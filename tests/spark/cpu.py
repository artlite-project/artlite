"""Spark-X2.5 / X2.6 (GGUF ``spark2_5``) on the CPU in fp32: the reference of the NPU path, on a GGUF file's dequantized
tensors (GGUF names; Q / K rows as stored: NeoX pairs), the graph of llama.cpp's ``src/models/spark2-5.cpp`` and HF's
``modeling_spark.py`` (``model_config.SparkConfig``). GELU: "erf" (HF's ``gelu``, the model's) or "tanh" (llama.cpp's
GEGLU; they differ by KL ~1e-5).

``SparkCpu.run(tokens, state)`` runs tokens after a state's cached positions and returns their output-normed last
hidden rows (``taps``: the residual entering given layers, a DSpark drafter's inputs); ``logits(seq, rows)`` runs a
sequence from position 0 in chunks and returns the logits of the positions ``rows``. The arithmetic's steps are methods
(``mm``, ``attend``, ``store_kv``, ``q_rows``, ``residual``, ``gate``, ``act``) so that models of other rounding
override them."""

from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import SparkConfig


@dataclass
class SparkState:
    """The cached positions: per layer K (after RoPE), V (kv heads, positions, head_dim)."""

    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    n: int = 0


def gelu(x: torch.Tensor, form: str) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh" if form == "tanh" else "none")


class SparkCpu:
    """A ``spark2_5`` GGUF file's model in fp32 (weights dequantized at construction)."""

    def __init__(self, gguf: GGUFFile | str, *, gelu_form: str = "erf", layers: list[int] | None = None):
        self.gguf = gguf = GGUFFile(gguf) if isinstance(gguf, str) else gguf
        self.cfg = cfg = SparkConfig.from_gguf(gguf)
        self.gelu_form = gelu_form
        self.run_layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)

        def w(name: str) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(gguf.tensor(name).dequantize(), np.float32))

        def v(name: str) -> torch.Tensor:
            return w(name).reshape(-1)

        self.emb = w("token_embd.weight")
        self.head_w = self.emb if cfg.tied else w("output.weight")
        self.norm = v("output_norm.weight")
        self.layers: dict[int, dict[str, torch.Tensor]] = {}
        for L in self.run_layers:
            b = f"blk.{L}."
            if b + "attn_qkv.weight" in gguf:
                qkv = w(b + "attn_qkv.weight")
            else:
                qkv = torch.cat([w(b + "attn_q.weight"), w(b + "attn_k.weight"), w(b + "attn_v.weight")])
            self.layers[L] = {"an": v(b + "attn_norm.weight"), "qkv": qkv, "g": w(b + "attn_gate.weight"),
                              "o": w(b + "attn_output.weight"), "fn": v(b + "ffn_norm.weight"),
                              "gate": w(b + "ffn_gate.weight"), "up": w(b + "ffn_up.weight"),
                              "down": w(b + "ffn_down.weight")}
        self.inv = {k: torch.from_numpy((cfg.rope_base if k == "full" else cfg.rope_base_swa)
                                        ** (-np.arange(cfg.rot(k), dtype=np.float64) / cfg.rot(k)))
                    for k in ("full", "swa")}

    # -- operations (overridden by models of other arithmetic) ---------------------------------------------------

    def rms(self, x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return y if w is None else y * w

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x (T, K) @ w (N, K)^T."""
        return x @ w.T

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def act(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """The FFN's gelu(gate) * up."""
        return gelu(g, self.gelu_form) * u

    def rope(self, x: torch.Tensor, pos: torch.Tensor, kind: str) -> torch.Tensor:
        """x (heads, T, hd): NeoX pairs (i, i + r) of the first 2 r dims turned by p * inv_freq[i]."""
        inv = self.inv[kind]
        r = inv.numel()
        a = pos.double()[:, None] * inv[None]
        c, s = torch.cos(a).float(), torch.sin(a).float()
        x1, x2 = x[..., :r], x[..., r:2 * r]
        out = x.clone()
        out[..., :r], out[..., r:2 * r] = x1 * c - x2 * s, x2 * c + x1 * s
        return out

    def q_rows(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def store_kv(self, state: SparkState, L: int, k: torch.Tensor, v: torch.Tensor, p0: int) -> None:
        if p0 == 0:
            state.keys[L], state.values[L] = k, v
        else:
            state.keys[L] = torch.cat([state.keys[L][:, :p0], k], 1)
            state.values[L] = torch.cat([state.values[L][:, :p0], v], 1)

    def attend(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """softmax(q K^T / sqrt(hd)) V: q (heads, T, hd), K, V (n, hd), mask (T, n) True = attended."""
        return torch.nn.functional.scaled_dot_product_attention(q[None], K[None, None], V[None, None],
                                                                attn_mask=mask[None, None])[0]

    def gate(self, g: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(g)

    # -- layers ----------------------------------------------------------------------------------------------------

    def attn_op(self, L: int, xn: torch.Tensor, state: SparkState, p0: int) -> torch.Tensor:
        cfg, lw, kind = self.cfg, self.layers[L], self.cfg.kinds[L]
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        T = xn.shape[0]
        pos = torch.arange(p0, p0 + T)
        qkv = self.mm(xn, lw["qkv"])
        q = qkv[:, :nh * hd].reshape(T, nh, hd).transpose(0, 1)
        k = qkv[:, nh * hd:(nh + nkv) * hd].reshape(T, nkv, hd).transpose(0, 1)
        v = qkv[:, (nh + nkv) * hd:].reshape(T, nkv, hd).transpose(0, 1).contiguous()
        q = self.q_rows(self.rope(q, pos, kind))
        k = self.rope(k, pos, kind)
        self.store_kv(state, L, k, v, p0)
        lo = max(0, p0 - cfg.sliding_window + 1) if kind == "swa" else 0
        K, V = state.keys[L][:, lo:p0 + T], state.values[L][:, lo:p0 + T]
        kpos = torch.arange(lo, p0 + T)[None, :]
        mask = kpos <= pos[:, None]
        if kind == "swa":
            mask = mask & (pos[:, None] - kpos < cfg.sliding_window)
        per = nh // nkv
        o = torch.cat([self.attend(q[h * per:(h + 1) * per], K[h], V[h], mask) for h in range(nkv)])  # (nh, T, hd)
        o = o * self.gate(self.mm(xn, lw["g"])).T[:, :, None]
        return self.mm(o.transpose(0, 1).reshape(T, nh * hd), lw["o"])

    def ffn(self, L: int, xn: torch.Tensor) -> torch.Tensor:
        lw = self.layers[L]
        return self.mm(self.act(self.mm(xn, lw["gate"]), self.mm(xn, lw["up"])), lw["down"])

    def layer(self, L: int, x: torch.Tensor, state: SparkState, p0: int) -> torch.Tensor:
        lw = self.layers[L]
        x = self.residual(x + self.attn_op(L, self.rms(x, lw["an"]), state, p0))
        return self.residual(x + self.ffn(L, self.rms(x, lw["fn"])))

    @torch.no_grad()
    def run(self, tokens: list[int], state: SparkState, *, taps: dict[int, list] | None = None) -> torch.Tensor:
        """The output-normed last hidden rows (T, hidden) of tokens after the state's positions. ``taps``: layer ->
        a list that gains the residual entering that layer (T, hidden) (DSpark's target features)."""
        p0 = state.n
        x = self.emb[torch.tensor(tokens)]
        for L in self.run_layers:
            if taps is not None and L in taps:
                taps[L].append(x.clone())
            x = self.layer(L, x, state, p0)
        state.n = p0 + len(tokens)
        return self.rms(x, self.norm)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.mm(h, self.head_w)

    @torch.no_grad()
    def outputs(self, seq: list[int], rows: slice | list[int], *, chunk: int = 1024,
                state: SparkState | None = None) -> torch.Tensor:
        state = SparkState() if state is None else state
        want = torch.zeros(len(seq), dtype=torch.bool)
        want[rows] = True
        out = []
        for c0 in range(0, len(seq), chunk):
            h = self.run(seq[c0:c0 + chunk], state)
            sel = want[c0:c0 + chunk]
            if sel.any():
                out.append(h[sel])
        return torch.cat(out)

    @torch.no_grad()
    def logits(self, seq: list[int], rows: slice | list[int], *, chunk: int = 1024,
               state: SparkState | None = None) -> torch.Tensor:
        return self.head(self.outputs(seq, rows, chunk=chunk, state=state))
