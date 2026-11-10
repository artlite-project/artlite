"""LFM3 on the CPU in fp32: the reference of the NPU path, on a GGUF file's dequantized tensors (GGUF names; Q / K
rows as stored: NeoX pairs), the graph of llama.cpp's ``src/models/lfm3.cpp`` (conv layers: ``lfm-base.cpp``'s
``build_lfm_shortconv``).

``LfmCpu.run(tokens, state)`` runs tokens after a state's cached positions (``LfmState``: the attention layers' K / V,
the conv layers' histories) and returns their output-normed last hidden rows; ``logits(seq, rows)`` runs a sequence
from position 0 in chunks and returns the logits of the positions ``rows``. A non-causal file (d1's decision models)
reads a prompt whole from position 0: its full layers see it both ways, as a state and questions after it (``sets``:
the bounds S, e1, .., en; the state's tokens see every token, a question's the state's and their own: d1's ``set``
attention), the sliding-window and conv layers stay causal. The arithmetic's steps are methods
(``mm``, ``conv``, ``attend``, ``store_kv``, ``q_rows``, ``residual``) so that models of other rounding can
override them (N0's simulations, the NPU's host models)."""

import itertools
from dataclasses import dataclass, field

import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import LfmConfig


@dataclass
class LfmState:
    """The cached positions' state: per attention layer K, V (kv heads, positions, head_dim) after norm and RoPE,
    per conv layer the last conv_width - 1 rows of B * x (width - 1, hidden; zeros before position 0)."""

    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv: dict[int, torch.Tensor] = field(default_factory=dict)
    n: int = 0
    sets: tuple[int, ...] | None = None  # (non-causal: the prompt's bounds, ``LfmCpu`` docstring; None: all both ways)


class LfmCpu:
    """An LFM3 GGUF file's model in fp32 (its weights dequantized at construction). ``layers``: the layers to run (a
    slice: the first one reads the embeddings, its post-norm residual chain as the model's), ``vocab``: the LM head's
    first rows (default all)."""

    def __init__(self, gguf: GGUFFile | str, *, layers: list[int] | None = None, vocab: int | None = None):
        self.gguf = gguf = GGUFFile(gguf) if isinstance(gguf, str) else gguf
        self.cfg = cfg = LfmConfig.from_gguf(gguf)
        self.run_layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)

        def w(name: str) -> torch.Tensor:
            return gguf.tensor(name).dequantize().float()

        def v(name: str) -> torch.Tensor:
            return w(name).reshape(-1)

        self.emb = w("token_embd.weight")
        self.head_w = self.emb if vocab is None else self.emb[:vocab]
        self.norm = v("output_norm.weight")
        self.layers: dict[int, dict[str, torch.Tensor]] = {}
        for L in self.run_layers:
            kind = cfg.kinds[L]
            b = f"blk.{L}."
            lw = {"an": v(b + "attn_norm.weight"), "pan": v(b + "post_attention_norm.weight"),
                  "fn": v(b + "ffn_norm.weight"), "pfn": v(b + "post_ffw_norm.weight"),
                  "gate": w(b + "ffn_gate.weight"), "up": w(b + "ffn_up.weight"), "down": w(b + "ffn_down.weight")}
            if kind == "conv":
                lw |= {"in": w(b + "shortconv.in_proj.weight"), "out": w(b + "shortconv.out_proj.weight"),
                       "conv": w(b + "shortconv.conv.weight").reshape(cfg.hidden_size, cfg.conv_width)}
            else:
                lw |= {"q": w(b + "attn_q.weight"), "k": w(b + "attn_k.weight"), "v": w(b + "attn_v.weight"),
                       "g": w(b + "attn_gate.weight"), "o": w(b + "attn_output.weight"),
                       "qn": v(b + "attn_q_norm.weight"), "kn": v(b + "attn_k_norm.weight")}
            self.layers[L] = lw
        self.inv = {k: torch.from_numpy(cfg.rope_inv_freq(k)) for k in ("full", "swa")}

    # ------------------------------------------------------------------
    # Operations (overridden by models of other arithmetic)
    # ------------------------------------------------------------------

    def rms(self, x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return y if w is None else y * w

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x (T, K) @ w (N, K)^T."""
        return x @ w.T

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """The residual stream after an update (fp32 here)."""
        return x

    def rope(self, x: torch.Tensor, pos: torch.Tensor, kind: str) -> torch.Tensor:
        """x (heads, T, hd): pairs (i, i + hd / 2) for i < rot turned by p * inv_freq[i], the rest as they are."""
        inv = self.inv[kind]
        r, h = inv.numel(), x.shape[-1] // 2
        a = pos.double()[:, None] * inv[None]
        c, s = torch.cos(a).float(), torch.sin(a).float()
        x1, x2 = x[..., :r], x[..., h:h + r]
        out = x.clone()
        out[..., :r], out[..., h:h + r] = x1 * c - x2 * s, x2 * c + x1 * s
        return out

    def q_rows(self, q: torch.Tensor) -> torch.Tensor:
        """The queries (heads, T, hd) after norm and RoPE, as the attention takes them."""
        return q

    def store_kv(self, state: LfmState, L: int, k: torch.Tensor, v: torch.Tensor, p0: int) -> None:
        """Append K, V (kv heads, T, hd) of positions p0 .. to layer L's cache."""
        if p0 == 0:
            state.keys[L], state.values[L] = k, v
        else:
            state.keys[L] = torch.cat([state.keys[L][:, :p0], k], 1)
            state.values[L] = torch.cat([state.values[L][:, :p0], v], 1)

    def attend(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """softmax(q K^T / sqrt(hd)) V for the heads of a KV head: q (heads, T, hd), K, V (n, hd), mask (T, n) True =
        attended."""
        return torch.nn.functional.scaled_dot_product_attention(q[None], K[None, None], V[None, None],
                                                                attn_mask=mask[None, None])[0]

    def gate(self, g: torch.Tensor) -> torch.Tensor:
        """The output gates sigmoid(g) (T, heads)."""
        return torch.sigmoid(g)

    def conv(self, L: int, bx: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """The causal depthwise conv of B * x (T, hidden) after the history (width - 1, hidden): (T, hidden)."""
        w = self.layers[L]["conv"]  # (hidden, width): tap width - 1 on the token itself
        full = torch.cat([hist, bx])
        T, W = bx.shape[0], w.shape[1]
        return sum(full[j:j + T] * w[:, j] for j in range(W))

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def conv_op(self, L: int, xn: torch.Tensor, state: LfmState) -> torch.Tensor:
        cfg, lw = self.cfg, self.layers[L]
        H = cfg.hidden_size
        bcx = self.mm(xn, lw["in"])
        b, c, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        bx = b * x
        hist = state.conv.get(L)
        if hist is None:
            hist = torch.zeros(cfg.conv_width - 1, H)
        y = c * self.conv(L, bx, hist)
        state.conv[L] = torch.cat([hist, bx])[-(cfg.conv_width - 1):]
        return self.mm(y, lw["out"])

    def attn_op(self, L: int, xn: torch.Tensor, state: LfmState, p0: int) -> torch.Tensor:
        cfg, lw, kind = self.cfg, self.layers[L], self.cfg.kinds[L]
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        T = xn.shape[0]
        pos = torch.arange(p0, p0 + T)
        q = self.mm(xn, lw["q"]).view(T, nh, hd).transpose(0, 1)
        k = self.mm(xn, lw["k"]).view(T, nkv, hd).transpose(0, 1)
        v = self.mm(xn, lw["v"]).view(T, nkv, hd).transpose(0, 1)
        q = self.q_rows(self.rope(self.rms(q, lw["qn"]), pos, kind))
        k = self.rope(self.rms(k, lw["kn"]), pos, kind)
        self.store_kv(state, L, k, v, p0)
        K, V = state.keys[L][:, :p0 + T], state.values[L][:, :p0 + T]
        kpos = torch.arange(p0 + T)[None, :]
        mask = kpos <= pos[:, None] if cfg.causal or kind == "swa" else torch.ones(T, p0 + T, dtype=torch.bool)
        if kind == "swa":
            mask = mask & (pos[:, None] - kpos < cfg.sliding_window)
        elif not cfg.causal and state.sets is not None:  # (a question's rows: the state's keys and their own)
            assert p0 == 0 and state.sets[-1] == T, (p0, state.sets, T)
            lo, hi = torch.zeros(T, dtype=torch.long), torch.full((T,), T)
            for a, b in itertools.pairwise(state.sets):
                lo[a:b], hi[a:b] = a, b
            mask = (kpos < hi[:, None]) & ((kpos < state.sets[0]) | (kpos >= lo[:, None]))
        per = nh // nkv
        o = torch.cat([self.attend(q[h * per:(h + 1) * per], K[h], V[h], mask) for h in range(nkv)])  # (nh, T, hd)
        o = o * self.gate(self.mm(xn, lw["g"])).T[:, :, None]
        return self.mm(o.transpose(0, 1).reshape(T, nh * hd), lw["o"])

    def ffn(self, L: int, xn: torch.Tensor) -> torch.Tensor:
        lw = self.layers[L]
        return self.mm(torch.nn.functional.silu(self.mm(xn, lw["gate"])) * self.mm(xn, lw["up"]), lw["down"])

    def layer(self, L: int, x: torch.Tensor, state: LfmState, p0: int) -> torch.Tensor:
        lw = self.layers[L]
        xn = self.rms(x, lw["an"])
        y = self.conv_op(L, xn, state) if self.cfg.kinds[L] == "conv" else self.attn_op(L, xn, state, p0)
        x = self.residual(x + self.rms(y, lw["pan"]))
        return self.residual(x + self.rms(self.ffn(L, self.rms(x, lw["fn"])), lw["pfn"]))

    @torch.no_grad()
    def run(self, tokens: list[int], state: LfmState) -> torch.Tensor:
        """The output-normed last hidden rows (T, hidden) of tokens after the state's positions (the state gains
        them)."""
        p0 = state.n
        x = self.emb[torch.tensor(tokens)]
        for L in self.run_layers:
            x = self.layer(L, x, state, p0)
        state.n = p0 + len(tokens)
        return self.rms(x, self.norm)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """The logits (n, vocab) of output-normed hidden rows."""
        return self.mm(h, self.head_w)

    @torch.no_grad()
    def outputs(self, seq: list[int], rows: slice | list[int], *, chunk: int = 1024,
                state: LfmState | None = None, sets: tuple[int, ...] | None = None) -> torch.Tensor:
        """The output-normed last hidden rows (n, hidden) of the positions ``rows`` of seq (from position 0, or after
        a given state's positions: rows of seq); a non-causal file's prompt whole (``sets``: its bounds)."""
        state = LfmState(sets=sets) if state is None else state
        if not self.cfg.causal:
            assert state.n == 0, "a non-causal prompt runs whole from position 0"
            chunk = max(chunk, len(seq))
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
               state: LfmState | None = None, sets: tuple[int, ...] | None = None) -> torch.Tensor:
        """The fp32 logits (n, vocab) of the positions ``rows`` of seq."""
        return self.head(self.outputs(seq, rows, chunk=chunk, state=state, sets=sets))


def kl(ref: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """KL(softmax(ref) || softmax(got)) per row (float64)."""
    la, lb = torch.log_softmax(ref.double(), -1), torch.log_softmax(got.double(), -1)
    return (la.exp() * (la - lb)).sum(-1)

