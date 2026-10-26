"""Gemma 4 on the CPU in fp32: the reference of the NPU path (the graph of llama.cpp's
``src/models/gemma4.cpp`` on the dequantized weights of a GGUF file).

``GemmaCpu.forward`` runs a batch of tokens after the cached positions (a prompt, a
chunk of one, or one token). The matrices are dequantized on first use and kept in fp32
(E4B: ~18 GB with the LM head); the per-layer token embedding is a lookup in the mapped
file. Checked against llama.cpp's CPU backend (BF16 file: KL 2e-7 on the next-token
distribution; llama.cpp itself rounds activations to bf16 there).

The operations are methods (``mm``, ``norm``, ``gelu``, ``rope``, ``attend``) so that models of
other arithmetic can override them (ref.py: ``mm`` with the NPU's rounding). The embedding and the
per-layer inputs are the NPU app's (``models.gemma4.host.GemmaHost``).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from models.gemma4 import host
from models.gemma4.host import GemmaHost, rms_norm

SOURCES = [__file__, host.__file__]  # the reference's code (tests' cached_ref)


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Final logit soft-capping: cap * tanh(logits / cap) (monotonic)."""
    return torch.tanh(logits / cap) * cap if cap else logits


def rope_angles(inv_freq: np.ndarray, positions: np.ndarray | range | list[int]) -> torch.Tensor:
    """(n, head_dim) fp32 angle rows of the positions: cos, sin of p * inv_freq[i] interleaved."""
    f = np.outer(np.asarray(positions, np.float64), inv_freq)
    a = np.empty((f.shape[0], 2 * f.shape[1]), np.float32)
    a[:, 0::2] = np.cos(f)
    a[:, 1::2] = np.sin(f)
    return torch.from_numpy(a)


def rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """"Two halves" RoPE of x (T, heads, head_dim) with angle rows (T, head_dim) = cos, sin
    interleaved: pair i is (x[i], x[i + head_dim / 2])."""
    c, s = angles[:, None, 0::2], angles[:, None, 1::2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU in its tanh form (ggml_gelu, gelu_pytorch_tanh)."""
    return torch.nn.functional.gelu(x, approximate="tanh")


@dataclass
class GemmaState:
    """Per layer with K / V projections: K, V (kv heads, positions, head_dim) of the cached
    positions (every position; the sliding window is a mask)."""

    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    n: int = 0  # cached positions


class GemmaCpu(GemmaHost):
    """The model of a GGUF file."""

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def layer_names(self, layer: int) -> list[str]:
        pfx = f"blk.{layer}."
        return [n for n in self.gguf.tensor_info if n.startswith(pfx)]

    def preload(self, layers: range | None = None, *, lm_head: bool = True) -> None:
        """Dequantize the matrices of ``layers`` (default: all) in parallel."""
        names = [n for layer in (layers or range(self.cfg.num_hidden_layers)) for n in self.layer_names(layer)]
        names += ["output_norm.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight"]
        names = [n for n in names if n in self.gguf]
        if lm_head:
            names.append("token_embd.weight")
        with ThreadPoolExecutor(self._threads) as ex:  # numpy releases the GIL
            list(ex.map(self.mat, names))

    # ------------------------------------------------------------------
    # Operations (overridden by models of the NPU arithmetic; mm, norm: GemmaHost)
    # ------------------------------------------------------------------

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x)

    def rope(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return rope(x, angles)

    def attend(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """softmax(q K^T) V for one head: q (T, hd), K, V (n, hd), mask (T, n) True = not attended
        (score scale 1.0)."""
        return torch.softmax((q @ K.T).masked_fill(mask, float("-inf")), -1) @ V

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def qkv(self, layer: int, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The projections of the normed input xn (T, hidden): Q (T, heads * head_dim) and, for a
        layer with its own cache, K, V (T, kv heads * head_dim)."""
        pfx = f"blk.{layer}."
        q = self.mm(pfx + "attn_q.weight", xn)
        if self.cfg.kv_source(layer) != layer:
            return q, None, None
        return q, self.mm(pfx + "attn_k.weight", xn), self.mm(pfx + "attn_v.weight", xn)

    def attend_qkv(self, layer: int, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None,
                   state: GemmaState, p0: int) -> torch.Tensor:
        """Attention of the projections (``qkv``) of T tokens at positions p0 ..: norms, RoPE, the
        cache update and softmax(Q K^T) V per head -> (T, heads * head_dim)."""
        cfg = self.cfg
        T, hd, nh = q.shape[0], cfg.head_dim(layer), cfg.num_attention_heads
        pfx, swa = f"blk.{layer}.", cfg.swa[layer]
        ang = rope_angles(cfg.rope_inv_freq(swa=swa), range(p0, p0 + T))
        q = self.rope(self.norm(q.view(T, nh, hd), pfx + "attn_q_norm.weight"), ang)
        src = cfg.kv_source(layer)
        nkv = cfg.kv_heads[src]
        if src == layer:
            k = self.rope(self.norm(k.view(T, nkv, hd), pfx + "attn_k_norm.weight"), ang).transpose(0, 1)
            v = self.norm(v.view(T, nkv, hd), None).transpose(0, 1)
            self.store_kv(state, layer, k, v, p0)
        K, V = state.keys[src][:, :p0 + T], state.values[src][:, :p0 + T]
        kpos = torch.arange(p0 + T)[None, :]
        qpos = torch.arange(p0, p0 + T)[:, None]
        mask = kpos > qpos
        if swa:
            mask = mask | (qpos - kpos >= cfg.sliding_window)
        out = [self.attend(q[:, h], K[h // (nh // nkv)], V[h // (nh // nkv)], mask) for h in range(nh)]
        return torch.stack(out, 1).reshape(T, nh * hd)

    def attention(self, layer: int, xn: torch.Tensor, state: GemmaState, p0: int) -> torch.Tensor:
        """The attention sub-layer's output before the O projection: (T, heads * head_dim)."""
        return self.attend_qkv(layer, *self.qkv(layer, xn), state, p0)

    def store_kv(self, state: GemmaState, layer: int, k: torch.Tensor, v: torch.Tensor, p0: int) -> None:
        """Append K, V (kv heads, T, head_dim) of positions p0 .. to the layer's cache."""
        state.keys[layer] = k if p0 == 0 else torch.cat([state.keys[layer][:, :p0], k], 1)
        state.values[layer] = v if p0 == 0 else torch.cat([state.values[layer][:, :p0], v], 1)

    def layer(self, layer: int, x: torch.Tensor, pli: torch.Tensor | None, state: GemmaState, p0: int,
              trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """One layer on x (T, hidden). trace: every sub-layer output by name (``o.3``: the O
        projection of layer 3 before its norm, ...)."""
        pfx = f"blk.{layer}."

        def keep(name: str, t: torch.Tensor) -> torch.Tensor:
            if trace is not None:
                trace[f"{name}.{layer}"] = t
            return t

        xn = self.norm(x, pfx + "attn_norm.weight")
        att = keep("attn", self.attention(layer, xn, state, p0))
        o = keep("o", self.mm(pfx + "attn_output.weight", att))
        x = keep("attn_out", self.norm(o, pfx + "post_attention_norm.weight") + x)
        xn = self.norm(x, pfx + "ffn_norm.weight")
        h = keep("h", self.gelu(self.mm(pfx + "ffn_gate.weight", xn)) * self.mm(pfx + "ffn_up.weight", xn))
        d = keep("down", self.mm(pfx + "ffn_down.weight", h))
        x = keep("ffn_out", self.norm(d, pfx + "post_ffw_norm.weight") + x)
        if pli is not None:
            g = keep("gate", self.gelu(self.mm(pfx + "inp_gate.weight", x)) * pli[:, layer])
            x = x + self.norm(keep("proj", self.mm(pfx + "proj.weight", g)), pfx + "post_norm.weight")
        if pfx + "layer_output_scale.weight" in self.gguf:
            x = x * self.mat(pfx + "layer_output_scale.weight")
        return keep("l_out", x)

    def hidden(self, tokens: list[int], state: GemmaState | None = None,
               trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """The output-normed hidden states (T, hidden) of ``tokens`` at the positions after the
        state's cached ones; the state gains their K, V."""
        state = state if state is not None else GemmaState()
        p0 = state.n
        x = self.embed(tokens)
        pli = self.per_layer_inputs(tokens, x)
        if trace is not None:
            trace["x0"], trace["pli"] = x, pli
        for layer in range(self.cfg.num_hidden_layers):
            x = self.layer(layer, x, pli, state, p0, trace)
        state.n = p0 + len(tokens)
        return self.norm(x, "output_norm.weight")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(T, vocab) soft-capped logits of output-normed hidden states (T, hidden)."""
        name = "output.weight" if "output.weight" in self.gguf else "token_embd.weight"
        return softcap(self.mm(name, h), self.cfg.final_logit_softcapping)

    def forward(self, tokens: list[int], state: GemmaState | None = None) -> torch.Tensor:
        """Logits of the last token (vocab,)."""
        return self.logits(self.hidden(tokens, state)[-1:])[0]


class GemmaAssistant:
    """The multi-token-prediction drafter of a Gemma 4 model (a ``gemma4-assistant`` GGUF file; the
    graph of llama.cpp's ``src/models/gemma4-assistant.cpp``), fp32.

    A draft step takes a token and a hidden state h (the target's output-normed hidden state of the
    position before the token, or the h the previous draft step returned) and predicts the token
    after it. Its layers have no K / V of their own: they attend to the target's caches of its last
    sliding-window and its last full-attention layer with K / V projections."""

    def __init__(self, gguf: GGUFFile, target: GemmaCpu):
        self.gguf, self.target = gguf, target
        assert gguf.arch == "gemma4-assistant", f"{gguf.path}: architecture {gguf.arch!r}"
        m = gguf.arch_meta
        self.n_layer = int(m("block_count"))
        self.swa = [bool(x) for x in m("attention.sliding_window_pattern")]
        self.hd = [int(m("attention.key_length_swa")) if s else int(m("attention.key_length")) for s in self.swa]
        self.heads = int(m("attention.head_count"))
        self.eps = float(m("attention.layer_norm_rms_epsilon"))
        self.w: dict[str, torch.Tensor] = {}
        for n in gguf.tensor_info:
            t = gguf.tensor(n).dequantize()
            self.w[n] = t.reshape(-1) if t.shape[0] == 1 else t

    def step(self, token: int, h: torch.Tensor, state: GemmaState, pos: int,
             trace: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """A draft step at position pos on the target's cache (positions < pos): the logits of the
        next token (vocab,) and the hidden state for the step after (every step of a draft runs at
        the same position: the one of its first token). trace: the sub-layer outputs by name."""
        t, w, eps = self.target, self.w, self.eps
        cfg = t.cfg

        def keep(name: str, x: torch.Tensor) -> torch.Tensor:
            if trace is not None:
                trace[name] = x
            return x

        cur = keep("pre", w["nextn.pre_projection.weight"] @ torch.cat([t.embed([token])[0], h]))
        for il in range(self.n_layer):
            b = f"blk.{il}."
            swa, hd = self.swa[il], self.hd[il]
            q = keep(f"q.{il}", w[b + "attn_q.weight"] @ rms_norm(cur, w[b + "attn_norm.weight"], eps))
            ang = rope_angles(cfg.rope_inv_freq(swa=swa), [pos])
            q = rope(rms_norm(q.view(1, self.heads, hd), w[b + "attn_q_norm.weight"], eps), ang)[0]
            src = cfg.kv_layers - (2 if swa else 1)  # the target's last SWA / full cache
            lo = max(0, pos - cfg.sliding_window + 1) if swa else 0
            K, V = state.keys[src][:, lo:pos], state.values[src][:, lo:pos]
            per = self.heads // K.shape[0]
            att = torch.cat([torch.softmax(q[hh] @ K[hh // per].T, -1) @ V[hh // per] for hh in range(self.heads)])
            o = keep(f"o.{il}", w[b + "attn_output.weight"] @ keep(f"attn.{il}", att))
            a = rms_norm(o, w[b + "post_attention_norm.weight"], eps) + cur
            xn = rms_norm(a, w[b + "ffn_norm.weight"], eps)
            f = keep(f"h.{il}", gelu(w[b + "ffn_gate.weight"] @ xn) * (w[b + "ffn_up.weight"] @ xn))
            f = keep(f"down.{il}", w[b + "ffn_down.weight"] @ f)
            cur = keep(f"l_out.{il}", (rms_norm(f, w[b + "post_ffw_norm.weight"], eps) + a) * w[b + "layer_output_scale.weight"])
        cur = rms_norm(cur, w["output_norm.weight"], eps)
        return w["token_embd.weight"] @ cur, w["nextn.post_projection.weight"] @ cur

