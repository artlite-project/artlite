"""References of Flint: numpy oracles of its blocks (llama.cpp's ``src/models/flint.cpp``; its goldens: ``goldens.text``).

Oracles in float32 (llama.cpp's CPU ops) with tokens as the leading axis: ``qk_norm``, ``rope_neox`` (the sliding-window
layers only), ``attention`` (GQA, the window of the sliding-window layers, a learned sink per query head), the dense
SwiGLU FFN ``dense_ffn``, the MoE block ``moe_block`` (sigmoid router, top 6, weights normalized, SwiGLU experts, the
ungated shared expert; the pieces: ``moe.ref``), ``head_logits``; whole layers ``layer``.

Q4_1 weights multiply activations that llama.cpp rounds to Q8_1 (blocks of 32: a scale and the sum of the codes, both
fp16): ``mm`` does the same from the weights' exact fields (``Q41``), so a matmul agrees with llama.cpp's to its
summation order (~1e-6) instead of the activations' rounding (~1e-2); ``q8=False`` multiplies the plain fp32 values.
llama.cpp's amax rescale of the experts' activations (a device-range workaround) is exact up to rounding and left out.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import FlintConfig

from ..moe import ref as moe_ref
from ..np_ops import f32, rms_norm, silu
from ..qwen35.ref import rope_partial

# ----------------------------------------------------------------------
# Q4_1 matmuls with llama.cpp's Q8_1 activations
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Q41:
    """A Q4_1 matrix (rows, K) as llama.cpp's dot product reads it: ``qd`` = d q (rows, K) and ``m`` (rows, K / 32),
    float32; the values are ``qd`` + ``m`` repeated over each block."""

    qd: np.ndarray
    m: np.ndarray

    def dense(self) -> np.ndarray:
        return self.qd + np.repeat(self.m, 32, axis=1)

    @property
    def rows(self) -> int:
        return self.qd.shape[0]


def q41(t: QTensor) -> Q41:
    """A Q4_1 ``QTensor`` (2-D: rows x K) as ``Q41``."""
    assert t.qtype == "Q4_1", t.qtype
    f = t.fields()
    return Q41((f["d"][..., None] * f["q"].reshape(t.rows, -1, 32)).reshape(t.rows, -1).astype(np.float32),
               f["m"].astype(np.float32))


def q41_slice(g: GGUFFile, name: str, lo: int, hi: int) -> Q41:
    """Rows lo .. hi of a Q4_1 tensor's (rows x K) view (an expert of a 3-D tensor: its F or H rows)."""
    t = g.tensor(name)
    return q41(QTensor(name=name, qtype=t.qtype, rows=hi - lo, cols=t.cols, raw=np.asarray(t.raw[lo:hi])))


def q81(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """llama.cpp's Q8_1 rounding of x (T, K): per block of 32 d = max|x| / 127 (fp16), codes round(x 127 / max|x|)
    (ties to even), s = d sum(codes) (fp16). Returns the values d q (T, K) and s (T, K / 32), float32."""
    xb = f32(x).reshape(x.shape[0], -1, 32)
    mx = np.abs(xb).max(-1)
    with np.errstate(divide="ignore"):
        idv = np.where(mx != 0, np.float32(127) / mx, np.float32(0)).astype(np.float32)
    q = np.rint(xb * idv[..., None])
    d = (mx / np.float32(127)).astype(np.float32)
    d16 = d.astype(np.float16).astype(np.float32)
    s16 = (d * q.sum(-1, dtype=np.float32)).astype(np.float16).astype(np.float32)
    return (q * d16[..., None]).reshape(x.shape), s16


def mm(x: np.ndarray, w: Q41 | np.ndarray, *, q8: bool = True) -> np.ndarray:
    """x (T, K) @ w.T for a Q4_1 matrix (``Q41``; ``q8``: llama.cpp's Q8_1 activations) or a float matrix (rows, K)."""
    if not isinstance(w, Q41):
        return f32(x) @ f32(w).T
    if not q8:
        return f32(x) @ w.dense().T
    a, s = q81(x)
    return a @ w.qd.T + s @ w.m.T


def weight(g: GGUFFile, name: str) -> Q41 | np.ndarray:
    """A 2-D weight of the file: ``Q41`` for Q4_1, else fp32 (the router, the norms)."""
    t = g.tensor(name)
    return q41(t) if t.qtype == "Q4_1" else f32(t.dequantize()).reshape(t.rows, t.cols)


def ffn_q41(x: np.ndarray, gate: Q41, up: Q41, down: Q41, *, q8: bool = True) -> np.ndarray:
    """A SwiGLU FFN on Q4_1 matrices (llama.cpp's graph: gate and up from one Q8_1 rounding of x, down from another of
    silu(gate) * up)."""
    a = silu(mm(x, gate, q8=q8)) * mm(x, up, q8=q8)
    return mm(a, down, q8=q8)


# ----------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------


def qk_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm of every head of x (T, heads, d) with a weight per dim w (d,) (as stored: no 1 +)."""
    return rms_norm(x, w, eps)


def rope_neox(x: np.ndarray, cfg: FlintConfig, positions: np.ndarray, kind: str) -> np.ndarray:
    """NeoX RoPE of x (T, heads, d) at ``positions``: pairs (i, i + d / 2) on a "swa" layer (``cfg.rope_theta``), none
    on a "full" one (no positional encoding)."""
    return f32(x) if cfg.rot(kind) == 0 else rope_partial(x, cfg.rope_theta(positions, kind))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, sinks: np.ndarray, *, window: int | None,
              q_pos: np.ndarray) -> np.ndarray:
    """Softmax attention with a sink: q (T, Hq, d) at positions ``q_pos`` (T,), k, v (P, Hkv, d) at positions 0 .. P - 1;
    query head h attends KV head h // (Hq / Hkv) at the positions p <= q_pos and, with a ``window`` w, p > q_pos - w;
    scores scaled by 1 / sqrt(d). ``sinks`` (Hq,): per head one more logit that takes part in the maximum and the sum
    and has no value. Returns (T, Hq, d)."""
    q, k, v = f32(q), f32(k), f32(v)
    T, Hq, d = q.shape
    P, Hkv, _ = k.shape
    qg = q.reshape(T, Hkv, Hq // Hkv, d)
    s = np.einsum("tgjd,pgd->tgjp", qg, k) * np.float32(1.0 / np.sqrt(d))
    p = np.arange(P)[None, None, None, :]
    qp = np.asarray(q_pos)[:, None, None, None]
    masked = p > qp if window is None else (p > qp) | (qp - p >= window)
    s = np.where(masked, -np.inf, s).astype(np.float32)
    sk = f32(sinks).reshape(1, Hkv, Hq // Hkv, 1)
    m = np.maximum(s.max(-1, keepdims=True), sk)
    e = np.exp(s - m)
    den = e.sum(-1, keepdims=True, dtype=np.float32) + np.exp(sk - m)
    return np.einsum("tgjp,pgd->tgjd", e / den, v).reshape(T, Hq, d).astype(np.float32)


def attention_layer(g: GGUFFile, cfg: FlintConfig, layer: int, xn: np.ndarray, positions: np.ndarray, *,
                    q8: bool = True) -> dict[str, np.ndarray]:
    """A layer's attention from its normed input xn (T, H) at consecutive ``positions`` from 0: the projections ``q`` /
    ``k`` / ``v`` (T, heads, d), the normed and rotated ones (``qr``, ``kr``), the heads' output ``o`` (T, Hq d) and
    ``y`` = o @ Wo (T, H)."""
    b, hd, kind = f"blk.{layer}.", cfg.head_dim, cfg.kinds[layer]
    T = xn.shape[0]
    q = mm(xn, weight(g, b + "attn_q.weight"), q8=q8).reshape(T, -1, hd)
    k = mm(xn, weight(g, b + "attn_k.weight"), q8=q8).reshape(T, -1, hd)
    v = mm(xn, weight(g, b + "attn_v.weight"), q8=q8).reshape(T, -1, hd)
    qr = rope_neox(qk_norm(q, weight(g, b + "attn_q_norm.weight"), cfg.rms_norm_eps), cfg, positions, kind)
    kr = rope_neox(qk_norm(k, weight(g, b + "attn_k_norm.weight"), cfg.rms_norm_eps), cfg, positions, kind)
    sinks = weight(g, b + "attn_sinks.weight").reshape(-1)
    o = attention(qr, kr, v, sinks, window=cfg.sliding_window if kind == "swa" else None, q_pos=positions)
    o = o.reshape(T, -1)
    return {"q": q, "k": k, "v": v, "qr": qr, "kr": kr, "o": o, "y": mm(o, weight(g, b + "attn_output.weight"), q8=q8)}


# ----------------------------------------------------------------------
# FFN
# ----------------------------------------------------------------------


def dense_ffn(g: GGUFFile, layer: int, x: np.ndarray, *, q8: bool = True) -> np.ndarray:
    """The dense SwiGLU FFN of the first layers (x: the normed input (T, H))."""
    b = f"blk.{layer}."
    return ffn_q41(x, *(weight(g, b + f"ffn_{n}.weight") for n in ("gate", "up", "down")), q8=q8)


def router(g: GGUFFile, layer: int, x: np.ndarray) -> np.ndarray:
    """The router's logits (T, E), fp32 (the official file's router is F32)."""
    return mm(x, weight(g, f"blk.{layer}.ffn_gate_inp.weight"))


def expert_weights(g: GGUFFile, layer: int) -> Callable[[int], tuple[Q41, Q41, Q41]]:
    """Expert e of a layer as ``Q41`` (gate, up, down): rows e F .. (e + 1) F of ``ffn_gate_exps`` / ``ffn_up_exps``,
    e H .. (e + 1) H of ``ffn_down_exps``."""
    b = f"blk.{layer}."
    F = g.tensor(b + "ffn_down_exps.weight").cols
    H = g.tensor(b + "ffn_gate_exps.weight").cols

    def get(e: int) -> tuple[Q41, Q41, Q41]:
        return (q41_slice(g, b + "ffn_gate_exps.weight", e * F, (e + 1) * F),
                q41_slice(g, b + "ffn_up_exps.weight", e * F, (e + 1) * F),
                q41_slice(g, b + "ffn_down_exps.weight", e * H, (e + 1) * H))

    return get


def moe_block(g: GGUFFile, cfg: FlintConfig, layer: int, x: np.ndarray, *, q8: bool = True) -> dict[str, np.ndarray]:
    """A MoE layer's FFN of its normed input x (T, H): ``logits``, ``ids`` / ``weights`` (T, k), ``moe`` (the routed
    experts' sum), ``shexp`` (the shared expert), ``out`` = their sum."""
    b = f"blk.{layer}."
    logits = router(g, layer, x)
    ids, w = moe_ref.moe_route(logits, cfg.n_experts_used, "sigmoid")
    moe = moe_ref.moe_ffn(x, ids, w, expert_weights(g, layer), lambda x, *m: ffn_q41(x, *m, q8=q8))
    shexp = ffn_q41(x, *(weight(g, b + f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down")), q8=q8)
    return {"logits": logits, "ids": ids, "weights": w, "moe": moe, "shexp": shexp, "out": moe + shexp}


def layer(g: GGUFFile, cfg: FlintConfig, index: int, x: np.ndarray, positions: np.ndarray, *,
          q8: bool = True) -> dict[str, np.ndarray]:
    """A whole layer on x (T, H) at consecutive ``positions`` from 0: ``attn_norm``, ``attention_layer``'s tensors,
    ``ffn_inp``, ``ffn_norm``, ``dense_ffn``'s ``out`` or ``moe_block``'s, ``l_out``."""
    b, eps = f"blk.{index}.", cfg.rms_norm_eps
    xn = rms_norm(x, weight(g, b + "attn_norm.weight").reshape(-1), eps)
    a = attention_layer(g, cfg, index, xn, positions, q8=q8)
    ffn_inp = f32(x) + a["y"]
    fn = rms_norm(ffn_inp, weight(g, b + "ffn_norm.weight").reshape(-1), eps)
    f = moe_block(g, cfg, index, fn, q8=q8) if cfg.moe_layer(index) else {"out": dense_ffn(g, index, fn, q8=q8)}
    return {"attn_norm": xn, **a, "ffn_inp": ffn_inp, "ffn_norm": fn, **{"ffn_" + k: v for k, v in f.items()},
            "l_out": ffn_inp + f["out"]}


def head_logits(g: GGUFFile, cfg: FlintConfig, h: np.ndarray, rows: np.ndarray | None = None, *, step: int = 1 << 14,
                q8: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The final norm and the LM head of the last layer's output h (T, H): result_norm (T, H) and logits (T, V) (or of
    the vocabulary rows ``rows``), computed in pieces of ``step`` rows."""
    out = g.tensor("output.weight")
    hn = rms_norm(h, weight(g, "output_norm.weight").reshape(-1), cfg.rms_norm_eps)
    idx = np.arange(out.rows) if rows is None else np.asarray(rows)
    logits = np.empty((h.shape[0], len(idx)), np.float32)
    for lo in range(0, len(idx), step):
        sel = idx[lo:lo + step]
        raw = np.ascontiguousarray(out.raw[sel])
        logits[:, lo:lo + len(sel)] = mm(hn, q41(QTensor(name=out.name, qtype="Q4_1", rows=len(sel), cols=out.cols,
                                                         raw=raw)), q8=q8)
    return hn, logits
