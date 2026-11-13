"""References of the qwen35 models (Qwen3.5 / 3.6 / 3.8): numpy oracles of the blocks (llama.cpp's goldens:
``goldens.text``).

Oracles: the exact math of each block in float32 (llama.cpp's CPU ops), on numpy arrays with tokens as the
leading axis (the basic ops: ``np_ops``; the MoE FFN: ``moe.ref``): ``conv_seq``, ``gdn_gates``, ``gdn_seq`` /
``gdn_step``, ``gated_norm``, ``rope_partial``, ``attention_gqa``, ``q_and_gate``; a multi-token-prediction block:
``mtp_block`` (a DFlash2 block drafter: ``dflash_ref``).

The cores' arithmetic (their rounding points, float64 elsewhere): ``core_gdn_step`` (the decode's GDN core),
``core_gdn_scan`` / ``core_gdn_out`` (the prefill's GDN scan and gated output), ``core_attention`` (a decode
attention column).
"""

import numpy as np

from artlite.model_config import QwenConfig
from models.qwen35.attn import HD, ROT
from models.qwen35.gdn import D

from ..moe.ref import experts, moe_ffn, moe_route, shared_expert, swiglu
from ..np_ops import f32, rms_norm, sigmoid, silu, softmax, softplus
from ..numerics import bf16_round

# ----------------------------------------------------------------------
# Oracles (float32, tokens first)
# ----------------------------------------------------------------------


def conv_seq(x: np.ndarray, w: np.ndarray, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The GDN's causal depthwise conv (no bias, before its SiLU): x (T, C) raw inputs, w (C, K) taps (the
    GGUF ``ssm_conv1d``), state (C, K - 1) the last K - 1 inputs per channel (llama.cpp's conv state layout;
    None: zeros) -> y (T, C) with y_c(t) = sum_i w[c, i] x_c(t - K + 1 + i), and the new state."""
    x, w = f32(x), f32(w)
    C, K = w.shape
    s = np.zeros((C, K - 1), np.float32) if state is None else f32(state)
    win = np.concatenate([s.T, x], axis=0)  # (K - 1 + T, C)
    T = x.shape[0]
    y = np.zeros((T, C), np.float32)
    for i in range(K):
        y += win[i:i + T] * w[:, i]
    return y, np.ascontiguousarray(win[T:].T)


def split_qkv(qkv: np.ndarray, cfg: QwenConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, conv channels) [q | k | v] -> q, k (T, k heads, d), v (T, v heads, d)."""
    T, d, hk = qkv.shape[0], cfg.ssm_state, cfg.ssm_k_heads
    q = qkv[:, :hk * d].reshape(T, hk, d)
    k = qkv[:, hk * d:2 * hk * d].reshape(T, hk, d)
    return q, k, qkv[:, 2 * hk * d:].reshape(T, -1, d)


def gdn_gates(alpha: np.ndarray, beta: np.ndarray, dt_bias: np.ndarray, ssm_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(g, beta): g = softplus(alpha + dt_bias) * ssm_a (the GGUF's ssm_a = -exp(A_log)), beta = sigmoid(beta)."""
    return softplus(f32(alpha) + f32(dt_bias)) * f32(ssm_a), sigmoid(beta)


def gdn_seq(q: np.ndarray, k: np.ndarray, v: np.ndarray, g: np.ndarray, beta: np.ndarray,
            state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The gated delta rule over T tokens, token by token (llama.cpp's ``ggml_gated_delta_net``): q, k (T, Hk,
    d) l2-normed, v (T, Hv, d), g, beta (T, Hv); state (Hv, d, d) as llama.cpp stores it: state[h, j, i] =
    S[i][j] (i: k dim, j: v dim; None: zeros). v head h reads q / k head h % Hk. Per token and v head:
    S <- e^g S; u = S^T k; delta = beta (v - u); S += k delta^T; o = S^T q / sqrt(d). Returns o (T, Hv, d)
    and the final state."""
    q, k, v, g, beta = f32(q), f32(k), f32(v), f32(g), f32(beta)
    T, Hk, d = q.shape
    Hv = v.shape[1]
    M = np.zeros((Hv, d, d), np.float32) if state is None else f32(state).copy()  # M[h, j, i] = S[i][j]
    kh = np.arange(Hv) % Hk
    scale = np.float32(1.0 / np.sqrt(d))
    o = np.empty((T, Hv, d), np.float32)
    for t in range(T):
        qt, kt = q[t, kh], k[t, kh]  # (Hv, d)
        M *= np.exp(g[t])[:, None, None]
        u = np.einsum("hji,hi->hj", M, kt)
        delta = beta[t][:, None] * (v[t] - u)
        M += delta[:, :, None] * kt[:, None, :]
        o[t] = np.einsum("hji,hi->hj", M, qt) * scale
    return o, M


def gdn_step(q: np.ndarray, k: np.ndarray, v: np.ndarray, g: np.ndarray, beta: np.ndarray,
             state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One token of ``gdn_seq``: q, k (Hk, d), v (Hv, d), g, beta (Hv,) -> o (Hv, d), the new state."""
    o, s = gdn_seq(q[None], k[None], v[None], f32(g)[None], f32(beta)[None], state)
    return o[0], s


def gated_norm(o: np.ndarray, z: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """The GDN output gate: rms(o) * ssm_norm * silu(z) per head (o, z (..., d))."""
    return rms_norm(o, w, eps) * silu(z)


def q_and_gate(qfull: np.ndarray, heads: int) -> tuple[np.ndarray, np.ndarray]:
    """W_q's output (T, heads * 2 d), per head [q (d) | gate (d)] -> q (T, heads, d), gate (T, heads, d)."""
    x = f32(qfull).reshape(qfull.shape[0], heads, 2, -1)
    return x[:, :, 0], x[:, :, 1]


def rope_partial(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """NeoX RoPE of the first 2 n dims (pairs (i, i + n), i < n) of x (T, heads, hd) by theta (T, n)
    (``QwenConfig.rope_theta``); the dims from 2 n on pass unchanged."""
    x = f32(x).copy()
    n = theta.shape[-1]
    c, s = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]
    a, b = x[..., :n].copy(), x[..., n:2 * n].copy()
    x[..., :n] = a * c - b * s
    x[..., n:2 * n] = a * s + b * c
    return x


def attention_gqa(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float, q_pos: np.ndarray) -> np.ndarray:
    """Causal softmax attention: q (T, Hq, d) at positions q_pos (T,), k, v (P, Hkv, d) at positions 0 .. P - 1;
    query head h attends KV head h // (Hq / Hkv). Returns (T, Hq, d)."""
    q, k, v = f32(q), f32(k), f32(v)
    Hq = q.shape[1]
    P, Hkv, _ = k.shape
    kv = np.arange(Hq) // (Hq // Hkv)
    s = np.einsum("thd,phd->thp", q, k[:, kv]) * np.float32(scale)
    s = np.where(np.arange(P)[None, None, :] > np.asarray(q_pos)[:, None, None], -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("thp,phd->thd", p, v[:, kv]).astype(np.float32)


def mtp_block(gguf: object, cfg: QwenConfig, emb: np.ndarray, h: np.ndarray, pos: np.ndarray, *,
              head: np.ndarray | None = None, swap: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """An MTP block (llama.cpp's ``graph_mtp``, the block at index ``cfg.num_hidden_layers`` of an MTP file, or of a
    model file with a block of its own) over rows of consecutive positions ``pos`` from 0 (T,), each the cache of the
    rows after it: emb (T, H) the rows' token embeddings, h (T, H) the model's output-normed hidden state of the
    position before each (zeros at 0). Its FFN: a ``qwen35moe`` file's MoE, a ``qwen35`` file's dense SwiGLU.
    Returns its tensors by llama.cpp's names (``mtp_eh_proj``: the layer's input x0, ``mtp_attn_residual``,
    ``mtp_attn_post_norm``, ``mtp_ffn_out``, ``mtp_post_ffn``, ``h_nextn``: the next draft's h), an MoE's experts' ids
    and weights (``ids``, ``weights``) and the logits of the tokens ``head`` (None: none). ``swap`` (T,) bool: rows whose
    router takes its 9th expert instead of its 8th (a near tie's other side)."""
    L, eps = cfg.num_hidden_layers, cfg.rms_norm_eps
    b = f"blk.{L}."

    def mat(name: str) -> np.ndarray:
        t = gguf.tensor(b + name)
        return f32(t.dequantize()).reshape(t.rows, t.cols)

    def vec(name: str) -> np.ndarray:
        return f32(gguf.tensor(b + name).dequantize()).reshape(-1)

    T, d = emb.shape[0], cfg.head_dim
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    out = {}
    cat = np.concatenate([rms_norm(emb, vec("nextn.enorm.weight"), eps), rms_norm(h, vec("nextn.hnorm.weight"), eps)], -1)
    x0 = out["mtp_eh_proj"] = cat @ mat("nextn.eh_proj.weight").T
    xa = rms_norm(x0, vec("attn_norm.weight"), eps)
    q, gate = q_and_gate(xa @ mat("attn_q.weight").T, nh)
    k = (xa @ mat("attn_k.weight").T).reshape(T, nkv, d)
    v = (xa @ mat("attn_v.weight").T).reshape(T, nkv, d)
    theta = cfg.rope_theta(pos)
    q = rope_partial(rms_norm(q, vec("attn_q_norm.weight"), eps), theta)
    k = rope_partial(rms_norm(k, vec("attn_k_norm.weight"), eps), theta)
    o = attention_gqa(q, k, v, scale=1 / np.sqrt(d), q_pos=np.asarray(pos))
    x1 = out["mtp_attn_residual"] = (o * sigmoid(gate)).reshape(T, -1) @ mat("attn_output.weight").T + x0
    xm = out["mtp_attn_post_norm"] = rms_norm(x1, vec("post_attention_norm.weight"), eps)
    if cfg.moe:
        k, lg = cfg.n_experts_used, xm @ mat("ffn_gate_inp.weight").T
        ids, w = moe_route(lg, k)
        if swap is not None:
            ids = np.where(swap[:, None] & (np.arange(k) == k - 1), moe_route(lg, k + 1)[0][:, k:], ids)
            w = np.take_along_axis(softmax(lg), ids, -1)
            w = (w / w.sum(-1, keepdims=True)).astype(np.float32)
        out["ids"], out["weights"] = ids, w
        y = moe_ffn(xm, ids, w, experts(gguf, L)) + shared_expert(
            xm, mat("ffn_gate_shexp.weight"), mat("ffn_up_shexp.weight"), mat("ffn_down_shexp.weight"),
            vec("ffn_gate_inp_shexp.weight"))
    else:
        y = swiglu(xm, mat("ffn_gate.weight"), mat("ffn_up.weight"), mat("ffn_down.weight"))
    out["mtp_ffn_out"] = y
    x2 = out["mtp_post_ffn"] = x1 + y
    hn = out["h_nextn"] = rms_norm(x2, vec("nextn.shared_head_norm.weight"), eps)
    if head is not None:
        out["logits"] = hn @ f32(gguf.rows("output.weight", head)).T
    return out


# ----------------------------------------------------------------------
# The cores' arithmetic (float64 where the kernels are accurate to ~2^-16)
# ----------------------------------------------------------------------


def _silu64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / (1 + np.exp(-x))


def core_gdn_step(*, x: np.ndarray, z: np.ndarray, alpha: np.ndarray, beta: np.ndarray, conv_state: np.ndarray,
                  conv_w: np.ndarray, dt: np.ndarray, ssm_a: np.ndarray, norm: np.ndarray, state: np.ndarray,
                  eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One GDN step of a column as the decode's GDN core computes it (``qwen35.gdn``). x (CH) raw qkv (bf16 values),
    z (HC, 128) bf16 values, alpha / beta / dt / ssm_a (HC) fp32, conv_state (3, CH) bf16 values, conv_w (CH, 4), norm
    (128), state (HC, j, i) llama.cpp's layout. Returns (out (HC, 128) bf16 values, the new conv state, the new
    state)."""
    hc = len(alpha)
    xs = np.concatenate([np.asarray(conv_state, np.float32), np.asarray(x, np.float32)[None]])  # (4, CH)
    y = (np.asarray(conv_w, np.float64).T * xs).sum(0)
    act = _silu64(bf16_round(y))
    q = act[:2 * D].reshape(2, D)
    k = act[2 * D:4 * D].reshape(2, D)
    v = act[4 * D:].reshape(hc, D)
    q = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + eps)
    k = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + eps)
    a64 = np.asarray(alpha, np.float64) + np.asarray(dt, np.float64)
    sp = np.where(a64 > 20, a64, np.log1p(np.exp(np.minimum(a64, 20))))
    d = np.expm1(np.asarray(ssm_a, np.float64) * sp)
    b = 1 / (1 + np.exp(-np.asarray(beta, np.float64)))
    S = np.asarray(state, np.float64).transpose(0, 2, 1)  # (h, i, j)
    kh = np.arange(hc) & 1
    ut = np.einsum("hij,hi->hj", S, k[kh])
    wt = np.einsum("hij,hi->hj", S, q[kh])
    u = ut + d[:, None] * ut
    dl = b[:, None] * (v - u)
    o = (wt + d[:, None] * wt + (k[kh] * q[kh]).sum(-1)[:, None] * dl) / np.sqrt(D)
    S = S + d[:, None, None] * S + k[kh][:, :, None] * dl[:, None, :]
    rms = 1 / np.sqrt((o ** 2).mean(-1, keepdims=True) + eps)
    out = bf16_round(o * rms * np.asarray(norm, np.float64) * _silu64(np.asarray(z, np.float32)))
    new_cs = np.concatenate([xs[1:]])
    return out, new_cs, S.transpose(0, 2, 1).astype(np.float32)


def core_gdn_out(o: np.ndarray, z: np.ndarray, norm: np.ndarray, eps: float) -> np.ndarray:
    """The gated output as the prefill's GdnOut computes it (``qwen35.gdn_prefill``; before the panels' rounding):
    o, z (..., 128) bf16 values."""
    o = np.asarray(o, np.float64)
    inv = 1 / np.sqrt((o ** 2).mean(-1, keepdims=True) + eps)
    return (o * inv * np.asarray(norm, np.float64) * bf16_round(_silu64(z))).astype(np.float32)


def core_gdn_scan(*, qkv: np.ndarray, ab: np.ndarray, conv_state: np.ndarray, conv_w: np.ndarray, dt: np.ndarray,
                  ssm_a: np.ndarray, state: np.ndarray, n: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """One column's scan as the prefill's cores compute it (``qwen35.gdn_prefill``): qkv (T, CH) raw bf16 values, ab
    (T, 2 HC) bf16 values [alpha | beta], conv_state (3, CH), conv_w (CH, 4), dt / ssm_a (HC), state (HC, j, i)
    llama.cpp's layout, n valid tokens. Returns (o (T, HC, 128): the tokens from n on 0, the new state)."""
    T = qkv.shape[0]
    hc = ab.shape[1] // 2
    S = np.asarray(state, np.float64).transpose(0, 2, 1).copy()  # (h, i, j)
    win = np.asarray(conv_state, np.float64)
    o = np.zeros((T, hc, D), np.float32)
    for t in range(n):
        xs = np.concatenate([win, np.asarray(qkv[t], np.float64)[None]])
        y = (np.asarray(conv_w, np.float64).T * xs).sum(0)
        win = xs[1:]
        act = bf16_round(_silu64(bf16_round(y)))  # (the cores' table SiLU: bf16 results within 0.59 ULP)
        q, k, v = act[:2 * D].reshape(2, D), act[2 * D:4 * D].reshape(2, D), act[4 * D:].reshape(hc, D)
        q = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + eps)
        k = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + eps)
        a64 = np.asarray(ab[t, :hc], np.float64) + np.asarray(dt, np.float64)
        sp = np.where(a64 > 20, a64, np.log1p(np.exp(np.minimum(a64, 20))))
        d = np.expm1(np.asarray(ssm_a, np.float64) * sp)
        b = 1 / (1 + np.exp(-np.asarray(ab[t, hc:], np.float64)))
        kh = np.arange(hc) & 1
        ut = np.einsum("hij,hi->hj", S, k[kh])
        wt = np.einsum("hij,hi->hj", S, q[kh])
        dl = b[:, None] * (v - (ut + d[:, None] * ut))
        o[t] = (wt + d[:, None] * wt + (k[kh] * q[kh]).sum(-1)[:, None] * dl) / np.sqrt(D)
        S = S + d[:, None, None] * S + k[kh][:, :, None] * dl[:, None, :]
    return o, S.transpose(0, 2, 1).astype(np.float32)


def core_attention(q_gate: np.ndarray, k_new: np.ndarray, v_new: np.ndarray, k_cache: np.ndarray,
                   v_cache: np.ndarray, *, q_norm: np.ndarray, k_norm: np.ndarray, cs: tuple[np.ndarray, np.ndarray],
                   eps: float) -> tuple[np.ndarray, np.ndarray]:
    """One column's decode attention (``qwen35.attn``) in fp64: q_gate (QC, 512) [q | gate] raw, k_new / v_new (256)
    raw of its KV head, the cache before the position (P, 256) (K normed and rotated), cs: the position's cos, sin
    (``attn.rope_cs``). Returns (o * sigmoid(gate) (QC, 256), K of the position (normed, rotated))."""
    c, s = (np.asarray(x, np.float64) for x in cs)

    def nr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * np.asarray(w, np.float64)
        a, b = y[..., :ROT].copy(), y[..., ROT:2 * ROT].copy()
        y[..., :ROT], y[..., ROT:2 * ROT] = a * c - b * s, a * s + b * c
        return y

    q = nr(q_gate[:, :HD], q_norm)
    kn = nr(k_new, k_norm)
    K = np.concatenate([np.asarray(k_cache, np.float64), kn[None]])
    V = np.concatenate([np.asarray(v_cache, np.float64), np.asarray(v_new, np.float64)[None]])
    sc = q @ K.T / 16
    pr = np.exp(sc - sc.max(-1, keepdims=True))
    o = (pr @ V) / pr.sum(-1, keepdims=True)
    return o / (1 + np.exp(-np.asarray(q_gate[:, HD:], np.float64))), kn
