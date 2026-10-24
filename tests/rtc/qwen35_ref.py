"""References of the qwen35 models (Qwen3.5 / 3.6 / 3.8): llama.cpp goldens and numpy oracles of the blocks.

Goldens: ``golden`` runs a token sequence through llama.cpp on the CPU (``qwen35_dump.cpp``, built against the
llama.cpp tree of ``model_config.LLAMA_CPP_DIR``) and returns the tensors it dumped: per layer the named
intermediates of llama.cpp's graph (``src/models/qwen35.cpp``: e.g. ``attn_norm``, ``linear_attn_qkv_mixed``,
``z``, ``alpha``, ``beta``, ``conv_output_silu``, ``attn_output`` - the GDN core output, but after W_o in the
full-attention layers -, ``new_state``, ``final_output``, ``Qcur_full``, ``Kcur.1`` - after RoPE -,
``attn_pregate``, ``l_out``; a ``qwen35moe`` layer's MoE in ``src/llama-graph.cpp``'s ``build_moe_ffn`` and
``src/models/qwen35moe.cpp``: ``attn_post_norm`` - its input -, ``ffn_moe_logits``, ``ffn_moe_probs``, ``ffn_moe_topk``
- the experts' ids, int32 -, ``ffn_moe_weights_norm``, ``ffn_moe_gate`` / ``_up`` / ``_down`` (tokens, k, ...),
``ffn_moe_weighted``, ``ffn_moe_out``, ``ffn_shexp``, ``shared_expert_gate``, ``ffn_out``), and the logits of every
position; with an MTP file (``mtp``) its block over the same positions (``src/models/qwen35moe.cpp``'s ``graph_mtp``:
``mtp_eh_proj``, ``mtp_attn_out``, ``mtp_attn_residual``, ``mtp_ffn_out``, ``mtp_post_ffn``, ``h_nextn``, its logits:
``g.mtp(name, layer)``, ``g.mtp_logits``), each row fed the model's ``h_nextn`` of the row before. They are cached under ``tests/rtc/refs/<model>/golden_<hash>/`` (gitignored; ``--refresh-refs``
recomputes). llama.cpp quantizes the activations of its Q4_K / Q6_K matmuls to Q8_K, so its tensors carry that
error, and its prompt and one-token graphs differ by ~1% after a few layers: compare layers by tolerance, the model
by KL. In a ``qwen35moe`` model they differ by ~15% at the last layer and in the top 8 of 40-70% of the tokens
there: any rounding change moves a near-tied expert, and the difference grows (a 1e-7 change of the router's logits
reaches the same KL), so its layers are compared where the routing agrees.

Oracles: the exact math of each block in float32 (llama.cpp's CPU ops), on numpy arrays with tokens as the
leading axis: ``rms_norm``, ``conv_seq``, ``l2norm``, ``gdn_gates``, ``gdn_seq`` / ``gdn_step``,
``gated_norm``, ``rope_partial``, ``attention_gqa``, ``q_and_gate``; the MoE FFN: ``moe_route``, ``moe_ffn`` (over
``experts``), ``shared_expert``; a multi-token-prediction block: ``mtp_block``.

The cores' arithmetic (their rounding points, float64 elsewhere): ``core_gdn_step`` (the decode's GDN core),
``core_gdn_scan`` / ``core_gdn_out`` (the prefill's GDN scan and gated output), ``core_attention`` (a decode
attention column).
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

import numpy as np
import pytest
import torch

from artlite.model_config import LLAMA_CPP_DIR, QwenConfig, occamy_gguf, qwen_gguf  # (tests import them from here)
from artlite.operators.base import CACHE_DIR
from artlite.operators.qwen35.attn import HD, ROT
from artlite.operators.qwen35.gdn import D

from . import common

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_SRC = os.path.join(HERE, "qwen35_dump.cpp")


def size_gguf(size: str) -> str:
    """The file a test runs for a size: lmstudio's Q4_K_M of it (``qwen_gguf``), or "occamy": Occamy 1.0 APEX
    (``occamy_gguf``: Qwen3.6-35B-A3B's shapes in Q3_K, IQ3_XXS, Q5_K, Q6_K, Q8_0 and F32 tensors)."""
    return occamy_gguf() if size == "occamy" else qwen_gguf("Q4_K_M", size)


# ----------------------------------------------------------------------
# llama.cpp goldens
# ----------------------------------------------------------------------


def llama_cpp_bin(name: str) -> str:
    """A file of the llama.cpp CPU build (``build/bin/<name>``), or skip the test without one."""
    path = os.path.join(LLAMA_CPP_DIR, "build", "bin", name)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing: no llama.cpp CPU build (ARTLITE_LLAMA_CPP_DIR)")
    return path


def _llama_lib() -> str:
    return llama_cpp_bin("libllama.so")


def _lib_identity() -> str:
    """The llama.cpp build the goldens come from (a rebuild invalidates them)."""
    lib = os.path.realpath(_llama_lib())
    return f"{lib}:{os.path.getsize(lib)}:{int(os.path.getmtime(lib))}"


def dump_tool() -> str:
    """The path of the built ``qwen35_dump`` (compiled on first use into the artlite cache)."""
    with open(DUMP_SRC, "rb") as f:
        key = hashlib.sha256(f.read() + _lib_identity().encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"qwen35_dump_{key}")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        bindir = os.path.dirname(_llama_lib())
        tmp = f"{path}.{os.getpid()}.tmp"
        subprocess.run(["g++", "-O2", "-std=c++17", f"-I{LLAMA_CPP_DIR}/include", f"-I{LLAMA_CPP_DIR}/ggml/include",
                        DUMP_SRC, "-o", tmp, f"-L{bindir}", "-lllama", "-lggml", "-lggml-base",
                        f"-Wl,-rpath,{bindir}"], check=True)
        os.replace(tmp, path)
    return path


def llama_tokenize(model: str, text: str) -> list[int]:
    """llama.cpp's token ids of a text (``llama-tokenize``: special tokens parsed, no BOS)."""
    exe = llama_cpp_bin("llama-tokenize")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([exe, "-m", model, "-f", path, "--ids", "--log-disable", "--no-bos", "--no-escape"],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out.strip().splitlines()[-1])


class Golden:
    """The tensors of one llama.cpp run (memory-mapped .npy files): ``g["attn_norm", 3]``, ``g["result_norm"]``,
    ``g.logits`` ((tokens, vocab)); ``g.meta``: what ran."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)

    def _file(self, name: str, layer: int | None) -> str:
        """"Kcur.1", 3 -> Kcur-3.1.npy (the second tensor of that name in a graph)."""
        base, dot, dup = name.rpartition(".")
        if not (dot and dup.isdigit()):
            base, dup = name, ""
        return os.path.join(self.path, (base if layer is None else f"{base}-{layer}") + (f".{dup}" if dup else "") + ".npy")

    def __contains__(self, key: str | tuple[str, int]) -> bool:
        name, layer = key if isinstance(key, tuple) else (key, None)
        return os.path.exists(self._file(name, layer))

    def __getitem__(self, key: str | tuple[str, int]) -> np.ndarray:
        name, layer = key if isinstance(key, tuple) else (key, None)
        path = self._file(name, layer)
        if not os.path.exists(path):
            have = sorted(f[:-4] for f in os.listdir(self.path) if f.endswith(".npy"))
            raise KeyError(f"{key}: not dumped (have {have})")
        return np.load(path, mmap_mode="r")

    @property
    def logits(self) -> np.ndarray:
        return self["logits"]

    def mtp(self, name: str, layer: int | None = None) -> np.ndarray:
        """A tensor of the MTP block's run (``golden(mtp=)``)."""
        return self[("mtp." + name, layer) if layer is not None else "mtp." + name]

    @property
    def mtp_logits(self) -> np.ndarray:
        return self["mtp.logits"]


def golden(model: str, tokens: np.ndarray | list[int], *, ubatch: int = 0, split: int = -1,
           layers: tuple[int | str, ...] = (), names: str = "", logits: str = "f16", states: str = "last",
           threads: int = 16, mtp: str = "") -> Golden:
    """llama.cpp's tensors for the token sequence (teacher forcing). ``ubatch``: tokens per graph (0: all in
    one; 1: decode graphs); ``split`` = P: the first P tokens in ubatches of ``ubatch``, then one per graph.
    ``layers``: layer indices whose tensors matching ``names`` (a regex over the names without the layer
    suffix) are dumped ("g": the tensors without a layer, "all": every layer). ``logits``: "f32", "f16" or
    "none". ``states``: the recurrent states ("new_state", ...) of the last ubatch, or "all" of them. ``mtp``: an MTP
    file whose block then runs over the same rows (its ``layers`` / ``names`` tensors, its logits)."""
    tokens = np.ascontiguousarray(tokens, np.int32)
    key = (common.model_key(model), hashlib.sha256(tokens.tobytes()).hexdigest(), ubatch, split,
           tuple(str(x) for x in layers), names, logits, states, _lib_identity())
    if mtp:
        key += (common.model_key(mtp),)
    h = hashlib.sha256(repr(key).encode())
    with open(DUMP_SRC, "rb") as f:
        h.update(f.read())
    folder = common.model_refs(model)
    path = os.path.join(folder, f"golden_{h.hexdigest()[:16]}")
    if os.path.exists(os.path.join(path, "meta.json")) and not common.REFRESH_REFS:
        return Golden(path)
    os.makedirs(folder, exist_ok=True)
    tool = dump_tool()
    tmp = tempfile.mkdtemp(dir=folder, prefix=".tmp_")
    try:
        tokens.tofile(os.path.join(tmp, "tokens.i32"))
        cmd = [tool, "-m", model, "-t", os.path.join(tmp, "tokens.i32"), "-o", tmp, "-u", str(ubatch),
               "--split", str(split), "--logits", logits, "--states", states, "--threads", str(threads)]
        if layers:
            cmd += ["-l", ",".join(str(x) for x in layers), "-n", names or ".*"]
        if mtp:
            cmd += ["--mtp", mtp]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"{' '.join(cmd)}: exit {r.returncode}\n{r.stderr[-4000:]}")
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return Golden(path)


# ----------------------------------------------------------------------
# Oracles (float32, tokens first)
# ----------------------------------------------------------------------


def f32(x: object) -> np.ndarray:
    return np.asarray(x, np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    return x / (np.float32(1) + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    return np.float32(1) / (np.float32(1) + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    """llama.cpp's: x above 20, else log(1 + e^x)."""
    x = f32(x)
    with np.errstate(over="ignore"):
        return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(np.float32)


def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    """x / sqrt(mean(x^2) + eps) * w over the last axis (every qwen35 norm weight already holds its 1 +)."""
    x = f32(x)
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    y = (x / np.sqrt(ms + eps)).astype(np.float32)
    return y if w is None else y * f32(w)


def l2norm(x: np.ndarray, eps: float) -> np.ndarray:
    """x / sqrt(sum(x^2) + eps) over the last axis (the GDN's q / k norm)."""
    x = f32(x)
    ss = np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ss + eps)).astype(np.float32)


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


def softmax(x: np.ndarray) -> np.ndarray:
    """Over the last axis, as ``ggml_soft_max``: the maximum subtracted, the sum in float64."""
    x = f32(x)
    e = np.exp(x - x.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True, dtype=np.float64)).astype(np.float32)


def moe_route(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The router's choice (``build_moe_ffn``: softmax gating, the weights renormalized, no scale, no groups): logits
    (T, E) -> the experts of the k largest probabilities (T, k), largest first, and their weights p_e / their sum."""
    p = softmax(logits)
    ids = np.argsort(-p, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(p, ids, -1)
    return ids, (w / np.maximum(w.sum(-1, keepdims=True), np.float32(6.103515625e-5))).astype(np.float32)


def swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """A SwiGLU FFN: down (silu(gate x) * up x); x (T, H), gate / up (F, H), down (H, F)."""
    x = f32(x)
    return (silu(x @ f32(gate).T) * (x @ f32(up).T)) @ f32(down).T


def experts(gguf: object, layer: int) -> Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A ``qwen35moe`` layer's expert e as fp32 (gate (F, H), up (F, H), down (H, F)): rows e F .. (e + 1) F of the
    GGUF's ``ffn_gate_exps`` / ``ffn_up_exps`` (3-D tensors, an expert's matrix contiguous), e H .. of
    ``ffn_down_exps``."""
    b = f"blk.{layer}."
    F, H = gguf.tensor(b + "ffn_down_exps.weight").cols, gguf.tensor(b + "ffn_gate_exps.weight").cols

    def get(e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def rows(name: str, n: int) -> np.ndarray:
            return gguf.rows(b + name, np.arange(e * n, (e + 1) * n)).numpy()

        return rows("ffn_gate_exps.weight", F), rows("ffn_up_exps.weight", F), rows("ffn_down_exps.weight", H)

    return get


def moe_ffn(x: np.ndarray, ids: np.ndarray, weights: np.ndarray,
            expert: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """The routed experts' sum (``ffn_moe_out``): y_t = sum_i weights[t, i] swiglu_e(x_t), e = ids[t, i] (``moe_route``);
    x (T, H); ``expert(e)``: its (gate, up, down) (``experts``)."""
    x = f32(x)
    y = np.zeros_like(x)
    for e in np.unique(ids):
        t, i = np.nonzero(ids == e)
        y[t] += f32(weights)[t, i][:, None] * swiglu(x[t], *expert(int(e)))
    return y


def shared_expert(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray,
                  gate_inp: np.ndarray) -> np.ndarray:
    """The shared expert times its gate (``ffn_shexp_gated``): sigmoid(gate_inp . x) swiglu(x); gate_inp (H,) (the
    GGUF's ``ffn_gate_inp_shexp``). A layer's FFN output is ``moe_ffn`` + this."""
    return sigmoid(f32(x) @ f32(gate_inp).reshape(-1, 1)) * swiglu(x, gate, up, down)


def mtp_block(gguf: object, cfg: QwenConfig, emb: np.ndarray, h: np.ndarray, pos: np.ndarray, *,
              head_rows: int = 0) -> dict[str, np.ndarray]:
    """A ``qwen35moe`` MTP block (llama.cpp's ``graph_mtp``, the block at index ``cfg.num_hidden_layers`` of an MTP
    file) over rows of consecutive positions ``pos`` from 0 (T,), each the cache of the rows after it: emb (T, H) the
    rows' token embeddings, h (T, H) the model's output-normed hidden state of the position before each (zeros at 0).
    Returns its tensors by llama.cpp's names (``mtp_eh_proj``: the layer's input x0, ``mtp_attn_residual``,
    ``mtp_attn_post_norm``, ``mtp_ffn_out``, ``mtp_post_ffn``, ``h_nextn``: the next draft's h), the experts' ids and
    weights (``ids``, ``weights``) and the logits of the head's first ``head_rows`` rows (0: none)."""
    L, eps = cfg.num_hidden_layers, cfg.rms_norm_eps
    b = f"blk.{L}."

    def mat(name: str) -> np.ndarray:
        t = gguf.tensor(b + name)
        return f32(t.dequantize().numpy()).reshape(t.rows, t.cols)

    def vec(name: str) -> np.ndarray:
        return f32(gguf.tensor(b + name).dequantize().numpy()).reshape(-1)

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
    ids, w = moe_route(xm @ mat("ffn_gate_inp.weight").T, cfg.n_experts_used)
    out["ids"], out["weights"] = ids, w
    y = moe_ffn(xm, ids, w, experts(gguf, L)) + shared_expert(
        xm, mat("ffn_gate_shexp.weight"), mat("ffn_up_shexp.weight"), mat("ffn_down_shexp.weight"),
        vec("ffn_gate_inp_shexp.weight"))
    out["mtp_ffn_out"] = y
    x2 = out["mtp_post_ffn"] = x1 + y
    hn = out["h_nextn"] = rms_norm(x2, vec("nextn.shared_head_norm.weight"), eps)
    if head_rows:
        out["logits"] = hn @ f32(gguf.rows("output.weight", np.arange(head_rows)).numpy()).T
    return out


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max |b|."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# ----------------------------------------------------------------------
# The cores' arithmetic (float64 where the kernels are accurate to ~2^-16)
# ----------------------------------------------------------------------


def _bf(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


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
    act = _silu64(_bf(y))
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
    out = _bf(o * rms * np.asarray(norm, np.float64) * _silu64(np.asarray(z, np.float32)))
    new_cs = np.concatenate([xs[1:]])
    return out, new_cs, S.transpose(0, 2, 1).astype(np.float32)


def core_gdn_out(o: np.ndarray, z: np.ndarray, norm: np.ndarray, eps: float) -> np.ndarray:
    """The gated output as the prefill's GdnOut computes it (``qwen35.gdn_prefill``; before the panels' rounding):
    o, z (..., 128) bf16 values."""
    o = np.asarray(o, np.float64)
    inv = 1 / np.sqrt((o ** 2).mean(-1, keepdims=True) + eps)
    return (o * inv * np.asarray(norm, np.float64) * _bf(_silu64(z))).astype(np.float32)


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
        act = _bf(_silu64(_bf(y)))  # (the cores' table SiLU: bf16 results within 0.59 ULP)
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
