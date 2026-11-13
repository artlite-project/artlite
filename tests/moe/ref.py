"""The MoE FFN's oracles (llama.cpp's ``build_moe_ffn``) in float32 numpy: the router's choice, an expert's SwiGLU, the routed
experts' sum, the shared expert."""

from collections.abc import Callable

import numpy as np

from ..np_ops import f32, sigmoid, silu, softmax


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
            return gguf.rows(b + name, np.arange(e * n, (e + 1) * n))

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
