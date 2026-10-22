"""The host side of an MoE layer's prefill: the routing of a chunk's tokens (the router GEMM's logits -> the top k of
every token and their weights: ``route``) and the experts' chunks dealt to the columns (``Plan``), for the columns'
control streams of the work-list operators (``expert_gemm``, ``gather``: their ``stream``).

The routing is llama.cpp's (``build_moe_ffn``: softmax gating, the top k, the weights renormalized): the k largest
logits (ties to the lower id: the decode's ``route.top_k``), w = e^(l - l_max) over the k, normalized; the shared
expert's gate sigmoid(l_s).

A chunk is up to ``CHUNK`` of one expert's (token, slot) pairs, in token order; an expert of r pairs takes ceil(r /
CHUNK) chunks (its last one partial: the padding rows repeat its first token, their outputs never read). The chunks
go to the columns in turn (expert by expert, each column at most one chunk more than another: every chunk streams one
expert's weights, the work the columns share). Column c's chunk k reads its tokens' rows into its slot k of the
gathered inputs (XP) and writes its slot k of H and of Y; the combine finds token t's slot i output at ``pos[t, i]``
(chunk slots of Y, column-major: column c's slots after column c - 1's capacity; ``ExpertGemmOperator.y_offset``)."""

from dataclasses import dataclass

import numpy as np

from ... import _native
from ..design import COLS
from .expert_gemm import CHUNK


def route(logits: np.ndarray, k: int, *, E: int = 0, per_col: int = 0, pitch: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(ids (T, k) int32, largest first, ties to the lower id; weights (T, k) float32 summing to 1) of fp32 logits:
    (T, E), or rows of the router GEMM's layout (expert e at (e / per_col) pitch + e % per_col of its row)
    (``_native.moe_route``)."""
    lg = np.ascontiguousarray(logits, np.float32)
    T = lg.shape[0]
    E = E or lg.shape[1]
    ids, w = np.empty((T, k), np.int32), np.empty((T, k), np.float32)
    _native.moe_route(lg.reshape(-1), T, lg.shape[1], per_col or E, pitch, E, ids, w)
    return ids, w


@dataclass
class Plan:
    """An MoE layer's chunks for T tokens' routing (``ids`` (T, k)), dealt to the columns of ``cap`` slots each:
    ``experts[c]`` the experts of column c's chunks, ``sizes[c]`` their pairs, ``rows[c]`` (n_c, CHUNK) their tokens
    (padding: the chunk's first), ``pos`` (T, k) the row (c cap CHUNK + k CHUNK + j) of each (token, slot)."""

    experts: list[np.ndarray]
    sizes: list[np.ndarray]
    rows: list[np.ndarray]
    pos: np.ndarray

    @classmethod
    def of(cls, ids: np.ndarray, *, cap: int, E: int) -> "Plan":
        """(``_native.moe_plan``: E experts)"""
        ids = np.ascontiguousarray(ids, np.int32)
        ex, counts, sizes = np.empty((COLS, cap), np.int32), np.empty(COLS, np.int32), np.empty((COLS, cap), np.int32)
        rows, pos = np.empty((COLS, cap, CHUNK), np.int32), np.empty(ids.shape, np.int64)
        _native.moe_plan(ids, E, CHUNK, ex, counts, sizes, rows, pos)
        return cls(*([a[c, :counts[c]] for c in range(COLS)] for a in (ex, sizes, rows)), pos)

    @property
    def counts(self) -> list[int]:
        return [len(e) for e in self.experts]

