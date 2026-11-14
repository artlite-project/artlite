"""The prefill attention's shapes: ``Shape`` (the head dims, panels and cache objects of a model's attention), the
scratchpad parameters the cores read, and the cos / sin tables (module docstring of ``prefill.attn``)."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ...decode.kv import Records
from ...operators.design import COLS
from ...operators.util import cos_sin_rows
from .layout import L1_LIMIT, core_bytes

P_CS = "pa_cs"  # bf16 offset of the chunk's rows in the cos / sin table (2 hd p0)
P_KV = "pa_kv"  # bf16 offset of the chunk's first cache group in a KV head's cache
P_P8 = "pa_p8"  # p0 / 8 (the attention cores' masks)
P_FG = "pa_fg"  # the cache groups every pass streams first (p0 / pg + 1: the history and the chunk's first)
P_FQ = "pa_fq"  # their chunks of CR records (the cores' count)
P_FO = "pa_fo"  # bf16 offset of the chunk's second group
P_G0 = "pa_g0"  # bf16 offset of the chunk's first group (``AttnOperator(ring=)``)
P_FH = "pa_fh"  # the history's chunks of CR records (``AttnOperator(ring=)``: the cores')
P_HK = "pa_hk"  # the chunk's first keys every row sees (packed: a lead, ``AttnOperator.params``)
P_SW = "pa_sw"  # (sliding window) bf16 offset of the chunk's first cache group from a KV head's group 0




@dataclass(frozen=True)
class Shape:
    """The attention's shapes: head_dim, query heads per column, K panels per record, the cache's object bytes, Q panels
    per row block (``qp``: 1 single bfp16, 2 hi + lo), P panels per row block and record (``pp``: 1, 2), columns per KV
    head (``per``: 1, Llama's; 2, LFM3's: the KV head's K and V halves in the two columns' G slices), a sigmoid output
    gate per head (``gate``: its logits the G slices' block "g", per token group the column's heads' gates after their
    Q units in QS)."""

    hd: int
    qh: int
    kp: int
    a_bytes: int
    qp: int = 1
    pp: int = 1
    per: int = 1
    gate: bool = False

    @cached_property
    def kvh(self) -> int:  # KV heads (of the array's columns)
        return COLS // self.per

    @cached_property
    def gw(self) -> int:  # bytes of a token group's gates of a column (8 tokens x qh bf16)
        return 16 * self.qh if self.gate else 0

    @cached_property
    def tgu(self) -> int:  # bytes of a token group of a column in QS: its heads' Q units, its gates
        return self.qh * self.qunit + self.gw

    @cached_property
    def nd(self) -> int:
        return self.hd // 8

    @cached_property
    def rec(self) -> Records:
        return Records(self.hd, self.kp)

    @cached_property
    def unit(self) -> int:  # bytes of a stream unit (module docstring)
        return 144 * self.nd

    @cached_property
    def qunit(self) -> int:  # bytes of a row block's Q panels (8 queries x 8 dims per panel; hi + lo: [hi ND | lo ND])
        return 72 * self.nd * self.qp

    @cached_property
    def rg(self) -> int:  # records per cache group
        return self.rec.rg(self.a_bytes)

    @cached_property
    def pg(self) -> int:  # positions per cache group
        return 8 * self.rg

    @cached_property
    def objects(self) -> int:  # cache objects per group
        return self.kp + 1

    @cached_property
    def used(self) -> int:  # bytes of an object's records (the rest padding)
        return self.rec.kper(self.a_bytes) * self.rec.k_bytes

    @cached_property
    def group(self) -> int:  # bytes of a cache group
        return self.objects * self.a_bytes

    @cached_property
    def scale(self) -> float:  # the score constant (log2 units)
        return 1 / math.sqrt(self.hd) / math.log(2)

    @cached_property
    def blocks(self) -> tuple[int, int]:
        """(QB, CR): the most row blocks per head that fit L1 (2 or 1), chunks of 4 records."""
        return next((qb, 4) for qb in (2, 1) if core_bytes(self, qb, 4) <= L1_LIMIT)


def cs_rows(inv_freq: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """(n, hd) fp32: per position [cos hd / 2 | sin hd / 2] of the pairs' angles position inv_freq[i] (the decode's
    ``decode.attn.rope_tables``)."""
    return cos_sin_rows(np.outer(np.asarray(positions, np.float64), np.asarray(inv_freq, np.float64)))


def cs_table(inv_freq: np.ndarray, n: int) -> np.ndarray:
    """``cs_rows`` of the positions 0 .. n - 1."""
    return cs_rows(inv_freq, np.arange(n))
