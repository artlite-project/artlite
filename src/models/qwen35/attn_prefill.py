"""The prefill of the qwen35 full-attention layers: a T-token chunk at position p0 (a multiple of 32) from the attn_in
GEMMs' output G (``prefill.gemm``: per column [q + gate of its QC heads | K part | V part]: KV head h's dims in the NPH
= 8 / KV heads columns NPH h ..) to the attn_out GEMM's A.

``AttnPrepOperator``: per column c, cores 2 .. 1 + QC one per q head (the column's heads QC c ..), core 2 + QC the
column's KV role. Per 8 tokens a Q core turns its head's rows into the attention's operands: q / rms(q) q_norm, the
NeoX rotation of dims 0 .. 63, x log2(e) / 16 (scores in log2 units), as bfp16 panels of 8 queries x 8 dims, per d-block
hi then lo (4608 B), then sigmoid(gate) (256 bf16 per query, padded to 4608 B: per d-block 144 B as a Q panel pair).
QS holds per column and token group [the column's heads' Q panels | their sigmoid(gate)] (``SLOT`` bytes per head each).
The 2 x KV heads jobs of the KV cores (``AttnPrepOperator.kv_job``: one every 8 / (2 KV heads) columns; 9B / 27B every
column, MiniCPM-V every other): K head h (rms(k) k_norm, rotated) as records of blocks of 8 dims, then V head h as records of
blocks over the 8 positions, in the decode's cache layout (``decode.kv``, d-block major: two K records [d-block][record 2],
two V records [d-block][record 2]): the chunk's groups from p0 / 32 on (scratchpad
parameter ``QA_KV``), so the decode continues from them. cos / sin of the
tokens' positions from a table (``cs_table``; parameter ``QA_CS``: the row of the chunk's first token): the static rows
of every text position, or rows the host writes past them for a chunk of image tokens (M-RoPE: ``cs_rows``).

``PrefillAttnOperator``: per column 4 cores, passes of 32 QG queries (QG = 2 when a column holds one q head, the L1 of
the 9B's two heads: half the history's re-streams, else 1); core r takes the QG token groups QG r .. of the pass for
each of the column's heads (QC QG row panels: an O of QC QG x 8 KB) and streams the cache groups up to the pass's (the
KV head's stream broadcast to the 4 NPH cores of its columns; with two row panels per core and 4 columns per KV head on
two streams, ``two_streams``: one stream's memtile channel bound the history's re-streams): per K record the scores on
the MMUL (Q hi + lo against K, two products: one loop over a pair of records' panels [d-block][record 2], the Q slots'
[d-block][hi | lo]; P V over the V object's pairs of records [d-block][record 2]), per group of 32 positions the softmax (a
row's maximum is raised, rescaling o and l, only when the group's exceeds it by 2^16: the per-dimension norm weights
leave scores unbounded), P = 2^(s - m) by the vexp2 exp2 (vexp2 builds an exact 2^n; 0.59 ULP) as bfp16 panels (hi + lo
with ``pp`` 2: the readout models'), o += P V; at the pass's end o / l sigmoid(gate) as the attn_out GEMM's bfp16 A
panels. Within 0.7% of the exact attention of its own operands (hi + lo P: its output's own rounding, against 1.5-3x of
it), ~1% rms of llama.cpp's attn_gated.

Time (T = 512, per layer alone, at p0 = 0 / 4K / 16K): the 27B 1.38 / 6.89 / 22.1 ms (its compute compiled out at
16K: 20.6, the KV re-streams of the 32-query passes at the DDR ceiling), the 1B 0.97 / 2.47 / 7.23 (5.99: the cores'
QK binds), the 35B-A3B (two streams) 1.12 / 3.44 / 10.6 (8.8).
"""

import math

import numpy as np
from aie.dialects.aiex import npu_write32
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.model_config import QwenConfig
from artlite.npu_regs import window_counter_writes
from artlite.operators import kmath, templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.bfp16 import PANEL_BYTES, PANEL_ELEMS, panel_elems
from artlite.operators.chain import ChainPart, chained_fill
from artlite.operators.design import (
    COLS,
    MAX_BD_ITER,
    MAX_INNER_WORDS,
    SHIM_QUEUE,
    check_bd_iterations,
    range32,
    stream_walk,
)
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.l1_layout import REGION, pin_l1_buffers
from artlite.operators.util import Attr, arg_elems, bf16_hilo, cos_sin_rows, mask64, regions
from artlite.prefill.gemm import Layout

from . import attn as AT

HD = AT.HD
ND = AT.ND
SLOT = 2 * AT.REC_K  # bytes per head of a token group's Q panels (and of its sigmoid(gate): 4096 B used); of a pair of records
GROUP = AT.OBJS * AT.OBJ_BYTES  # bytes of a cache group (32 positions: a K object, a V object)
OBJ_ELEMS = AT.OBJ_BYTES // 2  # bf16 of a cache object
A_SET = ND * PANEL_ELEMS  # bf16 of a set of 32 A panels (ND panels of 72 B)
PREP_ROW = 576  # bf16 of a row of the prep's QS and KV taps (a BD's inner dimension: at most ``MAX_INNER_WORDS`` words)
PASSES_IN_FLIGHT = 2  # a pass queues one task per shim channel: its fills wait for the pass before the last
assert PREP_ROW // 2 <= MAX_INNER_WORDS and PASSES_IN_FLIGHT <= SHIM_QUEUE
# Scratchpad parameters. QA_KV and QA_OFF are one per operator size ("qa_kv<T>"): a parameter's range is the tightest
# of the operators that read it, and a smaller chunk may start where a larger one would overrun the cache.
QA_CS = "qa_cs"  # bf16 offset of the chunk's rows in the cos / sin table (128 p0)
QA_KV = "qa_kv"  # bf16 offset of the chunk's first cache group (GROUP / 2 * p0 / 32)
QA_N = "qa_n"  # p0 / 32 + 1 (cache groups every attention pass streams first; the cores' count); non-causal: the
# prompt's groups
QA_NK = "qa_nk"  # (non-causal) the prompt's keys in its last group (1 .. 32)
QA_OFF = "qa_off"  # bf16 offset of the chunk's second cache group
PW = 2 * 512  # bytes of the attention softmax's scratch (P and f)


def pass_groups(cfg: QwenConfig, T: int) -> int:
    """QG: the token groups per core and head in a 32 QG-token attention pass (2 when a column holds one q head and T
    allows, the L1 of the 9B's two heads: half the history's re-streams); QS holds the slots pass by pass."""
    return 2 if cfg.num_attention_heads == COLS and T % 64 == 0 else 1


def two_streams(cfg: QwenConfig, T: int) -> bool:
    """Whether the prefill attention of a T-token chunk takes each KV head's cache on two streams (two row panels per
    core: one q head per column and QG = 2, or two q heads; a KV head's 4 columns at least, each stream from two
    shims)."""
    qc = cfg.num_attention_heads // COLS
    return COLS // cfg.num_key_value_heads >= 4 and qc * pass_groups(cfg, T) == 2


def part_walk() -> TensorAccessPattern:
    """The two-stream attention's (``two_streams``) memtile walk of one shim channel's part of a stream object (the two
    parts make the object): d-blocks 8 i .. of a stream's half of a K or V object's two pairs of records or of two slots
    ([piece 2][d-block 8][144 B]) -> [d-block][piece][144 B]. (One stream's objects are a K pair, a V pair or a slot: as
    they lie in DDR.)"""
    return stream_walk(SLOT // 4, [(ND // 4, 72), (2, SLOT // 8), (72, 1)])


def cs_rows(cfg: QwenConfig, positions: np.ndarray | range) -> np.ndarray:
    """(n, 64) fp32: per position (text, or M-RoPE (T, H, W)) [cos 32 | sin 32] of the rotated pairs
    (``QwenConfig.rope_theta``)."""
    return cos_sin_rows(cfg.rope_theta(positions))


def cs_table(cfg: QwenConfig, n: int) -> np.ndarray:
    """(n, 64) fp32: ``cs_rows`` of the text positions 0 .. n - 1."""
    return cs_rows(cfg, range(n))


_PREP = templating.load("models/qwen35/attn_prefill.prep.cc.jinja")


class AttnPrepOperator(ArtcOperator):
    """The attention prep of a chunk (module docstring). Runtime sequence: G (attn_in's output, ``layout``), CS (the
    cos / sin table of ``cs_positions`` rows, as bf16 pairs), NRM (q_norm | k_norm fp32), QS (per column, token group
    and head a ``SLOT``, pass-major: [32 QG-token pass][column][token group 4 QG][Q | gate][head], ``pass_groups``;
    a sub-chunk's part of a chunk's QS contiguous),
    KV (one layer's cache: the KV heads' of ``attn.AttnShape.head_bytes``). Parameters: ``QA_CS`` = 128 x the table row
    of the first token (default p0), ``QA_KV`` (T's) = GROUP / 2 * p0 / 32 (``params``)."""

    STACK = 3072
    pfx = kernel_prefix("qap")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, head_bytes: int, cs_positions: int):
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * AT.ROT and COLS % (2 * self.kvh) == 0
        assert T % 32 == 0, "whole cache groups"
        check_bd_iterations(T // 8, "AttnPrep")
        assert {"q", "k", "v"} <= set(layout.off)
        self.T, self.cfg, self.layout = T, cfg, layout
        self.nph, self.per = COLS // self.kvh, self.kvh * HD // COLS  # (a KV head's columns, its dims in one)
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 3, "a core per q head and the KV core in a column"
        self.head_bytes, self.cs_positions = head_bytes, cs_positions
        self.qg = pass_groups(cfg, T)
        assert T % (32 * self.qg) == 0
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return {"nrm": 0, "row": 8 * HD, "stage": 12 * HD, "kf": 44 * HD, "size": 44 * HD + AT.REC_K}

    def get_kernel_src(self) -> str:
        p = self.pfx
        qs = bf16_hilo(math.log2(math.e) / 16)
        return _PREP.render(p=p, HD=HD, ND=ND, w=Attr(self.workspace()), eps=repr(self.cfg.rms_norm_eps), qs=qs,
                            tmax=repr(kmath.SIGMOID.t_max))

    def kv_job(self, c: int) -> tuple[str, int] | None:
        """Column c's KV core's job: ("k" or "v", the KV head), or None (one every COLS / (2 KV heads) columns)."""
        step = COLS // (2 * self.kvh)
        if c % step:
            return None
        j = c // step
        return ("k" if j < self.kvh else "v"), j % self.kvh

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * self.layout.N,), bfloat16),
                ArgSpec("in", (self.cs_positions * 128,), bfloat16),
                ArgSpec("in", (2 * HD * 2,), bfloat16),
                ArgSpec("out", (self.n_qs,), bfloat16),
                ArgSpec("out", (self.kvh * self.head_bytes // 2,), bfloat16)]

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument."""
        return COLS * self.T // 8 * self.qc * SLOT

    def params(self, p0: int, cs_row: int | None = None) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 whose tokens' cos / sin start at table row ``cs_row``
        (default p0)."""
        r = p0 if cs_row is None else cs_row
        assert p0 % 32 == 0 and 0 <= r and r + self.T <= self.cs_positions
        return {QA_CS: 128 * r, f"{QA_KV}{self.T}": GROUP // 2 * (p0 // 32)}

    def get_mlir_artifact(self) -> Module:
        p, T, qc, lay = self.pfx, self.T, self.qc, self.layout
        N, pitch = lay.N, lay.pitch
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(8 * 2 * HD,), bf]  # 8 rows of [q | gate]
        kv_ty = np.ndarray[(8 * HD,), bf]  # 8 rows of a KV head
        in_ty = np.ndarray[(qc * 8 * 2 * HD + 8 * HD,), bf]
        inq_ty = np.ndarray[(qc * 8 * 2 * HD,), bf]  # (a column without a KV job)
        jobs = {c: self.kv_job(c) for c in range(COLS) if self.kv_job(c) is not None}
        cs_ty = np.ndarray[(8 * 128,), bf]  # 8 positions x 64 fp32 (the norms: 512 fp32)
        slot_ty = np.ndarray[(SLOT // 2,), bf]
        qs_ty = np.ndarray[(qc * SLOT // 2,), bf]
        vr_ty = np.ndarray[(AT.REC_V,), bf]  # (a pair of K or V records)
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_q = Kernel(f"{p}_q", "kernel.o", [row_ty, cs_ty, ws_ty, slot_ty])
        k_sg = Kernel(f"{p}_sg", "kernel.o", [row_ty, ws_ty, slot_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [kv_ty, cs_ty, ws_ty, vr_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [kv_ty, cs_ty, ws_ty, vr_ty, np.int32])
        p_cs, p_kv = ScratchpadParameter(QA_CS, np.int32), ScratchpadParameter(f"{QA_KV}{T}", np.int32)
        n_tg = T // 8

        a3 = [ObjectFifo(in_ty if c in jobs else inq_ty, name=f"qp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([8 * 2 * HD * h for h in range(qc + (c in jobs))], tile=Tile(c, 1),
                                 depths=[2] * (qc + (c in jobs)), obj_types=[row_ty] * qc + [kv_ty] * (c in jobs),
                                 names=[f"qp_ah{c}_{h}" for h in range(qc + (c in jobs))]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"qp_b{c}", depth=2) for c in range(COLS)]
        o3 = [ObjectFifo(qs_ty, name=f"qp_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([SLOT // 2 * h for h in range(qc)], tile=Tile(c, 1), depths=[2] * qc,
                                obj_types=[slot_ty] * qc, names=[f"qp_oh{c}_{h}" for h in range(qc)])
              for c in range(COLS)]
        k = {c: ObjectFifo(vr_ty, name=f"qp_k{c}", depth=2) for c in jobs}

        def core(a_h, b_h, o_h, prm, fn, ws, sg=None):
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    o = o_h.acquire(1)
                    fn(x, cs, ws, o)
                    o_h.release(1)
                    if sg is not None:  # (a Q core: then the token group's sigmoid(gate))
                        o = o_h.acquire(1)
                        sg(x, ws, o)
                        o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        def core_q(a_h, b_h, o_h, prm, fn, sg, ws):
            core(a_h, b_h, o_h, prm, fn, ws, sg)

        def core_v(a_h, b_h, o_h, prm, fn, ws):  # (a KV job: two token groups' records per object)
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg // 2):
                    o = o_h.acquire(1)
                    for half in range(2):
                        x = a_h.acquire(1)
                        cs = b_h.acquire(1)
                        fn(x, cs, ws, o, half)
                        b_h.release(1)
                        a_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for h in range(qc + (c in jobs)):
                ws = Buffer(ws_ty, name=f"qp_ws{c}_{h}")
                if h < qc:
                    fn, args = core_q, [a1[c][h].cons(), b[c].cons(), o1[c][h].prod(), k_prm, k_q, k_sg, ws]
                else:
                    args = [a1[c][h].cons(), b[c].cons(), k[c].prod(), k_prm, k_k if jobs[c][0] == "k" else k_v, ws]
                    fn = core_v
                workers.append(Worker(fn, args, tile=Tile(c, 2 + h), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_qs, n_kv = arg_elems(self, 1), arg_elems(self, 3), arg_elems(self, 4)
        nph, per = self.nph, self.per
        blk = 4 * self.qg * qc * SLOT  # (bf16 of a column's slots in a pass)
        hb, g2, o2 = self.head_bytes // 2, GROUP // 2, AT.OBJ_BYTES // 2

        def seq(G, CS, NRM, QS, KV, ah, bh, oh, khs):
            kh = dict(zip(k, khs))
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                taps = [TensorAccessPattern((T * N,), c * pitch + lay.off["q"], [n_tg, qc, 8, 2 * HD],
                                            [8 * N, 2 * HD, N, 1])]
                if c in jobs:  # (the KV head's rows: its dims in its NPH columns)
                    kind, h = jobs[c]
                    taps.append(TensorAccessPattern((T * N,), nph * h * pitch + lay.off[kind], [n_tg, 8, nph, per],
                                                    [8 * N, N, pitch, 1]))
                chained_fill(ah[c], G, taps, tg)
                bh[c].fill(NRM, TensorAccessPattern((2 * HD * 2,), 0, [1, 1, 1, 2 * HD * 2], [0, 0, 0, 1]), group=tg)
                bh[c].fill(CS, TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 128], [0, 0, 0, 1]), offset_parameter=p_cs,
                           group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * blk,
                                                    [1, n_tg // (4 * self.qg), blk // PREP_ROW, PREP_ROW],
                                                    [0, COLS * blk, PREP_ROW, 1]), group=tg, wait=True)
                if c not in jobs:
                    continue
                kind, h = jobs[c]
                # (the K or V object of group g + i: its 4 records)
                tap = TensorAccessPattern((n_kv,), h * hb + (kind == "v") * o2, [1, T // 32, 8, PREP_ROW],
                                          [0, g2, PREP_ROW, 1])
                kh[c].drain(KV, tap, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(2 * HD * 2,), bf],
                           np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in k.items()]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = templating.load("models/qwen35/attn_prefill.cc.jinja")


class PrefillAttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``AttnPrepOperator``). Column c (q heads QC c .., KV head c // NPH), core row r; passes of 32 QG queries (``qg``):
    in pass b the core takes the QG token groups 4 QG b + QG r .. of each of the column's heads (QC QG row panels),
    streams the cache groups 0 .. p0 / 32 + QG (b + 1) - 1 (the KV head's stream broadcast to the 4 NPH cores of its
    columns) and writes their outputs as the attn_out GEMM's A panels. Runtime sequence: QS (the prep's), KV (one layer's cache), A (T x heads x 256 as bfp16
    panels [T / 8][heads x 32]). Parameters ``QA_N`` = p0 / 32 + 1 (the history groups every pass streams first, and the
    cores' trip count), ``QA_OFF`` (T's): the bf16 offset of the chunk's second group (``params``).

    Non-causal (``causal`` False: ``QwenConfig.causal``, vllm-d3's decision models): a prompt of n tokens in one chunk
    at position 0, every pass streams the prompt's groups (``QA_N`` of them) and each query sees all of its keys, those
    after the prompt's in its last group masked (``QA_NK``: the prompt's keys there).

    Packed (``pack``: several inputs, segments, in a chunk): each row sees the history (the positions before p0: a
    prefix every segment continues; causal only) and its segment's keys from its first up to itself (causal) or to its
    last (non-causal, at position 0), the others masked (-16384: P 0 against any of its own; a row's groups before its
    segment, all masked, rescaled to 0 by its first own one); the SEG argument (``seg``) per pass, an object after the Q
    slots (two streams: a pair), holds the rows' first and last keys and the history's groups."""

    STACK = 3072
    pfx = kernel_prefix("qaa")

    def __init__(self, *, T: int, cfg: QwenConfig, head_bytes: int, ablate: tuple[str, ...] = (), causal: bool = True,
                 pp: int = 1, pack: bool = False):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv", packed: "mask": wrong
        results); "ev_qk", "ev_soft", "ev_pv": an event window around each call of the kernel, the cores' counters
        programmed (``npu_regs.window_counter_writes``; two streams only). ``pp``: P as single bfp16 panels (1) or
        hi + lo (2: P V's error gone; one stream: the lo pair in the same loop, two streams: P V once per part)."""
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and COLS % self.kvh == 0
        assert T % 32 == 0
        self.T, self.cfg, self.head_bytes, self.causal, self.pp, self.pack = T, cfg, head_bytes, causal, pp, pack
        self.nph = COLS // self.kvh
        self.ablate = tuple(ablate)
        self.qc = cfg.num_attention_heads // COLS
        self.qg = pass_groups(cfg, T)  # (query groups per core and head)
        self.R = self.qc * self.qg  # (row panels)
        self.H = cfg.num_attention_heads
        # (two row panels per core, a KV head's 4 columns: each KV head's cache on two streams, half of every object
        # each, so a core takes it on both input channels; its Q and gates ride ahead and behind on them; else one
        # stream, Q apart)
        self.two = two_streams(cfg, T)
        self.ob = self.R <= 2 and not self.two  # (o's second copy fits: regions 2 and 3)
        super().__init__()

    def workspace(self) -> dict[str, int]:
        R = self.R
        sizes = [("sc", R * 1024), ("m", R * 256), ("b", R * 256), ("l", R * 256)] + \
            ([("lh", 96 * self.qg), ("lv", 256 * self.qg), ("jl", 128)] if self.pack else []) + [("o", R * 8192)]
        return regions(sizes + [("ob", R * 8192)] if self.ob else sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        tri = mask64(lambda q, k: k > q)
        keys = [f"{mask64(lambda q, k, v=v: k >= v):016X}" for v in range(8)]
        return _ATTN.render(p=p, w=Attr(self.workspace()), R=self.R, QG=self.qg, HD=HD, ND=ND, SLOT=SLOT,
                            tri=f"{tri:016X}", keys=keys, ablate=self.ablate, TWO=self.two, OB=self.ob,
                            CAUSAL=self.causal, PP=self.pp, PACK=self.pack)

    @property
    def n_seg(self) -> int:
        """bf16 of the SEG argument (packed): per pass two slots (an object per stream)."""
        return self.T // (32 * self.qg) * SLOT

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        seg = [ArgSpec("in", (self.n_seg,), bfloat16)] if self.pack else []
        return [ArgSpec("in", (COLS * T // 8 * 2 * self.qc * SLOT // 2,), bfloat16),
                ArgSpec("in", (self.kvh * self.head_bytes // 2,), bfloat16),
                ArgSpec("out", (panel_elems(T * self.H * HD),), bfloat16), *seg]

    def seg(self, lo: np.ndarray, hi: np.ndarray, p0: int = 0) -> np.ndarray:
        """The SEG argument's bytes (uint8) of a packed chunk at p0: per token (T,) its segment's first and last key
        (chunk positions; a padding token's: itself) -> per pass [its rows' lo | their hi] int16 in the first stream's
        object (two streams: lo in the first's, hi in the second's), at the start of the part its first shim fills; the
        first's value 64 the history's groups (p0 / 32)."""
        assert self.pack and lo.shape == hi.shape == (self.T,) and (lo <= np.arange(self.T)).all()
        assert p0 % 32 == 0 and p0 // 32 < 1 << 15 and (self.causal or not p0)
        pq = 32 * self.qg
        assert 2 * pq <= 64 or (self.two and pq <= 64), "value 64 free"
        out = np.zeros((self.T // pq, 2, SLOT // 2), np.int16)
        out[:, 0, :pq] = lo.reshape(-1, pq)
        if self.two:
            out[:, 1, :pq] = hi.reshape(-1, pq)
        else:
            out[:, 0, pq:2 * pq] = hi.reshape(-1, pq)
        out[:, 0, 64] = p0 // 32
        return out.view(np.uint8).reshape(-1)

    def params(self, p0: int, n: int = 0) -> dict[str, int]:
        """(n: the valid tokens of a non-causal chunk, the prompt's)"""
        assert p0 % 32 == 0 and not (self.pack and p0 and not self.causal), "a non-causal packed chunk at position 0"
        if not self.causal:
            assert p0 == 0 and 0 < n <= self.T, (p0, n)
            return {QA_N: -(-n // 32)} | ({} if self.pack else {QA_NK: n - 32 * (-(-n // 32) - 1)})
        return {QA_N: p0 // 32 + 1, f"{QA_OFF}{self.T}": GROUP // 2 * (p0 // 32 + 1)}

    def get_mlir_artifact(self) -> Module:
        if self.two:
            return self._two_artifact()
        p, T, qc, H, qg, R = self.pfx, self.T, self.qc, self.H, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        kv_ty = np.ndarray[(SLOT // 2,), bf]  # a pair of K or V records, or a slot
        o_ty = np.ndarray[(A_SET,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * A_SET,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [kv_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        k_seg = Kernel(f"{p}_seg", "kernel.o", [kv_ty, ws_ty, np.int32]) if self.pack else None
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(f"{QA_OFF}{T}", np.int32)
        causal, pack = self.causal, self.pack
        p_nk = None if causal or pack else ScratchpadParameter(QA_NK, np.int32)
        n_pass, nph, n_kvh = T // (32 * qg), self.nph, self.kvh
        assert nph >= 2, "a KV head's stream from two shims of its columns"
        n_q = nph * 4 * qg * qc  # (a pass's slots of a KV head's cores, one object each)

        # the KV head's stream from the shims of columns NPH h and NPH h + 1 (d-blocks 16 i .. of every piece from part
        # i: a K record, a V pair, a slot), joined by its memtile: one panel stream in the kernels' loops
        kv = [ObjectFifo(kv_ty, name=f"pa_kv{h}", depth=2) for h in range(n_kvh)]
        kv3 = [kv[h].prod().join([0, SLOT // 4], tile=Tile(nph * h, 1), depths=[2, 2],
                                 obj_types=[np.ndarray[(SLOT // 4,), bf]] * 2, names=[f"pa_ks{h}_{i}" for i in range(2)])
               for h in range(n_kvh)]
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([A_SET * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(kv_h, o_h, init, qsel, qk, soft, pv, fin, ws, pw, qb, n_p, row, me, *seg):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q):
                        a = kv_h.acquire(1)
                        qsel(a, qb, i, me)
                        kv_h.release(1)
                    if seg:  # (packed: the pass's SEG object)
                        a = kv_h.acquire(1)
                        seg[0](a, ws, row)
                        kv_h.release(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills (non-causal: the
                    # prompt's groups; row: the prompt's keys in the last; packed: the core's row)
                    n = n_p.read() + b * qg + (qg - 1) if causal else n_p.read()
                    last = row if causal or seg else row.read()
                    for g in range32(n):
                        for k in range(2):
                            kr = kv_h.acquire(1)
                            qk(qb, kr, ws, k)
                            kv_h.release(1)
                        soft(ws, pw, n - 1 - g, last, g)
                        for half in range(2):
                            v = kv_h.acquire(1)
                            pv(v, ws, half)
                            kv_h.release(1)
                    for i in range_(n_q):  # (the gates into the Q slots' buffer)
                        a = kv_h.acquire(1)
                        qsel(a, qb, i, me)
                        kv_h.release(1)
                    for h in range(R):
                        o = o_h.acquire(1)
                        fin(ws, qb, h, o)
                        o_h.release(1)

        workers = [Worker(core, [kv[c // nph].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk, k_soft, k_pv, k_fin,
                                 Buffer(ws_ty, name=f"pa_ws{c}_{r}"), Buffer(pw_ty, name=f"pa_pw{c}_{r}"),
                                 Buffer(q_ty, name=f"pa_qb{c}_{r}"), p_n, r if causal or pack else p_nk,
                                 (c % nph * 4 * qg + qg * r) * qc, *([k_seg] if pack else [])],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        n_qs, n_kv, n_a = (arg_elems(self, i) for i in range(3))
        hb, sl, blk = self.head_bytes // 2, SLOT // 2, 4 * qg * qc * SLOT  # (blk: a column's slots in a pass)

        n_seg = self.n_seg

        def seq(QS, KV, A, *rest):
            *seg, kvh, oh = rest  # (packed: SEG)
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, lock stall)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in window_counter_writes():
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):
                # (a joined object: the 4 rows' panel i; QG > 1 (one head): token groups i, QG + i, .. of the pass)
                # (passes in runs of at most MAX_BD_ITER: a BD repeats at most that many times)
                g_st = A_SET if qg == 1 else H * A_SET
                for b0 in range(0, n_pass, MAX_BD_ITER):
                    oh[c].drain(A, TensorAccessPattern((n_a,), c * qc * A_SET + b0 * 4 * qg * H * A_SET,
                                                       [min(MAX_BD_ITER, n_pass - b0), R, 4, A_SET],
                                                       [4 * qg * H * A_SET, g_st, qg * H * A_SET, 1]), group=tg,
                                wait=True)
            inflight = []  # (a pass: one task of up to four BDs per part, ``PASSES_IN_FLIGHT`` passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1 if causal else 0  # (the chunk's groups after its first through the pass's)
                for (h, i), f in zip(((h, i) for h in range(n_kvh) for i in range(2)), kvh, strict=True):
                    # (part i: d-blocks 16 i .. of every piece (a cache object's two pairs of records); the pass's slots of the head's
                    # columns [column NPH][token group 4 QG][head] one after the other in QS, a token group [Q |
                    # sigmoid(gate)][head])
                    q0, k0 = (b * COLS + nph * h) * blk + i * SLOT // 4, h * hb + i * SLOT // 4
                    q_dims, q_str = [1, nph * 4 * qg, qc, SLOT // 4], [0, qc * SLOT, sl, 1]
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0, q_dims, q_str))]
                    if seg:  # (its half of the pass's SEG object)
                        parts.append(ChainPart(seg[0], TensorAccessPattern((n_seg,), b * SLOT + i * SLOT // 4,
                                                                           [1, 1, 2, SLOT // 8], [0, 0, SLOT // 8, 1])))
                    parts += [ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, AT.OBJS, 2, SLOT // 4],
                                                                [0, OBJ_ELEMS, SLOT // 2, 1]), length_parameter=p_n)]
                    if more:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, AT.OBJS * more, 2, SLOT // 4],
                                                                       [0, OBJ_ELEMS, SLOT // 2, 1]),
                                               offset_parameter=p_off))
                    parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + qc * sl, q_dims, q_str)))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) >= PASSES_IN_FLIGHT:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           *([np.ndarray[(n_seg,), bf]] if pack else []),
                           [kv3[h][i].prod(tile=Tile(nph * h + i, 0)) for h in range(n_kvh) for i in range(2)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def _two_artifact(self) -> Module:
        """Two row panels per core (``two_streams``: QG token groups of one q head, or a token group of QC = 2
        heads): stream s of KV head h carries d-blocks 16 s .. 16 s + 15 of every cache object (a K or V object's two
        pairs of records) from two shims (columns NPH h + 2 s and the next:
        d-blocks 8 i .. of each from part i) through its memtile (column NPH h + 2 s: the pieces interleaved per
        d-block, ``part_walk``), broadcast to the head's 4 NPH cores, which take the two on their two input channels:
        with one stream the shim channel and each core's input bound the history's re-streams (the 1B at 16K: 24.5
        GB/s for its 2 KV heads), with one shim channel per stream ~40 GB/s for the 4.
        Per pass each stream carries first the halves of the pass's Q slots of the head's cores ([column NPH][token
        group 4 QG][head QC], from the pass-major QS), two slots an object, then the history and the chunk's groups
        before the pass's, then the halves of their sigmoid(gate) slots: each core copies its own pair (``qsel``)."""
        p, T, H, qc, qg, R = self.pfx, self.T, self.H, self.qc, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        h_ty = np.ndarray[(SLOT // 2,), bf]  # (half a K or V object, the halves of two slots)
        o_ty = np.ndarray[(A_SET,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * A_SET,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [h_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, h_ty, ws_ty])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [h_ty, ws_ty])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        k_seg = Kernel(f"{p}_seg", "kernel.o", [h_ty, h_ty, ws_ty, np.int32]) if self.pack else None
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(f"{QA_OFF}{T}", np.int32)
        causal, pack = self.causal, self.pack
        p_nk = None if causal or pack else ScratchpadParameter(QA_NK, np.int32)
        n_pass, nph, n_kvh = T // (32 * qg), self.nph, self.kvh
        n_q = nph * 4 * qg * qc  # (a pass's slots of a KV head's cores)

        streams = {(h, s): nph * h + 2 * s for h in range(n_kvh) for s in range(2)}
        # (each stream from the shims of columns NPH h + 2 s and the next, d-blocks 8 i .. of every piece from part i,
        # joined by its memtile: one shim channel each paced the history's re-streams)
        kv = {hs: ObjectFifo(h_ty, name=f"pa_k{'ab'[hs[1]]}{hs[0]}", depth=2, to_stream=part_walk()) for hs in streams}
        kv3 = {hs: kv[hs].prod().join([0, SLOT // 4], tile=Tile(c, 1), depths=[2, 2],
                                      obj_types=[np.ndarray[(SLOT // 4,), bf]] * 2,
                                      names=[f"pa_ks{hs[0]}_{hs[1]}_{i}" for i in range(2)])
               for hs, c in streams.items()}
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([A_SET * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, b_h, o_h, init, qsel, qk, soft, pv, fin, ws, pw, qb, n_p, row, me, *seg):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q // 2):
                        a = a_h.acquire(1)
                        b_h.acquire(1)  # (b's object follows a's: l1_layout)
                        qsel(a, qb, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    if seg:  # (packed: the pass's SEG objects)
                        a = a_h.acquire(1)
                        bb = b_h.acquire(1)
                        seg[0](a, bb, ws, row)
                        a_h.release(1)
                        b_h.release(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills (non-causal: the
                    # prompt's groups; row: the prompt's keys in the last; packed: the core's row)
                    n = n_p.read() + b * qg + (qg - 1) if causal else n_p.read()
                    last = row if causal or seg else row.read()
                    for g in range32(n):
                        ka = a_h.acquire(1)
                        b_h.acquire(1)
                        qk(qb, ka, ws)
                        a_h.release(1)
                        b_h.release(1)
                        soft(ws, pw, n - 1 - g, last, g)
                        va = a_h.acquire(1)
                        b_h.acquire(1)
                        pv(va, ws)
                        a_h.release(1)
                        b_h.release(1)
                    for i in range_(n_q // 2):  # (the gates into the Q slots' buffer)
                        a = a_h.acquire(1)
                        b_h.acquire(1)
                        qsel(a, qb, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    for h in range(R):
                        o = o_h.acquire(1)
                        fin(ws, qb, h, o)
                        o_h.release(1)

        workers = []
        for c in range(COLS):
            h = c // nph
            for r in range(4):
                workers.append(Worker(core, [kv[h, 0].cons(), kv[h, 1].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk,
                                             k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"pa_ws{c}_{r}"),
                                             Buffer(pw_ty, name=f"pa_pw{c}_{r}"),
                                             Buffer(q_ty, name=f"pa_qb{c}_{r}"), p_n, r if causal or pack else p_nk,
                                             (c % nph) * 4 + r, *([k_seg] if pack else [])],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = (arg_elems(self, i) for i in range(3))
        hb, sl = self.head_bytes // 2, SLOT // 2
        blk = 4 * qg * qc * SLOT  # (a column's slots in a pass: per token group [its heads' Q | their sigmoid(gate)])
        n_seg = self.n_seg

        def seq(QS, KV, A, *rest):
            *seg, kvh, oh = rest  # (packed: SEG)
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, lock stall)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in window_counter_writes():
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):
                g_st = A_SET if qg == 1 else H * A_SET  # (the row panels: heads, or token groups)
                for b0 in range(0, n_pass, MAX_BD_ITER):  # (runs of at most MAX_BD_ITER passes)
                    oh[c].drain(A, TensorAccessPattern((n_a,), c * qc * A_SET + b0 * 4 * qg * H * A_SET,
                                                       [min(MAX_BD_ITER, n_pass - b0), R, 4, A_SET],
                                                       [4 * qg * H * A_SET, g_st, qg * H * A_SET, 1]),
                                group=tg, wait=True)
            inflight = []  # (a pass: one task of up to four BDs per stream, ``PASSES_IN_FLIGHT`` passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1 if causal else 0  # (the chunk's groups after its first through the pass's)
                for ((h, s), i), f in zip(((hs, i) for hs in streams for i in range(2)), kvh, strict=True):
                    # (part i: d-blocks 16 s + 8 i .. of every piece (a cache object's two); QS pass-major: the pass's
                    # slots of the head's columns [column NPH][token group 4 QG] one after the other, a token group's [Q
                    # | sigmoid(gate)] of its QC heads)
                    q0 = (b * COLS + nph * h) * blk + (2 * s + i) * SLOT // 8  # (its quarter of the slots)
                    k0, run = h * hb + s * SLOT // 4 + i * SLOT // 8, SLOT // 8
                    q_dims, q_str = [1, nph * 4 * qg, qc, SLOT // 8], [0, qc * SLOT, SLOT // 2, 1]
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0, q_dims, q_str))]
                    if seg:  # (its part of the pass's SEG object of the stream)
                        parts.append(ChainPart(seg[0], TensorAccessPattern((n_seg,), b * SLOT + s * SLOT // 2
                                                                           + i * SLOT // 4, [1, 1, 2, SLOT // 8],
                                                                           [0, 0, SLOT // 8, 1])))
                    parts += [ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, AT.OBJS, 2, run],
                                                                [0, OBJ_ELEMS, SLOT // 2, 1]), length_parameter=p_n)]
                    if more:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, AT.OBJS * more, 2, run],
                                                                       [0, OBJ_ELEMS, SLOT // 2, 1]),
                                               offset_parameter=p_off))
                    parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + qc * sl, q_dims, q_str)))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) >= PASSES_IN_FLIGHT:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           *([np.ndarray[(n_seg,), bf]] if pack else []),
                           [kv3[hs][i].prod(tile=Tile(c + i, 0)) for hs, c in streams.items() for i in range(2)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (aiecc puts the stack at 0): 0 the K / V records, an output buffer and the
        softmax's scratch, 1 the Q panels held through a pass, the other output buffer (and with o's second copy the
        scores, m, l), 2 and 3 o (its two copies, else the workspace): QK streams Q (1) against K (0), PV V (0) from one
        o copy into the other (two streams: their objects in regions 0 and 1). The 9B: all 64 KB, the 27B 61. The
        pinned buffers' bounds and overlaps are checked where they are pinned (``pin_l1_buffers``)."""
        kv, o, q, w = SLOT, ND * PANEL_BYTES, self.R * SLOT, self.workspace()
        pw = self.STACK + 2 * kv + o
        if self.two:  # (the two streams' objects i in region i, b's after a's: the kernels read both from a's pointer; the
            # Q slots, then the gates, and the scores in region 2, o in 3)
            return {"pa_ka": [self.STACK, REGION], "pa_kb": [self.STACK + kv, REGION + kv],
                    "pa_or": [self.STACK + 2 * kv, REGION + 2 * kv], "pa_qb": [2 * REGION], "pa_pw": [pw],
                    "pa_ws": [3 * REGION - w["o"]]}  # (o of the workspace at the start of region 3)
        ws = (2 * REGION - w["o"] if self.ob else 2 * REGION)
        return {"pa_kv": [self.STACK, self.STACK + kv], "pa_or": [self.STACK + 2 * kv, REGION + q], "pa_qb": [REGION],
                "pa_pw": [pw], "pa_ws": [ws]}
