"""The attention of the Gemma prefill: a T-token chunk at position p0 (a multiple of 32) of one layer, from the qkv
GEMMs' output G (``prefill.gemm``: per column c [q of head c (hd) | the column's quarter of [K_h; V_h], h = c / 4,
own-cache layers]) to the O GEMM's A panels, with the K / V records in the decode's cache layout (``decode.kv``:
per KV head groups of 3 objects of 9216 B, [K obj | K obj | V obj]; a sliding-window cache after its guard,
``overlay.GUARD``), so the decode continues from them. Head dims: 256 (sliding-window layers: every pair (i, i + 128)
rotated, theta 10K, a window of 512 positions) and 512 (full layers: pairs (i, i + 256) of i < 64, theta 1M);
q / k RMS-normed per head with the product of their constant weights (and log2 e) as one score constant (the decode's
``attn.score_constant``), v RMS-normed without a weight; scores unscaled.

``GemmaPrepOperator``: per column c core (c, 2) turns head c's q rows into the attention's operands, per 8 tokens:
q / rms(q), rotated (fp32), times the score constant, as bfp16 panels of 8 queries x 8 dims, per d-block hi then lo
(QS: per token group and column; head_dim 512: pass-major, [pass][KV head][token group][column], and per half of the
d-blocks hi panels then lo, which the attention's memtile interleaves). Cores (4 h, 3) and (4 h + 2, 3) write KV head
h's K (normed, rotated, hi + lo panel records) and V (bf16(v / rms(v)), per d-block a panel of blocks over the 8 positions) into the cache: the chunk's
groups from p0 on (scratchpad parameters ``params``). cos / sin of the positions from a table of every position
(``cs_table``). The score constant comes as the first object of the cos / sin stream (``prm_object``).

``GemmaAttnOperator``: per column c (query head c) 4 cores, passes of 32 queries; core r takes queries 8 r .. of the
pass and streams the KV head's cache (broadcast to the 16 cores of the head's 4 columns; head_dim 512: two streams, one
per input channel of each core, each with half of every cache object: ``GemmaAttnOperator._full_artifact``): per K
record the scores on the MMUL (Q hi + lo against K hi + lo: three products), per group the softmax with a running maximum
(qwen35's ``attn_prefill``: raised only by 2^16; masked scores far below a floor, so a row without keys in a group
adds nothing), P = 2^(s - m) as bfp16 hi + lo panels (single ones cost 2x the cache error by layer 23), o += P V; at the pass's end o / l as the O GEMM's A panels.
Sliding-window passes read 17 groups of 32 positions (the window's 16 before the pass and the pass's own: one BD
iterated per pass); groups before position 0 (the guard) are skipped, the window's first and the pass's own masked
per record. Full passes read every group from position 0 (the history's length a parameter, as qwen35's).
"""

import math
from dataclasses import dataclass

import numpy as np
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

from artlite.gemm.gemm_q import ChainPart, chained_fill
from artlite.model_config import GemmaConfig
from artlite.operators import templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.design import COLS, range32, stream_walk
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.l1_layout import pin_l1_buffers
from artlite.operators.util import Attr, regions
from artlite.prefill.gemm import Layout

from . import attn as AT

OBJ = 9216  # a cache object
QH = AT.QH  # query heads per KV head
WINDOW_GROUPS = 17  # groups a sliding-window pass streams (the window's 16 before it, its own)
# scratchpad parameters
QA_CS = {256: "g4_cs_s", 512: "g4_cs_f"}  # bf16 offset of the chunk's rows in the cos / sin table
QA_KV = {256: "g4_kv_s", 512: "g4_kv_f"}  # bf16 offset of the chunk's first cache group (a sliding-window cache: after
# its guard)
QA_SW = "g4_sw"  # sliding-window attention: bf16 offset of the window of the chunk's first pass (p0 / 32 groups)
QA_FN = "g4_fn"  # full attention: the history's groups every pass streams first (p0 / 16 + 2)
QA_FO = "g4_fo"  # full attention: bf16 offset of the group after them
QA_P8 = "g4_p8"  # p0 / 8 (the full attention cores' masks)
QA_P32 = "g4_p32"  # p0 / 32 (the sliding-window attention cores' masks)


@dataclass(frozen=True)
class Kind:
    """A layer kind's attention shapes."""

    hd: int
    rot: int  # rotated pairs (i, i + hd / 2), i < rot
    kp: int = 2  # K panels per record (2: hi + lo, 1: single)

    @property
    def nd(self) -> int:
        return self.hd // 8

    @property
    def swa(self) -> bool:
        return self.hd == 256

    @property
    def pg(self) -> int:  # positions per cache group
        return 32 if self.swa else 16

    @property
    def rec_k(self) -> int:  # bytes of a K record (8 positions: hi | lo panels, or single ones)
        return 72 * self.kp * self.nd

    @property
    def group(self) -> int:  # bytes of a cache group (kp K objects, the V object)
        return (self.kp + 1) * OBJ

    @property
    def unit(self) -> int:  # bytes of the stream's unit: a hi + lo K record, two single ones, a V object's two records
        return 144 * self.nd

    @property
    def qp(self) -> int:  # bytes of a token group's Q panels
        return 144 * self.nd

    @property
    def recs(self) -> int:  # records per group
        return self.pg // 8

    @property
    def k_units(self) -> int:  # K units per group
        return self.pg // 8 if self.kp == 2 else self.pg // 16

    @property
    def v_units(self) -> int:  # V units (two records) per group
        return self.pg // 16


def kind_of(cfg: GemmaConfig, hd: int, kp: int = 2) -> Kind:
    return Kind(hd, AT.GemmaAttn(cfg).rot[0 if hd == 256 else 1], kp)


def cs_table(cfg: GemmaConfig, kind: Kind, n: int) -> np.ndarray:
    """(n, 2 rot) fp32: per position [cos rot | sin rot] of the kind's rotated pairs."""
    inv = cfg.rope_inv_freq(swa=kind.swa)[:kind.rot]
    th = np.outer(np.arange(n, dtype=np.float64), inv)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


def prm_object(kind: Kind, c: float) -> np.ndarray:
    """The prep's parameter object (one cos / sin object of 8 positions, fp32): the score constant first."""
    out = np.zeros(8 * 2 * kind.rot, np.float32)
    out[0] = c
    return out


_PREP = templating.load("models/gemma4/attn_prefill.prep.cc.jinja")

_NR0 = templating.load("models/gemma4/attn_prefill.nr0.cc.jinja")


class GemmaPrepOperator(ArtcOperator):
    """The attention prep of T tokens of one layer kind (module docstring). Runtime sequence: G (the qkv GEMM's
    output, ``layout``), CS (the kind's ``cs_table`` of every position, as bf16 pairs), PRM (``prm_object``), QS (per
    token group and column the Q panels, head_dim 512 pass-major: ``n_qs``; a sub-chunk's part of a chunk's), KV (the
    layer's cache: 2 KV heads of ``kv_bytes`` each, from the first head's guard; a sub-chunk's from its first group:
    the cache's groups it writes lie inside, the bytes past the caches are never touched). Parameters ``params``."""

    STACK = 2048
    pfx = kernel_prefix("g4ap")

    def __init__(self, *, T: int, cfg: GemmaConfig, hd: int, layout: Layout, kv_bytes: int, cs_positions: int,
                 kp: int = 2):
        """``kp``: the cache's K panels per record (2: hi + lo, 1: single)."""
        assert cfg.num_attention_heads == COLS and all(k == COLS // QH for k in cfg.kv_heads)
        self.kind = kind_of(cfg, hd, kp)
        assert T % 32 == 0 and T // 8 <= 64 and {"q", "kv"} <= set(layout.off)
        self.T, self.cfg, self.layout, self.kv_bytes, self.cs_positions = T, cfg, layout, kv_bytes, cs_positions
        super().__init__()

    def workspace(self) -> dict[str, int]:
        """Byte offsets: c (the score constant, 64 fp32), row (hd fp32), stage (Q / K cores: [nd][8][8] fp32) or vn (V
        cores: [8][hd] bf16)."""
        hd = self.kind.hd
        return {"c": 0, "row": 256, "stage": 256 + 4 * hd, "vn": 256 + 4 * hd, "size": 256 + 36 * hd}

    def get_kernel_src(self) -> str:
        p = self.pfx
        k = self.kind
        args = {"p": p, "HD": k.hd, "ND": k.nd, "ROT": k.rot, "eps": repr(self.cfg.rms_norm_eps), "KP": k.kp}
        return _NR0.render(**args) + _PREP.render(w=Attr(self.workspace()), HALVES=not k.swa,
                                                                         **args)

    @property
    def n_cs(self) -> int:
        return self.cs_positions * 4 * self.kind.rot

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument: [token group][column][Q panels] (head_dim 512: [pass][KV head][token group]
        [column])."""
        return self.T // 8 * COLS * self.kind.qp // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        k = self.kind
        return [ArgSpec("in", (self.T * self.layout.N,), bfloat16),
                ArgSpec("in", (self.n_cs,), bfloat16),
                ArgSpec("in", (8 * 4 * k.rot,), bfloat16),
                ArgSpec("out", (self.n_qs,), bfloat16),
                ArgSpec("out", (COLS // QH * self.kv_bytes // 2,), bfloat16)]

    def params(self, p0: int, guard: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (``guard``: the cache's groups before group 0)."""
        k = self.kind
        assert p0 % 32 == 0 and p0 + self.T <= self.cs_positions
        return {QA_CS[k.hd]: 4 * k.rot * p0, QA_KV[k.hd]: k.group // 2 * (guard + p0 // k.pg)}

    def get_mlir_artifact(self) -> Module:
        p, T, lay, k = self.pfx, self.T, self.layout, self.kind
        hd, N, pitch = k.hd, lay.N, lay.pitch
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of a head
        cs_ty = np.ndarray[(8 * 4 * k.rot,), bf]  # 8 positions x [cos | sin] fp32 (the parameters: the score constant)
        qp_ty = np.ndarray[(k.qp // 2,), bf]
        kr_ty = np.ndarray[(k.rec_k // 2,), bf]
        vr_ty = np.ndarray[(36 * k.nd,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_q = Kernel(f"{p}_q", "kernel.o", [row_ty, cs_ty, ws_ty, qp_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kr_ty])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, cs_ty, ws_ty, vr_ty])
        p_cs, p_kv = ScratchpadParameter(QA_CS[hd], np.int32), ScratchpadParameter(QA_KV[hd], np.int32)
        n_tg = T // 8
        roles = {QH * h: "k" for h in range(COLS // QH)} | {QH * h + 2: "v" for h in range(COLS // QH)}

        # per column [q rows | the role's K or V rows] per token group, split in the memtile
        a3, a1 = [], []
        for c in range(COLS):
            if c in roles:
                f = ObjectFifo(np.ndarray[(2 * 8 * hd,), bf], name=f"gp_a{c}", depth=2)
                a1.append(f.cons().split([0, 8 * hd], tile=Tile(c, 1), depths=[2, 2], obj_types=[row_ty, row_ty],
                                         names=[f"gp_aq{c}", f"gp_ak{c}"]))
            else:
                f = ObjectFifo(row_ty, name=f"gp_a{c}", depth=2)
                a1.append([f])
            a3.append(f)
        b = [ObjectFifo(cs_ty, name=f"gp_b{c}", depth=2) for c in range(COLS)]
        od = 2 if k.swa else 1  # (output objects: head_dim 512's would not fit L1 twice)
        o = [ObjectFifo(qp_ty, name=f"gp_o{c}", depth=od) for c in range(COLS)]
        kv = {c: ObjectFifo(kr_ty if r == "k" else vr_ty, name=f"gp_k{c}", depth=od) for c, r in roles.items()}

        def core(a_h, b_h, o_h, prm, fn, ws):
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, cs, ws, out)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        workers = []
        for c in range(COLS):
            workers.append(Worker(core, [a1[c][0].cons(), b[c].cons(), o[c].prod(), k_prm, k_q,
                                         Buffer(ws_ty, name=f"gp_ws{c}")],
                                  tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
            if c in roles:
                workers.append(Worker(core, [a1[c][1].cons(), b[c].cons(), kv[c].prod(), k_prm,
                                             k_k if roles[c] == "k" else k_v, Buffer(ws_ty, name=f"gp_wk{c}")],
                                      tile=Tile(c, 3), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_cs, n_kv, n_qs = self.n_cs, COLS // QH * self.kv_bytes // 2, self.n_qs
        qh = k.qp // 2  # (a token group's Q panels of a column, bf16: rows of 1152 for a shim BD)
        hb, g2 = self.kv_bytes // 2, k.group // 2
        n_grp = T // k.pg

        def seq(G, CS, PRM, QS, KV, ah, bh, oh, kh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                taps = [TensorAccessPattern((T * N,), c * pitch + lay.off["q"], [n_tg, 1, 8, hd], [8 * N, 0, N, 1])]
                if c in roles:  # (the KV head's halves: this column's and the next one's kv blocks)
                    taps.append(TensorAccessPattern((T * N,), c * pitch + lay.off["kv"], [n_tg, 8, 2, hd // 2],
                                                    [8 * N, N, pitch, 1]))
                chained_fill(ah[c], G, taps, tg)
                bh[c].fill(PRM, TensorAccessPattern((8 * 4 * k.rot,), 0, [1, 1, 1, 8 * 4 * k.rot], [0, 0, 0, 1]),
                           group=tg)
                bh[c].fill(CS, TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 4 * k.rot], [0, 0, 0, 1]),
                           offset_parameter=p_cs, group=tg)
                if k.swa:
                    q_tap = TensorAccessPattern((n_qs,), c * qh, [1, n_tg, qh // 1152, 1152], [0, COLS * qh, 1152, 1])
                else:  # (pass-major: [pass][KV head][token group of the pass][column of the head])
                    q_tap = TensorAccessPattern((n_qs,), (4 * QH * (c // QH) + c % QH) * qh,
                                                [T // 32, 4, qh // 1152, 1152], [4 * COLS * qh, QH * qh, 1152, 1])
                oh[c].drain(QS, q_tap, group=tg, wait=True)
                if c in roles:
                    h = c // QH
                    if roles[c] == "k":  # a group's K records: its first kp objects
                        tap = TensorAccessPattern((n_kv,), h * hb, [1, n_grp, 8 * k.kp, OBJ // 16],
                                                  [0, g2, OBJ // 16, 1])
                    else:  # its V records: the last object
                        tap = TensorAccessPattern((n_kv,), h * hb + k.kp * OBJ // 2, [1, n_grp, 8, OBJ // 16],
                                                  [0, g2, OBJ // 16, 1])
                    kh[list(kv).index(c)].drain(KV, tap, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(8 * 4 * k.rot,), bf],
                           np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o)],
                           [f.cons(tile=Tile(c, 0)) for c, f in kv.items()]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = templating.load("models/gemma4/attn_prefill.cc.jinja")


class GemmaAttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 of one layer kind (module docstring) on the cache that already holds
    the chunk's K and V (``GemmaPrepOperator``). Column c (query head c, KV head c / 4), core row r; passes of 32
    queries: in pass b the core takes the queries 32 b + 8 r .., streams the KV head's cache (broadcast to the 16 cores
    of its columns; head_dim 512 on both input channels) and writes their outputs as the O GEMM's A panels. Runtime
    sequence: QS (the prep's), KV (the
    layer's cache, from the first head's guard), A (T x 8 hd as bfp16 panels [T / 8][hd]). Parameters ``params``."""

    STACK = 3072
    pfx = kernel_prefix("g4aa")

    def __init__(self, *, T: int, cfg: GemmaConfig, hd: int, kv_bytes: int, kv_pad: int = 0, kp: int = 2,
                 ablate: tuple[str, ...] = ()):
        """``kv_pad``: bytes of the KV argument past the last head's cache (the walks of a chunk near the cache's end
        may reach past its tail: their parameters' range); ``kp``: the cache's K panels per record (2: hi + lo, 1:
        single); ``ablate`` (timing experiments, wrong results): kernel bodies compiled out ("qk", "soft", "pv")."""
        assert cfg.num_attention_heads == COLS and all(k == COLS // QH for k in cfg.kv_heads)
        self.kind = kind_of(cfg, hd, kp)
        assert not self.kind.swa or cfg.sliding_window == (WINDOW_GROUPS - 1) * 32
        # query blocks of 8 per core and pass (head_dim 256: two, 64-query passes; 512: one, L1)
        self.qb = 2 if self.kind.swa else 1
        self.pq = 32 * self.qb  # queries per pass
        self.n_pass = -(-T // self.pq)  # (a last pass past T: its queries' panels and outputs in the buffers' padding)
        assert T % 32 == 0 and self.n_pass <= 64
        self.wg = WINDOW_GROUPS + self.qb - 1  # (the groups a sliding-window pass streams)
        self.T, self.cfg, self.kv_bytes, self.kv_pad, self.ablate = T, cfg, kv_bytes, kv_pad, tuple(ablate)
        super().__init__()

    def workspace(self) -> dict[str, int]:
        rg, qb = self.kind.recs, self.qb
        sizes = [("sc", 256 * rg * qb), ("m", 256 * qb), ("l", 256 * qb), ("pp", 144 * rg * qb), ("rm", 256), ("st", 64)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        k = self.kind
        tri = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j > q)
        return _ATTN.render(p=p, w=Attr(self.workspace()), ND=k.nd, RG=k.recs, KP=k.kp, SWA=k.swa, QB=self.qb,
                            tri=f"{tri:016X}", ablate=self.ablate)

    @property
    def n_qs(self) -> int:
        return COLS * self.n_pass * self.pq // 8 * self.kind.qp // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        hd = self.kind.hd
        return [ArgSpec("in", (self.n_qs,), bfloat16),
                ArgSpec("in", ((COLS // QH * self.kv_bytes + self.kv_pad) // 2,), bfloat16),
                ArgSpec("out", (self.n_pass * self.pq * COLS * hd * 9 // 16,), bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        assert p0 % 32 == 0
        if self.kind.swa:
            return {QA_P32: p0 // 32, QA_SW: self.kind.group // 2 * (p0 // 32)}
        return {QA_P8: p0 // 8, QA_FN: p0 // 16 + 2, QA_FO: self.kind.group // 2 * (p0 // 16 + 2)}

    def get_mlir_artifact(self) -> Module:
        if not self.kind.swa:
            return self._full_artifact()
        p, k, qb = self.pfx, self.kind, self.qb
        nd, hd = k.nd, k.hd
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        q_ty = np.ndarray[(qb * k.qp // 2,), bf]  # the core's query blocks' Q panels
        qm_ty = np.ndarray[(4 * qb * k.qp // 2,), bf]
        kv_ty = np.ndarray[(k.unit // 2,), bf]  # a K record, or two V records
        o_ty = np.ndarray[(qb * nd * 36,), bf]  # ND A panels per query block
        om_ty = np.ndarray[(4 * qb * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[(64 * nd * qb,), f32]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty])
        k_grp = Kernel(f"{p}_grp", "kernel.o", [ws_ty, np.int32, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, wo_ty])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, wo_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty])
        p_p8 = ScratchpadParameter(QA_P32, np.int32)
        p_sw = ScratchpadParameter(QA_SW, np.int32)
        n_pass, wg = self.n_pass, self.wg
        n_kv_heads = COLS // QH

        q3 = [ObjectFifo(qm_ty, name=f"ga_q{c}", depth=2) for c in range(COLS)]
        # (two query blocks: their Q panels interleaved per d-block, [db][block][hi | lo], one stream in the kernel)
        qw = k.qp // 2
        ilq = stream_walk(qb * qw, [(nd, 72), (qb, qw), (72, 1)])
        q1 = [q3[c].cons().split([qb * qw * r for r in range(4)], tile=Tile(c, 1), depths=[1] * 4,
                                 obj_types=[q_ty] * 4, names=[f"ga_qr{c}_{r}" for r in range(4)], to_stream=[ilq] * 4)
              for c in range(COLS)]
        # the KV head's units through its memtile, each re-laid per d-block ([hi | lo] of a K record, the two records
        # of a V unit): one panel stream in the kernels' loops (the stream of a head reads ~6 GB/s through one shim
        # channel: the bound of a sliding-window pass; a unit's halves through two channels joined in the memtile would
        # need 7 of its 6 input channels, with the column's Q split and output join)
        kv3 = [ObjectFifo(kv_ty, name=f"ga_ks{h}", depth=2) for h in range(n_kv_heads)]
        ilv = stream_walk(k.unit // 2, [(nd, 36), (2, nd * 36), (36, 1)])
        kv = [kv3[h].cons().forward(tile=Tile(QH * h, 1), depth=2, name=f"ga_kv{h}", to_stream=ilv)
              for h in range(n_kv_heads)]
        o3 = [ObjectFifo(om_ty, name=f"ga_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([qb * nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"ga_or{c}_{r}" for r in range(4)]) for c in range(COLS)]
        ku, vu = k.k_units, k.v_units

        def core(q_h, kv_h, o_h, init, grp, qk, soft, pv, fin, ws, wo, p8, row):
            for _ in forever_():
                for b in range_(n_pass):
                    q = q_h.acquire(1)
                    init(ws, wo)
                    x = p8.read() + b + b  # (p0 / 32 + 2 b; read after the first object: written before the fills)
                    for g in range_(wg):
                        grp(ws, g, x, row)
                        for kk in range(ku):
                            kr = kv_h.acquire(1)
                            qk(q, kr, ws, kk)
                            kv_h.release(1)
                        soft(ws, wo)
                        for u in range(vu):
                            v = kv_h.acquire(1)
                            pv(v, ws, wo, u)
                            kv_h.release(1)
                    q_h.release(1)
                    o = o_h.acquire(1)
                    fin(ws, wo, o)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for r in range(4):
                workers.append(Worker(core, [q1[c][r].cons(), kv[c // QH].cons(), o1[c][r].prod(), k_init, k_grp, k_qk,
                                             k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"ga_ws{c}_{r}"),
                                             Buffer(wo_ty, name=f"ga_wo{c}_{r}"), p_p8, r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = self.n_qs, (n_kv_heads * self.kv_bytes + self.kv_pad) // 2, n_pass * self.pq * COLS * hd * 9 // 16
        hb, g2, uw = self.kv_bytes // 2, k.group // 2, k.unit // 2  # (bf16: a head's cache, a group, a unit)

        def seq(QS, KV, A, qh, kvh, oh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 qb b .. of the column's head
                qh[c].fill(QS, TensorAccessPattern((n_qs,), c * qw, [1, n_pass * 4 * qb, qw // 1152, 1152],
                                                   [0, COLS * qw, 1152, 1]), group=tg)
                oh[c].drain(A, TensorAccessPattern((n_a,), c * nd * 36, [n_pass, 4 * qb, nd, 36],
                                                   [4 * qb * COLS * nd * 36, COLS * nd * 36, 36, 1]), group=tg, wait=True)
            # per pass the window's 16 groups before it and its own (one BD iterated per pass): their units' halves
            for h in range(n_kv_heads):
                kvh[h].fill(KV, TensorAccessPattern((n_kv,), h * hb, [n_pass, 1, wg * k.group // k.unit, uw],
                                                    [qb * g2, 0, uw, 1]), offset_parameter=p_sw, group=tg)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(q3)],
                           [f.prod(tile=Tile(QH * h, 0)) for h, f in enumerate(kv3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return self._pinned(Program(NPU2(), rt, workers=workers).resolve_program())

    def _full_artifact(self) -> Module:
        """Head_dim 512: a core's two input channels each carry half of the KV head's cache stream, as a fused attention
        whose tiles take Q + K on one inbound path and V on the other (arXiv 2609.21264), balanced: stream s of a KV
        head carries d-blocks 32 s .. 32 s + 31 of every cache object (a K record's hi and lo panels, a V object's two
        records), from its own shim through its own memtile (which interleaves the two parts per d-block),
        broadcast to the head's 16 cores; the kernels run a loop per stream object. With one stream of
        whole objects, each core's 8 B / cycle input channel bounds the layer (every position's 1728 bytes to each core
        once per 32-query pass). A pass's 16 Q objects ([token group of the pass][column of the head], the prep's
        pass-major QS, per half [hi | lo] panels) ride ahead of its cache objects, their halves on the two streams;
        each core copies its own."""
        p, k = self.pfx, self.kind
        nd, hd = k.nd, k.hd
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        half = k.unit // 4  # (bf16 of a stream object: half a K record, one V record, half a Q object)
        h_ty = np.ndarray[(half,), bf]
        q_ty = np.ndarray[(k.qp // 2,), bf]  # the core's Q panels
        o_ty = np.ndarray[(nd * 36,), bf]  # ND A panels
        om_ty = np.ndarray[(4 * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[(64 * nd,), f32]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty])
        k_grp = Kernel(f"{p}_grp", "kernel.o", [ws_ty, np.int32, np.int32, np.int32])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [h_ty, h_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, h_ty, h_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, wo_ty])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [h_ty, h_ty, ws_ty, wo_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty])
        p_p8 = ScratchpadParameter(QA_P8, np.int32)
        p_fn = ScratchpadParameter(QA_FN, np.int32)
        p_fo = ScratchpadParameter(QA_FO, np.int32)
        n_pass, n_kv_heads, ku = self.n_pass, COLS // QH, k.k_units
        assert self.qb == 1 and k.v_units == 1 and k.qp // 2 == 2 * half

        streams = {(h, s): QH * h + 2 * s for h in range(n_kv_heads) for s in range(2)}  # (its shim's and memtile's
        # column)
        kv3 = {hs: ObjectFifo(h_ty, name=f"ga_ks{hs[0]}_{hs[1]}", depth=2) for hs in streams}
        ilv = stream_walk(half, [(nd // 2, 36), (2, half // 2), (36, 1)])
        kv = {hs: kv3[hs].cons().forward(tile=Tile(c, 1), depth=2, name=f"ga_k{'ab'[hs[1]]}{hs[0]}", to_stream=ilv)
              for hs, c in streams.items()}
        o3 = [ObjectFifo(om_ty, name=f"ga_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"ga_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, b_h, o_h, init, grp, qsel, qk, soft, pv, fin, ws, wo, q, p8, fn, me, row):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(4 * QH):
                        q0 = a_h.acquire(1)
                        q1 = b_h.acquire(1)
                        qsel(q0, q1, q, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    init(ws, wo)
                    n = fn.read() + b + b  # (the history's groups, then the chunk's two per pass before this one)
                    x = p8.read() + b + b + b + b
                    for g in range32(n):
                        grp(ws, g, x, row)
                        for kk in range(ku):
                            kh = a_h.acquire(1)
                            kl = b_h.acquire(1)
                            qk(q, kh, kl, ws, kk)
                            a_h.release(1)
                            b_h.release(1)
                        soft(ws, wo)
                        v0 = a_h.acquire(1)
                        v1 = b_h.acquire(1)
                        pv(v0, v1, ws, wo, 0)
                        a_h.release(1)
                        b_h.release(1)
                    o = o_h.acquire(1)
                    fin(ws, wo, o)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for r in range(4):
                workers.append(Worker(core, [kv[c // QH, 0].cons(), kv[c // QH, 1].cons(), o1[c][r].prod(), k_init,
                                             k_grp, k_qsel, k_qk, k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"ga_ws{c}_{r}"),
                                             Buffer(wo_ty, name=f"ga_wo{c}_{r}"), Buffer(q_ty, name=f"ga_qb{c}_{r}"),
                                             p_p8, p_fn, QH * r + c % QH, r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = self.n_qs, (n_kv_heads * self.kv_bytes + self.kv_pad) // 2, n_pass * self.pq * COLS * hd * 9 // 16
        hb, ob, qw = self.kv_bytes // 2, OBJ // 2, k.qp // 2  # (bf16: a head's cache, a cache object, a Q object)

        def seq(QS, KV, A, kvh, oh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 b .. of the column's head
                oh[c].drain(A, TensorAccessPattern((n_a,), c * nd * 36, [n_pass, 4, nd, 36],
                                                   [4 * COLS * nd * 36, COLS * nd * 36, 36, 1]), group=tg, wait=True)
            inflight = []  # (a pass: one task of three BDs per stream, at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                for (h, s), f in zip(streams, kvh, strict=True):  # the pass's Q halves, the history (QA_FN groups),
                    # then the chunk's groups before the pass's: the objects' d-block halves s (of both parts)
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), (4 * COLS * b + 4 * QH * h) * qw + s * half,
                                                               [1, 4 * QH, 2, half // 2], [0, qw, half // 2, 1])),
                             ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * half // 2,
                                                               [1, k.kp + 1, 2, half // 2],
                                                               [0, ob, half, 1]), length_parameter=p_fn)]
                    if b:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * half // 2,
                                                                       [1, 2 * (k.kp + 1) * b, 2, half // 2],
                                                                       [0, ob, half, 1]),
                                               offset_parameter=p_fo))
                    chained_fill(f, KV, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [kv3[hs].prod(tile=Tile(c, 0)) for hs, c in streams.items()],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return self._pinned(Program(NPU2(), rt, workers=workers).resolve_program())

    def _pinned(self, module: Module) -> Module:
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for kk, v in lay.items() if name.startswith(kk)), None),
                       buffer=lambda name: next((v[0] for kk, v in lay.items() if name.startswith(kk)), None))
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (the stack at 0): QK streams Q against K, PV V into O. Head_dim 256 (two
        query blocks): 0 the K / V units, 1 Q and an output buffer, 2 O (16 KB), 3 the workspace and the other output
        buffer; 512: 0 and 1 stream A's and B's objects (1 also an output buffer), 2 O (16 KB), 3 Q, the workspace and
        the other output buffer."""
        k = self.kind
        u, o, q, w = k.unit, self.qb * k.nd * 72, self.qb * k.qp, -(-self.workspace()["size"] // 64) * 64
        S, R = self.STACK, 0x4000
        if k.swa:
            lay = {"ga_kv": [S, S + u], "ga_or": [R + q, 3 * R + w], "ga_qr": [R], "ga_wo": [2 * R], "ga_ws": [3 * R]}
            assert S + 2 * u <= R and R + q + o <= 2 * R and 3 * R + w + o <= 4 * R
        else:
            h = u // 2
            lay = {"ga_ka": [S, S + h], "ga_kb": [R, R + h], "ga_or": [R + 2 * h, 3 * R + q + w], "ga_wo": [2 * R],
                   "ga_qb": [3 * R], "ga_ws": [3 * R + q]}
            assert S + 2 * h <= R and R + 2 * h + o <= 2 * R and 3 * R + q + w + o <= 4 * R
        assert 64 * 4 * k.nd * self.qb <= R
        return lay


# ----------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------


def score_constant(q_norm: float, k_norm: float) -> float:
    """q_norm k_norm log2(e): the scores in log2 units (the decode: ``attn.score_constant``, as bf16 hi + lo)."""
    return q_norm * k_norm / math.log(2)
