"""``AttnOperator``: the attention of a chunk (module docstring of ``prefill.attn``), its variants as ``AttnMode``."""

import itertools

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

from ...npu_regs import EVENT_LOCK_STALL, EVENT_MEMORY_STALL, window_counter_writes
from ...operators import templating
from ...operators.base import ArgSpec, ArtcOperator
from ...operators.chain import ChainPart, chained_fill
from ...operators.design import COLS, plus_i32, range32, stream_walk
from ...operators.kernel_source import kernel_prefix
from ...operators.kmath import ACC_C
from ...operators.l1_layout import pin_l1_buffers
from ...operators.util import Attr, regions

# scratchpad parameters
from .layout import OS, STACK, l1_layout, two_copies, workspace_regions
from .mode import AttnMode, Kind
from .ring import MT_BYTES, mt_bytes, ring_memtiles
from .shape import P_FG, P_FH, P_FO, P_FQ, P_G0, P_HK, P_P8, P_SW, Shape

_ATTN = templating.load("artlite/prefill/attn/attention.cc.jinja")


class AttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``PrepOperator``). Column c = KV head c (query heads qh c ..), core row r; in pass b the core takes the query
    blocks 4 QB b + QB r .. of each of the column's heads. Runtime sequence: QS (the prep's of sub-chunks of ``sub``
    tokens, each [column][token group][head][unit]), KV (the layer's cache: COLS KV heads of ``kv_bytes``, ``kv_pad``
    past them), A (T x heads hd as bfp16 panels [T / 8][heads hd / 8]). Parameters ``params``.

    Packed (``window`` W > 0: several inputs, segments, in the chunk): each row sees the history (the cache before p0)
    and its segment's keys up to itself, from its segment's first key lo (the SEG argument: ``seg``, an object per pass
    on stream B ahead of the V objects). Pass b streams the history and the chunk's groups max(1, D - W + 1) .. D (D:
    its queries' group; a window fixed by the design: no runtime parameters), so a segment spans at most
    ``max_segment`` positions; key chunks below every one of a core's rows' segments are skipped, those below some
    masked (-1e30). It reads at the DDR port's rate (each pass the history and its window again).

    The ring variant (``ring`` H > 0: hi + lo K, a pass per cache group, a history of at most H groups) reads each group
    once: per column the memory tile holds H groups from position 0 (the history's, the rest skipped) and a ring of W + 1
    of the chunk's groups (slot b % (W + 1) group b), and sends every pass the history and the whole ring
    (``ring.ring_memtiles``); the cores skip the slots outside their window (the chunk's groups max(0, D - W + 1) .. D), so
    a pass takes its window's key chunks in the slots' order (o, l the same sums in another order: not bit-identical to
    the window variant). A key chunk the cores skip still crosses their input streams: the ring wins from 2048 tokens.

    Non-causal (``causal`` False: LFM3's d1 decision models' full-attention layers, a prompt in one chunk at position 0):
    every pass streams the prompt's cache groups (the causal design's history part, from group 0: ``set_params``);
    row q sees the keys below the state's end S (``P_HK``) and its own, lo .. hi - 1 (the SEG argument, ``set_seg``: an
    object per pass after the GATE object, per row lo then hi, int16): the state's rows and the padding's every key of
    the prompt, a question's rows the state's and their question's. Key chunks no row of a core sees are skipped, those
    some row sees in part masked (-1e30)."""

    pfx = kernel_prefix("paa")
    # stats builds: core counters 0: cycles in the region, 1: lock (memory) stall cycles, 2: all cycles, 3: windows
    # (``npu_regs.window_counter_writes``; values at ``CORE.address(k)``)

    def __init__(self, *, T: int, shape: Shape, kv_bytes: int, kv_pad: int = 0, sub: int = 0,
                 ablate: tuple[str, ...] = (), window: int = 0, ring: int = 0, swa: int = 0, kv0: int = 0,
                 bcast: bool = True, causal: bool = True):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_<region>"
        (qk, soft, pv, raise, chunk: grp to the second pv) marks a region with event0 / event1 and the sequence programs the
        cores' performance counters (``CTRL``; "memstall": memory instead of lock stalls in counter 1). ``window``: the
        packed variant's window of cache groups; ``ring``: its history's groups held in the memory tiles (class
        docstring); ``swa``: a sliding window of that many positions (whole cache groups: each pass streams the window's
        groups before its own and its own, from ``kv0`` bytes of a KV head's cache, its guard, before group 0); ``bcast``
        (two columns per KV head, ``Shape.per``): the KV head's streams read once, by column 2 h's shim, and broadcast by
        its memtile to the cores of both columns (each taking its column's Q units and gates of both), else each column
        reads them (twice the cache's bytes: a reference); ``causal`` False: the non-causal design (class docstring)."""
        s = shape
        self.qb, self.cr = s.blocks
        self.pq = 32 * self.qb  # queries per pass
        self.m = s.pg // self.pq  # passes per cache group
        self.sub = sub or T
        assert T % s.pg == 0 and T // self.pq <= 64 and s.rg % self.cr == 0 and self.m >= 1
        assert self.qb * s.qh % 2 == 0 and T % self.sub == 0 and self.sub % self.pq == 0, "row blocks in pairs"
        assert 4 * self.qb * s.qh % 4 == 0  # (a pass's Q units: whole B objects of 4)
        assert self.pq <= 72, "a pass's segment starts in its object's first 72 values (stream B's walk)"
        self.T, self.shape, self.kv_bytes, self.kv_pad, self.ablate = T, s, kv_bytes, kv_pad, tuple(ablate)
        self.n_pass = T // self.pq
        self.mode = AttnMode.make(s, T=T, m=self.m, window=window, ring=ring, swa=swa, kv0=kv0, bcast=bcast,
                                  causal=causal)
        if ring:
            assert mt_bytes(s, ring=ring, window=window, qb=self.qb) <= MT_BYTES, "a column's memory tile"
        self.oc = 2 if two_copies(self) is not None else 1  # (o's copies: ``l1_layout``)
        super().__init__()

    @property
    def win(self) -> int:
        return self.mode.window

    @property
    def ring(self) -> int:
        return self.mode.ring

    @property
    def swa(self) -> int:
        return self.mode.swa

    @property
    def kv0(self) -> int:
        return self.mode.kv0

    @property
    def bc(self) -> bool:
        """One stream per KV head, broadcast to its columns' cores (``AttnMode.bcast``)."""
        return self.mode.bcast

    @property
    def nc(self) -> bool:
        return self.mode.kind is Kind.NONCAUSAL

    @property
    def pp_bytes(self) -> int:
        """Bytes of the P panels of a chunk (the kernels' pp)."""
        return 72 * self.qb * self.shape.qh * self.cr * self.shape.pp

    @property
    def o_bytes(self) -> int:
        """Bytes of one copy of o."""
        return 256 * self.shape.nd * self.qb * self.shape.qh

    def workspace(self) -> dict[str, int]:
        gt = [("gt", 32 * self.qb * self.shape.qh)] if self.shape.gate else []
        return regions(workspace_regions(self.qb * self.shape.qh, self.cr, self.qb if self.win else 0,
                                    self.qb if self.nc else 0) + gt)

    @property
    def max_segment(self) -> int:
        """The packed variant's longest segment: its window holds its keys wherever in a cache group it starts."""
        return (self.win - 1) * self.shape.pg + 1 if self.win else self.T

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        tri = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j > q)
        tril = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j <= q)
        w = self.workspace()
        cur = f" + {OS} * AW(int32_t, {w['st']})[15]" if self.oc == 2 else ""
        return _ATTN.render(p=p, w=Attr(w), ND=s.nd, QB=self.qb, QH=s.qh, RB=self.qb * s.qh, CR=self.cr,
                            OC=self.oc, OS=OS, CUR=cur, c=ACC_C,
                            KP=s.kp, QP=s.qp, PP=s.pp, M=self.m, QUNIT=s.qunit, tri=f"{tri:016X}", ablate=self.ablate,
                            WIN=self.win, RG=s.rg, RING=self.ring, R=self.win + 1, GQ=s.rg // self.cr,
                            HBC=self.ring * s.rg // self.cr, SWA=self.swa > 0, WR=self.swa // 8,
                            WG=self.swa // s.pg, tril=f"{tril:016X}", GATE=s.gate, NC=self.nc)

    @property
    def n_qs(self) -> int:
        return self.T // 8 * COLS * self.shape.tgu // 2

    @property
    def n_kv(self) -> int:
        return (self.shape.kvh * self.kv_bytes + self.kv_pad) // 2

    @property
    def n_a(self) -> int:
        return self.T * COLS * self.shape.qh * self.shape.hd * 9 // 16

    @property
    def n_seg(self) -> int:
        """bf16 of the SEG argument (packed): an object of two units per pass."""
        return self.n_pass * self.shape.unit

    def get_arg_spec(self) -> list[ArgSpec]:
        seg = [ArgSpec("in", (self.n_seg,), bfloat16)] if self.mode.has_seg else []
        return [ArgSpec("in", (self.n_qs,), bfloat16), ArgSpec("in", (self.n_kv,), bfloat16),
                ArgSpec("out", (self.n_a,), bfloat16), *seg]

    def seg(self, lo: np.ndarray) -> np.ndarray:
        """The SEG argument's bytes (uint8) of a packed chunk: ``lo`` (T,) per token the chunk-relative first key of its
        segment (a padding token's: itself) -> per pass its rows' as int16 at its object's start."""
        assert self.win and lo.shape == (self.T,) and (lo <= np.arange(self.T)).all() and (lo >= 0).all()
        out = np.zeros((self.n_pass, self.shape.unit), np.int16)
        out[:, :self.pq] = lo.reshape(self.n_pass, self.pq)
        return out.view(np.uint8).reshape(-1)

    def set_seg(self, sets: tuple[int, ...]) -> np.ndarray:
        """The SEG argument's bytes (uint8) of a non-causal prompt of n = sets[-1] tokens, a state and questions after it
        (``sets``: bounds S, e1, .., en, as ``NpuModel.outputs`` takes them) -> per pass its rows' lo (values 0 ..) and
        hi (values 72 ..: stream B's walk puts them 288 bf16 into the unit) as int16: the state's rows and the padding's
        0, n; question k's e(k-1), ek."""
        assert self.nc and len(sets) >= 1 and sets[0] >= 1 and list(sets) == sorted(sets) and sets[-1] <= self.T, sets
        n = sets[-1]
        lo, hi = np.zeros(self.T, np.int16), np.full(self.T, n, np.int16)
        for a, b in itertools.pairwise(sets):
            lo[a:b], hi[a:b] = a, b
        out = np.zeros((self.n_pass, self.shape.unit), np.int16)
        out[:, :self.pq] = lo.reshape(self.n_pass, self.pq)
        out[:, 72:72 + self.pq] = hi.reshape(self.n_pass, self.pq)
        return out.view(np.uint8).reshape(-1)

    def set_params(self, sets: tuple[int, ...]) -> dict[str, int]:
        """The scratchpad parameters of a non-causal prompt (``set_seg``'s ``sets``): its cache groups every pass streams
        from position 0, the state's end."""
        s = self.shape
        g = -(-sets[-1] // s.pg)
        return {P_P8: 0, P_FG: g, P_FQ: g * s.rg // self.cr, P_HK: sets[0]}

    def params(self, p0: int, lead: int = 0) -> dict[str, int]:
        """The scratchpad parameters of a chunk at p0 (packed: ``lead``, its first keys every row sees: 0 or a cache
        group, a prompt every segment continues; its rows a segment of their own)."""
        s = self.shape
        assert not self.nc, "non-causal: set_params"
        assert p0 % s.pg == 0 and lead in (0, s.pg) and (self.win or not lead)
        if self.swa:
            return {P_P8: p0 // 8, P_SW: s.group // 2 * (p0 // s.pg)}
        fg = p0 // s.pg + 1
        if self.ring:
            assert fg - 1 + lead // s.pg <= self.ring, (p0, lead, self.ring, "a history the memory tiles hold")
            return {P_P8: p0 // 8, P_FH: (fg - 1 + lead // s.pg) * s.rg // self.cr, P_G0: s.group // 2 * (fg - 1),
                    P_HK: lead}
        out = {P_P8: p0 // 8, P_FG: fg, P_FQ: fg * s.rg // self.cr} | ({P_HK: lead} if self.win else {})
        if self.n_pass > self.m:  # (a pass streams the chunk's groups after its first)
            out[P_FO] = s.group // 2 * fg
        return out

    def get_mlir_artifact(self) -> Module:
        p, s, qb, cr, m = self.pfx, self.shape, self.qb, self.cr, self.m
        nd, qh, rb = s.nd, s.qh, qb * s.qh
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        u_ty = np.ndarray[(s.unit // 2,), bf]
        q_ty = np.ndarray[(rb * s.qunit // 2,), bf]
        o_ty = np.ndarray[(qb * nd * 36,), bf]  # a head's A panels of the core's blocks
        om_ty = np.ndarray[(4 * qb * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[((OS if self.oc == 2 else 0) + 64 * nd * rb,), f32]
        pp_ty = np.ndarray[(self.pp_bytes,), np.dtype[np.uint8]]
        win = self.win
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty, np.int32, np.int32, np.int32] + [np.int32] * (
            2 if win else 1 if self.nc else 0))
        k_seg = Kernel(f"{p}_seg", "kernel.o", [u_ty, ws_ty, np.int32]) if win or self.nc else None
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [u_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, u_ty, u_ty, ws_ty, np.int32, np.int32] if s.kp == 1 else
                      [q_ty, u_ty, ws_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [u_ty, ws_ty, wo_ty, pp_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty, np.int32])
        k_gsel = Kernel(f"{p}_gsel", "kernel.o", [u_ty, ws_ty, np.int32]) if s.gate else None
        p_p8, p_fq = ScratchpadParameter(P_P8, np.int32), ScratchpadParameter(P_FH if self.ring else P_FQ, np.int32)
        p_fg, p_fo = ScratchpadParameter(P_FG, np.int32), ScratchpadParameter(P_FO, np.int32)
        p_sw = ScratchpadParameter(P_SW, np.int32)
        n_pass, ku, gq = self.n_pass, cr * s.kp // 2, s.rg // cr

        # stream A: the K records (hi + lo: a group's [K obj | padding | K obj], the memtile reads the objects' records)
        ilv = stream_walk(s.unit // 2, [(nd, 36), (2, nd * 36), (36, 1)])  # (per d-block the unit's two halves)
        if self.ring:  # (a group's K records without the padding; the memory tiles' own buffers: ``ring.ring_memtiles``)
            ka_ty = np.ndarray[(s.used,), bf]
            walk = stream_walk(s.used, [(s.used // 576, 576), (576, 1)])
        elif s.kp == 2:
            ka_ty = np.ndarray[((s.a_bytes + s.used) // 2,), bf]
            walk = stream_walk((s.a_bytes + s.used) // 2, [(2, s.a_bytes // 2), (s.used // 1152, 576), (576, 1)],
                               pad=[(0, 0)] * 3)
        else:  # (a unit's two records interleaved per d-block)
            ka_ty, walk = u_ty, ilv
        scols = list(range(0, COLS, s.per)) if self.bc else list(range(COLS))  # (the columns whose shims stream)
        a3 = {c: ObjectFifo(ka_ty, name=f"la_ka{c}", depth=2) for c in scols}
        # (single K: a core takes a chunk's two K units at once, its two L1 objects; the memtile's end one deeper, so
        # the cores' depth is the stated 2, not the acquire's 3)
        a1 = {c: a3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=3 if s.kp == 1 else 2, name=f"la_k{c}",
                                      to_stream=walk) for c in scols}
        # stream B: objects of two units, four slices of nd panels (4 V records; 4 Q units: two pairs of row blocks)
        # -> per d half [d-block pair][slice][d-block 2]
        vwalk = stream_walk(s.unit, [(2, nd // 2 * 36), (nd // 4, 72), (4, nd * 36), (72, 1)])
        b3 = {c: ObjectFifo(np.ndarray[(s.unit,), bf], name=f"la_vb{c}", depth=2) for c in scols}
        b1 = {c: b3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=2, name=f"la_v{c}", to_stream=vwalk)
              for c in scols}
        o3 = [ObjectFifo(om_ty, name=f"la_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([qb * nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"la_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb, nq, seg=None,
                 hk=None, full=False, gsel=None):
            """(a core's share of a pass: its Q units of the pass's nq, its segments and lead (packed), the history, its output;
            ``full``: the packed window's last groups only, ``win`` of them; the ring: the held history's and the ring's
            chunks, every pass)"""
            for i in range_(nq):
                u = b_h.acquire(1)
                qsel(u, q, i, me)
                b_h.release(1)
            if gsel is not None:  # (the pass's GATE object: two units, the first read)
                u = b_h.acquire(1)
                gsel[0](u, ws, gsel[1])
                b_h.release(1)
                b_h.acquire(1)
                b_h.release(1)
            if seg is not None:  # (the pass's SEG object: two units, the first read)
                u = b_h.acquire(1)
                seg(u, ws, row)
                b_h.release(1)
                b_h.acquire(1)
                b_h.release(1)
            if win:
                init(ws, wo, p8.read(), bg, bb, fq.read(), hk.read())
            elif self.nc:
                init(ws, wo, p8.read(), bg, bb, hk.read())
            else:
                init(ws, wo, p8.read(), bg, bb)
            if self.swa:  # (the window's groups and the pass's)
                n = gq * (self.swa // s.pg + 1)
            else:
                n = fq.read()  # (read after this dispatch's first object: the host wrote it before the fills)
            if self.swa or self.nc:  # (non-causal: the prompt's groups, every pass)
                pass
            elif self.ring:
                n = gq * (self.ring + win + 1)
            elif full:
                n = plus_i32(n, gq * win)
            else:
                for _ in range(gq):
                    n = n + bg
            for t in range32(n):  # (a chunk: ``qk`` of its first K unit starts it, ``pv`` of its first V half the softmax)
                if s.kp == 1:  # (its two K units at once)
                    k = a_h.acquire(2)
                    qk(q, k[0], k[1], ws, t, row)
                    a_h.release(2)
                else:
                    for kk in range(ku):
                        k = a_h.acquire(1)
                        qk(q, k, ws, kk, t, row)
                        a_h.release(1)
                for uu in range(2):
                    v = b_h.acquire(1)
                    pv(v, ws, wo, pp, uu)
                    b_h.release(1)
            for h in range(qh):
                o = o_h.acquire(1)
                fin(ws, wo, o, h)
                o_h.release(1)

        npg = n_pass // m  # (the chunk's groups)

        def core(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, *extra):
            seg, gs = (extra[:-2], extra[-2:]) if s.gate else (extra, None)
            if self.ring:  # (every pass the held history and the whole ring)
                for _ in forever_():
                    for bg in range_(npg):
                        turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, 0,
                             2 * rb * s.qp, *seg)
                return
            for _ in forever_():
                # (packed: the passes whose window is every group of the chunk after its first, then the rest's last
                # win)
                for bg in range_(min(npg, win + 1) if win else npg):
                    for bb in range(m):  # (the pass's units to every core: each keeps its own)
                        turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb,
                             2 * rb * s.qp * (s.per if self.bc else 1), *seg, gsel=gs)
                if win and npg > win + 1:
                    for bg in range_(win + 1, npg):
                        for bb in range(m):
                            turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb,
                                 2 * rb * s.qp, *seg, full=True)

        def src(c: int) -> int:  # (the column whose streams column c's cores take)
            return c - c % s.per if self.bc else c

        def jb(c: int) -> int:  # (column c's place among a broadcast's columns)
            return c % s.per if self.bc else 0

        workers = [Worker(core, [a1[src(c)].cons(depth=2), b1[src(c)].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk, k_pv,
                                 k_fin, Buffer(ws_ty, name=f"la_ws{c}_{r}"),
                                 Buffer(wo_ty, name=f"la_wo{c}_{r}"), Buffer(pp_ty, name=f"la_pp{c}_{r}"),
                                 Buffer(q_ty, name=f"la_q{c}_{r}"), p_p8, p_fq,
                                 rb * (r + 4 * jb(c)), r,
                                 *([k_seg, ScratchpadParameter(P_HK, np.int32)] if win or self.nc else []),
                                 *([k_gsel, r + 4 * jb(c)] if s.gate else [])],
                          tile=Tile(c, 2 + r), stack_size=STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        n_qs, n_kv, n_a, NH = self.n_qs, self.n_kv, self.n_a, COLS * qh
        uw, hb, g2, o2, ow = s.qunit // 2, self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2
        rows, sg = ow // 576, self.sub // 8  # (a sub-chunk's token groups)
        if s.kp == 2:  # (a group's [K obj | padding | K obj])
            n2 = (s.a_bytes + s.used) // 2
            k_tap = [1, 1, 16, n2 // 16], [0, g2, n2 // 16, 1]
        else:
            k_tap = [1, 1, rows, 576], [0, g2, 576, 1]

        n_seg = self.n_seg

        def seq(QS, KV, A, *rest):
            *seg, ah, bh, oh = rest  # (packed: SEG)

            def kv_part(off: int, n: int = 0, *, sizes: list[int], strides: list[int]) -> ChainPart:
                """A KV head's groups from ``off`` (bf16): the history's (P_FG groups, n = 0) or n of the chunk's from
                P_FO on (sizes [1, 1, d1, d0]: a group's part)."""
                if n:
                    return ChainPart(KV, TensorAccessPattern((n_kv,), off, [1, n, *sizes[2:]], strides),
                                     offset_parameter=p_fo)
                return ChainPart(KV, TensorAccessPattern((n_kv,), off, [1, 2, *sizes[2:]], strides),
                                 length_parameter=p_fg, length_unit=sizes[2] * sizes[3])

            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):
                mc = [int(a[3:]) for a in self.ablate if a.startswith("mc_")]  # (memory-module events, e.g. bank conflicts)
                ctrl = window_counter_writes(stall=EVENT_MEMORY_STALL if "memstall" in self.ablate else EVENT_LOCK_STALL,
                                             memory_events=mc)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in ctrl:
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 qb b .. of the column's heads (a pass's 4 cores joined)
                oh[c].drain(A, TensorAccessPattern((n_a,), c * qh * nd * 36, [n_pass, qh, 4 * qb, nd * 36],
                                                   [4 * qb * NH * nd * 36, nd * 36, NH * nd * 36, 1]),
                            group=tg, wait=True)
            if self.ring:
                ring_fills(QS, KV, seg[0], ah, bh, tg)
                return
            # (a pass's fills: at most three in flight, within the shim tiles' BDs; packed or gated: two, their SEG / GATE
            # parts more)
            inflight = []
            tw, gw2, wg = s.tgu // 2, s.gw // 2, self.swa // s.pg  # (QS's token groups; a group's gates; window groups)
            for b in range(n_pass):
                tp = TaskGroup()
                ng = b // m  # (the chunk's groups after its first that the pass streams; packed: the last win of them)
                j0, nw = (max(1, ng - win + 1), min(ng, win)) if win else (1, 0 if self.nc else ng)
                t0 = 4 * qb * b  # (the pass's first token group: its sub-chunk, the group in it)
                q0 = t0 // sg * sg * COLS * tw + t0 % sg * tw
                for i, c in enumerate(scols):
                    h = c // s.per  # (the column's KV head)
                    k0, v0 = h * hb + self.kv0 // 2, h * hb + self.kv0 // 2 + s.kp * o2
                    if self.swa:  # (the window's groups before the pass's and its own, from the guard on)
                        w0 = (ng - wg) * g2
                        parts = [ChainPart(KV, TensorAccessPattern((n_kv,), k0 + w0, [1, wg + 1, *k_tap[0][2:]],
                                                                   [0, *k_tap[1][1:]]), offset_parameter=p_sw)]
                    else:
                        parts = [kv_part(k0, sizes=k_tap[0], strides=k_tap[1])]
                        if nw:
                            parts.append(kv_part(k0 + (j0 - 1) * g2, nw, sizes=k_tap[0], strides=k_tap[1]))
                    chained_fill(ah[i], KV, parts, tp, wait=True)
                    qc, nc = q0 + c * sg * tw, s.per if self.bc else 1  # (the stream's columns: their Q units, gates)
                    if s.gate:  # (the pass's Q units, then its GATE object: its token groups' gates, padding to 2 units)
                        parts = [ChainPart(QS, TensorAccessPattern((n_qs,), qc, [1, nc, 4 * qb, qh * uw],
                                                                   [0, sg * tw, tw, 1])),
                                 ChainPart(QS, TensorAccessPattern((n_qs,), qc + qh * uw, [1, nc, 4 * qb, gw2],
                                                                   [0, sg * tw, tw, 1])),
                                 ChainPart(QS, TensorAccessPattern((n_qs,), 0, [1, 1, 1, s.unit - nc * 4 * qb * gw2],
                                                                   [0, 0, 0, 1]))]
                    else:
                        parts = [ChainPart(QS, TensorAccessPattern((n_qs,), qc, [1, 1, 4 * qb * qh, uw], [0, 0, uw, 1]))]
                    if seg:
                        parts.append(ChainPart(seg[0], TensorAccessPattern((n_seg,), b * s.unit,
                                                                           [1, 1, s.unit // 576, 576], [0, 0, 576, 1])))
                    if self.swa:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), v0 + w0, [1, wg + 1, rows, 576],
                                                                       [0, g2, 576, 1]), offset_parameter=p_sw))
                    else:
                        parts.append(kv_part(v0, sizes=[1, 1, rows, 576], strides=[0, g2, 576, 1]))
                        if nw:
                            parts.append(kv_part(v0 + (j0 - 1) * g2, nw, sizes=[1, 1, rows, 576],
                                                 strides=[0, g2, 576, 1]))
                    chained_fill(bh[i], KV, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > (1 if win or s.gate else 2):  # (a GATE object: two parts more)
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        def ring_fills(QS: object, KV: object, SEG: object, ah: list, bh: list, tg: TaskGroup) -> None:
            """The ring variant's fills: per column the held history (groups from position 0, the chunk's first among
            them) once, then pass b's group b (stream A: its K; B: the pass's Q units, SEG, its V), each read once."""
            p_g0, hold = ScratchpadParameter(P_G0, np.int32), self.ring
            hist = TaskGroup()
            for c in range(COLS):  # (a group's K: its two objects' records, without the first one's padding)
                k0, v0 = c * hb, c * hb + s.kp * o2
                chained_fill(ah[c], KV, [TensorAccessPattern((n_kv,), k0, [hold, 2, rows, 576], [g2, o2, 576, 1])],
                             hist, wait=True)
                chained_fill(bh[c], KV, [TensorAccessPattern((n_kv,), v0, [hold, 1, rows, 576], [g2, 0, 576, 1])], hist,
                             wait=True)
            inflight = [hist]
            for b in range(n_pass):
                tp = TaskGroup()
                t0 = 4 * qb * b
                q0 = t0 // sg * sg * COLS * qh * uw + t0 % sg * qh * uw
                for c in range(COLS):
                    k0, v0 = c * hb + b * g2, c * hb + s.kp * o2 + b * g2
                    chained_fill(ah[c], KV, [ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 2, rows, 576],
                                                                               [0, o2, 576, 1]), offset_parameter=p_g0)],
                                 tp, wait=True)
                    chained_fill(bh[c], KV, [
                        ChainPart(QS, TensorAccessPattern((n_qs,), q0 + c * sg * qh * uw, [1, 1, 4 * qb * qh, uw],
                                                          [0, 0, uw, 1])),
                        ChainPart(SEG, TensorAccessPattern((n_seg,), b * s.unit, [1, 1, s.unit // 576, 576],
                                                           [0, 0, 576, 1])),
                        ChainPart(KV, TensorAccessPattern((n_kv,), v0, [1, 1, rows, 576], [0, g2, 576, 1]),
                                  offset_parameter=p_g0)], tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 2:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           *([np.ndarray[(n_seg,), bf]] if win or self.nc else []),
                           [a3[c].prod(tile=Tile(c, 0)) for c in scols],
                           [b3[c].prod(tile=Tile(c, 0)) for c in scols],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = l1_layout(self)
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        if self.ring:
            ring_memtiles(self, module)
        return module
