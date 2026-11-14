"""``Gemm``: y = x W^T + b (+ a residual) of the vision overlay for every column's tokens, its output kinds (``Out``)
and the memtile tasks and B stream of the op.

Per column: ``ntile`` tiles of m tokens. Core (c, r) computes row r's N / 4 output channels in passes of 80. Memtile tasks
per column: per core one send per tile (a BD per pass: the k-tiles of the column's panels, d3 over the k-tiles; with a
residual a chained BD reads the pass's C-tile object), and receives (per tile the passes' outputs: bfp16 panels
scattered into a panel tensor [kb][token panel], or bf16 C-tile objects; statistics records into ``rec``).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from ml_dtypes import bfloat16

from .. import core as K
from .. import design as D
from .. import layout as L
from ..roles import Role
from ..seq import ColOps, MemTask, RowOps
from .common import (
    ROPE2_TAB,
    ROPE40_TAB,
    Command,
    check_heap,
    check_memtile,
    contig,
    f32,
    gn_window,
    mac_walks,
    panel_recv,
)
from .conv import ConvSpec


class Out(StrEnum):
    """What a GEMM outputs: bf16 C tiles or bfp16 panels, transformed by an epilogue (``OUT``)."""

    BF16 = "bf16"
    BFP16 = "bfp16"
    GEGLU = "geglu"
    VT = "vt"
    DS = "ds"
    GELU = "gelu"
    ROPE = "rope"
    ROPE2 = "rope2"
    ROPE40 = "rope40"
    PROD = "prod"
    F32 = "f32"
    ISIG = "isig"


class Act(StrEnum):
    """The activation of a GEGLU / GELU output."""

    GELU = "gelu"
    SILU = "silu"


class Stats(StrEnum):
    """The statistics of an op's output: GroupNorm group sums, LayerNorm per-token sums."""

    GN = "gn"
    LN = "ln"


@dataclass(frozen=True)
class OutKind:
    """One output kind: ``core``'s output code (``code``) and epilogue word (``epi``); ``panels``: its panel tensor has N /
    panels kb (None: C tiles, or fixed-size passes); ``role``: the page its epilogue is (``silu``: with ``act`` silu); the
    shapes it takes: ``m`` (None: either), ``one_tile``, ``res`` (True: required, False: none, None: either), ``qkn``
    (the heads' norm weights: required), ``stats`` (False: none), ``conv`` (False: none)."""

    code: int
    panels: int | None = None
    role: Role | None = None
    silu: Role | None = None
    epi: int = 0
    m: int | None = None
    one_tile: bool = False
    res: bool | None = None
    qkn: bool = False
    stats: bool = True
    conv: bool = True


OUT: dict[Out, OutKind] = {
    Out.BF16: OutKind(K.O_BF16),
    Out.BFP16: OutKind(K.O_BFP16, panels=8),
    Out.GEGLU: OutKind(K.O_GEGLU, panels=16, role=Role.LN, silu=Role.D),
    Out.VT: OutKind(K.O_VT, panels=8),
    Out.DS: OutKind(K.O_DS, res=False, stats=False),  # (a stride-2 row conv's: ``ConvSpec.rc``, rstep 2)
    Out.GELU: OutKind(K.O_GELU, panels=8, role=Role.LN, silu=Role.R),
    Out.ROPE: OutKind(K.O_ROPE, panels=8, role=Role.R, m=64, one_tile=True, res=True, stats=False, conv=False),
    Out.ROPE2: OutKind(K.O_ROPE, panels=8, role=Role.R, m=16, res=True, qkn=True, stats=False, conv=False),
    Out.ROPE40: OutKind(K.O_ROPE, panels=8, role=Role.E, epi=K.E_ROPE40, m=16, res=True, qkn=True, stats=False,
                        conv=False),
    Out.PROD: OutKind(K.O_GELU, role=Role.D, one_tile=True, stats=False, conv=False),  # (role d's code at GELU's place)
    Out.F32: OutKind(K.O_ROPE, role=Role.E, epi=K.E_F32, m=16, one_tile=True, res=False, stats=False, conv=False),
    Out.ISIG: OutKind(K.O_ROPE, role=Role.E, epi=K.E_GATE, res=False, stats=False, conv=False),
}
STATS_CODE = {None: K.S_NONE, Stats.GN: K.S_GN, Stats.LN: K.S_LN}
PASS_F32, PASS_PROD = 5760, 90  # (bytes per pass of the fp32 tile; of a prod tile per token)


@dataclass
class Gemm:
    """The layer's shape and memtile tensors: x (panels of the column's m ntile tokens x K, ``x_np`` token panels per
    kb), y (panels if out is a panel kind (``OutKind.panels``), else C tiles), res (C tiles of N channels, or None).
    out "vt": panels of the transposes ([kb][token panel] of [8 channels][8 tokens], the attention's V^T); "geglu": the
    weights' rows [hid N / 2 | gate N / 2], output hid gelu(gate) (N / 2 channels; act "silu": hid silu(gate), SwiGLU,
    role d); "prod": the same pairs' products hid gate as bf16 ([pass][rb][5 cb][64] in m 90 B per pass, role d; one
    tile); "gelu": gelu of the outputs as panels (act "silu": silu of them, role r); "rope": each pass a head of 40 pairs
    of channels (k, 40 + k) turned by its tokens' angles, as panels: res is then the tile's cos | sin (one C-tile object
    of 80 channels, the same for every pass: ``core``'s RoPE; m 64, one tile, no statistics), with ``qkn`` (80 weights,
    the pairs' layout) each head RMS-normed first (over ``qkn_n`` channels, eps ``qkn_eps``: LFM3-VL's q / k norms);
    "rope2" (m 16): a head in two passes (A, B: the pairs (k, 40 + k) of each), normed over both (``qkn``: 160 weights, A's
    then B's) and turned, res the tiles' tables [A | B] (``ROPE2_TAB`` bytes a tile: two C-tile layouts [rb][cb][64] of
    cos | sin; LFM3-Audio's heads of 96); "rope40" (m 16): each pass two heads of 40 channels (32 dims: LFM3-Audio's
    depthformer), each normed over its first 32 (``qkn``: 80 weights, a head's 40 twice) and turned (NEOX pairs (k, 16 +
    k) at its channels (k, 16 + k)), res the tile's table (``ROPE40_TAB`` bytes: [rb][cb][64] bf16, cos in cb 0, 1, sin
    in cb 2, 3), as panels; "f32" (m 16, one tile): the fp32 C tile as it is ([pass][rb][cb][64] in 5760 B per pass);
    "isig": 1 + 2^y of the outputs y as bf16 C tiles (with the weights -log2 e times LFM3's attention gate's, 1 /
    sigmoid of its logits: the attention's factor of its row sums, ``DepthAttention``, ``WinAttention``); "ds": a
    stride-2 conv's outputs (``ConvSpec`` rc, rstep 2).
    stats "gn" (groups of ``gsize`` channels of a tensor whose channel ``ch_off`` is this output's first) / "ln": records
    at ``rec`` ([tile][core][OCH] (LN), [core][OCH] (GN)); stf: ``core.ST_*`` (a GroupNorm record over several ops). The
    op runs passes [p0, p0 + npo) of each row's N / 4 channels (default all). K 0: a copy of res (C tiles) into y with
    statistics (no weights).

    conv: a 3x3 conv of x (``ConvSpec``): K input channels, the weights (N, K, 3, 3), a residual's C tiles after each
    pass's chunks (an op of 2 npo tasks per core: ``seq.Seq.op`` issues them in rounds).

    ch_step: a core row's channels in the statistics' channel space (default its passes'), ch_real: how many of them
    count (the record window). tsplit: core rows per channel slice: row r computes slice r // tsplit (N tsplit / 4
    channels) for the (r % tsplit)-th of tsplit parts of the tiles (``tiles``).

    hilo: the weights as two bfp16 parts, hi = bfp16(w) and lo = bfp16(w - hi) (about bf16's precision): K doubled, per
    pass the hi k-tiles, then the lo ones on the A panels walked again (twice the MACs and weight bytes). packed (bf16,
    N 320): row r's block of tile t at y + r ``packed_block`` + t m 160 (the C tiles without their chunks' padding: a
    block's tokens linear over the tiles, a drain's walk)."""

    m: int
    ntile: int
    K: int
    N: int
    x: int
    y: int
    res: int | None = None
    out: Out = Out.BFP16
    stats: Stats | None = None
    rec: int = 0
    gsize: int = 0
    ch_off: int = 0
    x_np: int | None = None
    y_np: int | None = None
    stf: int = 0
    p0: int = 0
    npo: int | None = None
    conv: ConvSpec | None = None
    ch_step: int = 0
    ch_real: int = 0
    tsplit: int = 1
    hilo: bool = False
    act: Act = Act.GELU
    qkn: np.ndarray | None = None
    qkn_n: int = 64
    qkn_eps: float = 1e-5
    packed: bool = False

    def __post_init__(self) -> None:
        self.out, self.act = Out(self.out), Act(self.act)
        self.stats = Stats(self.stats) if self.stats else None
        kind = OUT[self.out]
        assert self.m in (64, 16) and self.N * self.tsplit % 320 == 0 and self.K % 8 == 0, \
            f"a GEMM of m {self.m}, N {self.N} x tsplit {self.tsplit}, K {self.K}"
        self._check_out(kind)
        self.npass = self.N * self.tsplit // 320
        self.nt = self.ntile // self.tsplit  # (a core's tiles)
        self.npo = self.npo or self.npass - self.p0
        assert 0 <= self.p0 and self.p0 + self.npo <= self.npass, "the passes"
        self.nkt = -(-self.K // D.KT) * (2 if self.hilo else 1)
        assert not self.hilo or self.K and self.conv is None, "hilo: weights of K > 0, no conv"
        self.nch = 0
        if self.conv:
            self.conv.check(m=self.m, ntile=self.ntile, kdim=self.K)
            self.nch = self.K // 64
            self.nkt = 9 * self.nch
        self.np = self.m * self.ntile // 8  # (the column's token panels)
        self.x_np = self.x_np or self.np
        self.junk = 2 if self.out == Out.GEGLU and self.m == 16 else 0  # (m 16 GEGLU: 720 B per pass, a padded chunk)
        self.y_np = self.y_np or self.np + self.junk
        self.nres = -(-self.m * 160 // D.A_OBJ) if self.res is not None else 0
        self.aobj = self.nch if self.conv else -(-self.nkt * self.m // 128)  # (A objects of a pass's k-tiles)
        assert self.npass <= 16 and self.y_np >= self.np + self.junk, "the passes or y's token panels"
        ct = L.ct_bytes(self.m)
        if kind.panels:
            ybytes = self.N // kind.panels * self.y_np * L.PANEL
        elif self.out == Out.PROD:
            ybytes = self.N // 80 * self.m * PASS_PROD
        elif self.out == Out.F32:
            ybytes = self.N // 80 * PASS_F32
        else:
            ybytes = self.N // 80 * (1 if self.out == Out.DS else self.ntile) * ct
        check_heap(self.y, ybytes, "a GEMM's output tensor")
        check_heap(self.x, self.K // 8 * self.x_np * L.PANEL, "a GEMM's input tensor")
        walk = self.aobj * (128 // self.m) - (self.nkt // 2 if self.hilo else 0)  # (k-tiles of x a walk reads; hilo:
        # the second, the longer)
        if self.conv is None:
            check_memtile(self.x, walk * 8 * self.x_np * L.PANEL, "a GEMM's A objects' walk")
        nres_obj = 1 if self.out == Out.ROPE else self.ntile if self.out in (Out.ROPE2, Out.ROPE40) else \
            self.N // 80 * self.ntile  # (the residual's C-tile objects)
        if self.res is not None:
            check_memtile(self.res, (nres_obj - 1) * ct + self.nres * D.A_OBJ, "a GEMM's residual objects")
        assert self.stats != Stats.GN or self.gsize, "a GroupNorm's group size"
        assert self.stats != Stats.LN or not self.stf or self.nt == 1, "a LayerNorm record over ops of one tile"

    def _check_out(self, kind: OutKind) -> None:
        """The shapes the output kind takes (``OutKind``) and the options that need one."""
        out = self.out
        assert self.act == Act.GELU or out in (Out.GEGLU, Out.GELU), f"act {self.act} with out {out}"
        assert kind.m in (None, self.m), f"out {out}: m {kind.m}, not {self.m}"
        assert not kind.one_tile or self.ntile == 1, f"out {out}: one tile"
        assert kind.stats or self.stats is None, f"out {out}: no statistics"
        assert kind.conv or self.conv is None, f"out {out}: no conv"
        assert kind.res is None or (self.res is not None) == kind.res, f"out {out}: res {'required' if kind.res else 'none'}"
        assert self.qkn is not None or not kind.qkn, f"out {out}: the heads' norm weights (qkn)"
        assert self.qkn is None or kind.qkn or out == Out.ROPE, f"out {out}: no norm weights (qkn)"
        if out == Out.DS:
            assert self.conv and self.conv.rc and self.conv.rstep == 2, "out ds: a stride-2 row conv"
        if out == Out.ROPE2:
            assert (self.npo or self.N // 320) % 2 == 0 and self.p0 % 2 == 0, "out rope2: passes in pairs"
        assert not self.packed or out == Out.BF16 and self.N == 320 and self.tsplit == 1 and self.res is None, \
            "packed: bf16 outputs of N 320, one slice, no residual"
        assert self.tsplit in (1, 2, 4) and self.ntile % self.tsplit == 0 and (self.tsplit == 1 or out == Out.BF16), \
            f"tsplit {self.tsplit} of {self.ntile} tiles"

    @property
    def packed_block(self) -> int:
        """Bytes between the packed output's blocks (``packed``: a block's tiles m x 160 bytes apart, then a chunk's
        padding)."""
        return self.ntile * self.m * 160 + L.ct_bytes(self.m) - self.m * 160

    @property
    def role(self) -> Role | None:
        """The page of the output's epilogue (None: the resident code)."""
        kind = OUT[self.out]
        return kind.silu if self.act == Act.SILU and kind.silu else kind.role

    def ch0(self, row: int) -> int:
        """Row ``row``'s first output channel in the statistics' channel space."""
        return self.ch_off + row // self.tsplit * (self.ch_step or self.npass * 80)

    def command(self, bias: np.ndarray | None, row: int) -> np.ndarray:
        """Row ``row``'s command slot (Q bytes); bias: (N,) or None (a conv's: its bias + the time bias)."""
        cmd = Command(K.K_GEMM)
        kind = OUT[self.out]
        cmd.set([K.C_M, K.C_NTILE, K.C_NPASS, K.C_NKT, K.C_NRES, K.C_OUT, K.C_STATS, K.C_STF],
                (self.m, self.nt, self.npo, self.nkt, self.nres, kind.code, STATS_CODE[self.stats], self.stf))
        if self.stats == Stats.GN:
            c0 = self.ch0(row)
            cmd.set([K.C_GSIZE, K.C_CH0, K.C_G0, K.C_GINV],
                    (self.gsize, c0 + 80 * self.p0, gn_window(c0, self.ch_real or self.npass * 80, self.gsize),
                     K.group_inv(self.gsize)))
        if self.conv is None:
            assert K.C_BIAS * 4 + self.npo * 160 <= K.C_WALKS * 4, "the passes' biases past the walks"
            cmd.at(K.C_WALKS, mac_walks(self.m))
        if kind.epi:
            cmd.set([K.C_EPI], [kind.epi])
        if self.qkn is not None:
            two, r40 = self.out == Out.ROPE2, self.out == Out.ROPE40
            cmd.set([K.C_QKN, K.C_INVN, K.C_EPS, K.C_ROPE2], (not (two or r40), f32(1.0 / self.qkn_n), f32(self.qkn_eps), two))
            w = np.asarray(self.qkn, np.float32).reshape(160 if two else 80).astype(bfloat16)
            cmd.at(K.C_QKW, w.view(np.int32))
        if self.conv:
            cv = self.conv
            cmd.set([K.C_CONV, K.C_NCH, K.C_R, K.C_ROT, K.C_TPAR],
                    (cv.stride, self.nch, cv.R, cv.R == 2, cv.tile_word(self.tiles(row)[0], self.ntile)))
            cmd.at(K.C_CALLS, cv.words(self.nch))
            assert K.C_BIAS * 4 + self.npo * 160 <= K.C_MASKS * 4, "the passes' biases past the conv's masks"
            cmd.bytes()[4 * K.C_MASKS:4 * K.C_MASKS + 1152] = np.frombuffer(K.shift_masks(cv.joined), np.uint8)
        out = cmd.bytes().copy()
        if bias is not None:
            b = self.bias_bytes(bias, row)
            out[4 * K.C_BIAS:4 * K.C_BIAS + b.size] = b
        return out

    def tiles(self, row: int) -> range:
        """Row ``row``'s tiles of the op (``tsplit``: the (row % tsplit)-th of tsplit parts)."""
        k = row % self.tsplit
        return range(k * self.nt, (k + 1) * self.nt)

    def bias_bytes(self, bias: np.ndarray, row: int) -> np.ndarray:
        """The command's biases (at word C_BIAS): row ``row``'s of its passes, bf16."""
        return np.asarray(bias, np.float32)[self.weight_rows(row)].astype(bfloat16).view(np.uint8)

    def weight_rows(self, row: int) -> np.ndarray:
        """Row ``row``'s weight rows of the op's passes in order (GEGLU: per pass [hid 40 | gate 40])."""
        n0 = row // self.tsplit * self.npass * 80
        ps = range(self.p0, self.p0 + self.npo)
        if self.out not in (Out.GEGLU, Out.PROD):
            return np.array([n0 + 80 * p + i for p in ps for i in range(80)])
        h, g = self.N // 2, n0 // 2
        return np.array([g + 40 * p + i if i < 40 else h + g + 40 * p + i - 40 for p in ps for i in range(80)])

    @property
    def col_free(self) -> bool:
        """The op's tasks are the same in every column (none reads a neighbour's memtile or the image's edge)."""
        return self.conv is None or self.conv.col_free(self.ntile)

    def col_ops(self, col: int = 0) -> ColOps:
        """Per core its sends and receives. Tiles: the receives one task whose BDs step over the tiles; the sends one
        task (execution t NPO + p: with one pass per core or one tile, its BDs step linearly), else a task per tile."""
        m, ct, NT = self.m, L.ct_bytes(self.m), self.ntile
        P0, NPO = self.p0, self.npo
        XN, YN, TP = self.x_np, self.y_np, m // 8 * 18  # (TP: a tile's token panels in words)
        co = ColOps()
        for r in range(4):
            sends, recvs, sl = [], [], r // self.tsplit  # (sl: the row's channel slice)
            if self.conv:  # (per tile; with a residual per pass its chunks, then its residual objects)
                repeat = self.nch * (1 if self.res is not None else NPO)
                for t in self.tiles(r):
                    cs = self.conv.sends(x=self.x, x_np=XN, nch=self.nch, repeat=repeat, col=col, t=t)
                    if self.res is None:
                        sends.append(cs)
                        continue
                    ow = self.nres * D.A_OBJ // 4
                    res = MemTask([{"addr": self.res + ((sl * self.npass + P0) * NT + t) * ct, "nwords": ow,
                                    "dims": contig(ow), "iteration": (NT * ct // 4, NPO)}])
                    for _ in range(NPO):
                        sends += [cs, res]
            else:
                def chain(t: int, it: bool, sl: int = sl) -> list[dict]:  # (tile t's; it: the BDs step over the row's
                    # tiles (one pass) or passes (one tile))
                    out = []
                    if self.aobj:  # (hilo: x walked twice, the hi k-tiles' A, then the lo ones')
                        nw = self.aobj * D.A_OBJ // 4
                        for n in ([self.nkt // 2 * m * 18, nw - self.nkt // 2 * m * 18] if self.hilo else [nw]):
                            out.append({"addr": self.x + t * TP * 4, "nwords": n,
                                        "dims": [(36, 1), (8, XN * 18), (m // 16, 36), (0, 8 * XN * 18)],
                                        **({"iteration": (TP, self.nt)} if it else {})})
                    if self.out in (Out.ROPE, Out.ROPE2, Out.ROPE40):  # (the tile's cos | sin: every pass's; rope2:
                        # [A | B] tables)
                        ow = self.nres * D.A_OBJ // 4
                        tab = {Out.ROPE2: ROPE2_TAB, Out.ROPE40: ROPE40_TAB}.get(self.out, ct)
                        out.append({"addr": self.res + t * tab, "nwords": ow, "dims": contig(ow)})
                    elif self.res is not None:  # (the object (sl npass + P0 + p, t))
                        ow = self.nres * D.A_OBJ // 4
                        it_res = (ct // 4, self.nt) if NPO == 1 else (NT * ct // 4, NPO)
                        out.append({"addr": self.res + ((sl * self.npass + P0) * NT + t) * ct, "nwords": ow,
                                    "dims": contig(ow), "iteration": it_res if it else (NT * ct // 4, NPO)})
                    return out
                if self.nt == 1 or NPO == 1:
                    sends.append(MemTask(chain(self.tiles(r)[0], True), repeat=self.nt * NPO))
                else:
                    sends += [MemTask(chain(t, False), repeat=NPO) for t in self.tiles(r)]
            if OUT[self.out].panels and self.out != Out.GEGLU:  # (pass p: [rb][cb] -> kb (r npass + p) 10 + cb, token
                # panel (m / 8) t + rb)
                data = panel_recv(self.y + (r * self.npass + P0) * 10 * YN * L.PANEL, YN, m // 8, 10, NPO)
                data["iteration"] = (TP, NT)
            elif self.out in (Out.PROD, Out.F32):  # (pass p: [rb][5 cb][64] bf16 in m 90 B or [rb][cb][64] fp32 in
                # 5760 B (and a chunk's padding) at (r npass + p))
                pb = self.m * PASS_PROD if self.out == Out.PROD else PASS_F32
                data = {"addr": self.y + (r * self.npass + P0) * pb, "nwords": NPO * pb // 4}
            elif self.out == Out.GEGLU:  # (pass p: [rb][5 cb] -> kb (r npass + p) 5 + cb)
                data = panel_recv(self.y + (r * self.npass + P0) * 5 * YN * L.PANEL, YN, m // 8, 5, NPO, self.junk)
                data["iteration"] = (TP, NT)
            elif self.out == Out.DS:  # (tile t's pass p: groups 0 and 2 (2 x 2560 B, each in two chunks) -> the 16x16
                # C-tile object (r npass + p), its row t: the second call's data over the first's chunk padding)
                data = {"addr": self.y + (r * self.npass + P0) * ct, "nwords": NT * NPO * 1440,
                        "dims": [(720, 1), (2, 640), (NPO, ct // 4), (0, 1280)]}
            elif self.packed:  # (row r's block r of tile t at r packed_block + t m 160: its chunks' padding over the
                # next tile's start, before that tile's data lands)
                data = {"addr": self.y + r * self.packed_block, "nwords": ct // 4, "iteration": (m * 40, NT)}
            else:  # (pass p: the C-tile object (sl npass + p, t))
                data = {"addr": self.y + ((sl * self.npass + P0) * NT + self.tiles(r)[0]) * ct, "nwords": NPO * ct // 4,
                        "dims": contig(ct // 4) + [(NPO, NT * ct // 4)], "iteration": (ct // 4, self.nt)}
            rc = [data]
            if self.stats == Stats.LN and not self.stf & K.ST_HOLD:
                rc.append({"addr": self.rec + r * D.OCH, "nwords": D.OCH // 4, "iteration": (D.OCH, NT)})
            recvs.append(MemTask(rc, repeat=1 if self.out == Out.DS else self.nt))
            if self.stats == Stats.GN and not self.stf & K.ST_HOLD:
                recvs.append(MemTask([{"addr": self.rec + r * D.OCH, "nwords": D.OCH // 4}]))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, img: int, wimg: int | None = None) -> RowOps:
        """img: the device address of the row's image: its command slot, then its weights (one run; the weights
        again per further tile); wimg: the weights' own image."""
        wb = self.npo * self.nkt * D.Q
        runs = [(img, D.Q + wb)] + [(img + D.Q, wb)] * (self.nt - 1 if wb else 0) if wimg is None else \
            [(img, D.Q)] + [(wimg, wb)] * self.nt
        return RowOps(runs=runs, b_objects=1 + self.nt * self.npo * self.nkt, parity=self.m == 64 and self.nkt > 0,
                      role=self.role)
