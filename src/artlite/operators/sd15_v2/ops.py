"""sd15_v2's layers as ops of the sequencer (``seq``): per column its memtile tasks, per row its B stream.

``Gemm``: y = x W^T + b (+ a residual) for every column's tokens (``ntile`` tiles of m). Core (c, r) computes row r's
N / 4 output channels in passes of 80. Memtile tasks per column: per core one send per tile (a BD per pass: the
k-tiles of the column's panels, d3 over the k-tiles; with a residual a chained BD reads the pass's C-tile object), and
receives (per tile the passes' outputs: bfp16 panels scattered into a panel tensor [kb][token panel], or bf16 C-tile
objects; statistics records into ``rec``).

``Apply``: GroupNorm (+ SiLU) / LayerNorm of a bf16 C-tile tensor into panels: the i-th core of its rows
(``design.APPLY_ROWS``) normalizes the 80-channel blocks [blk0 + i nblk, ..) for the column's
tokens.

``Attention``: 8 heads of d 160 (16x16, 8x8): the queries of a column's tokens against the keys of their image (K / V^T
slots in DDR, written by ``Attention.kv_drains``, on the B streams of the attention rows).

Tensors are memtile byte addresses, the same in every column; ``np`` is a panel tensor's token panels per channel block
(the column's tokens / 8, or more: a padded tensor), in (image row, group, image) order: token panel tp is image tp % 2.
"""

import struct
from dataclasses import dataclass

import numpy as np
import torch
from ml_dtypes import bfloat16

from ... import txn
from ..design import COLS
from . import core as K
from . import design as D
from . import layout as L
from .seq import ColOps, MemTask, RowOps


def _contig(words: int) -> list[tuple[int, int]]:
    """Dims of ``words`` contiguous words (d0 <= 1023)."""
    d0 = next(d for d in range(min(words, 1023), 0, -1) if words % d == 0)
    return [(d0, 1), (words // d0, d0)] if words > d0 else [(d0, 1)]


def _dims(*ds: tuple[int, int]) -> list[tuple[int, int]]:
    """A memtile walk (size, stride) innermost first without its 1-dims; a fourth dim's size is implied."""
    out = [d for d in ds[:-1] if d[0] != 1] + [ds[-1]]
    assert len(out) <= 4, ds
    if len(out) == 4:
        out[3] = (0, out[3][1])
    return out


def _f32(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def _panel_recv(addr: int, np_: int, nrb: int, ncb: int, nblk: int, junk_rb: int = 0) -> dict:
    """A receive BD of one tile's output panels [block][rb][cb] -> panel tensor kb = 10 block + cb (ncb per block) from
    ``addr``'s kb 0, token panel rb (junk_rb more padding rows per block: panels beyond the tokens)."""
    rbs = nrb + junk_rb
    return {"addr": addr, "nwords": nblk * rbs * ncb * 18,
            "dims": [(18, 1), (ncb, np_ * 18), (rbs, 18), (0, ncb * np_ * 18)]}


def mac_walks(m: int) -> np.ndarray:
    """The MAC's walks of A, B and C (``core.C_WALKS``, ``C_PATS``: three dims_3d_t) of a GEMM's m-token C tile (m 64:
    the band pairs (0, 2), (1, 3) alternate per block, C's halves in two L1 regions; m 16: one band; a d-80 attention's
    P V^T: one band of a query pair)."""
    if m == 64:
        w = [(1, 1152, 4, -3456, -2304), (1, -1152, 4, 0, -5760), (1, 2560, 4, -2432, -1792)]
    else:
        w = [(4, -1152, m // 16 - 1, 0, 0), (4, 0, m // 16 - 1, -5760, -5760), (4, 128, m // 16 - 1, 768, 768)]
    return np.array([x for d in w for x in (*d, 0, 0)], np.int64).astype(np.int32)


def gn_window(ch0: int, n: int, gsize: int) -> int:
    """The first group of the 16-group window of a GroupNorm record (16-aligned) holding channels [ch0, ch0 + n): a
    producer's record and its consumer's must agree."""
    g0 = ch0 // gsize // 16 * 16
    assert (ch0 + n - 1) // gsize < g0 + 16, (ch0, n, gsize)
    return g0


@dataclass
class Gemm:
    """The layer's shape and memtile tensors: x (panels of the column's m ntile tokens x K, ``x_np`` token panels per
    kb), y (panels if out != "bf16", else C tiles), res (C tiles of N channels, or None). out "vt": panels of the
    transposes ([kb][token panel] of [8 channels][8 tokens], the attention's V^T); "geglu": the weights' rows
    [hid N / 2 | gate N / 2], output hid gelu(gate) (N / 2 channels). stats "gn" (groups of ``gsize`` channels of a
    tensor whose channel ``ch_off`` is this output's first) / "ln": records at ``rec`` ([tile][core][OCH] (LN), [core][OCH]
    (GN)); stf: ``core.ST_*`` (a GroupNorm record over several ops). The op runs passes [p0, p0 + npo) of each row's
    N / 4 channels (default all). K 0: a copy of res (C tiles) into y with statistics (no weights).

    conv 1 / 2: a 3x3 conv (padding 1, stride ``conv``) of x (K input channels) for the column's tile (m = 16 R Lg
    tokens: R rows of Lg groups, x = g + Lg lane; the input's rows R conv of Lg conv groups); weights (N, K, 3, 3);
    the halo rows from the neighbour columns' memtiles (``zero``: a memtile region of zeros for the image's edges,
    ``conv_zero_bytes``); with a residual the sends are per pass the chunks' task, then the residual's (an op of
    2 npo tasks per core: ``seq.Seq.op`` issues them in rounds). Rows (``rc``: x holds the column's rc rows of 4 groups,
    32x32): the op's ntile tiles are output rows r0 .. of 4 groups (m 64); per chunk an A object per dy: the 8 (row,
    group) pairs from input row r + dy - 1's predecessor group (slot 0, then the row's groups in slots 1-4, slot 5 the
    next row's group 0), into whose slots 0 and 5 the core shifts the row's group 3 one lane down and its group 0 one
    lane up: every tap's 4 bands are slots dx .. dx + 3 (``conv_words``). il (64x64, Lg 8): x holds rc rows of 8 groups
    (the halo rows included), a row's even groups then its odd ones; tile t = output row r0 + rstep (t // 2)'s even
    (t even) or odd groups (par0: row r0 + rstep t's even groups, a stride-2 conv's outputs); per chunk an A object per
    dy: input row r + dy - 1 (``_il_words``)."""

    m: int
    ntile: int
    K: int
    N: int
    x: int
    y: int
    res: int | None = None
    out: str = "bfp16"
    stats: str | None = None
    rec: int = 0
    gsize: int = 0
    ch_off: int = 0
    x_np: int | None = None
    y_np: int | None = None
    stf: int = 0
    p0: int = 0
    npo: int | None = None
    conv: int = 0
    R: int = 0
    Lg: int = 0
    zero: int = 0
    rc: int = 0
    r0: int = 0
    rstep: int = 1
    il: bool = False
    par0: bool = False

    def __post_init__(self) -> None:
        assert self.m in (64, 16) and self.N % 320 == 0 and self.K % 8 == 0 and \
            self.out in ("bf16", "bfp16", "geglu", "vt", "ds")
        assert self.out != "ds" or self.rc and self.rstep == 2 and self.stats is None and self.res is None
        self.npass = self.N // 320
        self.npo = self.npo or self.npass - self.p0
        assert 0 <= self.p0 and self.p0 + self.npo <= self.npass
        self.nkt = -(-self.K // D.KT)
        if self.conv:
            assert self.m * (1 + self.il) == 16 * self.R * self.Lg and (self.ntile == 1 or self.rc) and self.K % 64 == 0 \
                and self.conv in (1, 2)
            if self.il:  # (the tiles' input rows in x)
                assert (self.R, self.Lg, self.conv) == (1, 8, 1) and (self.par0 or self.ntile % 2 == 0) and self.r0 >= 1 \
                    and self.r0 + self.rstep * ((self.ntile if self.par0 else self.ntile // 2) - 1) + 1 < self.rc
            else:
                assert not self.rc or (self.R, self.Lg, self.conv) == (1, 4, 1) and self.r0 + self.rstep * (self.ntile - 1) < self.rc
            self.nch = self.K // 64
            self.Li, self.Rc = self.Lg * self.conv, self.R * self.conv  # (the input's groups, the column's input rows)
            self.Ri = self.Rc + (2 if self.conv == 1 else 1)  # (an object's rows: the halos above / below)
            self.nkt = 9 * self.nch
        self.np = self.m * self.ntile // 8  # (the column's token panels)
        self.x_np = self.x_np or self.np
        self.junk = 2 if self.out == "geglu" and self.m == 16 else 0  # (m 16 GEGLU: 720 B per pass, a padded chunk)
        self.y_np = self.y_np or self.np + self.junk
        self.nres = -(-self.m * 160 // D.A_OBJ) if self.res is not None else 0
        self.aobj = self.nch if self.conv else -(-self.nkt * self.m // 128)  # (A objects of a pass's k-tiles)
        assert self.npass <= 16 and self.y_np >= self.np + self.junk
        ybytes = {"bfp16": self.N // 8, "vt": self.N // 8, "geglu": self.N // 16}.get(self.out, 0) * self.y_np * L.PANEL or \
            self.N // 80 * (1 if self.out == "ds" else self.ntile) * L.ct_bytes(self.m)
        assert self.y + ybytes <= D.HEAP and self.x + self.K // 8 * self.x_np * L.PANEL <= D.HEAP, "tensors past the heap"
        assert self.conv or self.x + self.aobj * (128 // self.m) * 8 * self.x_np * L.PANEL <= 512 * 1024, \
            "the A objects' walk past the memtile (an over-read hangs)"
        assert self.res is None or self.res + (self.N // 80 * self.ntile - 1) * L.ct_bytes(self.m) + self.nres * D.A_OBJ <= \
            512 * 1024, "the residual's last objects past the memtile"
        assert self.stats != "gn" or self.gsize, "a GroupNorm's group size"

    def ch0(self, row: int) -> int:
        """Row ``row``'s first output channel in the statistics' channel space."""
        return self.ch_off + row * self.npass * 80

    # (the 4-band convs' taps: those of object rows 0, 1 (region 2) and rows 2, 3 (region 3) alternating with the B
    # slots' regions, in odd chunks rotated by one (9 B slots a chunk); before index 1, 5, 7 the shift phases)
    ORDER2 = ((1, 1), (2, 2), (0, 0), (2, 1), (0, 1), (2, 0), (0, 2), (1, 0), (1, 2))

    def tap_order(self, c: int) -> list[tuple[int, int]]:
        """The taps (dy, dx) of chunk c in the order of its B slots."""
        if self.il:
            return [(dy, dx) for dy in range(3) for dx in (1, 2, 0)]
        if self.R == 1:
            return [(dy, dx) for dy in range(3) for dx in range(3)]
        return list(self.ORDER2[c % 2:]) + list(self.ORDER2[:c % 2])

    def conv_words(self) -> np.ndarray:
        """The conv's command words from ``core.C_CALLS`` (``core``'s v2_conv): per tap (dy, dx) of the order its MAC
        calls (the bands (g, r), g-major, each group's source: object row r conv + dy's group conv g + dx - 1 (its
        byte offset: ``design.a_byte``), or a lane-shifted copy in cbuf (the phase's: a row's last group one lane down,
        its group 0 one lane up); a call per group where the two sources' row strides differ), the B and C walks, the
        taps' words, the shift phases."""
        if self.il:
            return self._il_words()
        if self.rc:
            return self._row_words()
        R, LG, Li, st = self.R, self.Lg, self.Li, self.conv
        assert (R, LG) in ((1, 1), (2, 2)), (R, LG)
        rs, il = Li * 1152, R == 2  # (il: 4 bands, per group its rows alternate per block: C's halves in two regions)

        def row(k: int, g: int = 0) -> int:  # (object row k's group g)
            return D.a_byte(k * rs) + g * 1152
        if il:  # (per phase: (sd, su) object rows per output row: cbuf slots [r][down, up])
            assert self.nch % 2 == 0 and st == 1  # (every pass starts at the same B parity)
            phases, at = [((0, 2), (1, 3)), ((2, 0), (3, 1)), ((1, 1), (2, 2))], {1: 0, 5: 1, 7: 2}
        else:
            phases, at = [((dy, dy),) for dy in range(3)], {0: 0, 3: 1, 6: 2}
        pw = np.zeros((3, 2, 4), np.int64)
        for p, rows in enumerate(phases):
            for r, (kd, ku) in enumerate(rows):
                pw[p, r] = (row(kd, Li - 1), 2304 * r, row(ku), 2304 * r + 1152)
        cbuf, slots = D.L1["tab"], [D.L1["r23"] + D.R23[n] for n in ("a0", "a1")]
        calls, tapw, ph = [], [], None
        for i, (dy, dx) in enumerate(self.tap_order(0)):
            if i in at:
                ph = at[i]
            prov = {}  # ((down?, object row) -> cbuf offset) of the phase
            if ph is not None:
                for r, (kd, ku) in enumerate(phases[ph]):
                    prov[(True, kd)], prov[(False, ku)] = 2304 * r, 2304 * r + 1152
            src = []  # (per group: cbuf?, its offset, its row stride)
            for g in range(LG):
                x = st * g + dx - 1
                if 0 <= x < Li:
                    src.append((False, row(dy, x), row(dy + 1, x) - row(dy, x) if il else 0))
                else:
                    o = [prov[(x < 0, dy + r)] for r in range(R)]
                    src.append((True, o[0], o[-1] - o[0]))
            first = len(calls)
            if il and src[0][2] != src[1][2]:
                for g in range(2):
                    sg = src[g][2]
                    calls.append((src[g:g + 1], g, 2, (1, sg - 1152, 4, -sg - 1152, 0)))
            elif il:
                s0 = src[0][2]
                calls.append((src, 0, 4, (1, s0 - 1152, 4, -s0 - 1152, src[1][1] - src[0][1] - s0 - 1152)))
            else:
                calls.append((src, 0, 1, (4, -1152, 0, 0, -1152)))
            tapw.append(first * K.CALL | (len(calls) - first) << 10 | (at[i] + 1 if i in at else 0) << 12)
        words = []
        for k, (src, g, nbr, dims) in enumerate(calls):  # (the walk's last step per A slot: cbuf <-> the object)
            c0, c1 = src[0][0], src[-1][0]
            last = [dims[4] + (cbuf - a if c1 and not c0 else a - cbuf if c0 and not c1 else 0) for a in slots]
            assert k in K.CROSS or last[0] == last[1]
            words += [(1 << 31 if c0 else 0) | (1 << 30 if g else 0) | nbr << 16 | src[0][1], *dims, 0, 0, *last]
        assert len(calls) <= K.NCALL
        pb = (1, -1152, 4, 0, -5760) if il else (4, 0, 0, -5760, -5760)
        pl = (1, 2560, 4, -2432, -1792) if il else (4, 128, 0, 768, 768)
        out = np.zeros(K.C_PHASE + 24 - K.C_CALLS, np.int64)
        out[:len(words)] = words
        out[K.C_PATS - K.C_CALLS:K.C_TAPW - K.C_CALLS] = [*pb, 0, 0, *pl, 0, 0]
        out[K.C_TAPW - K.C_CALLS:K.C_PHASE - K.C_CALLS] = [w | w << 16 for w in tapw]  # (odd tiles: the same)
        out[K.C_PHASE - K.C_CALLS:] = pw.reshape(-1)
        assert K.C_PHASE + 24 <= D.WS // 4
        return (out & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

    def _row_words(self) -> np.ndarray:
        """``conv_words`` of the row tiles (``rc``): 4 MAC calls relative to an object, the same for its 3 taps (dx): bands
        b = groups 0..3 in slots dx + b (slot s at ``design.a_byte`` (1152 s): slots 4.. in the second half); one call of
        the 4 bands where the pairs (b0, b2), (b1, b3) are equally far apart (the il walk), else a call per pair; the
        shift phase (slot 4 one lane down into slot 0, slot 1 up into slot 5) and a new A object before each dy's taps."""
        A = [D.a_byte(1152 * k) for k in range(6)]

        def walk(bands: list[int]) -> tuple:  # (the A walk of 4 bands or of one pair)
            a = [A[k] for k in bands]
            if len(a) == 2:
                return 2, (1, a[1] - a[0] - 1152, 4, a[0] - a[1] - 1152, 0)
            assert a[2] - a[0] == a[3] - a[1]
            return 4, (1, a[2] - a[0] - 1152, 4, a[0] - a[2] - 1152, a[1] - a[2] - 1152)
        calls, first = [], []
        for dx in range(3):
            first.append(len(calls))
            sl = [dx + b for b in range(4)]
            if A[sl[2]] - A[sl[0]] == A[sl[3]] - A[sl[1]]:
                calls.append((sl[0], 0, *walk(sl)))
            else:
                calls += [(sl[0], 0, *walk([sl[0], sl[2]])), (sl[1], 1, *walk([sl[1], sl[3]]))]
        first.append(len(calls))
        words = []
        for src, g, nbr, dims in calls:
            words += [(1 << 30 if g else 0) | nbr << 16 | A[src], *dims, 0, 0, dims[4], dims[4]]
        tapw = [first[dx] * K.CALL | (first[dx + 1] - first[dx]) << 10 | (1 if dx == 0 else 0) << 12 |
                (1 if dx == 0 and dy else 0) << 15 for dy, dx in self.tap_order(0)]
        tapw = [w | w << 16 for w in tapw]  # (odd tiles: the same)
        out = np.zeros(K.C_PHASE + 24 - K.C_CALLS, np.int64)
        out[:len(words)] = words
        out[K.C_PATS - K.C_CALLS:K.C_TAPW - K.C_CALLS] = [1, -1152, 4, 0, -5760, 0, 0, 1, 2560, 4, -2432, -1792, 0, 0]
        out[K.C_TAPW - K.C_CALLS:K.C_PHASE - K.C_CALLS] = tapw
        out[K.C_PHASE - K.C_CALLS:K.C_PHASE - K.C_CALLS + 4] = (A[4], A[0] | K.SH_OBJ, A[1], A[5] | K.SH_OBJ)
        return (out & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

    def _il_words(self) -> np.ndarray:
        """``conv_words`` of the 64x64 row tiles (``il``): the object row r + dy - 1's groups [g0 g2 g4 g6 | g1 g3 g5 g7]
        (slots A0..A7, ``design.a_byte`` (1152 k)), its taps in the order dx 0, +1, -1 (``tap_order``). Even tiles:
        [A0..A3], [A4..A7], then g7 one lane down into slot A0 (both read), dx -1 from [A0 A4 A5 A6] (the pairs (A0, A5),
        (A4, A6): a call each); odd tiles: [A4..A7], g0 one lane up into A4, dx +1 from [A1 A2 A3 A4] (the pairs (A1, A3),
        (A2, A4)), [A0..A3]. Each shift phase's other half lands in cbuf (unused)."""
        A = [D.a_byte(1152 * k) for k in range(8)]

        def il4(k: int) -> list[int]:  # (4 slots from k: the il walk)
            a = A[k:k + 4]
            return [4 << 16 | a[0], 1, a[2] - a[0] - 1152, 4, a[0] - a[2] - 1152, a[1] - a[2] - 1152, 0, 0,
                    a[1] - a[2] - 1152, a[1] - a[2] - 1152]

        def pair(g: int, x: int, y: int) -> list[int]:
            return [(1 << 30 if g else 0) | 2 << 16 | x, 1, y - x - 1152, 4, x - y - 1152, 0, 0, 0, 0, 0]
        words = il4(0) + il4(4) + pair(0, A[0], A[5]) + pair(1, A[4], A[6]) + pair(0, A[1], A[3]) + pair(1, A[2], A[4])
        use = ({1: (0, 1), 2: (1, 1), 0: (2, 2)}, {1: (1, 1), 2: (4, 2), 0: (0, 1)})  # (per parity, dx: first call, calls)
        phase = ({0: 1}, {2: 2})  # (per parity: its shift before that tap)
        tapw = []
        for dy, dx in self.tap_order(0):
            w = [use[p][dx][0] * K.CALL | use[p][dx][1] << 10 | phase[p].get(dx, 0) << 12 | (dx == 1 and dy > 0) << 15
                 for p in range(2)]
            tapw.append(w[0] | w[0 if self.par0 else 1] << 16)
        out = np.zeros(K.C_PHASE + 24 - K.C_CALLS, np.int64)
        out[:len(words)] = words
        out[K.C_PATS - K.C_CALLS:K.C_TAPW - K.C_CALLS] = [1, -1152, 4, 0, -5760, 0, 0, 1, 2560, 4, -2432, -1792, 0, 0]
        out[K.C_TAPW - K.C_CALLS:K.C_PHASE - K.C_CALLS] = tapw
        out[K.C_PHASE - K.C_CALLS:K.C_PHASE - K.C_CALLS + 16] = [A[7], A[0] | K.SH_OBJ, A[0], 0, 0, 0, 0, 0,
                                                                 A[7], 1152, A[0], A[4] | K.SH_OBJ, 0, 0, 0, 0]
        return (out & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

    def command(self, bias: torch.Tensor | None, row: int) -> np.ndarray:
        """Row ``row``'s command slot (Q bytes); bias: (N,) or None (a conv's: its bias + the time bias)."""
        cmd = np.zeros(D.Q // 4, np.int32)
        om = {"bf16": K.O_BF16, "bfp16": K.O_BFP16, "geglu": K.O_GEGLU, "vt": K.O_VT, "ds": K.O_DS}[self.out]
        st = {None: K.S_NONE, "gn": K.S_GN, "ln": K.S_LN}[self.stats]
        cmd[[K.C_KIND, K.C_M, K.C_NTILE, K.C_NPASS, K.C_NKT, K.C_NRES, K.C_OUT, K.C_STATS, K.C_STF]] = (
            K.K_GEMM, self.m, self.ntile, self.npo, self.nkt, self.nres, om, st, self.stf)
        if self.stats == "gn":
            c0 = self.ch0(row)
            cmd[[K.C_GSIZE, K.C_CH0, K.C_G0, K.C_GINV]] = (self.gsize, c0 + 80 * self.p0,
                                                           gn_window(c0, self.npass * 80, self.gsize), K.group_inv(self.gsize))
        if not self.conv:
            assert K.C_BIAS * 4 + self.npo * 160 <= K.C_WALKS * 4
            cmd[K.C_WALKS:K.C_PATS + 14] = mac_walks(self.m)
        if self.conv:
            cmd[[K.C_CONV, K.C_NCH, K.C_R, K.C_ROT]] = self.conv, self.nch, self.R, self.R == 2
            cmd[K.C_CALLS:K.C_PHASE + 24] = self.conv_words()
            assert K.C_BIAS * 4 + self.npo * 160 <= K.C_MASKS * 4
            cmd.view(np.uint8)[4 * K.C_MASKS:4 * K.C_MASKS + 1152] = np.frombuffer(K.shift_masks(), np.uint8)
        out = cmd.view(np.uint8).copy()
        if bias is not None:
            b = self.bias_bytes(bias, row)
            out[4 * K.C_BIAS:4 * K.C_BIAS + b.size] = b
        return out

    def bias_bytes(self, bias: torch.Tensor, row: int) -> np.ndarray:
        """The command's biases (at word C_BIAS): row ``row``'s of its passes, bf16."""
        return bias[self._rows(row)].float().numpy().astype(bfloat16).view(np.uint8)

    def _rows(self, row: int) -> torch.Tensor:
        """Row ``row``'s weight rows of the op's passes in order (GEGLU: per pass [hid 40 | gate 40])."""
        n0 = row * self.npass * 80
        ps = range(self.p0, self.p0 + self.npo)
        if self.out != "geglu":
            return torch.tensor([n0 + 80 * p + i for p in ps for i in range(80)])
        h, g = self.N // 2, n0 // 2
        return torch.tensor([g + 40 * p + i if i < 40 else h + g + 40 * p + i - 40 for p in ps for i in range(80)])

    def weights(self, w: torch.Tensor, row: int) -> np.ndarray:
        """Row ``row``'s weight image (a conv's k-tiles: per chunk of 64 input channels its taps, ``tap_order``)."""
        ws = w[self._rows(row)]
        if self.conv:
            t = ws.reshape(-1, self.nch, 64, 3, 3).permute(0, 1, 3, 4, 2)  # (n, chunk, dy, dx, k)
            ws = torch.stack([torch.stack([t[:, c, dy, dx] for dy, dx in self.tap_order(c)], 1) for c in range(self.nch)],
                             1).reshape(ws.shape[0], -1)
        return L.gemm_weights(ws, 0, self.npo)

    def conv_zero_bytes(self) -> int:
        """Bytes of the zeros an edge column's missing halo row reads (one row of 8 kb: the iteration stays on it)."""
        return self.Li * 8 * 2 * L.PANEL

    def _row_sends(self, col: int, t: int) -> MemTask:
        """Tile t's A objects of the row tiles (``rc``), per chunk 3 (dy): the pairs (row, group) from input row
        r + dy - 1's predecessor ([pair 8][kb 8][2], pair q at q 144 B of a kb's panels: q -1 is before x); the row above
        the column's first from the west column (the image's top: zeros), below its last from the east (the bottom)."""
        kbw, r = self.x_np * 18, self.r0 + self.rstep * t
        obj = [(36, 1), (8, kbw), (8, 36), (0, 144)]
        it = (8 * kbw, self.nch)

        def own(dy0: int, n: int, base: int = txn.MEM_OWN, row0: int | None = None) -> dict:
            q = 4 * (r + dy0 - 1 if row0 is None else row0) - 1
            return {"addr": self.x + 144 * q, "nwords": n * D.A_OBJ // 4, "dims": obj, "base": base, "iteration": it}
        zero = {"addr": self.zero, "nwords": D.A_OBJ // 4, "dims": [(288, 1), (8, 1)]}
        lo, hi = (1 if r == 0 else 0), (2 if r == self.rc - 1 else 3)
        chain = []
        if r == 0:
            chain.append(zero if col == 0 else own(0, 1, txn.MEM_WEST, self.rc - 1))
        chain.append(own(lo, hi - lo))
        if r == self.rc - 1:
            chain.append(zero if col == COLS - 1 else own(2, 1, txn.MEM_EAST, 0))
        assert self.x >= 144, "the first object's slot 0 before x"
        return MemTask(chain, repeat=self.nch * (1 if self.res is not None else self.npo))

    def _il_sends(self, t: int) -> MemTask:
        """Tile t's A objects of the 64x64 row tiles (``il``), per chunk 3 (dy): input rows r - 1 .. r + 1 of x, each its
        groups [group 8][kb 8][2] (one BD: the rows are consecutive)."""
        kbw, r = self.x_np * 18, self.r0 + self.rstep * (t if self.par0 else t // 2)
        return MemTask([{"addr": self.x + 1152 * (r - 1), "nwords": 3 * D.A_OBJ // 4,
                         "dims": [(36, 1), (8, kbw), (8, 36), (0, 288)], "iteration": (8 * kbw, self.nch)}],
                       repeat=self.nch * (1 if self.res is not None else self.npo))

    def _conv_sends(self, col: int) -> MemTask:
        """The chunks' A objects: [rows Ri][Li groups][kb 8][2] from the west column's last row (the image's top: zeros),
        the column's rows, the east column's first row (stride 1; the bottom: zeros); the last BD reads on to fill
        A_OBJ."""
        kbw, gw = self.x_np * 18, self.Li * 36  # (words: a kb's panels, a row's groups)
        row = [(36, 1), (8, kbw), (self.Li, 36)]
        it = (8 * kbw, self.nch)
        last = self.x + ((self.Rc - 1) * self.Li * 2) * L.PANEL
        chain = [{"addr": self.zero, "nwords": 8 * gw} if col == 0 else
                 {"addr": last, "nwords": 8 * gw, "dims": row, "base": txn.MEM_WEST, "iteration": it},
                 {"addr": self.x, "nwords": self.Rc * 8 * gw, "dims": row + [(0, gw)], "iteration": it}]
        if self.conv == 1:
            chain.append({"addr": self.zero, "nwords": 8 * gw} if col == COLS - 1 else
                         {"addr": self.x, "nwords": 8 * gw, "dims": row, "base": txn.MEM_EAST, "iteration": it})
        pad = D.A_OBJ // 4 - self.Ri * 8 * gw
        assert pad >= 0, "an object past A_OBJ"
        chain[-1]["nwords"] += pad
        return MemTask(chain, repeat=self.nch * (1 if self.res is not None else self.npo))

    def col_ops(self, col: int = 0) -> ColOps:
        """Per core its sends and receives. Tiles: the receives one task whose BDs step over the tiles; the sends one
        task (execution t NPO + p: with one pass per core or one tile, its BDs step linearly), else a task per tile."""
        m, ct, NT = self.m, L.ct_bytes(self.m), self.ntile
        P0, NPO = self.p0, self.npo
        XN, YN, TP = self.x_np, self.y_np, m // 8 * 18  # (TP: a tile's token panels in words)
        co = ColOps()
        for r in range(4):
            sends, recvs = [], []
            if self.conv:  # (per tile; with a residual per pass its chunks, then its residual objects)
                for t in range(NT):
                    cs = self._il_sends(t) if self.il else self._row_sends(col, t) if self.rc else self._conv_sends(col)
                    if self.res is None:
                        sends.append(cs)
                        continue
                    ow = self.nres * D.A_OBJ // 4
                    res = MemTask([{"addr": self.res + ((r * self.npass + P0) * NT + t) * ct, "nwords": ow,
                                    "dims": _contig(ow), "iteration": (NT * ct // 4, NPO)}])
                    for _ in range(NPO):
                        sends += [cs, res]
            else:
                def chain(t: int, it: bool, r: int = r) -> list[dict]:  # (tile t's; it: the BDs step over the tiles)
                    out = []
                    if self.aobj:
                        out.append({"addr": self.x + t * TP * 4, "nwords": self.aobj * D.A_OBJ // 4,
                                    "dims": [(36, 1), (8, XN * 18), (m // 16, 36), (0, 8 * XN * 18)],
                                    **({"iteration": (TP, NT)} if it else {})})
                    if self.res is not None:  # (execution i = t NPO + p: the object (r npass + P0 + p, t))
                        ow = self.nres * D.A_OBJ // 4
                        out.append({"addr": self.res + ((r * self.npass + P0) * NT + t) * ct, "nwords": ow,
                                    "dims": _contig(ow), "iteration": (ct // 4, NT * NPO) if it else (NT * ct // 4, NPO)})
                    return out
                if NT == 1 or NPO == 1:
                    sends.append(MemTask(chain(0, True), repeat=NT * NPO))
                else:
                    sends += [MemTask(chain(t, False), repeat=NPO) for t in range(NT)]
            if self.out in ("bfp16", "vt"):  # (pass p: [rb][cb] -> kb (r npass + p) 10 + cb, token panel (m / 8) t + rb)
                data = _panel_recv(self.y + (r * self.npass + P0) * 10 * YN * L.PANEL, YN, m // 8, 10, NPO)
                data["iteration"] = (TP, NT)
            elif self.out == "geglu":  # (pass p: [rb][5 cb] -> kb (r npass + p) 5 + cb)
                data = _panel_recv(self.y + (r * self.npass + P0) * 5 * YN * L.PANEL, YN, m // 8, 5, NPO, self.junk)
                data["iteration"] = (TP, NT)
            elif self.out == "ds":  # (tile t's pass p: groups 0 and 2 (2 x 2560 B, each in two chunks) -> the 16x16
                # C-tile object (r npass + p), its row t: the second call's data over the first's chunk padding)
                data = {"addr": self.y + (r * self.npass + P0) * ct, "nwords": NT * NPO * 1440,
                        "dims": [(720, 1), (2, 640), (NPO, ct // 4), (0, 1280)]}
            else:  # (pass p: the C-tile object (r npass + p, t))
                data = {"addr": self.y + (r * self.npass + P0) * NT * ct, "nwords": NPO * ct // 4,
                        "dims": _contig(ct // 4) + [(NPO, NT * ct // 4)], "iteration": (ct // 4, NT)}
            rc = [data]
            if self.stats == "ln":
                rc.append({"addr": self.rec + r * D.OCH, "nwords": D.OCH // 4, "iteration": (D.OCH, NT)})
            recvs.append(MemTask(rc, repeat=1 if self.out == "ds" else NT))
            if self.stats == "gn" and not self.stf & K.ST_HOLD:
                recvs.append(MemTask([{"addr": self.rec + r * D.OCH, "nwords": D.OCH // 4}]))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, img: int) -> RowOps:
        """img: the device address of the row's image: its command slot, then its weights (one run; the weights
        again per further tile)."""
        wb = self.npo * self.nkt * D.Q
        return RowOps(runs=[(img, D.Q + wb)] + [(img + D.Q, wb)] * (self.ntile - 1 if wb else 0),
                      b_objects=1 + self.ntile * self.npo * self.nkt, parity=self.m == 64 and self.nkt > 0)


@dataclass
class Apply:
    """mode "gn" / "gnsilu" / "ln" of the bf16 C-tile tensor x (``ntile`` tiles of m tokens; blocks of 80 channels:
    ``nblk`` per core of ``rows`` from block ``blk0``) into the panel tensor y (``y_np`` token panels per kb; its kb of the
    i-th core's first channel: ``y_kb0`` + 10 i nblk). GroupNorm: groups of ``gsize`` channels, the i-th core's first
    channel ``ch_off`` + 80 i nblk in their space (its groups in one record window: ``gn_window``); the records come on
    the B stream (``row_ops(records=)``). LayerNorm: the producer's records at ``rec`` ([tile][core][OCH]); ``C``: the
    norm's channels (1 / C)."""

    m: int
    nblk: int
    mode: str
    x: int
    y: int
    ntile: int = 1
    blk0: int = 0
    y_kb0: int = 0
    y_np: int | None = None
    gsize: int = 0
    ch_off: int = 0
    rec: int = 0
    tokens: int = 0  # (GroupNorm: tokens of an image)
    C: int = 0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        assert self.mode in ("gn", "gnsilu", "ln") and self.m in (64, 16) and (self.m == 64 or self.ntile == 1)
        self.np = self.m * self.ntile // 8
        self.y_np = self.y_np or self.np
        self.gn = self.mode in ("gn", "gnsilu")
        self.rows = D.APPLY_ROWS
        ct = L.ct_bytes(self.m)  # (an object's row blocks, then its padding: m 64 a row block, m 16 80 words)
        assert ct - self.m * 160 == (1280 if self.m == 64 else 320)
        # (A halves: m 64 3 slots each (a tile's 8 row blocks and its padding in 3 halves), m 16 a tile per half)
        self.nhalf = self.nblk * self.ntile * (3 if self.m == 64 else 1)
        self.half_step = 960 if self.m == 64 else ct // 4
        self.aobj = -(-self.nhalf // 2) + (self.mode == "ln")
        assert self.nblk * 80 <= 640 and (not self.gn or (self.gsize >= 8 and self.tokens)) and (self.mode != "ln" or self.C)
        assert self.mode != "ln" or self.ntile <= 4, "LayerNorm records past their A object"
        last = self.x + (self.blk0 + (len(self.rows) - 1) * self.nblk) * self.ntile * ct + 4 * (self.nhalf - 1) * self.half_step
        assert last + D.A_OBJ // 2 <= 512 * 1024, "x's walk past the memtile"

    def ch0(self, row: int) -> int:
        """Row ``row``'s first channel in the GroupNorm's channel space."""
        return self.ch_off + self.rows.index(row) * self.nblk * 80

    def command(self, row: int, nrec: int = 0) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        mode = {"gn": K.A_GN, "gnsilu": K.A_GNSILU, "ln": K.A_LN}[self.mode]
        cmd[[K.C_KIND, K.C_M, K.C_NTILE, K.C_NBLK, K.C_MODE, K.C_NREC, K.C_EPS]] = (K.K_APPLY, self.m, self.ntile, self.nblk,
                                                                                mode, nrec, _f32(self.eps))
        if self.gn:
            ch0, n = self.ch0(row), self.nblk * 80
            g0 = gn_window(ch0, n, self.gsize)
            cmd[[K.C_GSIZE, K.C_CH0, K.C_G0, K.C_INVN, K.C_GINV]] = (
                self.gsize, ch0, g0, _f32(1.0 / (self.gsize * self.tokens)), K.group_inv(self.gsize))
            cmd[K.C_GTAB:K.C_GTAB + n // 8] = K.group_words(ch0, n, self.gsize, g0)
        elif self.mode == "ln":
            cmd[K.C_INVN] = _f32(1.0 / self.C)
        return cmd.view(np.uint8)

    def gamma_beta(self, gamma: torch.Tensor | None, beta: torch.Tensor | None, row: int) -> np.ndarray:
        """Row ``row``'s B slot: gamma, beta (fp32) of its channels (gamma, beta: of x's channels from block 0):
        GroupNorm [c16][gamma 16 | beta 16], else [gamma | beta]."""
        out = np.zeros(D.Q // 4, np.float32)
        n = self.nblk * 80
        if gamma is not None:
            c0 = (self.blk0 + self.rows.index(row) * self.nblk) * 80
            g, b = gamma[c0:c0 + n].float().numpy(), beta[c0:c0 + n].float().numpy()
            if self.gn:
                out[:2 * n] = np.stack([g.reshape(-1, 16), b.reshape(-1, 16)], 1).reshape(-1)
            else:
                out[:n], out[n:2 * n] = g, b
        return out.view(np.uint8)

    def col_ops(self) -> ColOps:
        ct = L.ct_bytes(self.m)
        co = ColOps()
        for i, r in enumerate(self.rows):
            chain = []
            if self.mode == "ln":  # (the tiles' records of the column's 4 cores: [tile][core][S1 m | S2 m], 576 B apart:
                # a tile's 2304 B never straddle the object's halves)
                chain.append({"addr": self.rec, "nwords": D.A_OBJ // 4, "dims": [(144, 1), (4, D.OCH // 4),
                                                                                 (self.ntile, D.OCH)]})
            blk = self.blk0 + i * self.nblk  # (its blocks' objects [block][tile], contiguous: per A half 3 slots (m 64:
            # a tile's 8 row blocks and its padding in 3 halves; m 16: a tile's 2 and its padding), then whatever follows)
            chain.append({"addr": self.x + blk * self.ntile * ct, "nwords": (self.aobj - (self.mode == "ln")) * D.A_OBJ // 4,
                          "dims": [(D.A_OBJ // 16, 1), (2, D.A_OBJ // 16), (self.nhalf, self.half_step)]})
            co.sends[r] = [MemTask(chain)]
            YN = self.y_np
            co.recvs[r] = [MemTask([_panel_recv(self.y + (self.y_kb0 + 10 * i * self.nblk) * YN * L.PANEL, YN, self.np, 10,
                                                self.nblk)])]
        return co

    def row_ops(self, img: int, records: list[tuple] = (), nslots: int = 0) -> RowOps:
        """img: the row's command and gamma / beta slots (one run); records (GroupNorm): the runs of its ``nslots``
        record slots after them (``core.REC_SLOT`` records each, the last one's rest anything). (Only the op's ``rows``
        have a B stream.)"""
        runs = [(img, 2 * D.Q)] + list(records)
        return RowOps(runs=runs, b_objects=2 + nslots, parity=False, role="b")


@dataclass
class Attention:
    """Attention of 8 heads of d 160 (16x16, 8x8) or 80 (32x32) over each image's keys, for the queries of tile ``tile``
    (4 query panels of each image, 8 token panels; 8x8: one panel of each), or of d 40 (64x64, role c) for all of the
    op's (16 query panels of each image, one op): row i of ``rows`` (in role a / c) runs the units (image, head) of heads
    [h i, h i + h), h = 8 / len(rows), image-major. q: the op's query panels ([kb 8 d][np], scaled by log2 e / sqrt d),
    o: the output panels (the same layout); kv: the device address of the K / V^T slots ([image][head][group of
    keys][K | V^T]: d 160 groups of 32 keys, d 80 of 64), written by ``kv_drains``. Self-attention (keys = the image's
    tokens, of ``nbands`` ops of np token panels: band b's column c holds keys [(8 b + c) kpi, ..) of its image): an
    image's token panels in a column are 2 apart. Cross-attention: nkeys (the context's: ``layout.kv_slots``, the last
    group's keys from nkeys masked). d 40: a key group's K and V^T in one slot."""

    np: int
    q: int
    o: int
    kv: int
    nkeys: int = 0
    d: int = 160
    nbands: int = 1
    tile: int = 0
    rows: tuple[int, ...] = D.ATTN_ROWS

    def __post_init__(self) -> None:
        assert self.d in (40, 80, 160) and len(self.rows) in (2, 4)
        self.hpr = 8 // len(self.rows)  # (heads per row)
        self.nd = self.d // 8  # (d-blocks of a head)
        self.gk = 4 if self.d == 160 else 8  # (key panels of a group)
        self.kpi = self.np // 2  # (an image's token panels in the op)
        self.kpc = 16 if self.d == 40 else min(self.kpi, 4)  # (query panels of a unit: an op per tile keeps a core's sends
        # and receives to two BDs each)
        self.role = "c" if self.d == 40 else "a"
        self.slots = 1 if self.d == 40 else 2  # (B slots of a key group)
        assert 8 * self.tile < self.np and (self.d != 40 or (self.kpi, self.tile) == (16, 0))
        keys = 8 * COLS * self.kpi * self.nbands
        self.ng = -(-(self.nkeys or keys) // (8 * self.gk))  # (key groups of an image)
        self.nk = self.nkeys - 8 * self.gk * (self.ng - 1) if self.nkeys else 8 * self.gk
        assert self.kpc in (1, 4, 16) and (self.d == 160 or self.kpc >= 4), self.np
        assert self.q + ((7 * self.nd + (8 if self.d == 40 else 32)) * self.np + 8) * L.PANEL <= D.HEAP, \
            "the last head's walk past the heap"

    def command(self) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        cmd[[K.C_KIND, K.C_NUNIT, K.C_NQP, K.C_NG, K.C_NK, K.C_ND, K.C_NKP]] = (K.K_ATTN, 2 * self.hpr, self.kpc, self.ng, self.nk,
                                                                             self.nd, self.gk)
        if self.d == 80:
            cmd[K.C_WALKS:K.C_PATS + 14] = mac_walks(16)  # (P V^T of d 80: one band)
        return cmd.view(np.uint8)

    def col_ops(self) -> ColOps:
        """Per attention row and image its 4 units' queries (per head an A object: query pairs [pair][d-block][2], a single
        panel paired with the next one; a pair per half: 32 d-blocks of the walk, the head's first), their outputs
        ([qp][d-block])."""
        co = ColOps()
        hstep, kb = self.nd * self.np * 18, self.np * 18  # (words: a head's kb, one kb)
        h = self.hpr
        nkb = 8 if self.d == 40 else 32  # (the walk's d-blocks per pair: d 40 [pair 8][d-block 8 (5 used)][2])
        for i, r in enumerate(self.rows):
            sends, recvs = [], []
            for img in range(2):
                p0 = (h * self.nd * i * self.np + 8 * self.tile + img) * L.PANEL
                sends.append(MemTask([{"addr": self.q + p0, "nwords": D.A_OBJ // 4,
                                       "dims": _dims((18, 1), (2, 36), (nkb, kb), (max(self.kpc // 2, 1), 72)),
                                       "iteration": (hstep, h)}], repeat=h))
                recvs.append(MemTask([{"addr": self.o + p0, "nwords": self.kpc * self.nd * 18,
                                       "dims": _dims((18, 1), (self.nd, kb), (self.kpc, 36)), "iteration": (hstep, h)}],
                                     repeat=h))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        """The attention rows' B streams: the command, then per image its units' K / V^T slots (contiguous)."""
        unit, h = self.ng * self.slots * D.Q, self.hpr
        return {r: RowOps(runs=[(cmd, D.Q)] + [(self.kv + (8 * img + h * i) * unit, h * unit) for img in range(2)],
                          b_objects=1 + 2 * h * self.slots * self.ng, parity=self.kpc == 4, role=self.role)
                for i, r in enumerate(self.rows)}

    def kv_bytes(self) -> int:
        return 16 * self.ng * self.slots * D.Q

    def kv_drains(self, k: int, vt: int, col: int, *, part: str = "kv", band: int = 0) -> ColOps:
        """Column ``col``'s drains of its keys' part of every slot (band ``band``'s: key panels kp0 .. of group g): from the
        K panels at ``k`` ([kb][np]) the K slot rows [key block][d-block][2], from the V^T panels at ``vt`` the V^T slot
        [d-block pair][key panel][2]; per image a task of the 8 heads. part: "k" / "v": only K's / V^T's."""
        g, kp0 = divmod((band * COLS + col) * self.kpi, self.gk)
        hstep, kb = self.nd * self.np * 18, self.np * 18
        unit = self.ng * self.slots * D.Q  # (a head's slots in DDR)
        co = ColOps()
        for img in range(2):
            slot = self.kv + 8 * img * unit + g * self.slots * D.Q
            if self.d == 40:  # (the column's keys fill two groups: per head their K, their V^T parts of d-blocks 0..3
                # ([d-block pair][key panel][2]) and of d-block 4 ([key panel])
                assert self.kpi == 2 * self.gk
                kw = _dims((18, 1), (2, 36), (5, kb), (8, 72))
                kd = (slot, D.Q, (720, 2, 1440, 2880), (unit // 4, 8))
                vws = [(0, _dims((18, 1), (2, kb), (16, 36), (2, 2 * kb)), (slot + D.Q // 2, 4 * 1152, (288, 2, 1440, 288),
                                                                           (unit // 4, 8))),
                       (4 * kb, _dims((18, 1), (16, 36)), (slot + D.Q // 2 + 2304, 2 * 576, (144, 2, 1440, 144),
                                                          (unit // 4, 8)))]
            elif self.kpi == self.gk:  # (the column's keys fill the group: whole slots)
                kw, kd = _dims((18, 1), (2, 36), (self.nd, kb), (self.gk // 2, 72)), (slot, D.Q, None, (unit // 4, 8))
                vw = _dims((18, 1), (2, kb), (self.gk, 36), (self.nd // 2, 2 * kb))
                vd = (slot + D.Q, D.Q, None, (unit // 4, 8))
            else:  # (d 160, one key panel: every other panel of the K slot's pair rows, a pair of every V^T row of 4)
                assert self.d == 160 and self.kpi == 1
                kw = _dims((18, 1), (20, kb))
                kd = (slot + (kp0 // 2 * 40 + kp0 % 2) * L.PANEL, 20 * L.PANEL, (18, 20, 36, 720), (unit // 4, 8))
                vw = _dims((18, 1), (2, kb), (10, 2 * kb))
                vd = (slot + D.Q + 2 * kp0 * L.PANEL, 20 * L.PANEL, (36, 10, 144, 1440), (unit // 4, 8))
            if "k" in part:
                co.drains.append(MemTask([{"addr": k + img * L.PANEL, "nwords": self.kpi * self.nd * 18, "dims": kw,
                                           "iteration": (hstep, 8)}], repeat=8, ddr=kd))
            if "v" in part and self.d == 40:
                for off, w, dd in vws:
                    co.drains.append(MemTask([{"addr": vt + img * L.PANEL + 4 * off, "nwords": dd[1] // 4, "dims": w,
                                               "iteration": (hstep, 8)}], repeat=8, ddr=dd))
            elif "v" in part:
                co.drains.append(MemTask([{"addr": vt + img * L.PANEL, "nwords": self.kpi * self.nd * 18, "dims": vw,
                                           "iteration": (hstep, 8)}], repeat=8, ddr=vd))
        return co
