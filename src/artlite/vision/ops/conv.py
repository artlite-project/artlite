"""The 3x3 convs of the vision overlay's GEMM (``Gemm.conv``): the geometry of a conv's tile and input rows, the command's
tap tables (``ConvSpec.words``: the taps' MAC calls, walks and shift phases of ``core``'s ``sd15_conv``) and the memtile
tasks that bring a tile's A objects.

A conv's tile is the column's tokens, x = g + groups lane: ``R`` rows of ``Lg`` groups of 8 lanes (m = 16 R Lg tokens),
the input's rows ``R`` stride of ``Lg`` stride groups; the halo rows come from the neighbour columns' memtiles (``zero``: a
memtile region of zeros for the image's edges). Variants:
- plain (m 16 R Lg of one tile; ``R`` 1 or 2): an A object per chunk of 64 input channels of the input band rows with
  the halos;
- ``rc`` (32x32: x holds the column's rc rows of 4 groups): the op's ntile tiles are output rows r0 .. of 4 groups
  (m 64); per chunk an A object per dy: the 8 (row, group) pairs from input row r + dy - 1's predecessor group (slot 0,
  then the row's groups in slots 1-4, slot 5 the next row's group 0), into whose slots 0 and 5 the core shifts the row's
  group 3 one lane down and its group 0 one lane up: every tap's 4 bands are slots dx .. dx + 3;
- ``il`` (64x64, Lg 8): x holds rc rows of 8 groups (the halo rows included), a row's even groups then its odd ones;
  tile t = output row r0 + rstep (t // 2)'s even (t even) or odd groups (par0: row r0 + rstep t's even groups, a
  stride-2 conv's outputs); per chunk an A object per dy: input row r + dy - 1;
- ``wide`` (256 / 512 px rows, an op's tiles of one row): x holds 3 rows of ``wpos`` positions, a tile's object per dy a
  window of them;
- ``joined``: x's two images are one image's halves (``core.shift_masks``).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ... import txn
from ...operators.design import COLS
from .. import core as K
from .. import design as D
from .. import layout as L
from ..seq import MemTask
from .common import B_WALK_1, B_WALK_4, C_WALK_1, C_WALK_4, PAIR, PATS_4, ZERO_BYTES, band_walk, i32


class Wide(StrEnum):
    """The two classes of a wide conv's tiles in an op (the tap words' low / high half): first and middle, middle and
    last, or both middle."""

    FM = "fm"
    ML = "ml"
    MM = "mm"


ORDER2 = ((1, 1), (2, 2), (0, 0), (2, 1), (0, 1), (2, 0), (0, 2), (1, 0), (1, 2))
TPAR_ODD = 0xAAAAAAAA  # (tpar: the odd tiles' tap words)


def tap_word(first: int, ncalls: int, phase: int = 0, cross: bool = False) -> int:
    """A tap's word: its first MAC call, the number of calls, the shift phase before it, whether its A object is a new one."""
    return first * K.CALL | ncalls << 10 | phase << 12 | int(cross) << 15


class ConvTable:
    """A conv's command words from ``core.C_CALLS``: the taps' MAC calls (``K.CALL`` words each), the B and C walks, a
    word per tap (``tap_word``; the low half the even tiles', the high half the odd tiles'), the shift phases."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def add(self, words: list[int]) -> int:
        """Append one call's words; returns its index."""
        assert len(words) == K.CALL, len(words)
        self.calls += words
        return len(self.calls) // K.CALL - 1

    def finish(self, pats: tuple[int, ...], taps: list[int], phase: list[int]) -> np.ndarray:
        """The words from ``C_CALLS`` (``K.CONV_WORDS``), int32."""
        assert len(self.calls) <= K.NCALL * K.CALL and len(taps) == K.C_PHASE - K.C_TAPW, "the conv's calls or taps"
        out = np.zeros(K.CONV_WORDS, np.int64)
        out[:len(self.calls)] = self.calls
        out[K.C_PATS - K.C_CALLS:K.C_TAPW - K.C_CALLS] = pats
        out[K.C_TAPW - K.C_CALLS:K.C_PHASE - K.C_CALLS] = taps
        out[K.C_PHASE - K.C_CALLS:K.C_PHASE - K.C_CALLS + len(phase)] = phase
        return i32(out)

    def band_calls(self, a: list[int], bands: list[int]) -> tuple[int, int]:
        """The MAC calls of 4 bands in slots ``bands`` (byte offsets ``a``): one call of the 4 bands where the pairs (b0,
        b2), (b1, b3) are equally far apart (the il walk), else a call per pair. Returns (the first call, the number of
        calls)."""
        def call(src: int, g: int, nbr: int, walk: tuple) -> int:
            return self.add([(1 << 30 if g else 0) | nbr << 16 | a[src], *walk, 0, 0, walk[4], walk[4]])
        first = len(self.calls) // K.CALL
        if a[bands[2]] - a[bands[0]] == a[bands[3]] - a[bands[1]]:
            call(bands[0], 0, *band_walk([a[k] for k in bands]))
        else:
            call(bands[0], 0, *band_walk([a[bands[0]], a[bands[2]]]))
            call(bands[1], 1, *band_walk([a[bands[1]], a[bands[3]]]))
        return first, len(self.calls) // K.CALL - first


@dataclass
class ConvSpec:
    """A 3x3 conv (padding 1) of a GEMM: stride 1 or 2; the tile's R rows of Lg groups (the module docstring); zero: the
    memtile region of zeros for the image's edges (``Gemm.x`` holds the input); the variants rc / r0 / rstep, il / par0,
    wide / wpos, joined; tpar: per tile of the op the half of the tap words it takes (bit t set: the high)."""

    stride: int
    R: int
    zero: int
    Lg: int = 0
    rc: int = 0
    r0: int = 0
    rstep: int = 1
    il: bool = False
    par0: bool = False
    joined: bool = False
    wide: Wide | None = None
    wpos: int = 0
    tpar: int = TPAR_ODD

    def __post_init__(self) -> None:
        assert self.stride in (1, 2), self.stride
        self.wide = Wide(self.wide) if self.wide else None
        self.Li, self.Rc = self.Lg * self.stride, self.R * self.stride  # (the input's groups, the column's input rows)
        self.Ri = self.Rc + (2 if self.stride == 1 else 1)  # (an object's rows: the halos above / below)
        if self.il:
            assert (self.R, self.Lg, self.stride) == (1, 8, 1) and self.r0 >= 1, "il: 64x64 rows of 8 groups"
        elif self.wide:
            assert (self.R, self.Lg, self.stride) == (1, 4, 1) and 1 <= self.r0 < self.rc - 1, "wide: rows of 4 groups"
        elif self.rc:
            assert (self.R, self.Lg, self.stride) == (1, 4, 1), "rc: rows of 4 groups"

    def check(self, *, m: int, ntile: int, kdim: int) -> None:
        """The conv fits its GEMM's tile (m tokens), tiles and input channels."""
        assert m * (1 + self.il) == 16 * self.R * self.Lg and (ntile == 1 or self.rc) and kdim % 64 == 0, \
            f"conv: m {m}, R {self.R}, Lg {self.Lg}, ntile {ntile}, K {kdim}"
        if self.il:  # (the tiles' input rows in x)
            assert (self.par0 or ntile % 2 == 0) and \
                self.r0 + self.rstep * ((ntile if self.par0 else ntile // 2) - 1) + 1 < self.rc, "il: the tiles' rows"
        elif self.wide:  # (the op's tiles of one row: windows of 8 positions from 4 t)
            assert self.wpos >= 4 * ntile + 4, "wide: the tiles' windows"
        elif self.rc:
            assert self.r0 + self.rstep * (ntile - 1) < self.rc, "rc: the tiles' rows"

    def tile_word(self, first: int, ntile: int) -> int:
        """C_TPAR of the tiles from ``first`` (their bits of ``tpar``), as int32."""
        tpar = {Wide.FM: 0xFFFFFFFE, Wide.ML: 1 << (ntile - 1), Wide.MM: 0}[self.wide] if self.wide else self.tpar
        return int(i32((tpar & 0xFFFFFFFF) >> first))

    def col_free(self, ntile: int) -> bool:
        """The op's tasks are the same in every column (none reads a neighbour's memtile or the image's edge)."""
        return self.il or bool(self.wide) or (bool(self.rc) and self.r0 >= 1 and self.r0 + self.rstep * (ntile - 1) < self.rc - 1)

    def tap_order(self, c: int) -> list[tuple[int, int]]:
        """The taps (dy, dx) of chunk c in the order of its B slots (the 4-band convs': those of object rows 0, 1
        (region 2) and rows 2, 3 (region 3) alternating with the B slots' regions, in odd chunks rotated by one (9 B slots a
        chunk); before index 1, 5, 7 the shift phases)."""
        if self.il:
            return [(dy, dx) for dy in range(3) for dx in (1, 2, 0)]
        if self.R == 1:
            return [(dy, dx) for dy in range(3) for dx in range(3)]
        return list(ORDER2[c % 2:]) + list(ORDER2[:c % 2])

    # -- the command's words
    def words(self, nch: int) -> np.ndarray:
        """The conv's command words from ``core.C_CALLS`` (``core``'s sd15_conv): per tap (dy, dx) of the order its MAC
        calls (the bands (g, r), g-major, each group's source: object row r conv + dy's group conv g + dx - 1 (its byte
        offset: ``design.a_byte``), or a lane-shifted copy in cbuf (the phase's: a row's last group one lane down, its
        group 0 one lane up); a call per group where the two sources' row strides differ), the B and C walks, the taps'
        words, the shift phases. nch: chunks of 64 input channels."""
        if self.il:
            return self._il_words()
        if self.wide:
            return self._wide_words()
        if self.rc:
            return self._row_words()
        R, LG, Li, st = self.R, self.Lg, self.Li, self.stride
        assert (R, LG) in ((1, 1), (2, 2)), (R, LG)
        rs, il = Li * PAIR, R == 2  # (il: 4 bands, per group its rows alternate per block: C's halves in two regions)

        def row(k: int, g: int = 0) -> int:  # (object row k's group g)
            return D.a_byte(k * rs) + g * PAIR
        if il:  # (per phase: (sd, su) object rows per output row: cbuf slots [r][down, up])
            assert nch % 2 == 0 and st == 1  # (every pass starts at the same B parity)
            phases, at = [((0, 2), (1, 3)), ((2, 0), (3, 1)), ((1, 1), (2, 2))], {1: 0, 5: 1, 7: 2}
        else:
            phases, at = [((dy, dy),) for dy in range(3)], {0: 0, 3: 1, 6: 2}
        pw = np.zeros((3, 2, 4), np.int64)
        for p, rows in enumerate(phases):
            for r, (kd, ku) in enumerate(rows):
                pw[p, r] = (row(kd, Li - 1), 2 * PAIR * r, row(ku), 2 * PAIR * r + PAIR)
        cbuf, slots = D.L1["tab"], [D.L1["r23"] + D.R23[n] for n in ("a0", "a1")]
        calls, tapw, ph = [], [], None
        for i, (dy, dx) in enumerate(self.tap_order(0)):
            if i in at:
                ph = at[i]
            prov = {}  # ((down?, object row) -> cbuf offset) of the phase
            if ph is not None:
                for r, (kd, ku) in enumerate(phases[ph]):
                    prov[(True, kd)], prov[(False, ku)] = 2 * PAIR * r, 2 * PAIR * r + PAIR
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
                    calls.append((src[g:g + 1], g, 2, (1, sg - PAIR, 4, -sg - PAIR, 0)))
            elif il:
                s0 = src[0][2]
                calls.append((src, 0, 4, (1, s0 - PAIR, 4, -s0 - PAIR, src[1][1] - src[0][1] - s0 - PAIR)))
            else:
                calls.append((src, 0, 1, (4, -PAIR, 0, 0, -PAIR)))
            tapw.append(tap_word(first, len(calls) - first, at[i] + 1 if i in at else 0))
        table = ConvTable()
        for k, (src, g, nbr, walk) in enumerate(calls):  # (the walk's last step per A slot: cbuf <-> the object)
            c0, c1 = src[0][0], src[-1][0]
            last = [walk[4] + (cbuf - a if c1 and not c0 else a - cbuf if c0 and not c1 else 0) for a in slots]
            assert k in K.CROSS or last[0] == last[1]
            table.add([(1 << 31 if c0 else 0) | (1 << 30 if g else 0) | nbr << 16 | src[0][1], *walk, 0, 0, *last])
        pats = (*B_WALK_4, 0, 0, *C_WALK_4, 0, 0) if il else (*B_WALK_1, 0, 0, *C_WALK_1, 0, 0)
        return table.finish(pats, [w | w << 16 for w in tapw], pw.reshape(-1))  # (odd tiles: the same)

    def _row_words(self) -> np.ndarray:
        """``words`` of the row tiles (``rc``): 4 MAC calls relative to an object, the same for its 3 taps (dx): bands
        b = groups 0..3 in slots dx + b (slot s at ``design.a_byte`` (``PAIR`` s): slots 4.. in the second half); one call of
        the 4 bands where the pairs (b0, b2), (b1, b3) are equally far apart (the il walk), else a call per pair; the
        shift phase (slot 4 one lane down into slot 0, slot 1 up into slot 5) and a new A object before each dy's taps."""
        A = [D.a_byte(PAIR * k) for k in range(6)]
        table, first = ConvTable(), []
        for dx in range(3):
            first.append(table.band_calls(A, [dx + b for b in range(4)]))
        tapw = [tap_word(first[dx][0], first[dx][1], dx == 0, dx == 0 and dy > 0) for dy, dx in self.tap_order(0)]
        phase = [A[4], A[0] | K.SH_OBJ, A[1], A[5] | K.SH_OBJ]
        return table.finish(PATS_4, [w | w << 16 for w in tapw], phase)  # (odd tiles: the same)

    def _il_words(self) -> np.ndarray:
        """``words`` of the 64x64 row tiles (``il``): the object row r + dy - 1's groups [g0 g2 g4 g6 | g1 g3 g5 g7]
        (slots A0..A7, ``design.a_byte`` (``PAIR`` k)), its taps in the order dx 0, +1, -1 (``tap_order``). Even tiles:
        [A0..A3], [A4..A7], then g7 one lane down into slot A0 (both read), dx -1 from [A0 A4 A5 A6] (the pairs (A0, A5),
        (A4, A6): a call each); odd tiles: [A4..A7], g0 one lane up into A4, dx +1 from [A1 A2 A3 A4] (the pairs (A1, A3),
        (A2, A4)), [A0..A3]. Each shift phase's other half lands in cbuf (unused)."""
        A = [D.a_byte(PAIR * k) for k in range(8)]
        table = ConvTable()

        def il4(k: int) -> None:  # (4 slots from k: the il walk)
            a = A[k:k + 4]
            table.add([4 << 16 | a[0], 1, a[2] - a[0] - PAIR, 4, a[0] - a[2] - PAIR, a[1] - a[2] - PAIR, 0, 0,
                       a[1] - a[2] - PAIR, a[1] - a[2] - PAIR])

        def pair(g: int, x: int, y: int) -> None:
            table.add([(1 << 30 if g else 0) | 2 << 16 | x, 1, y - x - PAIR, 4, x - y - PAIR, 0, 0, 0, 0, 0])
        il4(0)
        il4(4)
        pair(0, A[0], A[5])
        pair(1, A[4], A[6])
        pair(0, A[1], A[3])
        pair(1, A[2], A[4])
        use = ({1: (0, 1), 2: (1, 1), 0: (2, 2)}, {1: (1, 1), 2: (4, 2), 0: (0, 1)})  # (per parity, dx: first call, calls)
        phase = ({0: 1}, {2: 2})  # (per parity: its shift before that tap)
        tapw = []
        for dy, dx in self.tap_order(0):
            w = [tap_word(*use[p][dx], phase[p].get(dx, 0), dx == 1 and dy > 0) for p in range(2)]
            tapw.append(w[0] | w[0 if self.par0 else 1] << 16)
        zeros = [0] * 5
        return table.finish(PATS_4, tapw, [A[7], A[0] | K.SH_OBJ, A[0], *zeros, A[7], PAIR, A[0], A[4] | K.SH_OBJ, 0, 0, 0, 0])

    def _wide_words(self) -> np.ndarray:
        """``words`` of the wide rows' tiles (``wide``: a tile = 4 consecutive groups of a row, its object per dy the
        window of 8 positions [g 4t - 1 .. g 4t + 6] of input row r + dy - 1, a row's positions being [its last group,
        groups 0 .. G - 1, its group 0]): bands b in slots dx + b, except in the row's first tile, where the dx -1
        band 0 is slot 0 (the last group) one lane down into slot 7, and in its last, where the dx +1 band 3 is slot 5
        (group 0) one lane up into slot 6. The op's tiles are two classes (``tpar``: the tap words' low / high half):
        "fm" first / middle, "ml" middle / last, "mm"."""
        A = [D.a_byte(PAIR * k) for k in range(8)]
        sets = {"m0": [0, 1, 2, 3], "f0": [7, 1, 2, 3], "1": [1, 2, 3, 4], "2": [2, 3, 4, 5], "l2": [2, 3, 4, 6]}
        table = ConvTable()
        at = {key: table.band_calls(A, sl) for key, sl in sets.items()}

        def tap(c: str, dy: int, dx: int) -> int:
            key = {0: "f0" if c == "f" else "m0", 1: "1", 2: "l2" if c == "l" else "2"}[dx]
            phase = {"f": 1, "l": 2}.get(c, 0) if dx == 0 else 0
            return tap_word(*at[key], phase, dx == 0 and dy > 0)
        lo, hi = self.wide.value
        tapw = [tap(lo, dy, dx) | tap(hi, dy, dx) << 16 for dy, dx in self.tap_order(0)]
        zeros = [0] * 5
        return table.finish(PATS_4, tapw, [A[0], A[7] | K.SH_OBJ, A[0], *zeros, A[0], PAIR, A[5], A[6] | K.SH_OBJ, 0, 0, 0, 0])

    # -- the memtile tasks
    def sends(self, *, x: int, x_np: int, nch: int, repeat: int, col: int, t: int) -> MemTask:
        """The A objects of tile t of the op (x, x_np: the input tensor; nch: chunks of 64 input channels; repeat: the
        task's executions)."""
        if self.il:
            return self._il_sends(x, x_np, nch, repeat, t)
        if self.wide:
            return self._wide_sends(x, x_np, nch, repeat, t)
        if self.rc:
            return self._row_sends(x, x_np, nch, repeat, col, t)
        return self._plain_sends(x, x_np, nch, repeat, col)

    def _row_sends(self, x: int, x_np: int, nch: int, repeat: int, col: int, t: int) -> MemTask:
        """Tile t's A objects of the row tiles (``rc``), per chunk 3 (dy): the pairs (row, group) from input row
        r + dy - 1's predecessor ([pair 8][kb 8][2], pair q at q 144 B of a kb's panels: q -1 is before x); the row above
        the column's first from the west column (the image's top: zeros), below its last from the east (the bottom)."""
        kbw, r = x_np * 18, self.r0 + self.rstep * t
        obj = [(36, 1), (8, kbw), (8, 36), (0, 144)]
        it = (8 * kbw, nch)

        def own(dy0: int, n: int, base: int = txn.MEM_OWN, row0: int | None = None) -> dict:
            q = 4 * (r + dy0 - 1 if row0 is None else row0) - 1
            return {"addr": x + 144 * q, "nwords": n * D.A_OBJ // 4, "dims": obj, "base": base, "iteration": it}
        zero = {"addr": self.zero, "nwords": D.A_OBJ // 4, "dims": [(288, 1), (8, 1)]}
        lo, hi = (1 if r == 0 else 0), (2 if r == self.rc - 1 else 3)
        chain = []
        if r == 0:
            chain.append(zero if col == 0 else own(0, 1, txn.MEM_WEST, self.rc - 1))
        chain.append(own(lo, hi - lo))
        if r == self.rc - 1:
            chain.append(zero if col == COLS - 1 else own(2, 1, txn.MEM_EAST, 0))
        assert x >= 144, "the first object's slot 0 before x"
        return MemTask(chain, repeat=repeat)

    def _wide_sends(self, x: int, x_np: int, nch: int, repeat: int, t: int) -> MemTask:
        """Tile t's A objects of the wide rows (``wide``), per chunk 3 (dy): rows r0 - 1 .. r0 + 1 of x (``wpos``
        positions each: [kb][row][position][2]), each the window of 8 positions from 4 t ([position 8][kb 8][2]: one
        BD)."""
        kbw, rw = x_np * 18, self.wpos * 36  # (words: a kb's panels, a row's positions)
        return MemTask([{"addr": x + 144 * (self.wpos * (self.r0 - 1) + 4 * t), "nwords": 3 * D.A_OBJ // 4,
                         "dims": [(36, 1), (8, kbw), (8, 36), (0, rw)], "iteration": (8 * kbw, nch)}], repeat=repeat)

    def _il_sends(self, x: int, x_np: int, nch: int, repeat: int, t: int) -> MemTask:
        """Tile t's A objects of the 64x64 row tiles (``il``), per chunk 3 (dy): input rows r - 1 .. r + 1 of x, each its
        groups [group 8][kb 8][2] (one BD: the rows are consecutive)."""
        kbw, r = x_np * 18, self.r0 + self.rstep * (t if self.par0 else t // 2)
        return MemTask([{"addr": x + PAIR * (r - 1), "nwords": 3 * D.A_OBJ // 4,
                         "dims": [(36, 1), (8, kbw), (8, 36), (0, 288)], "iteration": (8 * kbw, nch)}], repeat=repeat)

    def _plain_sends(self, x: int, x_np: int, nch: int, repeat: int, col: int) -> MemTask:
        """The chunks' A objects: [rows Ri][Li groups][kb 8][2] from the west column's last row (the image's top: zeros),
        the column's rows, the east column's first row (stride 1; the bottom: zeros); the last BD reads on to fill
        A_OBJ."""
        kbw, gw = x_np * 18, self.Li * 36  # (words: a kb's panels, a row's groups)
        assert 4 * 8 * gw <= ZERO_BYTES, "an edge column's missing halo row reads past the zeros"
        row = [(36, 1), (8, kbw), (self.Li, 36)]
        it = (8 * kbw, nch)
        last = x + ((self.Rc - 1) * self.Li * 2) * L.PANEL
        chain = [{"addr": self.zero, "nwords": 8 * gw} if col == 0 else
                 {"addr": last, "nwords": 8 * gw, "dims": row, "base": txn.MEM_WEST, "iteration": it},
                 {"addr": x, "nwords": self.Rc * 8 * gw, "dims": row + [(0, gw)], "iteration": it}]
        if self.stride == 1:
            chain.append({"addr": self.zero, "nwords": 8 * gw} if col == COLS - 1 else
                         {"addr": x, "nwords": 8 * gw, "dims": row, "base": txn.MEM_EAST, "iteration": it})
        pad = D.A_OBJ // 4 - self.Ri * 8 * gw
        assert pad >= 0, "an object past A_OBJ"
        chain[-1]["nwords"] += pad
        return MemTask(chain, repeat=repeat)
