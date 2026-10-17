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
    2 npo tasks per core: ``seq.Seq.op`` issues them in rounds)."""

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

    def __post_init__(self) -> None:
        assert self.m in (64, 16) and self.N % 320 == 0 and self.K % 8 == 0 and self.out in ("bf16", "bfp16", "geglu", "vt")
        self.npass = self.N // 320
        self.npo = self.npo or self.npass - self.p0
        assert 0 <= self.p0 and self.p0 + self.npo <= self.npass
        self.nkt = -(-self.K // D.KT)
        if self.conv:
            assert self.m == 16 * self.R * self.Lg and self.ntile == 1 and self.K % 64 == 0 and self.conv in (1, 2)
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
            self.N // 80 * self.ntile * L.ct_bytes(self.m)
        assert self.y + ybytes <= D.HEAP and self.x + self.K // 8 * self.x_np * L.PANEL <= D.HEAP, "tensors past the heap"
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
        if self.R == 1:
            return [(dy, dx) for dy in range(3) for dx in range(3)]
        return list(self.ORDER2[c % 2:]) + list(self.ORDER2[:c % 2])

    def conv_words(self) -> np.ndarray:
        """The conv's command words from ``core.C_CALLS`` (``core``'s v2_conv): per tap (dy, dx) of the order its MAC
        calls (the bands (g, r), g-major, each group's source: object row r conv + dy's group conv g + dx - 1 (its
        byte offset: ``design.a_byte``), or a lane-shifted copy in cbuf (the phase's: a row's last group one lane down,
        its group 0 one lane up); a call per group where the two sources' row strides differ), the B and C walks, the
        taps' words, the shift phases."""
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
        out[K.C_TAPW - K.C_CALLS:K.C_PHASE - K.C_CALLS] = tapw
        out[K.C_PHASE - K.C_CALLS:] = pw.reshape(-1)
        assert K.C_PHASE + 24 <= D.WS // 4
        return (out & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

    def command(self, bias: torch.Tensor | None, row: int) -> np.ndarray:
        """Row ``row``'s command slot (Q bytes); bias: (N,) or None (a conv's: its bias + the time bias)."""
        cmd = np.zeros(D.Q // 4, np.int32)
        om = {"bf16": K.O_BF16, "bfp16": K.O_BFP16, "geglu": K.O_GEGLU, "vt": K.O_VT}[self.out]
        st = {None: K.S_NONE, "gn": K.S_GN, "ln": K.S_LN}[self.stats]
        cmd[[K.C_KIND, K.C_M, K.C_NTILE, K.C_NPASS, K.C_NKT, K.C_NRES, K.C_OUT, K.C_STATS, K.C_STF]] = (
            K.K_GEMM, self.m, self.ntile, self.npo, self.nkt, self.nres, om, st, self.stf)
        if self.stats == "gn":
            c0 = self.ch0(row)
            cmd[[K.C_GSIZE, K.C_CH0, K.C_G0, K.C_GINV]] = (self.gsize, c0 + 80 * self.p0,
                                                           gn_window(c0, self.npass * 80, self.gsize), K.group_inv(self.gsize))
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
        m, ct = self.m, L.ct_bytes(self.m)
        P0, NPO = self.p0, self.npo
        co = ColOps()
        for r in range(4):
            sends, recvs = [], []
            for t in range(self.ntile):
                XN = self.x_np
                if self.conv and self.res is not None:  # (per pass: its chunks, then its residual objects)
                    ow = self.nres * D.A_OBJ // 4
                    res = MemTask([{"addr": self.res + (r * self.npass + P0) * ct, "nwords": ow, "dims": _contig(ow),
                                    "iteration": (ct // 4, NPO)}])
                    for _ in range(NPO):
                        sends += [self._conv_sends(col), res]
                    chain = []
                elif self.conv:
                    sends.append(self._conv_sends(col))
                    chain = []
                elif not self.aobj:  # (K 0: C := bias + the residual, a copy with statistics)
                    chain = []
                else:
                    chain = [{"addr": self.x + 2 * (m // 16) * t * L.PANEL, "nwords": self.aobj * D.A_OBJ // 4,
                              "dims": [(36, 1), (8, XN * 18), (m // 16, 36), (0, 8 * XN * 18)]}]
                if self.res is not None and not self.conv:
                    ow = self.nres * D.A_OBJ // 4
                    chain.append({"addr": self.res + ((r * self.npass + P0) * self.ntile + t) * ct, "nwords": ow,
                                  "dims": _contig(ow), "iteration": (self.ntile * ct // 4, NPO)})
                if chain:
                    sends.append(MemTask(chain, repeat=NPO))
                YN = self.y_np
                if self.out in ("bfp16", "vt"):  # (pass p: [rb][cb] -> kb (r npass + p) 10 + cb, token panel (m / 8) t + rb)
                    data = _panel_recv(self.y + ((r * self.npass + P0) * 10 * YN + m // 8 * t) * L.PANEL, YN, m // 8, 10,
                                       NPO)
                elif self.out == "geglu":  # (pass p: [rb][5 cb] -> kb (r npass + p) 5 + cb)
                    data = _panel_recv(self.y + ((r * self.npass + P0) * 5 * YN + m // 8 * t) * L.PANEL, YN, m // 8, 5,
                                       NPO, self.junk)
                else:  # (pass p: the C-tile object (r npass + p, t))
                    data = {"addr": self.y + ((r * self.npass + P0) * self.ntile + t) * ct, "nwords": NPO * ct // 4,
                            "dims": _contig(ct // 4) + [(NPO, self.ntile * ct // 4)]}
                chain = [data]
                if self.stats == "ln":
                    chain.append({"addr": self.rec + (4 * t + r) * D.OCH, "nwords": D.OCH // 4})
                recvs.append(MemTask(chain))
            if self.stats == "gn" and not self.stf & K.ST_HOLD:
                recvs.append(MemTask([{"addr": self.rec + r * D.OCH, "nwords": D.OCH // 4}]))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, img: int) -> RowOps:
        """img: the device address of the row's image: its command slot, then its weights (one run; the weights
        again per further tile)."""
        wb = self.npo * self.nkt * D.Q
        return RowOps(runs=[(img, D.Q + wb)] + [(img + D.Q, wb)] * (self.ntile - 1),
                      b_objects=1 + self.ntile * self.npo * self.nkt, parity=self.m == 64 and self.nkt > 0)


@dataclass
class Apply:
    """mode "gn" / "gnsilu" / "ln" of the bf16 C-tile tensor x (one tile of m tokens; blocks of 80 channels: ``nblk``
    per core of ``rows`` from block ``blk0``) into the panel tensor y (``y_np`` token panels per kb; its kb of the i-th core's first
    channel: ``y_kb0`` + 10 i nblk). GroupNorm: groups of ``gsize`` channels, the i-th core's first channel ``ch_off`` +
    80 i nblk in their space (its groups in one record window: ``gn_window``); the records come on the B stream
    (``row_ops(records=)``). LayerNorm: the producer's records at ``rec`` ([core][OCH]); ``C``: the norm's
    channels (1 / C)."""

    m: int
    nblk: int
    mode: str
    x: int
    y: int
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
        assert self.mode in ("gn", "gnsilu", "ln") and self.m in (64, 16)
        self.np = self.m // 8
        self.y_np = self.y_np or self.np
        self.gn = self.mode in ("gn", "gnsilu")
        self.rows = D.APPLY_ROWS
        ct = L.ct_bytes(self.m)  # (an object's row blocks, then its padding: m 64 a row block, m 16 80 words)
        assert ct - self.m * 160 == (1280 if self.m == 64 else 320)
        self.nhalf = self.nblk * (3 if self.m == 64 else 1)  # (A halves: 3 slots each, 1 + np slots a block)
        self.half_step = 960 if self.m == 64 else ct // 4
        self.aobj = -(-self.nhalf // 2) + (self.mode == "ln")
        assert self.nblk * 80 <= 640 and (not self.gn or (self.gsize >= 16 and self.tokens)) and (self.mode != "ln" or self.C)
        assert self.mode != "ln" or 4 * 8 * self.m <= D.A_OBJ // 2, "LayerNorm records past an A half"
        last = self.x + (self.blk0 + (len(self.rows) - 1) * self.nblk) * ct + 4 * (self.nhalf - 1) * self.half_step
        assert last + D.A_OBJ // 2 <= 512 * 1024, "x's walk past the memtile"

    def ch0(self, row: int) -> int:
        """Row ``row``'s first channel in the GroupNorm's channel space."""
        return self.ch_off + self.rows.index(row) * self.nblk * 80

    def command(self, row: int, nrec: int = 0) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        mode = {"gn": K.A_GN, "gnsilu": K.A_GNSILU, "ln": K.A_LN}[self.mode]
        cmd[[K.C_KIND, K.C_M, K.C_NBLK, K.C_MODE, K.C_NREC, K.C_EPS]] = K.K_APPLY, self.m, self.nblk, mode, nrec, _f32(self.eps)
        if self.gn:
            ch0, n = self.ch0(row), self.nblk * 80
            g0 = gn_window(ch0, n, self.gsize)
            cmd[[K.C_GSIZE, K.C_CH0, K.C_G0, K.C_INVN, K.C_GINV]] = (
                self.gsize, ch0, g0, _f32(1.0 / (self.gsize * self.tokens)), K.group_inv(self.gsize))
            cmd[K.C_GTAB:K.C_GTAB + n // 16] = K.group_words(ch0, n, self.gsize, g0)
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
        m, ct = self.m, L.ct_bytes(self.m)
        co = ColOps()
        for i, r in enumerate(self.rows):
            chain = []
            if self.mode == "ln":  # (the tile's records of the column's 4 cores: [core][S1 m | S2 m])
                chain.append({"addr": self.rec, "nwords": D.A_OBJ // 4, "dims": [(2 * m, 1), (4, D.OCH // 4)]})
            blk = self.blk0 + i * self.nblk  # (its blocks' objects, contiguous: per A half 3 slots (m 64: a block's
            # 8 row blocks and its padding in 3 halves; m 16: a block's 2 and its padding), then whatever follows)
            chain.append({"addr": self.x + blk * ct, "nwords": (self.aobj - (self.mode == "ln")) * D.A_OBJ // 4,
                          "dims": [(D.A_OBJ // 16, 1), (2, D.A_OBJ // 16), (self.nhalf, self.half_step)]})
            co.sends[r] = [MemTask(chain)]
            YN = self.y_np
            co.recvs[r] = [MemTask([_panel_recv(self.y + (self.y_kb0 + 10 * i * self.nblk) * YN * L.PANEL, YN, m // 8, 10,
                                                self.nblk)])]
        return co

    def row_ops(self, img: int, records: list[tuple] = (), nslots: int = 0) -> RowOps:
        """img: the row's command and gamma / beta slots (one run); records (GroupNorm): the runs of its ``nslots``
        record slots after them (``core.REC_SLOT`` records each, the last one's rest anything). (Only the op's ``rows``
        have a B stream.)"""
        runs = [(img, 2 * D.Q)] + list(records)
        return RowOps(runs=runs, b_objects=2 + nslots, parity=False)


@dataclass
class Attention:
    """Attention of 8 heads of d 160 over each image's keys: row i of ``design.ATTN_ROWS`` runs the units (image, head)
    of heads [4 i, 4 i + 4), image-major. q: the column's query panels ([kb 160][np], scaled by log2 e / sqrt 160), o:
    the output panels (the same layout); kv: the device address of the K / V^T slots ([image][head][group of 32
    keys][K | V^T slot]). Self-attention (keys = the image's tokens): ``kv_drains`` writes each column's part from its K
    and V^T panels; an image's token panels in a column are 2 apart (``kpc`` of them: its queries, its keys / 8).
    Cross-attention: nkeys (the context's: ``layout.kv_slots``, the last group's keys from nkeys masked)."""

    np: int
    q: int
    o: int
    kv: int
    nkeys: int = 0

    def __post_init__(self) -> None:
        self.kpc = self.np // 2
        self.ng = -(-self.nkeys // 32) if self.nkeys else 8 * self.kpc // 4  # (key groups of an image)
        self.nk = self.nkeys - 32 * (self.ng - 1) if self.nkeys else 32
        assert self.kpc in (1, 4), self.np
        assert self.q + (172 * self.np + 8) * L.PANEL <= D.HEAP, "the last head's walk past the heap"

    def command(self) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        cmd[[K.C_KIND, K.C_NUNIT, K.C_NQP, K.C_NG, K.C_NK]] = K.K_ATTN, 8, self.kpc, self.ng, self.nk
        return cmd.view(np.uint8)

    def col_ops(self) -> ColOps:
        """Per attention row and image: the units' queries (per head an A object: query pairs [pair][d-block][2], a
        single panel paired with the next one; a pair per half: 12 more d-blocks of the walk after its 20), their
        outputs ([qp][d-block])."""
        co = ColOps()
        hstep, kb = 20 * self.np * 18, self.np * 18  # (words: a head's 20 kb, one kb)
        for i, r in enumerate(D.ATTN_ROWS):
            sends, recvs = [], []
            for img in range(2):
                p0 = (80 * i * self.np + img) * L.PANEL
                sends.append(MemTask([{"addr": self.q + p0, "nwords": D.A_OBJ // 4,
                                       "dims": _dims((18, 1), (2, 36), (32, kb), (max(self.kpc // 2, 1), 72)),
                                       "iteration": (hstep, 4)}], repeat=4))
                recvs.append(MemTask([{"addr": self.o + p0, "nwords": self.kpc * 20 * 18,
                                       "dims": _dims((18, 1), (20, kb), (self.kpc, 36)), "iteration": (hstep, 4)}],
                                     repeat=4))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        """The attention rows' B streams: the command, then per image its 4 units' K / V^T slots (contiguous)."""
        unit = self.ng * 2 * D.Q
        return {r: RowOps(runs=[(cmd, D.Q)] + [(self.kv + (8 * img + 4 * i) * unit, 4 * unit) for img in range(2)],
                          b_objects=1 + 8 * 2 * self.ng, parity=self.kpc == 4)
                for i, r in enumerate(D.ATTN_ROWS)}

    def kv_bytes(self) -> int:
        return 16 * self.ng * 2 * D.Q

    def kv_drains(self, k: int, vt: int, col: int, *, part: str = "kv") -> ColOps:
        """Column ``col``'s drains of its keys' part of every slot (key panels kp0 .. of group g): from the K panels at
        ``k`` ([kb][np]) the K slot rows [key pair][d-block][2], from the V^T panels at ``vt`` the V^T slot [d-block
        pair][key panel][2]; per image a task of the 8 heads. part: "k" / "v": only K's / V^T's."""
        g, kp0 = divmod(col * self.kpc, 4)
        hstep, kb = 20 * self.np * 18, self.np * 18
        unit = self.ng * 2 * D.Q  # (a head's slots in DDR)
        co = ColOps()
        for img in range(2):
            slot = self.kv + 8 * img * unit + g * 2 * D.Q
            if self.kpc == 4:  # (the column's keys fill the group: whole slots)
                kw, kd = _dims((18, 1), (2, 36), (20, kb), (2, 72)), (slot, D.Q, None, (unit // 4, 8))
                vw, vd = _dims((18, 1), (2, kb), (4, 36), (10, 2 * kb)), (slot + D.Q, D.Q, None, (unit // 4, 8))
            else:  # (one key panel: every other panel of the K slot's pair rows, a pair of every V^T row of 4)
                kw = _dims((18, 1), (20, kb))
                kd = (slot + (kp0 // 2 * 40 + kp0 % 2) * L.PANEL, 20 * L.PANEL, (18, 20, 36, 720), (unit // 4, 8))
                vw = _dims((18, 1), (2, kb), (10, 2 * kb))
                vd = (slot + D.Q + 2 * kp0 * L.PANEL, 20 * L.PANEL, (36, 10, 144, 1440), (unit // 4, 8))
            if "k" in part:
                co.drains.append(MemTask([{"addr": k + img * L.PANEL, "nwords": self.kpc * 20 * 18, "dims": kw,
                                           "iteration": (hstep, 8)}], repeat=8, ddr=kd))
            if "v" in part:
                co.drains.append(MemTask([{"addr": vt + img * L.PANEL, "nwords": self.kpc * 20 * 18, "dims": vw,
                                           "iteration": (hstep, 8)}], repeat=8, ddr=vd))
        return co
