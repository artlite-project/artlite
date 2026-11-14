"""The attention ops of the vision overlay: ``Attention`` (d 40 / 80 / 160, roles c, a, s), ``WinAttention`` (sliding-window
attention of window pairs on the d-160 core, roles s, g) and ``DepthAttention`` (a few tokens on the d-40 core, role c).

Attention ``Attention``: 8 heads of d 160 (16x16, 8x8): the queries of a column's tokens against the keys of their image (K
/ V^T slots in DDR, written by ``Attention.kv_drains``, on the B streams of the attention rows).
"""

from dataclasses import dataclass

import numpy as np

from ...operators.design import COLS, ROWS
from .. import core as K
from .. import design as D
from .. import layout as L
from ..roles import Role
from ..seq import ColOps, MemTask, RowOps
from .common import CBAND, CCOL, PAIR, Command, check_heap, dims, i32


@dataclass
class Attention:
    """Attention of ``heads`` (8, or 12 / 16 of d 80) heads of d 160 (16x16, 8x8) or 80 (32x32; 16x16: a key group over two
    columns' keys) over each image's keys, for the queries of tile ``tile`` (4 query panels of each image, 8 token
    panels; 8x8: one panel of each), or of d 40 (64x64, role c) for all of the op's (16 query panels of each image, one
    op): row i of ``rows`` (in role a, s or c) runs the units (image, head) of heads [h i, h i + h), h = heads / len(rows),
    image-major. q: the op's query panels ([kb 8 d][np], scaled by log2 e / sqrt d),
    o: the output panels (the same layout); kv: the device address of the K / V^T slots ([image][head][group of
    keys][K | V^T]: d 160 groups of 32 keys, d 80 of 64), written by ``kv_drains``. Self-attention (keys = the image's
    tokens, of ``nbands`` ops of np token panels: band b's column c holds keys [(8 b + c) kpi, ..) of its image): an
    image's token panels in a column are 2 apart. Cross-attention: nkeys (the context's: ``layout.kv_slots``, the last
    group's keys from nkeys masked). d 40: a key group's K and V^T in one slot. raise_max: a key group whose P saturates
    (a score 63 past the softmax's reference max, the first group's: an attention sink) runs again with the reference
    raised (d 80 / 160), else P saturates. valid (d 80 / 160): per
    image a boolean array over its keys in the slots' order, the others masked (default: the first nkeys, or all).
    joint (d 80 / 160 self-attention): the images' keys one sequence (an entry over both images of its bands: MiniCPM-V's
    lone entries): image 1's groups after image 0's in one slot region, which both images' units stream (valid: per
    image over that sequence). dense (joint, d 80; any number of groups): the groups up to the last with a valid key
    computed, every one before it whole (valid: over the joint sequence, both images' the same; None: all), the rest
    streamed and skipped (``core``'s C_DENSE: the command's masks of that last live group only). kv_heads (grouped-query
    attention, d 80): the K / V^T tensors' heads, each the keys of the query heads kv_heads apart (query head h reads
    KV head h mod kv_heads: ``kv_drains`` writes each KV head's slots for each of its query heads)."""

    np: int
    q: int
    o: int
    kv: int
    nkeys: int = 0
    d: int = 160
    nbands: int = 1
    tile: int = 0
    rows: tuple[int, ...] = tuple(range(ROWS))
    raise_max: bool = False
    heads: int = 8
    valid: tuple | None = None
    joint: bool = False
    dense: bool = False
    kv_heads: int = 0

    def __post_init__(self) -> None:
        assert self.d in (40, 80, 160) and len(self.rows) in (2, 4) and (self.heads == 8 or self.d == 80 and self.heads in (12, 16))
        self.kvh = self.kv_heads or self.heads
        assert self.heads % self.kvh == 0 and (self.kvh == self.heads or self.d == 80)
        self.hpr = self.heads // len(self.rows)  # (heads per row)
        self.nd = self.d // 8  # (d-blocks of a head)
        self.gk = 4 if self.d == 160 else 8  # (key panels of a group)
        self.kpi = self.np // 2  # (an image's token panels in the op)
        self.kpc = 16 if self.d == 40 else min(self.kpi, 4)  # (query panels of a unit: an op per tile keeps a core's sends
        # and receives to two BDs each)
        self.role = {40: Role.C, 80: Role.A, 160: Role.S}[self.d]
        self.slots = 1 if self.d == 40 else 2  # (B slots of a key group)
        assert 8 * self.tile < self.np and (self.d != 40 or (self.kpi, self.tile) == (16, 0))
        keys = 8 * COLS * self.kpi * self.nbands
        self.ng = -(-(self.nkeys or keys) // (8 * self.gk))  # (key groups of an image; joint: of both)
        self.ngi = self.ng  # (an image's own groups: the second image's first in the joint sequence)
        if self.joint:
            assert self.d != 40 and not self.nkeys
            self.ng *= 2
        assert not self.dense or self.joint and self.d == 80
        self.nk = self.nkeys - 8 * self.gk * (self.ng - 1) if self.nkeys else 8 * self.gk
        assert self.kpc in (1, 4, 16) and (self.d == 160 or self.kpc >= 4), self.np
        check_heap(self.q, (((self.heads - 1) * self.nd + (8 if self.d == 40 else 32)) * self.np + 8) * L.PANEL,
                   "the last head's walk of the queries")

    def command(self) -> np.ndarray:
        cmd = Command(K.K_ATTN)
        cmd.set([K.C_NUNIT, K.C_NQP, K.C_NG, K.C_NK, K.C_ND, K.C_NKP, K.C_RAISE],
                (2 * self.hpr, self.kpc, self.ng, self.nk, self.nd, self.gk, self.raise_max))
        if self.dense:  # (masks: the last live group's, every key's, none's: two words each)
            ok = np.ones((self.ng, 64), bool)
            if self.valid is not None:
                v = np.asarray(self.valid[0], bool)
                assert np.array_equal(v, np.asarray(self.valid[1], bool)) and v.size <= ok.size, "dense: one sequence"
                ok[:] = False
                ok.reshape(-1)[:v.size] = v
            live = ok.any(1)
            nlive = int(np.flatnonzero(live)[-1]) + 1 if live.any() else 1
            assert ok[:nlive - 1].all() and not live[nlive:].any(), "dense: whole groups, then the last live one"
            words = [int(sum(1 << k for k in range(32) if ok[nlive - 1, 32 * j + k])) for j in range(2)]
            m = i32(words + [-1, -1, 0, 0])
            cmd.set([K.C_DENSE, K.C_NLIVE], (1, nlive))
            for img in range(2):
                cmd.at(K.C_KMASK[img], m)
        elif self.d != 40:
            assert self.ng <= 32, "key masks"
            km = self.key_masks().reshape(2, -1)
            for img in range(2):
                cmd.at(K.C_KMASK[img], km[img])
        if self.d == 80:
            # (P V^T of d 80: a band per query pair, block by block in turn (its O in the C buffer's two halves: a
            # block's C stores and the next one's loads in two L1 regions), its P from the scratch buffers s0, s1)
            sp = D.R23["s1"] - D.R23["s0"]
            w = [(1, sp - PAIR, 0, 0, -sp - PAIR), (1, -PAIR, 0, 0, 0), (1, CBAND, 0, 0, CCOL - CBAND)]
            cmd.at(K.C_WALKS, i32([x for d in w for x in (*d, 0, 0)]))
        return cmd.bytes()

    def key_masks(self) -> np.ndarray:
        """C_KMASK: per image and key group its keys' valid bits (bit k: key k of the group; two words, -1 past a group
        of 32 keys)."""
        per = 8 * self.gk  # (keys of a group)
        out = np.full((2, self.ng, 2), -1, np.int64)
        for img in range(2):
            ok = np.zeros(self.ng * per, bool)
            if self.valid is not None:
                v = np.asarray(self.valid[img], bool)
                ok[:v.size] = v
            else:
                ok[:(self.ng - 1) * per + self.nk] = True
            bits = ok.reshape(self.ng, per // 32 if per >= 32 else 1, -1)
            for g in range(self.ng):
                for j in range(per // 32):
                    out[img, g, j] = int(sum(1 << k for k in range(32) if bits[g, j, k]))
        return (out & 0xFFFFFFFF).astype(np.uint32).view(np.int32).reshape(-1)

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
                                       "dims": dims((18, 1), (2, 36), (nkb, kb), (max(self.kpc // 2, 1), 72)),
                                       "iteration": (hstep, h)}], repeat=h))
                recvs.append(MemTask([{"addr": self.o + p0, "nwords": self.kpc * self.nd * 18,
                                       "dims": dims((18, 1), (self.nd, kb), (self.kpc, 36)), "iteration": (hstep, h)}],
                                     repeat=h))
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        """The attention rows' B streams: the command, then per image its units' K / V^T slots (contiguous)."""
        unit, h = self.ng * self.slots * D.Q, self.hpr
        return {r: RowOps(runs=[(cmd, D.Q)] + [(self.kv + (self.heads * img * (not self.joint) + h * i) * unit, h * unit)
                                               for img in range(2)],
                          b_objects=1 + 2 * h * self.slots * self.ng, parity=self.kpc == 4, role=self.role)
                for i, r in enumerate(self.rows)}

    def kv_bytes(self) -> int:
        return (1 if self.joint else 2) * self.heads * self.ng * self.slots * D.Q

    def kv_drains(self, k: int, vt: int, col: int, *, part: str = "kv", band: int = 0) -> ColOps:
        """Column ``col``'s drains of its keys' part of every slot (band ``band``'s: key panels kp0 .. of group g): from the
        K panels at ``k`` ([kb][np]) the K slot rows [key block][d-block][2], from the V^T panels at ``vt`` the V^T slot
        [d-block pair][key panel][2]; per image a task of the heads. part: "k" / "v": only K's / V^T's."""
        g, kp0 = divmod((band * COLS + col) * self.kpi, self.gk)
        hstep, kb = self.nd * self.np * 18, self.np * 18
        unit = self.ng * self.slots * D.Q  # (a head's slots in DDR)
        co = ColOps()
        H = self.kvh
        for img, rep in ((i, r) for i in range(2) for r in range(self.heads // H)):
            slot = self.kv + (g + img * self.ngi if self.joint else self.heads * img * self.ng + g) * self.slots * D.Q + \
                rep * H * unit
            if self.d == 40:  # (the column's keys fill two groups: per head their K, their V^T parts of d-blocks 0..3
                # ([d-block pair][key panel][2]) and of d-block 4 ([key panel])
                assert self.kpi == 2 * self.gk
                kw = dims((18, 1), (2, 36), (5, kb), (8, 72))
                kd = (slot, D.Q, (720, 2, 1440, 2880), (unit // 4, H))
                vws = [(0, dims((18, 1), (2, kb), (16, 36), (2, 2 * kb)), (slot + D.Q // 2, 4 * 1152, (288, 2, 1440, 288),
                                                                           (unit // 4, H))),
                       (4 * kb, dims((18, 1), (16, 36)), (slot + D.Q // 2 + 2304, 2 * 576, (144, 2, 1440, 144),
                                                          (unit // 4, H)))]
            elif self.kpi == self.gk:  # (the column's keys fill the group: whole slots)
                kw, kd = dims((18, 1), (2, 36), (self.nd, kb), (self.gk // 2, 72)), (slot, D.Q, None, (unit // 4, H))
                vw = dims((18, 1), (2, kb), (self.gk, 36), (self.nd // 2, 2 * kb))
                vd = (slot + D.Q, D.Q, None, (unit // 4, H))
            elif self.d == 80 and 2 * self.kpi == self.gk:  # (half a group: the K slot's key blocks kp0 / 2 .., the V^T
                # slot's key panels kp0 .. of each d-block pair row)
                n = self.kpi * self.nd * L.PANEL
                kw = dims((18, 1), (2, 36), (self.nd, kb), (self.kpi // 2, 72))
                kd = (slot + kp0 // 2 * 2 * self.nd * L.PANEL, n, None, (unit // 4, H))
                vw = dims((18, 1), (2, kb), (self.kpi, 36), (self.nd // 2, 2 * kb))
                row = 2 * self.gk * 18  # (words of a d-block pair row)
                vd = (slot + D.Q + 2 * kp0 * L.PANEL, n, (2 * self.kpi * 18, self.nd // 2, row, self.nd // 2 * row),
                      (unit // 4, H))
            else:  # (d 160, one key panel: every other panel of the K slot's pair rows, a pair of every V^T row of 4)
                assert self.d == 160 and self.kpi == 1
                kw = dims((18, 1), (20, kb))
                kd = (slot + (kp0 // 2 * 40 + kp0 % 2) * L.PANEL, 20 * L.PANEL, (18, 20, 36, 720), (unit // 4, H))
                vw = dims((18, 1), (2, kb), (10, 2 * kb))
                vd = (slot + D.Q + 2 * kp0 * L.PANEL, 20 * L.PANEL, (36, 10, 144, 1440), (unit // 4, H))
            if "k" in part:
                co.drains.append(MemTask([{"addr": k + img * L.PANEL, "nwords": self.kpi * self.nd * 18, "dims": kw,
                                           "iteration": (hstep, H)}], repeat=H, ddr=kd))
            if "v" in part and self.d == 40:
                for off, w, dd in vws:
                    co.drains.append(MemTask([{"addr": vt + img * L.PANEL + 4 * off, "nwords": dd[1] // 4, "dims": w,
                                               "iteration": (hstep, H)}], repeat=H, ddr=dd))
            elif "v" in part:
                co.drains.append(MemTask([{"addr": vt + img * L.PANEL, "nwords": self.kpi * self.nd * 18, "dims": vw,
                                           "iteration": (hstep, H)}], repeat=H, ddr=vd))
        return co



@dataclass
class WinAttention:
    """Sliding-window attention of window pairs on the d-160 core (role s, ``core``'s C_QMASK): LFM3-Audio's encoder (8
    query heads, 4 KV heads, heads of 96 in 160). A band's rows in the interleaved layout (column c's token panel j: rows
    64 j + 8 c ..): window pair j (64 rows: window 2 j in columns 0-3, 2 j + 1 in 4-7, one query panel per column) is a
    unit. Core row r runs KV head r: per unit its query pair (query heads 2 r, 2 r + 1 of the column's panel j: the
    query object's first half, ``q``'s panels [kb][np]) against the unit's 4 key groups on the B stream ([own, previous]
    of window 2 j, then of 2 j + 1: the K / V^T image ``kv_image``'s unit), each column's per-query masks (the object's
    second half: ``masks``, at memtile ``mask``) keeping its window's two groups (own first: the diagonal key sets the
    reference max), outputs into the panels ``o`` (the same layout as q). image_units: the units of the layer's K / V^T
    image (``kv_image``), which set its KV heads' stride."""

    q: int
    o: int
    mask: int
    kv: int
    image_units: int
    units: int = 8
    np: int = 8
    gate: int | None = None  # (the gates' C tile, ``Gemm(out="isig")``: ``col_ops``)

    ND, GROUPS = 20, 4

    def __post_init__(self) -> None:
        assert self.image_units >= self.units > 0, f"the layer's image of {self.image_units} units holds {self.units}"
        self.kv_head_bytes = self.kv_image(self.image_units) // ROWS  # (a KV head's units in the image)

    def command(self) -> np.ndarray:
        cmd = Command(K.K_ATTN)
        cmd.set([K.C_NUNIT, K.C_NQP, K.C_NG, K.C_NK, K.C_ND, K.C_NKP, K.C_RAISE, K.C_QMASK],
                (self.units, 2, self.GROUPS, 32, self.ND, 4, 1, 1))
        return cmd.bytes()

    @staticmethod
    def unit_bytes() -> int:
        """Bytes of a unit's K / V^T groups of one KV head."""
        return WinAttention.GROUPS * 2 * D.Q

    @staticmethod
    def kv_image(units: int) -> int:
        """Bytes of a layer's K / V^T image: [KV head 4][unit units + 1][group 4][K | V^T] (the last unit: where the
        dispatch's last group goes as the next one's previous group, ``kv_drains``)."""
        return ROWS * (units + 1) * WinAttention.unit_bytes()

    def col_ops(self) -> ColOps:
        """Per core row its units' query objects (the pair's 32 d-blocks: head 2 r's 20 and what follows; then the
        column's masks of the unit and what follows; with ``gate`` (role g), the masks, then per query of the unit's panel
        j its word of heads 2 r, 2 r + 1 from the gates' C tile (m 16: row block j; ``core.WIN_GATE``), then the mask
        image again) and outputs ([qp 2][d-block 20] -> kb 20 (2 r + qp) + db)."""
        co = ColOps()
        kb = self.np * 18  # (words of a kb)
        assert self.gate is None or self.units <= 2, "the gates of one m-16 tile"
        for r in range(ROWS):
            h0 = 2 * r * self.ND * kb
            tail = [{"addr": self.mask, "nwords": D.A_OBJ // 8, "iteration": (32, self.units)}] if self.gate is None else [
                {"addr": self.mask, "nwords": 32, "iteration": (32, self.units)},
                {"addr": self.gate + 4 * r, "nwords": 8, "dims": [(8, 4)], "iteration": (320, self.units)},
                {"addr": self.mask, "nwords": D.A_OBJ // 8 - 40}]
            co.sends[r] = [MemTask([{"addr": self.q + 4 * h0, "nwords": D.A_OBJ // 8, "dims": dims((18, 1), (2, self.ND * kb), (32, kb)),
                                     "iteration": (18, self.units)}] + tail,
                                   repeat=self.units)]
            co.recvs[r] = [MemTask([{"addr": self.o + 4 * h0, "nwords": 2 * self.ND * 18,
                                     "dims": dims((18, 1), (self.ND, kb), (2, self.ND * kb)), "iteration": (18, self.units)}],
                                   repeat=self.units)]
        return co

    def row_ops(self, cmd: int, unit0: int) -> dict[int, RowOps]:
        """The rows' B streams: the command, then KV head r's groups of units unit0 .. + units (``kv``: the image)."""
        ub, hb = self.unit_bytes(), self.kv_head_bytes
        return {r: RowOps(runs=[(cmd, D.Q), (self.kv + r * hb + unit0 * ub, self.units * ub)],
                          b_objects=1 + self.units * self.GROUPS * 2, parity=True, role=Role.S if self.gate is None else Role.G)
                for r in range(ROWS)}

    @staticmethod
    def masks(col: int, units: int, start: bool = False, window: int = 32) -> np.ndarray:
        """Column ``col``'s per-query masks of ``units`` units ([unit][group 4][key panel 4][2] uint32: bit 8 q + k of word
        h: query 4 h + q of the column's panel, key 8 kp + k of the group): its window's own group (keys up to the query's
        row) and previous one (keys after it), the last ``window`` rows (<= 32) up to the query's; start: unit 0's
        previous group masked (the stream's first rows)."""
        out = np.zeros((units, 4, 4, 2), np.uint32)
        w, i0 = col // 4, 8 * (col % 4)
        for kp in range(4):
            for h in range(2):
                own = prev = 0
                for q in range(4):
                    for k in range(8):
                        i, key = i0 + 4 * h + q, 8 * kp + k
                        own |= (i - window < key <= i) << (8 * q + k)
                        prev |= (key - 32 > i - window) << (8 * q + k)
                out[:, 2 * w, kp, h], out[:, 2 * w + 1, kp, h] = own, prev
        if start:
            out[0, 2 * w + 1] = 0 if w == 0 else out[0, 2 * w + 1]
        return out

    def kv_drains(self, k: int, vt: int, col: int, band: int, image: int) -> ColOps:
        """Column ``col``'s drains of band ``band``'s keys (its panels j: group g = 2 j + col // 4, key panel col % 4):
        K from the panels at ``k``, V^T from those at ``vt`` ([kb][np], 20 kb a KV head) into both slots of each group
        (even g = 2 j: unit j's groups 0 and 3; odd: unit j's group 2 and unit j + 1's group 1) of each KV head; image:
        the layer's K / V^T image (device address)."""
        kb, ub, hb = self.np * 18, self.unit_bytes(), self.kv_head_bytes
        kp, odd = col % 4, col // 4
        dests = [(2, 0), (1, 1)] if odd else [(0, 0), (3, 0)]  # ((group, unit offset) of the two slots)
        co = ColOps()
        for g, du in dests:
            slot = image + (band * self.units + du) * ub + g * 2 * D.Q
            # (K: the slot's key block kp // 2, panel kp % 2 of each d-block pair; V^T: its key panel kp of each
            # d-block pair row: ``layout.kv_slots``)
            co.drains.append(MemTask([{"addr": k, "nwords": self.units * self.ND * 18,
                                       "dims": dims((18, 1), (self.ND, kb), (self.units, 18)), "iteration": (self.ND * kb, ROWS)}],
                                     repeat=ROWS, ddr=(slot + (kp // 2 * 40 + kp % 2) * L.PANEL, self.units * self.ND * L.PANEL,
                                                      (18, self.ND, 36, ub // 4), (hb // 4, ROWS))))
            co.drains.append(MemTask([{"addr": vt, "nwords": self.units * self.ND * 18,
                                       "dims": dims((18, 1), (2, kb), (self.ND // 2, 2 * kb), (self.units, 18)),
                                       "iteration": (self.ND * kb, ROWS)}],
                                     repeat=ROWS, ddr=(slot + D.Q + 2 * kp * L.PANEL, self.units * self.ND * L.PANEL,
                                                      (36, self.ND // 2, 144, ub // 4), (hb // 4, ROWS))))
        return co



@dataclass
class DepthAttention:
    """Bidirectional attention of a few tokens on the d-40 core (role c, ``core``'s C_NPAIR, C_GATE): LFM3-Audio's
    depthformer (32 query heads, 8 KV heads of 32 dims in 40, its slots the tokens of column 0's m-16 tile: the other
    columns run the same on their tokens, against column 0's keys). Query head u of the NPU's order (``Q``'s rows,
    ``O``'s columns, the gates': HF's head 4 (u % 8) + u // 8) reads KV head u % 8; core row r runs heads 8 r .. 8 r + 7,
    a unit each: its query object [the head's two panels' 8 d-blocks (q: panels [kb 5 u ..][2]) | the gates of its block
    of 8 heads (the gates' C tile at ``gate``, ``Gemm(out="isig")``, m 16: [rb 2][cb 10][64] bf16, block r: 64 words, the
    head's channel u % 8)
    | the query panels again], one key group of 64 (``nk`` valid) in its B slot (``kv``: [unit 32][K | V^T], written by
    ``kv_drains``), its output O g / l -> o's panels [kb 5 u ..][2] (and a chunk's padding into ``junk``, 720 B)."""

    q: int
    o: int
    kv: int
    gate: int
    junk: int
    nk: int = 12

    HEADS, KVH, ND = 32, 8, 5
    GATE = 1152  # (the gate words' byte in a query object: query pair 1's place)

    @staticmethod
    def order() -> np.ndarray:
        """Per query head of the NPU's order its HF head (module docstring)."""
        u = np.arange(DepthAttention.HEADS)
        return 4 * (u % DepthAttention.KVH) + u // DepthAttention.KVH

    def command(self) -> np.ndarray:
        cmd = Command(K.K_ATTN)
        cmd.set([K.C_NUNIT, K.C_NG, K.C_NK, K.C_NPAIR, K.C_GATE], (self.HEADS // ROWS, 1, self.nk, 1, self.GATE))
        return cmd.bytes()

    def col_ops(self) -> ColOps:
        co = ColOps()
        hb = self.ND * 2 * L.PANEL  # (a head's panels: 720 B)
        for r in range(ROWS):
            u0 = r * self.HEADS // ROWS
            co.sends[r] = [MemTask([{"addr": self.q + u0 * hb, "nwords": 288, "iteration": (hb // 4, 8)},
                                    {"addr": self.gate + 128 * r, "nwords": 64, "dims": [(32, 1), (2, 320)]},
                                    {"addr": self.q, "nwords": D.A_OBJ // 4 - 352}], repeat=8)]
            co.recvs[r] = [MemTask([{"addr": self.o + u0 * hb, "nwords": hb // 4, "dims": [(18, 1), (5, 36), (2, 18)],
                                     "iteration": (hb // 4, 8)},
                                    {"addr": self.junk, "nwords": D.OCH // 4 - hb // 4}], repeat=8)]
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        """The rows' B streams: the command, then their units' slots."""
        n = self.HEADS // ROWS
        return {r: RowOps(runs=[(cmd, D.Q), (self.kv + r * n * D.Q, n * D.Q)], b_objects=1 + n, parity=False, role=Role.C)
                for r in range(ROWS)}

    @staticmethod
    def kv_bytes() -> int:
        return DepthAttention.HEADS * D.Q

    def kv_drains(self, k: int, vt: int) -> list[dict[int, ColOps]]:
        """Column 0's drains of its keys (two key panels) into every unit's slot (unit u: KV head u % 8), three ops (a
        task each: 32 executions): from the K panels at ``k`` ([kb 5 h ..][2]: key block 0 of the slot's K [key block 4]
        [d-block 5][2]), from the V^T panels at ``vt`` (the slot's V^T: [d-block pair 2][key panel 8][2], then [key panel
        8] of d-block 4: key panels 0, 1). The rest of each slot is never written (zeros: the masked keys)."""
        hb, Q = self.ND * 2 * L.PANEL, D.Q
        it = (Q // 4, self.HEADS)
        return [{0: ColOps(drains=[t])} for t in (
            MemTask([{"addr": k, "nwords": hb // 4, "iteration": (hb // 4, self.KVH)}], repeat=self.HEADS,
                    ddr=(self.kv, hb, None, it)),
            MemTask([{"addr": vt, "nwords": 144, "dims": [(18, 1), (2, 36), (2, 18), (0, 72)],
                      "iteration": (hb // 4, self.KVH)}], repeat=self.HEADS, ddr=(self.kv + Q // 2, 576, (72, 2, 288, 288), it)),
            MemTask([{"addr": vt + 8 * L.PANEL, "nwords": 36, "iteration": (hb // 4, self.KVH)}], repeat=self.HEADS,
                    ddr=(self.kv + Q // 2 + 2304, 144, None, it)))]
