"""GemmQ's / GemmQd's data movement: the shim taps of the weight fills, the A strips and the C drains, and the memtile
walks of the FIFOs (``Dataflow``), pure host arithmetic over a ``Geometry`` that ``tests/gemm/test_dma.py`` checks
against the layout the kernels expect.

B (weights): shim -> memtile objects of ``b_obj`` words (one super-block slice of the pass's tiles, written with
``b_wdims``) -> L1 objects of ``b_l1`` words (one 64-k tile, sent with ``b_rdims``, which may zero-pad it) to the
column's cores (``b_parts`` = 2: the memtile object in two segments, each written with ``b_wdims``, segment h read with
``b_rdims`` to the cores of half h). A: shim -> memtile objects of ``a_obj`` (bfp16 panels: the row's strip if ``a_res``,
sent ``a_repeat`` times, else K chunks; written with ``a_wdims`` in MAC pop order) -> L1 objects of ``a_l1`` (one k-tile).
C: each core's objects of ``c_l1`` joined in the memtile, sent to the shim with ``c_dims``. Sizes in elements of the
FIFO object types: weights i32 words, A and C bf16 units.

The weights of all passes come as one chained task per column and token chunk where a fill per pass would need a task
each (``b_chain``): per-pass groups blew the prefill's control code past a 64 MB heap BO, and more than ``SHIM_QUEUE``
queued fills on a channel stall the controller before the A fills are issued (a hang).
"""

import itertools
from dataclasses import dataclass

from aie.helpers.taplib import TensorAccessPattern

from ..formats import FORMATS, KT, slot
from ..operators.design import MAX_BD_ITER, MAX_CHAIN_BDS, stream_walk
from .geometry import Geometry

A_RESIDENT_MAX = 448 * 1024  # bytes of a core row's A strip kept in its memtile (one buffer: the
# memtile resends it per pass with a BD repeat; depth x repeat BDs must stay within 24 per channel; one
# memtile BD moves < 512 KB (17-bit length; 442 KB strips run) and a split chain cannot carry the repeat)
MEMTILE_BYTES = 512 * 1024


@dataclass(frozen=True)
class Dataflow:
    """The walks of a GEMM's FIFOs (the fields) and the taps of its runtime sequence (the methods), of the geometry
    ``geo``."""

    geo: Geometry
    b_obj: int
    b_l1: int
    b_wdims: TensorAccessPattern | None
    b_rdims: TensorAccessPattern | None
    b_parts: int  # (pair: the memtile object in 2 segments, each written with b_wdims and read with b_rdims)
    a_res: bool
    a_obj: int
    a_l1: int
    a_wdims: TensorAccessPattern | None
    a_repeat: int
    c_l1: int
    c_dims: TensorAccessPattern | None

    @classmethod
    def make(cls, geo: Geometry) -> "Dataflow":
        """The walks of ``geo`` (its tokens per core row ``m`` chosen)."""
        assert geo.m, "the walks scale with the tokens per core row"
        m, K, no = geo.m, geo.KA, geo.n_out
        r = t = 8
        cb = geo.c_bfp16
        noperm = "noperm" in geo.ablate
        obj_w = geo.tiles * geo.slice // 4  # words per memtile object (the pass's tiles x one slice)
        rpad = None
        if geo.planar:
            # tile slices of 32-byte planes per row (plane major: [plane][row R][8 words], Q5_K / Q6_K; row major:
            # the Q4 nibbles [row R][step s][8 words]): S2MM writes [plane][g = t R + r][8 words]; MM2S reads
            # k-tile objects [plane][word ii 2][g] (words 2 kt + ii: groups 2 kt + ii) of the first n_read rows,
            # zero-padded to n (64-byte aligned runs of 16 rows for the dequant's loads)
            R, T, TR, P = geo.R, geo.tiles, geo.TR, FORMATS[geo.fmt].planes
            if geo.units:  # (the unit of the pass's tiles as it arrives, each tile's padded to whole lines; in
                # parts: the stream [part][tile] written as [tile][part])
                uw, pw, us = geo.units // 4, geo.unit_pitch // 4, geo.us
                assert us == 1 or pw == uw, (us, pw, uw)
                wdims = [(us, uw // us), (T, uw), (uw // us, 1)] if us > 1 else None
                rdims = [(T * us, uw // us), (uw // us, 1)]
                rpad = [(0, 0), (0, pw - uw)] if pw > uw else None
            elif FORMATS[geo.fmt].word_major:
                # word-major slices ([word][row R], Q2_K / Q3_K: 2 P words per k-tile): S2MM writes [word][g];
                # MM2S reads k-tile objects [word 2 P][g] (words 2 P kt ..)
                wdims = [(T, R), (8 * P, TR), (R, 1)]
                rdims = [(4, 2 * P * TR), (2 * P, TR), (geo.n_read, 1)]
            else:
                if FORMATS[geo.fmt].plane_major:
                    wdims = [(T, 8 * R), (P, 8 * TR), (8 * R, 1)]
                elif geo.us > 1:  # (sub-units: the stream [sub-unit][tile][row 16][step][8 words], g = its row)
                    wdims = [(TR, 8), (P, 8 * TR), (8, 1)]
                else:
                    wdims = [(T, 8 * R), (R, 8), (P, 8 * TR), (8, 1)]
                rdims = [(4, 2), (P, 8 * TR), (2, 1), (geo.n_read, 8)]
            # (a padded MM2S BD moves its padded walk, not the object: reading fewer rows than the memtile
            # holds needs a padding, if only of zeros)
            if not geo.units and (geo.n > geo.n_read or geo.n_read < TR):
                rpad = [(0, 0)] * (len(rdims) - 1) + [(0, geo.n - geo.n_read)]
        elif (dims := slot(geo.fmt).tile_major_dims(geo)) is not None:  # (BFP16)
            wdims, rdims = dims
        else:
            # Q4: tile rows store words of 8 consecutive k ((s, i) = k in [32 i + 8 s, +8)), a MMUL panel
            # is 8 rows x 8 k: S2MM writes each slice [tile][i][s][row], MM2S reads k-tile objects
            # [kt][tile][i % 2][s][row] (i = 2 kt + i % 2) = packed panels in the dequant's order
            # (pair: the memtile object in two segments of tq tiles, each written and read the same way: a
            # distribute link writes its input per segment; segment h to the cores of half h)
            wdims = [(geo.tq, 512), (16, 1), (4, 16), (8, 64)]
            rdims = [(4, 128), (geo.tq, 512), (2, 64), (64, 1)]
        # A strip of a core row: resident in the memtile and resent per pass when it fits next to the
        # column's B and C FIFOs, else refetched per pass in K chunks
        c_obj = m * no * 9 // 8 * geo.c_parts if cb else (m * geo.n_c + geo.c_tail) * 2
        others = 2 * obj_w * 4 + 2 * geo.nrows * c_obj  # (B, C: depth 2)
        a_res = m * K * 9 // 8 <= min(A_RESIDENT_MAX, MEMTILE_BYTES - others - 4096)
        ak = K if a_res else geo.kc
        # bfp16 panels (36 bf16 units) arrive [kt][token group g = 2 band + half][k-step][panel] (shim
        # pattern); the memtile stores them [kt][band][k-step][half][panel] = the MAC pop order
        a_wdims = [(ak // KT * (m // 16), 8 * 72), (2, 36), (8, 72), (36, 1)]
        a_obj = m * ak * 9 // 16
        c_l1 = m * no * 9 // 16 * geo.c_parts if cb else m * geo.n_c + geo.c_tail
        parts = 2 if geo.pair else 1
        return cls(
            geo=geo, b_obj=obj_w, b_l1=geo.b_l1_words,
            b_wdims=None if noperm or wdims is None else stream_walk(obj_w // parts, wdims),
            b_rdims=None if noperm else stream_walk(obj_w // parts, rdims, rpad), b_parts=parts,
            a_res=a_res, a_obj=a_obj, a_l1=m * 36, a_wdims=stream_walk(a_obj, a_wdims),
            a_repeat=geo.passes * geo.G if a_res and geo.passes * geo.G > 1 else 1,
            c_l1=c_l1,
            # the join of the 4 cores' 8x8 sub-blocks is un-panelized to row-major on the way to the shim
            # (packed C objects: row-major already, the columns past n_out dropped: a partial read of each
            # core's segment of the join)
            # (whole-line windows: win columns of each row, the rest from the next row's or the object's tail, ``lines``)
            # (two bfp16 parts: [part][group][panels] to the groups' rows of both parts' panels)
            c_dims=(stream_walk(c_l1, [(m // 8, no // 8 * 36), (2, m // 8 * no // 8 * 36), (no // 8 * 36, 1)])
                    if geo.c_parts == 2 else
                    stream_walk(c_l1, [(m, geo.n_c), (geo.win, 1)], [(0, 0), (0, 0)])
                    if geo.n_c != geo.win else
                    None if cb or geo.c_pack or geo.c_pairs else
                    stream_walk(c_l1, [(m // r, r * no), (r, t), (no // t, r * t), (t, 1)])))

    def _hg_bds(self, c: int, j0: int, np_: int) -> list[TensorAccessPattern]:
        """``hg`` > 1: the weight fill of passes j0 .. (row tiles j0 + tiles p) as BDs [np_ (passes), 3 dims]: per
        header group its unit's two 16-row sub-units ([tile][chunks 2 su, 2 su + 1][1 KB]; a group of fewer than 4
        chunks reads its last tile's header and first codes into the unused slots), then per chunk its codes
        ([sub-unit 4][tile][2 KB] after the header)."""
        geo = self.geo
        tw, (nsl, sw) = geo.tile_bytes // 2, geo.fill_units
        nt, g, rs, n_rt = geo.tiles, geo.kc_group, geo.rt_stride, geo.region_rt
        base = c * geo.col_bytes // 2
        w_dims = (1, geo.w_bytes // 2)
        hw = geo.slice // 2 // geo.hg  # (a tile's header, bf16 units)

        def at(k: int) -> int:  # tile (j0, k)
            return base + ((k // g) * n_rt * rs + j0 * rs + k % g) * tw

        out = []
        for k0, r in geo.hdr_groups:
            for su in range(2):
                qa, qb = min(2 * su, r - 1), min(2 * su + 1, r - 1)
                step = at(k0 + qb) - at(k0 + qa) if qb > qa else hw  # (no zero strides in a shim BD)
                out.append(TensorAccessPattern(w_dims, at(k0 + qa), [np_, nt, 2, hw], [nt * rs * tw, rs * tw, step, 1]))
            out += [TensorAccessPattern(w_dims, at(k) + hw, [np_, nsl, nt, sw], [nt * rs * tw, sw, rs * tw, 1])
                    for k in range(k0, k0 + r)]
        return out

    def b_taps(self, c: int) -> list[tuple[int, TensorAccessPattern]]:
        """(first pass, tap) of the weight fills of column c in bf16 units (one fill for all passes of a
        block where one pattern covers them): per pass and K chunk the header units and the super-block
        slices of the pass's tiles."""
        geo = self.geo
        tw, (nsl, sw) = geo.tile_bytes // 2, geo.fill_units
        nt, nk, g, rs = geo.tiles, geo.n_kc, geo.kc_group, geo.rt_stride
        # units start hdr_skip bytes before each tile (a header shorter than a slice: see __init__); the
        # W argument starts hdr_skip bytes before the matrix
        base = c * geo.col_bytes // 2
        w_dims = (1, geo.w_bytes // 2)
        n_rt = geo.region_rt  # row tiles of the column
        out: list[tuple[int, TensorAccessPattern]] = []
        p0 = 0
        for rt0, np_ in geo.blocks:
            if geo.hg > 1:  # (per pass: the header groups' units and the chunks' codes, ``_hg_bds``)
                for p_ in range(p0, p0 + np_):
                    out += [(p_, t) for t in self._hg_bds(c, geo.pass_rt[p_], 1)]
            elif nk == 1:
                out.append((p0, TensorAccessPattern(w_dims, base + rt0 * rs * tw, [np_, nsl, nt, sw],
                                                    [nt * rs * tw, sw, rs * tw, 1])))
            elif g == nk and tw == nsl * sw:  # (chunk, unit) as one dimension: tiles of whole slices, no padding
                out.append((p0, TensorAccessPattern(w_dims, base + rt0 * rs * tw, [np_, nk * nsl, nt, sw],
                                                    [nt * rs * tw, sw, rs * tw, 1])))
            else:  # per pass (more than the 3 dimensions + repeat of a shim BD): tile (j, k) of row tile j and
                # chunk k is at (k // g) n_rt rs + j rs + k % g
                for p_ in range(p0, p0 + np_):
                    j0 = geo.pass_rt[p_]
                    if g in (1, nk):  # the chunks at one stride: one fill per pass
                        sk = tw if g == nk else n_rt * rs * tw
                        out.append((p_, TensorAccessPattern(w_dims, base + j0 * rs * tw, [nk, nsl, nt, sw],
                                                            [sk, sw, rs * tw, 1])))
                    else:
                        out += [(p_, TensorAccessPattern(w_dims, base + (G * n_rt + j0) * rs * tw, [g, nsl, nt, sw],
                                                         [tw, sw, rs * tw, 1])) for G in range(nk // g)]
            p0 += np_
        return out

    def b_chain(self, c: int) -> list[list[TensorAccessPattern]] | None:
        """The weight fill of column c as one chained task per block, when b_taps needs fills per pass:
        one BD per K chunk, each 4-dimensional [pass (BD iteration), unit, tile, words] (the task repeats
        once per pass of the block and every BD of the chain advances its iteration then), or None."""
        geo = self.geo
        if geo.hg > 1:
            assert max(n for _, n in geo.blocks) <= MAX_BD_ITER, "a BD iterates at most MAX_BD_ITER times"
            chains = [self._hg_bds(c, rt0, np_) for rt0, np_ in geo.blocks]
            assert sum(map(len, chains)) <= MAX_CHAIN_BDS, "chained weight fills of more than MAX_CHAIN_BDS shim BDs"
            return chains
        taps = self.b_taps(c)
        if (len(taps) == len(geo.blocks) and geo.k_rep == 1) or max(n for _, n in geo.blocks) > MAX_BD_ITER:
            return None  # (one fill per block already; a BD iterates at most MAX_BD_ITER times)
        tw, (nsl, sw) = geo.tile_bytes // 2, geo.fill_units
        nt, nk, g, rs = geo.tiles, geo.n_kc, geo.kc_group, geo.rt_stride
        base = c * geo.col_bytes // 2
        n_rt = geo.region_rt
        w_dims = (1, geo.w_bytes // 2)
        # chunk k: tile (j, k) at (k // g) n_rt rs + j rs + k % g; pass p's tiles j = rt0 + nt p ..
        return [[TensorAccessPattern(w_dims, base + ((k // g) * n_rt * rs + rt0 * rs + k % g) * tw, [np_, nsl, nt, sw],
                                     [nt * rs * tw, sw, rs * tw, 1]) for k in list(range(nk)) * geo.k_rep]
                for rt0, np_ in geo.blocks]

    def a_tap(self, ch: int, row: int) -> TensorAccessPattern:
        """Core row ``row``'s A strip of token chunk ``ch`` (bf16 units): once (resident, resent by
        the memtile) or once per pass."""
        geo = self.geo
        m, K, aK, df = geo.m, geo.KA, geo.a_K, self
        n = 1 if df.a_res else geo.passes
        # panels [group][k8][36]: read [kt][group][8 k-steps x panel]
        return TensorAccessPattern((geo.M * aK * 9 // 16,),
                                   (ch * geo.nrows + row) * (m // 8) * (aK // 8) * 36 + geo.a_k0 // 8 * 36,
                                   [n, K // KT, m // 8, 8 * 36], [0, 8 * 36, aK // 8 * 36, 1])

    def c_chain(self, ch: int, c: int) -> list[TensorAccessPattern]:
        """Row groups: the C drain of column c for token chunk ``ch`` as one chained task, a BD per C object of a pass
        (its slots' rows of the pass's tiles: [token][tile G R apart][slots R]), each iterating over the passes (bf16
        units; the passes' outputs evenly spaced)."""
        geo = self.geo
        m, rows, G, R = geo.m, geo.nrows, geo.G, geo.R
        cols = geo.out_cols[c]
        step = cols[1] - cols[0] if len(cols) > 1 else G * geo.TR
        assert all(b - a == step for a, b in itertools.pairwise(cols)) and geo.passes <= MAX_BD_ITER, cols
        assert (geo.slots * R) % 2 == 0 and (G * R) % 2 == 0, "C runs of whole 32-bit words"
        return [TensorAccessPattern((geo.M, geo.N), ch * rows * m * geo.N + cols[0] + geo.slots * R * v,
                                    [geo.passes, rows * m, geo.tiles, geo.slots * R], [step, geo.N, G * R, 1])
                for v in range(geo.vp)]

    def c_out_cols(self, c: int) -> list[int]:
        """The first output column of each C object of column c (``out_cols``; two per pass in halves; of two parts:
        one per pass, or per pair of passes)."""
        geo = self.geo
        if geo.c_parts == 2:
            return list(geo.out_cols[c]) if geo.halves == 2 else list(geo.out_cols[c][0::2])
        return [o + geo.n_out * h for o in geo.out_cols[c] for h in range(geo.halves)]

    def c_taps(self, ch: int, c: int) -> list[TensorAccessPattern]:
        """The C drains of column c for token chunk ``ch`` (bf16 units): one per run of evenly
        spaced C objects' outputs."""
        geo = self.geo
        m, no, rows = geo.m, geo.n_out * geo.c_parts, geo.nrows
        cols = self.c_out_cols(c)
        if geo.carry is not None:  # (whole-line windows from each pass's line boundary)
            cols, no = [o - cr for o, (cr, _) in zip(cols, geo.carry, strict=True)], geo.win
        taps = []
        for p0, cnt, step in drain_runs(cols):
            if geo.c_bfp16:  # panels [group][N / 8][36]: [pass][the chunk's groups][no / 8 blocks][panel]
                taps.append(TensorAccessPattern((geo.M * geo.N * 9 // 16,),
                                                (ch * rows * m // 8 * geo.N + cols[p0]) // 8 * 36,
                                                [cnt, rows * m // 8, no // 8, 36],
                                                [step // 8 * 36, geo.N // 8 * 36, 36, 1]))
            else:
                taps.append(TensorAccessPattern((geo.M, geo.N), ch * rows * m * geo.N + cols[p0],
                                                [1, cnt, rows * m, no], [0, step, geo.N, 1]))
        return taps


def drain_runs(cols: list[int]) -> list[tuple[int, int, int]]:
    """Split pass output columns into runs (first pass, count, column step) of at most MAX_BD_ITER (a drain BD's iteration
    count)."""
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(cols):
        step = cols[i + 1] - cols[i] if i + 1 < len(cols) else 1
        j = i + 1
        while j < len(cols) and cols[j] - cols[j - 1] == step and step > 0 and j - i < MAX_BD_ITER:
            j += 1
        out.append((i, j - i, max(step, 1)))
        i = j
    return out
