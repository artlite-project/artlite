"""GemmQOperator's data movement on the host (``dma_model``; no NPU, no compile).

Uniquely numbered weights, activations and outputs go through the operator's own
shim taps and memtile patterns (``GemmQOperator.dataflow`` / ``b_taps`` / ``a_tap``
/ ``c_taps``); every object a core receives, and every output element, is compared
with the layout the kernels expect (module docstring of gemm/gemm_q.py).
"""

import numpy as np
import pytest

from artlite.formats import FORMATS
from artlite.gemm.gemm_q import COLS, KT, GemmQOperator
from artlite.gemm.gemm_qd import GemmQdOperator

from .. import dma_model as dm

TILE_BYTES = {"Q4_0": 18496, "Q4_1": 20480, "Q4_K": 18496, "BFP16": 18432}


def _op(fmt: str, *, M: int, K: int, passes: int, nrows: int = 4, silu: bool = False,
        cb: bool = False) -> GemmQOperator:
    """Tiles of 2048 columns, a column's K chunks outer (``kc_group`` 1: chunk kc's row tiles after chunk kc - 1's)."""
    R = 8 if fmt == "BFP16" else 16
    S = TILE_BYTES[fmt]
    tiles = 64 // R
    n_kc = K // 2048
    col_bytes = max(n_kc, 1) * passes * tiles * S
    step = 32 if silu else 64
    N = COLS * passes * step
    out_cols = [[c * passes * step + step * p for p in range(passes)] for c in range(COLS)]
    return GemmQOperator(fmt=fmt, rows=R, M=M, K=K, N=N, tile_bytes=S, col_bytes=col_bytes, passes=passes,
                         out_cols=out_cols, kc_group=1 if n_kc > 1 else 0, silu=silu, nrows=nrows, c_bfp16=cb)


def _op3b(*, M: int, K: int, passes: int, kc_group: int = 0, silu: bool = False, fmt: str = "Q4_K") -> GemmQOperator:
    """Q4_K or Q4_0 tiles of 1024 columns (Llama: 9280-B objects, the header half a slice), a row tile's chunks
    consecutive (or in groups of kc_group)."""
    S, n_kc = 9280, K // 1024
    col_bytes = n_kc * passes * 4 * S
    step = 32 if silu else 64
    N = COLS * passes * step
    out_cols = [[c * passes * step + step * p for p in range(passes)] for c in range(COLS)]
    return GemmQOperator(fmt=fmt, rows=16, M=M, K=K, N=N, tile_bytes=S, col_bytes=col_bytes, passes=passes,
                         out_cols=out_cols, silu=silu, c_bfp16=silu, kc=1024, kc_group=kc_group)


def _op_planar(fmt: str, *, R: int, S: int, kc: int, M: int, K: int, tiles: int, blocks: tuple[tuple[int, int], ...],
               region_rt: int, kc_group: int = 0, silu: bool = False, mac_rows: int = 3, m: int = 64) -> GemmQdOperator:
    """GemmQd on planar tiles (``gemm_q.planar``): ``blocks`` of passes of ``tiles`` row tiles each, packed C (or the
    SiLU groups' outputs), a column's region of ``region_rt`` row tiles (a row tile's K chunks consecutive, or in
    groups of ``kc_group``)."""
    n_kc = K // kc
    g = kc_group or n_kc
    col_bytes = region_rt * n_kc * S
    passes = sum(n for _, n in blocks)
    w = 64 if silu else tiles * R + tiles * R % 2  # C columns per pass
    pitch = passes * w
    out_cols = [[c * pitch + w * p for p in range(passes)] for c in range(COLS)]
    return GemmQdOperator(fmt=fmt, rows=R, M=M, K=K, N=COLS * pitch, tile_bytes=S, col_bytes=col_bytes, passes=passes,
                          out_cols=out_cols, silu=silu, c_bfp16=silu, kc=kc, kc_group=g,
                          tiles=tiles, c_pack=not silu, region_rt=region_rt, blocks=blocks, mac_rows=mac_rows, m=m)


def _tile(op: GemmQOperator, c: int, p: int, kc: int, t: int) -> int:
    """Index (in its column) of tile t of pass p, K chunk kc (a row tile's chunks in groups of
    ``kc_group``: (kc // g) n_rt g + j g + kc % g)."""
    g, j = op.kc_group, op.pass_rt[p] + t
    return (kc // g) * op.region_rt * g + j * g + kc % g


def _expected_b(op: GemmQOperator, c: int, half: int = 0) -> list[np.ndarray]:
    """Word indices (into W) of the L1 weight objects of one token chunk, in consumption order (dequantized tile-major
    tiles: those of the cores of ``half``, its ``tq`` tiles)."""
    words = []
    # (the W argument starts hdr_skip bytes before the matrix: unit sl of a tile is at its start -
    # hdr_skip + sl slices in the matrix)
    col = c * op.col_bytes // 4
    tw = op.tile_bytes // 4
    fmt = op.fmt

    def units(p: int) -> list[list]:
        """Per memtile object of pass p: per tile the word address of each gathered unit row (``None``: the rows
        of a slice from its start; hg > 1: a header unit's rows of the group's chunks, ``GemmQOperator.hg``)."""
        out = []
        if op.hg == 1:
            for kc in range(op.n_kc):
                for sl in range(op.hdr_units + op.nsb):
                    out.append([col + _tile(op, c, p, kc, t) * tw + sl * op.slice // 4 for t in range(op.tiles)])
            return out
        hw = op.slice // 4 // op.hg  # (a tile's header, words)
        for k0, r in op.hdr_groups:
            rows = []
            for t in range(op.tiles):
                start = []
                for u in range(op.R):  # unit row u: chunk slot q = u / 8, row u % 8 of its header
                    q, su = u // 8, u // 16
                    qa, qb = min(2 * su, r - 1), min(2 * su + 1, r - 1)
                    if q % 2 == 0 or qb == qa:
                        a = col + _tile(op, c, p, k0 + qa, t) * tw + (hw if q % 2 and qb == qa else 0)
                    else:
                        a = col + _tile(op, c, p, k0 + qb, t) * tw
                    start.append(a + (u % 8) * 32)
                rows.append(start)
            out.append(rows)
            for kc in range(k0, k0 + r):
                for sl in range(op.nsb):
                    out.append([col + _tile(op, c, p, kc, t) * tw + hw + sl * op.slice // 4 for t in range(op.tiles)])
        return out

    for p in range(op.passes):
        for base in units(p):
            if op.units:  # (super-block units: one object of the pass's tiles' units, each padded to whole lines)
                uw, pw = op.units // 4, op.unit_pitch // 4
                words.append(np.concatenate([np.r_[base[t] + np.arange(uw), np.zeros(pw - uw, np.int64)]
                                             for t in range(op.tiles)]))
                continue
            for kt in range(4):
                if op.planar and FORMATS[fmt].word_major:  # [word][row g]: word 2 P kt + i of tile row g
                    R, P = op.R, FORMATS[fmt].planes
                    rows = [(g // R, g % R) for g in range(op.n_read)] + [(None, 0)] * (op.n - op.n_read)
                    w = [0 if t is None else base[t] + (2 * P * kt + i) * R + r
                         for i in range(2 * P) for t, r in rows]
                elif op.planar:  # [plane][word ii][row g]: word 2 kt + ii of plane pl of tile row g; padding rows 0
                    R, P, pm = op.R, FORMATS[fmt].planes, FORMATS[fmt].plane_major
                    if op.us > 1:  # (sub-units: gathered rows [sub-unit][tile][row 16])
                        rows = [((g % (op.tiles * 16)) // 16, g // (op.tiles * 16) * 16 + g % 16)
                                for g in range(op.n_read)]
                    else:
                        rows = [(g // R, g % R) for g in range(op.n_read)]
                    rows += [(None, 0)] * (op.n - op.n_read)
                    at = [b if isinstance(b, list) else [b + r * P * 8 for r in range(R)] for b in base]  # (unit rows)
                    w = [0 if t is None else (base[t] + (pl * R + r) * 8 if pm else at[t][r] + pl * 8) + 2 * kt + ii
                         for pl in range(P) for ii in range(2) for t, r in rows]
                elif fmt == "BFP16":  # [16-row block][k8][8-row half][panel of 18 words]
                    w = [base[2 * b + h] + (8 * kt + q) * 18 + e
                         for b in range(4) for q in range(8) for h in range(2) for e in range(18)]
                else:  # [tile][i % 2][s][row 16]: word (s, i = 2 kt + i % 2) of each row
                    w = [base[t] + r * 32 + s * 8 + 2 * kt + il
                         for t in range(half * op.tq, (half + 1) * op.tq)
                         for il in range(2) for s in range(4) for r in range(16)]
                words.append(np.array(w))
    return words


def _expected_a(op: GemmQOperator, ch: int, row: int) -> list[np.ndarray]:
    """Element indices (bf16 units into A) of one pass's L1 A objects: MAC pop order
    [band][k-step][half][8 x 8 panel]."""
    m, K = op.m, op.K
    out = []
    for kt in range(K // KT):  # panels [token group][k8][36 units]
        g0 = (ch * op.nrows + row) * (m // 8)
        out.append(np.array([((g0 + 2 * band + h) * (K // 8) + 8 * kt + q) * 36 + u
                             for band in range(m // 16) for q in range(8) for h in range(2) for u in range(36)]))
    return out


CASES = [
    ("Q4_0", {"M": 256, "K": 2048, "passes": 3}),
    ("Q4_0", {"M": 512, "K": 2048, "passes": 2, "silu": True, "cb": True}),
    ("Q4_0", {"M": 256, "K": 8192, "passes": 2}),  # down: A refetched per pass in K chunks
    ("Q4_0", {"M": 64, "K": 2048, "passes": 2, "nrows": 2}),
    ("Q4_1", {"M": 256, "K": 2048, "passes": 2, "silu": True, "cb": True}),
    ("Q4_K", {"M": 256, "K": 8192, "passes": 2}),
    ("BFP16", {"M": 256, "K": 2048, "passes": 2, "silu": True, "cb": True}),
    ("Q4_K", {"M": 64, "K": 2048, "passes": 2, "nrows": 2}),
]


CASES_3B = [
    {"M": 256, "K": 3072, "passes": 6},  # a row tile's chunks consecutive, per-pass fills
    {"M": 512, "K": 3072, "passes": 3, "silu": True},
    {"M": 256, "K": 8192, "passes": 2, "kc_group": 2},  # down: the decode's activation objects
]

# Q4_0 tiles of 1024 columns (Llama 1B Q4_0: the header half a slice, as Q4_K's)
CASES_Q40_1K = [
    {"M": 256, "K": 2048, "passes": 3},
    {"M": 1024, "K": 2048, "passes": 2, "silu": True},
]

ALL = CASES + [("Q4_K-3B", kw) for kw in CASES_3B] + [("Q4_0-1K", kw) for kw in CASES_Q40_1K]


@pytest.mark.parametrize("fmt,kw", ALL, ids=[f"{f}-" + "-".join(f"{k}{v}" for k, v in kw.items()) for f, kw in ALL])
def test_gemm_q_dataflow(fmt: str, kw: dict) -> None:
    op = _op3b(**kw) if fmt == "Q4_K-3B" else _op3b(**kw, fmt="Q4_0") if fmt == "Q4_0-1K" else _op(fmt, **kw)
    df = op.dataflow()
    dm.check_memtile_object(df.b_obj * 4, df.b_wdims)
    dm.check_memtile_object(df.a_obj * 2, df.a_wdims, repeated=df.a_repeat > 1)

    _check_weights(op, df)
    _check_activations(op, df)
    _check_outputs(op, df)


def _check_weights(op: GemmQOperator, df: object) -> None:
    """Weights: column 0 and the last one (the column base offset matters)."""
    w_units = np.arange(op.w_bytes // 2, dtype=np.int64)
    for c in (0, COLS - 1):
        stream = np.concatenate([dm.read(w_units, tap) for _, tap in op.b_taps(c)])
        for _, tap in op.b_taps(c):
            dm.check_shim_tap(tap, 2)
        chains = op.b_chain(c)
        if chains is not None:  # one task per block: [repeat (each BD's iteration)][BD of the chain][3 dims]
            parts = []
            for chain in chains:
                reps = chain[0].sizes[0]
                parts.append(np.concatenate([dm.read(w_units, t).reshape(reps, -1) for t in chain], axis=1).reshape(-1))
            assert (np.concatenate(parts) == stream).all(), "the chained fill differs"
        pairs = stream.reshape(-1, 2)
        assert (pairs[:, 1] == pairs[:, 0] + 1).all() and (pairs[:, 0] % 2 == 0).all(), "weights move in words"
        # the memtile object in b_parts segments, each written with b_wdims; segment h read to the cores of half h
        nparts = df.b_parts
        segs = dm.receive(pairs[:, 0] // 2, df.b_obj // nparts, df.b_wdims)
        for h in range(nparts):
            l1 = dm.receive(dm.send(segs[h::nparts], df.b_rdims), df.b_l1)
            want = _expected_b(op, c, h)
            assert len(l1) == len(want)
            for i, (got, exp) in enumerate(zip(l1, want)):
                assert (got == exp).all(), f"column {c}, half {h}: weight object {i} differs"


def _check_activations(op: GemmQOperator, df: object) -> None:
    """Every core row of every token chunk, all passes."""
    a_units = np.arange(op.M * op.K * 9 // 16, dtype=np.int64)
    for ch in range(op.mc):
        for row in range(op.nrows):
            tap = op.a_tap(ch, row)
            dm.check_shim_tap(tap, 2)
            l2 = dm.receive(dm.read(a_units, tap), df.a_obj, df.a_wdims)
            l1 = dm.receive(dm.send(l2, None, repeat=df.a_repeat), df.a_l1)
            want = _expected_a(op, ch, row) * op.passes
            assert len(l1) == len(want), (len(l1), len(want))
            for i, (got, exp) in enumerate(zip(l1, want)):
                assert (got == exp).all(), f"chunk {ch} row {row}: A object {i} differs"


def _check_outputs(op: GemmQOperator, df: object) -> None:
    """Outputs: numbered per (chunk, pass, core row, element) in the kernel's C layout."""
    m, no, N = op.m, op.n_out, op.N
    c_len = op.M * N * 9 // 16 if op.c_bfp16 else op.M * N
    for c in (0, COLS - 1):
        C = np.full(c_len, -1, np.int64)
        expect = np.full(c_len, -1, np.int64)
        uid = 0
        for ch in range(op.mc):
            objs = []  # per core row, per C object (halves: two per pass)
            for row in range(op.nrows):
                objs.append([])
                for p, col0 in enumerate(op.c_out_cols(c)):
                    ids = uid + np.arange(df.c_l1)
                    uid += df.c_l1
                    objs[row].append(ids)
                    t0 = (ch * op.nrows + row) * m
                    cols = col0 + np.arange(no)
                    if op.c_bfp16:  # panels [group][col block][36 units]
                        v = ids.reshape(m // 8, no // 8, 36)
                        for g in range(m // 8):
                            for j in range(no // 8):
                                dst = ((t0 // 8 + g) * (N // 8) + cols[8 * j] // 8) * 36
                                expect[dst:dst + 36] = v[g, j]
                    elif op.c_pack:  # row-major rows of n_c columns, the first n_out reach C
                        v = ids.reshape(m, op.n_c)[:, :no]
                        for tok in range(m):
                            expect[(t0 + tok) * N + cols] = v[tok]
                    else:  # 8x8 row-major sub-blocks [token panel][col block][8][8]
                        v = ids.reshape(m // 8, no // 8, 8, 8).transpose(0, 2, 1, 3).reshape(m, no)
                        for tok in range(m):
                            expect[(t0 + tok) * N + cols] = v[tok]
            # (the memtile walks each core's segment of the join)
            stream = dm.send([seg for obj in dm.join(objs) for seg in np.split(obj, op.nrows)], df.c_dims)
            pos = 0
            for tap in op.c_taps(ch, c):
                dm.check_shim_tap(tap, 2)
                n = int(np.prod(tap.sizes))
                dm.write(C, tap, stream[pos:pos + n])
                pos += n
            assert pos == stream.size, "the drains consume the whole C stream"
        assert (C == expect).all(), f"column {c}: C layout differs at {np.flatnonzero(C != expect)[:4]}"


def test_walk_offsets_outer_loop() -> None:
    """An object longer than its walk: the outermost dimension keeps stepping; padding inserts -1."""
    from aie.helpers.taplib import TensorAccessPattern

    walk = TensorAccessPattern((32,), 0, [2, 2], [8, 1])
    assert dm.walk_offsets(walk, 8).tolist() == [0, 1, 8, 9, 16, 17, 24, 25]
    assert dm.walk_offsets(walk.pad([(0, 0), (0, 1)]), 4).tolist() == [0, 1, -1, 8, 9, -1]


def test_check_shim_tap() -> None:
    from aie.helpers.taplib import TensorAccessPattern

    dm.check_shim_tap(TensorAccessPattern((4096,), 0, [2, 2046], [2048, 1]), 2)
    dm.check_shim_tap(TensorAccessPattern((8192,), 0, [2, 4096], [4096, 1]), 2)  # contiguous: one linear run
    with pytest.raises(ValueError):
        dm.check_shim_tap(TensorAccessPattern((16384,), 0, [2, 4096], [8192, 1]), 2)


# planar tiles (GemmQd): (format, rows, tile bytes, columns, kwargs)
CASES_PLANAR = [
    ("Q5_K", 13, 18496, 2048, {"M": 192, "K": 2048, "tiles": 5, "blocks": ((0, 4), (20, 1), (25, 1)), "region_rt": 30}),
    ("Q5_K", 13, 18496, 2048, {"M": 192, "K": 2048, "tiles": 10, "blocks": ((0, 4),), "region_rt": 40, "silu": True}),
    ("Q5_K", 13, 9280, 1024, {"M": 64, "K": 3072, "tiles": 10, "blocks": ((0, 3),), "region_rt": 30, "silu": True,
                              "mac_rows": 1}),
    ("Q4_1", 14, 18496, 2048, {"M": 192, "K": 8192, "tiles": 4, "blocks": ((0, 5),), "region_rt": 19, "kc_group": 1}),
    ("Q4_1", 14, 9280, 1024, {"M": 128, "K": 8192, "tiles": 4, "blocks": ((0, 7),), "region_rt": 28, "kc_group": 2,
                              "mac_rows": 2}),
    ("Q6_K", 11, 9280, 1024, {"M": 192, "K": 3072, "tiles": 6, "blocks": ((0, 2),), "region_rt": 12}),
    # SiLU groups whose padding rows reach the MMUL (Q6_K's d chunks; a half-slice Q4 header): outputs dropped
    ("Q6_K", 11, 9280, 1024, {"M": 144, "K": 3072, "tiles": 12, "blocks": ((0, 2),), "region_rt": 24, "silu": True,
                              "m": 48}),
    ("Q4_1", 14, 9280, 1024, {"M": 96, "K": 3072, "tiles": 10, "blocks": ((0, 2),), "region_rt": 20, "silu": True,
                              "m": 48, "mac_rows": 2}),
    # word-major tiles ([word][row]: Q2_K, Q3_K; header units from before the tile, row-direct)
    ("Q2_K", 26, 9408, 1024, {"M": 192, "K": 3072, "tiles": 3, "blocks": ((0, 2), (6, 1)), "region_rt": 9}),
    ("Q2_K", 26, 9408, 1024, {"M": 192, "K": 3072, "tiles": 5, "blocks": ((0, 2),), "region_rt": 10, "silu": True}),
    ("Q3_K", 21, 9408, 1024, {"M": 192, "K": 8192, "tiles": 4, "blocks": ((0, 3),), "region_rt": 12, "kc_group": 2}),
    ("Q3_K", 22, 20160, 2048, {"M": 128, "K": 2048, "tiles": 6, "blocks": ((0, 2),), "region_rt": 12, "silu": True,
                               "mac_rows": 2}),
    ("Q3_K", 22, 20160, 2048, {"M": 64, "K": 2048, "tiles": 3, "blocks": ((0, 3),), "region_rt": 9, "mac_rows": 1}),
    # Gemma's Q4_0 tiles of 32 rows x 512 columns: 16-row sub-units, a header unit per 4 K chunks (groups of 4 and 1)
    ("Q4_0", 32, 9216, 512, {"M": 288, "K": 2560, "tiles": 2, "blocks": ((0, 5),), "region_rt": 10, "m": 96}),
    ("Q4_0", 32, 9216, 512, {"M": 288, "K": 2560, "tiles": 4, "blocks": ((0, 3),), "region_rt": 12, "silu": True,
                             "m": 96}),
    ("Q4_0", 32, 9216, 512, {"M": 288, "K": 4096, "tiles": 2, "blocks": ((0, 2),), "region_rt": 4, "kc_group": 4,
                             "m": 96}),
    ("Q4_0", 32, 9216, 512, {"M": 288, "K": 512, "tiles": 2, "blocks": ((0, 3),), "region_rt": 6, "m": 96}),
    # super-block units (Q8_0 of 1024 columns: 8-row tiles, 2176-byte units; IQ3_XXS: 2320 bytes, padded to 2336)
    ("Q8_0", 8, 9280, 1024, {"M": 192, "K": 2048, "tiles": 8, "blocks": ((0, 2), (16, 1)), "region_rt": 24}),
    ("Q8_0", 8, 9280, 1024, {"M": 64, "K": 3072, "tiles": 4, "blocks": ((0, 3),), "region_rt": 12, "mac_rows": 1}),
]


@pytest.mark.parametrize("fmt,R,S,kc,kw", CASES_PLANAR,
                         ids=[f"{f}-r{r}-kc{k}-" + "-".join(f"{a}{v}" for a, v in kw.items() if a != "blocks")
                              for f, r, _, k, kw in CASES_PLANAR])
def test_gemm_qd_planar_dataflow(fmt: str, R: int, S: int, kc: int, kw: dict) -> None:
    """The planar gather: [plane][row] k-tile objects of the pass's rows (any row count, zero-padded to the
    MMUL's n), blocks of passes, packed C rows (odd row counts: one column more), SiLU passes of whole groups."""
    op = _op_planar(fmt, R=R, S=S, kc=kc, **kw)
    assert op.planar
    df = op.dataflow()
    dm.check_memtile_object(df.b_obj * 4, df.b_wdims)
    dm.check_memtile_object(df.a_obj * 2, df.a_wdims, repeated=df.a_repeat > 1)
    _check_weights(op, df)
    _check_activations(op, df)
    _check_outputs(op, df)
