"""GemmSpec and Geometry on the host: the request's key, the derived sizes, the drain variants (no NPU, no compile)."""

import dataclasses

import pytest

from artlite.gemm.gemm_q import GemmQOperator
from artlite.gemm.gemm_qd import GemmQdOperator
from artlite.gemm.geometry import LINE, GemmSpec, Geometry, L1Plan, line_window
from artlite.operators.design import COLS


def _spec(**kw: object) -> GemmSpec:
    """Q4_K tiles of 1024 columns (Llama's), 4 passes of 64 outputs per column."""
    passes = kw.pop("passes", 4)
    base = {"fmt": "Q4_K", "rows": 16, "M": 256, "K": 2048, "N": COLS * passes * 64, "tile_bytes": 9280,
            "col_bytes": passes * 4 * 2 * 9280, "passes": passes, "kc": 1024,
            "out_cols": [[c * passes * 64 + 64 * p for p in range(passes)] for c in range(COLS)]}
    return GemmSpec(**{**base, **kw})


def _planar(**kw: object) -> GemmSpec:
    """Q6_K tiles (11 rows x 1024 columns) in passes of 5 tiles: 55 rows, packed C with whole-line drains."""
    passes = 4
    w = 56
    pitch = -(-(passes * w) // LINE) * LINE + LINE
    base = {"fmt": "Q6_K", "rows": 11, "M": 192, "K": 2048, "N": COLS * pitch, "tile_bytes": 9280,
            "col_bytes": 20 * 2 * 9280, "passes": passes, "kc": 1024, "tiles": 5, "c_pack": True,
            "blocks": ((0, passes),), "region_rt": 20, "lines": True,
            "out_cols": [[c * pitch + w * p for p in range(passes)] for c in range(COLS)]}
    return GemmSpec(**{**base, **kw})


def test_spec_is_a_key() -> None:
    a = _spec()
    b = GemmSpec(**{**dataclasses.asdict(a), "out_cols": [list(c) for c in a.out_cols]})
    assert a == b and hash(a) == hash(b)  # (lists and tuples: one spec)
    assert a != _spec(silu=True, c_bfp16=True) and _spec(M=512) != a
    with pytest.raises(TypeError):
        GemmSpec(**{**dataclasses.asdict(a), "kc_gruop": 2})  # (a misspelled option)
    assert GemmSpec(**a.operator_kwargs()) == a


def test_variants_plain_and_lines() -> None:
    spec = _planar()
    plain, lines = Geometry.variants(spec, dq_core=True)
    assert lines is not None and plain.carry is None and lines.carry is not None
    assert plain.win == plain.n_out and lines.win == line_window(lines.carry, lines.n_out)
    assert lines.n_c >= plain.n_c and lines.c_tail == lines.win - lines.n_c
    assert plain.m == lines.m == 0
    # (everything else is the same geometry)
    keep = [f.name for f in dataclasses.fields(Geometry) if f.name not in ("carry", "n_c", "win", "c_parts")]
    assert all(getattr(plain, k) == getattr(lines, k) for k in keep)
    assert Geometry.variants(_planar(lines=False), dq_core=True)[1] is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        plain.n = 1  # type: ignore[misc]
    assert plain.with_m(48).mc == 192 // (plain.nrows * 48) and plain.m == 0


def test_geometry_of_tile_major_tiles() -> None:
    g, lines = Geometry.variants(_spec(), dq_core=False)
    assert lines is None and not g.planar and g.pair and g.tiles == 4 and g.TR == 64 and g.n == g.n_out == 64
    assert g.pass_rt == (0, 4, 8, 12) and g.n_kc == 2 and g.kc_group == 2 and g.hdr_skip == 1024
    assert g.w_bytes >= COLS * g.col_bytes
    with pytest.raises(AssertionError):  # (planar tiles: GemmQd's)
        Geometry.variants(_planar(), dq_core=False)


def test_operators_read_their_geometry() -> None:
    q = GemmQOperator(**_spec().operator_kwargs())
    assert isinstance(q.geo, Geometry) and q.plan == L1Plan() and q.m in (64, 48, 32, 16)
    assert (q.n, q.n_out, q.passes, q.mc, q.w_bytes) == (q.geo.n, q.geo.n_out, q.geo.passes, q.geo.mc, q.geo.w_bytes)
    qd = GemmQdOperator(mac_rows=3, m=64, **_planar().operator_kwargs())
    assert qd.geo.m == 64 and qd.nrows == 3 and qd.M == 192 and qd.T == 192
    assert (qd.c_depth, qd.sxp, qd.b_depth) == (qd.plan.c_depth, qd.plan.sxp, qd.plan.b_depth)
    assert qd.flow.geo is qd.geo and qd.flow.a_res == (qd.geo.m * qd.geo.KA * 9 // 8 <= 448 * 1024)
