"""The render manifest (operators/render_cache.py): keys from constructor arguments, a compile that finds its build
without rendering (no NPU)."""

import os

import numpy as np
import pytest

from artlite.operators import FusedOperator
from artlite.operators import render_cache as rc

from .elementwise import GeluOperator, SiluOperator, StreamOperator


def test_keys(request: pytest.FixtureRequest) -> None:
    """Equal constructor arguments: one key; other arguments, a shadowed method, a function argument: another or
    none."""
    if request.config.getoption("--dump-designs"):
        pytest.skip("--dump-designs")  # (its ArtcOperator.compile, a function of conftest: no operator has a key)
    a, b = SiluOperator(size=2048), SiluOperator(size=2048)
    assert rc._op_key(a) is not None and rc._op_key(a) == rc._op_key(b)
    assert rc._op_key(SiluOperator(size=4096)) != rc._op_key(a)
    assert rc._op_key(GeluOperator(size=2048)) != rc._op_key(a)
    c = SiluOperator(size=2048)
    c.get_kernel_src = lambda: ""  # (a variant build)
    assert rc._op_key(c) is None
    with pytest.raises(rc._NoKey):
        rc._describe(lambda: 0, set(), set())
    assert rc._describe({"x": np.arange(3)}, set(), set()) != rc._describe({"x": np.arange(1, 4)}, set(), set())

    def fused(ops: list[StreamOperator]) -> FusedOperator:
        return FusedOperator(name="t_rc", runlist=[(ops[0], "x", "y"), (ops[1], "y", "z")], input_args=["x"],
                             output_args=["z"])

    # one operator object twice (one design) vs two equal ones (two designs)
    assert rc._fused_key(fused([a, a])) != rc._fused_key(fused([a, b]))
    assert rc._fused_key(fused([a, b])) == rc._fused_key(fused([b, a]))


def test_hit() -> None:
    """A second compile of equal operators attaches the first's build without rendering (an ArtcOperator, a
    FusedOperator)."""
    first = SiluOperator(size=1024, tile_size=256).compile()
    f1 = FusedOperator(name="t_rc_hit", runlist=[(SiluOperator(size=1024, tile_size=256), "x", "y")],
                       input_args=["x"], output_args=["y"]).compile()
    again = SiluOperator(size=1024, tile_size=256)
    f2 = FusedOperator(name="t_rc_hit", runlist=[(SiluOperator(size=1024, tile_size=256), "x", "y")],
                       input_args=["x"], output_args=["y"])

    def no_render() -> None:
        raise AssertionError("rendered")

    assert rc._op_key(again) is not None  # (memoized: the shadowed build_spec does not void it)
    again.build_spec = f2.build_spec = no_render
    again.compile()
    f2.compile()
    assert again._program.pdi_path == first._program.pdi_path
    assert f2.elf_path == f1.elf_path
    assert all(os.path.exists(os.path.join(os.path.dirname(f2.elf_path), n)) for n in f2.artifacts)
