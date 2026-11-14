"""The vision ops' host side (no NPU): the walks' literals, the checks that refuse a configuration that would hang or
read the wrong bytes, the roles."""

import numpy as np
import pytest

from artlite.txn import HardwareLimitError
from artlite.vision import design as D
from artlite.vision.ops import ConvSpec, Gemm, Out, WinAttention, mac_walks
from artlite.vision.ops import common as C
from artlite.vision.roles import ROLES, SPECS, Role
from artlite.vision.seq import RowOps


def test_walks() -> None:
    """The MAC's walks are the literal words ``core``'s kernels were written against."""
    assert mac_walks(64).tolist() == [1, 1152, 4, -3456, -2304, 0, 0, 1, -1152, 4, 0, -5760, 0, 0,
                                      1, 2560, 4, -2432, -1792, 0, 0]
    assert mac_walks(16).tolist() == [4, -1152, 0, 0, 0, 0, 0, 4, 0, 0, -5760, -5760, 0, 0, 4, 128, 0, 768, 768, 0, 0]
    assert C.PATS_4 == (1, -1152, 4, 0, -5760, 0, 0, 1, 2560, 4, -2432, -1792, 0, 0)


def test_b_objects() -> None:
    """A row's B objects are counted against its runs' bytes: a count off by one is refused."""
    RowOps(runs=[(0, D.Q), (D.Q, 3 * D.Q)], b_objects=4)
    with pytest.raises(HardwareLimitError, match="off by one"):
        RowOps(runs=[(0, D.Q), (D.Q, 3 * D.Q)], b_objects=3)


def test_roles() -> None:
    """A role is a ``Role`` or None; a mistyped one is refused where the op is built."""
    assert RowOps(runs=[], b_objects=0, role="gn").role is Role.GN
    with pytest.raises(ValueError):
        RowOps(runs=[], b_objects=0, role="gx")
    assert [r for r in ROLES if SPECS[r].clobbers_table] == [Role.C]


def test_memtile() -> None:
    """Tensors and walks past the memtile are refused with a message that names them."""
    Gemm(m=64, ntile=1, K=640, N=640, x=0x4000, y=0x20000)
    with pytest.raises(HardwareLimitError, match="output tensor"):
        Gemm(m=64, ntile=1, K=640, N=640, x=0x4000, y=D.HEAP - 1024)
    with pytest.raises(HardwareLimitError, match="residual"):
        Gemm(m=64, ntile=1, K=640, N=640, x=0x4000, y=0x20000, res=C.MEMTILE - 4096, out=Out.BF16)


def test_gemm_options() -> None:
    """Output kinds take their shapes only; an option another kind owns is refused."""
    base = {"m": 64, "ntile": 1, "K": 640, "N": 640, "x": 0x4000, "y": 0x20000}
    assert Gemm(**base, out="gelu").out is Out.GELU
    with pytest.raises(ValueError):
        Gemm(**base, out="gelu2")
    with pytest.raises(AssertionError, match="out rope: res required"):
        Gemm(**base, out="rope")
    with pytest.raises(AssertionError, match="no norm weights"):
        Gemm(**base, out="bf16", qkn=np.zeros(80, np.float32))
    with pytest.raises(AssertionError, match="act silu"):
        Gemm(**base, out="bf16", act="silu")


def test_conv() -> None:
    """A conv's tile geometry is checked against its GEMM; its tap words are packed once, per variant."""
    cv = ConvSpec(stride=1, R=1, Lg=4, zero=0x60000, rc=4, r0=1)
    g = Gemm(m=64, ntile=2, K=128, N=320, x=0x4000, y=0x20000, conv=cv)
    assert g.nkt == 18 and g.conv.tile_word(0, 2) == -0x55555556
    with pytest.raises(AssertionError, match="conv: m 16"):
        Gemm(m=16, ntile=1, K=128, N=320, x=0x4000, y=0x20000, conv=cv)
    with pytest.raises(AssertionError, match="rc: the tiles' rows"):
        Gemm(m=64, ntile=4, K=128, N=320, x=0x4000, y=0x20000, conv=cv)


def test_win_attention() -> None:
    """A window attention's KV heads' stride comes from its image's units: none is not a zero stride."""
    at = WinAttention(q=0, o=0, mask=0, kv=0, image_units=8)
    assert at.kv_head_bytes == WinAttention.kv_image(8) // 4 > 0
    with pytest.raises(AssertionError, match="image"):
        WinAttention(q=0, o=0, mask=0, kv=0, image_units=0)
