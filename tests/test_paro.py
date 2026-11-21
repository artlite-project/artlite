"""PARO checkpoints on the host (``artlite.paro``, ``artlite.hf_model``): the AWQ and fifth-bit unpacking against a
packer written here, a checkpoint's linears as ``ParoTensor`` (their fields, W_eff = W M^T against the rotation applied
to rows), the tan form of the rounds against the exact transform (and with a norm's gamma folded into D_0) on every
linear of a layer, the rows / groups in another order as index maps, the quantization_config checks, the native
transcode and P4_128 packer against the slot's tiles (no NPU)."""

import numpy as np
import pytest

from artlite.decode.image import pack_rows
from artlite.formats import pack_tile, slice_fields
from artlite.hf_model import HfModel
from artlite.model_config import checkpoint
from artlite.paro import ParoTensor, Rotation, awq_unpack, bitplane_unpack, check_config

from .model_files import require

ORDER = [0, 4, 1, 5, 2, 6, 3, 7]  # (AWQ: logical column j of a word in nibble ORDER[j])


def awq_pack(codes: np.ndarray) -> np.ndarray:
    """codes (rows, cols) 0 .. 15 -> AWQ int32 words (rows, cols / 8)."""
    c = codes.astype(np.uint32).reshape(codes.shape[0], -1, 8)
    w = np.zeros(c.shape[:2], np.uint32)
    for j in range(8):
        w |= c[..., j] << (4 * ORDER[j])
    return w.view(np.int32)


def bitplane_pack(bits: np.ndarray) -> np.ndarray:
    """bits (rows, cols) 0 / 1 -> words (rows, cols / 32): bit c % 32 of word c // 32."""
    b = bits.astype(np.uint64).reshape(bits.shape[0], -1, 32)
    return (b << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32).view(np.int32)


def random_paro(K: int, N: int, *, g: int = 128, bits: int = 4, seed: int = 0) -> tuple[ParoTensor, dict]:
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 1 << bits, (K, N))
    z = rng.integers(0, 1 << bits, (K // g, N))
    s = rng.uniform(0.001, 0.02, (K // g, N)).astype(np.float16)
    pairs = np.stack([np.concatenate([rng.permutation(g) for _ in range(K // g)]) for _ in range(8)])
    rot = Rotation(rng.uniform(-3, 3, (8, K // 2)), pairs, rng.uniform(-2, 2, K), g)
    hi = {"qweight_hi": bitplane_pack(q >> 4), "qzeros_hi": bitplane_pack(z >> 4)} if bits == 5 else {}
    t = ParoTensor(name="w", qtype=f"P{bits}_{g}", rows=N, cols=K, raw=awq_pack(q & 15), qweight=awq_pack(q & 15),
                   qzeros=awq_pack(z & 15), scales=s, group=g, rotation=rot, **hi)
    return t, {"q": q, "z": z, "s": s.astype(np.float32)}


@pytest.mark.parametrize("bits", [4, 5])
def test_unpack(bits: int) -> None:
    t, ref = random_paro(512, 256, bits=bits)
    assert np.array_equal(awq_unpack(awq_pack(ref["q"] & 15)), ref["q"] & 15)
    assert np.array_equal(bitplane_unpack(bitplane_pack(ref["q"] >> 4)), ref["q"] >> 4)
    f = t.fields()
    assert t.bits == bits
    assert np.array_equal(f["q"], ref["q"].T) and np.array_equal(f["z"], ref["z"].T) and np.array_equal(f["s"], ref["s"].T)
    w = (ref["q"].reshape(4, 128, -1) - ref["z"][:, None]) * ref["s"][:, None]
    assert np.array_equal(t.dequantize(), w.reshape(512, -1).T.astype(np.float32))


def test_permuted() -> None:
    """rows_of / groups_of: the same linear with its outputs and its groups (channels, codes, rotation) reordered."""
    t, _ = random_paro(512, 256)
    rows, groups = np.random.default_rng(1).permutation(256), np.array([2, 0, 3, 1])
    k = (groups[:, None] * 128 + np.arange(128)).reshape(-1)
    p = ParoTensor(**{**t.__dict__, "rows_of": rows, "groups_of": groups})
    assert np.array_equal(p.dequantize(), t.dequantize()[rows][:, k])
    assert np.allclose(p.effective(), t.effective()[rows][:, k], rtol=0, atol=1e-12)


def _linears(case: str) -> tuple[HfModel, list[str]]:
    require(checkpoint(case))
    hf = HfModel(checkpoint(case))
    return hf, [n for n, (_, q, _, _) in hf.tensor_info.items() if q.startswith("P4_") and n.startswith("blk.3.")]


@pytest.mark.parametrize("case", ["qwen3-0.6B-paro", "qwen3.5-0.8B-paro"])
def test_checkpoint_linears(case: str) -> None:
    """Every PARO linear of a layer: y = x W_eff^T is rotate(x) W^T; the tan form (with and without a gamma folded in)
    gives rotate(x) in storage order to fp64 rounding."""
    hf, names = _linears(case)
    assert names
    rng = np.random.default_rng(0)
    for n in names:
        t = hf.tensor(n)
        assert isinstance(t, ParoTensor) and t.qtype == "P4_128" and t.group == 128
        x = rng.standard_normal((4, t.cols))
        xr = t.rot().apply(x)
        y = xr @ t.dequantize().astype(np.float64).T
        assert np.allclose(x @ t.effective().T, y, rtol=0, atol=1e-9 * np.abs(y).max()), n
        gamma = rng.uniform(0.5, 2.0, t.cols)
        for gm in (None, gamma):
            tf = t.rot().tan_form(gm)
            want = (t.rot().apply(x * gm) if gm is not None else xr)[:, tf.chan_index()]
            got = tf.apply(x)
            assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max(), n
            assert tf.partner.dtype == np.uint8 and tf.partner.max() < 128
            assert np.array_equal(np.sort(tf.chan.reshape(-1, 128), 1), np.tile(np.arange(128), (t.cols // 128, 1)))


def test_tan_bounded() -> None:
    """|T| is at most the ratio of the pair's scales: the tangent's magnitude stays at most 1 (the swap)."""
    hf, names = _linears("qwen3-0.6B-paro")
    t = hf.tensor(names[0])
    r = t.rot()
    tf = Rotation(r.theta, r.pairs, np.ones(t.cols), 128).tan_form()
    assert np.abs(tf.T[0]).max() <= 1 + 1e-12  # (round 0 with unit scales: |T| = |tan| or |cot| <= 1)


def test_config_checks() -> None:
    assert check_config("x", {"bits": 4, "group_size": 128, "krot": 8}) == (4, 128)
    assert check_config("x", {"bits": 5, "group_size": 128, "format": "int5-bitplane"}) == (5, 128)
    for qc in ({"bits": 3}, {"bits": 4, "group_size": 32}, {"bits": 5}, {"bits": 4, "format": "mxfp4"}):
        with pytest.raises(NotImplementedError):
            check_config("x", qc)


@pytest.mark.parametrize("bits", [4, 5])
def test_pack_native(bits: int) -> None:
    """``ParoTensor.blocks`` (``_native.awq_rows``) and the native packer give the slot's tiles (``formats.awq.pack``:
    P4_128, int5 P5_128) of the fields: rows, groups and each group's channels in another order, 3 K chunks."""
    t, _ = random_paro(3072, 256, bits=bits, seed=2)
    rng = np.random.default_rng(3)
    t = ParoTensor(**{**t.__dict__, "rows": 48, "rows_of": rng.permutation(256)[:48], "groups_of": rng.permutation(24),
                      "chan": np.concatenate([rng.permutation(128) for _ in range(24)])})
    fmt, S = t.qtype, 9216 if bits == 4 else 11264
    tiles = pack_rows([t.blocks()], np.arange(48), fmt, 16, S, 3, kc=1024)
    f = t.fields()
    for g in range(3):
        for k in range(3):
            fc = slice_fields(f, np.arange(16 * g, 16 * g + 16), k0=1024 * k, cols=t.cols, kc=1024)
            assert np.array_equal(tiles[g, k], pack_tile(fmt, fc, S)), (g, k)


def test_tan_dead_channel() -> None:
    """A norm weight of zero (a dead channel, Qwen3.8-27B-PARO-int5's layer 7): the unit tan form zeroes that channel's
    input (w), the rounds as for a weight of 1, exactly the transform of x * gamma."""
    t, _ = random_paro(512, 64, seed=4)
    r = t.rot()
    gamma = np.random.default_rng(5).uniform(0.5, 2.0, 512)
    gamma[[3, 200, 511]] = 0.0
    tf = r.tan_form(gamma, unit=True)
    x = np.random.default_rng(6).standard_normal((3, 512))
    assert np.isfinite(tf.T).all() and np.isfinite(tf.w).all()
    want = r.apply(x * gamma)[:, tf.chan_index()]
    assert np.abs(tf.apply(x) - want).max() < 1e-12 * np.abs(want).max()
