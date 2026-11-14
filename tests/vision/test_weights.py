"""The vision overlay's weights (artlite.vision.weights, no NPU): ``Wt``'s maps against torch's operations on the
values, and the native packer (``Slots`` -> ``_native.pack_slots``) bit for bit against a numpy reference of the B
slots (``slot_ref``) for every source type, row and column maps, scales, GEGLU's rows, hi + lo planes, the convs' tap
orders, several parts, the identity, and the model files' sources (GGUF, safetensors)."""

import json
import struct

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.operators import bfp16
from artlite.vision import design as D
from artlite.vision import weights as W
from artlite.vision.ops import ConvSpec, Gemm

ROWS = 4


def slot_ref(g: Gemm, w: np.ndarray, row: int) -> np.ndarray:
    """Core row ``row``'s weight image of a Gemm (w: its float32 values (N, K) or a conv's (N, K, 3, 3)): its rows
    (``Gemm.weight_rows``), a conv's k-tiles per chunk of 64 input channels its taps (``ConvSpec.tap_order``), K padded to whole
    k-tiles; per pass its k-tiles [16-row block 5][k-step 8][half 2][row 8][9 B] (hilo: the hi planes' k-tiles, then
    the lo ones: lo = bfp16(w - hi))."""
    ws = np.asarray(w, np.float32)[g.weight_rows(row)]
    if g.conv:
        t = ws.reshape(-1, g.nch, 64, 3, 3).transpose(0, 1, 3, 4, 2)  # (n, chunk, dy, dx, k)
        ws = np.stack([np.stack([t[:, c, dy, dx] for dy, dx in g.conv.tap_order(c)], 1) for c in range(g.nch)],
                      1).reshape(ws.shape[0], -1)
    half = (g.nkt // 2 if g.hilo else g.nkt) * D.KT
    wp = np.zeros((ws.shape[0], half), np.float32)
    wp[:, :ws.shape[1]] = ws
    planes = [bfp16.block_bytes(wp.reshape(wp.shape[0], -1, 8))]
    if g.hilo:
        lo = wp - bfp16.block_values(planes[0]).reshape(wp.shape)
        planes.append(bfp16.block_bytes(lo.reshape(wp.shape[0], -1, 8)))
    nkt = half // D.KT
    out = [b.reshape(g.npo, 5, 2, 8, nkt, 8, 9).transpose(0, 4, 1, 5, 2, 3, 6) for b in planes]  # p kt j s h n 9
    return np.concatenate(out, 1).reshape(-1)


def packed(g: Gemm, w: W.Wt | np.ndarray) -> list[np.ndarray]:
    """Every core row's image of g by the native packer."""
    per = g.npo * g.nkt * D.Q
    s = W.Slots()
    s.add(g, W.as_wt(w), {r: ("w", r * per) for r in range(ROWS)})
    out = np.zeros(ROWS * per, np.uint8)
    s.pack_to({"w": out})
    return [out[r * per:(r + 1) * per] for r in range(ROWS)]


def check(g: Gemm, w: W.Wt | np.ndarray) -> None:
    vals = W.values(w)
    for r, img in enumerate(packed(g, w)):
        ref = slot_ref(g, vals, r)
        assert np.array_equal(img, ref), f"row {r}: {np.count_nonzero(img != ref)} of {ref.size} bytes differ"


def source(kind: str, x: np.ndarray, name: str = "t") -> W.Wt:
    """A matrix of x's values as a source of kind (Q8_0: x quantized to blocks of 32 first)."""
    x = np.ascontiguousarray(x, np.float32)
    r, c = x.shape[0], x[0].size
    if kind == "F32":
        raw = x.reshape(r, c).view(np.uint8)
    elif kind == "F16":
        raw = x.reshape(r, c).astype(np.float16).view(np.uint8)
    elif kind == "BF16":
        raw = x.reshape(r, c).astype(bfloat16).view(np.uint8)
    else:
        b = x.reshape(r, c // 32, 32)
        d = (np.abs(b).max(-1) / 127).astype(np.float16)
        q = np.clip(np.round(b / np.maximum(d.astype(np.float32), 1e-30)[..., None]), -127, 127).astype(np.int8)
        raw = np.concatenate([d[..., None].view(np.uint8), q.view(np.uint8)], -1).reshape(r, -1)
    return W.Wt.of(W.Source(name, kind, r, c, np.ascontiguousarray(raw)), x.shape)


def _gemm(N: int, K: int, **kw: object) -> Gemm:
    return Gemm(m=64, ntile=1, K=K, N=N, x=0, y=0, **kw)


@pytest.mark.parametrize("kind", W.KINDS[:4])
def test_kinds(kind: str) -> None:
    """A plain GEMM's weights of every source type (K past whole k-tiles: zero columns), values of a wide range
    (zero blocks, tiny and huge exponents)."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((640, 352)).astype(np.float32) * np.exp2(rng.integers(-12, 12, (640, 1)))
    w[3] = 0
    w[5, :8] = 0
    check(_gemm(640, 352), source(kind, w))


def test_maps() -> None:
    """Row and column maps with zeros (heads padded, a sliced row range), a scale, GEGLU's rows, passes of a part of
    the rows (p0, npo), several parts (column blocks of two sources and the identity, as ``cat``), tsplit."""
    rng = np.random.default_rng(1)
    a = source("BF16", rng.standard_normal((1152, 1152)).astype(np.float32), "a")
    b = source("F16", rng.standard_normal((1280, 576)).astype(np.float32), "b")
    heads = np.array([h * 72 + d if d < 72 else -1 for h in range(16) for d in range(80)])
    q = W.take(W.take(a, heads, 0), W.pad_index(1280, 1152), 1) * 0.17
    check(_gemm(1280, 1280), q)
    check(_gemm(1280, 1280, p0=1, npo=2), q)
    check(_gemm(1280, 1280, out="geglu"), q)
    check(Gemm(m=64, ntile=2, K=1280, N=640, x=0, y=0, tsplit=2, out="bf16"), q[:640])
    e = W.take(W.take(W.eye(1280), W.pad_index(1280, 1152), 0), W.pad_index(1280, 1152), 1) * 0.25
    check(_gemm(1280, 1280 + 576 + 1280), W.cat([q, b, e], 1))


def test_hilo() -> None:
    """hi + lo planes (a block whose negative maximum rounds to -128 among them)."""
    rng = np.random.default_rng(2)
    w = rng.standard_normal((320, 200)).astype(np.float32)
    w[0, :8] = [-127.75, 1, 2, 3, 0.5, -0.25, 7, 9]
    check(_gemm(320, 200, hilo=True), source("F32", w))


@pytest.mark.parametrize("kw", [{"R": 1, "Lg": 4}, {"R": 2, "Lg": 2}, {"R": 1, "Lg": 8, "il": True, "rc": 4, "r0": 1,
                                                                        "ntile": 2}])
def test_conv(kw: dict) -> None:
    """A 3x3 conv's weights (N, K, 3, 3) per chunk of 64 input channels in its taps' order (rows of 4 groups, of 2 rows,
    64x64's il order), an input-channel slice of a wider conv (``w[:, a:b]``), padded channels (zeros)."""
    rng = np.random.default_rng(3)
    w = source("F32", rng.standard_normal((320, 384, 3, 3)).astype(np.float32))
    m = 16 * kw["R"] * kw["Lg"] // (2 if kw.get("il") else 1)
    g = Gemm(m=m, ntile=kw.get("ntile", 1), K=128, N=320, x=0, y=0,
             conv=ConvSpec(stride=1, zero=0, **{k: v for k, v in kw.items() if k != "ntile"}))
    check(g, w[:, 64:192])
    check(g, W.take(w, np.array([0, 1, 2, 3] + [-1] * 124), 1))


def test_wt_ops() -> None:
    """``Wt``'s maps against torch's operations on the values: rows, channels, spatial positions, flatten, cat, the
    scale, ``take`` on tensors."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 6, 3, 3)).astype(np.float32)
    w, t = source("F32", x), torch.from_numpy(x)
    torch.testing.assert_close(torch.from_numpy(w[2:5].numpy()), t[2:5], rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(w[:, 1:4].numpy()), t[:, 1:4], rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(w[:, :, 0, 0].numpy()), t[:, :, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(w.flatten(1).numpy()), t.flatten(1), rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy((w * 0.3).numpy()), t * 0.3, rtol=0, atol=0)
    rows = torch.tensor([7, 0, 3])
    torch.testing.assert_close(torch.from_numpy(w[rows].numpy()), t[rows], rtol=0, atol=0)
    two = W.cat([w[:2], w[5:]])
    torch.testing.assert_close(torch.from_numpy(two.numpy()), torch.cat([t[:2], t[5:]]), rtol=0, atol=0)
    assert len(two.parts) == 1  # (one source's rows: one part)
    idx = np.array([1, -1, 0])
    torch.testing.assert_close(torch.from_numpy(W.take(w, idx, 1).numpy()), torch.from_numpy(W.take(t, idx, 1)), rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(W.take(w, idx, 0).numpy()), torch.from_numpy(W.take(t, idx, 0)), rtol=0, atol=0)
    assert w[1:3].key == w[1:3].key and w[1:3].key != w[1:4].key


def test_files(tmp_path: object) -> None:
    """A safetensors file's matrices (F32, F16, BF16) as sources of its mapping, its vectors as tensors; a GGUF file's
    tensors packed from its mapping."""
    rng = np.random.default_rng(5)
    arrays = {"a": rng.standard_normal((320, 64, 3, 3)).astype(np.float32),
              "b": rng.standard_normal((320, 128)).astype(np.float16),
              "c": rng.standard_normal(320).astype(np.float32)}
    header, blob = {}, b""
    for n, a in arrays.items():
        header[n] = {"dtype": {"float32": "F32", "float16": "F16"}[a.dtype.name], "shape": list(a.shape),
                     "data_offsets": [len(blob), len(blob) + a.nbytes]}
        blob += a.tobytes()
    h = json.dumps(header).encode()
    path = f"{tmp_path}/w.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blob)
    st = W.safetensors(path)
    assert isinstance(st["c"], np.ndarray) and np.array_equal(st["c"], arrays["c"])
    assert np.array_equal(st["a"].numpy(), arrays["a"]) and np.array_equal(st["b"].numpy(), arrays["b"].astype(np.float32))
    assert st["a"].parts[0].source.file[0] == path
    check(Gemm(m=64, ntile=1, K=64, N=320, x=0, y=0, conv=ConvSpec(stride=1, R=1, Lg=4, zero=0)), st["a"])
    check(_gemm(320, 128), st["b"])
