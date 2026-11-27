"""The layer program's host side: the code pages of the design, the weight images' layouts and the arithmetic the
program's conversions do - gated against the CPU reference of the same layer.
"""

import numpy as np

from .moss_layer import (
    A_PAGE_BUDGET,
    FFN,
    FFN_REAL,
    HD,
    HEADS,
    HP,
    PAGES,
    QSCALE,
    C,
    H,
    block,
    block_reference,
    compiled,
    layer_front,
    layer_weights,
    page_sizes,
    program_memory,
    reference,
)


def seeded(rows: int = 1536) -> dict[str, np.ndarray]:
    """A layer's HF-shaped tensors (small seeded values: every projection stays in the operands' range) and the input."""
    rng = np.random.default_rng(11)
    return {"q_w": rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02, "q_b": rng.standard_normal(1024) * 0.01,
            "k_w": rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02, "v_w": rng.standard_normal((1024, 1024)) * 0.02,
            "v_b": rng.standard_normal(1024) * 0.01, "o_w": rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02,
            "o_b": rng.standard_normal(1024) * 0.01, "ln_w": 1.0 + rng.standard_normal(1024) * 0.1,
            "ln_b": rng.standard_normal(1024) * 0.05,
            "fc1_w": rng.standard_normal((4096, 1024)) * 0.02, "fc1_b": rng.standard_normal(4096) * 0.01,
            "fc2_w": rng.standard_normal((1024, 4096)) * 0.02, "fc2_b": rng.standard_normal(1024) * 0.01,
            "f_ln_w": 1.0 + rng.standard_normal(1024) * 0.1, "f_ln_b": rng.standard_normal(1024) * 0.05,
            "_x": rng.standard_normal((rows, 1024)) * 0.5}


def test_pages() -> None:
    """The design compiles; its ``a`` page is inside the 8.6 KB budget and every page the layer runs fits beside the
    resident code."""
    sizes = page_sizes(compiled())
    assert sizes["a"] <= A_PAGE_BUDGET, f"the a page is {sizes['a']} B (budget {A_PAGE_BUDGET}): {sizes}"
    fits = program_memory(sizes)
    assert all(n <= 16384 for n in fits.values()), fits
    assert set(PAGES) <= set(sizes), sorted(sizes)


def test_weights_layout() -> None:
    """The program's layouts: q / k / v rows and o's columns head-major (each head's 64 dims at its 80-channel slot,
    zero pads), q and its bias scaled by log2 e / sqrt 64, the K each matrix reads (q / k / v the 1,024 real channels, o
    the padded q width), the biases at the heads' rows, the pre-LN's vectors at the residual width."""
    src = seeded()
    w = layer_weights(src)
    for tag in ("q", "k", "v"):
        assert w[tag].shape == (HEADS * HP, H), (tag, w[tag].shape)
        scale = QSCALE if tag == "q" else 1.0
        for h in (0, 7, 15):
            assert np.allclose(w[tag][h * HP:h * HP + HD],
                               src[tag + "_w"][h * HD:(h + 1) * HD] * scale, atol=1e-6), (tag, h)
            assert np.allclose(w[tag][h * HP + HD:(h + 1) * HP], 0.0), (tag, h)
        assert np.allclose(w[tag + "_b"][h * HP + HD:h * HP + HP], 0.0), tag
    assert w["o"].shape == (C, HEADS * HP)
    for h in (0, 15):
        assert np.allclose(w["o"][:H, h * HP:h * HP + HD], src["o_w"][:, h * HD:(h + 1) * HD], atol=1e-6), h
    assert np.allclose(w["o"][H:], 0.0) and np.allclose(w["o_b"][H:], 0.0)
    assert np.allclose(w["q_b"][:HD], src["q_b"][:HD] * QSCALE, atol=1e-6) and np.allclose(w["k_b"], 0.0)
    assert np.allclose(w["ln_g"][H:], 0.0) and np.allclose(w["ln_b"][H:], 0.0)
    assert w["fc1"].shape == (FFN, H) and w["fc2"].shape == (C, FFN) and w["ln_f_g"].shape == (C,)
    assert np.allclose(w["fc1"][FFN_REAL:], 0.0) and np.allclose(w["fc2"][H:], 0.0)
    assert np.allclose(w["fc1"][:FFN_REAL, :H], src["fc1_w"], atol=1e-6)
    assert np.allclose(w["fc2"][:H, :FFN_REAL], src["fc2_w"], atol=1e-6)
    assert np.allclose(w["ln_f_g"][H:], 0.0)


def test_block_gate() -> None:
    """The whole layer's arithmetic against HF's layer in float64 (the gate A5: 3 %): the attention half (1 %) and the
    FFN - the post attention pre-LN's panels, fc1 4,096 -> 4,160 with the erf GELU, fc2 (K 4,160) and the bf16 residual."""
    src = seeded()
    got = block(src["_x"], layer_weights(src), keys=1500)
    ref = block_reference(src["_x"], src, keys=1500)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert got.shape == (1536, H) and np.isfinite(got).all() and np.isfinite(ref).all()
    assert rel <= 0.03, f"the layer is {rel:.4f} from the reference (gate 0.03)"


def test_front_gate() -> None:
    """The program's arithmetic against the CPU reference of the same layer (the gate: 1 %): the C tiles' bf16, the
    pre-LN's fp32 statistics, the panels' and weights' bfp16 blocks, the command biases' bf16, the d-80 attention's
    per-query maximum with P rounded once to bfp16, the output projection and the bf16 residual."""
    src = seeded()
    got = layer_front(src["_x"], layer_weights(src), keys=1500)
    ref = reference(src["_x"], src)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert got.shape == (1536, H) and np.isfinite(got).all() and np.isfinite(ref).all()
    assert rel <= 0.01, f"the layer's front is {rel:.4f} from the reference (gate 0.01)"


def test_ablation_program() -> None:
    """The layer without the attention page's work builds and verifies (the ops' streams are the layer's; the queries
    stand in for the context): no attention command image, no ``a`` page in the transaction."""
    from artlite.vision.builder import Builder
    from artlite.vision.device import VisionDevice
    from artlite.vision.program import Program

    from ..txn_dump import fake_vision
    from .moss_layer import KEYS, LV, NAME, front, layer_wt

    src = seeded()
    with fake_vision():
        npu = VisionDevice()
        npu.alloc(f"{NAME}.ab.kv", KEYS.attention(LV).kv_bytes())
        bd = Builder(npu)
        front(bd, layer_wt(src, f"{NAME}.ab"), name=f"{NAME}.ab", lv=LV, kv_ddr=npu.address(f"{NAME}.ab.kv"),
              keys=KEYS, attention=False)
        tr = Program.of(npu, bd).transaction(npu)
    assert len(tr.words) > 0 and f"{NAME}.ab.attn" not in set(npu.names())


def test_sequence_mapping() -> None:
    """The joint key sequence's positions are a bijection of the 1,536-row grid, and its masks keep exactly the 1,500
    rows of the sequence (the grid's 36 padding rows are the 36 masked keys - they sit in the last group of each image
    and are masked by the key words, not by whole groups)."""
    from .moss_layer import LV, sequence_rows, valid_keys

    rows = sequence_rows(LV)
    assert rows.shape == (1536,) and np.unique(rows).size == 1536 and rows.max() == 1535
    ok = valid_keys(LV)
    assert ok.sum() == 1500 and (~ok).sum() == 36
    assert np.array_equal(np.sort(rows[ok]), np.arange(1500))


def test_transaction() -> None:
    """The layer's front builds as one dispatch of the vision overlay's design on a fake device (no NPU: the design
    compiles from the cache, its role pages link, the transaction's BDs are verified): the 3 bands' passes, the keys'
    image, the output's bands and the 4 matrices' weight images."""
    from artlite.vision.device import VisionDevice

    from ..txn_dump import fake_vision
    from .moss_layer import NAME, Layer, layer_wt

    src = seeded()
    with fake_vision() as log:
        npu = VisionDevice()
        lay = Layer(npu, layer_wt(src))
        lay.write(src["_x"])
        lay.run()
    images = set(lay.program.images)
    for b in range(3):
        assert {f"{NAME}.x.{b}", f"{NAME}.y.{b}", f"{NAME}.x.rec.{b}"} <= images, sorted(images)
    assert f"{NAME}.kv" in images and f"{NAME}.attn" in images
    assert len(lay.program.words) > 0 and len(log.entries) == 1


def test_chain() -> None:
    """Two layers of the same program on a fake device (no NPU): each layer's own weight images, the bands alternating
    x -> y -> x, the K / V^T slots' two images; both transactions verified."""
    from artlite.vision.device import VisionDevice

    from ..txn_dump import fake_vision
    from .moss_layer import Chain

    src = seeded()
    with fake_vision() as log:
        npu = VisionDevice()
        chain = Chain(npu, [src, src])
        chain.write(src["_x"])
        chain.run()
    assert len(chain.transactions) == 2 and len(log.entries) == 2
    names = set(npu.names())
    for n in ("moss.kv0", "moss.kv1", "moss.x.0", "moss.y.0"):
        assert n in names, sorted(names)
    assert chain.out == "moss.x"  # (the second layer writes back into x)
