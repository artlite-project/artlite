"""The prefill GEMMs (prefill/gemm.py: GemmQ / GemmQd) on the models' decode images (Llama's ``LlamaImage``,
qwen35's ``QwenImage``, Gemma 4's ``GemmaImage``: slices of a few layers, read in place): each phase's GEMMs as the
model's prefill plans them, with random bfp16 activations, against the dequantized GGUF rows: input projections at
their column-slice layout, output projections at d_pitch (K parts: the parts' sum), gate/up's act(gate) up panels.
The files cover the tile formats: Q4_K (tile-major GemmQ), Q5_K / Q6_K (planar GemmQd, 13 / 11 / 12-row tiles), Q4_0 /
Q4_1 (planar at 1024 columns; Gemma's at 512), Q2_K / Q3_K (word-major planar), BFP16 (tile-major, 8-row tiles of a BF16
file), Q8_0 (super-block units, Occamy's GDN z), F32 rows as BFP16 plane pairs (Occamy's GDN alpha: fp32 sums)."""

import os

import numpy as np
import pytest
import torch

import artlite
from artlite.gguf_model import GGUFFile
from artlite.model_config import gemma_gguf, llama_gguf, occamy_gguf, qwen_gguf
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.prefill import gemm as PG

# (model, GGUF quantization, size, layers to pack, the layer and phase tested, tokens)
CASES = [("llama", "Q4_K_M", "1B", (0, 3), 3, "qkv", 64), ("llama", "Q4_K_M", "1B", (0, 3), 3, "gate_up", 256),
         ("llama", "Q4_K_M", "1B", (0, 3), 0, "down", 256), ("llama", "Q4_K_M", "1B", (0, 3), 3, "o", 64),
         ("llama", "Q4_K_M", "3B", (0, 3), 3, "qkv", 256), ("llama", "Q4_K_M", "3B", (0, 3), 3, "down", 512),
         ("llama", "Q4_K_M", "3B", (0, 3), 0, "qkv", 64), ("llama", "Q4_K_M", "3B", (0, 3), 3, "gate_up", 64),
         ("llama", "Q4_K_M", "3B", (0, 3), 0, "o", 256),
         ("llama", "Q4_K_S", "8B", (0,), 0, "qkv", 64), ("llama", "Q4_K_S", "8B", (0,), 0, "down", 256),
         ("llama", "Q4_K_S", "8B", (0,), 0, "gate_up", 256),
         ("llama", "Q5_K_M", "1B", (0, 3), 3, "qkv", 192), ("llama", "Q5_K_M", "1B", (0, 3), 3, "gate_up", 64),
         ("llama", "Q5_K_M", "1B", (0, 3), 3, "o", 64), ("llama", "Q5_K_M", "1B", (0, 3), 0, "down", 256),
         ("llama", "Q6_K", "1B", (0,), 0, "gate_up", 192), ("llama", "Q6_K", "1B", (0,), 0, "down", 64),
         ("llama", "Q4_0", "1B", (0,), 0, "qkv", 64), ("llama", "Q4_0", "1B", (0,), 0, "down", 192),
         ("llama", "Q2_K", "3B", (0,), 0, "gate_up", 192), ("llama", "Q2_K", "3B", (0,), 0, "down", 64),
         ("llama", "Q3_K_M", "3B", (0,), 0, "qkv", 64), ("llama", "Q3_K_L", "1B", (0,), 0, "gate_up", 64),
         ("llama", "BF16", "1B", (0,), 0, "qkv", 256), ("llama", "BF16", "1B", (0,), 0, "gate_up", 256),
         ("llama", "BF16", "1B", (0,), 0, "down", 64), ("llama", "Q4_1", "1B", (0,), 0, "gate_up", 256),
         ("llama", "Q4_1", "1B", (0,), 0, "o", 64), ("llama", "Q4_0", "1B", (0,), 0, "gate_up", 256),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gdn_in", 64), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gdn_out", 64),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gate_up", 64), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "down", 64),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 4, "down", 256), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 3, "attn_in", 256),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 3, "attn_out", 64), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gdn_in", 256),
         ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 0, "gdn_in", 256), ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 8, "gdn_in", 64),
         ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 0, "gate_up", 256), ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 0, "down", 256),
         ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 8, "down", 64), ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 3, "attn_in", 64),
         ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 3, "attn_out", 256),
         ("qwen35", "Q4_K_M", "1B", (0, 3), 0, "gdn_in", 256), ("qwen35", "Q4_K_M", "1B", (0, 3), 0, "gdn_out", 64),
         ("qwen35", "Q4_K_M", "1B", (0, 3), 0, "gate_up", 256), ("qwen35", "Q4_K_M", "1B", (0, 3), 0, "down", 256),
         ("qwen35", "Q4_K_M", "1B", (0, 3), 3, "down", 64), ("qwen35", "Q4_K_M", "1B", (0, 3), 3, "attn_in", 256),
         ("qwen35", "Q4_K_M", "1B", (0, 3), 3, "attn_out", 64),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "qkv", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 5, "qkv", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "o", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 5, "o", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate_up", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 0, "down", 288),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 0, "proj", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate_up", 1024), ("gemma4", "Q4_0", "E4B", (0, 5), 0, "down", 1024),
         ("gemma4", "Q4_0", "E4B", (0, 5), 5, "qkv", 1024),
         ("occamy", "", "", (0, 3), 0, "gdn_in", 256), ("occamy", "", "", (0, 3), 0, "gdn_in", 64),
         ("occamy", "", "", (0, 3), 0, "gdn_out", 256), ("occamy", "", "", (0, 3), 3, "attn_in", 256),
         ("occamy", "", "", (0, 3), 3, "attn_out", 64)]

_images: dict[tuple, tuple] = {}


def _image(model: str, quant: str, size: str, layers: tuple[int, ...]) -> tuple[GGUFFile, object, np.ndarray]:
    """The model's image of the layers (a truncated vocabulary), packed."""
    path = {"llama": lambda: llama_gguf(quant, size), "qwen35": lambda: qwen_gguf(quant, size),
            "gemma4": lambda: gemma_gguf(quant), "occamy": occamy_gguf}[model]()
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    key = (path, layers)
    if key not in _images:
        _images.clear()
        g = GGUFFile(path)
        if model == "llama":
            from models.llama.image import LlamaImage as Image
        elif model in ("qwen35", "occamy"):
            from models.qwen35.image import QwenImage as Image
        else:
            from models.gemma4.image import GemmaImage as Image
        img = Image(g, layers=list(layers), vocab=COLS * 64)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W, release=False)
        _images[key] = (g, img, W)
    return _images[key]


def _plan(model: str, img: object, region: object, T: int) -> PG.PhasePlan:
    """The phase's GEMMs with the model's prefill options (Gemma: GELU, down in K parts)."""
    pitch = PG.d_pitch(img, T) if region.name in PG.OUT else 0
    if model == "gemma4":
        from models.gemma4.prefill import plan_gemm
    elif model == "llama":
        from models.llama.prefill import plan_gemm
    else:
        return PG.plan_phase(img, region, T, pitch=pitch)
    return plan_gemm(img, region, T, pitch=pitch)


def _rows(img: object, tensors: tuple[str, ...], rows: np.ndarray, K: int = 0) -> torch.Tensor:
    """Dequantized rows of the stacked tensors as the image holds them (Llama's Q / K rows in rotate-half order, Gemma's
    proj in its slotted copy; K: zero columns past the tensors' up to K, a K padded to whole chunks)."""
    w = torch.cat([img.tensor(t).dequantize().float().reshape(img.tensor(t).rows, -1) for t in tensors])
    w = w[torch.from_numpy(rows)]
    return torch.nn.functional.pad(w, (0, max(K - w.shape[1], 0)))


def _run(op: object, A: torch.Tensor, W: np.ndarray, w_off: int, a_off: int = 0,
         C: torch.Tensor | None = None) -> torch.Tensor:
    """One GEMM: A (op.M, K) as panels from element a_off on (a K part's columns), the W argument from the image
    (zeros outside it), C (bf16 (M, N); default zeros: the columns another GEMM of the phase writes; given: the
    phase's output so far, at least op.M rows, which this one's writes overwrite)."""
    dev = op.device
    w = np.zeros(op.w_bytes, np.uint8)
    a, b = max(w_off, 0), min(w_off + op.w_bytes, W.size)
    w[a - w_off:b - w_off] = W[a:b]
    init = C if C is not None else torch.zeros((op.M * op.N * 9 // 16,) if op.silu else (op.M, op.N),
                                               dtype=torch.bfloat16)
    out = artlite.from_torch(init.clone(), dev)
    pan = bfp16.to_panels(A)
    op.dispatch(artlite.from_torch(torch.cat([pan[a_off:], torch.zeros(a_off, dtype=pan.dtype)]), dev),
                artlite.from_torch(torch.from_numpy(w).view(torch.bfloat16), dev), out)
    out.sync_from_device()
    c = out.to_torch()
    return bfp16.from_panels(c, op.M, op.N) if op.silu else c


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm())


def _tol(K: int) -> float:
    """C rounded to bf16 after each 64-k tile: ~2^-9 sqrt(K / 64); the weights dequantized to bfp16 panels: a floor
    whatever K (Q4_K: ~1.1% at K = 1024, MiniCPM-V's input projections), the bound's at K = 2048."""
    return 1.35 * 2 ** -9 * (max(K, 2048) / 64) ** 0.5


@pytest.mark.parametrize(("model", "quant", "size", "layers", "layer", "phase", "T"), CASES,
                         ids=[f"{m}-{s}-{q}-L{L}-{p}-T{T}" for m, q, s, _, L, p, T in CASES])
def test_prefill_gemm(model: str, quant: str, size: str, layers: tuple[int, ...], layer: int, phase: str,
                      T: int) -> None:
    _, img, W = _image(model, quant, size, layers)
    cfg = img.cfg
    region = img.region(layer, phase)
    pp = _plan(model, img, region, T)
    for gm in pp.gemms:
        gm.op.compile()
    M, K = max(gm.op.M for gm in pp.gemms), region.K  # (GemmQd pads the tokens to whole passes)
    A = (torch.randn(M, K, generator=torch.Generator().manual_seed(T + K)) * 0.5).bfloat16()
    Aq = bfp16.from_panels(bfp16.to_panels(A), M, K)  # (the GEMM's operand values)
    if pp.layout is not None:  # (an input projection's GEMMs write one output in turn, as the prefill's: M rows)
        Cb = torch.zeros((M, pp.gemms[0].op.N), dtype=torch.bfloat16)
        for gm in pp.gemms:
            Cb = _run(gm.op, A[:gm.op.M], W, gm.w_off, gm.a_off, Cb)
        C = Cb[:T].float()
    else:
        Cs: list[torch.Tensor | None] = [None] * pp.parts
        for gm in pp.gemms:  # (the GEMMs of a part write disjoint columns)
            C = _run(gm.op, A[:gm.op.M], W, gm.w_off, gm.a_off)[:T].float()
            Cs[gm.part] = C if Cs[gm.part] is None else Cs[gm.part] + C
        C = sum(Cs)
    if phase == "gate_up":
        (b,) = region.blocks
        ffn = cfg.intermediate_size // COLS
        idx = np.concatenate([ffn * c + np.arange(ffn) for c in range(COLS)])
        gt, up = (_rows(img, (t,), idx) for t in b.tensors)
        y_g, y_u = Aq[:T] @ gt.T, Aq[:T] @ up.T
        act = (lambda y: torch.nn.functional.gelu(y, approximate="tanh")) if model == "gemma4" else \
            torch.nn.functional.silu
        want = act(y_g.bfloat16().float()) * y_u
        assert _rel(C[:, idx], want) < 3.5e-2  # (and the bfp16 C panels)
        return
    if phase in PG.OUT:
        (b,) = region.blocks
        h8 = cfg.hidden_size // COLS
        cols = np.concatenate([pitch_c + np.arange(h8) for pitch_c in PG.d_pitch(img, T) * np.arange(COLS)])
        rows = np.concatenate([b.rows[c][b.rows[c] >= 0] for c in range(COLS)])
        want = Aq[:T] @ _rows(img, b.tensors, rows, K).T
        assert _rel(C[:, cols], want) < _tol(K)
        return
    lay = pp.layout
    for b in region.blocks:
        for c in range(COLS):
            rows = b.rows[c][b.rows[c] >= 0]
            o = lay.pitch * c + lay.off[b.name]
            if b.pairs:  # (fp32 sums of the planes: the F32 rows times the norm's w / bf16(w), as the image packs them)
                got = Cb[:T, o:o + rows.size].contiguous().view(torch.float32).float()
                want = Aq[:T] @ (_rows(img, b.tensors, rows[0::2] // 2) * _ratio(img, region)[None]).T
                assert _rel(got, want) < 5e-5, (b.name, c, _rel(got, want))  # (hi + lo: ~2^-16 of a row)
                continue
            got = C[:, o + np.arange(rows.size)]
            assert _rel(got, Aq[:T] @ _rows(img, b.tensors, rows).T) < _tol(K), (b.name, c)


def _ratio(img: object, region: object) -> torch.Tensor:
    """The norm weight's w / bf16(w) of a region's input (the factor F32 rows are packed with: ``QwenImage.pack``)."""
    w = img.gguf.tensor(f"blk.{region.layer}.attn_norm.weight").dequantize().float().reshape(-1)
    bw = w.bfloat16().float()
    return torch.where(bw != 0, w / torch.where(bw != 0, bw, 1), 1)
