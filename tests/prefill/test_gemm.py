"""The prefill GEMMs (prefill/gemm.py: GemmQ / GemmQd) on the models' decode images (Llama's ``LlamaImage``,
qwen35's ``QwenImage``, Gemma 4's ``GemmaImage``: slices of a few layers, read in place): each phase's GEMMs as the
model's prefill plans them, with random bfp16 activations, against the dequantized GGUF rows: input projections at
their column-slice layout, output projections at d_pitch (K parts: the parts' sum), gate/up's act(gate) up panels.
The files cover the tile formats: Q4_K (tile-major GemmQ), Q5_K / Q6_K (planar GemmQd, 13 / 11 / 12-row tiles), Q4_0 /
Q4_1 (planar at 1024 columns; Gemma's at 512), Q2_K / Q3_K (word-major planar), BFP16 (tile-major, 8-row tiles of a BF16
file), Q8_0 (super-block units: Occamy's GDN z; every role of vllm-d3's d3-lite, gate/up in half groups; Gemma's
16-row copy of the per-layer projection), F32 rows as BFP16 plane pairs (Occamy's GDN alpha: fp32 sums); LFM3's conv layers'
in / out and its qkv with a gate block of one tile (two rows a column). The GEMMs
run in a library program as the prefill's do, paced (``gemm_q``'s "Pacing") at level 0 and at ``LEVEL``, and as unpaced
twins: their outputs bit-identical."""


import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import artlite
from artlite.decode.image import pack_rows
from artlite.formats import FORMATS
from artlite.gemm.gemm_q import PACE, pace_word
from artlite.gemm.gemm_qd import GemmQdOperator
from artlite.gguf_model import GGUFFile
from artlite.model_config import (
    decision_gguf,
    gemma_gguf,
    lfm3_gguf,
    llama_gguf,
    occamy_gguf,
    qwen3_gguf,
    qwen_gguf,
    swift_gguf,
)
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.fused import Library
from artlite.prefill import gemm as PG

from ..common import to_numpy, to_torch
from ..models import require
from ..numerics import rel_native

LEVEL = 3  # (a paced level: the paced copy of each worker body)

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
         ("gemma4", "Q4_0", "E4B", (0, 5), 5, "qkv", 1024), ("gemma4", "Q4_0", "E4B", (0, 5), -1, "ple", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), -1, "ple", 1024),
         ("occamy", "", "", (0, 3), 0, "gdn_in", 256), ("occamy", "", "", (0, 3), 0, "gdn_in", 64),
         ("occamy", "", "", (0, 3), 0, "gdn_out", 256), ("occamy", "", "", (0, 3), 3, "attn_in", 256),
         ("occamy", "", "", (0, 3), 3, "attn_out", 64),
         ("d3-lite", "", "", (0, 3), 0, "gdn_in", 256), ("d3-lite", "", "", (0, 3), 0, "gate_up", 256),
         ("d3-lite", "", "", (0, 3), 0, "gate_up", 64), ("d3-lite", "", "", (0, 3), 0, "down", 256),
         ("d3-lite", "", "", (0, 3), 3, "attn_in", 64), ("d3-lite", "", "", (0, 3), 3, "attn_out", 256),
         ("qwen3", "", "embedding-4B", (0, 4), 4, "down", 256), ("qwen3", "", "embedding-4B", (0, 4), 0, "down", 256),
         ("qwen3", "", "embedding-4B", (0, 4), 4, "qkv", 256), ("qwen3", "", "embedding-4B", (0, 4), 4, "gate_up", 256),
         ("qwen3", "", "embedding-4B", (0, 4), 4, "o", 64), ("qwen3", "", "embedding-0.6B", (0,), 0, "down", 256),
         ("swift", "", "", (0, 16), 0, "down", 256), ("swift", "", "", (0, 16), 16, "down", 64),
         ("swift", "", "", (0, 16), 0, "gate_up", 256), ("swift", "", "", (0, 16), 16, "gate_up", 256),
         ("swift", "", "", (0, 16), 16, "gate_up", 64), ("swift", "", "", (0, 16), 0, "gdn_in", 256),
         ("swift", "", "", (0, 16), 16, "gdn_in", 64),
         ("lfm", "QAD-Q4_0", "", (0, 2), 0, "in", 256), ("lfm", "QAD-Q4_0", "", (0, 2), 0, "out", 256),
         ("lfm", "QAD-Q4_0", "", (0, 2), 2, "qkv", 256), ("lfm", "QAD-Q4_0", "", (0, 2), 2, "o", 64),
         ("lfm", "QAD-Q4_0", "", (0, 2), 0, "gate_up", 256), ("lfm", "QAD-Q4_0", "", (0, 2), 0, "down", 256),
         ("lfm", "QAD-Q4_0", "", (0, 2), 0, "in", 64), ("lfm", "Q4_K_M", "", (1, 2), 2, "qkv", 256),
         ("lfm", "Q4_K_M", "", (1, 2), 1, "down", 256), ("lfm", "Q4_K_M", "", (1, 2), 1, "in", 256),
         ("lfm", "Q8_0", "", (0, 2), 2, "qkv", 256), ("lfm", "Q8_0", "", (0, 2), 0, "gate_up", 256),
         ("lfm", "BF16", "", (0, 2), 0, "in", 256)]

_images: dict[tuple, tuple] = {}


def _image(model: str, quant: str, size: str, layers: tuple[int, ...]) -> tuple[GGUFFile, object, np.ndarray]:
    """The model's image of the layers (a truncated vocabulary), packed."""
    path = {"llama": lambda: llama_gguf(quant, size), "qwen35": lambda: qwen_gguf(quant, size),
            "gemma4": lambda: gemma_gguf(quant), "occamy": occamy_gguf, "d3-lite": decision_gguf, "swift": swift_gguf,
            "qwen3": lambda: qwen3_gguf(size), "lfm": lambda: lfm3_gguf(quant)}[model]()
    require(path)
    key = (path, layers)
    if key not in _images:
        _images.clear()
        g = GGUFFile(path)
        if model in ("llama", "qwen3"):
            from models.llama.image import LlamaImage as Image
        elif model in ("qwen35", "occamy", "d3-lite", "swift"):
            from models.qwen35.image import QwenImage as Image
        elif model == "lfm":
            from models.lfm.image import LfmImage as Image
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
    elif model in ("llama", "qwen3"):
        from models.llama.prefill import plan_gemm
    elif model == "lfm":
        from models.lfm.prefill import plan_gemm
    else:
        return PG.plan_phase(img, region, T, pitch=pitch)
    return plan_gemm(img, region, T, pitch=pitch)


def _tensor(img: object, name: str) -> object:
    """The image's tensor (Gemma's Q8_0 copy of the per-layer projection: made again, a packed image has dropped it)."""
    t = img.tensor(name)
    if not t.raw.size:
        img._ple()
        t = img.tensor(name)
    return t


def _rows(img: object, tensors: tuple[str, ...], rows: np.ndarray, K: int = 0) -> torch.Tensor:
    """Dequantized rows of the stacked tensors as the image holds them (Llama's Q / K rows in rotate-half order, Gemma's
    proj in its slotted copy; K: zero columns past the tensors' up to K, a K padded to whole chunks)."""
    w = torch.cat([torch.from_numpy(_tensor(img, t).dequantize()).reshape(img.tensor(t).rows, -1) for t in tensors])
    w = w[torch.from_numpy(rows)]
    return torch.nn.functional.pad(w, (0, max(K - w.shape[1], 0)))


def _twin(op: object) -> object:
    """The operator unpaced (its constructor's arguments, pace None)."""
    args, kwargs = op._ctor
    return type(op)(*args, **{**kwargs, "pace": None})


def _run(lib: Library, op: object, A: torch.Tensor, W: np.ndarray, w_off: int, a_off: int = 0,
         C: torch.Tensor | None = None) -> torch.Tensor:
    """One GEMM of the library: A (op.M, K) as panels from element a_off on (a K part's columns), the W argument from
    the image (zeros outside it), C (bf16 (M, N); default zeros: the columns another GEMM of the phase writes; given:
    the phase's output so far, at least op.M rows, which this one's writes overwrite)."""
    dev = artlite.Device()
    w = np.zeros(op.w_bytes, np.uint8)
    a, b = max(w_off, 0), min(w_off + op.w_bytes, W.size)
    w[a - w_off:b - w_off] = W[a:b]
    init = to_numpy(C) if C is not None else np.zeros((op.M * op.N * 9 // 16,) if op.silu else (op.M, op.N), bfloat16)
    out = artlite.from_numpy(init, dev)
    pan = bfp16.to_panels(to_numpy(A))
    bufs = [artlite.from_numpy(np.concatenate([pan[a_off:], np.zeros(a_off, bfloat16)]), dev),
            artlite.from_numpy(w.view(bfloat16), dev), out]
    lib.run(lib.items([(op, "A", "W", "C")], dict(zip("AWC", (x._raw for x in bufs)))), bufs=[x._raw for x in bufs])
    out.sync_from_device()
    c = out.numpy()
    return to_torch(bfp16.from_panels(c, op.M, op.N) if op.silu else c.copy())


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
    ops = list(dict.fromkeys(gm.op for gm in pp.gemms))
    twins = {id(op): _twin(op) for op in ops}
    lib = Library("t_prefill_gemm", [*ops, *twins.values()]).compile().load()
    M, K = max(gm.op.M for gm in pp.gemms), region.K  # (GemmQd pads the tokens to whole passes)
    A = (torch.randn(M, K, generator=torch.Generator().manual_seed(T + K)) * 0.5).bfloat16()
    Aq = to_torch(bfp16.from_panels(bfp16.to_panels(to_numpy(A)), M, K))  # (the GEMM's operand values)

    def outputs(level: int | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """C (T rows, fp32) and an input projection's whole bf16 output Cb: paced at ``level`` (None: the twins)."""
        lib.set_params({PACE: pace_word({op.pace: level or 0 for op in ops})})
        run = (lambda op: twins[id(op)]) if level is None else (lambda op: op)
        if pp.layout is not None:  # (an input projection's GEMMs write one output in turn, as the prefill's: M rows)
            Cb = torch.zeros((M, pp.gemms[0].op.N), dtype=torch.bfloat16)
            for gm in pp.gemms:
                Cb = _run(lib, run(gm.op), A[:gm.op.M], W, gm.w_off, gm.a_off, Cb)
            return Cb[:T].float(), Cb
        Cs: list[torch.Tensor | None] = [None] * pp.parts
        for gm in pp.gemms:  # (the GEMMs of a part write disjoint columns)
            C = _run(lib, run(gm.op), A[:gm.op.M], W, gm.w_off, gm.a_off)[:T].float()
            Cs[gm.part] = C if Cs[gm.part] is None else Cs[gm.part] + C
        return sum(Cs), None

    variants = {level: outputs(level) for level in (0, None, LEVEL)}
    C, Cb = variants[0]

    def agree(f: object) -> None:
        """The values checked (``f(C, Cb)``) bit-identical in every variant (as bits: fp32 halves can be NaN)."""
        def bits(x: torch.Tensor) -> torch.Tensor:
            return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)

        ref = bits(f(C, Cb))
        for level, out in variants.items():
            assert torch.equal(bits(f(*out)), ref), f"{'unpaced' if level is None else f'paced at {level}'} != level 0"

    if phase == "gate_up":
        (b,) = region.blocks
        ffn = cfg.intermediate_size // COLS
        idx = np.concatenate([ffn * c + np.arange(ffn) for c in range(COLS)])
        gt, up = (_rows(img, (t,), idx, K) for t in b.tensors)
        y_g, y_u = Aq[:T] @ gt.T, Aq[:T] @ up.T
        act = (lambda y: torch.nn.functional.gelu(y, approximate="tanh")) if model == "gemma4" else \
            torch.nn.functional.silu
        want = act(y_g.bfloat16().float()) * y_u
        assert rel_native(C[:, idx], want) < 3.5e-2  # (and the bfp16 C panels)
        agree(lambda C, Cb: C[:, idx])
        return
    if phase in PG.OUT:
        (b,) = region.blocks
        h8 = cfg.hidden_size // COLS
        cols = np.concatenate([pitch_c + np.arange(h8) for pitch_c in PG.d_pitch(img, T) * np.arange(COLS)])
        rows = np.concatenate([b.rows[c][b.rows[c] >= 0] for c in range(COLS)])
        want = Aq[:T] @ _rows(img, b.tensors, rows, K).T
        assert rel_native(C[:, cols], want) < _tol(K)
        agree(lambda C, Cb: C[:, cols])
        return
    lay = pp.layout
    for b in region.blocks:
        for c in range(COLS):
            rows = b.rows[c][b.rows[c] >= 0]
            o = lay.pitch * c + lay.off[b.name]
            if b.pairs:  # (fp32 sums of the planes: the F32 rows times the norm's w / bf16(w), as the image packs them)
                got = Cb[:T, o:o + rows.size].contiguous().view(torch.float32).float()
                want = Aq[:T] @ (_rows(img, b.tensors, rows[0::2] // 2) * _ratio(img, region)[None]).T
                assert rel_native(got, want) < 5e-5, (b.name, c, rel_native(got, want))  # (hi + lo: ~2^-16 of a row)
                agree(lambda C, Cb, o=o, n=rows.size: Cb[:T, o:o + n])
                continue
            got = C[:, o + np.arange(rows.size)]
            assert rel_native(got, Aq[:T] @ _rows(img, b.tensors, rows, K).T) < _tol(K), (b.name, c)
            agree(lambda C, Cb, o=o, n=rows.size: C[:, o:o + n])


def _ratio(img: object, region: object) -> torch.Tensor:
    """The norm weight's w / bf16(w) of a region's input (the factor F32 rows are packed with: ``QwenImage.pack``)."""
    w = torch.from_numpy(img.gguf.tensor(f"blk.{region.layer}.attn_norm.weight").dequantize()).reshape(-1)
    bw = w.bfloat16().float()
    return torch.where(bw != 0, w / torch.where(bw != 0, bw, 1), 1)


# (format, tiles per pass): super-block unit formats in passes no test file holds (IQ3_S and IQ4 outside gate / up)
UNIT = [("IQ3_S", 6), ("IQ3_S", 3), ("IQ4_NL", 8), ("IQ4_XS", 4), ("IQ4_XS", 1)]


@pytest.mark.parametrize(("fmt", "tiles"), UNIT, ids=[f"{f}-t{t}" for f, t in UNIT])
def test_unit_gemm(fmt: str, tiles: int) -> None:
    """GemmQd on super-block unit tiles of random GGUF blocks (packed C of two passes per column, K of two chunks; 3 and 2
    MAC rows) against the product of the dequantized weights."""
    from ..tensors import random_tensor

    S, kc, K, T, passes = 9280, 1024, 2048, 192, 2
    R = FORMATS[fmt].rows(S, kc // 256)
    n_rt, w = passes * tiles, tiles * R + tiles * R % 2
    t = random_tensor(fmt, COLS * n_rt * R, K, 7)
    W = np.concatenate([pack_rows([t], np.arange(c * n_rt * R, (c + 1) * n_rt * R), fmt, R, S, K // kc, kc=kc).reshape(-1)
                        for c in range(COLS)])
    pitch = passes * w

    def gemm(mac_rows: int) -> GemmQdOperator:
        return GemmQdOperator(fmt=fmt, rows=R, M=T, K=K, N=COLS * pitch, tile_bytes=S, col_bytes=n_rt * (K // kc) * S,
                              passes=passes, out_cols=[[c * pitch + w * p for p in range(passes)] for c in range(COLS)],
                              kc=kc, kc_group=K // kc, tiles=tiles, c_pack=True, region_rt=n_rt,
                              blocks=((0, passes),), mac_rows=mac_rows)

    ops = [gemm(3), gemm(2)]  # (a library holds two designs at least)
    lib = Library("t_unit_gemm", ops).compile().load()
    A = (torch.randn(max(op.M for op in ops), K, generator=torch.Generator().manual_seed(1)) * 0.5).bfloat16()  # (padded M)
    Aq = to_torch(bfp16.from_panels(bfp16.to_panels(to_numpy(A)), A.shape[0], K))
    want = Aq[:T] @ torch.from_numpy(t.dequantize()).T
    for op in ops:
        C = _run(lib, op, A, W, -op.hdr_skip)[:T].float()
        cols = np.concatenate([c * pitch + w * p + np.arange(tiles * R) for c in range(COLS) for p in range(op.passes)])
        rows = np.concatenate([c * n_rt * R + np.arange(op.passes * tiles * R) for c in range(COLS)])
        assert rel_native(C[:, cols], want[:, rows]) < _tol(K), (op.passes, rel_native(C[:, cols], want[:, rows]))
