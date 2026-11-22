"""The prefill GEMMs (prefill/gemm.py: GemmQ / GemmQd) on the models' decode images (Llama's ``LlamaImage``,
qwen35's ``QwenImage``, Gemma 4's ``GemmaImage``: slices of a few layers, read in place): each phase's GEMMs as the
model's prefill plans them, with random bfp16 activations, against the dequantized GGUF rows: input projections at
their column-slice layout, output projections at d_pitch (K parts: the parts' sum), gate/up's act(gate) up panels.
The files cover the tile formats: Q4_K (tile-major GemmQ), Q5_K / Q6_K (planar GemmQd, 13 / 11 / 12-row tiles), Q4_0 /
Q4_1 (planar at 1024 columns; Gemma's at 512), Q4_1_64 (tile-major: Flint's), P4_128 (tile-major: Qwen3-0.6B-PARO's), P5_128 (planar: Qwen3.8-27B-PARO-int5's, 13-row tiles), Q2_K / Q3_K (word-major planar), BFP16 (tile-major, 8-row tiles of a BF16
file), Q8_0 (super-block units: Occamy's GDN z; vllm-d3's d3-lite: GDN in, attn in, gate/up in half groups, down; Gemma's
16-row copy of the per-layer projection), F32 rows as BFP16 plane pairs (Occamy's GDN alpha: fp32 sums); LFM3's conv layers'
in / out and its qkv with a gate block of one tile (two rows a column). The GEMMs
run in a library program as the prefill's do, paced (``gemm.pacing``) at level 0 and at ``LEVEL``, and as unpaced
twins: their outputs bit-identical."""


import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import artlite
from artlite.decode.image import pack_rows
from artlite.formats import FORMATS
from artlite.gemm.gemm_qd import GemmQdOperator
from artlite.gemm.pacing import PACE, pace_word
from artlite.gguf_model import GGUFFile
from artlite.hf_model import open_weights
from artlite.model_config import (
    checkpoint,
    decision_gguf,
    flint_gguf,
    gemma_gguf,
    lfm3_gguf,
    llama_gguf,
    occamy_gguf,
    qwen3_gguf,
    qwen_gguf,
    spark_gguf,
    swift_gguf,
)
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.fused import Library
from artlite.prefill import gemm as PG

from ..common import to_numpy, to_torch
from ..model_files import require
from ..numerics import rel_native

LEVEL = 3  # (a paced level: the paced copy of each worker body)

# (model, GGUF quantization, size, layers to pack, the layer and phase tested, tokens)
CASES = [("llama", "Q4_K_M", "1B", (0, 3), 3, "qkv", 64), ("llama", "Q4_K_M", "1B", (0, 3), 3, "gate_up", 256),
         ("llama", "Q4_K_M", "3B", (0, 3), 3, "down", 512), ("llama", "Q4_K_S", "8B", (0,), 0, "gate_up", 256),
         ("llama", "Q5_K_M", "1B", (0, 3), 3, "qkv", 192), ("llama", "Q5_K_M", "1B", (0, 3), 3, "o", 64),
         ("llama", "Q6_K", "1B", (0,), 0, "gate_up", 192), ("llama", "Q6_K", "1B", (0,), 0, "down", 64),
         ("llama", "Q4_0", "1B", (0,), 0, "qkv", 64), ("llama", "Q4_0", "1B", (0,), 0, "gate_up", 256),
         ("llama", "Q2_K", "3B", (0,), 0, "down", 64), ("llama", "Q3_K_M", "3B", (0,), 0, "qkv", 64),
         ("llama", "BF16", "1B", (0,), 0, "qkv", 256), ("llama", "BF16", "1B", (0,), 0, "down", 64),
         ("llama", "Q4_1", "1B", (0,), 0, "gate_up", 256),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gdn_in", 64), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gdn_out", 64),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 0, "gate_up", 64), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 4, "down", 256),
         ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 3, "attn_in", 256), ("qwen35", "Q4_K_M", "9B", (0, 3, 4), 3, "attn_out", 64),
         ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 0, "gdn_in", 256), ("qwen35", "Q4_K_M", "27B", (0, 3, 8), 0, "down", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "qkv", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 5, "o", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate_up", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 0, "down", 288),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate", 256), ("gemma4", "Q4_0", "E4B", (0, 5), 0, "proj", 256),
         ("gemma4", "Q4_0", "E4B", (0, 5), 0, "gate_up", 1024), ("gemma4", "Q4_0", "E4B", (0, 5), 5, "qkv", 1024),
         ("gemma4", "Q4_0", "E4B", (0, 5), -1, "ple", 256),
         ("occamy", "", "", (0, 3), 0, "gdn_in", 256), ("occamy", "", "", (0, 3), 0, "gdn_out", 256),
         ("occamy", "", "", (0, 3), 3, "attn_in", 256),
         ("d3-lite", "", "", (0, 3), 0, "gdn_in", 256), ("d3-lite", "", "", (0, 3), 0, "gate_up", 256),
         ("d3-lite", "", "", (0, 3), 0, "down", 256), ("d3-lite", "", "", (0, 3), 3, "attn_in", 64),
         ("qwen3", "", "embedding-4B", (0, 4), 4, "down", 256), ("qwen3", "", "embedding-4B", (0, 4), 4, "gate_up", 256),
         ("qwen3", "", "embedding-0.6B", (0,), 0, "down", 256),
         ("swift", "", "", (0, 16), 0, "gate_up", 256), ("swift", "", "", (0, 16), 0, "gdn_in", 256),
         ("lfm", "QAD-Q4_0", "", (0, 2), 0, "in", 256), ("lfm", "QAD-Q4_0", "", (0, 2), 0, "out", 256),
         ("lfm", "QAD-Q4_0", "", (0, 2), 2, "qkv", 256), ("lfm", "QAD-Q4_0", "", (0, 2), 0, "gate_up", 256),
         ("lfm", "Q4_K_M", "", (1, 2), 1, "down", 256), ("lfm", "Q8_0", "", (0, 2), 2, "qkv", 256),
         ("lfm", "BF16", "", (0, 2), 0, "in", 256),
         # Spark-X: the gate rows (BFP16, or the F32 gate's BF16 copy) after q, k, v; the 4B's hidden 2560 and the 1.7B's
         # ffn 6656 padded to whole K chunks (3072, 7168); gate / up with GELU's erf form (GemmQ, GemmQd; a split block's
         # activation: ``test_gate_up.py``)
         ("spark", "Q4_K_M", "4B", (0, 3), 0, "attn_in", 256), ("spark", "Q4_K_M", "4B", (0, 3), 0, "gate_up", 256),
         ("spark", "Q4_K_M", "4B", (0, 3), 0, "down", 256), ("spark", "Q4_K_M", "1.7B", (0, 3), 3, "down", 1024),
         ("spark", "Q8_0", "4B", (0,), 0, "gate_up", 256), ("spark-bartowski", "Q4_0", "4B", (0,), 0, "gate_up", 256),
         # (the image's regions of half a chunk short, ``SparkImage(tails=True)``: their tail tiles' halves; gate / up's
         # Q4_K and Q4_0 blocks in pairs of row tiles: pass pairs)
         ("spark-tails", "Q4_K_M", "4B", (0, 4), 0, "gate_up", 256), ("spark-tails", "Q8_0", "4B", (0,), 0, "gate_up", 256),
         ("spark-bartowski-tails", "Q4_0", "4B", (0,), 0, "gate_up", 256),
         # (the Qwen3-4B family's and Kev-4B's half regions: "-halves" a readout model's image (every tail tile read for
         # its first half), "-tails" a decoding model's (gate / up in pass pairs); qkv's Q4_K and planar Q6_K, gdn_in's
         # planar Q8_0)
         ("qwen3-halves", "", "embedding-4B", (0, 4), 0, "qkv", 256),
         ("qwen3-halves", "", "embedding-4B", (0, 4), 4, "gate_up", 1024),
         ("qwen3-tails", "", "embedding-4B", (0, 4), 4, "gate_up", 256),
         ("kev-halves", "", "", (0, 3), 0, "gdn_in", 256), ("kev-tails", "", "", (0, 3), 0, "gate_up", 1024),
         # (down half a chunk short in one K group: MiniCPM-V 1B's and d3-lite's ffn 3584, unpaired; gate / up's rows
         # of 3584)
         ("qwen35-halves", "Q4_K_M", "1B", (0, 3), 0, "down", 256), ("qwen35-halves", "Q4_K_M", "1B", (0, 3), 0, "gate_up", 256),
         ("d3-lite-halves", "", "", (0, 3), 0, "gate_up", 256),
         # Flint's official file: Q4_1_64 (tile-major, the compact header); o's K 4096, down's 6144
         ("flint", "int4", "", (0, 2), 0, "qkv", 256), ("flint", "int4", "", (0, 2), 0, "o", 256),
         ("flint", "int4", "", (0, 2), 0, "gate_up", 256), ("flint", "int4", "", (0, 2), 0, "down", 256),
         # P4_128 (a PARO checkpoint's linears: tile-major GemmQ, the K of each group in its rotation's storage order)
         ("qwen3-paro", "P4_128", "0.6B", (0, 3), 0, "q", 64), ("qwen3-paro", "P4_128", "0.6B", (0, 3), 3, "gate", 256),
         ("qwen3-paro", "P4_128", "0.6B", (0, 3), 0, "down", 256), ("qwen3-paro", "P4_128", "0.6B", (0, 3), 3, "o", 64),
         # P5_128 (int5: GemmQd, Q5_K's dequant; 13-row tiles of a qwen35 image): Qwen3.8-27B-PARO-int5's
         ("qwen35-paro5", "P5_128", "27B", (0, 3), 0, "gin_q", 256), ("qwen35-paro5", "P5_128", "27B", (0, 3), 0, "gdn_out", 256),
         ("qwen35-paro5", "P5_128", "27B", (0, 3), 0, "gate", 128), ("qwen35-paro5", "P5_128", "27B", (0, 3), 3, "down", 256)]

_images: dict[tuple, tuple] = {}


def _base(model: str) -> str:
    """A case's model without its layout suffix ("-tails": half regions and the image's pairs; "-halves": without
    pairs)."""
    return model.removesuffix("-tails").removesuffix("-halves")


def _image(model: str, quant: str, size: str, layers: tuple[int, ...]) -> tuple[GGUFFile, object, np.ndarray]:
    """The model's image of the layers (a truncated vocabulary), packed; its half regions as the suffix says (none
    without one: padded to whole chunks)."""
    base = _base(model)
    path = {"llama": lambda: llama_gguf(quant, size), "qwen35": lambda: qwen_gguf(quant, size),
            "gemma4": lambda: gemma_gguf(quant), "occamy": occamy_gguf, "d3-lite": decision_gguf, "swift": swift_gguf,
            "kev": lambda: decision_gguf("kev"), "qwen3": lambda: qwen3_gguf(size), "lfm": lambda: lfm3_gguf(quant),
            "spark": lambda: spark_gguf(size, quant), "spark-bartowski": lambda: spark_gguf(size, quant, "bartowski"),
            "flint": lambda: flint_gguf(quant), "qwen3-paro": lambda: checkpoint("qwen3-0.6B-paro"),
            "qwen35-paro5": lambda: checkpoint("qwen3.8-27B-paro-int5")}[base]()
    require(path)
    key = (path, layers, model)
    if key not in _images:
        _images.clear()
        g = open_weights(path)
        layout = {"tails": model != base}
        if base in ("llama", "qwen3", "qwen3-paro"):
            from models.llama.image import LlamaImage as Image
        elif base in ("qwen35", "occamy", "d3-lite", "swift", "kev", "qwen35-paro5"):
            from models.qwen35.image import QwenImage as Image
        elif base == "lfm":
            from models.lfm.image import LfmImage as Image
        elif base.startswith("spark"):
            from models.spark.image import SparkImage as Image
        elif base == "flint":
            from models.flint.image import FlintImage as Image
        else:
            from models.gemma4.image import GemmaImage as Image
        if base in ("llama", "qwen3", "qwen3-paro", "qwen35", "occamy", "d3-lite", "swift", "kev"):
            layout["pairs"] = model.endswith("-tails")
        elif not base.startswith("spark"):
            layout = {}
        img = Image(g, layers=list(layers), vocab=COLS * 64, **layout)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W)
        _images[key] = (g, img, W)
    return _images[key]


def _plan(model: str, img: object, region: object, T: int) -> PG.PhasePlan:
    """The phase's GEMMs with the model's prefill options (Gemma: GELU, down in K parts)."""
    pitch = PG.d_pitch(img, T) if region.name in PG.OUT else 0
    if model == "gemma4":
        from models.gemma4.prefill import plan_gemm
    elif _base(model) in ("llama", "qwen3", "qwen3-paro"):
        from models.llama.prefill import plan_gemm
    elif model == "lfm":
        from models.lfm.prefill import plan_gemm
    elif model.startswith("spark"):
        from models.spark.prefill import plan_gemm
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
    M, K = max(gm.op.M for gm in pp.gemms), region.K_in  # (GemmQd pads the tokens to whole passes)
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
    if model != _base(model):  # (the padded image's GEMMs on A's columns and zeros: bit for bit, the k-tiles less
        # exact zeros)
        _, pimg, PW = _image(_base(model), quant, size, layers)
        ppp = _plan(model, pimg, pimg.region(layer, phase), T)
        pops = list(dict.fromkeys(gm.op for gm in ppp.gemms))
        plib = Library("t_prefill_gemm_pad", [*pops, *map(_twin, pops)]).compile().load()
        Ap = torch.nn.functional.pad(A, (0, pimg.region(layer, phase).K - K))
        if ppp.layout is not None:  # (the blocks' valid rows: a pass's rows past its block read other tiles)
            assert ppp.layout == pp.layout
            Pb = torch.zeros((M, ppp.gemms[0].op.N), dtype=torch.bfloat16)
            for gm in ppp.gemms:
                Pb = _run(plib, gm.op, Ap[:gm.op.M], PW, gm.w_off, gm.a_off, Pb)
            cols = np.concatenate([pp.layout.pitch * c + pp.layout.off[b.name] + np.arange(int((b.rows[c] >= 0).sum()))
                                   for b in region.blocks for c in range(COLS)])
            assert torch.equal(Pb[:T, cols].view(torch.int16), Cb[:T, cols].view(torch.int16)), "the padded image's"
        else:
            P = sum(_run(plib, gm.op, Ap[:gm.op.M], PW, gm.w_off, gm.a_off)[:T].float() for gm in ppp.gemms)
            n = C.shape[1]  # (gate / up's rows: down's K, a half region's of fewer columns; the padded image's zero past)
            assert torch.equal(P[:, :n].view(torch.int32), C.view(torch.int32)), "the padded image's output"
            assert not P[:, n:].any()
        _image(model, quant, size, layers)

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
            torch.nn.functional.gelu if model.startswith("spark") else torch.nn.functional.silu
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


# (format, tiles per pass): super-block unit formats in passes no test file holds (IQ3_S and IQ4 outside gate / up, IQ2)
UNIT = [("IQ3_S", 6), ("IQ4_NL", 8), ("IQ4_XS", 4), ("IQ2_XXS", 6), ("IQ2_XS", 3), ("IQ2_S", 6), ("IQ1_S", 4),
        ("IQ1_M", 3)]


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


def test_gate_up_pair_tails() -> None:
    """A PARO Gemma's gate and up (K 2560 of tiles of 1024 columns, their row tiles in pairs sharing the tail tile,
    ``Block.pt``): ``plan_gate_up_pair``'s GEMMs of the pairs' even and odd row tiles (``GemmSpec.ptail_part``) into GU
    as [gate | up] halves of bf16 panels of 64: every row of both against the dequantized rows (the operand's
    values)."""
    path = checkpoint("gemma-4-E4B-paro")
    require(path)
    from models.gemma4.image import GemmaImage

    g = open_weights(path)
    img = GemmaImage(g, layers=[0], vocab=COLS * 64)
    W = np.zeros(img.nbytes, np.uint8)
    img.pack(W)
    gate, up = img.region(0, "gate"), img.region(0, "up")
    assert gate.blocks[0].pt and PG.gate_up_split(gate, up)
    T, K, ffn = 256, gate.K_in, img.cfg.intermediate_size
    pp = PG.plan_gate_up_pair(img, gate, up, T, {})
    ops = list(dict.fromkeys(gm.op for gm in pp.gemms))
    assert len(pp.gemms) == 4
    lib = Library("t_gate_up_pairs", ops).compile().load()
    A = (torch.randn(T, K, generator=torch.Generator().manual_seed(5)) * 0.5).bfloat16()
    Aq = to_torch(bfp16.from_panels(bfp16.to_panels(to_numpy(A)), T, K))
    GU = torch.zeros((T * 2 * ffn,), dtype=torch.bfloat16)
    for gm in pp.gemms:
        dev = artlite.Device()
        w = np.zeros(gm.op.w_bytes, np.uint8)
        a, b = max(gm.w_off, 0), min(gm.w_off + gm.op.w_bytes, W.size)
        w[a - gm.w_off:b - gm.w_off] = W[a:b]
        bufs = [artlite.from_numpy(bfp16.to_panels(to_numpy(A)), dev), artlite.from_numpy(w.view(bfloat16), dev),
                artlite.from_numpy(to_numpy(GU), dev)]
        lib.run(lib.items([(gm.op, "A", "W", "C")], dict(zip("AWC", (x._raw for x in bufs)))), bufs=[x._raw for x in bufs])
        bufs[2].sync_from_device()
        GU = to_torch(bufs[2].numpy().copy())
    got = GU.view(T // 8, 2, ffn // 8, 8, 8).permute(1, 0, 3, 2, 4).reshape(2, T, ffn).float()  # ([gate | up][t][row])
    for i, r in enumerate((gate, up)):
        want = Aq @ _rows(img, r.blocks[0].tensors, np.arange(ffn), K).T
        err = rel_native(got[i], want)
        print(r.name, f"{err:.2e}")
        assert err < _tol(K), (r.name, err)


def test_gate_up_dual() -> None:
    """The same gate and up in GEMMs of both (``plan_gate_up_pair``'s ``act``, ``GemmSpec.dual``: per parity a pass
    pair, gate's tiles on A's first K columns, up's on the next K) whose epilogue forms h = gelu(bf16(gate)) up, bf16
    panels of 64: against the dequantized rows on the two inputs (the operands' values)."""
    path = checkpoint("gemma-4-E4B-paro")
    require(path)
    from models.gemma4.image import GemmaImage

    g = open_weights(path)
    img = GemmaImage(g, layers=[0], vocab=COLS * 64)
    W = np.zeros(img.nbytes, np.uint8)
    img.pack(W)
    gate, up = img.region(0, "gate"), img.region(0, "up")
    T, K, ffn = 256, gate.K_in, img.cfg.intermediate_size
    pp = PG.plan_gate_up_pair(img, gate, up, T, {}, 2 * K, act="gelu")
    assert len(pp.gemms) == 2
    lib = Library("t_gate_up_dual", [gm.op for gm in pp.gemms]).compile().load()
    A = (torch.randn(T, 2 * K, generator=torch.Generator().manual_seed(6)) * 0.5).bfloat16()
    Aq = to_torch(bfp16.from_panels(bfp16.to_panels(to_numpy(A)), T, 2 * K))
    H = torch.zeros((T * ffn,), dtype=torch.bfloat16)
    for gm in pp.gemms:
        dev = artlite.Device()
        w = np.zeros(gm.op.w_bytes, np.uint8)
        a, b = max(gm.w_off, 0), min(gm.w_off + gm.op.w_bytes, W.size)
        w[a - gm.w_off:b - gm.w_off] = W[a:b]
        bufs = [artlite.from_numpy(bfp16.to_panels(to_numpy(A)), dev), artlite.from_numpy(w.view(bfloat16), dev),
                artlite.from_numpy(to_numpy(H), dev)]
        lib.run(lib.items([(gm.op, "A", "W", "C")], dict(zip("AWC", (x._raw for x in bufs)))), bufs=[x._raw for x in bufs])
        bufs[2].sync_from_device()
        H = to_torch(bufs[2].numpy().copy())
    got = H.view(T // 8, ffn // 8, 8, 8).permute(0, 2, 1, 3).reshape(T, ffn).float()
    yg = (Aq[:, :K] @ _rows(img, gate.blocks[0].tensors, np.arange(ffn), K).T).bfloat16().float()
    yu = Aq[:, K:] @ _rows(img, up.blocks[0].tensors, np.arange(ffn), K).T
    want = torch.nn.functional.gelu(yg, approximate="tanh") * yu
    err = rel_native(got, want)
    print(f"h {err:.2e}")
    assert err < 2 * _tol(K), err
