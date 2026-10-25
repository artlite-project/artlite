"""Weight tile formats of the GGUF weight images (the decode overlays, the prefill GEMMs): the registry of the format
slots.

A slot (``common.Slot``, one module per format family: ``q4`` Q4_0 / Q4_1, ``q4k``, ``q5k``, ``q6k``, ``bf16`` BFP16
(of a BF16 tensor), ``q2k``, ``q3k``, ``q8`` (Q8_0; its 512-column tiles decode only), ``iq3`` (IQ3_XXS); ``k23``:
their shared parts, a slot without formats)
holds for its formats the ``TileFormat`` records (tile geometry), the host-side packing (a lossless re-layout of the
GGUF fields; only fp16 -> bf16 scales change, BFP16 rounds bf16 weights once), the kernel-source fragments of the
prefill GEMMs (``gemm/<fmt>``: ``gemm_q``, ``gemm_qd``) and of the decode GEMV (``mmv/<fmt>``: ``overlay.gemv``, which
dispatches on ``TileFormat.fid``), and the hooks of both.

Tile: R rows x one K chunk of ``nsb`` super-blocks (256 columns each; the image's chunk: 1024 columns = 4
super-blocks for Llama and Qwen, 512 for Gemma). Every format stores its integer codes as 32-bit words of 8
consecutive k (one row, one quantization group). Within a row, super-block ``sb`` (256 k) holds 32 words in the order
(s, i), word (s, i) = k in [32 i + 8 s, +8): 64 vector lanes are 8 groups (i) x 8 consecutive elements, and step
s = 0..3 walks the 32 elements of each group.

Layout: the per-row scale planes come first ("header", R x ``header_sb`` B per
super-block), then the codes super-block major (``slice_bytes``: one super-block of all
rows), so a GEMM fetches one super-block of several tiles as long DDR runs (the slots'
docstrings give each format's bytes per row and super-block).
"""

import math

import numpy as np
from jinja2 import FunctionLoader, Template

from . import bf16, common, iq3, k23, q2k, q3k, q4, q4k, q5k, q6k, q8
from .common import KC, KT, SB, DqKernels, Slot, TileFormat

__all__ = ["FORMATS", "KC", "KT", "SB", "SLOTS", "DqKernels", "Slot", "TileFormat", "choose_tile_bytes", "gu_tiles",
           "has_tile_format", "kernel_template", "pack_tile", "slice_fields", "slot", "tile_format"]

# the slots
SLOTS: tuple[Slot, ...] = (q4.SLOT, bf16.SLOT, q4k.SLOT, q5k.SLOT, q6k.SLOT, k23.SLOT, q2k.SLOT, q3k.SLOT, q8.SLOT,
                           iq3.SLOT)

FORMATS: dict[str, TileFormat] = {f.name: f for f in sorted((f for s in SLOTS for f in s.formats), key=lambda f: f.fid)}
_SLOT_OF = {f.name: s for s in SLOTS for f in s.formats}

# Jinja fragments by include name
SOURCES: dict[str, str] = dict(common.SOURCES)
for _s in SLOTS:
    assert not set(_s.sources) & set(SOURCES), set(_s.sources) & set(SOURCES)
    SOURCES.update(_s.sources)


def slot(fmt: str) -> Slot:
    """The slot of a tile format."""
    return _SLOT_OF[fmt]


def tile_format(qtype: str) -> str:
    """Tile format of a GGUF tensor type (BF16 and F16 tensors: BFP16, their values rounded once to bfp16ebs8)."""
    fmt = "BFP16" if qtype in ("BF16", "F16") else qtype
    if fmt not in FORMATS:
        raise NotImplementedError(f"GGUF tensor type {qtype} has no tile format (supported: {sorted(FORMATS)})")
    return fmt


def has_tile_format(qtype: str) -> bool:
    """A GGUF tensor type with a tile format (``tile_format``)."""
    return qtype in ("BF16", "F16") or qtype in FORMATS


def pack_tile(fmt: str, f: dict[str, np.ndarray], tile_bytes: int) -> np.ndarray:
    """One tile from the fields of R rows x one K chunk (QTensor.fields sliced; the chunk's
    super-blocks nsb = columns / 256)."""
    s = slot(fmt)
    q = f[s.field]
    R, nsb = q.shape[0], q.shape[1] // SB
    out = np.zeros(tile_bytes, np.uint8)
    data = np.concatenate([x.reshape(-1) for x in s.pack(fmt, f, R, nsb)])
    assert data.size == FORMATS[fmt].used_bytes(R, nsb) <= tile_bytes
    out[:data.size] = data
    return out


def _load(name: str) -> str:
    # (one newline more: Jinja drops a template's last one, an included fragment keeps its lines whole)
    return SOURCES[name] + "\n"


def kernel_template(source: str, **globals: object) -> Template:
    """A kernel template (``base.JINJA_ENV``) that includes the slots' fragments (``{% include "gemm/" ~ fmt %}``)
    with the including template's variables."""
    from ..base import JINJA_ENV

    env = JINJA_ENV.overlay(loader=FunctionLoader(_load))
    return env.from_string(source, globals=globals)


# ----------------------------------------------------------------------
# Tile layout helpers (the weight images of the decode overlays)
# ----------------------------------------------------------------------


def gu_tiles(r_gu: int) -> int:
    """Tiles of a gate/up SiLU group (128 rows: 8 x [8 gate | 8 up]) of R_gu-row tiles; a group that is not
    a whole number of tiles ends in padding rows (Q5_K's 13: 10 tiles, 2 padding rows)."""
    return -(-128 // r_gu)


def choose_tile_bytes(formats: dict[str, int], nsb: int = 8, gu: dict[str, int] | None = None) -> int:
    """Object size minimizing the streamed bytes; formats: {fmt: rows streamed} (rows counted
    per K chunk of nsb super-blocks); gu: {fmt: gate/up SiLU groups streamed (per K chunk)}, each
    ``gu_tiles`` tiles. A format with a fixed tile size (BFP16: 8 rows) imposes it."""
    gu = gu or {}
    formats = {**{f: 0 for f in gu}, **formats}
    fixed = {FORMATS[f].tile_bytes(nsb) for f in formats} - {None}
    if fixed:
        assert len(fixed) == 1, f"formats with different fixed tile sizes: {formats}"
        return fixed.pop()
    best = None
    lo, hi = 2304 * nsb, 2568 * nsb  # 18432 .. 20544 for 2048-column tiles
    for fmt in formats:
        for r in range(8, FORMATS[fmt].max_rows + 1):
            s = -(-FORMATS[fmt].used_bytes(r, nsb) // 64) * 64
            if s < lo or s > hi:
                continue
            if any(FORMATS[f].rows(s, nsb) < 8 for f in formats):
                continue
            total = sum(math.ceil(n / FORMATS[f].rows(s, nsb)) * s for f, n in formats.items())
            total += sum(g * gu_tiles(FORMATS[f].rows(s, nsb)) * s for f, g in gu.items())
            if best is None or total < best[0]:
                best = (total, s)
    assert best is not None, formats
    return best[1]


def slice_fields(f: dict[str, np.ndarray], rows: np.ndarray, *, k0: int, cols: int, kc: int) -> dict[str, np.ndarray]:
    """Rows and the kc-column chunk at k0 of QTensor.fields() arrays (columns past ``cols``: zero fields, a K padded
    to whole chunks as the packer pads it)."""
    out = {}
    for k, v in f.items():
        per = 1 if k in ("q", "w") else cols // v.shape[1] if v.ndim == 2 else 256  # (columns per entry along K)
        x = v[rows, k0 // per:(k0 + kc) // per]
        n = kc // per - x.shape[1]
        out[k] = np.concatenate([x, np.zeros((x.shape[0], n, *x.shape[2:]), x.dtype)], axis=1) if n else x
    return out
