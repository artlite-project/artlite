"""The prefills' residual stream: RMSNorms (with the residual update) whose normalized rows go out as bfp16ebs8
panels of 8 tokens x 8 columns, [T / 8][E / 8][72 B] (the layout GemmQOperator reads: half the bytes of bf16 through the
GEMM's A port).

``ResidualNormOperator`` (Gemma): the residual x (T, E) fp32, modes

- ``in`` (the first layer's input): xn = x / rms(x) w1;
- ``post`` (Gemma after the O projection and after a layer's per-layer projection): x' = (x + d / rms(d) w_post) s,
  then xn = x' / rms(x') w1 (the next GEMM's input);
- ``plain`` (Gemma after the down projection): x' = x + d / rms(d) w_post, xn = x' (the per-layer input gate reads the
  residual itself);

with d the projection's output (bf16, column c's E / 8 values at c d_pitch of a row of 8 d_pitch; ``deltas`` = 2, 4:
the sum of K parts, d_stride elements apart), s the layer's output scale (``post`` after the per-layer projection; 1
elsewhere). x' overwrites x.

The arithmetic: d w_post exact in fp32 (d1 w + d2 w for two parts), times a = s / rms(d) (fp32, as bf16 hi / lo; the
product's hi / lo parts: three products), plus x s (x hi / lo times s hi / lo, or x itself when s = 1): within ~2^-16
of the exact update. The norm's input xn = bf16(x') w1 / rms(x') (rms of the bf16
values; 1 / rms as bf16 hi / lo) as the GEMM's single bfp16 panels.

``cols`` cores (one per column) take the token groups of 8 in turn (core c: groups c, c + cols, ..). Per group the
memtile holds x (8 x E fp32) and d and sends them to the core once per pass (``repeat_count``), in chunks of 8 tokens x
``ch`` columns; each pass works in panels (8 tokens x 8 columns: the lanes 8 t + e): pass A sums d's squares (``in``:
x's), pass B computes x' (and its squares; ``plain``: also its panels), pass C x' again (the same
arithmetic) and its panels. The parameters (w_post, w1, s: ``residual_params``, [8 rows][E] bf16: row 0 w_post, row 1
w1, row 2 the hi / lo bits of s) are d's first object.

``PrefillNormOperator`` (Qwen, Llama): for a (T, E) bf16 prompt activation x (T a multiple of 64, E the hidden size: a
multiple of 1024, or above 2048 of 256 columns, Qwen3-Embedding-4B's 2560)
  add:    x += delta (bf16, written back), xn = rmsnorm(x) * w
  no add: xn = rmsnorm(x) * w
with xn as panels.

``cols`` cores (one per column), 8 tokens per step: core c takes token groups c, c + cols, ...
Per group, pass A streams x (and delta) in chunks of 8 tokens x ``ch`` columns (256; E / 8 below 2048: the norm weight
is one chunk-sized object of the x stream), keeps x (+ delta)
in the group's x_out object (16 E bytes, the buffer of pass B) and sums the squares; pass B
gathers 8 x 8 panels, multiplies by w and 1 / rms (bf16 hi + lo) and converts them to
bfp16. The norm weight is the first object of the x stream.

E > 2048 ("replay": a group does not fit L1 next to the streams): x (and delta) reach the core
through its memtile, which holds the group and sends it twice; pass A only sums the squares,
pass B adds x + delta again, writes x_out chunk by chunk and emits the panels. The norm weight
is then the first group object of the x stream (w read 8 times: a group-shaped object).

More than two deltas ("hold": the replay's memtile would hold a group's deltas twice over): each core a contiguous
token range in groups of 4, a group's x + d1 + .. + dn (4 x E bf16, at most 40 KB) held in core L1 between pass A (x
chunks straight from the shim, the deltas' chunks re-laid by the memtile) and pass B, which emits half panels (the
group's 4 tokens of each panel's 8, two per 64-lane vector); the memtile joins two groups' halves into whole panel
rows for the shim; every byte read once.

Runtime sequence: (x, [delta,] w, [x_out,] xn) with x_out = x in place (add mode; ``deltas`` = n > 1:
delta = [d1 | d2 | ..] (``d_stride`` apart), x + d1 + .. + dn summed in fp32 and rounded once, the outputs of a
projection in K parts; two: replay mode whatever E, more: hold mode).
``d_pitch``: delta's column c (E / 8 values) at c d_pitch of a row of 8 d_pitch (a Q6_K down
projection's padding rows after each column's values).

The package: ``prefill_norm`` (``PrefillNormOperator``: its three designs ``plain``, ``replay`` and ``hold``) and
``residual`` (``ResidualNormOperator``).
"""

from .prefill_norm import Design, PrefillNormOperator
from .residual import MODES, RCH, ResidualNormOperator, residual_params

__all__ = ["MODES", "RCH", "Design", "PrefillNormOperator", "ResidualNormOperator", "residual_params"]
