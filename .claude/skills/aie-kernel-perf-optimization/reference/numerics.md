# Numerics on AIE2P: fast and accurate enough

What the vector unit supports natively, how to get fp32-level accuracy from bf16 operations, and the
numerical traps encountered in Artlite's models. Helpers live in `include/kernel_lib.h`
(namespace `al`: exact building blocks) and `operators/kmath/` (`km.exp2`, `km.recip`, `km.sigmoid`, ...: generated per
contract, with every variant tested bit-exact on the NPU against its numpy model in `tests/operators/kmath_sim.py`,
which also models the fp32 adder, bf16 rounding, and hi/lo products for precision studies).

## Contents
- What is native
- fp32 through bf16 splits
- LUTs and their errors
- Recipes: exp2, softmax, SiLU, GELU
- Block float (bfp16ebs8)
- Rounding
- Model-level accuracy rules
- Precision studies: simulate the rounding points

## What is native

- **bf16 x bf16 -> fp32 multiply / MAC** (`vmul.f`, `vmac.f`): 6-cycle latency.
  Dependent MACs on the same accumulator can issue every 3 cycles (`machine-model.md`).
- **fp32 add / sub**, on the 2048-bit accumulators: 64 lanes per op. Not IEEE-compliant: before adding, the smaller
  operand is rounded (nearest-even) onto the larger operand's 23-bit mantissa grid (AM027). During cancellation,
  precision is limited to that grid—for example, `x - round(x)` drops the LSB of `x` whenever `round(x)` steps up to
  the next power of two. A MAC (accumulator + exact bf16 product) uses the same adder when the product is smaller.
  When the product is larger, the accumulator shifts onto a grid ~3 bits finer (matching the larger exponent) before
  rounding the sum. This matches `kmath_sim`'s `mac` on 99.9% of single MACs with cancellation (within 1 ULP on the
  rest), and chains like `al::rot2` match ~99.5% of the time (within 2 ULP).
- **Denormals are flushed to zero** based on the exact (pre-rounding) result of a product, conversion, or MAC sum (the
  MAC's intermediate product is not flushed before adding). Because of this, hi/lo splits of `x` lose the lo part when
  it falls below `2^-126` (`km.sigmoid`'s error increases below `x = -79`).
- **Float literals passed to `aie::broadcast<bfloat16>`** are rounded at compile time if they are not exact bf16
  values; write the exact bf16 value instead.
- **fp32 max**: `aie::max` is emulated (an accfloat subtraction, its sign as a bf16 compare, a select: ~10 cycles per
  16 lanes; a 64-lane clamp costs ~25 cycles and spills registers). `al::fmax` is exact from integer compares in 3
  mv-slot ops (the signed order of fp32 bits, reversed where both are negative). Float min / abs are still emulated:
  - float `aie::neg` / `abs` may fail to legalize (`G_FNEG`): use `min(x, 0 - x)`, or operate on the integer bits
    directly.
- **No fp32 multiply**: `aie::mul` on fp32 vectors is emulated. `aie::inv` / `aie::invsqrt` on an fp32 vector emit one
  scalar `inv` per lane (`aie_api`'s elementary fallback), costing ~3 instructions of program memory per lane; only
  invert the lanes that matter (4 rows of a set, not 16 lanes: `al::rows4_inv`). Both are within 1 fp32 ULP of the correctly rounded
  result (~88% exact), depending deterministically on the mantissa (and exponent parity for `invsqrt`). `kmath_sim`
  models this using measured NPU outputs over `[1, 4)` (`load_inv`, read by `test_kmath`'s fixture). **Scalar float
  math** falls back to soft-float (`__mulsf3`, `__gtsf2`: function calls, 0.9 KB of program memory). `checks.py` warns
  about both.
- **The bfp16 8x8x8 MMUL** (`mac_8x8_8x8T`) is the only hardware matrix path; bf16 `aie::mmul` is emulated.

## fp32 through bf16 splits

`x = hi + lo` with `hi = bf16(x)`, `lo = bf16(x - hi)`: about 16 bits of precision.

| helper | computes | products | error |
|---|---|---|---|
| `al::mulc<N>(x, c)` | fp32 `x` times an exact bf16 constant `c` | 2 | ~2^-16 rel. |
| `al::mul_hilo<N, Hi, Lo>(x)` | fp32 `x` times an fp32 constant `Hi + Lo` | 3 | ~2^-16 rel. |
| `al::mulff<N>(a, b)` | fp32 x fp32 | 3 | ~2^-16 rel. |
| `al::bf16_hilo` | a scale as bf16 hi + lo for a vector-wide multiply | | |

- Each split costs 2 conversions and a subtraction. A chain of splits adds 20-30 cycles of latency.
- `x²` for a bf16 `x` is exact in fp32 with a single `aie::mul(x, x)`.

## LUTs and their errors

| op | lanes per instruction | slot | error |
|---|---|---|---|
| `vexp2` | 16 | mv | the chord `2^n (1 + f)` between integer powers: up to +6% on bf16 inputs, 15% on fp32 |
| `vtanh` | 16 | mv | coarse (max 8.5% rel.); returns its argument below `|x| = 0.5`; bf16 saturation near 1 |
| gather (`load_4x64_lo / hi`) | 4 per instruction | ldb | exact: 64-bit table entries (`km.act`, `kmath/`) |

- `vexp2` matches Schraudolph's algorithm bit-for-bit (verified across all bf16 and fp32 accumulator values):
  `bits = floor(128 y) + 16256`, or `16255 - trunc(128 |y|)` (one's complement) for `y < 0`, where `y` is truncated to
  bf16 (producing 1/4-step quantization and up to 19% error for `|y| >= 32`). It is exact at non-negative integers
  (exact `2^n`, used by the `exp2` recipe below) and `2^n (1 - 2^-8)` at negative integers. Because the error depends
  only on the fractional part, fixing it requires a range split anyway. When softmax only needs an unscaled `P`,
  combining two offset chords (`km.exp2(bias=0, prec="chord2")`: `vexp2(x) + vexp2(x + 1/2) / sqrt 2` — two `vexp2`
  calls, an add, and two products) cancels the chord error to within 1.2% of `2.068 * 2^x` for `|x| <= 2` (minimax
  weights over `|x| <= 2` gave higher error on SD's UNet).
- The gather: lane `k` of an instruction reads table `ab` (`k % 4 < 2`) or `cd`, with odd lanes 32 bytes further;
  offset bits `5:4` select the entry in a 64-byte group and bits `3:0` are ignored. Each table is therefore stored in
  4 copies: groups of 4 entries duplicated in `ab` and `cd`, placed in two 16 KB regions (`bank-conflicts.md`). Each
  base in the address vector must be a separate global or pointer parameter; otherwise the reads lose their memory
  operands (`compiler-feedback.md`).
- Both take the argument truncated to bf16 and share the `mv` slot with compares, selects, and min / max.
- A 32-lane activation with 2 LUT calls, a compare, and a select already needs 5+ `mv` ops per 32 lanes.

## Recipes

### exp2 of fp32 x to 0.4 bf16 ULP (`km.exp2(N)`)

1. `n = round(x)`: add `1.5 * 2^23`, read the float's bits, subtract `0x4B400000`.
2. `f = x - n`, with `|f| <= 0.5`.
3. Compute `2^f` with a polynomial whose coefficients are fitted with bf16 rounding points (degree 2: `c1 = 0.6996`,
   `c2 = 0.236`; or `km.exp2`'s cubic).
4. Shift `n` into the exponent field (`upshift(n, 23)`) and add it to the polynomial's bits. Range handling: either
   clamp `x` at `-126` in fp32 first (a masked score of `-1e30` becomes `2^-126`, or exact `0` from `-127` downward
   with `below="zero"`), or clamp `n` **as an integer** (`below="int"`: avoids fp32 max for inputs above `-2^22`).

No branches, no `-inf` handling. Processes 64 lanes per step at ~55 cycles per 64 lanes; does not software-pipeline,
so batch the work.

### exp2 without a table or fp32 vectors

1. `A = y + 192` in an accumulator (a preceding MAC chain can fold this into its initial constant);
   `v = bf16(A) = 192 + round(y)` for `|y| < 64`; `f = bf16(A - v)`, `|f| <= 1/2` (exact for bf16 `y`; an fp32 `y` loses
   `f`'s low bits, adding +0.17 ULP).
2. `P = vexp2(v - 128) = 2^(n + 64)` exactly (non-negative inputs only, as noted above); the `2^-64` factor is folded
   into the constants.
3. `y = bf16(P * 2^-64 + bf16(P f) * c1' + bf16(P f^2) * hi')`, where `hi' = bf16(c2' + c3' f)` and `c' = c * 2^-64`.
   `P f` and `P f^2` are exact scalings, leaving only one rounding at the end. Fit `c1` first, then fit `c2` and `c3`
   against the rounded `c1` (`0.69140625`, `0.2412109375`, `0.0625`; rounding `c1` last degrades error to 0.72 ULP).

Max error 0.585 ULP, 99.80% correctly rounded over bf16 `y` in `[-63, 63)`; II 10 per 32 lanes (10 MACs, vec-bound),
~370 cycles per 1024 elements on the NPU, bit-exact against its numpy emulation. A quadratic (constant `hi'`:
`c1 = 0.703125`, `c2 = 0.2412109375`) saves 2 MACs: max 0.85 ULP, mean 0.26 (vs. 0.56 / 0.21 for the cubic). For
comparison, `km.exp2`'s fp32 cubic achieves 0.4 ULP at ~880 cycles unpipelined, and a gather-LUT exp would cost ~850
cycles and 3 KB like the SiLU table.

### Softmax in the MMUL panel layout (decode attention)

- Keep scores as 8x8 accumulator panels (lane `8q + k`).
- Compute row max using `shuffle_down_rotate` by 1, 2, and 4 with max.
- Take a per-query integer reference max from the first KV chunk (+1), compute `P = 2^(s - m)` with the clamp above,
  and skip `O` rescaling.
- This breaks if attention sinks appear after the first chunk: in MiniCPM-V's ViT, later scores exceed the first 32
  keys' max by 71 log2 units (saturating `P` at `2^63`; updating the max drops embedding KL from 0.0090 to 0.0074).
  Check each later group against `m + 16` in bf16 and rescale `O` and `l` on the rare updates
  (`ops.Attention(raise_max=)` in the vision overlay, and qwen35 prefill attention). Holding `m` fixed from the first
  chunk broke the 35B-A3B sub-step core on 32K chat prompts (producing garbage tokens at step >= 2), even though raw
  essays stayed clean at every threshold (essay prefixes look like normal text, whereas chat prefixes are system
  prompts). Keep rescaling enabled unless you have specifically tested on chat prompts, and inspect actual generated
  text with and without drafts.
- Take the max of the raw inputs and scale it with the same multiply as the elements; otherwise bf16 rounding can push
  the largest argument above 0.

### SiLU through the gather LUT (`km.act(km.SILU)`: GDN prefill)

- `silu(x) = relu(x) + g(|x|)`, where `g(t) = silu(-t)` on `[0, 24]` is divided into 1/4-wide segments, each
  approximated by a quadratic in the offset `r` from its center (`c0` in fp32, `c1` and `c2` in bf16, packed into one
  64-bit entry; minimax over the segment's bf16 points in result ULPs).
- Range reduction in bf16: `v = bf16(t + 32)` rounds `t` to the grid (`[32, 56]` is a binade with ULP 1/4),
  `r = (32 - v) + t` is exact (compute `32 - v` first, or small `t` values lose their low bits), and the table offset
  is `vfloor(64 v)`.
- Max error 0.59 ULP, 99.84% correctly rounded, ~2000 cycles per 4096 elements on the NPU (compared to 6.6 ULP at
  `x = -2.2` and 2682 cycles for the `vtanh` / `vexp2` version below). Emulation in numpy is bit-exact (`emulate` in
  `tests/operators/elementwise.py`).
- II was limited by register pressure rather than functional slots (~550 variants tested). In a single loop, constant
  vectors (`1`, `24`, the abs mask, `0`, and the lane table pointers) took 5 of the 12 `x` registers, causing
  post-pipeliner register allocation to fail (II 23-30 vs. ResMII 12-15, spilling constants to `lf` / `bm` registers and
  moving them back every iteration). Splitting the loop fixes this: compute `t = min(|x|, 24)` in a separate loop
  (II 2), then run the table pass in-place (II 14 vs. bound 9; post-RA scheduling gets II 21 because regalloc fails below
  that, while a full-table pass gets II 12 post-RA vs. 27 if pipelined first). Shuffle coefficients into 32 lanes using
  `T256_2x2_lo / hi` on two `T16_16x4_lo` results rather than concatenating 256-bit halves (which inserts extra moves).

### Sigmoid through the gather LUT (`km.SIGMOID`: qwen35 attention prep gate)

No odd symmetry to fold: `ActTable(full=True)` tabulates `f` directly over `[-t_max, t_max]` (16 here), using the same
range reduction as SiLU with `x + 48` in a binade of ULP 1/4 (129 segments) and clamping outside that range. Error is
within 0.54 bf16 ULP with 99.9% correctly rounded (`test_sigmoid_lut`); runs in 1718 cycles instead of 10273 cycles per
8 tokens of 256 gates (fp32 `exp2` + reciprocal).

### SiLU (GemmQ's gate/up epilogue: `vtanh` + small-|x| fit, `km.silu_tanh`)

- `vtanh` form (`al::tanh_acc` / `al::exp2_acc`): `u (1 + tanh(u))`, where `u = x / 2`. Below `x = -3`, switch to
  `u * 2^(x log2e) * (1 + |t|)`; the `exp2` tail avoids cancellation against `vtanh`'s saturation. II 10 per 32 lanes.
- `silu_small` fixes `|x| < 2.5` (where `vtanh` error increased first-token KL by 2.6x) using a bf16 fit
  `u + s * (P0 + P1 s + P2 s²)` with `s = u²`. Run it as a **second in-place loop**; combining both into one loop
  prevents pipelining (48-59 cycles per 32 lanes).
- This form is too coarse for a recurrent state: on a 128-token GDN state it produced 2.6x the error of exact SiLU
  (6.7e-3 vs 2.6e-3; starting the `vexp2` tail at `-2.5` raised that to 3.4x). Using the identity `a (1 + t)` for
  `x >= 0` and `a e^x (1 - t)` for `x < 0` (since `1 + t = e^x (1 - t)`, avoiding cancellation), with `e^x` in a
  separate loop (below) and the `silu_tanh` fit for `|x| < 2.5`, drops this to 1.2x at 3.2K cycles per 8 tokens x 288
  values. The LUT SiLU above matches exact SiLU's error and is faster (27B scan: 4.57 vs 4.84 ms/layer; GdnOut: 0.72 vs
  0.89 ms). For fp32 consumers, the unrounded quadratic (`km.act(f32=True)`, ~0.09 bf16 ULP) is the best option at any
  speed: 7e-4 max relative error vs. 3.9e-3 for `km.silu`, at ~1 cycle/element vs. 16.

### e^x of fp32 x in a pipelined loop

The fp32 version of `km.exp` (which applies `km.exp2` to `al::log2e<32>(x)`, or uses two products for bf16 inputs via
`km.exp(arg="bfloat16")`) does not pipeline (86 bundles per 32 lanes due to fp32 vector ops). Because `al::log2e` has
~2^-16 relative error, `e^x` drifts by `~1.6e-5 * |x|` (reaching `2^-9.5` at `|x| = 87` in `km.expneg`). Instead, compute
`x log2 e` as two bf16 products (`log2 e` hi + lo) into an accumulator, apply the `1.5 * 2^23` split, evaluate `2^f`
with a quadratic in bf16 MACs (minimax on `[-0.5, 0.5]`: 0.27% error, ~0.7 bf16 ULP; cubic gets 0.014%), and add the
exponent as integers. This pipelines at II 15 per 32 lanes (1131 vs 6238 cycles per 2304 values; 1997 cycles for the
cubic). On the GDN state, the quadratic's error is negligible (3.14e-3 vs 3.08e-3).

### tanh of fp32 x at fp32 precision, batched (Inkling's decode blend, `decode/design.cc.jinja` `htanh`)

Computed as `sign * (1 - e) / (1 + e)` with `e = e^-2|x|` using `km.exp`'s fp32 form and `km.recip`'s Newton steps.
Called 16 lanes at a time, each call is a single latency-bound chain taking 29 cycles per value. Batching 288 values
and splitting into L1 passes (range reduction, each Taylor step, reciprocal initial guess, each Newton step, and
quotient) at 32 lanes per step cuts this to 15.6 cycles per value. Individual passes do not pipeline because each fp32
chain is a full-length recurrence (e.g., RecMII 51 vs. ResMII 10 on a Newton step). To overlap chains, unroll each pass
by 3 (491 -> 241 cycles), limited by code size and stack space (unrolling the quotient pass spilled into a 1.6 KB stack
frame).

### GELU through the gather LUT (`km.GELU`: Gemma prefill gate/up epilogue and per-layer-input op)

Uses the SiLU table scheme (`kmath.ActTable`: `f(x) = relu(x) + g(min(|x|, t_max))` for functions satisfying
`f(x) - f(-x) = x`) with 65 segments of width `1/8` over `[0, 8]`. Error is within 0.59 bf16 ULP for `x >= -4` and
1.42 ULP in `[-6, -4)` (1.4e-7 absolute; because the tail decays faster than a per-segment quadratic can track, `floor`
sets an absolute error target there). The table takes ~2.2 KB and is passed as a worker `Buffer` argument
(`km.act(table_arg=True)`: SKILL.md section 2). `walk=(2, 128)` reads the gate halves of a GEMM epilogue's gate/up C
tile in place; `walk=(1, 64)` processes the PLE slots; the vision overlay runs a GELU table on SiLU's grid using
`2|x|` (`scale=2`, `clamp`). The clamp pass defaults to a separate noinline function. Setting `inline=True` avoids the
7-cycle call overhead—which helps the PLE—but increases register spills in the GEMM epilogue and vision overlay row
loops. The decode overlay's GELU (`decode/design.cc.jinja`) uses `km.act(km.GELU)`, 32 values at a time: 2.4x faster
than the `vtanh` form (bf16 fit below `|w| = 1.25`, `e^(2w) (1 + |tanh w|)` for `w < 0`), which gets 1.7 ULP at ~7
cycles per element.

## Block float (bfp16ebs8)

- **Layout**: 8 values share one 8-bit exponent (9 bytes per block). `sizeof(bfp16ebs8)` is 9; `sizeof(uint4)` is 1.
  Never do pointer arithmetic on `bfp16ebs8 *` or `uint4 *`; offset a `uint8_t *` and cast when constructing the stream.
- **Loads**: 576-bit blocks load only via FIFO pops (`vlda / vldb.pop.576`). Each input stream occupies one of the 2
  `lf` registers, so at most two input streams can be live at once.
- **Fills and seeks**: `aie_api`'s stream refills once every 8 pops, anchored at pop 0. Calling `seek(p)` with
  `p % 8 != 0` returns garbage later. Either create streams inside the loop iteration, or keep a persistent stream and
  `seek` by multiples of 8 (`seek` resets the counter). Otherwise a stream carried across loop iterations refills before
  every pop.
- **Output streams**: `push` performs a push + flush. Create output streams per iteration.
- **Unsigned mantissas**: unsigned-mantissa MMUL mode (`sgn_x = sgn_y = 0`) represents an 8-bit unsigned significand
  exact.
- **Precision**: 7 bits relative to the block maximum (~1-2% relative error per operand on real Q / K / P / V). For
  bf16-level accuracy, split each operand into bfp16 hi + lo and issue 3 MMULs (`hh + hl + lh`). Exact splitting on the
  MMUL uses a `-I` panel (`restructurings.md` R4).
- **Idempotence**: conversion is not idempotent. A `-128` mantissa keeps its exponent, and converting the value again
  raises the exponent. Host models should insert the kernel's blocks directly rather than re-rounding values.
- **Exact integers**: integers in `[-128, 127]` are represented exactly with exponent byte 133 (scale 1), though
  `bfp16.blocks` avoids this exponent if a block contains `-128` (use `artlite.vision.layout.int_panels` instead).
  MiniCPM-V's encoder uses this to pass 8-bit pixels directly as `(p - 128)`, folding normalization into the patch
  embedding weights and placing the offset in a dedicated constant-input block—avoiding input rounding and making host
  packing cheaper than float panels.
- **Host packing**: use round-to-nearest-even when packing on the host (`bfp16.blocks`), not truncation (as
  mlir-aie's `floatToBfp16` helper does): on a Llama Q4_0 prefill, nearest-even gets KL 0.00085 vs 0.00145 for
  truncation.

## Rounding

- The default rounding mode is floor. Set `aie::set_rounding(aie::rounding_mode::conv_even)` at the entry of every
  kernel that performs rounding; the mode is per-core state that persists across kernels.
- Any kernel that relies on a previous kernel to set the mode will silently truncate every tile when that earlier
  kernel is absent (on a GEMM, KL jumps from 0.0008 to 0.038).
- Floor rounding also introduces bias on bf16 write-backs (~6% at large K).

## Model-level accuracy rules (found on Gemma 4)

- Any scale factor that multiplies an entire vector (such as `1 / rms`) must be passed as bf16 hi + lo
  (`al::bf16_hilo`), never rounded to a single bf16.
- Cached K: using one 7-bit bfp16 block per 8 dims shifts a position's scores coherently, which softmax turns into
  output error on short vectors (Gemma and Llama keep hi + lo blocks for K). At `head_dim` 256 this averages out, so
  qwen35 uses one block per 8 dims for K. On Qwen3.8-27B, K hi alone vs. hi + lo gives: 512 teacher-forced tokens KL
  0.0029 / 0.0030, agreement 0.953 / 0.951 (llama.cpp prompt vs. decode: 0.0031, 0.957); 128 steps after a 4096-token
  prefill gives KL 0.0040 / 0.0032, agreement 0.961 / 0.945. Evaluate over several hundred tokens; looking only at the
  first 128 gave noisy agreement numbers (0.938 / 0.969). V tolerates single blocks.
- Single bfp16 panels are acceptable for GEMM inputs: simulated KL is 0.00059 vs 0.00042 with hi + lo.
- Judge accuracy by KL across multiple prompts or rounding variants; single short runs can vary by up to 40x due to
  rounding luck.
- **Flash attention: store `P = 2^(s - m)` as bfp16 hi + lo panels**, like Q and K. Single bfp16 P blocks (where 8
  positions share an exponent and small weights lose precision) doubled Gemma prefill's cache error at the last
  own-cache layer (K: 5.4% vs 3.7%, V: 8.0% vs 5.2% against fp32) and raised its KL (0.00068 vs 0.00046). With hi + lo
  P, attention error drops down to the bfp16 rounding error of the output itself (single P inflates error by 1.13-1.39x
  on the Llama path and 1.5-3x on qwen35: `prefill.attn.Shape.pp`, `attn_prefill`'s `pp`). In DMA-bound attention, the
  extra MMUL per record adds negligible overhead (Qwen3 0.6B at 512 tokens: +1.3%); when compute-bound, it does
  (Llama 3B's 2048-token chunk at p0 8K goes from 14.5 to 17.6 ms/layer, adding 50% more MMULs per record). Use a
  separate accumulator chain for the lo products when sharing one would serialize (`QB = 1`). Using single 8-row panels
  for Q and P during decode works without accuracy loss when hi + lo won't fit (e.g., qwen35's 8-slot designs:
  Qwen3.8-27B teacher-forced KL is 0.0022-0.0023, matching llama.cpp's 0.0022 prompt-vs-decode baseline).
- **hi + lo weights** where a GEMM is sensitive and cheap: `artlite.vision.ops.Gemm(hilo=)` doubles K (running the hi
  k-tiles, then the lo ones while re-walking the A panels). MiniCPM-V's encoder uses hi + lo weights only at the patch
  embedding and final merger (which together account for a quarter of the total weight error); other layers contributed
  1-3% each and weren't worth the extra compute.
- **A GEMM's bf16 C tile between k-tiles** (`gemm/mac.py`: storing C as bf16 every 64 k) rounds the running sum `K / 64`
  times, with error growing with the number of roundings and the magnitude of the partial sums. Splitting K into parts
  that each emit a bf16 C—summed in fp32 by the consumer (e.g., a residual add)—cuts this error (Gemma's down
  projection, `K = 10240`, split into 4 parts via `prefill.gemm.plan_phase(k_part=)`). Increasing the round-trip interval
  from `K = 64` to `128` did not help measurably.
- **Q4_0 weights as bfp16 panels** (GemmQd's dequant: `q * d` per element, 8 `k` sharing an exponent) round the fp16
  scale `d` into 8-bit mantissas, adding ~0.5% error per weight. Exact-integer panels (used in decode GEMV) avoid this,
  but require applying the scale per group after the MMUL—which the prefill MAC loop lacks spare accumulators to do.
- In Gemma prefill, these three sources (single bfp16 inputs, bfp16 weights, and the bf16 C tile) contribute roughly
  equally (~2-3% each at layer 23 vs fp32), so removing any single one barely changes the total. Fix the outlier (`P`
  above) first.

## Precision studies: simulate the rounding points

Before changing kernels for accuracy, rank candidates on the CPU: subclass the fp32 reference model and insert the
kernel rounding points where they occur (one flag per rounding point: GEMM inputs as bfp16 / hi + lo, weights as bfp16
panels, bf16 outputs, the C tile per k-tile, K parts, and P/V blocks in attention).

- Measure what the acceptance test measures: per-layer cache error against the fp32 caches, and KL of teacher-forced
  tokens after the prompt.
- Average over several prompts: a single prompt's error at the final layers varies by ~+-30% due to rounding luck, and a
  64-token KL mean can be dominated by a few outlier tokens (e.g., max 0.014 at mean 0.0006).
- Rank rounding points one at a time against the exact fp32 model. If you toggle a single site inside the fully
  quantized model, that change cascades and re-rolls all downstream rounding decisions—so a single run mostly measures
  noise (in MiniCPM-V's encoder, toggling any early site shifted KL by the same ~0.01). Single-site errors also combine
  cleanly in quadrature, unlike perturbations in the full model.
- A few sensitive tokens can dominate squared error (in MiniCPM-V's ViT, the worst 1% of tokens accounted for 67-80%
  of the error and shifted by 57-99% under tiny changes), so rank variants by median per-token error instead. These
  outliers can come from the model itself rather than a specific op or rounding point (e.g., an FFN half switching on a
  massive activation in vision ViTs—see `artlite.vision.vit`'s `Precision`; llama.cpp fails on the same tokens). To
  separate model sensitivity from kernel bugs: dump the NPU residual after each layer, run each layer in isolation from
  float64 inputs on CPU with matching rounding points, and compare per-token error against llama.cpp.
- llama.cpp's CPU path quantizes Q4_K weights' activations to Q8, so any change to the input re-rolls that rounding,
  creating a KL floor (~0.005 on MiniCPM-V 1B over 96 tokens). Its F32 requant
  (`llama-quantize --allow-requantize IN OUT F32`) avoids this floor.
- Average KL over multiple noise realizations: adding a tiny dither (`1e-6` relative, or single bf16 ULPs on 0.1% of an
  input) re-rolls all downstream rounding; 10-20 runs per config resolve ~5% differences. Report prompts separately when
  their KL hinges on a single near-tie token (e.g., reading a month from small print in MiniCPM-V): minor numerical
  biases tip the tie one way on CPU and the other on the NPU, even when all other prompts agree.
- Check the simulated model against the NPU layer by layer. Whenever they diverge, a rounding point is missing from the
  simulation (this is how attention's `P` was found: the simulation matched the NPU through layer 17 and drifted after
  that until `P` was modeled).
- `bfp16.values` emulates the conversion; insert blocks where the kernel converts and never re-round (see above).