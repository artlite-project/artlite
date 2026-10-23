# Numerics on AIE2P: fast and accurate enough

What the vector unit does natively, how to get fp32-level results from bf16 operations, and the
accuracy traps met in Artlite's models. The helpers live in `operators/kernel_lib.h`
(namespace `al`: exact building blocks) and `operators/kmath.py` (`km.exp2`, `km.recip`, `km.sigmoid`, ...: generated per
contract, each variant tested on the NPU bit-exact against its numpy model, `tests/rtc/kmath_sim.py`, which also
models the fp32 adder, bf16 rounding and the hi / lo products for precision studies).

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

- **bf16 x bf16 -> fp32 multiply / MAC** (`vmul.f`, `vmac.f`): the result lands at +6 cycles.
  Dependent MACs on one accumulator run every 2 cycles.
- **fp32 add / sub**, on the 2048-bit accumulators: 64 lanes per op. Not IEEE: the smaller operand
  is rounded (to nearest even) onto the larger's grid of 23 fractional bits before the add (AM027),
  so a cancellation keeps only that grid: x - round(x) loses x's last bit where round(x) is a power
  of two above x. A MAC's sum (accumulator + exact bf16 product) is the same adder.
- **Denormals are flushed to zero**, judged by the exact result: a product, a MAC's sum (of the exact product, not
  flushed itself) before its rounding, a conversion. So x hi / lo splits lose the lo part where it falls below 2^-126
  (`km.sigmoid`'s error rises below x = -79).
- **A bf16 constant written as a float literal** that is no bf16 value is rounded at compile time
  (`aie::broadcast<bfloat16>`): write the exact bf16 value.
- **fp32 max** lowers to a bf16 compare plus a select. Float min / max / abs are expensive:
  - a 64-lane clamp cost ~25 cycles and spilled;
  - float `aie::neg` / `abs` may not legalize (G_FNEG): use `min(x, 0 - x)`, or integer ops on
    the bits.
- **No fp32 multiply**: `aie::mul` on fp32 vectors is emulated. **Scalar float math** is
  soft-float (`__mulsf3`, `__gtsf2`: calls, 0.9 KB of program memory). `checks.py` warns about
  both.
- **The bfp16 8x8x8 MMUL** (`mac_8x8_8x8T`) is the only hardware matrix path; bf16 `aie::mmul`
  is emulated.

## fp32 through bf16 splits

x = hi + lo with hi = bf16(x), lo = bf16(x - hi): about 16 bits.

| helper | computes | products | error |
|---|---|---|---|
| `al::mulc<N>(x, c)` | fp32 x times an exact bf16 constant c | 2 | ~2^-16 rel. |
| `al::mul_hilo<N, Hi, Lo>(x)` | fp32 x times an fp32 constant Hi + Lo | 3 | ~2^-16 rel. |
| `al::mulff<N>(a, b)` | fp32 x fp32 | 3 | ~2^-16 rel. |
| `al::bf16_hilo` | a scale as bf16 hi + lo for a vector-wide multiply | | |

- Each split costs 2 conversions and a subtraction. A chain of them is 20-30 cycles of latency.
- x² of a bf16 x is exact in fp32: one `aie::mul(x, x)`.

## LUTs and their errors

| op | lanes per instruction | slot | error |
|---|---|---|---|
| `vexp2` | 16 | mv | the chord 2^n (1 + f) between integer powers: up to +6% on bf16 inputs, 15% on fp32 |
| `vtanh` | 16 | mv | coarse (max 8.5% rel.); returns its argument below |x| = 0.5; bf16 saturation near 1 |
| gather (`load_4x64_lo / hi`) | 4 per instruction | ldb | exact: 64-bit table entries (`km.act`, `kmath.py`) |

- vexp2 is Schraudolph's exp bit for bit (probed over every bf16 and fp32 accumulators): bits = floor(128 y) +
  16256, for y < 0 16255 - trunc(128 |y|) (one's complement), y the argument truncated to bf16 (steps of 1/4 from
  |y| = 32: up to 19%). Exact at non-negative integers (an exact 2^n: the exp2 recipe below); at negative integers
  2^n (1 - 2^-8). Its error depends on the fraction only: a correction needs the fraction, i.e. the range split
  anyway. Where a softmax's scale-free P suffices, two chords (`km.exp2(bias=0, prec="chord2")`: vexp2(x) +
  vexp2(x + 1/2) / sqrt 2, two vexp2, an add, two products) are within 1.2% of 2.068 2^x for |x| <= 2 (1 / sqrt 2:
  the chords' errors cancel; weights minimax for |x| <= 2 gave SD's UNet more error).
- The gather: lane k of an instruction reads table ab (k % 4 < 2) or cd, its odd lanes 32 bytes further; offset bits
  5:4 select the entry of a 64-byte group, bits 3:0 are ignored. So a table is 4 copies: groups of 4 entries stored
  twice, in ab and cd, the two copies in two 16 KB regions (bank-conflicts.md). Each base of the address vector must
  be a global or a pointer parameter of its own, or the reads carry no memory operand (compiler-feedback.md).

- Both take the argument truncated to bf16. They share the mv slot with compares, selects and
  min / max.
- A 32-lane activation with 2 LUT calls, a compare and a select already needs 5+ mv ops per 32
  lanes.

## Recipes

### exp2 of fp32 x to 0.4 bf16 ULP (`km.exp2(N)`)

1. n = round(x): add 1.5·2^23, read the float's bits, subtract 0x4B400000.
2. f = x - n, with |f| ≤ 0.5.
3. 2^f by a polynomial (degree 2 with its bf16 rounding points fitted: c1 = 0.6996, c2 = 0.236;
   or `km.exp2`'s cubic).
4. Shift n into the exponent field (`upshift(n, 23)`) and add it to the polynomial's bits. The
   range: x clamped at -126 in fp32 first (a masked score of -1e30 becomes 2^-126, or exactly 0
   from -127 on: `below="zero"`), or n clamped **as an integer** (`below="int"`: no fp32 max,
   arguments above -2^22).

No branches, no -inf handling. 64 lanes per step, about
55 cycles per 64 lanes; it does not software-pipeline, so batch the work.

### exp2 without a table or fp32 vectors

1. A = y + 192 in an accumulator (a caller's MAC chain adds it as its init constant); v = bf16(A) = 192 + round(y)
   for |y| < 64; f = bf16(A - v), |f| <= 1/2 (exact for bf16 y; an fp32 y loses f's low bits: +0.17 ULP).
2. P = vexp2(v - 128) = 2^(n + 64) exactly (non-negative arguments only, above); the 2^-64 goes into the constants.
3. y = bf16(P 2^-64 + bf16(P f) c1' + bf16(P f^2) hi'), hi' = bf16(c2' + c3' f), c' = c 2^-64: P f and P f^2 are
   exact scalings, so one rounding at the end. Fit c1 first, then c2, c3 to its rounding (0.69140625, 0.2412109375,
   0.0625; c1 rounded last: 0.72 ULP).

Max 0.585 ULP, 99.80% correctly rounded over the bf16 y in [-63, 63); II 10 per 32 lanes (its 10 MACs: the vec
bound), ~370 cycles per 1024 on the NPU, bit-exact against its numpy emulation. A quadratic (hi' constant: c1 0.703125, c2 0.2412109375)
saves 2 MACs: max 0.85 ULP, mean 0.26 (the cubic's 0.56 / 0.21). `km.exp2`'s fp32 cubic: 0.4 ULP, ~880 cycles, not
pipelined; a gather-LUT exp would cost a SiLU table's ~850 and 3 KB.

### Softmax in the MMUL panel layout (decode attention)

- Keep scores as 8x8 accumulator panels (lane 8q + k).
- Row max by `shuffle_down_rotate` 1, 2, 4 and max.
- A per-query integer reference max from the first KV chunk (+1), P = 2^(s - m) with the clamp
  above, and no O rescaling.
- Not with attention sinks past the first chunk: MiniCPM-V's ViT scores reach 71 log2 units above its first 32 keys'
  maximum (P saturates at 2^63; embeddings' KL 0.0090 -> 0.0074 once raised). Check each later group against m + 16
  in bf16 and rescale o and l on the rare raise (sd15 `ops.Attention(raise_max=)`, the qwen35 prefill attention).
- Take the max of the raw inputs and scale it with the same multiply as the elements; otherwise
  bf16 rounding can put the largest argument above 0.

### SiLU through the gather LUT (`km.act(km.SILU)`: the GDN prefill's)

- silu(x) = relu(x) + g(|x|), g(t) = silu(-t) on [0, 24] in segments of 1/4, a quadratic per segment in the offset r
  from its centre (c0 fp32, c1, c2 bf16: one 64-bit entry; minimax on the segment's bf16 points, in result ULPs).
- Range reduction in bf16: v = bf16(t + 32) rounds t to the grid ([32, 56] is a binade of ULP 1/4), r = (32 - v) + t
  exactly ((32 - v) first, or tiny t lose their bits), the table offset is vfloor(64 v).
- Max 0.59 ULP, 99.84% correctly rounded, 3287 cycles per 4096 elements on the NPU; the vtanh / vexp2 form below:
  6.6 ULP (at x = -2.2), 2682 cycles. Its emulation in numpy is bit-exact (`emulate` in `tests/rtc/elementwise.py`).
- What set the II (~550 variants): register pressure, not slots. In one loop the
  constant vectors (1, 24, the abs mask, 0, the lanes' table pointers) took 5 of the 12 x registers, and the
  post-pipeliner's allocation failed (II 23-30 at ResMII 12-15; constants parked in lf / bm registers and moved back
  every iteration). Fission: t = min(|x|, 24) in a loop of its own (II 2), the table pass reads it in place (II 22,
  bound 9: the post-pipeliner's register allocation fails below). Shuffle the coefficients into 32 lanes (`T256_2x2_lo / hi` of two `T16_16x4_lo`), not `concat` of 256-bit
  halves (moves).

### Sigmoid through the gather LUT (`km.SIGMOID`: the qwen35 attention prep's gate)

No odd symmetry to fold: `ActTable(full=True)` tabulates f itself over [-t_max, t_max] (here 16), the range
reduction as SiLU's with x + 48 in one binade of ULP 1/4 (129 segments), beyond it clamped: within 0.54 bf16 ULP,
99.9% correctly rounded (`test_sigmoid_lut`); 1718 instead of 10273 cycles per 8 tokens of 256 gates (an fp32 exp2
and reciprocal).

### SiLU (GemmQ's gate/up epilogue: the vtanh form + `al::silu_small`)

- vtanh form (`al::tanh_acc` / `al::exp2_acc`): u (1 + tanh(u)), u = x / 2. Below x = -3,
  u 2^(x log2e) (1 + |t|): the exp2 tail avoids cancellation against vtanh's saturation. II 10 per
  32 lanes.
- `silu_small` fixes |x| < 2.5 (vtanh's small-argument error cost 2.6x the first-token KL) with a
  bf16 fit u + s (P0 + P1 s + P2 s²), s = u². It runs as a **second loop** in place (fission): one
  loop with both was not pipelined (48-59 cycles per 32).
- Too coarse for a recurrent state: in a 128-token GDN state it put 2.6x the exact SiLU's error
  (6.7e-3 vs 2.6e-3; the vexp2 tail from -2.5 on: 3.4x). The exact identity a (1 + t) for x >= 0,
  a e^x (1 - t) for x < 0 (1 + t = e^x (1 - t): no cancellation), e^x in a loop of its own (below),
  `al::silu_small` below |x| = 2.5: 1.2x, 3.2K cycles per 8 tokens x 288 values. The table SiLU
  (above) has the exact SiLU's error and is faster (the 27B scan 4.57 vs 4.84 ms per layer, GdnOut
  0.72 vs 0.89).

### e^x of fp32 x in a pipelined loop

``km.exp2``'s fp32 forms (``{{ e }}(al::log2e<32>(x))``) do not pipeline; ``al::log2e`` is ~2^-16 relative, so e^x this
way is off by ~1.6e-5 |x| (2^-9.5 at |x| = 87: ``km.expneg``) (86 bundles per 32 lanes: its fp32 vector ops). x log2 e as
two bf16 products (log2 e hi + lo) into an accumulator, the 1.5·2^23 split, 2^f by a quadratic in bf16 MACs (minimax
on [-0.5, 0.5]: 0.27%, ~0.7 bf16 ULP; the cubic 0.014%), the exponent added as integers: II 15 per 32 (1131 vs 6238
cycles per 2304 values; the cubic 1997). In the GDN state the quadratic's error is invisible (3.14e-3 vs 3.08e-3).

### GELU through the gather LUT (`km.GELU`: the Gemma prefill's gate/up epilogue and per-layer-input op)

The SiLU table's scheme (`kmath.ActTable`: f(x) = relu(x) + g(min(|x|, t_max)) for f(x) - f(-x) = x) with 65
segments of 1/8 over [0, 8]: within 0.59 bf16 ULP from x = -4 on, 1.42 in [-6, -4) (absolute 1.4e-7; the tail decays
faster than a quadratic per segment follows, so `floor` sets an absolute error target there), ~2.2 KB of table, as
a worker `Buffer` argument (`km.table_pass(table_arg=True)`: SKILL.md section 2). `gate_up=True` renders the GEMM epilogue's
form (the gate halves of a gate/up C tile). The decode overlay's GELU (`overlay/kernels.py`) is `km.act(km.GELU)`,
32 values at a time: 2.4x faster than the tanh form through `vtanh` (a bf16 fit below |w| = 1.25, e^(2w) (1 + |tanh w|)
for w < 0), which is within 1.7 ULP at ~7 cycles per element.

## Block float (bfp16ebs8)

- **Layout**: 8 values share one 8-bit exponent, 9 bytes per block. `sizeof(bfp16ebs8)` is 9;
  `sizeof(uint4)` is 1. Never do pointer arithmetic on `bfp16ebs8 *` or `uint4 *`: offset a
  `uint8_t *` and cast at the stream constructor.
- **Loads**: 576-bit blocks load only as FIFO pops (`vlda / vldb.pop.576`). Each input stream holds
  one of the 2 `lf` registers, so at most two input streams are live at once.
- **Fills and seeks**: aie_api's stream fills once per 8 pops, anchored at pop 0. A `seek(p)`
  with `p % 8 != 0` returns garbage later. Create streams inside the iteration, or keep a
  persistent one and `seek` by multiples of 8 (`seek` zeroes the count). A stream carried across
  loop iterations without that fills before every pop.
- **Output streams**: `push` = push + flush. Create them per iteration.
- **Unsigned mantissas**: unsigned-mantissa MMUL mode (`sgn_x = sgn_y = 0`) makes an 8-bit
  unsigned significand exact.
- **Precision**: 7 bits relative to the block's maximum: ~1-2% relative error per operand on real
  Q / K / P / V. For bf16-level products split each operand into bfp16 hi + lo and issue 3 MMULs
  (hh + hl + lh). The exact split is on the MMUL with a -I panel (restructurings.md R4).
- **Idempotence**: conversion is not idempotent. A -128 mantissa keeps its exponent; converting
  the value again raises it. Host models insert the kernel's blocks; they do not re-round values.
- **Exact integers**: integers in [-128, 127] are exact with exponent byte 133 (a scale of 1), which `bfp16.blocks`
  does not pick for a block holding -128 (`sd15.layout.int_panels`). MiniCPM-V's encoder takes the 8-bit pixels so,
  as p - 128, with the normalization folded into the patch embedding's weights and its offset on a constant input in
  a block of its own: no rounding of the input, and cheaper host packing than float panels.
- **Host packing**: round to nearest even when packing on the host (`bfp16.blocks`), not truncate
  (as mlir-aie's `floatToBfp16` helper does): a Llama Q4_0 prefill's KL 0.00085 vs 0.00145 truncated.

## Rounding

- The default rounding mode is floor. Set `aie::set_rounding(aie::rounding_mode::conv_even)` in
  every kernel entry that rounds. The mode is per-core state that survives across kernels.
- A kernel that relies on another kernel having set the mode truncates every tile when that kernel
  is absent (a GEMM: KL 0.0008 -> 0.038).
- Floor rounding also biases bf16 write-backs (~6% at large K).

## Model-level accuracy rules (found on Gemma 4)

- A scale that multiplies a whole vector (1 / rms) goes in as bf16 hi + lo (`al::bf16_hilo`),
  never rounded to bf16.
- Cached K needs bfp16 hi + lo blocks: one 7-bit block shifts a position's scores coherently,
  and the softmax turns that into an output error. V tolerates single blocks.
- GEMM inputs as single bfp16 panels are acceptable: simulated KL 0.00059 vs 0.00042 with hi / lo.
- Judge by KL over several prompts or rounding variants. Single short runs vary up to 40x with
  rounding luck.
- **Flash attention: P = 2^(s - m) as bfp16 hi + lo panels**, like Q and K. Single bfp16 P blocks (8 positions
  share an exponent: the small weights of a block lose their bits) doubled the Gemma prefill's cache error at the
  last own-cache layer (K 5.4 -> 3.7%, V 8.0 -> 5.2% vs fp32) and its KL (0.00068 -> 0.00046). In a DMA-bound
  attention the second MMUL per record costs nothing measurable; give the lo products their own accumulator chain
  when one chain would serialize (QB = 1).
- **hi + lo weights** where a GEMM is sensitive and cheap: `sd15.ops.Gemm(hilo=)` doubles K (the hi k-tiles, then
  the lo ones on the A panels walked again). MiniCPM-V's encoder takes them at its patch embedding and its final
  merger (a quarter of its weights' error between them); any other layer's weights held 1-3% each and did not pay.
- **A GEMM's bf16 C tile between k-tiles** (`gemm/mac.py`: C stored as bf16 after every 64 k) rounds the running sum
  K / 64 times; the error grows with the number of roundings and the partial sums' size. Splitting K into parts
  that each emit a bf16 C, summed in fp32 by the consumer (a residual add), cuts it (Gemma's down, K = 10240, in 4
  parts: `prefill.gemm.plan_phase(k_part=)`). K = 128 per round trip instead of 64 did not help measurably.
- **Q4_0 weights as bfp16 panels** (GemmQd's dequant: q d per element, 8 k sharing an exponent) round the fp16 scale
  d into the 8-bit mantissas: ~0.5% per weight. Exact-integer panels (the decode's GEMV) avoid it but need the
  scale applied per group after the MMUL, which the prefill's MAC loop has no accumulators for.
- In the Gemma prefill these three (single bfp16 inputs, bfp16 weights, the bf16 C tile) cost about equally
  (each alone ~2-3% at layer 23 vs fp32); removing any one barely moved the sum. Fix the outlier first (P above).

## Precision studies: simulate the rounding points

Before changing kernels for accuracy, rank the candidates on the CPU: subclass the fp32 reference model and apply
the kernels' rounding points where they happen (one flag per rounding point: GEMM inputs as bfp16 / hi + lo,
weights as bfp16 panels, bf16 outputs, the C tile per k-tile, K parts, P and V blocks in the attention).

- Measure what the acceptance test measures: the per-layer cache error against the fp32 caches, and KL of
  teacher-forced tokens after the prompt.
- Average over several prompts: one prompt's error at the last layers varies by ~+-30% with the rounding luck,
  and a 64-token KL mean is dominated by its few largest tokens (max 0.014 at mean 0.0006).
- Rank rounding points one at a time in the exact model (only that point rounded). In the model with every rounding,
  any change re-draws all later roundings, so one run of it measures that noise: in MiniCPM-V's encoder every early
  site moved the KL by the same ~0.01. The one-site errors add up (as squares) where the full model's do not.
- A few fragile tokens can hold most of an output's squared error (MiniCPM-V's ViT: the worst 1% held 67-80%,
  single tokens changed by 57-99% by any small change): rank by the median per-token error.
- llama.cpp's CPU takes Q4_K weights' activations as Q8, so any change of the input re-draws that rounding: a KL
  floor (~0.005 on MiniCPM-V 1B over 96 tokens). Its F32 requant (`llama-quantize --allow-requantize IN OUT F32`)
  has none.
- Average the KL over realizations: a tiny dither (1e-6 relative, or single bf16 ulps in 0.1% of an input) re-draws
  every later rounding; 10-20 per config resolve ~5%. Report a prompt whose KL hinges on one near-tie token apart
  (MiniCPM-V: a month read from small print): the model's other biases set its sign, so it moved one way in the CPU
  model and the other on the NPU while the remaining prompts agreed.
- Check the model against the NPU per layer. Where they diverge, a rounding point is missing from the model (the
  attention's P was: the simulation matched the NPU through layer 17 and fell short after it until P was modeled).
- `bfp16.values` emulates the conversion; insert blocks where the kernel converts, never re-round (above).
