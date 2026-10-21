---
name: npu-prefill
description: Build, extend or speed up a model's prompt prefill on the NPU on top of its decode overlay (the qwen35 / gemma4 / llama pattern) - GEMMs on the decode weight image through GemmQ / GemmQd (prefill.gemm), layer-major chunk programs from a library ELF, the KV caches and states the decode continues from, the app's prompt path, accuracy gates and where a chunk's time goes. Use when porting a prefill to a new model or architecture (MoE, drafter), changing a prefill's operators or chunk program, or chasing its speed or accuracy.
---

# Prefill on the decode image

Reference implementations: `operators/qwen35/prefill.py` (`QwenPrefill`), `operators/gemma4/prefill.py`
(`GemmaPrefill`) and `operators/llama/prefill.py` (`LlamaPrefill`: pre-norm residual adds, one column per KV head, the
shared attention on the decode's KV record format `operators/prefill/attn.py`); their module docstrings describe each
layer's operators. Read the one closer to the new model first, then `operators/prefill/gemm.py` (the GEMM planner they
use), `operators/prefill/norm.py` (the residual norms: bf16 `PrefillNormOperator`, fp32 `ResidualNormOperator`)
and `operators/prefill/chunks.py` (`ChunkPrefill`, their base: chunk plan, pieces, buffer slices, library program; a
model supplies its runlist, static buffers and time model). Kernel and data-movement work inside it: the
**aie-kernel-perf-optimization** skill.

## The shape of a prefill

- **One device, no copies.** The prefill's buffers (the fp32 residual, bfp16 panels, GEMM outputs, per-layer
  inputs) and the overlay's (weight image, KV caches, recurrent states) live on the overlay's device; the chunk's
  program addresses slices of them. Weights are read in place: never a second, expanded copy.
- **GEMMs**: `prefill.gemm.plan_phase(image, region, T, ...)` turns a region of the decode image into GemmQ / GemmQd
  operators with their weight offsets; `ops=` (a dict shared across layers) makes identical GEMMs one object, so
  one design in the ELF. Options: `act` (the gate/up epilogue: silu / gelu), `k_part` (a long-K projection in K parts:
  A strip resident in the memtile, fewer C-tile roundings, the parts summed by the consumer's residual add), `pitch`
  (output column pitch: `phase_pitch` / `d_pitch`). Keep a model's options in one helper (`gemma4.prefill.plan_gemm`)
  that the prefill, its tests and the experiment scripts all call.
- **Chunk program: layer-major over sub-chunks.** A chunk of T tokens runs as S = T / SUB sub-chunks; each operator of
  a layer runs over the sub-chunks back to back, so consecutive items of one operator object share one ConfigureOp;
  the flash attention runs once over the whole chunk. SUB = the GEMMs' whole MAC tiles (GemmQd: mac_rows x m, e.g.
  3 x 96 = 288). Chunk sizes are multiples of SUB: a prompt just above a multiple pays a whole sub-chunk.
- **Short prompts: sizes of their own.** A chat prompt in a SUB-token chunk pays every operator's ConfigureOp and
  passes for tokens it does not have; sizes below SUB with operators planned at that T (`ChunkPrefill.SMALL`: one
  sub-chunk; Llama's and Qwen's 64 / 128) took Llama 3B's 26-token prompt from 183 to 133 ms, the Qwen 9B's / 27B's
  32-token one from ~222 / ~650 to 159 / 452 ms (the weights still stream once: ~0.6 of a sub-chunk's time at 64).
  Plan their input projections at SUB's pitch where buffers outlive a chunk in that layout (Qwen's conv states).
  Choose their operators by speed, not configuration size (configurations by control packets: GemmQ, the attention on
  4 rows, the norms on 4 columns; 3B's chat prompt 96 -> 70 ms). Long prompts the other way: chunks past 2048 in whole
  1024-token sub-chunks (Llama 3072 / 4096, where the flash attention's 64 passes allow) halve the ConfigureOps per
  token (1B 8K TTFT 1.29 -> 1.21 s).
- **Library ELF** (`fused.Library`): each design compiled once, whatever the depth and chunk sizes;
  `run(items, splits=, bufs=, hooks=)` assembles a chunk's program on the host and dispatches it in pieces of
  whole layers, each estimated under `PIECE_BUDGET_S` (the driver kills a dispatch after 2 s; DDR contention adds up
  to 40%). Keep a per-layer time estimate (`layer_seconds`) for the splits and the chunk planner, fitted to measured
  chunks (sizes x positions; Llama's within ~5%): a model 2x off put a 48-token prompt in a 256-token chunk (58 ms)
  instead of a 128-token one (44). Terms that scale with the model's dimensions fit one set of constants to all its
  sizes (Qwen: weight bytes, GEMM items, hidden size, v heads, q heads; the flash attention per cached position as
  the larger of its cache bytes and its cores' cycles: the 1B, 9B and 27B within ~2%); constants fitted to the
  largest model alone were 2-3.7x high for the 1B. A program's ConfigureOps alternate two PDI-load slots: `run` assembles the next
  run's pieces (the other parity) while one runs, else a size's second run pays it (1B +6 ms, 8B +105 ms).
- **Runtime values** (position, cache offsets, window starts) are scratchpad parameters set per chunk (`params()`):
  only for operators in the library (an unknown name raises), range-checked against the arguments' declared sizes
  (declare the cache argument with padding past its end). One parameter name has one value per program: two variants
  of an operator that need different values cannot share a program.
- **Caches in the decode's layout**: a prep operator writes the chunk's K / V records where the decode would, and the
  flash attention reads the chunk's history from the same cache, so the decode continues with no host work. Linear
  caches (also for sliding windows, after a guard of one window) keep the window reads one BD per pass. Recurrent
  states (GDN) the same: read and written in place, the decode's format.
- **Layers that share caches** (Gemma E4B's 24 .. 41) do nothing for the prompt's tokens but the last: the prefill
  stops at the last own-cache layer, and that layer only writes its cache.
- **The prompt's last token** needs logits: Qwen runs the LM head on the last row of x (`QwenOverlay.logits_of`);
  Gemma finishes the token with an overlay step from the prefill's last layer (`GemmaOverlay.step(start=)`, x the
  chunk's residual at that layer's input: `GemmaPrefill.residual`), so only the layers the prefill skipped stream.
- **App** (`models.common.OverlayModel.prompt` / `forward`): decode steps up to the next position where a chunk can
  start (whole cache groups), the planner's chunks, decode steps after; each chunk's inputs (embeddings, per-layer
  inputs) on a worker thread while the previous chunk runs; Gemma's first chunk starts with its embeddings and
  receives inputs that later operators read through a `Library.run` hook (`GemmaPrefill.run(pli=Future)`).

## Gates (test each before building on it)

1. **GEMMs alone** on the real image at the sub-chunk size: output against the host model (`test_prefill_gemm`),
   time and TMAC/s per phase, the plans (passes, tiles, m, A resident). New tile shapes: the DMA layout first, on
   the host (`tests/rtc/test_gemm_q_dma.py`: add cases with the model's tile geometry).
2. **Each other operator** against an fp64 model at the kernel's rounding points: norms within ~2^-16 for the fp32
   residual and within 1.1x the error of the exact values rounded to the output format (bfp16 panels); the
   attention within 1.1x the bfp16 rounding of the exact attention on the cores' own operands (their Q panels, the
   cache they read). Test edge positions: p0 = 0, a window start, the guard, chunks after a history.
3. **Layers in a program**: the caches a few layers leave against the decode path's (slow test).
4. **The model**: a prompt through the prefill vs the same prompt as decode steps: the caches at the last own-cache
   layer within a few % rms (K and V separately), KL(same) of the text after it against the fp32 reference (Gemma:
   <= ~0.0005), argmax agreement; `forward`'s last-token logits vs a whole decode step's. Once the numbers are known,
   tighten the asserts to just above them.
5. **The app** against the previous commit, run from a worktree in alternated processes (aie-kernel-perf-optimization
   `reference/measuring.md` "Comparing against another commit"); a ~1K and an ~8K prompt (the essay corpus).

## Accuracy

- Gemma's rules: the residual in fp32; scales of a whole vector (1 / rms) as bf16 hi + lo; Q, K and P as bfp16
  hi + lo panels in the attention (a single bfp16 P doubled the cache error); V and the GEMM inputs as single bfp16.
  Llama meets its gates with less: a bf16 residual (`PrefillNormOperator`, the deltas summed in fp32 and rounded once:
  half the norms' bytes) and single bfp16 Q, K and P panels.
- Judge a precision change on KL means over several sequences (Llama: three chat prompts, three essay windows): one
  or two prompts move by ~30% with small rounding changes.
- An MoE model is chaotic: any rounding change moves a near-tied expert (Qwen3.6-35B-A3B: llama.cpp's own prompt and
  one-token graphs pick another top 8 for 40-70% of the tokens by the last layers), and a 1e-7 change of the router's
  logits already reaches llama.cpp's prompt-vs-decode KL. So KL cannot rank its formats: rank them by their error in
  one block on llama.cpp's inputs (against llama.cpp's own Q8_K error there) and by the paired next-token NLL on an
  8K essay window over two trajectories (prompt graphs of two ubatch sizes: ~1e-2 nats of noise per run). Its router:
  BFP16 hi + lo rows against x hi + lo, the logits fp32 up to the top k (single BFP16 rows change 6% of the top-8 sets
  per layer and cost ~2.5e-2 nats per token; bf16 logits change 11%).
- Long-K output projections in K parts (`plan_phase(k_part=)`): each part's bf16 C tile sums fewer
  k-tiles, the residual add sums the parts in fp32. Llama 3B's down (K 8192, 128 k-tiles) in two parts: KL(same) on a
  chat prompt 0.00071 -> 0.00031, an essay's 0.00122 -> 0.00101.
- Rank fixes by simulating the rounding points on the CPU before writing kernels, averaged over several prompts
  (aie-kernel-perf-optimization `reference/numerics.md` "Precision studies", and its model-level rules: a GemmQd
  prefill's remaining error sources cost about equally).

## Speed: where a chunk's time goes

- Per operator kind: drop its items from the program and time the rest.
- Per design switch: a program of one item of the design alternating with another kind's, minus both repeated alone
  (no switch between them); or the program's configurations alone (its RunOps dropped). ~80-90 µs each (the reset
  ~40 µs; program memory, BDs, locks by control packets: `fused.ctrlpkt`): a layer of a dozen operators ~1 ms whatever
  the chunk size (Llama 1B's chunk of 128 tokens: 10 of 31 ms).
- Per GEMM: alone with parts compiled out (`ablate=("dq", "mm", "act")`) and its cores' windows (`ablate=("ev_kt",)`
  with the profiling counters); per attention kind and position: the attention operator alone at (T, p0).
- Gemma E4B, 1152 tokens at p0 = 0: GEMMs ~60% (gate/up the largest, MAC-bound), norms ~17% (the fp32 residual read
  and written per update: near their DMA floor), attention ~10% and growing with the position (full-attention layers:
  the history re-read per query pass, near the DDR ceiling with each core taking it on both input channels),
  ConfigureOps ~7%.
- Llama 3B Q4_K_M, 1024 tokens at p0 = 0 (341 ms): GEMMs ~70%, design switches ~13%, norms ~7%, attention ~6%. A
  layer whose V and down are Q6_K costs ~2.2 ms more than a Q4_K one (planar GemmQd: a dequantizing core per column,
  66 of 80 MAC columns, 1024 tokens as 1152): ~9% of the chunk.
- Levers that worked: the A strip resident through K parts (`prefill.gemm.resident_k_part`); reading each weight byte once (header units); more
  queries per attention pass where L1 allows (sliding window: 2 blocks of 32); a broadcast KV stream on both of each
  core's input channels (full attention at 30K: -41%); the last token from the prefill's last
  layer; the first chunk's inputs overlapped; sub-chunks of 1024 tokens for chunks made of them (Llama: GEMM items
  of 1024, fewer fills and configurations); GemmQ (4 MAC rows) on every 16-row tile format, whatever its K chunk (Q4_0
  / Q4_1 tiles of 1024 columns: the header expanded per k-tile slot; their planar GemmQd padded 1024 tokens to 1152 on
  3 rows: a 4096-token chunk 507 -> 465 ms). Did not: a C pingpong with one C object; deeper FIFOs on DMA-bound
  operators.

## Pitfalls

Besides AGENTS.md's (kernel names, statics, host writes and flushes, the 2 s and 4 GB limits) and aie2p-coding
section 8's (join offsets, memtile channels, chained fills within 14 shim BDs, zeroed strides of size-1 dimensions):
- A GemmQd stays within 64 passes per block.
- A bigger Worker `stack_size` costs L1 the operator's budget must still fit.
- Test fixtures: the overlay's capacity must hold a chunk plus its padding (`max_positions`).
- A GEMM's weight argument spans its region's 8 columns: in a column-major image that is nearly the whole image
  (Llama 8B: 4.3 GB, past the 4 GB argument limit). `TileImage` lays a column-major image out in blocks of whole
  layers under 3 GiB (`BLOCK_BYTES`); the decode's weight runs take a task per block.
