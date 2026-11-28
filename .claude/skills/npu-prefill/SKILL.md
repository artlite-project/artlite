---
name: npu-prefill
description: Build, extend or speed up a model's prompt prefill on the NPU on top of its decode overlay (the qwen35 / gemma4 / llama pattern) - GEMMs on the decode weight image through GemmQ / GemmQd (prefill.gemm), layer-major chunk programs from a library ELF, the KV caches and states the decode continues from, the app's prompt path, accuracy gates and where a chunk's time goes. Use when porting a prefill to a new model or architecture (MoE, drafter), changing a prefill's operators or chunk program, or chasing its speed or accuracy.
---

# Prefill on the decode image

Reference implementations: `models/qwen35/prefill.py` (`QwenPrefill`), `models/gemma4/prefill.py` (`GemmaPrefill`) and
`models/llama/prefill.py` (`LlamaPrefill`: pre-norm residual adds, one column per KV head, and the shared attention on
the decode's KV record format in `prefill/attn/`); their module docstrings describe each layer's operators. Read the
one closest to the new model first, then `prefill/gemm.py` (the GEMM planner they use), `prefill/norm/` (the residual
norms: bf16 `PrefillNormOperator`, fp32 `ResidualNormOperator`) and `prefill/chunks.py` (`ChunkPrefill`, their base
class, which handles the chunk plan, pieces, buffer slices and library program; a model supplies its runlist, static
buffers and time model). For kernel and data-movement work inside a prefill, use the **aie-kernel-perf-optimization**
skill.

## The shape of a prefill

- **One device, no copies.** The prefill's buffers (the fp32 residual, bfp16 panels, GEMM outputs, per-layer inputs)
  and the overlay's buffers (weight image, KV caches, recurrent states) live on the overlay's device, and the chunk's
  program addresses slices of them. Weights are read in place, never from a second, expanded copy.
- **GEMMs**: `prefill.gemm.plan_phase(image, region, T, ...)` turns a region of the decode image into GemmQ / GemmQd
  operators with their weight offsets; `ops=` (a dict shared across layers) turns identical GEMMs into one object, and
  so one design in the ELF. Options: `act` (the gate/up epilogue: silu / gelu), `k_part` (runs a long-K projection in
  K parts, with the A strip resident in the memtile, fewer C-tile roundings, and the parts summed by the consumer's
  residual add), and `pitch` (the output column pitch: `phase_pitch` / `d_pitch`). Keep a model's options in one helper
  (`gemma4.prefill.plan_gemm`) that the prefill, its tests and the experiment scripts all call.
- **Chunk program: layer-major over sub-chunks.** A chunk of T tokens runs as S = T / SUB sub-chunks. Each operator of
  a layer runs over the sub-chunks back to back, so consecutive items of one operator object share one ConfigureOp;
  the flash attention runs once over the whole chunk. SUB is the GEMMs' whole MAC tile (for GemmQd, mac_rows x m, e.g.
  3 x 96 = 288). Chunk sizes are multiples of SUB, so a prompt just above a multiple pays for a whole extra sub-chunk.
- **Short prompts get dedicated sizes.** A chat prompt in a SUB-token chunk pays every operator's ConfigureOp and
  passes for tokens it does not have. Sizes below SUB with operators planned at that T (`ChunkPrefill.SMALL`: one
  sub-chunk; 64 / 128 for Llama and Qwen) took Llama 3B's 26-token prompt from 183 to 133 ms, and the Qwen 9B's / 27B's
  32-token prompt from ~222 / ~650 to 159 / 452 ms (the weights still stream once, which costs ~0.6 of a sub-chunk's
  time at 64). Plan their input projections at SUB's pitch where buffers outlive a chunk in that layout (Qwen's conv
  states). Choose their operators by speed, not by configuration size (configurations go by control packets for GemmQ,
  the attention on 4 rows and the norms on 4 columns, also in qwen35; 3B's chat prompt drops from 96 to 70 ms). Long
  prompts go the other way: chunks past 2048 tokens built from whole 1024-token sub-chunks (Llama 3072 / 4096, where
  the flash attention's 64 passes allow it) halve the ConfigureOps per token (1B 8K TTFT 1.29 -> 1.21 s). Beyond the
  flash attention's 64 passes, a chunk's attention runs as items of 2048 tokens, each at its own positions
  (`AttnOperator(part=)`, with separate scratchpad parameters per item; the preps write the whole chunk's records
  first). This lets an MoE model's experts stream their weights once per chunk, which is weight-bound below ~4096
  tokens (Flint's 8192-token chunks: TTFT 8K 5.4 -> 3.8 s, 32K 31.8 -> 23.6 s, with the attention's planes below).
- **Library ELF** (`fused.Library`): each design is compiled once, whatever the depth and chunk sizes.
  `run(items, splits=, bufs=, hooks=)` assembles a chunk's program on the host and dispatches it in pieces of whole
  layers, each estimated to stay under `PIECE_BUDGET_S` (the driver kills a dispatch after 2 s, and DDR contention adds
  up to 40%), or under `PEER_PIECE_S` while another process uses the NPU (its requests wait for a piece to finish).
  Keep a per-layer time estimate (`layer_seconds`) for the splits and the chunk planner, fitted to measured chunks
  (sizes x positions; Llama's is within ~5%): a model that is off by 2x puts a 48-token prompt in a 256-token chunk
  (58 ms) instead of a 128-token one (44 ms). Terms that scale with the model's dimensions should fit one set of
  constants to all its sizes (for Qwen: weight bytes, GEMM items, hidden size, v heads and q heads, plus the flash
  attention per cached position as the larger of its cache bytes and its cores' cycles; this fits the 1B, 9B and 27B
  within ~2%); constants fitted to the largest model alone were 2-3.7x too high for the 1B. A program's ConfigureOps
  alternate between two PDI-load slots: `run` assembles the next run's pieces (the other parity) while one runs;
  otherwise a size's second run pays for it (1B +6 ms, 8B +105 ms).
- **Runtime values** (position, cache offsets, window starts) are scratchpad parameters set per chunk (`params()`).
  They exist only for operators in the library (an unknown name raises) and are range-checked against the arguments'
  declared sizes (declare the cache argument with padding past its end). One parameter name has one value per program,
  so two variants of an operator that need different values cannot share a program. A parameter's range is the
  tightest among all operators in the library that read it, in any program, so a variant with a smaller argument gets
  a separate name (`QA_KV` per size) or, where its value is fixed, no parameter at all.
- **Caches in the decode's layout**: a prep operator writes the chunk's K / V records where the decode would, and the
  flash attention reads the chunk's history from the same cache, so the decode continues with no host work. Linear
  caches (also used for sliding windows, after a guard of one window) keep the window reads at one BD per pass.
  Recurrent states (GDN) work the same way: they are read and written in place, in the decode's format.
- **Layers that share caches** (Gemma E4B's layers 24 .. 41) do no work for the prompt's tokens except the last one: the
  prefill stops at the last layer with its own cache, and that layer only writes its cache.
- **The prompt's last token** needs logits: Qwen runs the LM head on the last row of x (`QwenOverlay.logits_of`);
  Gemma finishes the token with an overlay step starting from the prefill's last layer (`GemmaOverlay.step(start=)`,
  with x set to the chunk's residual at that layer's input: `GemmaPrefill.residual`), so only the layers that the
  prefill skipped stream their weights.
- **Readouts and non-causal prompts**: `NpuModel.outputs` gathers a prompt's rows per chunk (with no decode-step tail
  and no LM head; the heads run on the host: `models.readouts`, `models.server.decisions`). A non-causal model (vllm-d3:
  `QwenConfig.causal`) runs a whole prompt in one chunk from position 0: all of the flash attention's passes stream the
  prompt's groups, the padding keys of its last group are masked (`PrefillAttnOperator(causal=False)`), and there are
  no decode steps. Its reset does not zero the GDN states (d3-lite: 18.9 MB, ~1 ms of a short readout). Instead, a chunk
  of up to SUB tokens runs the packed GDN operators as one segment from token 0 (the scan zeroes the state itself,
  exactly; its final state belongs to no sequence, which is why causal models cannot do this), and a larger chunk zeroes
  GSTATE first (the packed scan costs more there). LFM3's d1 (`LfmConfig.causal`): the full layers' passes stream the
  prompt's groups (`AttnOperator(causal=False)`, using the causal design's history part from group 0), and each row
  sees its own range of keys (lo, hi per row in a SEG object per pass; S, the keys every row sees, is a parameter).
  This puts a state and its questions in one sequence (`outputs(sets=)`); the SWA and conv layers stay causal.
- **Packed chunks** (a readout model's prefill, `pack`; `NpuModel.readouts`): several inputs (segments) are placed back
  to back in one chunk, so they share its ConfigureOps and weight passes. Only operators that mix tokens need packed
  variants; every other operator works row by row:
  - The attention: each row sees its segment's keys (plus a history that every row sees: a shared prefix). The Llama
    path's attention (`prefill.attn.AttnOperator(window=)`) is DMA-bound (with compute compiled out it still takes
    88-92% of its time), so skipping the compute of masked chunks saves nothing. Instead, each pass streams a fixed
    window of cache groups that ends at the pass's own group (with no runtime parameters, a sequence reads at most 32),
    which caps a segment at `PACK_SPAN`. After a history of up to `RING_HOLD` groups, chunks from 1024 tokens up use
    its ring variant (`ring=`, `("ring", T)` programs): each column's memory tile holds the history and the last W + 1
    groups, each read from DDR once, and the cores become the bound. A prompt of one cache group that the inputs
    continue, and that is not cached yet, rides in the first packed chunk as its first segment, with the keys before
    `P_HK` seen by every row (`run_packed(lead=)`, `NpuModel.readouts(lead=)`: used for a rerank query). qwen35's
    attention (`PrefillAttnOperator(pack=)`) streams as unpacked and builds its masks with vector compares against
    per-lane lo / hi (a scalar loop that builds them doubles a packed layer's time). Both take the rows' first / last
    keys on a stream (SEG, one object per pass, holding values in the order the memtile walk leaves them) and mask the
    rest.
  - qwen35's GDN prep / scan (`pack=`): the mask PM per token is 1 + the token's distance from its segment's first
    token, and d = -2 marks that first token (natural decays round to -1.0). The scan's reset is exact: the cut E terms
    are held at -1, and the values they multiply are zeroed by select (`al::mulff`'s bf16 hi + lo products leave 2^-17
    of x in x + (-1) x, so a long segment's state leaked 9% into the next one). The convs of a segment's first 3 tokens
    use only that segment's taps.
  - Tree chunks (a recurrent model's inputs after a cached prefix: qwen35's `tree_sizes`, `run_packed` at p0 > 0):
    each segment continues the prefix's states, starting from a 16-token scan chunk. The scan cores hold the prefix's
    state quarter next to the running one (their L1 comes from one-deep core FIFOs with the chunk's state update after
    their release, the v conv scratch in the union, and one SiLU table copy: `km.act(copies=1)`). A chunk whose first
    token starts a segment (a gate slot of the record holds the mask) starts from the prefix's state, with the conv
    windows taken from the prefix's rows; GdnOut writes the prefix's conv state back, so the chunk leaves the states as
    they were (no checkpoint, no restore). On Kev-4B, every row is bit-identical to the question run alone after the
    checkpoint; 8 x 64 tokens after 256 take 387 vs. 1006 ms.
  - Positions per token: the cos / sin rows are written past the table's text rows (rows that otherwise only padding
    tokens read).
  - Test a packed chunk against each segment run alone in a chunk of the same size, because GEMM K parts and norm splits
    differ between sizes. K records come out bit-identical. V's blocks span a record's 8 positions, so a segment that
    starts inside one rounds every V differently. Segments aligned to 32-token key chunks are bit-identical up to their
    last V record (with the window attention; the ring takes the key chunks in its slots' order).
  - `ChunkPrefill.pack` plans a request (first-fit-decreasing into each size, with each chunk's inputs moved into
    smaller chunks, or run alone, where that is not faster).
  - Qwen3-Reranker-0.6B's 4096-token packed chunk after a 64-token prompt takes 332 ms with the ring attention vs.
    346 ms with the window.
- **App** (`models.overlay_model.OverlayModel.prompt` / `forward`): decode steps run up to the next position where a
  chunk can start (whole cache groups), then the planner's chunks, then decode steps after them. Each chunk's inputs
  (embeddings, per-layer token rows) are prepared on a worker thread while the previous chunk runs. Gemma's first chunk
  starts with its embeddings and receives inputs that later operators read through a `Library.run` hook
  (`GemmaPrefill.run(e=Future)`).

## Gates (test each before building on it)

1. **GEMMs alone** on the real image at the sub-chunk size: check the output against the host model
   (`tests/prefill/test_gemm.py`), the time and TMAC/s per phase, and the plans (passes, tiles, m, A resident). For new
   tile shapes, check the DMA layout first, on the host (`tests/gemm/test_dma.py`: add cases with the model's tile
   geometry).
2. **Each other operator** against an fp64 model at the kernel's rounding points: norms within ~2^-16 for the fp32
   residual, and within 1.1x the error of the exact values rounded to the output format (bfp16 panels); the attention
   within 1.1x the bfp16 rounding of the exact attention computed on the cores' actual operands (their Q panels, the
   cache they read). Test edge positions: p0 = 0, a window start, the guard, and chunks after a history.
3. **Layers in a program**: compare the caches that a few layers leave against the decode path's (slow test).
4. **The model**: run a prompt through the prefill vs. the same prompt as decode steps. Check the caches at the last
   layer with its own cache (within a few % rms, K and V separately), KL(same) of the text after it against the fp32
   reference (Gemma: <= ~0.0005), argmax agreement, and `forward`'s last-token logits vs. a whole decode step's. Once
   the numbers are known, tighten the asserts to just above them.
5. **The drafter**, for a model that has one: compare its teacher-forced drafts after a fixed prompt set through the
   prefill vs. as decode steps (`tests/common.py` `draft_gate` over `SpeculativeMixin.forced_drafts`: for each position
   of the model's greedy replies, the tokens a round would give, whether the first guess equals the model's argmax, and
   its probability). The drafter reads what the prefill wrote (Qwen: the drafter's own records of the prompt's rows,
   built from the prefill's hidden states and the host's carry after decode steps; Gemma: the caches of the last layers
   with their own caches), which the model's KL does not see. On the 35B-A3B, records whose h had 20% rms noise cost 2%
   of the tokens per round, and records without h cost 25%, at a model KL of 0. Tokens per round react first (through
   the later guesses of a round); give every prefill path that the drafter reads a prompt in the set.
6. **The app** against the previous commit, run from a worktree in alternated processes (aie-kernel-perf-optimization
   `reference/measuring.md` "Comparing against another commit"), with a ~1K and an ~8K prompt (the essay corpus).

## Accuracy

- Gemma's rules: keep the residual in fp32, scales of a whole vector (1 / rms) as bf16 hi + lo, Q, K and P as bfp16
  hi + lo panels in the attention (a single bfp16 P doubled the cache error), and V and the GEMM inputs as single
  bfp16. Llama meets its gates with less: a bf16 residual (`PrefillNormOperator`, which sums the deltas in fp32 and
  rounds once, halving the norms' bytes) and single bfp16 Q, K and P panels. Per-dim q / k norm weights (Qwen3's reach
  ~100, putting a block's dims ~1e5 apart) need hi + lo Q and K (`prefill.attn.Shape.qp`, `kv_k` hilo: Qwen3 0.6B's
  chat KL 0.0053 -> 0.0037). The readout models (embedding, rerank and decision files: `models.readouts.readout_model`)
  also use hi + lo P (the prefills' `precise`): their prompts are short, so it costs only ~1% of the attention; a chat
  model's long prompts keep single P (its cores are the bound there: +20% at 8K).
- The GEMMs' accuracy floor is their 8-bit operands: weights dequantized into bfp16 blocks (Q8_0's bf16(d) q needs ~15
  bits) and bfp16 activations give ~0.15% of a row after 28 layers (a CPU model with those roundings matches the NPU's
  error); llama.cpp's CPU and Vulkan builds differ by ~0.05%. Embeddings and decision probabilities show it; tokens
  rarely do.
- Evaluate a precision change on KL means over several sequences (Llama: three chat prompts and three essay windows),
  since one or two prompts move by ~30% with small rounding changes. With a drafter, also check its gate (Gate 5): a
  prompt's tokens per round move by up to ~2% between equivalent paths, and the set's mean by up to ~0.6%.
- An MoE model is chaotic: any rounding change flips a near-tied expert (on Qwen3.6-35B-A3B, llama.cpp's own prompt
  and one-token graphs pick a different top 8 for 40-70% of the tokens by the last layers), and a 1e-7 change in the
  router's logits already reaches llama.cpp's prompt-vs-decode KL. So KL cannot rank its formats. Rank them instead by
  their error in one block on llama.cpp's inputs (against llama.cpp's own Q8_K error there) and by the paired
  next-token NLL on an 8K essay window over two trajectories (prompt graphs of two ubatch sizes: ~1e-2 nats of noise
  per run). Its router uses BFP16 hi + lo rows against x hi + lo, with fp32 logits up to the top k (single BFP16 rows
  change 6% of the top-8 sets per layer and cost ~2.5e-2 nats per token; bf16 logits change 11%).
- The attention's planes for an MoE model: decide K (the decode's caches too) on teacher-forced decode steps over
  several short texts (paired NLL, KL against llama.cpp), not on long prompts, whose routing chaos (~1e-2 nats over
  512 tokens) hides the effect. Flint's single K was within the noise after 32K-token prompts but worse on all six
  512-token essays (+0.013 nats per token); its single Q and P showed no difference (q / k norm weights up to 7.5) and
  made the full attention 1.2x slower at 30K.
- Long-K output projections in K parts (`plan_phase(k_part=)`): each part's bf16 C tile sums fewer k-tiles, and the
  residual add sums the parts in fp32. Splitting Llama 3B's down projection (K 8192, 128 k-tiles) into two parts cuts
  KL(same) on a chat prompt from 0.00071 to 0.00031, and on an essay from 0.00122 to 0.00101.
- Rank fixes by simulating the rounding points on the CPU before writing kernels, averaged over several prompts
  (aie-kernel-perf-optimization `reference/numerics.md` "Precision studies", and its model-level rules: a GemmQd
  prefill's remaining error sources contribute about equally).

## Speed: where a chunk's time goes

- Per piece of a real chunk: `Library` keeps each piece's shortest duration (`_cache._ends`, keyed by the items list
  and the piece's bounds), so run a prompt, then map the pieces to their items' operators. Per operator: run its items
  of one layer alone through `lib.run`, on the buffers a chunk left behind (~0.2 ms of dispatch each).
- Per operator kind: drop its items from the program and time the rest.
- Per design switch: time a program in which one item of the design alternates with another kind's item, minus both
  repeated alone (with no switch between them); or time the program's configurations alone (with its RunOps dropped).
  Each switch costs ~30-40 µs: ~0.1 µs per transaction operation (stream switch ports, the loader's route, the shim,
  the previous design's cores held in reset), plus program and data memory, BDs and locks written by control packets
  (`fused.ctrlpkt`, each tile quiesced first); the PDI load's array reset (37 µs) happens only at a piece's first
  configuration (`fused.program.Quiet`). A layer of a dozen operators spends ~0.4 ms on switches whatever the chunk
  size (d3-lite's 64-token chunk: 7.1 of 23.5 ms; Qwen3 0.6B's: 6.8 of 20.9 ms), which is a third of a 64-token chunk.
- Per GEMM: run it alone with parts compiled out (`ablate=("dq", "mm", "act")`) and read its cores' windows
  (`ablate=("ev_kt",)` with the profiling counters). Per attention kind and position: run the attention operator alone
  at (T, p0).
- Gemma E4B, 1152 tokens at p0 = 0: GEMMs take ~60% (gate/up is the largest, MAC-bound), norms ~17% (the fp32 residual
  is read and written per update, near their DMA floor), attention ~10%, growing with the position (in full-attention
  layers the history is re-read per query pass, near the DDR ceiling, with each core taking it on both input channels),
  and ConfigureOps ~7%.
- Llama 3B Q4_K_M, 1024 tokens at p0 = 0 (341 ms): GEMMs take ~70%, design switches ~13%, norms ~7% and attention ~6%.
  A layer whose V and down projections are Q6_K costs ~2.2 ms more than a Q4_K layer (planar GemmQd: one dequantizing
  core per column leaves 66 of 80 MAC columns, and 1024 tokens run as 1152), which is ~9% of the chunk.
- Other chips: Krackan Point and Strix Point have the same array at the same clock, also with all 32 cores busy, and
  Krackan has a faster DRAM path in every streaming and gather pattern (`artlite-probe`). However, on Krackan a write
  of part of a 64-byte line (an LPDDR5 masked write, i.e. a read-modify-write in the DRAM) costs 4.7x as much as a
  whole line (132 B rows: 15.6 GB/s, 128 B rows: 73.2 GB/s; on Halo 41.4 / 57.2 GB/s). The prefill writes whole lines:
  planar packed passes write in whole-line windows (`lines`, with the previous pass's last columns carried in the C
  object), gate/up writes its 32-output parts two per C object (576 B rows), and output projections write at `d_pitch`
  in whole lines; a layout leaves room for the last window (`PlanarPlan.line_pitches`). Count partial lines from the
  dispatched shim BDs' row starts and ends: per 1024-token chunk, the 1B and 8B write none, Gemma writes none, and the
  27B writes 8.2 M (its gdn_in's 96-column windows do not fit in L1). Windows maps the NPU's memory in 4 KB pages, so
  every DMA row on a separate page costs a translation (~50 M rows/s in total). The prefill's short rows (128-576 B at
  strides of several KB) pay this cost; longer contiguous rows would avoid it.
- Platform throttles: a Krackan Point laptop asserts PROCHOT (the whole SoC drops to its lowest clocks) within about a
  millisecond of each GEMM that runs its MMULs at 35-51% of their peak while it streams weights (the Q4_K GEMMs become
  2.3x slower), on any power source. MMULs alone at that duty, DMA alone at 97 GB/s, the Q6_K GEMMs (29-33%), the
  attention and the norms do not trip it. So the prefill's dense GEMMs are paced designs (`gemm.pacing`: a new one gets
  its class from `prefill.gemm.gemm_op`; an MoE layer's experts and `gemm_qd_op` GEMMs are excluded), with their levels
  calibrated per machine (`prefill.pace`) on random activations. The array's current follows its operands' bits (Strix
  Halo: 26-29 W socket power on zeros vs. 35-36 W on activations), and a GEMM on the zeroed buffers of a prefill that
  has not run a chunk yet does not trip it. The calibration probes every class because each class throttles
  independently, and a model's largest class can be one that never trips and that pacing hardly changes (the Q6_K
  classes of Llama 3.1 8B and Qwen3.8 27B). A design-wide runtime switch is free only as one branch per item between
  two copies of the core's loop nest; branches around each acquire cost GemmQ 3%.
- Levers that worked: keeping the A strip resident through K parts (`prefill.gemm.resident_k_part`); reading each
  weight byte once (header units); more queries per attention pass where L1 allows (sliding window: 2 blocks of 32); a
  broadcast KV stream on both of each core's input channels (full attention at 30K: -41%); the last token from the
  prefill's last layer; overlapping the first chunk's inputs; 1024-token sub-chunks for chunks built from them (Llama:
  GEMM items of 1024, fewer fills and configurations); and GemmQ (4 MAC rows) on every 16-row tile format, whatever its
  K chunk (Q4_0 / Q4_1 tiles of 1024 columns, with the header expanded per k-tile slot; their planar GemmQd padded 1024
  tokens to 1152 on 3 rows, and switching took a 4096-token chunk from 507 to 465 ms). Levers that did not work: a C
  ping-pong with one C object, and deeper FIFOs on DMA-bound operators.
- An MoE layer's limits (Qwen3.6-35B-A3B, `moe.prefill`; a 4096-token chunk takes ~2.1 s, with the expert GEMMs at ~31%
  and ~48 GB/s), for a design built from scratch:
  - A work item of a fixed 192 pairs streams its expert's weights once, so hot experts stream their weights again:
    1.45x the experts hit at 4096 tokens and 2.06x at 8192 (items of 384 pairs: 1.15x / 1.42x). Sizing the items to the
    expert's pairs (more tokens per weight pass) would fix this, but GemmQd's L1 (A, D, C and the SiLU scratch at 64
    tokens per MAC row) cannot hold them.
  - The down projection's outputs (bf16, 4 KB per pair) are ~40% of its bytes and ~80% of the combine's. Rows of bfp16
    blocks would halve them, if each C object's per-token run is made of whole blocks that the DMA can move (multiples
    of 32 values; Q6_K's 11-row tiles in pairs give runs of 88).
  - The router's rows per column (32 experts x [hi, lo] plus the shared gate's pair: 66) need a second 64-row pass for
    one pair (0.8 of its 2.0 ms per layer at 4096). Fix: a router that fits in one pass, with the shared gate placed
    elsewhere.

## Pitfalls

Besides the pitfalls in AGENTS.md (kernel names, the 2 s limit), the NPU pitfalls in `docs/development.md` (statics,
host writes and flushes, the 4 GB limit) and those in aie2p-coding section 8 (join offsets, memtile channels, chained
fills within 14 shim BDs, zeroed strides of size-1 dimensions):
- A GemmQd must stay within 64 passes per block.
- A bigger Worker `stack_size` takes L1 that the operator's budget must still fit.
- Test fixtures: the overlay's capacity must hold a chunk plus its padding (`max_positions`).
- A K that is half a chunk short of whole chunks (a hidden size of 2560, an ffn of 3584 at kc 1024: the image's
  `Region.half` regions, `tails`): the GEMMs read rows of `K_in`, so the buffers their A panels come from (the norms'
  `out_pitch`, HP) must have rows of `prefill.gemm.input_k` of the regions that read them, never the padded K (a padded
  pitch silently misreads every row after the first). The regions that read one buffer must either all be half regions
  or none of them.
- A script or test that calls the prefill's `run` itself (rather than `OverlayModel.prompt`) must mark the overlay stale
  before its next dispatch (`OverlayModel.stale`, or `dev.stale = True`): the chunk's configurations replace the
  overlay's code pages, so a paged overlay (Qwen3.6-35B-A3B, Occamy) runs stale pages and hangs (TDR) at its second
  dispatch after a chunk.
- A GEMM's weight argument spans its region's 8 columns, which in a column-major image is nearly the whole image
  (Llama 8B: 4.3 GB, past the 4 GB argument limit). `TileImage` lays out a column-major image in blocks of whole layers
  under 3 GiB (`BLOCK_BYTES`), and the decode's weight runs use one task per block.
