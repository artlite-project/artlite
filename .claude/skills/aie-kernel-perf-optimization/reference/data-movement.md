# Data movement: DMA, streams, overlays

For operators whose kernels wait for data. Judge changes here by e2e p10 and by bytes moved, not
by kernel II.

## Contents
- Rates and ceilings
- Layout: move fewer bytes, in pop order
- Streams, FIFOs and BDs
- Task queues and transactions (host-built sequences)
- Overlays, libraries and fused ELFs
- Host side of a dispatch

## Rates and ceilings (Strix, NPU2)

| path | rate |
|---|---|
| a DMA channel, a stream-switch port | 64 bit per cycle = 8 B at 1.8 GHz = 14.4 GB/s (AM027) |
| the core's own stream port | 32 bit (ObjectFifos do not use it) |
| a core's input DMA channel (S2MM) | 8 B per cycle: every consumer of a broadcast receives the whole stream at this rate |
| one shim MM2S channel reading DDR alone | ~13.8 GB/s: 7.7 B per array cycle (its 8 B stream) |
| one shim tile, both MM2S channels | ~27 GB/s: 15.5 B per cycle (the shim DMA's 128-bit AXI master, AM020) |
| all shim channels together | ~57 GB/s read or write: the NPU's one 32 B / cycle fabric port. Not full from 4 channels on 4 shims: one MM2S channel each into memtiles ~50 GB/s together, on 8 shims ~55; sd15_v2's 4 B streams into cores (5760-byte core BDs) 11.2 GB/s each, MAC on or off |
| reads and writes at once | ~76 GB/s together |
| DDR gathers | rows ≤ 256 B at a multiple-of-4 KB stride share one DRAM channel: ~15 GB/s in all (512 B: ~31), when the channels' rows start alike; offset per column (a GEMM's output slices) they run at the port rate |
| BD fields | 64 B bursts -16%, ≥ 128 B full; AxCACHE 0 halves the rate (Artlite: 512 B, AxCACHE 2) |

All shims' DDR traffic goes through one port to the Infinity Fabric, 32 B per array cycle (AMD's Strix Point SoC
slide: "NPU Datapath - 32B/cycle"; 31.7 B measured at 800, 1267 and 1810 MHz): not the fabric or memory clocks, not
IOMMU translation (4 KB vs 2 MB pages: the same rate), not latency. Gathers of rows ≥ 1 KB, or at a stride off
4 KB multiples by 64-256 B, run at the port rate.
No setting raises it; the SMU's `average_ipu_reads` (amdgpu `gpu_metrics`, MB/s) reads it directly.
Decode streams every weight once per token, so its tokens/s is this ceiling / bytes per token: the iGPU (~190-199
GB/s) decodes the same model ~3.6x faster. Single-token decode reads ~55-56 GB/s, ~97-98% of the ceiling (Llama 1B:
the A channels starved on memory half the time, backpressured 1%); the remaining levers are fewer bytes per token
(speculative decoding on the multi-token step, MoE, low-bit formats, tile padding).

Where the columns' streams start in DRAM matters, though no microbenchmark showed it: the 8 column streams of a
column-major image at an odd multiple of 256 KB apart (`overlay/image.py` `_col_stride`: each in its own 256 KB slot
of a 2 MB span) decode 1B Q4_K_M in 15.0 instead of 15.4 ms and Gemma 4 E4B in 55.8 instead of 57.5; starts near one
offset modulo 2 MB, or a third or a half of it apart, cost up to 3%.

There is no system cache on the NPU path: reuse distance is irrelevant, every re-read costs DDR
bandwidth.

## Layout: move fewer bytes, in pop order

- **Deliver data in consumption order**:
  - memtile `to_stream` (4 dims, 32-bit granularity) can re-lay a tile into the order the
    kernel pops;
  - host prepacking of weights is free (at load time, lossless, the same size);
  - producers write the format the next consumer reads (the prefill norms and GdnOut emit the next GEMM's
    bfp16 A panels; the decode's producers drain into the GEMV's token-interleaved layout).
- **bfp16 panels** `[rows / 8][K / 8][8 x 9 B]` carry activations in 56% of bf16's bytes.
- **Tokens per weight pass**: a weight-streaming GEMM re-reads all weights once per token chunk (GemmQ, 32 -> 64
  tokens per core row: GEMMs -15..26%); L1 bounds the tokens per pass.
- **Queries per pass over a KV cache**: a flash attention re-reads its KV head's history once per
  query block. Llama 3B prefill: 32-query blocks were DMA-bound on the re-reads (12.09 of 12.16 ms
  with the compute compiled out); 64-query blocks, each of the 4 core rows taking 16 queries of the
  column's 3 heads, halved them: -25%, then compute-bound.
- **Every column reads every input** (the decode overlays): a phase's input goes to all 8 columns, each from its own
  shim: 8 copies through the one port and 8 fill tasks at every phase boundary. Per step: Gemma E4B 42 MB of 3.1 GB
  (95% of it the same bytes in every column; packed to the step's slots, `overlay.decode.packed_slots`, from 100 MB),
  qwen35 9B 124 MB of 5.2 GB, 27B 351 MB of 16.4 GB (half of it per-column GDN states). One broadcast copy would save
  ~1-1.7%: the bytes, and 7 of the 8 fills between a drain wait and the next phase (their firmware ops are ~60% on
  the critical path). The overlays cannot take it as they are: the X stream also carries per-column objects
  (residual slices, attention and GDN objects), and each core's two input channels are taken (A, X). **A new design**:
  phase inputs on one broadcast stream from one shim (over 2-4 shims if one channel's 13.8 GB/s would pace a phase),
  per-column objects on another channel (the weight stream, or a neighbour core's free input channels).
- **Broadcast what every column needs**: a GEMM's activations per core row from one shim channel to its memtile,
  then to that core row of all 8 columns, are read once for all columns instead of 8 times (0.14 vs 1.15 GB per
  chunk). One stream per core row, from different shims: one shim channel for all rows paced the GEMMs at 96 tokens.
- **KV cache per KV head** on its own stream, broadcast to the head's attention cores; the kernels read cache objects
  in the FIFO buffers, with no staging copies.
- **A broadcast is bounded by its consumers' input ports, not only by the shim.** The Gemma prefill's full-attention
  layers stream a KV head's whole history to each of its 16 cores once per 32-query pass. Through one input channel
  per core (8 B / cycle) a 2048-token chunk at 30K positions took 259 ms, the same with every kernel compiled out.
  **Feed each core on both input channels** (as arXiv 2609.21264's fused attention: Q + K on one inbound path, V on
  the other): two streams per KV head, each from its own shim through its own memtile, each carrying half of every
  cache object: 153 ms, bit-exact (144 with the kernels compiled out: ~49 GB/s, near the DDR ceiling).
  - Split the objects so that each kernel loop still reads two block-vector streams: by d-block halves (d-blocks
    0-31 of a K record's hi and lo panels on one stream, 32-63 on the other; the memtile interleaves them per d-block;
    a loop per stream object, the accumulation order unchanged). Split by part (K hi on one stream, K lo on the
    other), the score loop reads three streams (SKILL.md section 4): II 12 instead of 6, 249 ms.
  - A core's per-pass input rides ahead on the same streams: the pass's 16 Q objects in a row, each core copies its
    own (the paper's duplicated Q broadcast); the producer writes them so that the memtile's walk restores them.
  - One shim task per pass and stream: Q, history and chunk as one BD chain (`chained_fill` with `ChainPart`s:
    BDs of other buffers, scratchpad offset / length parameters).
  Further: more queries per pass (fewer re-reads; bounded by L1: the fp32 O of head_dim 512), or cores splitting the
  positions and merging partial softmax results (below).
- **Read each byte of a tile once.** A gather of fixed-size units wastes the rest of a unit when a part is smaller:
  Gemma's 512-column Q4_0 tiles have a 1 KB header, read as a 4 KB slice-sized unit per K chunk (1.33x the tile
  bytes). One unit holding the headers of 4 consecutive K chunks (`GemmQOperator.hg`, `hdr_groups`) reads the
  bytes once (gate/up's DMA floor 0.81 -> 0.68 ms per 288 tokens; the GEMM itself only 1.18 -> 1.13: it was
  MAC-bound, measuring.md "Ablations").
- **K parts keep a long-K GEMM's A strip resident**: a GEMM whose A strip (m tokens x K) exceeds the memtile refetches
  it per pass. Split K into parts of whole chunks within a chunk group (`prefill.gemm.plan_phase(k_part=)`, GemmQ
  `kc_group` + `rt_stride`: the parts' tiles at a stride), each part's strip resident, the parts' outputs summed in
  fp32 by the consumer (the residual add: `ResidualNormOperator(deltas=)`). Gemma's down (K = 10240) in 4 parts:
  faster and fewer C-tile roundings (numerics.md). Parts of one width share one design (one ConfigureOp).
  `prefill.gemm.resident_k_part` picks the parts where a strip would be refetched, by bytes (the refetches against
  each part's C written and read by the residual add) and ~50 us per part: Qwen 27B's down in 4-6 parts, its output
  projections in 2 (prefill +3.6%); the residual add sums any number of parts (`PrefillNormOperator` hold design).
- **Write long runs, not scattered pieces**: an output scattered in 36-byte pieces at a 72-byte stride (two cores
  writing the halves of each panel) ran a 27B norm at 468 instead of 257 us for the same bytes. Assemble whole rows in
  the memtile (a one-input join whose `from_stream` interleaves the pieces: `prefill.norm` hold design) and let the
  shim write the rows.
- **Split a shared stream between cores instead of broadcasting it** when each core's stream or compute binds: a
  broadcast halves the DDR reads, but each core still consumes the whole stream. Decode attention of two cores on
  one KV head: each takes half the cached positions for all query heads of both, one leaves its partial (m, l, y) in
  its memory, and its neighbour merges it through shared memory and emits the output (qwen35 27B at 32K
  positions: 363 -> 335 ms per token; the broadcast: no change). A depth-1 ObjectFifo between the two serves as the
  token: the producer acquires it before overwriting its partial and releases it when complete, the consumer
  acquires it to merge and releases it after. Size L1 for the doubled query rows.

## Streams, FIFOs and BDs

What builds and what hangs (walks, links, pool sizes, BD counts and fields, shim BD iteration, runtime lengths and
offsets): aie2p-coding section 8. For speed:

- **Broadcast consumers run in lockstep**: an object is released when all consumers release it. The slowest core
  paces the row; balance the work per core.
- **Release early**: split work across kernel calls so an input object can be released as soon as its stage is done
  (a per-k-tile stage in its own call). A core that gathers what it needs from a shared stream copies it out and
  releases the object.
- **A shim producer can broadcast across columns**: one KV head's stream feeds the 8 cores of its two columns (qwen35
  prefill attention), halving the shim reads.
- **Deeper L1 / memtile FIFOs do not help a DMA-bound operator**, and they cost BD ids (24 per memtile channel
  parity): the Gemma decode overlay at A depth 20-22 instead of 12 was unchanged (rejected.md).
- **Strided fills**: a pass that needs rows [r0, r0 + n) of every super-block of a tile reads them with one 4-D
  pattern (`(a // 8, 2 n, a // 8, n_sb * a // 4)`-style dims) instead of n_sb tasks.
- **Runtime-variable lengths cost nothing**: a scratchpad `length_parameter`, applied by the firmware at dispatch (256
  fills per token: no measurable cost).

## Task queues and transactions (host-built sequences)

`artlite.txn` and `overlay.runtime.Sequence`:
- **Addresses**: shim BDs carry final addresses (paddr + 0x80000000).
- **Queue limit**: at most 4 queued tasks per shim channel between syncs. A BD is reusable once
  its task completed.
- **Rolling retirement** (`Sequence(rolling=True)`):
  - before a 5th task on a channel, the transaction waits for that channel's oldest;
  - 192 fills per column then cost the same as 3.
  - `sync()` waits for the drains, `sync(fills=True)` for everything;
  - a task carries a completion token only where a wait takes it (a channel completes its tasks in order: one wait
    on its last covers the others); the push is patched when the wait is issued (`Txn.set_token`).
- **Issue order decides deadlock.** A wait for the oldest weight fill needs the cores to consume
  it, and they need their X inputs first. Issue a phase's X fills and C drains before the bulk of
  its weight fills; prefetch the next command and first fills before a sync.
  - Many tasks per column: issue them in rounds over the columns, each round's drains before the next
    round's fills, so the oldest task a fifth one waits for never depends on a later round (all fills
    first hung: a fill waited for objects the core could only take after its drains).
  - A phase that reads what an earlier phase drained must wait for those drains (`seq.sync()` before its
    input fills; issue its weight fills before the sync so they keep streaming): the channels' queues run
    independently, and a fill issued after a drain can read DDR before the drain has written it.
- **Broadcast streams and several consumers**: every consumer of a broadcast object must take it, in an
  order compatible with the others'. A core that skips a phase's objects skips them in the order the
  phase's own core consumes them; otherwise each waits for an object the other holds back (deadlock).
- **Coherence**: host writes must be clflushed before a dispatch (`OverlayDevice.flush`); NPU
  writes must be fetched before the host reads them.
- **Task cost**: a BD write ~0.08 µs, a queue push ~0.15 µs, a token wait ~0.3 µs per column
  (timed in a host-built transaction); one column's fill -> the cores take it -> token -> the
  firmware's wait: 2.3 µs. On the Gemma decode step most waits were not the limit (a token and a wait per task vs per
  channel and sync: the same 57 ms): count tasks as well as bytes, but measure before cutting them.
- **The ops between a drain wait and the next phase's fills are on the critical path**: each one delays the next
  phase (a queue wait ~0.3 µs, a BD ~0.08, a push ~0.15, per column). `Sequence(inputs=("X",))` waits for the input
  fills before the drain wait (they are complete when the drains are), so the fills after it find empty queues:
  Gemma step 3751 -> 1366 waits, -0.25 ms (qwen35: no change). Extra BD writes on the X fills cost ~60% of their
  firmware time in step time; on the weight runs nothing (issued far ahead). Fewer tasks per boundary (one broadcast
  fill instead of one per column) is a design-time lever.
- **Watchdog**: a dispatch must finish within the driver's TDR (2 s): size chunks for < 1 s.
  A killed job fails with `DispatchError ... TIMEOUT (state 8)`; for a host-built transaction
  its message names the operations at the firmware's `txn_op_idx` (where it stopped).

## Overlays, libraries and fused ELFs

- **A change of design inside a fused ELF is a reset and a full reconfiguration**: ~80-90 µs in a library
  (the reset ~40 µs, the tiles' writes by control packets: `fused.ctrlpkt`). One design whose workers run every
  phase removes it (the Llama decode step: 13.4 ms on one overlay, 25 as a runlist of designs). Design for it from the
  start:
  - one weight-object size;
  - one broadcast stream: the inputs every column needs come once per phase (one fill, one copy through the port),
    per-column objects on another channel;
  - a fixed phase order, or command objects that tell the cores what to do (gemma).
- **Relocation addends are 32-bit**: a runtime-sequence argument must stay below 4 GB. The fused
  scratch goes as segments (`fused.layout.SEGMENT_BYTES`); host-built transactions carry 64-bit addresses.
- **`FusedOperator(expand_load_pdis=False)`** for a single-design ELF: the ConfigureOp is skipped
  when the PDI is unchanged (-0.28 ms per dispatch). Multi-design ELFs: keep the default.
- **ConfigureOp cost** (a library's): the PDI load's reset ~40 µs, ~0.1-0.13 µs per transaction operation left
  (stream switch ports, the cores' enables, the shim), a few µs of packets per column. With program memory by control
  packets, size each design for its work, not its configuration (Llama chunks of 64 / 128: GemmQ over 1-2-row GemmQd,
  the attention on 4 rows over 1, the norms on 4 columns over 2: -10 to -12%); by blockwrites (~1.5 µs per KB) fewer
  cores would win on short prompts.
- **An ELF's control code must fit its device's 64 MB heap** (each `FusedOperator` opens its own device):
  a multi-design runlist inlines ~0.25 MB of configuration per entry (qwen35 prefill: 310 entries = 79.6 MB,
  `CREATE_BO ... Resource temporarily unavailable`). Long programs: `fused.Library` (each design once) and
  `Library.run`, which assembles the program on the host and dispatches it in pieces. (One aiecc ELF of
  thousands of entries is no alternative: aiecc's time grows superlinearly, 14 min for 1216 entries.)
- **Count a layer-major program's design switches.** Every change of design is a ConfigureOp (above): the Gemma
  prefill switches design 12 times per layer (a 1152-token chunk: 279 configurations, 23 ms alone, 7%). Data memory
  contents still go by blockwrites (~1.5 µs per KB): tables as `initial_value` (the GELU epilogue's GEMM: 195 vs
  107 KB of PDI) cost on every switch; the per-design PDIs of a library ELF are in its cache dir
  (`design.mlir.prj/op<N>_<Class>.pdi`).
- **Static buffers are rewritten on every reconfiguration** (`.bss` zero-fill travels in the PDI):
  8 KB x 8 cores = +80 µs. Use worker `Buffer`s, without `initial_value` (it travels too: a zeroed workspace of 760 KB
  was most of a 1.29 MB PDI).
- **Program memory is the tight resource** in overlays (16 KB for all phases): SKILL.md section 7.
- **Library entries of one array design share one compiled device** (`fused.Library`: an overlay's
  operators differ only in their runtime sequences): one design compiled, not one per entry (the SD 1.5 UNet: 157
  entries, 3 devices, ~2 min). Runtime-task BD ids are numbered per device, so pin the static design's memtile BD ids
  (`Bd(bd_id=)`): variants of a design must configure alike.
- **Variants of one design switch by a program-memory patch** (`fused.program.program_memory_patch`: the differing
  words, no PDI load): ~0.2-0.4 ms instead of a 1.15 ms reconfiguration
  (SD 1.5: a conv variant and a transformer variant, 20 switches per step). The variants' shared code must be
  byte-identical (artlite.operators.sd15.overlay's docstring: weak trampolines, an ordering table, stubs with an
  opaque side effect, no deferred helpers); check `llvm-nm -n` of both core ELFs. The core main must be identical
  too, so a design opts in (`ArtcOperator.PROGRAM_MEMORY_VARIANTS`): the cores wait for the next entry past their
  loop's setup, so variants whose mains differ (GemmQd K parts of different k-chunk counts) would run with the first
  one's (NaN).
- **A ConfigureOp's single writes cost as much as its program memory.** aiecc's expansion writes, per core, every
  lock value (0 after the reset), the DMA channels' and the core's reset pulses and each stream switch port alone:
  ~2500 operations for a 32-core design. `program._prune` drops what the PDI load's array reset already holds and
  writes a tile's ports as one blockwrite per range (Llama 3B chunk of 1024 tokens: 512K -> 222K configuration
  operations, 362 -> 341 ms, bit-identical). `fused.ctrlpkt` sends each core's program memory and each core's and
  memtile's BDs, locks and DMA channels (start queues included, in their order) as control packets: per column the shim
  DMA streams them up a route set after the reset (a packet split at each tile: its id to TileControl, the rest
  North), a marker word polled, the route's ports zeroed by the design's own (a 24-core GEMM: 854 -> 306 operations;
  Llama 1B chunk of 128 tokens 43 -> 31 ms, Gemma / Qwen chat TTFT -20%). A slave port has 4 slots: match the ids
  above by a differing bit, not one slot each. Firmware PDI loads (`expand_load_pdis=False`) are slower: +1.1 ms per
  switch.
- **Control code is the fixed cost of a small operator**: ~0.1 us per TXN operation (measured with injected
  write32s), and an overlay operator issues 300-900 (memtile BDs and their address fields, shim BDs, address patches,
  queue pushes, task waits): 35-65 us per operator before any data moves. The library assembler leaves out memtile
  BD writes the registers already hold (not BDs with an iteration: the DMA advances word 6's current step), -24%. A
  program of one piece run again reuses its assembled control code.
- **Two cores on one object through shared memory**: post the other core's command and release it before
  computing your own half, or the halves run one after the other (a GEMV phase: 29 -> 47 GB/s).
- **Separate hardware contexts** (e.g. decode + prefill designs) cost ~0.8-2.5 ms per switch.
  Share device buffers instead of copying: `FusedOperator(external_buffers=...)`, or one weight
  image read by both designs (gemma).
- **Phase boundaries** in a DMA-bound overlay lose ~5% (cores wait for the next broadcast; the Gemma decode: ~11 us
  from a phase's last tile to the next phase's input, ~8 us of it the sync's round trip, then ~5-10 us of input
  strips, 295 boundaries per step). Queue the weights that follow a sync before it (they keep streaming). Route
  the weight stream through a memtile FIFO so DDR keeps streaming across the boundary.

## Host side of a dispatch

- **Waiting**: a blocking driver wait adds 0.3-1.7 ms and leaves the core cold. Nap in 0.2 ms
  slices until just before the expected end, then spin
  (`FusedOperator.__call__(spin_after_s=..., nap_s=0.0002)`).
- **Overlap host work**: prepare the next chunk's inputs on a worker thread while the NPU runs
  (gemma `NpuGemma.prompt`). A decode step's position patches (cache ranges, tails, token objects: ~0.15 ms of numpy)
  for the next step while this one runs, on a second transaction buffer (`ModelOverlay.AHEAD`, `OverlayDevice.submit`
  / `wait`; a patched step runs only at its position and ring state); the embedding row in C++
  (`_native.dequant_rows`): Llama 1B's step 14.92 -> 14.75 ms.
- **The first chunk's late inputs**: start the program with what its first operators read and write the rest
  before the piece that first reads it (`Library.run(hooks={item: fn})`: a hook runs before that piece is
  dispatched, while the previous one runs; the item starts a piece). The Gemma prefill's per-layer inputs (a 15 ms
  host projection) overlap layer 0's first operators (`GemmaPrefill.run` with a `Future`).
- **A prompt's last token**: a prefill that computes only the layers that own a cache still needs the last token's
  logits. Finish that token from the prefill's last layer with a decode step that starts there
  (`GemmaOverlay.step(start=)`; x from the chunk's residual) instead of a whole decode step: 19 of 42 layers' weights
  (-25 ms per prompt). If the overlay's image expects a header (post-norm update) at that layer, keep it and make
  the update zero.
- **Read only what the host needs**: top-k from block maxima, not 128K logits. numpy beats torch
  for 2-3K-element work.
