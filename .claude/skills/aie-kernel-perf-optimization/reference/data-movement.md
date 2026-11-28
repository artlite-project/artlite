# Data movement: DMA, streams, overlays

For operators whose kernels wait for data. Evaluate changes here by end-to-end p10 and by bytes moved, not by kernel
II.

## Contents
- Rates and ceilings
- Layout: move fewer bytes, in pop order
- Streams, FIFOs and BDs
- Task queues and transactions (host-built sequences)
- Overlays, libraries and fused ELFs
- Host side of a dispatch

## Rates and ceilings (NPU2: Strix Halo, and Krackan Point where it differs)

| path | rate |
|---|---|
| a DMA channel, a stream-switch port | 64 bit per cycle = 8 B at 1.8 GHz = 14.4 GB/s (AM027) |
| the core's direct stream port | 4 B per cycle (7.2 GB/s) each way; ObjectFifos do not use it, and the core spends an issue slot per 32-bit word (aie2p-coding section 2). A broadcast that includes a core port runs at that port's rate (a DMA consumer on it gets 7.0 instead of 13.2 GB/s) |
| a core's input DMA channel (S2MM) | 8 B per cycle: every consumer of a broadcast receives the whole stream at this rate |
| one shim MM2S channel reading DDR alone | ~13.8 GB/s, i.e. 7.7 B per array cycle of its 8 B/cycle stream; Krackan: 14.2 GB/s |
| one shim tile, both MM2S channels | ~27 GB/s, i.e. 15.5 B per cycle (the shim DMA's 128-bit AXI master, AM020); Krackan: 28.1 GB/s |
| all shim channels together | ~57 GB/s read or write, set by Strix Halo's single 32 B/cycle fabric port (Krackan: 97.8 GB/s read, 87.2 GB/s write, 54 B per cycle, 81% of its 120 GB/s bus). Four channels on 4 shims do not fill it: one MM2S channel per shim into memtiles reaches ~50 GB/s in total on 4 shims and ~55 GB/s on 8; the vision overlay's 4 B streams into cores (5760-byte core BDs) reach 11.2 GB/s each, with the MAC on or off |
| reads and writes at once | ~76 GB/s combined (Krackan: 88.2 GB/s, no more than its writes alone) |
| DDR gathers | rows <= 256 B at a multiple-of-4 KB stride hit the same DRAM channel, capping total bandwidth at ~15 GB/s (~31 GB/s for 512 B rows) when every channel's rows start at the same offset; offset per column (as a GEMM's output slices are), they run at the port rate. Krackan scales the same way: 28.6 / 55.9 / 95.6 GB/s for 256 B / 512 B / 1 KB rows, and 89.6 GB/s when skewed by 256 B |
| short DDR write rows | writing a GemmQ pass's C slice (64 columns per token = 128 B rows) adds ~25% to a short-K GEMM: qwen35 1B's `gdn_in` (K = 1024) takes 0.22 ms per item vs. 0.17 ms when each pass's C is written whole. Writing rows at the column's pitch (2.3 KB stride) instead of the full row (18 KB) did not help, because the row length is the bottleneck |
| DDR writes of partial lines | a row that starts or ends inside a 64-byte line writes only part of that line, which the memory then merges: 132 B rows sharing their end lines reach 41.4 GB/s on Halo vs. 57.2 GB/s for whole-line 128 B rows, and 15.6 vs. 73.2 GB/s on Krackan Point (LPDDR5 masked writes perform a read-modify-write in the DRAM; `artlite-probe`'s write rows). The prefill therefore writes whole lines: planar passes in windows that start at line boundaries (`GemmQOperator` `lines`, which carries the previous pass's last columns in the C object), gate/up's 32-output parts two per C object (576 B rows), and `d_pitch` in whole lines |
| a row per page | the NPU translates every page a DMA row touches. Rows on separate pages (stride >= 4 KB) run at ~45-60 M rows/s in total on Windows, which maps system memory for the NPU in 4 KB pages: the skewed 256 B gather reaches 14-16 GB/s, and 128 B write rows 5-6 GB/s (Halo with transparent huge pages off: 95-170 M rows/s, 29.3 / 12.2 GB/s). Linux host buffers of 2 MB and larger use 2 MB pages (`Device.alloc_host_buffer`); a streaming read over 4 KB pages loses little (Windows Krackan: 81 vs. 97 GB/s read). Rows sharing a page are free once its translation is cached (one cache shared by all channels: 512-1024 pages on Halo), but back-to-back row writes to an uncached page each pay a translation miss (8 rows of 128 B per page: 406 M rows/s read, 251 M rows/s written; revisiting 8 pages per channel: 433 / 434 M rows/s; `artlite-probe --pages`) |
| BD fields | 64 B bursts lose 16%, bursts >= 128 B run at the full rate; AxCACHE 0 halves the rate (Artlite uses 512 B bursts and AxCACHE 2) |

On Strix Halo, all shims' DDR traffic goes through a single port to the Infinity Fabric at 32 B per array cycle (31.7 B
measured at 800, 1267 and 1810 MHz). The limit is not the fabric or memory clocks, not address translation (a stream
runs at the same rate on 4 KB and 2 MB pages; for rows on separate pages, see above), and not latency. Gathers of rows
>= 1 KB, or at strides offset from 4 KB multiples by 64-256 B, run at the port rate. No setting raises it:
`ME_AXIMM_CONFIG.MASTER_SEL` (0x1E020) selects AXI-MM or streaming for the shim's single NMU rather than adding a second
master (AM025), and AM029's `NMU_Switches_Config` belongs to a different array-interface layout (its shim BDs are at
0x9000, NPU2's at 0x1D000). The SMU's `average_ipu_reads` (amdgpu `gpu_metrics`, MB/s) reports the rate directly. The
port is a property of Halo, not of the array: Krackan Point's NPU, at the same 1.81 GHz, reads 54 B per cycle (AMD's
Strix Point slide says "NPU Datapath - 32B/cycle", without a clock). A column's DRAM share is therefore ~3.9 B per array
cycle on Halo and ~6.8 B on Krackan; GemmQd's cost model (`gemm_qd.k_tile_cycles`) assumes Halo's.

Decode streams every weight once per token, so its tokens/s equals this ceiling divided by the bytes per token: the
iGPU (~190-199 GB/s) decodes the same model ~3.6x faster. Single-token decode reads ~55-56 GB/s, ~97-98% of the ceiling
(Llama 1B: the A channels were starved on memory half the time and backpressured 1% of it); the remaining levers are
fewer bytes per token (speculative decoding on the multi-token step, MoE, low-bit formats, tile padding). On Krackan
Point, two limits appear that Halo's port hides. First, each column streams its weights on one shim channel (A: MM2S
0): on Halo, the 8 channels saturate the port at 56.4 GB/s; on Krackan they reach 80.1 GB/s (~5.5 B per array cycle per
column, or one 9280-B tile every ~1700 cycles; all 16 channels reach 97.8 GB/s). Second, the main core: on Krackan, a
decode step takes max(bytes / 80.1 GB/s, main-core busy time) plus about 1 ms of phase boundaries (Llama 1B Q4_K_M with
row-major Q4_K tiles: 10.9 ms measured on Krackan vs. 9.2 ms of main-core work counted on Halo). Count main-core time on
Halo, both from per-core cycles minus lock stalls over a step (with the cores' counters programmed in the step's
transaction, measuring.md) and from the tile windows (`DecodeOverlayOperator(win=)`). Llama 1B's main core is busy 6.7
of 11.9 ms on Q4_0, 7.6 of 12.6 ms on Q4_K_M (word-major Q4_K tiles: 1086 cycles per half tile in the file's mix, plus
285 cycles per tile for the row epilogues), 12.8 of 15.0 ms on Q5_K_M, and 13.7 of 17.0 ms on Q6_K (row-major, where the
transposes take a quarter of a half tile). If a format's main-core time exceeds its transfer time at 80 GB/s, it is
core-bound on such a chip, and a second weight channel per column helps only below that threshold.

Where the column streams start in DRAM matters, even though no microbenchmark showed it: placing the 8 column streams of
a column-major image an odd multiple of 256 KB apart (`decode/image.py` `_col_stride`, which staggers each into a
separate 256 KB slot across a 2 MB span) decodes 1B Q4_K_M in 15.0 instead of 15.4 ms and Gemma 4 E4B in 55.8 instead
of 57.5 ms. Starts near the same offset modulo 2 MB, or a third or a half of 2 MB apart, cost up to 3%.

The NPU path has no system cache: reuse distance is irrelevant, and every re-read costs DDR bandwidth.

## Layout: move fewer bytes, in pop order

- **Deliver data in consumption order**:
  - memtile `to_stream` (4 dimensions, 32-bit granularity) can re-lay a tile into the order the kernel pops;
  - host prepacking of weights is free (done at load time, lossless, same size): store a weight tile the way its kernels
    load it. A MMUL panel is 8 rows x 8 k, one 4-byte word from each of 8 rows. With a row-major tile
    (`[row][its words]`), every reader has to build the panels: the decode GEMV with transposes on the core (Q4_K: 416
    of ~1180 cycles per half tile, and its stores into the tile collided with the tile stream's DMA; removing them cut
    memory stalls from 0.74 to 0.36 ms per step), and the prefill with a memtile walk and header kernels that picked
    rows apart. Word-major tiles (`[word][row][4 B]`, where the 32 B of a word's 8 rows form a panel,
    `TileFormat.word_major`) are free to consume: main-core work on Llama 1B Q4_K_M drops from 9.2 to 7.6 ms per step
    with bit-identical logits. Choose the word order the decode pops (`[i][s][row]`) and check the prefill side: its
    S2MM walk must fit within 4 dimensions (here it needs none, since the slice lies as the memtile holds it), and the
    header unit arrives through the same walk from before the tile. A tile whose row count is not a multiple of 8 (13
    for Q5_K, 11 for Q6_K) needs load-FIFO pops for its words and a separate path for its header unit, whose bytes a
    word-major walk would interleave with the preceding tile's;
  - producers write the format that the next consumer reads (prefill norms and GdnOut emit the next GEMM's bfp16 A
    panels; decode producers drain into the GEMV's token-interleaved layout).
- **bfp16 panels** `[rows / 8][K / 8][8 x 9 B]` carry activations in 56% of bf16's bytes.
- **Channels padded to whole passes** (a core's 80-channel C tile, with 128 real channels of 160): pad only the C tiles,
  and drain or read just the real kbs (memtile BDs with a slices dimension, or a 4-dim walk whose 4th size is implied
  by the task's words, i.e. 3 sizes plus a repeat). A kernel output padded to whole output chunks lands in junk rows of
  the receiving tensor (sd15's VAE decoder: `vae.Chans`, `Attention512`'s 80-kb output).
- **One image on a design built for two**: treat its left and right halves as the two images. A lane shift moves a
  whole slot (`[kb][image 2]` panels) as one byte stream, so the second half's cut lane receives the first half's edge
  lane, and the pixel across the seam comes for free. Only the masks change (sd15's VAE decoder:
  `core.shift_masks(joined)`), and the per-image statistics are summed (the GroupNorm apply's `C_JOIN`).
- **Tokens per weight pass**: a weight-streaming GEMM re-reads all weights once per token chunk (raising GemmQ from 32
  to 64 tokens per core row cut GEMM time by 15-26%); L1 bounds the tokens per pass. This helps only where the stream is
  the bottleneck: the vision overlay's m-64 GEMMs are core-bound (an fp32-C MAC call takes 1006 cycles per k-tile vs.
  ~930 cycles of B stream, and with the MAC compiled out an op still takes 76% of its time), and a second fp32 C tile
  does not fit in their L1 (rejected.md).
- **Queries per pass over a KV cache**: a flash attention re-reads its KV head's history once per query block. In Llama
  3B prefill, 32-query blocks were DMA-bound on the re-reads (12.09 of 12.16 ms with the compute compiled out); 64-query
  blocks, with each of the 4 core rows taking 16 queries of the column's 3 heads, halved the re-reads: -25%, after which
  the attention was compute-bound.
- **A sliding window from the memory tile** (`prefill.attn.AttnOperator(ring=)`): the Llama path's packed attention
  re-read its window of 9 cache groups from DDR on every pass, at the port rate. When each column's memory tile holds
  the history and a ring of the last W + 1 groups, sent whole on every pass (runtime memtile tasks: aie2p-coding section
  8), each group is read once, and the cores skip the slots outside their window. The attention then becomes core-bound
  rather than byte-bound (Qwen3-Reranker-0.6B, 4096 tokens: 3.25 -> 2.45 ms per layer on random data vs. ~1.9 ms by
  bytes): a key chunk that the cores skip still crosses their input channel at 8 B/cycle and, through two-deep FIFOs,
  does not overlap their compute. Send as few skipped objects as the static program allows (putting the chunk's first
  group in the ring rather than in the always-sent history saves 0.1 ms per layer).
- **Every column reads every input** (the decode overlays): a phase's input goes to all 8 columns, each through a
  separate shim, which means 8 copies through the single port and 8 fill tasks at every phase boundary. A phase's input
  is on the critical path (its tiles wait for it), so read only the step's slots (`decode.protocol.packed_slots`; the
  27B's 8-slot design read 8 slots at every S, and packing cut S 1 from 303.6 to 298.5 ms). Per step, Gemma E4B reads 42
  MB of 3.1 GB (95% of it identical bytes in every column; packing to the step's slots with
  `decode.protocol.packed_slots` brought it down from 100 MB), qwen35 9B 124 MB of 5.2 GB, and 27B 351 MB of 16.4 GB
  (half of it per-column GDN states). A single broadcast copy would save ~1-1.7%: the bytes, plus 7 of the 8 fills
  between a drain wait and the next phase (~60% of their firmware ops are on the critical path). The overlays cannot
  adopt it as they are: the X stream also carries per-column objects (residual slices, attention and GDN objects), and
  both of each core's input channels are taken (A, X). **What works** (`decode.design` "Broadcast inputs", 8-slot
  designs): packet-switching the memtile -> cores segments, so that a core's X channel takes a column's objects and a
  phase's input as a packet flow from one shim to all X consumers. This saves 3 ms per step at every S on the 27B. The
  flows share receivers, so aiecc requires `--allow-deadlock-prone-routing`; that is correct here because a broadcast
  fill follows a sync that drained the previous phase, and a step's first phase stays local.
- **A core's X at depth 1 adds the object's DMA to its work.** The next object cannot arrive while the core holds the
  current one: 544 cycles per 4352 B object on top of the strip. At 8 rows per k block, `down` takes twice as many
  objects (+1.4 ms per step on the 27B from S 4 to S 5). Depth 2 on a main core took ~2 ms off the 27B's step at every S.
  The 4352 B it needs came from moving the core's SiLU table copies into the helper's memory (a neighbour's memory reads
  like local memory, and the gathers run between the helper's tiles) and from moving a sigmoid gather table out of L1.
- **Broadcast what every column needs**: sending a GEMM's activations per core row from one shim channel to its
  memtile, then to that core row in all 8 columns, reads them once for all columns instead of 8 times (0.14 vs. 1.15 GB
  per chunk). Use one stream per core row, from different shims: a single shim channel for all rows paced the GEMMs at
  96 tokens.
- **KV cache: a dedicated stream per KV head**, broadcast to the head's attention cores; the kernels read cache objects
  in the FIFO buffers, with no staging copies.
- **A broadcast is bounded by its consumers' input ports, not only by the shim.** The Gemma prefill's full-attention
  layers stream a KV head's whole history to each of its 16 cores once per 32-query pass. With one input channel per
  core (8 B/cycle), a 2048-token chunk at 30K positions took 259 ms, the same as with every kernel compiled out. **Feed
  each core on both input channels** (as in arXiv 2609.21264's fused attention: Q + K on one inbound path, V on the
  other): two streams per KV head, each routed through a separate shim and memtile and each carrying half of every cache
  object, take 153 ms, bit-exact (144 ms with the kernels compiled out: ~49 GB/s, near the DDR ceiling). The qwen35
  35B-A3B (2 KV heads and two q heads per column, so a core's two row panels are two heads of one token group) takes
  17.2 ms per layer at 16K positions (T 512) on one stream per KV head, or 16.9 ms with every kernel compiled out; on two
  streams it takes 10.6 ms (8.8 ms).
  - Split the objects so that each kernel loop still reads two block-vector streams. Split by d-block halves: d-blocks
    0-31 of a K record's hi and lo panels go on one stream and 32-63 on the other, the memtile interleaves them per
    d-block, and there is one loop per stream object, with the accumulation order unchanged. Splitting by part instead
    (K hi on one stream, K lo on the other) makes the score loop read three streams (SKILL.md section 4): II 12 instead
    of 6, 249 ms.
  - A core's per-pass input rides ahead on the same streams: the pass's 16 Q objects arrive in a row, and each core
    copies out its own Q object (the paper's duplicated Q broadcast); the producer writes them so that the memtile's
    walk restores them.
  - Use one shim task per pass and stream, with Q, history and chunk as one BD chain (`chained_fill` with `ChainPart`s:
    BDs of other buffers, scratchpad offset / length parameters).
  - After that, the single shim channel per stream becomes the bottleneck (~10 GB/s each, so 4 streams reach ~40 GB/s).
    Feeding each stream from two shims, joined by its memtile (part i: d-blocks 8 i .. of every piece), cut the qwen35
    1B's floor from 5.8 to 4.1 ms per layer and the 9B's from 20.0 to 16.9 ms. A join treats each part as a separate
    transfer, so the memtile's walk (`to_stream`) operates within each part; split the parts so that each one can be
    re-laid independently. The join takes two of the memtile's 6 S2MM channels, so the attention's Q slots ride the KV
    stream instead of a dedicated input (the column's output join takes four).
  Further options: more queries per pass (fewer re-reads, bounded by L1: the fp32 O of head_dim 512), or cores that
  split the positions and merge partial softmax results (below).
- **An LM head's high plane** (`decode.screen`, `formats.q6h`): a decode step needs only a few of the vocabulary's
  logits, and the head is 7-40% of a step's bytes (27B: 1.04 of 15.8 GB, 9B: 0.83 of 5.0 GB, 35B-A3B: 0.42 of 1.96 GB,
  Llama 1B and MiniCPM-V 1B: 27-40%). The NPU reads a Q6_K head's scales and the top 2 bits of its codes (with Q2_K's
  tile layout and code loops: 27 rows per tile vs. 11, 39% of the bytes); the host computes the candidates' logits
  exactly from the file's rows, read in place (0.15 us per row and 1024 columns). 35B-A3B: 40.3 -> 36.8 ms per step, 9B:
  94.1 -> 85.4 ms, MiniCPM-V 1B: 11.5 -> 9.2 ms. What makes it exact:
  - per-row bounds computed from that row's codes and scales rather than a single error figure. With w = p + r (the
    plane's part and the rest of a code), the plane's logit is on average 1 - rho times the row's logit, where
    rho = sum a^2 r w / sum a^2 w^2 (0.083: the largest logits come out 2-3 short, which is 4-6 sigma of the leftover
    noise). The estimate is therefore the plane's logit / (1 - rho), and the bound is a multiple of the deviation of
    r - rho w. Without rho, the best token sat 0.5 sigma inside a 6-sigma bound;
  - refinement by rows on the host, not by blocks on the NPU (one candidate row in a block of 16-64 rows makes the whole
    block a candidate, which affects 14-76% of the blocks): seed with the exact logits of the largest screened ones,
    then add the rows whose upper bound reaches the largest exact logit, then a window below it for a sampler's k;
  - checking against the whole exact head on real steps (the estimates' error in sigma, reported separately for the
    largest logits; unused rows, such as reserved tokens filled with a constant code, err coherently and do not
    matter), and at model level (KL against llama.cpp: the exact logits are closer than the cores' bf16 logits).
  Tiles taller than 16 rows require a design's wide frame (in an MoE design, gate / up go through the windows and every
  format takes a page, costing ~1 ms of the 35B-A3B's step) and do not fit a dense 8-slot design (the 27B is 1.6 KB of
  L1 and 0.7 KB of program memory short).
- **Move only the valid rows of a work item**: a fixed-size work unit (an MoE expert GEMM's 192-row chunk, where most
  chunks have a few dozen valid rows) moved its padding rows through DDR along with the valid ones. At 1024 tokens, rows
  made up 44% of the expert GEMMs' DDR bytes, ~80% of them padding, and the down projection's Q4_K chunks ran at the
  combined read + write ceiling (~76 GB/s). Per 64-row strip, the control stream sends either the strip or a marker
  panel (finish-on-TLAST, aie2p-coding section 8); the cores skip the marker strips' outputs, row 0's outputs go
  straight to the shim, and rows 1 and 2's are joined. Chunks of 1024 / 2048 tokens: -14% / -7%, bit-identical.
- **Read each byte of a tile once.** A gather of fixed-size units wastes the rest of a unit when a part is smaller:
  Gemma's 512-column Q4_0 tiles have a 1 KB header, read as a 4 KB slice-sized unit per K chunk (1.33x the tile bytes).
  A single unit holding the headers of 4 consecutive K chunks (`GemmQOperator.hg`, `hdr_groups`) reads the bytes once
  (gate/up's DMA floor drops from 0.81 to 0.68 ms per 288 tokens, but the GEMM itself only from 1.18 to 1.13 ms, because
  it was MAC-bound: measuring.md "Ablations").
- **K parts keep a long-K GEMM's A strip resident**: a GEMM whose A strip (m tokens x K) exceeds the memtile refetches
  it on every pass. Split K into parts of whole chunks within a chunk group (`prefill.gemm.plan_phase(k_part=)`, GemmQ
  `kc_group` + `rt_stride`, which places the parts' tiles at a stride) so that each part's strip stays resident, and
  let the consumer sum the parts' outputs in fp32 (the residual add: `ResidualNormOperator(deltas=)`). Gemma's down
  projection (K = 10240) in 4 parts is faster and has fewer C-tile roundings (numerics.md). Parts of the same width
  share one design (one ConfigureOp). `prefill.gemm.resident_k_part` picks the parts where a strip would be refetched,
  based on bytes (the refetches vs. each part's C written and read by the residual add) plus ~50 us per part: Qwen 27B's
  down projection uses 4-6 parts and its output projections 2 (prefill +3.6%). The residual add sums any number of
  parts (`PrefillNormOperator` hold design).
- **Write long runs, not scattered pieces**: an output scattered in 36-byte pieces at a 72-byte stride (two cores
  writing the halves of each panel) made a 27B norm take 468 instead of 257 us for the same bytes. Assemble whole rows
  in the memtile (a one-input join whose `from_stream` interleaves the pieces: `prefill.norm` hold design) and let the
  shim write the rows.
- **Split a shared stream between cores instead of broadcasting it** when each core's stream or compute is the
  bottleneck: a broadcast halves the DDR reads, but each core still consumes the whole stream. In decode attention with
  two cores on one KV head, each core takes half the cached positions for all query heads of both; one leaves its
  partial (m, l, y) in its memory, and its neighbour merges it through shared memory and emits the output (qwen35 27B at
  32K positions: 363 -> 335 ms per token; broadcasting made no difference). A depth-1 ObjectFifo between the two serves
  as the token: the producer acquires it before overwriting its partial and releases it when the partial is complete;
  the consumer acquires it to merge and releases it afterward. Size L1 for the doubled query rows.

## Streams, FIFOs and BDs

For what builds and what hangs (walks, links, pool sizes, BD counts and fields, shim BD iteration, runtime lengths and
offsets), see aie2p-coding section 8. For speed:

- **Broadcast consumers run in lockstep**: an object is released only when all consumers have released it. The slowest
  core paces the row, so balance the work per core.
- **Release early**: split work across kernel calls so that an input object can be released as soon as its stage is
  done (a separate kernel call per k-tile stage). A core that gathers what it needs from a shared stream copies it out
  and releases the object.
- **Work between a release and the next acquire overlaps the next object's transfer** into a one-buffer FIFO (for a
  core without L1 for a second buffer), so place a kernel call after the release. qwen35's decode attention runs a
  group's softmax (which needs only the scores) after releasing the second K object, while the V object streams in
  (`attn.asoft`): this hides ~1100 of the ~1400 cycles that a 9280-B object takes at 8 B/cycle.
- **A shim producer can broadcast across columns**: one KV head's stream feeds the 8 cores of its two columns (qwen35
  prefill attention), halving the shim reads.
- **A broadcast chosen per fill** (`decode.design` "Shared A objects"): the shim -> memtile segment is packet-switched,
  with a second packet id whose flow reaches two memtiles. Nothing past the memtiles changes (the cores of both columns
  consume the shared objects as if they were local, including MoE's per-column object counts), and every other fill
  stays local, so a step that gains from one read for two columns uses it while the other steps keep a stream per
  column. Two sources keep no order between them: a shared fill follows a sync, and the other column's next fills wait
  for its fills (`Sequence.wait_fills`).
- **A stream is one shim channel**: a chain of cores on one stream reads at ~13.8 GB/s, whatever bandwidth the port has
  left. N columns sharing a KV head trade rows per core against groups per core against streams: N streams with all the
  rows (limited by L1), one stream with 1/N of the rows (half the port at 2 heads), or a combination in between.
- **Deeper L1 / memtile FIFOs do not help a DMA-bound operator**, and they cost BD ids (24 per memtile channel parity):
  the Gemma decode overlay at A depth 20-22 instead of 12 was unchanged (rejected.md). They do help a weight stream
  whose consumer pauses: in an MoE step, the main cores take no tiles during a GDN core's step (~80 us) or during a
  layer's last round trips (partials, combine, next input), and the shim fills the FIFO in the meantime. 14 objects
  instead of 8 took 0.6 ms off the 35B-A3B's 36 ms (12 objects: 0.5 ms). The BDs bound the depth: each FIFO object
  takes one BD from each half of the memtile's BDs, alongside the BDs of the X stream and the page stores
  (`DecodeOverlayOperator.moe_a_depth`, `pg_split`).
- **Strided fills**: a pass that needs rows [r0, r0 + n) of every super-block of a tile reads them with one 4-D pattern
  (`(a // 8, 2 n, a // 8, n_sb * a // 4)`-style dims) instead of n_sb tasks.
- **Runtime-variable lengths cost nothing**: the firmware applies a scratchpad `length_parameter` at dispatch (256 fills
  per token showed no measurable cost).
- **Several cores into one S2MM channel**: in out-of-order mode (IRON `DmaChannel(out_of_order=True)`, where each
  sender's BD inserts a header naming the receive BD), 4 cores sent into one memtile S2MM at 7.98 B/cycle with exact
  results, freeing three channels. A task is a packet count (1-256; Start_BD_ID is ignored), so re-arm it every round
  (start queue word `(packets - 1) << 16`); the receive BDs' count locks are 6-bit.
- **Zero compression**: memtile MM2S compresses and S2MM decompresses, and core tiles support it too. It is enabled by
  channel control bit 4 and BD bit 31 (memtile word 4) in host-built BDs, since mlir-aie does not expose it. Format: per
  64-byte block, two 32-bit nonzero-byte masks (one per 32-byte half), followed by the nonzero bytes, padded to a
  multiple of 8 bytes. A decompressing memtile S2MM writes faster than its stream delivers: one shim channel delivered
  15.5 / 20.9 / 28.3 GB/s of payload at 50 / 75 / 90% zero bytes (11.4 GB/s uncompressed). Quantized weights have < 1%
  zero bytes (compression makes them 1.125x larger), so only zero-rich data gains.

## Task queues and transactions (host-built sequences)

`artlite.txn` and `overlay.Sequence`:
- **Addresses**: shim BDs carry final addresses (paddr + 0x80000000).
- **Queue limit**: at most 4 queued tasks per shim channel between syncs. A BD can be reused once its task has
  completed.
- **Rolling retirement** (`Sequence`):
  - before a 5th task on a channel, the transaction waits for that channel's oldest task;
  - 192 fills per column then cost the same as 3;
  - `sync()` waits for the drains, `sync(fills=True)` for everything;
  - a task carries a completion token only where a wait uses it (a channel completes its tasks in order, so one wait on
    its last task covers the others); the push is patched when the wait is issued (`Txn.set_token`).
- **Issue order decides deadlock.** A wait for the oldest weight fill needs the cores to consume it, and the cores need
  their X inputs first. Issue a phase's X fills and C drains before the bulk of its weight fills; prefetch the next
  command and the first fills before a sync.
  - A round over the columns holds at most 4 tasks of a channel per column: a fifth task waits for the first task of
    the same round, column after column, so the columns run one at a time (a GDN layer's last v head: 59 -> 142 us per
    layer).
  - With many tasks per column, issue them in rounds over the columns, with each round's drains before the next round's
    fills, so that the oldest task a fifth one waits for never depends on a later round (issuing all fills first hung:
    a fill waited for objects that the core could take only after its drains).
  - A phase that reads what an earlier phase drained must wait for those drains (`seq.sync()` before its input fills;
    issue its weight fills before the sync so they keep streaming): the channels' queues run independently, and a fill
    issued after a drain can read DDR before the drain has written it.
- **Broadcast streams and several consumers**: every consumer of a broadcast object must take it, in an order
  compatible with the other consumers. A core that skips a phase's objects must skip them in the order in which that
  phase's core consumes them; otherwise each waits for an object the other holds back (deadlock).
- **Coherence**: host writes must be clflushed before a dispatch (`OverlayDevice.flush`); NPU writes must be fetched
  before the host reads them.
- **Task cost**: a BD write takes ~0.08 µs, a queue push ~0.15 µs, and a token wait ~0.3 µs per column (timed in a
  host-built transaction); one column's round trip (fill -> the cores take it -> token -> the firmware's wait) takes
  2.3 µs. On the Gemma decode step, most waits were not the limit (a token and a wait per task vs. per channel and sync
  gave the same 57 ms): count tasks as well as bytes, but measure before cutting them. Where the cores take a stream's
  objects as fast as they arrive (the 27B's GDN state panels at S 1, DMA-bound), each of the stream's tasks adds to the
  step time: splitting its panel fills into 8 / 32 tasks per pair and column cost +4 / +24 ms per step (~0.5-0.65 µs per
  task, round-robin over the columns). The same holds for a rotation phase's sub-phases (one X object each): the 27B
  PARO's rot_down on 8 slots ran 17 of them per column, with the metadata, X and A objects as one task each; one
  repeated BD per column, with the metadata objects XPAY apart like the X objects, made a step of 8 2.5% faster
  (`decode.rotated._rotx_fills`). With many units per layer, issue their tasks in rounds of one task over the columns,
  not a column's tasks in a row (a chunkwise GDN core with 4 v heads: 3.24 -> 2.82 ms per step at every S; otherwise the
  last column's first task trails the first column's by ~10 µs at every unit).
- **Hold an output object only while writing it**: a core that acquires a D object early (for example, to build a GDN
  conv block's history in it across two X objects) runs its next outputs single-buffered until that drain is queued.
  Building the history in its workspace instead and draining it with the head's deltas after its panels took the 8-slot
  GDN core's S 4 step from 331.7 to 313.8 ms (the same change reduced gdn_in's 5 drains to 3).
- **The ops between a drain wait and the next phase's fills are on the critical path**: each one delays the next phase
  (a queue wait ~0.3 µs, a BD ~0.08 µs, a push ~0.15 µs, per column). `Sequence(inputs=("X",))` waits for the input
  fills before the drain wait (they are complete when the drains are), so the fills after it find empty queues: the
  Gemma step drops from 3751 to 1366 waits, -0.25 ms (qwen35: no change). Extra BD writes on the X fills add ~60% of
  their firmware time to the step time, whereas extra writes on the weight fills are fully hidden (they are issued far
  ahead). Fewer tasks per boundary (one broadcast fill instead of one per column) is a design-time lever.
- **An op's fixed cost** (in the vision overlay, a gate per column followed by the op's tasks): an empty op costs 4.6
  µs, a fill-only op of 16 KB per column 7.9 µs, and a band's conv ~13 µs above its B-stream time. For many small ops,
  merge the next op's input fills (and its output's allocation) into the current op where the memtile holds both,
  double-buffering across ops (sd15 `vae.prefetch`: the VAE decoder's convs and norms 282 -> 258 ms).
- **Watchdog**: a dispatch must finish within the driver's TDR (2 s), so size chunks for < 1 s. A killed job fails with
  `DispatchError ... TIMEOUT (state 8)`; a note names the operations (a library's item) at the firmware's `txn_op_idx`
  (where it stopped; 0xFFFFFFFF means the firmware reported none).

## Overlays, libraries and fused ELFs

- **A change of design inside a fused ELF is a full reconfiguration**: ~30-40 µs in a library (the tiles' writes go by
  control packets, `fused.ctrlpkt`; the PDI load's array reset adds 37 µs, but only on the first configuration of a
  piece: `fused.program.Quiet`), or ~80-90 µs with the reset. A single design whose workers run every phase removes it
  (the Llama decode step: 13.4 ms on one overlay vs. 25 ms as a runlist of designs). Design for it from the start:
  - one weight-object size;
  - one broadcast stream: inputs that every column needs arrive once per phase (one fill, one copy through the port),
    with per-column objects on another channel;
  - a fixed phase order, or command objects that tell the cores what to do (gemma).
- **Relocation addends are 32-bit**: a runtime-sequence argument must stay below 4 GB. The fused scratch is passed as
  segments (`fused.layout.SEGMENT_BYTES`); host-built transactions carry 64-bit addresses.
- **`FusedOperator(expand_load_pdis=False)`** for a single-design ELF: the ConfigureOp is skipped when the PDI is
  unchanged (-0.28 ms per dispatch). For multi-design ELFs, keep the default.
- **ConfigureOp cost** (in a library): ~0.1-0.13 µs per transaction operation (stream switch ports, core enables, the
  shim), plus a few µs of packets per column; the PDI load's reset costs 37 µs, but only on the first configuration of a
  piece. When program memory is loaded by control packets, size each design for its work, not for its configuration
  cost (Llama chunks of 64 / 128 tokens: GemmQ over 1-2-row GemmQd, the attention on 4 rows over 1, the norms on 4
  columns over 2: -10 to -12%); with blockwrites (~1.5 µs per KB), fewer cores would win on short prompts.
- **An ELF's control code must fit its device's 64 MB heap** (each `FusedOperator` opens a separate device): a
  multi-design runlist inlines ~0.25 MB of configuration per entry (qwen35 prefill: 310 entries = 79.6 MB, failing with
  `CREATE_BO ... Resource temporarily unavailable`). For long programs, use `fused.Library` (each design once) and
  `Library.run`, which assembles the program on the host and dispatches it in pieces. (A single aiecc ELF of thousands
  of entries is no alternative: aiecc's time grows superlinearly, to 14 min for 1216 entries.)
- **Count a layer-major program's design switches.** Every change of design is a ConfigureOp (above): the Gemma prefill
  switches design 12 times per layer (a 1152-token chunk has 279 configurations, which take 23 ms in total, 7%). The
  per-design PDIs of a library ELF are in its cache dir (`design.mlir.prj/op<N>_<Class>.pdi`).
- **Static buffers are rewritten on every reconfiguration** (`.bss` zero-fill and tables travel in the PDI): with
  blockwrites (~1.5 µs per KB; an overlay, a `FusedOperator`), 8 KB x 8 cores adds 80 µs; a library sends them in its
  control packets (`fused.ctrlpkt`: a GDN operator's SiLU table copies, 100 KB on 32 cores, ~150 -> ~5 µs). Use worker
  `Buffer`s without `initial_value` (it travels too: a zeroed 760 KB workspace was most of a 1.29 MB PDI).
- **Program memory is the tight resource** in overlays (16 KB for all phases): see SKILL.md section 7.
- **Library entries of one array design share one compiled device** (`fused.Library`: an overlay's operators differ
  only in their runtime sequences), so one design is compiled, not one per entry. Runtime-task BD ids are numbered per
  device, so pin the static design's memtile BD ids (`Bd(bd_id=)`): variants of a design must configure identically.
- **A ConfigureOp's single writes cost as much as its program memory.** aiecc's expansion writes, per core, every lock
  value (0 after the reset), the DMA channels' and the core's reset pulses, and each stream switch port individually:
  ~2500 operations for a 32-core design. `program._prune` drops what the PDI load's array reset already holds and writes
  a tile's ports as one blockwrite per range (Llama 3B chunk of 1024 tokens: 512K -> 222K configuration operations,
  362 -> 341 ms, bit-identical). `fused.ctrlpkt` sends each core's program and data memory, and each core's and
  memtile's BDs, locks and DMA channels (including start queues, in order), as control packets. Per column, the shim
  DMA streams them up a route set up after the reset (a packet is split at each tile: its id goes to TileControl, the
  rest North), a marker word is polled, and the route's ports are then overwritten with the design's ports (a 24-core
  GEMM: 854 -> 306 operations; Llama 1B chunk of 128 tokens: 43 -> 31 ms; Gemma / Qwen chat TTFT: -20%). A slave port
  has 4 slots: match the ids above by a differing bit rather than one slot each. Firmware PDI loads
  (`expand_load_pdis=False`) are slower: +1.1 ms per switch.
- **A design that ends leaves its lock waits pending.** A core or DMA channel blocked on a lock when its design ends
  keeps its acquire pending in the lock's module: neither a core reset (Core_Control) nor a DMA channel reset cancels
  it, and the next design's first release of that lock goes to it. That object never reaches its consumer, which causes
  a hang that comes late and depends on timing when the wait was on a local lock; writing lock value 63 grants the
  pending acquire. Only the PDI load's array reset clears these waits, so a configuration without it
  (`fused.program.Quiet`) quiesces each tile through its packets (core and channels held in reset while every lock is
  written to 63, then 0). Before any packet, it also resets the previous design's cores that wait on a neighbour's lock
  and flushes those locks, because a neighbour's packets can arrive first (`fused.locks` derives those acquires from the
  cores' objects). The registers, stream switch counters and data memory showed nothing; per-port word counters (event
  port selection + counting `PORT_RUNNING`) and the lock values read during the hang did.
- **Which PDI load the firmware skips depends on the firmware's state, not the host's.** It skips a load of the PDI it
  loaded last (and the reset with it). A fresh context has loaded neither of a library's two PDIs, a TDR recovery keeps
  the last one, and the host's count of loads drifts when a piece is assembled but not dispatched, or when a hung piece
  stops anywhere. A configuration pruned for a reset array then runs on a live one: a hang. A dispatch whose first
  configuration needs the reset therefore loads the other PDI first (`fused.program`; it costs 0.25 µs when that PDI is
  the last one loaded and the firmware skips it).
- **Control code is the fixed cost of a small operator**: ~0.1 us per TXN operation (measured with injected write32s),
  and an overlay operator issues 300-900 of them (memtile BDs and their address fields, shim BDs, address patches, queue
  pushes, task waits): 35-65 us per operator before any data moves. The library assembler leaves out memtile BD writes
  that the registers already hold (except BDs with an iteration, since the DMA advances word 6's current step): -24%. A
  single-piece program that runs again reuses its assembled control code.
- **Two cores on one object through shared memory**: post the other core's command and release it before computing your
  own half, or the halves run one after the other (a GEMV phase: 29 -> 47 GB/s).
- **Separate hardware contexts** (e.g. decode + prefill designs) cost ~0.8-2.5 ms per switch. Share device buffers
  instead of copying: `FusedOperator(external_buffers=...)`, or one weight image read by both designs (gemma).
- **Phase boundaries** cost ~5% in a DMA-bound overlay, because the cores wait for the next broadcast (in the Gemma
  decode, ~11 us pass from a phase's last tile to the next phase's input, ~8 us of them the sync's round trip, followed
  by ~5-10 us of input strips, with 295 boundaries per step). Queue the weights that follow a sync before the sync (they
  keep streaming). Route the weight stream through a memtile FIFO so that DDR keeps streaming across the boundary.
- **Account for all time before blaming boundaries** (measuring.md, "The firmware's timeline"): the 35B-A3B's step lost
  4.4 ms of 36 ms vs. its bytes at the port rate, and its 253 boundaries were not the cause (the FIFO's prefetch hides a
  boundary of ~13 us). 2.0 ms was spent in the expert phases (core-bound row-group tiles, three DDR round trips at a
  layer's end) and 1.6 ms in the GDN cores' steps, under which no weights stream. A model core's step is the time to
  shorten or to stream weights under.
- **Outputs that nobody reads still cost tasks and port time**: a chunkwise GDN core drained 16 state panels per v head
  in every step, although only a step at the end of an 8-position block writes them (63 MB and 960 tasks per step: 0.6
  ms). Tally a step's bytes per stream and buffer from its transaction, and verify that each buffer actually has a
  consumer.

## Host side of a dispatch

- **Waiting**: a blocking driver wait adds 0.3-1.7 ms and leaves the CPU core cold. Nap in 0.2 ms slices until just
  before the expected end, then spin (`FusedOperator.__call__(spin_after_s=..., nap_s=0.0002)`). Keep the thread on that
  core between dispatches too: a thread released after each wait (with its affinity restored) ran the host work between
  a library program's pieces ~2.5x slower, on whatever core the scheduler woke for it (`Library.run` pins its thread to
  the interrupts' CPU with `portable.on_cpu`).
- **Overlap a step's position patch with the preceding step**: the host's patch of a decode step's transaction (the
  attention layers' words, angles and cache transfers: 0.37 ms of the 35B-A3B's 36 ms) ran between the steps while the
  NPU sat idle. `ModelOverlay.AHEAD` patches the next step in a second transaction buffer while a step runs (whatever
  the running step reads from shared buffers needs a copy per buffer; whatever the model writes only before the step is
  predicted and checked: `_ahead_state`). The second buffer's transactions are copies of the first's (1 ms), never a
  second build (~100 ms under a running step).
- **One-time host work must not reach a planner's times**: a transaction's first run (its build and its patch tables),
  timed as part of a drafter's round, tricked the draft planner into disabling a profitable drafter (`models.drafts`).
  Build the tables with the transaction, subtract `overlay.once_s` from every time that feeds a running average, and
  compare trees by the drafts taken as well as by tokens per second.
- **Host work between pieces is on the critical path** when the piece it overlaps is short (an MoE layer's routing under
  its shared expert's ~1.5 ms). Time it in the real run (a pieces timeline: dispatch, hook, wait), not alone (alone it
  ran 2-3x faster). Split it so that each piece's inputs come first and the rest overlaps the next piece (two hooks),
  and keep assembled pieces (`Library.PIECES_KEPT`, in bytes: a cache that was too small re-assembled a 1 MB piece,
  ~1.3 ms, every time).
- **Keep the next dispatch queued**: submit piece k, then wait for k - 1 (`dispatch_wait(keep=1)`), so the firmware
  starts k as soon as k - 1 ends. Waiting first left the NPU idle for ~35 us per boundary (the wait's wake-up and the
  submit) on top of the next piece's ConfigureOp; queuing ahead saves 0.3% on the qwen35 35B-A3B's 4096-token chunk
  (124 pieces).
- **The first dispatch after new host buffers** takes ~0.15 s longer (the driver maps them): a model's load ends with
  an empty dispatch (the vision overlay's `VisionDevice.warm`), or its first step pays the cost.
- **Overlap host work**: prepare the next chunk's inputs on a worker thread while the NPU runs (gemma
  `NpuGemma.prompt`), and compute whatever depends only on a step's index (SD's per-step time biases, 2.6 ms of GEMVs)
  for all steps before the first one (`NpuUNet.prepare`, started before the text encoder). Prepare a decode step's
  position patches (cache ranges, tails, token objects: ~0.15 ms of numpy) for the next step while the current one runs,
  on a second transaction buffer (`ModelOverlay.AHEAD`, `OverlayDevice.submit` / `wait`; a patched step runs only at
  its position and ring state), and compute the embedding row in C++ (`_native.dequant_rows`): Llama 1B's step 14.92 ->
  14.75 ms.
- **Host ops between dispatches belong in C++, not torch**: torch's OpenMP threads spin for milliseconds after every
  parallel op (a vocabulary's 128K-262K logits count as one), into the next dispatch. As torch ops, sampling's top-k,
  argmax and a draft's probability cost the MTP loops 160-400 us per call and 71-98 ms of host CPU per token (Gemma E4B,
  Qwen3.6 35B-A3B). As one AVX-512 pass over the bf16 logits (`_native.top_logits`, `max_prob`: a threshold at the k-th
  largest key so far, which almost no logit passes after the first few thousand), they take 10-40 us and 6 ms of CPU per
  token (Qwen), and the steps get 1-2.5% shorter.
- **The first chunk's late inputs**: start the program with what its first operators read, and write the rest before
  the piece that first reads it (`Library.run(hooks={item: fn})`: a hook runs before that piece is dispatched, while
  the previous one runs; the item starts a piece). The Gemma prefill's per-layer token rows (a lookup) overlap the first
  chunk's first operators (`GemmaPrefill.run` with a `Future`).
- **A prompt's last token**: a prefill that computes only the layers that own a cache still needs the last token's
  logits. Finish that token from the prefill's last layer with a decode step that starts there
  (`GemmaOverlay.step(start=)`, with x taken from the chunk's residual) instead of a whole decode step: it reads 19 of 42
  layers' weights (-25 ms per prompt). If the overlay's image expects a header (post-norm update) at that layer, keep
  the header and make the update zero.
- **Patch a step by cache lines**: rewriting a few words per step and then flushing their buffer's whole span costs
  milliseconds of clflush on the critical path: the 27B's metadata objects took 6.2-6.7 ms at every change of S (nearly
  every round of a generation, yet invisible in a step table of a single S), and a 2.2 MB transaction 1.2 ms per step.
  Write only what differs from the last S, patch through index arrays, and flush the touched lines with one fence
  (`OverlayDevice.flush_lines`, `Transaction.flush_lines`).
- **Read only what the host needs**: top-k from block maxima, not 128K logits. numpy beats torch for 2-3K-element work.
