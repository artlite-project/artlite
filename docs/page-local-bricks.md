# Page-local bricks: memory lines and pages on other machines

Artlite is measured on Strix Halo under Linux. Reports from a Krackan Point laptop on Linux and from Krackan Point and
Strix Point laptops on Windows ([issue #5](https://github.com/artlite-project/artlite/issues/5)) showed two costs that
Strix Halo hides. Both concern the rows the NPU's DMA reads and writes in DRAM:

- **Writes of part of a 64-byte line.** On Krackan Point such a write costs 4.7x a whole line's. Artlite's prefill
  writes whole lines.
- **Rows on pages of their own.** The NPU translates the address of every memory page a DMA row touches. With 2 MB
  pages (Linux) that is nearly free; with 4 KB pages, the only size Windows gives the NPU, the translations cap the
  NPU at about 50 million page visits a second. The prefill's many short rows pay it: this is most of the prefill's
  gap on Windows.

The second cost needs a change of the prefill's data layouts: **page-local bricks**, a proposal described in the last
sections. It is not implemented yet.

## Rows, lines and pages

The NPU's shim DMA moves data between DRAM and the AI Engine array as rows: a buffer descriptor (BD) describes up to
three dimensions of sizes and strides, and each innermost run of contiguous bytes is a row. A prefill GEMM writes its
output as one row per token per pass (128 bytes for 64 bf16 columns), at a stride of the output's whole row (several
KB). Two properties of a row matter:

- **Lines.** DRAM is written in 64-byte lines. A row that starts or ends inside a line writes part of it, and the memory
  then reads the line and merges the new bytes in.
- **Pages.** The NPU addresses memory by virtual addresses, translated page by page (Linux: through the IOMMU and the
  process's page tables; Windows: through the video memory manager's GPU page tables). A row that starts on another
  page than the channel's previous row needs a translation, a *page visit*.

`artlite-probe` measures both ([probe](probe.md)). Rows below are 16 shim channels at once, each row next to a row of a
later group, the rows 64 KB + one row apart:

| GB/s | Strix Halo | Strix Halo, 4 KB pages | Krackan Point, Linux | Krackan Point, Windows | Strix Point, Windows |
|---|---:|---:|---:|---:|---:|
| gather 256 B rows | 56.9 | 29.3 | 89.6 | 15.6 | 14.0 |
| write 128 B rows (whole lines) | 57.2 | 12.2 | 73.2 | 6.0 | ? |
| write 132 B rows (parts of their end lines) | 41.4 | 12.1 | 15.6 | 5.4 | ? |
| read stream | 56.8 | 56.8 | 96.9 | 81.0 | 53.2 |
| write stream | 57.6 | 57.2 | 85.3 | 53.1 | 39.9 |

Strix Halo with 4 KB pages: Linux with transparent huge pages off (`echo never >
/sys/kernel/mm/transparent_hugepage/enabled`). Krackan Point on Windows: power mode `performance` (the top clock, as
`default` on Linux); Strix Point on Windows: `default`. All at 1.81 GHz.

## Parts of lines: whole-line writes

On Krackan Point, which has no cache in front of DRAM, 132-byte rows write at 15.6 GB/s against 73.2 for whole-line
128-byte rows (4.7x); Strix Halo's cache merges them (41.4 against 57.2). Writing each GEMM pass's own columns where
they fall, a 1024-token prefill chunk writes parts of 4.4 million lines for Llama 3.2 1B Q4_K_M, 10.8 M for Llama 3.1
8B, 83 M for Qwen3.8 27B and 1.1 M for Gemma 4 E4B (counted from the dispatched control code's BDs: every row's start
and end). Their sources and what Artlite does:

- **Planar GEMM passes** (Q6_K tiles of 11 rows: passes of 66 columns, pass p at column 66 p). Each pass writes a window
  of whole lines from the line boundary at or before its first column: the previous pass's last columns first (the
  core keeps them in its output buffer and moves its own columns behind them), its own columns, then columns that the
  next pass rewrites. The output layouts leave room for the last window.
- **gate/up's output** (bfp16 panels, 32 outputs per pass: 288-byte rows, half of them ending inside a line). Two
  passes, or a pass's two halves, share one output buffer, drained as 576-byte rows: whole lines.
- **The output projections' column pitch** is a whole number of lines.

The prefill then writes no partial line for the 1B, the 8B and Gemma, and 8.2 M for the 27B (its gdn_in projection,
whose windows do not fit the cores' memory), for 3-9% more bytes written. Strix Halo's prefill speeds are unchanged
(1B, 8B, Gemma E4B, Qwen3.8 27B, Qwen3.6 35B-A3B, measured alternately with and without).

## Pages: 2 MB on Linux, 4 KB on Windows

On Linux, Artlite's host buffers of 2 MB or more are the process's own memory with 2 MB pages (transparent huge pages,
`madvise` or `always`; `artlite-probe` prints `host buffers: 2 MB pages`), weights and activations alike. Windows maps
system memory for the NPU in 4 KB pages: its video memory manager manages system memory in 4 KB pages only
([GPU virtual address](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address)), and
large pages for an application need a privilege (`SeLockMemoryPrivilege`) and would not change those page tables. The
probe rows on both Windows laptops behave as Strix Halo's with 4 KB pages, at about half its rate.

What a page visit costs, measured on Strix Halo with 4 KB and with 2 MB pages (16 channels unless noted; 256-byte rows,
one per page, 4 KB + 256 B apart, so that neighbouring rows' page-table entries are neighbours too; millions of rows
per second):

| pattern | 2 MB pages | 4 KB pages |
|---|---:|---:|
| read: 1 / 2 / 4 / 8 / 16 channels (256 pages each) | 50 / 98 / 187 / 212 / 217 | 50 / 98 / 166 / 148 / 170 |
| read: 4 / 16 / 64 / 128 / 256 / 512 pages per channel | 213 / 217 / 217 / 217 / 217 / 217 | 213 / 217 / 217 / 173 / 171 / 169 |
| write: 1 / 2 / 4 / 16 channels (256 pages each) | 51 / 103 / 205 / 220 | 51 / 103 / 141 / 137 |
| write: 4 / 16 / 64 / 256 pages per channel | 215 / 220 / 220 / 220 | 215 / 220 / 152 / 137 |
| 128 B rows, a page each: read / write | 437 / 436 | 178 / 146 |
| 512 B, 1 KB, 2 KB rows, a page each | as many bytes as a stream | the same |
| 128 B rows 512 B apart (8 rows a page): read / write | 437 / 436 | 439 / 435 |

- **One limit for the whole NPU**, not per channel: one or two channels never reach it; from four on, all channels
  together get ~140-170 million page visits a second on Strix Halo, about 50 million on the Windows laptops.
- **A translation cache.** While all channels cycle through no more than ~1000 pages (reads) or ~250 (writes), page
  visits cost nothing.
- **Page-table locality.** Rows 64 KB apart (each row's page-table entry on another 64-byte line of the page table)
  cost more than rows 4 KB apart: 95 against 146 million 128-byte write rows a second.
- **Rows that share a page are free**, however short.

A 1024-token prefill chunk of Llama 3.2 1B Q4_K_M visits 7.5 million pages:

| source | rows | page visits |
|---|---|---:|
| GEMM outputs | 128-192 bytes per token and pass, a whole output row apart | 3.1 M |
| weights | 2 KB slices of the weight tiles, a tile (9 KB) apart | 1.9 M |
| norms and attention prep reading activations | 128-512 bytes per token, an activation row apart | 1.1 M |
| GEMM inputs (bfp16 panels) | 576 bytes per 8 tokens and 64 columns, a panel row apart | 0.6 M |
| gate/up outputs, norm and attention outputs | 128-576 bytes | 0.7 M |

At Windows' ~50 million a second that is ~150 ms of a chunk whose work takes ~105 ms on Strix Halo: most of the
prefill's ~2x gap on Windows. As streams, the chunk's 4.6 GB would need ~1.1 million page visits.

Decode reads each column's weights as one stream and pays little (Krackan Point on Windows reads 81 GB/s as a stream
against 97 on Linux).

## Page-local bricks

The fix is not long rows everywhere but rows that share pages: a layout in which the rows one DMA task issues fall into
few 4 KB pages. A **brick** is one 4 KB-aligned block that holds a small group of tokens across a range of features,
row-major inside. For bf16 values, G tokens x F features with G x F x 2 bytes = 4096: 8 x 256, 16 x 128 or 32 x 64. A
T x N activation is laid out as [T / G][N / F][G][F]: the parts of consecutive tokens in one feature range share a page.

What the prefill's accesses would cost:

| access | today | with bricks |
|---|---|---|
| a GEMM pass writes 64 columns of m tokens | m page visits (one per token) | m / G (G rows per brick; with 32 x 64 bricks the pass writes whole bricks) |
| a norm reads 8 tokens x 256 columns | 8 rows, 8 visits | one brick (8 x 256), contiguous 4 KB |
| the attention prep reads a head's rows | one visit per token | G tokens' rows per brick |

The GEMM inputs, bfp16 panels of 8 tokens x 8 columns stored [T / 8][K / 8][72 bytes], would get the same treatment
along K: [T / Tb][K / 64][panels of the block's tokens], so that the panels a GEMM core row needs for one 64-column
k-tile of a block of Tb tokens are contiguous (4.6 KB for Tb = 64). Stored in the order the multiply loop consumes them,
the memory tile's reordering of the GEMM's input then needs three dimensions instead of four.

The weights' 2 KB slices are the decode's tile layout. Interleaving the slices of the tiles a GEMM pass reads together
would make each of its reads one contiguous run, but it changes the decode's reads too: it belongs to a redesign of the
weight formats, not of the activation layouts.

Target for the 1B: from 7.5 to at most ~1.5 million page visits per 1024-token chunk, Strix Halo's speed unchanged or
better (writing a short-K GEMM's output in long rows is already faster there: ~25% on qwen35 1B's gdn_in).

### Constraints

- **Whole lines stay whole.** Bricks of whole 64-byte lines keep the prefill's whole-line writes; a planar pass's
  window that crosses a brick's edge needs two runs.
- **A column's DMA queue holds 4 tasks, a shim tile 16 BDs.** Splitting a pass's writes into more tasks than that hangs
  the NPU or fails to compile: one BD, with its iteration dimension, per pass where possible.
- **The decode's buffers keep their layouts**: the KV caches the prefill writes, the recurrent states, the weights.
- **The host reads activations** (the last token's hidden state, tests' references): one helper for the brick layout
  instead of reshapes in every app.
- **The cores' memory is full** in the planar GEMMs: the layouts must not grow core buffers.

### Plan

1. **Measure Windows' side**: `artlite-probe` rows with small and large page working sets (the table above, on
   Windows), to see whether its translation cache is as large as Strix Halo's. If it is, bricks are enough; if not,
   only long contiguous rows are.
2. **Design**: G, F and Tb per buffer, and for every prefill operator (GEMMs, norms, attention preps and attentions,
   the recurrent layers' operators, the mixture-of-experts gathers) its DMA patterns in the new layout, with BD counts
   and page visits per chunk.
3. **GEMM inputs** in the (block, k-tile) order, the GEMMs' input fills and their producers.
4. **Outputs and the residual stream** in bricks, the norms and attention preps reading them.
5. **Weights**, with a redesign of the weight formats, measured on decode as well.

Each step is measured on Strix Halo (prefill speed of every model, the page-visit count, the tests) before it lands;
on Windows by a report in [issue #5](https://github.com/artlite-project/artlite/issues/5).

## Measuring on your machine

- `artlite-probe` prints the page size of Artlite's buffers and the rows above. On Linux, `host buffers: 4 KB pages`
  means transparent huge pages are off (see [Installation](install.md)).
- On Linux, `echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled` gives Artlite 4 KB pages (restore
  `madvise` afterwards): a stand-in for Windows, whose page visits cost about twice Strix Halo's.
