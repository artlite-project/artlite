"""The attention of a prefill on the decode's KV record format (``decode.kv``), one column per KV head (Llama, Qwen3): a
T-token chunk at position p0 of one layer, from the qkv GEMMs' output G (``prefill.gemm``: per column c [q of its qh
heads | k | v] of KV head c) to the O GEMM's A panels, with the chunk's K / V records in the decode's cache, so the
decode continues from them. A KV head's cache is groups of kp + 1 objects of the image's tile size (kp K objects of
records of 8 positions, kp = 2: hi + lo panels, 1: single; the V object), linear in the positions; a chunk starts at a
group (``Shape.pg`` positions: 128 at head_dim 64, 64 at 128). RoPE on the pairs (i, i + hd / 2) (the image's Q / K rows
in rotate-half order), scores in log2 units: q k log2(e) / sqrt(hd).

``PrepOperator``: per column c, cores (c, 2) and (c, 3) the Q heads (the first half of the column's heads, the rest),
(c, 4) K, (c, 5) V. Per 8 tokens a Q core turns each of its heads' rows into the attention's operands: q rotated (fp32)
times the score constant, as single bfp16 panels of 8 queries x 8 dims (ND panels: a Q unit; ``Shape.qp`` 2: hi then
lo panels, 2 ND); QS holds per token group and column the qh heads' units. Qwen3's q and k are RMS-normed per head and
times their weights per dim first (``norm``: the layer's weights the cores' first object; its weights reach ~100, so a
row's dims span ~1e5 within a block: Q hi + lo, KL 0.0005-0.0023 of Qwen3 0.6B less on the CPU model with the
attention's roundings). The K core writes K rotated (fp32) as the cache's K
records (kp per 8 dims: hi + lo, or single panels), the V core V (bf16) as V records (per 8 dims a panel of blocks over
the 8 positions); each fills a group's K objects (V object) in L1 and the memtile joins them into the group's objects
(one transfer per group: a column's shim has two output channels). cos / sin from a table of every position
(``cs_table``); the chunk's rows and cache groups by scratchpad parameters (``PrepOperator.params``).

``AttnOperator``: per column c 4 cores, passes of 32 QB queries; core r takes the QB query blocks of 8 from block QB r
of the pass for each of the column's heads (QB qh row blocks), streams the KV head's cache groups from position 0 up to
the pass's and writes their outputs as the O GEMM's A panels. Two streams per column, each through its memtile and
broadcast to the column's 4 cores (a core's two input channels): A the K records (a single-K cache: its K objects,
each unit's two records interleaved per d-block; hi + lo: each group's [K | K] objects, the memtile skipping the first
one's padding), B the pass's Q units ahead of the V objects, both in objects of two units read per d half (a core keeps
its own Q units). A unit is 144 ND bytes: a hi + lo K record (two single ones), two V records, two row blocks' Q units
(hi + lo Q: one row block's). Per chunk of CR records (QB, CR: the most that L1 holds, ``Shape.blocks``):

- the scores on the MMUL, the accumulators starting at the rows' bias: per pair of row blocks one loop of two panel
  streams (Q | K0, K1: four chains, MMUL-bound; hi + lo K: a loop for K hi, then one for K lo; hi + lo Q: each Q panel
  pair's products into the same accumulators, Q lo against K hi only). A single-K chunk is one call on both its K units
  (the pairs' and units' loop pipelined as an outer loop: an iteration's setup under the previous one's MMULs and
  stores); each tile's epilogue writes the softmax's terms v = max(bf16(A), 129), f = bf16(A - v) (bf16, ``vf``: the
  exp's range reduction where the vector slot is idle) and a flag for the softmax when some v reaches 208;
- the softmax (``soft``, from the chunk's first ``pv``) with a running maximum per row, raised (o, l rescaled) only when
  the chunk's maximum exceeds it by 2^16 (qwen35's ``attn_prefill``: m is only the reference of 2^(s - m)); masks only
  on a diagonal chunk; P = 2^(s - m) times 2^64 (o / l cancels it) as 2^(v - 128) 2^f (km.exp2's steps), clamped at
  2^-63 of that (masked lanes v 129, f -1/2), as single bfp16 panels (half the P V MMULs of hi + lo), or hi + lo
  (``Shape.pp`` 2: the output then within its own bfp16 rounding, against 1.13-1.39x of it; LFM3's and the readout
  models'). Known error, cause open: the keys below a row's reference weigh ~0.4% too much
  (test_llama_npu_teacher_forced[Q3_K_L-1B-hilo] fails);
- o += P V per pair of row blocks (each V panel popped once for both: two MMUL chains; both panel streams on load slot B,
  o's accumulator loads on A: MMUL-bound), and the row sums l += P J on the MMUL (J a ones panel: no accumulator in the
  exp loop). Where L1 holds them o is in two copies a region apart, P V reading one and writing the other (in place,
  its loads and stores meet in one region's banks: ~60 cycles a call); ``l1_layout`` puts every hot loop's streams in
  distinct regions;

chunks past every one of the core's queries are skipped (their objects consumed). At the pass's end o / l per head as
the O GEMM's A panels. QB = 2 (64-query passes: half the cache re-reads) where L1 holds it (1B, 3B), else 1 (8B).
For 2048 tokens at p0 = 8K per layer: Llama's and Qwen3's shapes are cache-bound (3B 13.2 ms, 12.5 with the compute
compiled out); LFM3's broadcast design (``Shape.per`` 2) binds on its cores: 9.6 ms with hi + lo P (~450 cycles per
record and core: scores 124, softmax 110, P V 180, the rest calls and locks), 8.2 with single P, 6.2 with the compute
compiled out.

The package: ``shape`` (``Shape``, the scratchpad parameters, the cos / sin tables), ``prep`` (``PrepOperator``),
``attention`` (``AttnOperator``) with its variants as one validated ``mode.AttnMode``, ``layout`` (the cores' L1) and
``ring`` (the ring variant's memory tiles).
"""

from .attention import AttnOperator
from .layout import STACK, core_bytes, l1_layout, workspace_regions
from .mode import AttnMode, Kind
from .prep import PrepOperator
from .ring import MT_BYTES, mt_bytes, ring_groups
from .shape import P_CS, P_FG, P_FH, P_FO, P_FQ, P_G0, P_HK, P_KV, P_P8, P_SW, Shape, cs_rows, cs_table

__all__ = [
    "MT_BYTES", "P_CS", "P_FG", "P_FH", "P_FO", "P_FQ", "P_G0", "P_HK", "P_KV", "P_P8", "P_SW", "STACK", "AttnMode",
    "AttnOperator", "Kind", "PrepOperator", "Shape", "core_bytes", "cs_rows", "cs_table", "l1_layout", "mt_bytes",
    "ring_groups", "workspace_regions",
]
