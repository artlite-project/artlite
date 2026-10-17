"""Stable Diffusion 1.5's UNet level 64x64 on one static array design (an overlay; the 32x32 .. 8x8 levels:
``sd15_v2``): its operators share its configuration, so a library program of them never reconfigures; the operators
differ in their runtime sequences and in commands their cores read.

Tensors are arrays of objects [pass of 80 channels][tile of 128 tokens][C_OBJ]: a tile is R = 8 / L rows x L groups x
8 lanes x 2 pairs (the CFG pair: token (tg = (rr, g), pair, lane) is pixel (pair, t R + rr, g + L lane)); an object
holds a bf16 C tile [tg][pair][10][64] or bfp16 panels [tg][10 q][pair] (11520 B). 64x64 ops come in pairs (the top
and bottom 32 tiles: ``TiledOp`` t0 / TT).

Static design (per column c; cores k = 0..3 at rows 2..5):
- memtile c: a heap (HEAP bytes): ops place their data in [0, SLOT0); one output slot per core at SLOT0 + k C_OBJ.
- A: memtile MM2S k -> core k S2MM 0 (2 x A_OBJ ping-pong); B: B streams from the shims to the cores' S2MM 1 (a ring
  of NQ x Q): shim 2r -> the 8 cores of row r; C: core k MM2S 0 -> memtile S2MM k (static chain into slot k; its
  memtile BD ids pinned); DDR in: shim c MM2S 0 -> memtile S2MM 4 (op loads); DDR out: memtile MM2S 4 (static chain:
  slots 0..3 in turn) -> shim c S2MM 0.
- Core program: B command -> ws; pre-op B slots (GroupNorm / LayerNorm records, tables); the extension's phases;
  per tile: B init (3 slots), NR residual A objects, NK3 conv k-tiles (an A object: the halo-free band rows
  [R][S L][4 cb][2] bfp16, expanded in L1; 3 B slots), NKG GEMM k-tiles (1 B slot), B padding, the epilogue, C full.

Variants: the regular variant (conv, GroupNorm statistics and coefficients) and the transformer variant
(``conv=False``: LayerNorm, GEGLU and an extension's kernels, e.g. attention) have the same static design
and the same core main, and differ in a tail of program memory only: a library program switches between them by
writing the differing words (``fused.program.program_memory_patch``), not by reconfiguring. Their shared kernels must
be byte-identical: the variant entries go through trampolines (weak: no caller sees through them), every trampoline is
emitted before the impls (functions come out in the order of their first reference; an ordering table references the
ones only the core main calls), the stubs keep an opaque side effect (or their callers would drop the calls), and the
shared helpers are plain external functions (deferred templates and statics would land after the impls). In a
library's translation unit the variants link one copy of the code they share (``kernel_source``), so every definition
the variants share comes before the first that differs (``Overlay.source``): both link it at the same addresses.

Ops (``TiledOp``): a weight image of B objects (per stream: command, init and k-tile objects) and a runtime sequence:
heap loads (DDR -> memtile), sends (heap -> core k: BD taps), weight fills (shim -> B stream), and per column one shim
drain taking its cores' C objects in the static order.
"""

from collections.abc import Callable

import numpy as np
import torch
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Acquire,
    Bd,
    BdIteration,
    Buffer,
    DmaChannel,
    Flow,
    Kernel,
    Lock,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.controlflow import forever_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.operators import act_lut
from artlite.operators import bfp16 as _bfp16
from artlite.operators.base import ArgDirection, ArgSpec, ArtcOperator
from artlite.operators.design import COLS, ROWS, range32
from artlite.operators.kernel_source import kernel_prefix


def bfp16_blocks(x: np.ndarray) -> np.ndarray:
    """float (..., 8) -> bytes (..., 9): [E, m0..m7], rounded to nearest even."""
    e, m = _bfp16.blocks(x)
    return np.concatenate([e[..., None], m.view(np.uint8)], axis=-1)


PANEL, CBC, Q, NQ, C_OBJ, WS = 72, 4, 2880, 6, 20480, 64
A_OBJ = 8 * CBC * 2 * PANEL  # 4608: a conv k-tile's band rows (R L = 8 groups), a GEMM k-tile (8 tg x 4 kb x 2)
HEAP = 510 * 1024
SLOT0 = HEAP - ROWS * C_OBJ  # the output slots; ops use [0, SLOT0)
# ws words: the command's (copied by begin), then the core's state
(W_NT, W_NR, W_NK3, W_ROFF, W_NKG, W_NBP, W_NE, W_L, W_J, W_R, W_S, W_IM, W_NPRE, W_NFIN, W_NPASS, W_G, W_NSTS, W_INVN,
 W_EPS, W_ATT, W_NEMIT, W_RB, W_PM, W_G2, W_PS0, W_RI, W_APAR, W_BPOS, W_PK, W_PASS, W_FQ, W_CID, W_TILE) = range(33)
CMD_TILE = 64  # command words CMD_TILE + core id (c ROWS + r): the core's tile (LayerNorm records)
LREC = 1440  # a LayerNorm record in its B slot (2 per slot): [tile, ..][S1 8 | S2 8 per row (2 tg + pair) 16] fp32
# epilogue modes (W_NE)
E_BFP, E_GNST, E_GN, E_GNSILU, E_LNST, E_LN, E_GEGLU, E_BFPT = 1, 2, 3, 4, 5, 6, 7, 8
BAND = 8192  # the band buffer: conv band rows with halos (<= 6912 B), GN apply scratch + the SiLU table, records
REC = 576  # a GN record ([pair 2][S1 32 | S2 32] fp32 = 512 B) in its slot
NREC = 12  # records a core keeps (passes per op) in band, emitted as one object
ROFF = 128  # (bf16) the apply op's input in the C buffer: its panels are written in place from 0
SOBJ = 5 * A_OBJ  # a staged C object (+ the bytes that fill its last A object)
C_BUF = C_OBJ + 2 * ROFF
SCR = 3072  # per-core scratch: GN statistics / coefficients

# The variant region, last in .text (the module docstring): the regular variant links the conv kernel, the GroupNorm
# statistics and stubs for the transformer variant's entries; the transformer variant a conv stub (still referencing the
# mask table: equal .data), LayerNorm, GEGLU and the extension's impls.
_SRC_CONV = r"""
// conv k-tile: A object = halo-free band rows [R][S L][CBC][2] bfp16 (expanded RB rows at a time), 3 B slots
// [J][3 dx][CBC][2]
__attribute__((noinline)) void {{ p }}_conv3_impl(const int32_t *__restrict a0, const int32_t *__restrict a1,
                                                  int32_t *__restrict band, const int32_t *__restrict ring,
                                                  int32_t *__restrict ws, int32_t *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int L = ws[{{ W_L }}], J = ws[{{ W_J }}], R = ws[{{ W_R }}], S = 1 + ws[{{ W_S }}];
  const uint8_t *a = (const uint8_t *){{ p }}_a(a0, a1, ws);
  const int32_t *b = {{ p }}_b(ring, ws, 3);
  bfloat16 *cc = (bfloat16 *)c;
  const int Li = S * L, RB = ws[{{ W_RB }}];  // (band rows of the input level's Li groups, RB at a time; L: output groups)
  for (int r0 = 0; r0 < R; r0 += RB) {
    {{ p }}_expand(a + r0 * Li * 576, (uint8_t *)band, Li, RB);
    {{ p }}_k1<3>((const bfp16ebs8 *)band, (const bfp16ebs8 *)b, (const bfloat16 __aie_dm_resource_c *)(cc + r0 * L * J * 256),
                  (bfloat16 __aie_dm_resource_c *)(cc + r0 * L * J * 256), L, J, RB * L * J, S, 2);
  }
}
"""

_SRC_CONV_STUB = r"""
__attribute__((noinline)) void {{ p }}_conv3_impl(const int32_t *__restrict, const int32_t *__restrict,
                                                  int32_t *__restrict, const int32_t *__restrict, int32_t *__restrict ws,
                                                  int32_t *__restrict) {
  ws[{{ W_RI }}] = *(volatile const uint8_t *){{ p }}_hm[0];  // (unreachable: keeps the table linked)
}
"""

_SRC_R = r"""
namespace {
// a pass's channel sums rec [pair][S1 80 | S2 80] -> gacc [pair][S1 32 | S2 32] by group (channel c: group c / G)
static inline __attribute__((noinline, minsize)) void {{ p }}_group(const float *__restrict rec, int pass, int G,
                                                           float *__restrict gacc) {
  const aie::vector<float, 16> z16 = aie::zeros<float, 16>();
  for (int cb = 0; cb < 10; cb++) {
    const unsigned c0 = 80 * pass + 8 * cb;
    const int ga = c0 / (unsigned)G, gb = (c0 + 7) / (unsigned)G;
    const aie::mask<16> m = {{ p }}_lo((ga + 1) * G - c0);
    for (int k = 0; k < 4; k++) {  // (pair, statistic)
      const aie::vector<float, 8> v = aie::load_v<8>(rec + 80 * k + 8 * cb);
      const aie::vector<float, 16> vv = aie::concat(v, v);
      const float sa = aie::reduce_add(aie::select(z16, vv, m).extract<8>(0));
      const float sb = aie::reduce_add(aie::select(vv, z16, m).extract<8>(0));
      float *g = gacc + 32 * k;
      for (int h = 0; h < 32; h += 16) {  // (runtime group lanes: one-hot masks)
        aie::vector<float, 16> gv = aie::load_v<16>(g + h);
        const aie::mask<16> oa = aie::mask<16>::from_uint32(ga >= h && ga < h + 16 ? 1u << (ga - h) : 0u);
        const aie::mask<16> ob = aie::mask<16>::from_uint32(gb >= h && gb < h + 16 ? 1u << (gb - h) : 0u);
        gv = aie::select(gv, aie::add(gv, aie::broadcast<float, 16>(sa)), oa);
        gv = aie::select(gv, aie::add(gv, aie::broadcast<float, 16>(sb)), ob);
        aie::store_v(g + h, gv);
      }
    }
  }
}
}  // namespace

// st += the per-(pair, channel) sums of a C tile's values and squares
__attribute__((noinline)) void {{ p }}_stats_impl(const bfloat16 *__restrict c, float *__restrict st) {
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int pc = 0; pc < 20; pc++) {
    const bfloat16 *__restrict pa = c + 64 * pc;
    aie::accum<accfloat, 64> a1 = aie::zeros<accfloat, 64>(), a2 = aie::zeros<accfloat, 64>();
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int tg = 0; tg < 8; tg++) {
      const aie::vector<bfloat16, 64> v = aie::load_v<64>(pa);
      pa += 1280;
      a1 = aie::mac(a1, v, one);
      a2 = aie::mac(a2, v, v);
    }
    float *ps = st + 32 * pc;
    aie::store_v(ps, aie::add(aie::load_v<16>(ps), {{ p }}_r16(a1).template to_vector<float>()));
    aie::store_v(ps + 16, aie::add(aie::load_v<16>(ps + 16), {{ p }}_r16(a2).template to_vector<float>()));
  }
}

// gacc -> gst [pair][mean 32 | rstd 32]
static __attribute__((noinline, minsize)) void {{ p }}_final(const float *__restrict gacc, float *__restrict gst,
                                                             const int32_t *__restrict ws) {
  for (int pair = 0; pair < 2; pair++)
    for (int h = 0; h < 2; h++)
      {{ p }}_mrstd(aie::load_v<16>(gacc + 64 * pair + 16 * h), aie::load_v<16>(gacc + 64 * pair + 32 + 16 * h), ws,
                    gst + 64 * pair + 16 * h, gst + 64 * pair + 32 + 16 * h);
}

// gst, gb = [gamma 80 | beta 80] (fp32) of the pass -> coef [pc][s_hi 8 | s_lo 8 | t_hi 8 | t_lo 8]: y = x s + t
__attribute__((noinline, minsize)) void {{ p }}_coef_impl(const float *__restrict gst, const float *__restrict gb,
                                                          int pass, int G, bfloat16 *__restrict coef) {
  for (int pc = 0; pc < 20; pc++) {
    const int pair = pc >= 10 ? 1 : 0, cb = pc - 10 * pair;
    const unsigned c0 = 80 * pass + 8 * cb;
    const int ga = c0 / (unsigned)G, gb_ = (c0 + 7) / (unsigned)G;
    const aie::mask<16> ma = {{ p }}_lo((ga + 1) * G - c0);
    const float *g = gst + 64 * pair;  // ([mean 32 | rstd 32])
    const aie::vector<float, 16> rs = aie::select(aie::broadcast<float, 16>(g[32 + gb_]), aie::broadcast<float, 16>(g[32 + ga]), ma);
    const aie::vector<float, 16> mu = aie::select(aie::broadcast<float, 16>(g[gb_]), aie::broadcast<float, 16>(g[ga]), ma);
    const aie::vector<float, 8> g8 = aie::load_v<8>(gb + 8 * cb), b8 = aie::load_v<8>(gb + 80 + 8 * cb);
    const aie::vector<float, 16> s = al::mulff<16>(rs, aie::concat(g8, g8));
    const aie::vector<float, 16> t = aie::sub(aie::concat(b8, b8), al::mulff<16>(mu, s));
    aie::vector<bfloat16, 16> sh, sl, th, tl;
    al::split2<16>(s, sh, sl);
    al::split2<16>(t, th, tl);
    aie::store_v(coef + 32 * pc, aie::concat(sh.extract<8>(0), sl.extract<8>(0)));
    aie::store_v(coef + 32 * pc + 16, aie::concat(th.extract<8>(0), tl.extract<8>(0)));
  }
}

// after a pass's tiles (W_NFIN): the pass's GN record (its channel sums by group) -> band record W_FQ (REC B each; the
// channel sums in band + NREC REC); W_G2: a second record (groups of W_G2 channels, pass + W_PS0)
__attribute__((noinline, minsize)) void {{ p }}_fin_impl(int32_t *__restrict ws, int32_t *__restrict scr, int32_t *__restrict band) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *st = (float *)scr, *gacc = (float *)band + ws[{{ W_FQ }}]++ * {{ REC // 4 }}, *rec = (float *)band + {{ NREC * REC // 4 }};
  for (int pc = 0; pc < 20; pc++) {
    const int pair = pc >= 10 ? 1 : 0, cb = pc - 10 * pair;
    const aie::vector<float, 16> s1 = aie::load_v<16>(st + 32 * pc), s2 = aie::load_v<16>(st + 32 * pc + 16);
    aie::store_v(rec + 160 * pair + 8 * cb, aie::add(s1.extract<8>(0), s1.extract<8>(1)));
    aie::store_v(rec + 160 * pair + 80 + 8 * cb, aie::add(s2.extract<8>(0), s2.extract<8>(1)));
  }
  for (int j = 0; j < 128; j += 16) aie::store_v(gacc + j, aie::zeros<float, 16>());
  {{ p }}_group(rec, ws[{{ W_PASS }}], ws[{{ W_G }}], gacc);
  if (ws[{{ W_G2 }}]) {  // (a second record: the groups of a concatenation, the pass offset W_PS0)
    gacc = (float *)band + ws[{{ W_FQ }}]++ * {{ REC // 4 }};
    for (int j = 0; j < 128; j += 16) aie::store_v(gacc + j, aie::zeros<float, 16>());
    {{ p }}_group(rec, ws[{{ W_PASS }}] + ws[{{ W_PS0 }}], ws[{{ W_G2 }}], gacc);
  }
  for (int j = 0; j < 640; j += 16) aie::store_v(st + j, aie::zeros<float, 16>());
}

// pre-op slot k of ns: 5 GN records ([pair 2][S1 32 | S2 32] fp32 in 576 B) summed into gacc; after the last, the
// statistics gst
__attribute__((noinline, minsize)) void {{ p }}_grec_impl(const int32_t *__restrict b, int32_t *__restrict ws,
                                                          int32_t *__restrict scr, int k, int ns) {
  float *gacc = (float *)scr;
  if (k == 0)
    for (int j = 0; j < 128; j += 16) aie::store_v(gacc + j, aie::zeros<float, 16>());
  const float *r = (const float *)b;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 128; j += 16)
      aie::store_v(gacc + j, aie::add(aie::load_v<16>(gacc + j), aie::load_v<16>(r + {{ REC // 4 }} * i + j)));
  if (k == ns - 1) {{ p }}_final(gacc, (float *)scr + 128, ws);
}
"""

_SRC_T = r"""
// LayerNorm sums of a C tile: per row (2 tg + pair) and lane (token) S1, S2 over its 80 channels, added to acc
// [16 rows][S1 8 | S2 8] fp32
__attribute__((noinline)) void {{ p }}_lnacc_impl(const bfloat16 *__restrict c, float *__restrict acc) {
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int row = 0; row < 16; row++) {
    aie::accum<accfloat, 64> a1 = aie::zeros<accfloat, 64>(), a2 = aie::zeros<accfloat, 64>();
    AIE_LOOP_UNROLL_FULL
    for (int cb = 0; cb < 10; cb++) {
      const aie::vector<bfloat16, 64> v = aie::load_v<64>(c + 640 * row + 64 * cb);
      a1 = aie::mac(a1, v, one);
      a2 = aie::mac(a2, v, v);
    }
    const aie::vector<float, 16> s = aie::concat(al::group8<false>(a1.template to_vector<float>()),
                                                 al::group8<false>(a2.template to_vector<float>()));
    aie::store_v(acc + 16 * row, aie::add(s, aie::load_v<16>(acc + 16 * row)));
  }
}

// per-token sums [16 rows][S1 8 | S2 8] -> tok [16 rows][a_hi 8 | a_lo 8 | b_hi 8 | b_lo 8] bf16: u = x a + b, a = rstd,
// b = -mean rstd (W_INVN: 1 / C, W_EPS)
static __attribute__((noinline, minsize)) void {{ p }}_lntok(const float *__restrict sums, bfloat16 *__restrict tok,
                                                           const int32_t *__restrict ws) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *mr = (float *)(tok + 512);  // (scratch after tok: mean 16 | rstd 16)
  for (int row = 0; row < 16; row += 2) {
    const aie::vector<float, 16> u0 = aie::load_v<16>(sums + 16 * row), u1 = aie::load_v<16>(sums + 16 * row + 16);
    {{ p }}_mrstd(aie::concat(u0.extract<8>(0), u1.extract<8>(0)), aie::concat(u0.extract<8>(1), u1.extract<8>(1)), ws,
                  mr, mr + 16);
    const aie::vector<float, 16> r = aie::load_v<16>(mr + 16);
    const aie::vector<float, 16> bb = aie::sub(aie::zeros<float, 16>(), al::mulff<16>(aie::load_v<16>(mr), r));
    aie::vector<bfloat16, 16> ah, al_, bh, bl;
    al::split2<16>(r, ah, al_);
    al::split2<16>(bb, bh, bl);
    for (int h = 0; h < 2; h++) {
      bfloat16 *o = tok + 32 * (row + h);
      aie::store_v(o, aie::concat(ah.extract<8>(h), al_.extract<8>(h)));
      aie::store_v(o + 16, aie::concat(bh.extract<8>(h), bl.extract<8>(h)));
    }
  }
}

// GEGLU epilogue: the C tile's q 0..4 = hid, 5..9 = gate (40 channels each) -> per token hid gelu(gate) -> panels
// [tg][5 q][pair] from 0 (80 panels); band: gate z, |z| (-> gelu), the products, then the GELU table at 4096
__attribute__((noinline, minsize)) void {{ p }}_geglu_impl(int32_t *__restrict c_, int32_t *__restrict band_) {
  const bfloat16 *x = (const bfloat16 *)c_;
  bfloat16 *z = (bfloat16 *)band_, *t = z + 640, *y = z + 1280;
  const uint32_t *tab = (const uint32_t *)((uint8_t *)band_ + 4096);
  const aie::vector<bfloat16, 64> hi = aie::broadcast<bfloat16, 64>({{ 2 * gelu_tmax }}f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o((bfp16ebs8 *)c_);
  for (int tg = 0; tg < 8; tg++) {
    for (int i = 0; i < 10; i++) {  // (pair, q): gate z, its clamped |z|
      const aie::vector<bfloat16, 64> g = aie::load_v<64>(x + 1280 * tg + 640 * (i / 5) + 64 * (5 + i % 5));
      aie::store_v(z + 64 * i, g);
      const aie::vector<bfloat16, 64> ga = aie::abs(g);
      aie::store_v(t + 64 * i, aie::min(aie::add(ga, ga), hi));  // (2 |z|: the GELU table in the SiLU geometry)
    }
    {{ lut }}_tabu(z, t, t, t, tab);
    for (int i = 0; i < 10; i++)
      aie::store_v(y + 64 * i, aie::mul(aie::load_v<64>(t + 64 * i),
                                        aie::load_v<64>(x + 1280 * tg + 640 * (i / 5) + 64 * (i % 5))).template to_vector<bfloat16>());
    for (int q = 0; q < 5; q++) {
      o.push(aie::accum<accfloat, 64>(aie::load_v<64>(y + 64 * q)).template to_vector<bfp16ebs8>());
      o.push(aie::accum<accfloat, 64>(aie::load_v<64>(y + 64 * (5 + q))).template to_vector<bfp16ebs8>());
    }
  }
}

// pre-op slot k of ns: 2 LayerNorm records: those of the core's tile summed; after the last, the tokens' a, b
__attribute__((noinline, minsize)) void {{ p }}_lrec_impl(const int32_t *__restrict b, int32_t *__restrict ws,
                                                          int32_t *__restrict scr, int k, int ns) {
  float *sums = (float *)scr + 256;
  if (k == 0)
    for (int j = 0; j < 256; j += 16) aie::store_v(sums + j, aie::zeros<float, 16>());
  for (int i = 0; i < 2; i++) {
    const int32_t *r = b + {{ LREC // 4 }} * i;
    if (r[0] == ws[{{ W_TILE }}])
      for (int j = 0; j < 256; j += 16)
        aie::store_v(sums + j, aie::add(aie::load_v<16>(sums + j),  // (the second record is 1440 B in: unaligned)
                                        aie::load_unaligned_v<16>((const float *)r + 16 + j)));
  }
  if (k == ns - 1) {{ p }}_lntok(sums, (bfloat16 *)scr, ws);
}
"""

# the variant entries called from the shared kernels (regular variant: GroupNorm statistics and coefficients;
# transformer variant: LayerNorm, GEGLU)
R_ENTRIES = [("stats", "const bfloat16 *__restrict c, float *__restrict st", "c, st"),
             ("coef", "const float *__restrict gst, const float *__restrict gb, int pass, int G, bfloat16 *__restrict coef", "gst, gb, pass, G, coef"),
             ("fin", "int32_t *__restrict ws, int32_t *__restrict scr, int32_t *__restrict band", "ws, scr, band"),
             ("grec", "const int32_t *__restrict b, int32_t *__restrict ws, int32_t *__restrict scr, int k, int ns", "b, ws, scr, k, ns")]
T_ENTRIES = [("lnacc", "const bfloat16 *__restrict c, float *__restrict acc", "c, acc"),
             ("geglu", "int32_t *__restrict c_, int32_t *__restrict band_", "c_, band_"),
             ("lrec", "const int32_t *__restrict b, int32_t *__restrict ws, int32_t *__restrict scr, int k, int ns", "b, ws, scr, k, ns")]

CONV_ENTRY = ("conv3", ("const int32_t *__restrict a0, const int32_t *__restrict a1, int32_t *__restrict band, "
                        "const int32_t *__restrict ring, int32_t *__restrict ws, int32_t *__restrict c"),
              "a0, a1, band, ring, ws, c")

_SRC = r"""
namespace {
{{ mac_src }}

// K1: one k-tile (CBC ci-blocks; NDX taps) into the C tile [R][L][2 pairs][2 J][64] bf16; A: band rows of S L + H groups
// [CBC][2] panels (H: halo groups), B: [J][NDX][CBC][2]
template <int NDX>
static inline __attribute__((always_inline)) void {{ p }}_k1(
    const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
    const bfloat16 __aie_dm_resource_c *__restrict pCld, bfloat16 __aie_dm_resource_c *__restrict pCst,
    int L, int J, int nblk, int S, int H) {
  using AStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b>;
  using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
  constexpr int ks = NDX * {{ CBC }};
  AStream a_stream(pA);
  BStream b_stream(pB);
  // (S: the output groups' stride in the band's groups)
  dims_3d_t patA(J - 1, -2 * ks, L - 1, -2 * ks + 2 * S * {{ CBC }}, -2 * ks + 2 * (S + H) * {{ CBC }});
  dims_2d_t patB(J - 1, 0, -J * 2 * ks);
  const int c2 = J * 128;  // (pair 1's sub-blocks)
  // (one iterator per pointer: add_3d_ptr advances the iterator's counters)
  dims_3d_t patL(J - 1, 128, L - 1, 128 + c2, 128 + c2), patS(J - 1, 128, L - 1, 128 + c2, 128 + c2);
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int it = 0; it < nblk; it++) {
    MMUL acc0((aie::accum<accfloat, 64>(aie::load_v<64>(pCld))));
    MMUL acc1((aie::accum<accfloat, 64>(aie::load_v<64>(pCld + 64))));
    MMUL acc2((aie::accum<accfloat, 64>(aie::load_v<64>(pCld + c2))));
    MMUL acc3((aie::accum<accfloat, 64>(aie::load_v<64>(pCld + c2 + 64))));
    AIE_LOOP_UNROLL_FULL
    for (int s = 0; s < ks; s++) {
      aie::block_vector<bfp16ebs8, 64> A0, A1, B0, B1;
      if (s % 4 == 0) {
        B0 = b_stream.pop(); B1 = b_stream.pop(); A0 = a_stream.pop(); A1 = a_stream.pop();
      } else {
        A0 = a_stream.pop(); A1 = a_stream.pop(); B0 = b_stream.pop(); B1 = b_stream.pop();
      }
      acc0.mac(A0, aie::op_transpose(B0));
      acc1.mac(A0, aie::op_transpose(B1));
      acc2.mac(A1, aie::op_transpose(B0));
      acc3.mac(A1, aie::op_transpose(B1));
    }
    aie::store_v(pCst, acc0.template to_vector<bfloat16>());
    aie::store_v(pCst + 64, acc1.template to_vector<bfloat16>());
    aie::store_v(pCst + c2, acc2.template to_vector<bfloat16>());
    aie::store_v(pCst + c2 + 64, acc3.template to_vector<bfloat16>());
    a_stream.seek(patA);
    b_stream.seek(patB);
    pCld = add_3d_ptr(pCld, patL);
    pCst = add_3d_ptr(pCst, patS);
  }
}

// the expand's lane masks of a group's 9 vectors: 0x00 at the panels' first lane (row 0) / last lane (row 1)
alignas(64) const uint8_t {{ p }}_hm[2][576] = {{ hm }};

// halo-free rows [NR][Li][CBC][2] panels -> band rows [NR][Li + 2][CBC][2]: group 0 = group Li - 1 one lane down (lane
// 0 zero: the left image edge), group Li + 1 = group 0 one lane up (lane 7 zero). A group = 8 panels of 8 lanes x 9 B
// = 9 vectors of 64 B; a lane shift moves its bytes by 9, the masks zero the panels' first / last lane.
static inline __attribute__((noinline)) void {{ p }}_expand(const uint8_t *__restrict a, uint8_t *__restrict band,
                                                            int Li, int NR) {
  {  // main groups (II 1: int32 vectors, walking pointers)
    const int32_t *__restrict ps = (const int32_t *)a;
    int32_t *__restrict pd = (int32_t *)(band + 576);
    dims_2d_t dd(Li * 9 - 1, 16, 2 * 144 + 16);
    AIE_LOOP_MIN_ITERATION_COUNT(9)
    for (int i = 0; i < Li * 9 * NR; i++) {
      aie::store_v(pd, aie::load_v<16>(ps));
      ps += 16;
      pd = add_2d_ptr(pd, dd);
    }
  }
  {  // group 0: group Li - 1 one lane down (a shift by 9 bytes from the previous vector; the masks zero what crosses
     // panels and rows)
    const uint8_t *__restrict pl = a + (Li - 1) * 576;
    const uint8_t *__restrict pm = {{ p }}_hm[0];
    uint8_t *__restrict po = band;
    aie::vector<uint8, 64> prev = aie::zeros<uint8, 64>();
    dims_2d_t ds(8, 64, Li * 576 - 512), dm(8, 64, -512), dd(8, 64, (Li + 2) * 576 - 512);
    AIE_LOOP_MIN_ITERATION_COUNT(9)
    for (int i = 0; i < 9 * NR; i++) {
      const aie::vector<uint8, 64> v = aie::load_v<64>(pl);
      aie::store_v(po, aie::bit_and(aie::shuffle_up_fill(v, prev, 9), aie::load_v<64>(pm)));
      prev = v;
      pl = add_2d_ptr(pl, ds);
      pm = add_2d_ptr(pm, dm);
      po = add_2d_ptr(po, dd);
    }
  }
  {  // group Li + 1: group 0 one lane up
    const uint8_t *__restrict pr = a;
    const uint8_t *__restrict pm = {{ p }}_hm[1];
    uint8_t *__restrict po = band + (Li + 1) * 576;
    dims_2d_t ds(8, 64, Li * 576 - 512), dm(8, 64, -512), dd(8, 64, (Li + 2) * 576 - 512);
    AIE_LOOP_MIN_ITERATION_COUNT(9)
    for (int i = 0; i < 9 * NR; i++) {
      const aie::vector<uint8, 64> v = aie::load_v<64>(pr), nx = aie::load_v<64>(pr + 64);
      aie::store_v(po, aie::bit_and(aie::shuffle_down_fill(v, nx, 9), aie::load_v<64>(pm)));
      pr = add_2d_ptr(pr, ds);
      pm = add_2d_ptr(pm, dm);
      po = add_2d_ptr(po, dd);
    }
  }
}

// ---- sums and GroupNorm. scr: sum op: st [pc 20][S1 16 | S2 16] fp32 (2560 B; pc = 10 pair + cb, lanes [2][8 ch]);
// apply op: gacc [pair 2][S1 32 | S2 32] fp32 (512 B), gst [pair 2][32][mean, var, rstd, 0] (1024 B) at float 128, coef
// [pc 20][s_hi 8 | s_lo 8 | t_hi 8 | t_lo 8] bf16 (1280 B) at bf16 768 ----
}  // namespace (the shared kernels' helpers: external, emitted where defined, not at the end)
{{ silu }}
namespace {

// the first n lanes of each 8-lane half
static inline aie::mask<16> {{ p }}_lo(int n) {
  const unsigned bits = n >= 8 ? 0xFFu : (n <= 0 ? 0u : (1u << n) - 1);
  return aie::mask<16>::from_uint32(bits | bits << 8);
}

// 64 lanes [8 tokens][8 ch] (fp32) -> 16 lanes [2][8 ch]
static inline aie::accum<accfloat, 16> {{ p }}_r16(aie::accum<accfloat, 64> a) {
  const aie::accum<accfloat, 32> h = aie::add(a.extract<32>(0), a.extract<32>(1));
  return aie::add(h.extract<16>(0), h.extract<16>(1));
}

static inline aie::vector<bfloat16, 64> {{ p }}_rep(const bfloat16 *c) {
  return aie::load_v<8>(c).grow_replicate<64>();  // 8 channels -> [8 tokens][8 ch]
}



// gacc -> gst: mean, var = E[x^2] - E[x]^2, rstd per (pair, group); W_INVN: 1 / values per group (fp32 bits), W_EPS
// (sums s1, s2 of n values: inv_n = 1 / n) -> mean, rstd = 1 / sqrt(max(E[x^2] - mean^2, 0) + eps)
static inline __attribute__((noinline, minsize)) void {{ p }}_mrstd(aie::vector<float, 16> s1, aie::vector<float, 16> s2,
                                                                  const int32_t *__restrict ws, float *__restrict m_out,
                                                                  float *__restrict r_out) {
  const aie::vector<float, 16> inv_n = aie::broadcast<int32, 16>(ws[{{ W_INVN }}]).template cast_to<float>();
  const aie::vector<float, 16> eps = aie::broadcast<int32, 16>(ws[{{ W_EPS }}]).template cast_to<float>();
  const aie::vector<float, 16> m = al::mulff<16>(s1, inv_n);
  const aie::vector<float, 16> ve = aie::add(aie::max(aie::sub(al::mulff<16>(s2, inv_n), al::mulff<16>(m, m)),
                                                      aie::zeros<float, 16>()), eps);
  aie::vector<float, 16> r;
  for (int k = 0; k < 16; k++) r.set(aie::invsqrt(ve[k]), k);
  aie::store_v(m_out, m);
  aie::store_v(r_out, r);
}



// GN apply in place: the input C tile (bf16, at element ROFF) -> per tg, (SiLU of) x s + t rounded to bf16 per pair
// (z, |z| clamped in band; SiLU's output over the consumed input), then the tg's panels [q][pair] from byte 1440 tg (the
// output stays behind the unread input)
// 8 per-token values -> 64 lanes [token][8 ch]
static inline aie::vector<bfloat16, 64> {{ p }}_tok64(const bfloat16 *v) {
  aie::vector<bfloat16, 8> x = aie::load_v<8>(v);
  const auto z1 = aie::interleave_zip(x, x, 1);
  const aie::vector<bfloat16, 16> x16 = aie::concat(z1.first, z1.second);
  const auto z2 = aie::interleave_zip(x16, x16, 1);
  const aie::vector<bfloat16, 32> x32 = aie::concat(z2.first, z2.second);
  const auto z3 = aie::interleave_zip(x32, x32, 1);
  return aie::concat(z3.first, z3.second);
}

}  // namespace
__attribute__((noinline)) void {{ p }}_apply(int32_t *__restrict c_, const int32_t *__restrict scr,
                                                           int32_t *__restrict band_, bool silu, bool ln) {
  bfloat16 *x = (bfloat16 *)c_ + {{ ROFF }};
  const bfloat16 *coef = (const bfloat16 *)scr + 768;
  bfloat16 *z = (bfloat16 *)band_, *t = z + 640;
  const uint32_t *tab = (const uint32_t *)((uint8_t *)band_ + 2560);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f), hi = aie::broadcast<bfloat16, 64>(24.0f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o((bfp16ebs8 *)c_);
  for (int tg = 0; tg < 8; tg++) {
    for (int pair = 0; pair < 2; pair++) {
      bfloat16 *px = x + 640 * (2 * tg + pair);
      // (LayerNorm: u = bf16(x a + b) per token first; the tokens' a, b: [row 2 tg + pair][a_hi | a_lo | b_hi | b_lo])
      const bfloat16 *tk = (const bfloat16 *)scr + 32 * (2 * tg + pair);
      const aie::vector<bfloat16, 64> ah = {{ p }}_tok64(tk), al_ = {{ p }}_tok64(tk + 8);
      const aie::accum<accfloat, 64> bt = aie::mac(aie::mul({{ p }}_tok64(tk + 16), one), {{ p }}_tok64(tk + 24), one);
      for (int cb = 0; cb < 10; cb++) {
        const bfloat16 *cf = coef + 32 * (10 * pair + cb);
        const aie::accum<accfloat, 64> ta = aie::mac(aie::mul({{ p }}_rep(cf + 16), one), {{ p }}_rep(cf + 24), one);
        aie::vector<bfloat16, 64> v = aie::load_v<64>(px + 64 * cb);
        if (ln) v = aie::mac(aie::mac(bt, v, ah), v, al_).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> zb =
            aie::mac(aie::mac(ta, v, {{ p }}_rep(cf)), v, {{ p }}_rep(cf + 8)).template to_vector<bfloat16>();
        aie::store_v(z + 64 * cb, zb);
        aie::store_v(t + 64 * cb, aie::min(aie::abs(zb), hi));
      }
      if (silu)
        {{ lut }}_tabu(z, t, t, px, tab);
      else
        for (int cb = 0; cb < 10; cb++) aie::store_v(px + 64 * cb, aie::load_v<64>(z + 64 * cb));
    }
    for (int q = 0; q < 10; q++) {
      o.push(aie::accum<accfloat, 64>(aie::load_v<64>(x + 640 * (2 * tg) + 64 * q)).template to_vector<bfp16ebs8>());
      o.push(aie::accum<accfloat, 64>(aie::load_v<64>(x + 640 * (2 * tg + 1) + 64 * q)).template to_vector<bfp16ebs8>());
    }
  }
}
namespace {




// the next A object (ping-pong) / the next n B slots (ring): advance the positions in ws
static inline __attribute__((always_inline)) const int32_t *{{ p }}_a(const int32_t *a0, const int32_t *a1, int32_t *ws) {
  const int32_t *a = ws[{{ W_APAR }}] ? a1 : a0;
  ws[{{ W_APAR }}] ^= 1;
  return a;
}

static inline __attribute__((always_inline)) const int32_t *{{ p }}_b(const int32_t *ring, int32_t *ws, int n) {
  const int pos = ws[{{ W_BPOS }}];
  ws[{{ W_BPOS }}] = pos + n == {{ NQ }} ? 0 : pos + n;
  return ring + pos * {{ Q // 4 }};
}

}  // namespace

extern "C" {

// the command (B slot 0 of 3): ws words; the ring and ping-pong positions persist across ops
__attribute__((minsize)) void {{ p }}_begin(const int32_t *__restrict ring, int32_t *__restrict ws) {
  const int32_t *b = {{ p }}_b(ring, ws, 3);
  for (int i = 0; i < {{ W_RI }}; i++) ws[i] = b[i];
  if (ws[{{ W_NPASS }}] == 0) ws[{{ W_NPASS }}] = 1;
  ws[{{ W_PK }}] = 0;
  ws[{{ W_FQ }}] = 0;
  ws[{{ W_TILE }}] = b[{{ CMD_TILE }} + ws[{{ W_CID }}]];
}

// a GN-statistics op: the pass sums in scr from zero (an apply op leaves its statistics there); a LayerNorm-statistics
// op: its record (tile, then per-token sums) in band, emitted after the passes
__attribute__((minsize)) void {{ p }}_lnbegin(int32_t *__restrict ws, int32_t *__restrict band, int32_t *__restrict scr) {
  if (ws[{{ W_NE }}] == {{ E_GNST }})
    for (int j = 0; j < 640; j += 16) aie::store_v(scr + j, aie::zeros<int32, 16>());
  if (ws[{{ W_NE }}] != {{ E_LNST }}) return;
  for (int i = 0; i < {{ LREC // 64 }}; i++) aie::store_v(band + 16 * i, aie::zeros<int32, 16>());
  band[0] = ws[{{ W_TILE }}];
  ws[{{ W_FQ }}] = 2;  // (emit: 2 x REC bytes)
}

// piece i of a C object (5 A objects of A_OBJ bytes each) added to this core's C tile (from element W_ROFF): C += it
void {{ p }}_resid(const int32_t *__restrict a0, const int32_t *__restrict a1, int32_t *__restrict ws,
                   int32_t *__restrict c) {
  const bfloat16 *src = (const bfloat16 *){{ p }}_a(a0, a1, ws);
  const int ri = ws[{{ W_RI }}];  // (the piece of the object: 5 per C object)
  ws[{{ W_RI }}] = ri == 4 ? 0 : ri + 1;
  const int lo = ri * {{ A_OBJ // 2 }};
  const int n = {{ C_OBJ // 2 }} - lo < {{ A_OBJ // 2 }} ? {{ C_OBJ // 2 }} - lo : {{ A_OBJ // 2 }};
  bfloat16 *dst = (bfloat16 *)c + ws[{{ W_ROFF }}] + lo;
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int j = 0; j < n; j += 32) {
    aie::accum<accfloat, 32> s(aie::load_v<32>(dst + j));
    s = aie::add(s, aie::load_v<32>(src + j));
    aie::store_v(dst + j, s.template to_vector<bfloat16>());
  }
}

// GEMM k-tile of 32: A object [8 tg][4 kb][2 pairs] bfp16 (the MAC's pop order), 1 B slot [5][4][2]: C [16 r8][10][64]
void {{ p }}_gemm(const int32_t *__restrict a0, const int32_t *__restrict a1, const int32_t *__restrict ring,
                  int32_t *__restrict ws, int32_t *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int32_t *a = {{ p }}_a(a0, a1, ws);
  const int32_t *b = {{ p }}_b(ring, ws, 1);
  {{ mac }}((const bfp16ebs8 *)a, b, (const bfloat16 __aie_dm_resource_c *)c, (bfloat16 __aie_dm_resource_c *)c);
}

// padding B slots of a tile (keep the ring at 0 / 3)
__attribute__((minsize)) void {{ p }}_bskip(int32_t *__restrict ws) {
  const int pos = ws[{{ W_BPOS }}] + ws[{{ W_NBP }}];
  ws[{{ W_BPOS }}] = pos >= {{ NQ }} ? pos - {{ NQ }} : pos;
}

// the core's records (W_NEMIT, after its passes) -> C (one object)
__attribute__((minsize)) void {{ p }}_emit(int32_t *__restrict ws, int32_t *__restrict band, int32_t *__restrict c) {
  const int n = ws[{{ W_FQ }}] * {{ REC // 64 }};
  for (int i = 0; i < n; i++) aie::store_v(c + 16 * i, aie::load_v<16>(band + 16 * i));
  ws[{{ W_FQ }}] = 0;
}


}  // extern "C"
"""

# the shared kernels that call the variant entries (after every shared kernel: ``Overlay.source``)
_SRC_V = r"""
extern "C" {

// per tile: the init object (3 B slots): [2 pairs][80] bf16 bias (bytes 0 ..), [gamma 80 | beta 80] fp32 (bytes 640 ..),
// the pass (int32 at byte 1280). Mode 0: C := the bias broadcast over the sub-blocks' rows; mode 1 (GN apply): C := 0,
// coef := the pass's GN coefficients
__attribute__((minsize)) void {{ p }}_init(const int32_t *__restrict ring, int32_t *__restrict ws, int32_t *__restrict c,
                  int32_t *__restrict scr) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  ws[{{ W_RI }}] = 0;
  const bfloat16 *w = (const bfloat16 *){{ p }}_b(ring, ws, 3);
  ws[{{ W_PASS }}] = ((const int32_t *)w)[320];
  bfloat16 *cc = (bfloat16 *)c;
  if (ws[{{ W_IM }}]) {  // 1: GN (statistics), 4: LayerNorm / plain (gamma, beta as they are)
    for (int i = 0; i < {{ C_BUF // 2 }}; i += 32) aie::store_v(cc + i, aie::zeros<bfloat16, 32>());
    const float *gb = (const float *)((const uint8_t *)w + 640);
    if (ws[{{ W_IM }}] == 1) {
      {{ p }}_coef((const float *)scr + 128, gb, ws[{{ W_PASS }}], ws[{{ W_G }}], (bfloat16 *)scr + 768);
      return;
    }
    bfloat16 *coef = (bfloat16 *)scr + 768;
    for (int pc = 0; pc < 20; pc++) {
      const int cb = pc >= 10 ? pc - 10 : pc;
      aie::vector<bfloat16, 16> sh, sl, th, tl;
      const aie::vector<float, 8> g8 = aie::load_v<8>(gb + 8 * cb), b8 = aie::load_v<8>(gb + 80 + 8 * cb);
      al::split2<16>(aie::concat(g8, g8), sh, sl);
      al::split2<16>(aie::concat(b8, b8), th, tl);
      aie::store_v(coef + 32 * pc, aie::concat(sh.extract<8>(0), sl.extract<8>(0)));
      aie::store_v(coef + 32 * pc + 16, aie::concat(th.extract<8>(0), tl.extract<8>(0)));
    }
    return;
  }
  for (int rg = 0; rg < 16; rg++) {
    const int pp = rg % 2;
    for (int cb = 0; cb < 10; cb++) {
      aie::vector<bfloat16, 8> v8 = aie::load_v<8>(w + pp * 80 + cb * 8);
      aie::store_v(cc + (rg * 10 + cb) * 64, aie::concat(v8, v8, v8, v8, v8, v8, v8, v8));
    }
  }
}

// per-tile epilogue (mode W_NE): 1: the C tile's 160 sub-blocks (bf16) -> bfp16 panels in place, in their order (panel i
// written at 72 i, behind its source at 128 i); 2: GN sums of the tile into st; 3 / 4: GN apply (/ + SiLU) -> panels
__attribute__((minsize)) void {{ p }}_epi(int32_t *__restrict ws, int32_t *__restrict c, int32_t *__restrict scr, int32_t *__restrict band) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int m = ws[{{ W_NE }}];
  if (m == 1 || m == {{ E_BFPT }}) {  // (E_BFPT: each 8 x 8 sub-block transposed first)
    const bfloat16 *src = (const bfloat16 *)c;
    const bool tr = m != 1;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)c);
    for (int i = 0; i < 160; i++) {
      const aie::vector<bfloat16, 64> v = aie::load_v<64>(src + i * 64);
      out << aie::accum<accfloat, 64>(tr ? aie::transpose(v, 8, 8) : v).template to_vector<bfp16ebs8>();
    }
  } else if (m == {{ E_GNST }}) {
    {{ p }}_stats((const bfloat16 *)c, (float *)scr);
  } else if (m == {{ E_LNST }}) {
    {{ p }}_lnacc((const bfloat16 *)c, (float *)band + 16);
  } else if (m == {{ E_GEGLU }}) {
    {{ p }}_geglu(c, band);
  } else if (m >= {{ E_GN }}) {
    {{ p }}_apply(c, scr, band, m == {{ E_GNSILU }}, m == {{ E_LN }});
  }
}

// pre-op B slots (W_NPRE): the first W_NSTS hold 5 GN records each ([pair 2][S1 32 | S2 32] fp32 in 576 B): summed into
// gacc, then the statistics gst; the next two the SiLU table (3200 B) -> band + 2560
__attribute__((minsize)) void {{ p }}_pre(const int32_t *__restrict ring, int32_t *__restrict ws, int32_t *__restrict scr,
                 int32_t *__restrict band) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int32_t *b = {{ p }}_b(ring, ws, 1);
  const int k = ws[{{ W_PK }}]++, ns = ws[{{ W_NSTS }}];
  if (ws[{{ W_PM }}] == 1) {  // (LayerNorm records)
    {{ p }}_lrec(b, ws, scr, k, ns);
    return;
  }
  if (k < ns) {  // (GroupNorm records)
    {{ p }}_grec(b, ws, scr, k, ns);
  } else if (k - ns < 2) {  // a table (3200 B in 2 slots): SiLU -> band + 2560, GELU (W_PM 2) -> band + 4096
    const int off = (k - ns) * {{ Q }};
    const int n = off == 0 ? {{ Q // 64 }} : {{ (3200 - Q) // 64 }};
    int32_t *d = (int32_t *)((uint8_t *)band + (ws[{{ W_PM }}] == 2 ? 4096 : 2560) + off);
    for (int i = 0; i < n; i++) aie::store_v(d + 16 * i, aie::load_v<16>(b + 16 * i));
  }
}

}  // extern "C"
"""


def _silu() -> str:
    """The SiLU table pass (``{LUT}_tabu(z, t, t, y, tab)`` over 640 values, t = min(|z|, 24))."""
    return act_lut.SILU.render_unrolled(640)


def _gelu_table() -> tuple[np.ndarray, float]:
    """act_lut.GELU's table for the SiLU table pass (bit-exact): fed t' = 2 t, its segment index and remainder are
    exactly 2x GELU's (m 32 = 2 x 16, shift 6 = 7 - 1), so c1 / 2, c2 / 4; each copy padded to SiLU's 400 words."""
    import copy
    g = copy.copy(act_lut.GELU)
    c = act_lut.GELU.coefficients.copy()
    c[:, 1] /= 2
    c[:, 2] /= 4
    g.__dict__["coefficients"] = c
    w = g.table_words()
    nw = len(act_lut.SILU.table_words())
    w = np.concatenate([w, np.zeros(nw - len(w), np.uint32)])
    return np.concatenate([w, w]).view(np.uint8), act_lut.GELU.t_max


def silu_table() -> np.ndarray:
    return np.asarray(act_lut.SILU.table_buffer()).view(np.uint8)


def _masks() -> str:
    """The expand's byte masks of a group (576 B): 0 at the panels' first lane (bytes 72 p .. + 9) / last lane."""
    rows = [[0 if b % 72 < 9 else 255 for b in range(576)], [0 if b % 72 >= 63 else 255 for b in range(576)]]
    return "{" + ", ".join("{" + ", ".join(map(str, r)) + "}" for r in rows) + "}"


class Overlay:
    """The overlay design (the kernels take the level's geometry from the op's command: one design serves 64x64 and
    32x32) with a level's geometry for its ops: tiles of 128 tokens = R = 8 / L rows x L groups x 8 lanes x 2 pairs; A
    objects of A_OBJ B (a conv k-tile's halo-free band rows, a GEMM k-tile, a C-object piece); the conv kernel expands
    RB band rows at a time with their halo groups into the band buffer (BAND B)."""

    def __init__(self, *, W: int = 64, conv: bool = True):
        """conv False: the transformer blocks' variant (the same routes, buffers, locks and core program; the conv
        kernel is a stub, so K1 and the expand are not linked: their program memory holds the extensions' impls)."""
        assert W in (64, 32)
        self.conv = conv
        self.W, self.L = W, W // 8
        self.R = 8 // self.L
        self.G = self.L + 2
        self.A_OBJ = A_OBJ
        self.BAND = BAND
        assert NREC * REC + 1280 <= BAND  # (sum op: the records + a pass's channel sums)
        self.p = kernel_prefix("sd15")  # (the variants' kernels: one translation unit in a library)

    def source(self) -> str:
        """The kernels both variants share (one copy in a library's translation unit: ``kernel_source``), the shared
        kernels that call variant entries (``_SRC_V``), the entries' trampolines, then the variant region
        (``_SRC_CONV``). Every shared definition comes before the first that differs, in both variants: their cores
        link the same code at the same addresses up to the variant region."""
        from artlite.operators.base import JINJA_ENV
        from artlite.operators.gemm.mac import MAC, render_mac
        p = self.p
        entries = [CONV_ENTRY] + R_ENTRIES + T_ENTRIES + self.ext_entries()
        # (a stub's opaque side effect: otherwise the compiler deletes the shared code's calls of an empty entry, and
        # the variants' shared code would differ; one function for every stub: the others are aliases of it)
        stub = f'extern "C" __attribute__((noinline)) void {p}_stub(const char *a) {{ (void)*(volatile const char *)a; }}\n'

        def stubs(es: list) -> str:
            return "".join(f'void {p}_{n}_impl({prm}) __attribute__((alias("{p}_stub")));\n' for n, prm, _ in es)
        protos = 'extern "C" {\n' + "".join(f"void {p}_{n}({prm});\n" for n, prm, _ in entries) + '}  // extern "C"\n'
        var = "\n".join(f"void {p}_{n}_impl({prm});" for n, prm, _ in entries)
        # (functions are emitted in the order of their first reference: the table, after the shared code, places
        # the trampolines no shared kernel calls before every impl; weak: no caller inlines or sees through them)
        var += ('\nextern "C" {\nextern void (*const ' + p + '_entries[])() = {'
                + ", ".join(f"reinterpret_cast<void (*)()>(&{p}_{n})" for n, _, _ in entries) + "};\n"
                + "".join(f"__attribute__((weak, noinline)) void {p}_{n}({prm}) {{ {p}_{n}_impl({args}); }}\n"
                          for n, prm, args in entries) + '}  // extern "C"\n')
        if self.conv:
            var += _SRC_CONV + _SRC_R + stubs(T_ENTRIES + self.ext_entries())
        else:
            var += _SRC_CONV_STUB + stubs(R_ENTRIES) + _SRC_T
        ext = "" if self.conv else self.ext_source()
        return JINJA_ENV.from_string(_SRC + stub + protos + _SRC_V + var).render(
            p=self.p, mac=f"{MAC}<128, 80>", mac_src=render_mac(k=32), CBC=CBC, NQ=NQ, Q=Q, A_OBJ=A_OBJ,
            C_OBJ=C_OBJ, C_BUF=C_BUF, ROFF=ROFF, REC=REC, NREC=NREC, hm=_masks(), silu=_silu(), lut=act_lut.LUT,
            CMD_TILE=CMD_TILE, LREC=LREC, gelu_tmax=_gelu_table()[1],
            **{k: v for k, v in globals().items() if k.startswith("E_")},
            **{k: v for k, v in globals().items() if k.startswith("W_")}) + ext

    # Extension hooks (a subclass adds kernels and core-program phases without changing this design):
    def ext_entries(self) -> list[tuple[str, str, str]]:
        """The extension's kernel entries (name, C parameter list, argument list): the core program calls
        ``{p}_{name}``; the transformer variant's ``ext_source`` defines ``{p}_{name}_impl`` (noinline, non-static), the
        regular variant links empty stubs."""
        return []

    def ext_source(self) -> str:
        """Extra C++ of the transformer variant (namespace-scope names prefixed with self.p; no data-memory objects):
        the entries' impls, appended after the trampolines (not a template)."""
        return ""

    def ext_args(self, t: Tile, c: int, r: int, types: dict) -> list:
        """Extra Worker arguments of core (c, r) (Buffers, Locks, Kernels), passed to ext_phases as env["ext"]."""
        return []

    def ext_phases(self, env: dict) -> None:
        """Extra core-program code (IRON-traced), run after the pre-op slots and before the passes: env holds the
        core's a0, a1, ring, band, cb, scr, ws, the locks af, aa, bf, bb, cf, cc, the kernels by name (``k``) and the
        ext_args (``ext``). It runs in every op of both variants: gate it all with ``range32(ws[W_ATT])`` (command word
        W_ATT is 0 in this module's ops; the extension's ops set it), inside which its ops may reuse other command
        words."""

    def module(self, sequence: object, arg_types: list) -> object:
        i32 = np.dtype[np.int32]
        A_ty = np.ndarray[(self.A_OBJ // 4,), i32]
        B_ty = np.ndarray[(NQ * Q // 4,), i32]
        band_ty = np.ndarray[(self.BAND // 4,), i32]
        C_ty = np.ndarray[(C_BUF // 4,), i32]
        S_ty = np.ndarray[(SCR // 4,), i32]
        H_ty = np.ndarray[(HEAP // 4,), i32]
        ws_ty = np.ndarray[(WS,), i32]
        p = self.p
        k_begin = Kernel(f"{p}_begin", "kernel.o", [B_ty, ws_ty])
        k_init = Kernel(f"{p}_init", "kernel.o", [B_ty, ws_ty, C_ty, S_ty])
        k_resid = Kernel(f"{p}_resid", "kernel.o", [A_ty, A_ty, ws_ty, C_ty])
        k_conv3 = Kernel(f"{p}_conv3", "kernel.o", [A_ty, A_ty, band_ty, B_ty, ws_ty, C_ty])
        k_gemm = Kernel(f"{p}_gemm", "kernel.o", [A_ty, A_ty, B_ty, ws_ty, C_ty])
        k_bskip = Kernel(f"{p}_bskip", "kernel.o", [ws_ty])
        k_epi = Kernel(f"{p}_epi", "kernel.o", [ws_ty, C_ty, S_ty, band_ty])
        k_pre = Kernel(f"{p}_pre", "kernel.o", [B_ty, ws_ty, S_ty, band_ty])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, S_ty, band_ty])
        k_emit = Kernel(f"{p}_emit", "kernel.o", [ws_ty, band_ty, C_ty])
        k_lnbegin = Kernel(f"{p}_lnbegin", "kernel.o", [ws_ty, band_ty, S_ty])
        s = Seq(self)
        mems, shims = s.mems, s.shims
        cores = [[Tile(c, r + 2) for r in range(ROWS)] for c in range(COLS)]
        s.heap = [Buffer(type=H_ty, tile=mems[c], name=f"f3_heap_{c}") for c in range(COLS)]
        # (named flows: every op's design is the same, whatever was built before it in the process)
        s.d_in = [Flow(shims[c], mems[c], src_channel=0, dst_channel=4, name=f"f3_din{c}") for c in range(COLS)]
        s.d_out = [Flow(mems[c], shims[c], src_channel=4, dst_channel=0, name=f"f3_dout{c}") for c in range(COLS)]
        s.b = {r: Flow(shims[2 * r], [cores[c][r] for c in range(COLS)], src_channel=1, name=f"f3_b{r}")
               for r in range(ROWS)}  # (B stream r: row r's cores)
        flows = s.d_in + s.d_out + list(s.b.values())
        for c in range(COLS):
            for r in range(ROWS):
                s.a[c, r] = Flow(mems[c], cores[c][r], src_channel=r, dst_channel=0, name=f"f3_a{c}_{r}")
                s.c[c, r] = Flow(cores[c][r], mems[c], src_channel=0, dst_channel=r, name=f"f3_c{c}_{r}")
                flows += [s.a[c, r], s.c[c, r]]
        dmas, workers = [], []
        for c in range(COLS):  # staging (DDR -> memtile -> core k, per op): 2 slots per core
            for k in range(ROWS):
                s.st_free[c, k] = Lock(mems[c], init=2, name=f"f3_sf_{c}_{k}")
                s.st_full[c, k] = Lock(mems[c], init=0, name=f"f3_ss_{c}_{k}")
        for c in range(COLS):  # the output path: core k -> slot k (S2MM k), slots 0..3 in turn -> shim (MM2S 4)
            o_free = [Lock(mems[c], init=1, name=f"f3_of_{c}_{k}") for k in range(ROWS)]
            o_full = [Lock(mems[c], init=0, name=f"f3_oo_{c}_{k}") for k in range(ROWS)]
            slot = [Seq.htap(SLOT0 + k * C_OBJ, C_OBJ) for k in range(ROWS)]
            # (explicit BD ids, the top of each channel parity's range: aiecc would number them around the device's
            # runtime tasks, and the variants of a design must configure alike)
            dmas.append(TileDma(mems[c], [
                *[DmaChannel(DMAChannelDir.S2MM, s.c[c, k].endpoint(mems[c]),
                             [Bd(s.heap[c], tap=slot[k], acquires=[Acquire(o_free[k])], releases=[Release(o_full[k])],
                                 bd_id=(47 if k % 2 else 23) - k // 2)])
                  for k in range(ROWS)],
                DmaChannel(DMAChannelDir.MM2S, s.d_out[c].endpoint(mems[c]),
                           [Bd(s.heap[c], tap=slot[k], acquires=[Acquire(o_full[k])], releases=[Release(o_free[k])],
                               bd_id=21 - k) for k in range(ROWS)]),
            ]))
        for c in range(COLS):
            for r in range(ROWS):
                t = cores[c][r]
                a_buf = [Buffer(type=A_ty, tile=t, name=f"f3_a{i}_{c}_{r}") for i in range(2)]
                ring = Buffer(type=B_ty, tile=t, name=f"f3_ring_{c}_{r}")
                band = Buffer(type=band_ty, tile=t, name=f"f3_band_{c}_{r}")
                c_buf = Buffer(type=C_ty, tile=t, name=f"f3_c_{c}_{r}")
                scr = Buffer(type=S_ty, tile=t, name=f"f3_scr_{c}_{r}", initial_value=np.zeros(SCR // 4, np.int32))
                ws0 = np.zeros(WS, np.int32)
                ws0[W_CID] = c * ROWS + r
                ws = Buffer(type=ws_ty, tile=t, name=f"f3_ws_{c}_{r}", initial_value=ws0)
                a_free, a_full = Lock(t, init=2, name=f"f3_af_{c}_{r}"), Lock(t, init=0, name=f"f3_aa_{c}_{r}")
                b_free, b_full = Lock(t, init=NQ, name=f"f3_bf_{c}_{r}"), Lock(t, init=0, name=f"f3_bb_{c}_{r}")
                c_free, c_full = Lock(t, init=1, name=f"f3_cf_{c}_{r}"), Lock(t, init=0, name=f"f3_cc_{c}_{r}")
                ring_taps = TensorAccessPattern.full((NQ * Q // 4,)).partition(NQ)
                dmas.append(TileDma(t, [
                    DmaChannel(DMAChannelDir.S2MM, s.a[c, r].endpoint(t),
                               [Bd(b, acquires=[Acquire(a_free)], releases=[Release(a_full)]) for b in a_buf]),
                    DmaChannel(DMAChannelDir.S2MM, s.b[r].endpoint(t),
                               [Bd(ring, tap=tap, acquires=[Acquire(b_free)], releases=[Release(b_full)])
                                for tap in ring_taps]),
                    DmaChannel(DMAChannelDir.MM2S, s.c[c, r].endpoint(t),
                               [Bd(c_buf, tap=TensorAccessPattern((C_BUF // 4,), 0, [C_OBJ // 4], [1]),
                                   acquires=[Acquire(c_full)], releases=[Release(c_free)])]),
                ]))

                def core_fn(a0, a1, ring, band, cb, scr, ws, af, aa, bf, bb, cf, cc, begin, init, resid, conv3, gemm,
                            bskip, epi, pre, fin, emit, lnbegin, *ext):
                    for _ in forever_():
                        bb.acquire(3)
                        begin(ring, ws)
                        bf.release(3)
                        lnbegin(ws, band, scr)
                        for _ in range32(ws[W_NPRE]):  # (pre-op B slots: GN records, SiLU table)
                            bb.acquire(1)
                            pre(ring, ws, scr, band)
                            bf.release(1)
                        self.ext_phases({"a0": a0, "a1": a1, "ring": ring, "band": band, "cb": cb, "scr": scr, "ws": ws,
                                         "af": af, "aa": aa, "bf": bf, "bb": bb, "cf": cf, "cc": cc, "ext": ext,
                                         "k": {"init": init, "resid": resid, "conv3": conv3, "gemm": gemm, "epi": epi}})
                        for _ in range32(ws[W_NPASS]):
                            for _ in range32(ws[W_NT]):
                                cf.acquire(1)
                                bb.acquire(3)
                                init(ring, ws, cb, scr)
                                bf.release(3)
                                for _ in range32(ws[W_NR]):
                                    aa.acquire(1)
                                    resid(a0, a1, ws, cb)
                                    af.release(1)
                                for _ in range32(ws[W_NK3]):
                                    aa.acquire(1)
                                    bb.acquire(3)
                                    conv3(a0, a1, band, ring, ws, cb)
                                    bf.release(3)
                                    af.release(1)
                                for _ in range32(ws[W_NKG]):
                                    aa.acquire(1)
                                    bb.acquire(1)
                                    gemm(a0, a1, ring, ws, cb)
                                    bf.release(1)
                                    af.release(1)
                                for _ in range32(ws[W_NBP]):  # (B padding of the tile: the ring back at 0 / 3)
                                    bb.acquire(1)
                                    bf.release(1)
                                bskip(ws)
                                epi(ws, cb, scr, band)
                                cc.release(1)
                            for _ in range32(ws[W_NFIN]):  # (after a pass: its GN record into band)
                                fin(ws, scr, band)
                        for _ in range32(ws[W_NEMIT]):  # (after the passes: the records, one object)
                            cf.acquire(1)
                            emit(ws, band, cb)
                            cc.release(1)

                types = {"A": A_ty, "B": B_ty, "band": band_ty, "C": C_ty, "S": S_ty, "ws": ws_ty}
                workers.append(Worker(core_fn, [a_buf[0], a_buf[1], ring, band, c_buf, scr, ws, a_free, a_full, b_free,
                                                b_full, c_free, c_full, k_begin, k_init, k_resid, k_conv3, k_gemm,
                                                k_bskip, k_epi, k_pre, k_fin, k_emit, k_lnbegin,
                                                *self.ext_args(t, c, r, types)],
                                      tile=t, stack_size=2048, while_true=False))

        def seq(*args):
            s.tasks, s.waits = [], []
            sequence(args, s)
            for t in s.waits:
                t.await_()
                t.free()
            for t in s.tasks:
                t.free()

        rt = Runtime(seq, arg_types)
        for lk in list(s.st_free.values()) + list(s.st_full.values()):  # (in every op's design: the same configuration)
            rt.add_lock(lk)
        for f in flows:
            rt.add_flow(f)
        for d in dmas:
            rt.add_tile_dma(d)
        return Program(NPU2(), rt, workers=workers).resolve_program()


class Seq:
    """The routes and heaps of the design, with the runtime-sequence helpers ops use."""

    def __init__(self, ovl: Overlay):
        self.ovl = ovl
        self.mems = [Tile(c, 1) for c in range(COLS)]
        self.shims = [Tile(c, 0) for c in range(COLS)]
        self.a, self.c = {}, {}
        self.st_free, self.st_full = {}, {}
        self.tasks: list = []
        self.waits: list = []

    @staticmethod
    def htap(off: int, nbytes: int) -> TensorAccessPattern:
        """nbytes contiguous heap bytes from byte offset off (i32 units)."""
        n = nbytes // 4
        inner = next(d for d in (720, 512, 480, 360, 320, 256, 240, 180, 144, 128, 64, 32) if n % d == 0)
        return TensorAccessPattern((HEAP // 4,), off // 4, [1, 1, n // inner, inner], [0, 0, inner, 1])

    def send(self, c: int, k: int, bds: list[Bd], runs: int = 1) -> None:
        """heap -> core k of column c: the BDs in order, ``runs`` times."""
        t = self.a[c, k].endpoint(self.mems[c]).task(*bds, runs=runs)
        t.start()
        self.tasks.append(t)

    def stage(self, c: int, nsteps: int, base: int = 0) -> None:
        """DDR -> the column's cores' staging slots (core k: 2 x C_OBJ at heap byte base + 2 k C_OBJ) -> core k as 5 A
        objects (SOBJ bytes from the slot: the last A object filled with the bytes after it): nsteps objects per core,
        in [step][core] order."""
        def slot(k: int, i: int) -> int:
            return base + (2 * k + i) * C_OBJ
        # (a task-complete token, awaited at the op's end: every op's design then routes the memtiles' control ports
        # alike, as the ops that await their heap loads do); an odd count: a last task of slot 0
        for slots, runs in (((0, 1), nsteps // 2), ((0,), nsteps % 2)):
            if not runs:
                continue
            t = self.d_in[c].endpoint(self.mems[c]).task(
                *[Bd(self.heap[c], tap=self.htap(slot(k, i), C_OBJ), acquires=[Acquire(self.st_free[c, k])],
                     releases=[Release(self.st_full[c, k])]) for i in slots for k in range(ROWS)], runs=runs,
                wait=True)
            t.start()
            self.waits.append(t)
            for k in range(ROWS):
                t = self.a[c, k].endpoint(self.mems[c]).task(
                    *[Bd(self.heap[c], tap=self.htap(slot(k, i), SOBJ), acquires=[Acquire(self.st_full[c, k])],
                         releases=[Release(self.st_free[c, k])]) for i in slots], runs=runs)
                t.start()
                self.tasks.append(t)

    @staticmethod
    def stage_taps(n: int, steps: list[list[int]], S: int = 1, sstride: int = 0,
                   half: bool = False) -> list[TensorAccessPattern]:
        """The shim walks of a column's staging, in order: per step the S summands' (sstride apart) objects of the 4
        cores (bf16 offsets; groups of m contiguous ones, g apart), one walk per step (per step and summand when the
        summand stride exceeds a BD field). half: an object = the first halves of two consecutive objects (at its
        offset: a stride-2 conv's half tiles), one walk per step and summand."""
        C2, big = C_OBJ // 2, 1 << 21  # (BD strides: 20 bits of words)
        if half:
            st0 = steps[0]
            m = next(m for m in (4, 2, 1) if all(st[i + 1] - st[i] == 2 * C2 for st in steps
                                                 for i in range(ROWS - 1) if (i + 1) % m))
            g = st0[m] - st0[0] if m < ROWS else 0
            assert all(st[i] - st[0] == (i // m) * g + (i % m) * 2 * C2 for st in steps for i in range(ROWS)), steps
            assert 0 <= g < big and 0 <= sstride
            run = run_len(C2 // 2)
            return [TensorAccessPattern((n,), st[0] + k * sstride, [ROWS // m, 2 * m, C2 // 2 // run, run],
                                        [g, C2, run, 1]) for st in steps for k in range(S)]
        st0 = steps[0]
        m = next(m for m in (4, 2, 1) if all(st[i + 1] - st[i] == C2 for st in steps for i in range(ROWS - 1)
                                             if (i + 1) % m))
        g = st0[m] - st0[0] if m < ROWS else 0
        assert all(st[i] - st[0] == (i // m) * g + (i % m) * C2 for st in steps for i in range(ROWS)), steps
        assert 0 <= g < big
        run = run_len(m * C2)
        taps = []
        for st in steps:
            if 0 <= sstride < big or S == 1:
                taps.append(TensorAccessPattern((n,), st[0], [S, ROWS // m, m * C2 // run, run], [sstride, g, run, 1]))
            else:
                taps += [TensorAccessPattern((n,), st[0] + k * sstride, [1, ROWS // m, m * C2 // run, run],
                                             [0, g, run, 1]) for k in range(S)]
        return taps

    def stage_issue(self, src: object, taps: dict[int, list]) -> None:
        """The columns' staging fills, round-robin over the columns, at most 4 queued per shim channel (the oldest
        awaited before a fifth: the drains must be queued before)."""
        pending = {c: [] for c in taps}
        for i in range(max(len(t) for t in taps.values())):
            for c, ts in taps.items():
                if i >= len(ts):
                    continue
                if len(pending[c]) == 4:
                    t = pending[c].pop(0)
                    t.await_()
                    t.free()
                pending[c].append(self.d_in[c].fill(src, tap=ts[i], wait=True, managed=False))
        for ts in pending.values():
            for t in ts:
                t.await_()
                t.free()

    def load(self, c: int, src: object, tap: TensorAccessPattern, heap_tap: TensorAccessPattern,
             iteration: BdIteration | None = None, runs: int = 1) -> object:
        """DDR -> a heap region; returns the memtile task (``await_`` it before reading the region, then ``free``)."""
        t = self.d_in[c].endpoint(self.mems[c]).task(Bd(self.heap[c], tap=heap_tap, iteration=iteration), runs=runs,
                                                     wait=True)
        t.start()
        self.d_in[c].fill(src, tap=tap)
        return t

    def weights(self, r: int, w: object, n: int, off: int, count: int) -> None:
        """count B objects from bf16 offset off of the weight image (n bf16) -> B stream r."""
        self.b[r].fill(w, tap=TensorAccessPattern((n,), off, [1, count, 4, Q // 8], [0, Q // 2, Q // 8, 1]))


def run_len(n: int) -> int:
    """The largest divisor of n (bf16 units) that a DMA dimension walks (<= 1023 words)."""
    return next(d for d in range(min(n, 2046), 0, -1) if n % d == 0 and d % 2 == 0)


def affine2(seq: list[int]) -> tuple[list[int], list[int]] | None:
    """A walk [a][b] with non-negative strides that yields seq (offsets), as (sizes, strides), or None."""
    n = len(seq)
    for b in range(n, 0, -1):
        if n % b:
            continue
        a = n // b
        sb = seq[1] - seq[0] if b > 1 else 0
        sa = seq[b] - seq[0] if a > 1 else 0
        if sa >= 0 and sb >= 0 and all(seq[i * b + j] == seq[0] + i * sa + j * sb for i in range(a) for j in range(b)):
            return [a, b], [sa, sb]
    return None


class OverlayOp(ArtcOperator):
    PROGRAM_MEMORY_VARIANTS = True  # (the variants: the module docstring)

    def __init__(self, ovl: Overlay):
        super().__init__()
        self.ovl = ovl

    def get_kernel_src(self) -> str:
        return self.ovl.source()

    def get_mlir_artifact(self):
        types = [np.ndarray[(s.shape[0],), np.dtype[bfloat16]] for s in self.get_arg_spec()]
        return self.ovl.module(self._sequence, types)


class TiledOp(OverlayOp):
    """An op over T tiles of 128 tokens (or of 64: half tiles) x P passes of 80 outputs: every core takes one tile and
    every kpt-th pass from its first, j. A B stream's cores share j (the stream carries the command and j's passes:
    one contiguous part of the weight image) and hold distinct tiles; a column's tiles are consecutive.
    Output: the C-tile layout Y [P][T][C_OBJ] (bf16 C tiles [8 tg = (rr, g)][2 pairs][10][64]; token (tg, pair, lane) =
    pixel (pair, t R + rr, g + L lane); a half tile or bfp16 panels fill the object's first bytes)."""

    def _tiling(self, T: int, P: int, per_pass: int, TT: int | None = None, t0: int = 0) -> None:
        """T >= 8 tiles, kpt = 32 / T cores per tile: core (c, r) takes tile c tpc + r % tpc and its first pass r // tpc
        (tpc = 4 / kpt tiles per column; B stream r's cores share it). The op's tiles are tiles t0 .. t0 + T - 1 of
        tensors of TT tiles (64x64: two ops, the top and bottom 32 rows of both images)."""
        self.T, self.P, self.per_pass, self.TT, self.t0 = T, P, per_pass, TT or T, t0
        self.kpt = 32 // T
        assert P % self.kpt == 0, f"{P} passes over {self.kpt} cores per tile"
        assert 32 % T == 0 and self.kpt <= 4, f"{T} tiles"
        self.npc = P // self.kpt  # passes per core
        tpc = 4 // self.kpt
        self.jobs = jobs = {(c, r): (c * tpc + r % tpc, r // tpc) for c in range(COLS) for r in range(ROWS)}
        self.col_tiles = {c: sorted({jobs[c, r][0] for r in range(ROWS)}) for c in range(COLS)}
        for ts in self.col_tiles.values():
            assert ts == list(range(ts[0], ts[-1] + 1))
        self.tpc = max(len(ts) for ts in self.col_tiles.values())  # tiles per column

    npre = 0  # (pre-op B slots after the command: tables)
    pre_img: np.ndarray | None = None

    @property
    def part(self) -> int:
        """B objects of a stream's part of the weight image: [cmd 3][pre-op slots][its passes]."""
        return 3 + self.npre + self.npc * self.per_pass

    @property
    def w_bytes(self) -> int:
        return self.kpt * self.part * Q

    def tile_table(self, cmd: np.ndarray) -> None:
        """The cores' tiles (t0 + t) at command words CMD_TILE + core id."""
        for (c, r), (t, _) in self.jobs.items():
            cmd[CMD_TILE + c * ROWS + r] = self.t0 + t

    def _image(self, cmd: np.ndarray, pass_fn: Callable[[int], np.ndarray]) -> np.ndarray:
        """The weight image: per j [cmd][passes j, j + kpt, ..: pass_fn(ps) -> per_pass Q bytes]."""
        out = np.zeros(self.w_bytes, np.uint8)
        for j in range(self.kpt):
            base = j * self.part * Q
            out[base:base + Q] = cmd.view(np.uint8)
            if self.pre_img is not None:
                out[base + 3 * Q:base + 3 * Q + self.pre_img.size] = self.pre_img
            for n in range(self.npc):
                o = base + (3 + self.npre + n * self.per_pass) * Q
                out[o:o + self.per_pass * Q] = pass_fn(j + n * self.kpt)
        return out

    def _weights(self, s: Seq, Wt: object) -> None:
        for r in range(ROWS):  # (B stream r: the part of its cores' first pass)
            s.weights(r, Wt, self.w_bytes // 2, self.jobs[0, r][1] * self.part * Q // 2, self.part)

    @property
    def y_elems(self) -> int:
        return self.P * self.TT * C_OBJ // 2

    def obj(self, ps: int, t: int) -> int:
        """bf16 offset of object (pass, the op's tile t) in a [P][TT][C_OBJ] tensor."""
        return (ps * self.TT + self.t0 + t) * C_OBJ // 2

    def _drain(self, s: Seq, c: int, Y: object) -> None:
        """Column c's C objects (static order: per pass n, cores 0..3) -> Y[pass][tile]."""
        C2 = C_OBJ // 2
        seq = [self.obj(self.jobs[c, r][1] + n * self.kpt, self.jobs[c, r][0]) for n in range(self.npc) for r in range(ROWS)]
        aff = affine2(seq)
        assert aff is not None, f"column {c}: the drain order is no 2-D walk"
        (a, b), (sa, sb) = aff
        run = run_len(C2)
        s.d_out[c].drain(Y, tap=TensorAccessPattern((self.y_elems,), seq[0], [a, b, C2 // run, run], [sa, sb, run, 1]),
                         wait=True)


class ConvL(TiledOp):
    """conv3x3 (padding 1) at a level of width W over the ci chunk [ci0, ci0 + cic) of X: Y = bias + conv (the
    consumer's sum adds a conv's chunks, the residual and temb). Tiles of 128 px = R pair-rows x L groups (pairs: the
    two images, or at 64x64 the halves of one image); T = Hp / R tiles. A column's heap holds its tiles' rows (+ halos).
    X (``panels``): the producer's panel objects [ci / 80][TT][C_OBJ] ([tg][q 10][pair 2]; the heap rows gathered from
    them, zero rows from Z at the image edges); else the conv layout [S HT + 2][L][ci / 8 cb][2 pairs][72 B]
    (``conv_layout``: no halo groups). The op computes the output pair-rows y0 .. y0 + Hp of tensors of HT
    (``Hp_total``) rows. Y: C-tile layout.
    stride 2 (downsampling, in the input level's overlay): Hp output pair-rows of W / 2 pixels; an output tile = R rows
    x L / 2 groups (64 px: half a tile of the output level); the kernel's band walk steps 2 groups per output group,
    the band objects take every other input row. Arguments: X, [Z,] W, Y."""

    def __init__(self, ovl: Overlay, *, Hp: int, ci: int, co: int, ci0: int = 0, cic: int | None = None, stride: int = 1,
                 panels: bool = True, Hp_total: int | None = None, y0: int = 0, up: bool = False):
        """up: the input is the half-resolution level's panel objects, upsampled (nearest) by the heap loads (upper
        row y <- row y / 2, upper group g <- group g / 2, the lane kept)."""
        super().__init__(ovl)
        R = ovl.R
        assert stride in (1, 2)
        cic = ci - ci0 if cic is None else cic
        assert ci % 32 == 0 and ci0 % 32 == 0 and cic % 32 == 0 and ci0 + cic <= ci and co % 80 == 0 and Hp % R == 0
        assert not panels or (ci % 80 == 0 and ci0 % 80 == 0 and cic % 80 == 0)
        self.Hp, self.ci, self.co, self.ci0, self.cic, self.stride, self.panels = Hp, ci, co, ci0, cic, stride, panels
        self.HT = Hp_total or Hp
        assert y0 % R == 0 and self.HT % R == 0
        self.nk3 = 3 * cic // 32
        self._tiling(Hp // R, co // 80, 3 + 3 * self.nk3, self.HT // R, y0 // R)
        self.up = up
        assert not up or (panels and stride == 1)
        self.Ri, self.Li = (2 * R, ovl.L // 2) if up else (R, ovl.L)  # (the input level's tile rows, groups)
        self.TTi = (self.HT // 2 if up else stride * self.HT) // self.Ri  # (the input tensor's tiles)
        self.rb = ovl.L * ci // 8 * 144  # bytes of an X row
        self.rbc = ovl.L * cic // 8 * 144  # of a heap row (the chunk's channels)
        assert cic // 32 <= 64  # (BD iteration)
        self.nrows = stride * self.tpc * R + 3 - stride
        self.cbytes = R * ovl.L // stride * 2560  # (valid bytes of an output object)
        assert self.nrows * self.rbc <= SLOT0, f"{cic} ci: {self.nrows * self.rbc} B of rows"

    @staticmethod
    def max_chunk(ovl: Overlay, Hp: int, ci: int, co: int, stride: int = 1, cap: int = 640, panels: bool = True) -> int:
        """The largest ci chunk (a multiple of 32 (panels: 80) dividing ci, <= cap: bf16 C tiles) that fits."""
        q = 80 if panels else 32
        for n in range(1, ci // q + 1):
            if (ci // q) % n == 0 and ci // n <= cap and (ci // n) % 32 == 0:
                try:
                    ConvL(ovl, Hp=Hp, ci=ci, co=co, cic=ci // n, stride=stride, panels=panels)
                    return ci // n
                except AssertionError:
                    continue
        raise AssertionError("no chunk fits")

    def get_arg_spec(self) -> list[ArgSpec]:
        bf, I, O = torch.bfloat16, ArgDirection.IN, ArgDirection.OUT
        if self.panels:
            return [ArgSpec(I, (self.ci // 80 * self.TTi * C_OBJ // 2,), bf), ArgSpec(I, (self.rbc // 2,), bf),
                    ArgSpec(I, (self.w_bytes // 2,), bf), ArgSpec(O, (self.y_elems,), bf)]
        return [ArgSpec(I, ((self.stride * self.HT + 2) * self.rb // 2,), bf), ArgSpec(I, (self.w_bytes // 2,), bf),
                ArgSpec(O, (self.y_elems,), bf)]

    def pack(self, w: torch.Tensor, b: torch.Tensor | None = None) -> np.ndarray:
        """(co, cic, 3, 3) weights of the chunk, (co,) bias (None: zero) -> the weight image."""
        cmd = np.zeros(Q // 4, np.int32)
        cmd[W_NT], cmd[W_NK3] = self.npc, self.nk3
        cmd[W_L], cmd[W_J], cmd[W_R], cmd[W_S] = self.ovl.L // self.stride, 5, self.ovl.R, self.stride - 1
        R = self.ovl.R  # (band rows per expand: the most dividing R that fit the band buffer)
        cmd[W_RB] = max(n for n in range(1, R + 1) if R % n == 0 and n * (self.ovl.L + 2) * 576 <= BAND)

        def pass_fn(ps: int) -> np.ndarray:
            out = np.zeros(self.per_pass * Q, np.uint8)
            c0 = ps * 80
            if b is not None:
                init = np.zeros(Q // 2, bfloat16)
                init[:80], init[80:160] = b[c0:c0 + 80].float().numpy(), b[c0:c0 + 80].float().numpy()
                out[:Q] = init.view(np.uint8)
            # k-tiles (t, dy): [5 j][3 dx][4 cbc][2 h][8 n][8 k]
            t_ = w[c0:c0 + 80].float().reshape(5, 2, 8, self.cic // 32, CBC, 8, 3, 3).permute(3, 6, 0, 7, 4, 1, 2, 5)
            blk = bfp16_blocks(np.ascontiguousarray(t_.numpy())).reshape(-1)
            out[3 * Q:3 * Q + blk.size] = blk
            return out
        return self._image(cmd, pass_fn)

    def _loads(self, s: Seq, args: tuple) -> list:
        """The columns' heap rows (pair-rows S (t0 + t) R - 1 .. of their tiles, the chunk's channels)."""
        o = self.ovl
        R, L, S = o.R, o.L, self.stride
        rbc, C2, HTi = self.rbc, C_OBJ // 2, S * self.HT
        loads = []
        for c, ts in self.col_tiles.items():
            nr = S * len(ts) * R + 3 - S
            yf = S * (self.t0 + ts[0]) * R - 1
            if not self.panels:  # the conv layout: one walk (its rows -1 .. HTi)
                X = args[0]
                rb2, g2, gc2 = self.rb // 2, self.ci * 9, self.cic * 9
                n_x = (HTi + 2) * rb2
                if self.cic == self.ci:
                    run = run_len(rb2)
                    tap = TensorAccessPattern((n_x,), (yf + 1) * rb2, [1, nr, rb2 // run, run], [0, rb2, run, 1])
                else:
                    run = run_len(gc2)
                    tap = TensorAccessPattern((n_x,), (yf + 1) * rb2 + self.ci0 * 9, [nr, L, gc2 // run, run],
                                              [rb2, g2, run, 1])
                loads.append(s.load(c, X, tap, s.htap(0, nr * rbc)))
                continue
            X, Z = args[0], args[1]
            n_x, nps, cw = self.ci // 80 * self.TTi * C2, self.cic // 80, self.cic // 8 * 36
            Ri, Li = self.Ri, self.Li

            def src(y: int, Ri: int = Ri, Li: int = Li) -> int:  # (the input level's row y: its pass ci0 / 80)
                return (self.ci0 // 80 * self.TTi + y // Ri) * C2 + y % Ri * Li * 720
            rows = []  # (heap row, input row) of the image rows; zero rows from Z
            for h in range(nr):
                y = yf + h
                if not 0 <= y < HTi:
                    run = run_len(rbc // 2)
                    loads.append(s.load(c, Z, TensorAccessPattern((rbc // 2,), 0, [1, 1, rbc // 2 // run, run],
                                                                  [0, 0, run, 1]), s.htap(h * rbc, rbc)))
                else:
                    rows.append((h, y // 2 if self.up else y))
            sets = [[r for r in rows if (yf + r[0]) % 2 == p] for p in (0, 1)] if self.up else [rows]
            for part in sets:  # (up: even / odd upper rows: each lower row twice)
                i = 0
                while i < len(part):  # runs with one heap-row step and one source step
                    n, dh, ds = 1, 0, 0
                    if i + 1 < len(part):
                        dh, ds = part[i + 1][0] - part[i][0], src(part[i + 1][1]) - src(part[i][1])
                    while (i + n < len(part) and part[i + n][0] - part[i + n - 1][0] == dh
                           and src(part[i + n][1]) - src(part[i + n - 1][1]) == ds):
                        n += 1
                    h0, y0_ = part[i]
                    tap = TensorAccessPattern((n_x,), src(y0_), [n, nps, Li, 720], [ds, self.TTi * C2, 720, 1])
                    for gp in ((0, 1) if self.up else (0,)):  # (up: upper groups 2 g + gp <- group g)
                        heap = TensorAccessPattern((HEAP // 4,), h0 * rbc // 4 + gp * cw, [nps, Li, 360],
                                                   [360, (2 if self.up else 1) * cw, 1])
                        loads.append(s.load(c, X, tap, heap, BdIteration(n, dh * rbc // 4) if n > 1 else None, n))
                    i += n
        return loads

    def _sequence(self, args: tuple, s: Seq) -> None:
        Wt, Y = args[-2:]
        o = self.ovl
        R, L, S = o.R, o.L, self.stride
        rbc, nt = self.rbc, self.cic // 32
        loads = self._loads(s, args)
        self._weights(s, Wt)  # (the rings fill while the heaps load)
        for t in loads:
            t.await_()
            t.free()
        for c, ts in self.col_tiles.items():
            for r in range(ROWS):
                yl = S * (self.jobs[c, r][0] - ts[0]) * R  # heap row of the tile's input row -1
                # k-tiles (t, dy): [R rows (every S-th)][L][4 cb][2] of 32 channels t; per dy a BD whose iteration
                # steps t
                gw = self.cic // 8 * 36  # words of a heap group
                bds = []
                for dy in range(3):
                    off = (yl + dy) * rbc // 4
                    tap = (TensorAccessPattern((HEAP // 4,), off, [L, 144], [gw, 1]) if R == 1 else
                           TensorAccessPattern((HEAP // 4,), off, [R, L, 144], [S * rbc // 4, gw, 1]))
                    bds.append(Bd(s.heap[c], tap=tap, iteration=BdIteration(nt, 144)))
                s.send(c, r, bds, nt * self.npc)
            self._drain(s, c, Y)


class GemmL(TiledOp):
    """GEMM over T tiles of 128 tokens over the K chunk [k0, k0 + kc): Y = A W^T + b (the consumer sums the chunks).
    A: the producer's panel objects [K / cin][T][C_OBJ] ([tg][q][pair] panels: a norm op's output); a column's tiles'
    chunks are resident in its heap as [tg][kb][pair]; Y: the C-tile layout, bf16 or (``out_bfp16``) the 160 bfp16
    panels per C tile.
    Arguments: A, W, Y."""

    def __init__(self, ovl: Overlay, *, T: int, K: int, N: int, k0: int = 0, kc: int | None = None,
                 out_bfp16: bool = False, TT: int | None = None, t0: int = 0,
                 geglu: bool = False, cin: int = 80):
        """geglu: the output's N / 2 channels hid gelu(gate) (the weight's rows [hid N / 2 | gate N / 2]), objects of 40
        channels ([tg][5 q][pair] panels); cin: the channels of the input's objects (80, or 40 after a GEGLU)."""
        super().__init__(ovl)
        self.geglu, self.cin = geglu, cin
        kc = K - k0 if kc is None else kc
        assert K % 32 == 0 and k0 % 32 == 0 and kc % 32 == 0 and k0 + kc <= K and N % 80 == 0
        assert K % cin == 0 and k0 % cin == 0 and kc % cin == 0
        self.K, self.N, self.k0, self.kc, self.out_bfp16 = K, N, k0, kc, out_bfp16
        self.nkt = kc // 32
        self._tiling(T, N // 80, 3 + -(-self.nkt // 3) * 3, TT, t0)  # (pass: init 3, k-tiles, padding to 3 k)
        self.npre = 3 if geglu else 0  # (the GELU table, padding: the ring at 0 / 3)
        self.tile = K // 8 * 16 * PANEL  # bytes of a tile's A
        self.tilec = kc // 8 * 16 * PANEL  # of its chunk
        self.cbytes = 160 * PANEL if out_bfp16 else C_OBJ
        assert self.tpc * self.tilec <= SLOT0, f"K chunk {kc}: {self.tpc * self.tilec} B of A"

    @staticmethod
    def max_chunk(ovl: Overlay, T: int, K: int, N: int, cap: int = 640) -> int:
        for n in range(1, K // 32 + 1):
            if (K // 32) % n == 0 and K // n <= cap:
                try:
                    GemmL(ovl, T=T, K=K, N=N, kc=K // n)
                    return K // n
                except AssertionError:
                    continue
        raise AssertionError("no chunk fits")

    def get_arg_spec(self) -> list[ArgSpec]:
        bf, I, O = torch.bfloat16, ArgDirection.IN, ArgDirection.OUT
        n_a = self.K // self.cin * self.TT * C_OBJ // 2
        return [ArgSpec(I, (n_a,), bf), ArgSpec(I, (self.w_bytes // 2,), bf), ArgSpec(O, (self.y_elems,), bf)]

    def pack(self, w: torch.Tensor, b: torch.Tensor | None = None) -> np.ndarray:
        """(N, kc) weights of the chunk, (N,) bias (None: zero) -> per pass init, k-tiles [5 blk][4 kb][2][8 n][9 B],
        padding."""
        cmd = np.zeros(Q // 4, np.int32)
        cmd[W_NT], cmd[W_NKG], cmd[W_NBP], cmd[W_NE] = self.npc, self.nkt, self.per_pass - 3 - self.nkt, self.out_bfp16
        cmd[W_L], cmd[W_J], cmd[W_R] = self.ovl.L, 5, self.ovl.R
        if self.geglu:  # (pass p: hid rows 40 p .., gate rows N / 2 + 40 p ..; the GELU table in the pre-op slots)
            cmd[W_NE], cmd[W_NPRE], cmd[W_PM] = E_GEGLU, self.npre, 2
            self.pre_img = _gelu_table()[0]
            h = self.N // 2
            perm = torch.tensor([40 * (i // 80) + i % 80 if i % 80 < 40 else h + 40 * (i // 80) + i % 80 - 40
                                 for i in range(self.N)])
            w = w[perm]
            b = b[perm] if b is not None else None

        def pass_fn(ps: int) -> np.ndarray:
            out = np.zeros(self.per_pass * Q, np.uint8)
            n0 = ps * 80
            if b is not None:
                init = np.zeros(Q // 2, bfloat16)
                init[:80], init[80:160] = b[n0:n0 + 80].float().numpy(), b[n0:n0 + 80].float().numpy()
                out[:Q] = init.view(np.uint8)
            t = w[n0:n0 + 80].float().reshape(5, 2, 8, self.nkt, 4, 8).permute(3, 0, 4, 1, 2, 5)  # kt blk kb h n k
            blk = bfp16_blocks(np.ascontiguousarray(t.numpy())).reshape(-1)
            out[3 * Q:3 * Q + blk.size] = blk
            return out
        return self._image(cmd, pass_fn)

    def _sequence(self, args: tuple, s: Seq) -> None:
        A, Wt, Y = args
        n_a = self.get_arg_spec()[0].shape[0]
        C2, kw = C_OBJ // 2, self.kc // 8 * 36  # (words of a heap tg row)
        loads = []
        for c, ts in self.col_tiles.items():  # the column's tiles' chunks -> heap [tile][tg][kb][pair]
            # (object (pass, t): [tg][cin / 8 x 144 B] -> heap tg row, kb cin / 8 (pass - k0 / cin))
            ci, gb = self.cin, self.cin * 9  # (bf16 of an object's tg)
            tap = TensorAccessPattern((n_a,), self.obj(self.k0 // ci, ts[0]),
                                      [len(ts), self.kc // ci, 8, gb], [C2, self.TT * C2, gb, 1])
            heap = TensorAccessPattern((HEAP // 4,), 0, [self.kc // ci, 8, gb // 2], [gb // 2, kw, 1])
            loads.append(s.load(c, A, tap, heap, BdIteration(len(ts), self.tilec // 4), len(ts)))
        self._weights(s, Wt)
        for t in loads:
            t.await_()
            t.free()
        for c, ts in self.col_tiles.items():
            for r in range(ROWS):  # k-tile kt: [tg 8][kb 4 x 2 pairs x 72 B]; one BD whose iteration steps the k-tiles
                off = (self.jobs[c, r][0] - ts[0]) * self.tilec // 4
                bd = Bd(s.heap[c], tap=TensorAccessPattern((HEAP // 4,), off, [8, 144], [kw, 1]),
                        iteration=BdIteration(self.nkt, 144))
                s.send(c, r, [bd], self.npc * self.nkt)
            self._drain(s, c, Y)


class SumOp(TiledOp):
    """x = bias + the S summands (bf16 C-tile objects) over T tiles x P passes [+ GN records]. The tiling of TiledOp (a
    core: one tile, every kpt-th pass); per pass: init (the pass's bias), the summands' objects (staged, 5 A objects each), the epilogue's GN sums, the pass's record into
    band (fin); after the passes the core's records as one object (emit).
    X: [S][P][T][C_OBJ] (+ SOBJ - C_OBJ B of slack; ``half_in``: [S][P][2 T][C_OBJ], half tiles, tile t = objects
    2 t, 2 t + 1: a stride-2 conv's output); W: ``pack``; Y: [P][T][C_OBJ]; ST (stats): [COLS][ROWS][C_OBJ]
    (core (c, r): its passes' records, REC B apart; then zero records). Arguments: X, W, Y[, ST]."""

    def __init__(self, ovl: Overlay, *, T: int, P: int, S: int, stats: str | None = None, C: int | None = None,
                 TT: int | None = None, t0: int = 0, st_off: int = 0, n_st: int = 1,
                 stats2: tuple[int, int] | None = None, half_in: bool = False):
        """stats: None, "gn" (GroupNorm records per pass), "ln" (LayerNorm: one record per core, its tile's
        per-token sums over its passes). stats2 (C2, ps0): a second GN record per pass (records 2 n, 2 n + 1): the
        groups of C2 channels, x as passes ps0 .. of them (a concatenation's GroupNorm)."""
        super().__init__(ovl)
        assert stats in (None, "gn", "ln"), stats
        self.ln = stats == "ln"
        self.S, self.stats, self.C, self.st_off, self.n_st = S, bool(stats), C or 80 * P, st_off, n_st
        self.stats2, self.half_in = stats2, half_in
        assert not stats2 or (stats and not self.ln)
        self._tiling(T, P, 3, TT, t0)
        nr = 2 if stats2 else 1
        assert not stats or self.ln or nr * self.npc <= NREC

    def records(self, rset: int = 0) -> list[tuple[int, int]]:
        """(ST object, record) of the cores' records (LayerNorm: one per core); rset 1: the stats2 records."""
        cores = [(c, r) for c in range(COLS) for r in range(ROWS)]
        if self.ln:
            return [(self.st_off + c * ROWS + r, 0) for c, r in cores]
        assert rset == 0 or self.stats2
        k = 2 if self.stats2 else 1
        return [(self.st_off + c * ROWS + r, k * n + rset) for c, r in cores for n in range(self.npc)]

    def get_arg_spec(self) -> list[ArgSpec]:
        bf, I, O = torch.bfloat16, ArgDirection.IN, ArgDirection.OUT
        y = self.y_elems
        spec = [ArgSpec(I, (self.S * y * (2 if self.half_in else 1) + (SOBJ - C_OBJ) // 2,), bf),
                ArgSpec(I, (self.w_bytes // 2,), bf), ArgSpec(O, (y,), bf)]
        if self.stats:  # (n_st ops' records, st_off objects apart)
            spec.append(ArgSpec(O, (self.n_st * COLS * ROWS * C_OBJ // 2,), bf))
        return spec

    def pack(self, bias: torch.Tensor | None = None) -> np.ndarray:
        """bias: (2 pairs, 80 P) or None -> the B image."""
        cmd = np.zeros(Q // 4, np.int32)
        cmd[W_NPASS], cmd[W_NT], cmd[W_NR] = self.npc, 1, 5 * self.S
        if self.ln:
            cmd[W_NE], cmd[W_NEMIT] = E_LNST, 1
        else:
            cmd[W_NE], cmd[W_NFIN], cmd[W_NEMIT] = E_GNST if self.stats else 0, int(self.stats), int(self.stats)
        cmd[W_G] = self.C // 32
        if self.stats2:
            cmd[W_G2], cmd[W_PS0] = self.stats2[0] // 32, self.stats2[1]
        self.tile_table(cmd)

        def pass_fn(ps: int) -> np.ndarray:
            init = np.zeros(3 * Q, np.uint8)
            if bias is not None:
                init[:320] = bias[:, 80 * ps:80 * ps + 80].float().numpy().astype(bfloat16).view(np.uint8).reshape(-1)
            init[1280:1284] = np.array([ps], np.int32).view(np.uint8)
            return init
        return self._image(cmd, pass_fn)

    def _sequence(self, args: tuple, s: Seq) -> None:
        X, Wt, Y = args[:3]
        C2, y2 = C_OBJ // 2, self.y_elems
        h = 2 if self.half_in else 1
        n_x = self.S * h * y2 + (SOBJ - C_OBJ) // 2
        self._weights(s, Wt)
        taps = {}
        for c in range(COLS):
            s.stage(c, self.npc * self.S)
            self._drain(s, c, Y)
            if self.stats:
                run = run_len(C2)
                s.d_out[c].drain(args[3], tap=TensorAccessPattern((self.n_st * COLS * ROWS * C2,),
                                                                  (self.st_off + c * ROWS) * C2,
                                                                  [1, ROWS, C2 // run, run], [0, C2, run, 1]), wait=True)
            taps[c] = s.stage_taps(n_x, [[h * self.obj(self.jobs[c, r][1] + n * self.kpt, self.jobs[c, r][0])
                                          for r in range(ROWS)] for n in range(self.npc)], self.S, h * y2,
                                   self.half_in)
        s.stage_issue(X, taps)


class ApplyOp(TiledOp):
    """GroupNorm apply (+ SiLU) of x (bf16 C-tile objects) -> bfp16 panel objects ([tg][q 10][pair 2] panels: the first
    11520 B of each C_OBJ object). Every core reduces the sum op's GN records (B pre-op slots, 5 per slot) to the
    statistics; per pass the init object brings gamma / beta; x is staged (5 A objects per object, at ROFF in C).
    X: [P][T][C_OBJ]; ST: the sum op's records; W: ``pack``; Y: [P][T][C_OBJ]. Arguments: X, ST, W, Y."""

    def __init__(self, ovl: Overlay, *, T: int, P: int, sums: list, silu: bool = True, eps: float = 1e-5,
                 TT: int | None = None, t0: int = 0, mode: str = "gn"):
        """mode "gn" (GroupNorm (+ silu), the sums' GN records), "ln" (LayerNorm, their LN records), "plain" (x as
        it is: no statistics; sums may be empty)."""
        super().__init__(ovl)
        assert mode in ("gn", "ln", "plain")
        self.mode = mode
        sums = [sm if isinstance(sm, tuple) else (sm, 0) for sm in sums]  # ((sum op, record set))
        self.sums, self.silu, self.eps = sums, silu and mode == "gn", eps
        self.C = (sums[0][0].stats2[0] if sums[0][1] else sums[0][0].C) if sums else 80 * P
        self.recs = [rec for op, k in sums for rec in op.records(k)] if mode != "plain" else []
        self.nrec = len(self.recs)
        per = 2 if mode == "ln" else 5  # (records per B slot)
        self.nsts = -(-self.nrec // per)
        while (self.nsts + (2 if self.silu else 0)) % 3:  # (the ring at 0 / 3 for the 3-slot init objects: padding)
            self.nsts += 1
        self.npre = self.nsts + (2 if self.silu else 0)
        self._tiling(T, P, 3, TT, t0)

    @property
    def part(self) -> int:  # per stream: cmd 3, the table 2, per pass init 3 (the records come from ST)
        return 3 + 2 + self.npc * 3

    @property
    def w_bytes(self) -> int:
        return (self.kpt * self.part + 6) * Q  # (+ padding records: zeros (GN) or tile -1 (LN))

    def get_arg_spec(self) -> list[ArgSpec]:
        bf, I, O = torch.bfloat16, ArgDirection.IN, ArgDirection.OUT
        y = self.y_elems
        n_st = self.sums[0][0].get_arg_spec()[3].shape[0] if self.sums else C_OBJ // 2
        return [ArgSpec(I, (y + (SOBJ - C_OBJ) // 2,), bf), ArgSpec(I, (n_st,), bf),
                ArgSpec(I, (self.w_bytes // 2,), bf), ArgSpec(O, (y,), bf)]

    def pack(self, gamma: torch.Tensor | None = None, beta: torch.Tensor | None = None) -> np.ndarray:
        import struct
        if gamma is None:
            gamma, beta = torch.ones(80 * self.P), torch.zeros(80 * self.P)
        tokens = 64 * self.TT  # (values of a group per image: C / 32 channels x the image's tokens)
        cmd = np.zeros(Q // 4, np.int32)
        cmd[W_NT], cmd[W_NR], cmd[W_ROFF] = self.npc, 5, ROFF
        cmd[W_NE] = {"gn": E_GNSILU if self.silu else E_GN, "ln": E_LN, "plain": E_GN}[self.mode]
        cmd[W_IM] = 1 if self.mode == "gn" else 4
        cmd[W_PM] = 1 if self.mode == "ln" else 0
        cmd[W_NPRE], cmd[W_NSTS], cmd[W_G] = self.npre, self.nsts, self.C // 32
        inv = 1.0 / (self.C // 32 * tokens) if self.mode == "gn" else 1.0 / self.C
        cmd[W_INVN] = struct.unpack("<i", struct.pack("<f", inv))[0]
        cmd[W_EPS] = struct.unpack("<i", struct.pack("<f", self.eps))[0]
        self.tile_table(cmd)
        out = np.zeros(self.w_bytes, np.uint8)
        if self.mode == "ln":  # (padding records: tile -1)
            pad0 = self.kpt * self.part * Q
            for i in range(6 * Q // LREC):
                out[pad0 + i * LREC:pad0 + i * LREC + 4] = np.array([-1], np.int32).view(np.uint8)
        tab = silu_table()
        for j in range(self.kpt):
            base = j * self.part * Q
            out[base:base + Q] = cmd.view(np.uint8)
            out[base + 3 * Q:base + 3 * Q + tab.size] = tab
            for n in range(self.npc):
                ps = j + n * self.kpt
                init = np.zeros(Q, np.uint8)
                gb = np.concatenate([gamma[80 * ps:80 * ps + 80].float().numpy(), beta[80 * ps:80 * ps + 80].float().numpy()])
                init[640:1280] = gb.astype(np.float32).view(np.uint8)
                init[1280:1284] = np.array([ps], np.int32).view(np.uint8)
                o = base + (5 + 3 * n) * Q
                out[o:o + Q] = init
        return out

    def _sequence(self, args: tuple, s: Seq) -> None:
        X, ST, Wt, Y = args
        C2 = C_OBJ // 2
        n_x, n_w = self.y_elems + (SOBJ - C_OBJ) // 2, self.w_bytes // 2
        n_st = self.sums[0][0].get_arg_spec()[3].shape[0] if self.sums else 0
        rec_taps, per = [], 2 if self.mode == "ln" else 5
        rb = LREC if self.mode == "ln" else REC
        for op, k in self.sums if self.mode != "plain" else []:  # (per sum and set: its cores' records)
            objs = sorted({o for o, _ in op.records(k)})  # (any order)
            aff = affine2([o * C2 for o in objs])
            assert aff is not None, "records: no 2-D walk"
            (a, b), (sa, sb) = aff
            npc, nr = (1, 1) if self.mode == "ln" else (op.npc, 2 if op.stats2 else 1)
            assert a * b * npc <= 1023  # (one walk of 4 dims: the 2-D object walk, the records, a record)
            rec_taps.append(TensorAccessPattern((n_st,), objs[0] * C2 + k * rb // 2, [a, b, npc, rb // 2],
                                                [sa, sb, nr * rb // 2, 1]))
        pad = per * self.nsts - self.nrec
        for r in range(ROWS):
            base = self.jobs[0, r][1] * self.part * Q // 2
            s.weights(r, Wt, n_w, base, 3)
            for tap in rec_taps:
                s.b[r].fill(ST, tap=tap)
            if pad:
                s.b[r].fill(Wt, tap=TensorAccessPattern((n_w,), self.kpt * self.part * Q // 2, [1, 1, pad, rb // 2],
                                                        [0, 0, rb // 2, 1]))
            s.weights(r, Wt, n_w, base + 3 * Q // 2 if self.silu else base + 5 * Q // 2,
                      self.part - 3 if self.silu else self.part - 5)
        taps = {}
        for c in range(COLS):
            s.stage(c, self.npc)
            self._drain(s, c, Y)
            taps[c] = s.stage_taps(n_x, [[self.obj(self.jobs[c, r][1] + n * self.kpt, self.jobs[c, r][0])
                                          for r in range(ROWS)] for n in range(self.npc)])
        s.stage_issue(X, taps)

def to_panels(x2: torch.Tensor) -> np.ndarray:
    """(2 pairs, C, Hp, W) -> panel objects [C / 80][T][C_OBJ]: [8 tg = (rr, g)][10 q][2 pairs][8 lanes][9 B], zeros after."""
    _, C, Hp, W = x2.shape
    L = W // 8
    R = 8 // L
    t = x2.float().reshape(2, C // 80, 10, 8, Hp // R, R, 8, L)  # pair ps q ch t rr lane g
    t = t.permute(1, 4, 5, 7, 2, 0, 6, 3)  # ps t rr g q pair lane ch
    blk = bfp16_blocks(np.ascontiguousarray(t.numpy())).reshape(C // 80 * (Hp // R), 160 * PANEL)
    out = np.zeros((blk.shape[0], C_OBJ), np.uint8)
    out[:, :160 * PANEL] = blk
    return out.reshape(-1)


def ctile_values(y: torch.Tensor, bfp: bool = False, cbytes: int = C_OBJ) -> torch.Tensor:
    """Y (C_OBJ objects; their first cbytes valid: bf16 C tiles, half tiles, or bfp16 panels (160 x 72 B)) -> the dense
    bf16-equivalent C-tile layout as float [objects * cbytes / 2 values] (half tiles: consecutive halves = a tile)."""
    raw = y.view(torch.uint8).reshape(-1, C_OBJ)[:, :160 * PANEL if bfp else cbytes]
    if not bfp:
        return raw.contiguous().view(torch.bfloat16).float().reshape(-1)
    raw = raw.numpy().reshape(-1, 160, 8, 9)
    val = raw[..., 1:].view(np.int8).astype(np.float32) * np.ldexp(np.float32(1), raw[..., 0].astype(np.int32) - 133)[..., None]
    return torch.from_numpy(np.ascontiguousarray(val)).reshape(-1)


def pad_pairs(x2: torch.Tensor) -> torch.Tensor:
    """(2, C, H, W) (the two images) -> (2, C, H + 2, W): zero rows -1 and H."""
    return torch.nn.functional.pad(x2.float(), (0, 0, 1, 1))


def conv_layout(xp: torch.Tensor) -> np.ndarray:
    """(2 pairs, C, Hp + 2, W) (rows padded) -> the conv layout [Hp + 2][L groups][C / 8 cb][2 pairs][8 lanes][9 B]
    (lane r of group g: pixel x = g + L r)."""
    _, C, H2, W = xp.shape
    L = W // 8
    t = xp.float().reshape(2, C // 8, 8, H2, 8, L)  # pair cb ch y lane g
    t = t.permute(3, 5, 1, 0, 4, 2)  # y g cb pair lane ch
    return bfp16_blocks(np.ascontiguousarray(t.numpy())).reshape(-1)


def from_ctile(y: torch.Tensor, co: int, Hp: int, W: int) -> torch.Tensor:
    """C-tile layout [co / 80][T][R][L][2 pairs][10 cb][8 lanes][8 ch] -> (2 pairs, co, Hp, W)."""
    L = W // 8
    R = 8 // L
    t = y.float().reshape(co // 80, Hp // R, R, L, 2, 10, 8, 8)  # ps t rr g pair cb lane ch
    return t.permute(4, 0, 5, 7, 1, 2, 6, 3).reshape(2, co, Hp, W)


def to_ctile(x2: torch.Tensor) -> torch.Tensor:
    """(2 pairs, C, Hp, W) -> the C-tile layout [C / 80][T][8 tg = (rr, g)][2 pairs][10 cb][8 lanes][8 ch] bf16."""
    _, C, Hp, W = x2.shape
    L = W // 8
    R = 8 // L
    t = x2.float().reshape(2, C // 80, 10, 8, Hp // R, R, 8, L)  # pair ps cb ch t rr lane g
    return t.permute(1, 4, 5, 7, 0, 2, 6, 3).reshape(-1).bfloat16()


def from_panels(y: torch.Tensor, C: int, Hp: int, W: int) -> torch.Tensor:
    """Panel objects (C_OBJ each; [tg][q][pair][8 lanes][9 B] in the first 11520 B) -> (2 pairs, C, Hp, W) float."""
    L = W // 8
    R = 8 // L
    raw = y.view(torch.uint8).reshape(-1, C_OBJ)[:, :160 * PANEL].numpy().reshape(C // 80, Hp // R, R, L, 10, 2, 8, 9)
    val = raw[..., 1:].view(np.int8).astype(np.float32) * np.ldexp(np.float32(1), raw[..., 0].astype(np.int32) - 133)[..., None]
    t = torch.from_numpy(np.ascontiguousarray(val))  # ps t rr g q pair lane ch
    return t.permute(5, 0, 4, 7, 1, 2, 6, 3).reshape(2, C, Hp, W)
