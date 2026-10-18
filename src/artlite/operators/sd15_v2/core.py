"""The cores' program of sd15_v2's design (``design``): a C++ main reading commands from the row's B stream. The main
loop (``v2_run``), GEMMs and convs are shared code; the other kinds are a role's code (``ROLES``: "a" attention, "b"
the applies; together they do not fit 16 KB of program memory): a core of role x starts at ``v2_main_x``, which puts
its role's entry (``v2_ext_x``) into the ext word and enters ``v2_run``. Each role's ELF holds the shared code first,
at the same addresses in every role's (the role code is what only its main reaches, placed after it): a role switch
rewrites the program memory from there and the ext word (``runtime.role_code``, ``seq.Seq.role``).

A command is one B slot: the header words ``C_*`` and, from word ``C_BIAS``, the op's biases (bf16 [pass][80]); the
core copies it into its workspace and releases the slot. Kinds:
- ``K_NOP``: nothing (the host's parity filler: an op's command in B slot 0, ``seq``);
- ``K_TABLE``: the next B slot's first ``TAB`` bytes -> the activation table (``act_lut``: SiLU, or GELU fed 2 |x|);
- ``K_GEMM``: per token tile (C_NTILE) and pass (C_NPASS: 80 output channels each): C := the pass's bias; per k-tile
  (C_NKT) an A k-tile (``A_OBJ`` objects hold 128 / m of them, in two halves: ``design.R23``) and a B slot -> the MAC
  (m = C_M tokens, 64 or 16: an fp32 C tile, rounded to bf16 once after the residual); a conv (C_CONV, the tile = the
  column's tokens: C_R rows of groups, x = g + groups lane): per chunk of 64 input channels (C_NCH) an A object of its
  input band rows (the halo rows included) and per tap a B slot, in the host's tap order: the taps' A are the object's
  groups or lane-shifted copies (``v2_shift``) of its edge groups, their MAC calls, walks and shift phases host-built
  (``ops.Gemm.conv_words``, ``v2_conv``); then C_NRES A objects of a residual tile (bf16, the C-tile layout) added to
  C; statistics of the result (C_STATS: GroupNorm group sums of the
  op (C_STF: carried over from / to the next op), LayerNorm per-token sums of a tile, each emitted as one output
  chunk after the tile / the op); the output (C_OUT): the bf16 C tile (m x 160 B; O_DS: its groups 0 and 2, a
  stride-2 conv's x, in 2 x 2 chunks), its bfp16 panels (m x 90 B,
  [rb][cb]), their transposes (V^T: [rb][cb] of [8 channels][8 tokens], the exponent per channel), or GEGLU (the
  pass's channels [hid 40 | gate 40] -> hid gelu(gate): bfp16 panels [rb][5]) in ``OCH`` chunks (the last one padded);
- ``K_APPLY`` (role b): GroupNorm (+ SiLU) / LayerNorm of C_NBLK 80-channel blocks of a bf16 C tile (the A
  stream: 3 row blocks per A half, ``ops.Apply``) -> bfp16 panels [rb][cb] per block: the next B slot holds gamma,
  beta (GroupNorm [c16][gamma 16 | beta 16], LayerNorm [gamma | beta], fp32); GroupNorm: the B slots after it C_NREC
  group records (``REC_SLOT`` per slot, summed); LayerNorm: the first A object the tile's per-token records of the
  column's 4 cores;
- ``K_ATTN`` (role a): per unit (C_NUNIT: an image and head of d 160) its queries (an A object: C_NQP query
  panels (1, 2 or 4) as pairs [pair][d-block 20][2], a pair per half; a single panel's pair partner is junk), per key
  group (C_NG of 32
  keys, C_NK of them valid in the last one) a K slot [key pair 2][d-block 20][2] and a V^T slot [d-block pair 10][key
  block 4][2] (8 dims x 8 keys): per query pair S = Q K^T (the GEMM MAC loop, m 16) and the softmax, then P V^T of the
  pairs (m 16 npair); K is given back before V is taken (the next K arrives during P V^T). The softmax: the reference
  max of each query from the first group (never raised: P saturates at 2^63 above it; q pre-scaled by log2 e / sqrt
  d), P = 2^(s - m) to 0.59 bf16 ULP, l = the row sums of the bfp16 P; out: O / l as panels [query panel][d-block].

Statistics records (``REC`` bytes, an output chunk): GroupNorm [image 2][S1, S2][16 groups] fp32 of groups of C_GSIZE
channels counted from C_G0 (the core's first output channel C_CH0 + its passes); LayerNorm [S1 m | S2 m] fp32.
Token panels: (image row, group, image): row block rb of a tile is image rb % 2.
"""

import functools

import numpy as np

from .. import act_lut
from ..base import JINJA_ENV
from ..gemm.mac import render_mac
from ..kernel_source import kernel_prefix

# command header words
(C_KIND, C_M, C_NTILE, C_NPASS, C_NKT, C_NRES, C_OUT, C_STATS, C_GSIZE, C_CH0, C_G0, C_NBLK, C_MODE, C_NREC, C_INVN,
 C_EPS, C_NUNIT, C_NQP, C_NG, C_NK, C_CONV, C_NCH, C_R, C_ROT, C_STF, C_ND, C_NKP) = range(27)
C_GINV = 36  # (GroupNorm: ``group_inv(C_GSIZE)``, the group of channel c being c C_GINV >> 20)
C_MASKS = 256  # (conv: the lane shifts' byte masks, 1152 B: after the biases of at most 4 passes)
C_CALLS = 544  # (conv: the taps' MAC calls, CALL words each: ``ops.Gemm.conv_words``)
CALL, NCALL = 10, 11  # (words of a call, the most calls)
CROSS = (1, 2, 5, 6)  # (the calls of the 4-band convs' corner taps: their walk crosses between cbuf and the object)
C_PATS = C_CALLS + CALL * NCALL  # (the MAC's walks of B and C, a dims_3d_t each: a conv's, a GEMM's, attention's P V^T)
C_WALKS = C_PATS - 7  # (a GEMM's and attention's P V^T: the MAC's walk of A)
C_TAPW = C_PATS + 14  # (conv: per tap of the order its calls and shift phase)
C_PHASE = C_TAPW + 9  # (conv: the shift phases, per output row a v2_shift's offsets)
SH_OBJ = 1 << 31  # (a shift phase's destination offset in the A object, else in cbuf)
ST_KEEP, ST_HOLD = 1, 2  # (C_STF: a GroupNorm record continued from / kept for the next op: one record over several ops)
C_BIAS = 64  # (word of the biases: bf16 [pass][80])
C_GTAB = 64  # (GroupNorm apply: ``group_words``, two per 16 channels)
C_ZERO = 48  # (16 zero words: a residual of zeros)
K_NOP, K_TABLE, K_GEMM, K_APPLY, K_ATTN = 1, 2, 3, 4, 5
O_BF16, O_BFP16, O_GEGLU, O_VT, O_DS = 0, 1, 2, 3, 4
S_NONE, S_GN, S_LN = 0, 1, 2
A_GN, A_GNSILU, A_LN = 1, 2, 3
REC_GN = 256  # (bytes of a GroupNorm record)
REC_SLOT = 22  # (GroupNorm records per B slot)
P = kernel_prefix("v2")  # (the kernels' names: ``kernel_source``)
ROLES = ("a", "b")
WINS = ("gnc", "mac", "out", "acqa", "acqb", "apply", "aqk", "apv", "afin", "aexp", "nomac")  # (nomac: no GEMM / conv MAC, an ablation)

_SRC = r"""
{{ macf }}
{{ pvm }}

struct {{ p }}_st {  // the static chains' positions (A ping-pong, B slots, output ring) and the scratch buffers
  int32_t *a0, *a1, *b0, *b1, *outr, *s0, *s1;
  int apar, bpos, opos;
};

namespace {
constexpr unsigned {{ p }}_L_AF = 48, {{ p }}_L_AA = 49, {{ p }}_L_BF = 50, {{ p }}_L_BB = 51, {{ p }}_L_OF = 52, {{ p }}_L_OO = 53;  // (the core's locks 0..5)

// (take: the next slot, its position advanced; give: release it: two slots may be held at once)
inline __attribute__((always_inline)) int32_t *{{ p }}_take_a({{ p }}_st &s) {
  int32_t *p = s.apar ? s.a1 : s.a0;
  s.apar ^= 1;
{% if win == "acqa" %}
  event0();
{% endif %}
  acquire_greater_equal(p, {{ p }}_L_AA, 1);
{% if win == "acqa" %}
  event1();
{% endif %}
  return p;
}
inline __attribute__((always_inline)) void {{ p }}_give_a({{ p }}_st &, int32_t *p) { release(p, {{ p }}_L_AF, 1); }
inline __attribute__((always_inline)) int32_t *{{ p }}_take_b({{ p }}_st &s) {
  int32_t *p = s.bpos ? s.b1 : s.b0;
  s.bpos ^= 1;
{% if win == "acqb" %}
  event0();
{% endif %}
  acquire_greater_equal(p, {{ p }}_L_BB, 1);
{% if win == "acqb" %}
  event1();
{% endif %}
  return p;
}
inline __attribute__((always_inline)) void {{ p }}_give_b({{ p }}_st &, int32_t *p) { release(p, {{ p }}_L_BF, 1); }

// word w / byte o of the A object at a (its halves in two regions, ``design.R23``)
inline __attribute__((always_inline)) int32_t *{{ p }}_aw(int32_t *a, int w) {
  return a + w + (w >= {{ A_OBJ // 8 }} ? {{ A_HI // 4 }} : 0);
}
inline __attribute__((always_inline)) const uint8_t *{{ p }}_ab(const uint8_t *a, int o) {
  return a + o + (o >= {{ A_OBJ // 2 }} ? {{ A_HI }} : 0);
}

// 64 lanes [8 tokens][8 ch] (fp32) -> per channel the sum over the tokens
inline __attribute__((always_inline)) aie::vector<float, 8> {{ p }}_chsum(aie::accum<accfloat, 64> a) {
  const aie::vector<float, 64> v = a.template to_vector<float>();
  const aie::vector<float, 32> h = aie::add(v.template extract<32>(0), v.template extract<32>(1));
  const aie::vector<float, 16> q = aie::add(h.template extract<16>(0), h.template extract<16>(1));
  return aie::add(q.template extract<8>(0), q.template extract<8>(1));
}

// the first n lanes
inline __attribute__((always_inline)) aie::mask<8> {{ p }}_lo(int n) {
  return aie::mask<8>::from_uint32(n >= 8 ? 0xFFu : (n <= 0 ? 0u : (1u << n) - 1));
}

// 8 values -> 64 lanes [8 tokens][8 ch] (the 8 channels' values repeated per token)
inline __attribute__((always_inline)) aie::vector<bfloat16, 64> {{ p }}_rep(const bfloat16 *c) {
  return aie::load_v<8>(c).grow_replicate<64>();
}

// 8 per-token values -> 64 lanes [token][8 ch]
inline __attribute__((always_inline)) aie::vector<bfloat16, 64> {{ p }}_tok64(const bfloat16 *v) {
  aie::vector<bfloat16, 8> x = aie::load_v<8>(v);
  const auto z1 = aie::interleave_zip(x, x, 1);
  const aie::vector<bfloat16, 16> x16 = aie::concat(z1.first, z1.second);
  const auto z2 = aie::interleave_zip(x16, x16, 1);
  const aie::vector<bfloat16, 32> x32 = aie::concat(z2.first, z2.second);
  const auto z3 = aie::interleave_zip(x32, x32, 1);
  return aie::concat(z3.first, z3.second);
}
}  // namespace

{{ tabu }}


// (an fp32 C tile of nbr 16-row bands: a k-tile from A at a with the walks pa (A, bytes), pb (B, bytes), pl (C))
__attribute__((noinline)) void {{ p }}_ktf(const int32_t *__restrict a, const int32_t *__restrict b, float *__restrict c, int nbr,
                                      const dims_3d_t &pa, const dims_3d_t &pb, const dims_3d_t &pl) {
{% if win == "nomac" %}
  (void)a, (void)b, (void)c, (void)nbr, (void)pa, (void)pb, (void)pl;
{% else %}
  dims_3d_t patA = pa, patB = pb, patL = pl, patS = pl;
  {{ p }}_macf<64, 80>((const bfp16ebs8 *)a, b, (const float __aie_dm_resource_c *)c, (float __aie_dm_resource_c *)c, nbr, patA, patB, patL,
          patS);
{% endif %}
}

// the lane shifts of two groups of 16 panels (1152 B: [kb 8][image 2]): sd one lane down into dd (lane j <- j - 1,
// lane 0 zero), su one lane up into du (lane j <- j + 1, lane 7 zero): 9-byte moves across 64-byte vectors, each
// panel's cut lane cleared by the byte masks mk (a 576-B cycle: down, then up; the conv command's, ``C_MASKS``): the
// 64 bytes before sd and after su land in cut lanes (loaded again, not carried: post-RA pipelined, II 5)
__attribute__((noinline)) void {{ p }}_shift(const uint8_t *__restrict sd, uint8_t *__restrict dd, const uint8_t *__restrict su,
                                        uint8_t *__restrict du, const uint8_t *__restrict mk) {
  int cm = 0;
  for (int k = 0; k < 18; k++) {
    const aie::vector<uint8, 64> v = aie::load_v<64>(sd), pv = aie::load_v<64>(sd - 64);
    const aie::vector<uint8, 64> u = aie::load_v<64>(su), nu = aie::load_v<64>(su + 64);
    aie::store_v(dd, aie::bit_and(aie::shuffle_up_fill(v, pv, 9), aie::load_v<64>(mk)));
    aie::store_v(du, aie::bit_and(aie::shuffle_down_fill(u, nu, 9), aie::load_v<64>(mk + 576)));
    sd += 64;
    dd += 64;
    su += 64;
    du += 64;
    mk = add_2d_ptr(mk, 64 - 576, 8, cm, 64);
  }
}

// fp32 C := the pass's bias over nrb row blocks
__attribute__((noinline)) void {{ p }}_bias_f(const bfloat16 *__restrict bias, float *__restrict c, int nrb) {
  for (int cb = 0; cb < 10; cb++) {
    const aie::vector<float, 16> v16 = aie::accum<accfloat, 16>(aie::load_v<8>(bias + 8 * cb).grow_replicate<16>()).template to_vector<float>();
    float *p = c + 64 * cb;
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int rb = 0; rb < nrb; rb++) {
      aie::store_v(p, v16);
      aie::store_v(p + 16, v16);
      aie::store_v(p + 32, v16);
      aie::store_v(p + 48, v16);
      p += 640;
    }
  }
}

// bf16(cl[i] + residual value i) -> cs[i], i < n (a multiple of 32): the fp32 C tile to bf16 in place with an A
// object's values (rs 32, rj its halves' gap: ``design.A_HI`` / 2), or with zeros (rs, rj 0, r 32 zeros) (cl, cs: the
// same values as distinct restrict parameters, each read before it is written; the residual a MAC operand: a plain load,
// beside cl's two accumulator loads on load slot A; n >= 512: post-RA II 2)
__attribute__((noinline)) void {{ p }}_res_f2b(const bfloat16 *__restrict r, const float *__restrict cl, bfloat16 *__restrict cs, int n,
                                          int rs, int rj) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
  int cr = 0;
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < n / 32; i++) {
    aie::accum<accfloat, 32> s(aie::load_v<32>(cl));
    s = aie::mac(s, aie::load_v<32>(r), one);
    aie::store_v(cs, s.template to_vector<bfloat16>());
    cl += 32;
    r = add_2d_ptr(r, rs + rj, {{ A_OBJ // 128 - 1 }}, cr, rs);
    cs += 32;
  }
}

// the C tile's first n (>= 16) 8x8 sub-blocks (bf16) -> bfp16 panels in place, in their order (panel i at 72 i, read at
// 128 i: src and dst the same buffer, each sub-block read before its panel is written; post-RA II 2)
__attribute__((noinline)) void {{ p }}_bfp(const int32_t *__restrict src_, int32_t *__restrict dst, int n) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *src = (const bfloat16 *)src_;
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)dst);
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < n; i++) {
    out << aie::accum<accfloat, 64>(aie::load_v<64>(src)).template to_vector<bfp16ebs8>();
    src += 64;
  }
}

// the C tile's first n 8x8 sub-blocks (bf16 [8 tokens][8 ch]) -> bfp16 panels of their transposes in place (as {{ p }}_bfp)
__attribute__((noinline)) void {{ p }}_vt(const int32_t *__restrict src_, int32_t *__restrict dst, int n) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *src = (const bfloat16 *)src_;
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)dst);
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < n; i++) {
    out << aie::accum<accfloat, 64>(aie::transpose(aie::load_v<64>(src), 8, 8)).template to_vector<bfp16ebs8>();
    src += 64;
  }
}

// n (a multiple of 8, >= 128) words between 32-B aligned addresses
__attribute__((noinline)) void {{ p }}_copy8(const int32_t *__restrict s, int32_t *__restrict d, int n) {
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < n; i += 8) {
    aie::store_v(d, aie::load_v<8>(s));
    s += 8;
    d += 8;
  }
}

__attribute__((noinline, minsize)) void {{ p }}_zero(int32_t *__restrict d, int n) {
  for (int i = 0; i < n; i += 16) aie::store_v(d + i, aie::zeros<int32, 16>());
}

// GroupNorm sums of a pass: C [rb][cb][64] (bf16, nrb row blocks, rb of image rb % 2) -> rec [img][S1, S2][16
// groups] fp32 (+=): channel ch0 + k of the pass is in group (ch0 + k) / gs - g0
__attribute__((noinline)) void {{ p }}_gnst(const bfloat16 *__restrict c, float *__restrict rec, int nrb, int ch0, int gs,
                                       unsigned gi, int g0) {
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
  const aie::vector<float, 8> z8 = aie::zeros<float, 8>();
  for (int cb = 0; cb < 10; cb++) {
    const unsigned c0 = ch0 + 8 * cb;
    const int ga = c0 * gi >> 20, gb = (c0 + 7) * gi >> 20;
    const aie::mask<8> m = {{ p }}_lo((ga + 1) * gs - c0);
    const aie::mask<16> oa = aie::mask<16>::from_uint32(1u << (ga - g0)), ob = aie::mask<16>::from_uint32(1u << (gb - g0));
    for (int img = 0; img < 2; img++) {
      aie::accum<accfloat, 64> a1 = aie::zeros<accfloat, 64>(), a2 = aie::zeros<accfloat, 64>();
      for (int k = img; k < nrb; k += 2) {
        const aie::vector<bfloat16, 64> v = aie::load_v<64>(c + 640 * k + 64 * cb);
        a1 = aie::mac(a1, v, one);
        a2 = aie::mac(a2, v, v);
      }
      for (int st = 0; st < 2; st++) {
        const aie::vector<float, 8> s = {{ p }}_chsum(st ? a2 : a1);
        const float sa = aie::reduce_add(aie::select(z8, s, m)), sb = aie::reduce_add(aie::select(s, z8, m));
        float *g = rec + 32 * img + 16 * st;
        aie::vector<float, 16> gv = aie::load_v<16>(g);
        gv = aie::select(gv, aie::add(gv, aie::broadcast<float, 16>(sa)), oa);
        gv = aie::select(gv, aie::add(gv, aie::broadcast<float, 16>(sb)), ob);
        aie::store_v(g, gv);
      }
    }
  }
}

// LayerNorm sums of a pass: per token of the tile S1, S2 over the pass's 80 channels added to rec [S1 m | S2 m]
__attribute__((noinline)) void {{ p }}_lnst(const bfloat16 *__restrict c, float *__restrict rec, int nrb) {
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
  const int m = 8 * nrb;
  for (int rb = 0; rb < nrb; rb++) {
    aie::accum<accfloat, 64> a1 = aie::zeros<accfloat, 64>(), a2 = aie::zeros<accfloat, 64>();
    for (int cb = 0; cb < 10; cb++) {
      const aie::vector<bfloat16, 64> v = aie::load_v<64>(c + 640 * rb + 64 * cb);
      a1 = aie::mac(a1, v, one);
      a2 = aie::mac(a2, v, v);
    }
    float *p1 = rec + 8 * rb, *p2 = rec + m + 8 * rb;
    aie::store_v(p1, aie::add(aie::load_v<8>(p1), al::group8<false>(a1.template to_vector<float>())));
    aie::store_v(p2, aie::add(aie::load_v<8>(p2), al::group8<false>(a2.template to_vector<float>())));
  }
}

// GEGLU of a C tile: rb's channels [hid 5 cb | gate 5 cb] -> hid gelu(gate) as bfp16 panels [rb][5] at out (the tile's
// buffer: the panels trail the reads); z, t: scratch (640 bf16 each: gate, 2 |gate| -> gelu(gate)), tab: the GELU table
// (fed 2 |x|); two row blocks per table pass, the gate and hid blocks walked by 2-D pointers
__attribute__((noinline)) void {{ p }}_geglu(const int32_t *__restrict in, int32_t *__restrict out, bfloat16 *__restrict z,
                                        bfloat16 *__restrict t, const uint32_t *__restrict tab, int nrb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *g = (const bfloat16 *)in + 320, *h = (const bfloat16 *)in;
  int cg = 0, ch = 0;
  const aie::vector<bfloat16, 64> hi = aie::broadcast<bfloat16, 64>({{ 2 * gelu_tmax }}f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o((bfp16ebs8 *)out);
  for (int rb = 0; rb < nrb; rb += 2) {
    bfloat16 *zz = z, *tt = t;
    AIE_LOOP_MIN_ITERATION_COUNT(10)
    for (int i = 0; i < 10; i++) {  // (row block rb's gate blocks, then rb + 1's)
      const aie::vector<bfloat16, 64> gv = aie::load_v<64>(g);
      aie::store_v(zz, gv);
      const aie::vector<bfloat16, 64> ga = aie::abs(gv);
      aie::store_v(tt, aie::min(aie::add(ga, ga), hi));
      g = add_2d_ptr(g, 384, 4, cg, 64);
      zz += 64;
      tt += 64;
    }
    {{ lut }}_tabu(z, t, t, t, tab);
    const bfloat16 *tv = t;
    AIE_LOOP_MIN_ITERATION_COUNT(10)
    for (int i = 0; i < 10; i++) {
      o.push(aie::mul(aie::load_v<64>(tv), aie::load_v<64>(h)).template to_vector<bfp16ebs8>());
      h = add_2d_ptr(h, 384, 4, ch, 64);
      tv += 64;
    }
  }
}

// nbytes from src through the output ring, in OCH chunks (the last one padded with what follows src)
__attribute__((noinline, minsize)) void {{ p }}_out({{ p }}_st &s, const int32_t *src, int nbytes) {
{% if win == "out" %}
  event0();
{% endif %}
  for (int off = 0; off < nbytes; off += {{ OCH }}) {
    int32_t *o = s.outr + s.opos * {{ OSL // 4 }};
    acquire_greater_equal(o, {{ p }}_L_OF, 1);
    {{ p }}_copy8(src + off / 4, o, {{ OCH // 4 }});
    release(o, {{ p }}_L_OO, 1);
    s.opos = s.opos == {{ NO - 1 }} ? 0 : s.opos + 1;
  }
{% if win == "out" %}
  event1();
{% endif %}
}

// a conv pass (``K_GEMM`` with C_CONV, the module docstring): per chunk of 64 input channels its A object (band rows
// [groups][kb 8][2] each, whole rows in each half) and its taps in the host's order (``ops.Gemm.tap_order``: rotated by
// one in odd chunks where C_ROT), per tap (C_TAPW: its first call | calls << 10 | phase << 12): its shift phase if any
// (C_PHASE: per output row r < C_R the {{ p }}_shift of two object groups into cbuf, one lane down / up), a B slot and its
// MAC calls (C_CALLS: the first band's source (bit 31: cbuf, else the object; bits 0..15: its byte offset), bit 30:
// the second group's C bands, bits 16..23: the bands; the A walk (a dims_3d_t); its last step per A slot (the walk
// may cross between cbuf and the object: set per chunk); the B and C walks: C_PATS); the row tiles (``ops.Gemm`` rc): an
// A object per dy (bit 15 of its first tap: the next object), its shift phase's destinations in the object (SH_OBJ)
__attribute__((noinline, minsize)) void {{ p }}_conv({{ p }}_st &s, int32_t *ws, float *cf, uint8_t *cbuf) {
  static_assert(sizeof(dims_3d_t) == 28);
  const int nch = ws[{{ C_NCH }}], R = ws[{{ C_R }}], rot = ws[{{ C_ROT }}];
  const dims_3d_t &pb = *(const dims_3d_t *)(ws + {{ C_PATS }}), &pl = *(const dims_3d_t *)(ws + {{ C_PATS + 7 }});
  const uint8_t *mk = (const uint8_t *)(ws + {{ C_MASKS }});
  int32_t *const e0 = ws + {{ C_CALLS }};
  for (int ch = 0; ch < nch; ch++) {
    const uint8_t *a = (const uint8_t *){{ p }}_take_a(s);
    const int k = {{ CALL - 1 }} - s.apar;  // (the calls whose walk crosses between cbuf and the object: their last step)
{% for i in CROSS %}
    e0[{{ CALL * i + 5 }}] = e0[{{ CALL * i }} + k];
{% endfor %}
    for (int t = 0, i = rot & ch; t < 9; t++, i = i == 8 ? 0 : i + 1) {
      const unsigned tw = ws[{{ C_TAPW }} + i];
      if (tw >> 15) {  // (the row tiles: the next dy's object)
        {{ p }}_give_a(s, (int32_t *)a);
        a = (const uint8_t *){{ p }}_take_a(s);
      }
      if ((tw >> 12) & 7) {
        const int32_t *pw = ws + {{ C_PHASE }} + 8 * (((tw >> 12) & 7) - 1);
        for (int r = 0; r < R; r++, pw += 4)
          {{ p }}_shift(a + pw[0], (pw[1] < 0 ? (uint8_t *)a : cbuf) + (pw[1] & 0xFFFF), a + pw[2],
                   (pw[3] < 0 ? (uint8_t *)a : cbuf) + (pw[3] & 0xFFFF), mk);
      }
      int32_t *b = {{ p }}_take_b(s);
      const int32_t *e = e0 + (tw & 0x3FF);
      for (int n = (tw >> 10) & 3; n > 0; n--, e += {{ CALL }}) {
        const unsigned w = e[0];
        {{ p }}_ktf((const int32_t *)((w >> 31 ? cbuf : a) + (w & 0xFFFF)), b, cf + (w & 0x40000000 ? 1280 : 0), (w >> 16) & 0xFF,
               *(const dims_3d_t *)(e + 1), pb, pl);
      }
      {{ p }}_give_b(s, b);
    }
    {{ p }}_give_a(s, (int32_t *)a);
  }
}

__attribute__((noinline, minsize)) void {{ p }}_gemm({{ p }}_st &s, int32_t *cb, int32_t *ws, int32_t *tab) {
  const int m = ws[{{ C_M }}], ntile = ws[{{ C_NTILE }}], npass = ws[{{ C_NPASS }}], nkt = ws[{{ C_NKT }}];
  const int nres = ws[{{ C_NRES }}], om = ws[{{ C_OUT }}], stats = ws[{{ C_STATS }}], stf = ws[{{ C_STF }}];
  const int kpo = m == 64 ? 2 : 8, nrb = m / 8;  // (k-tiles per A object, row blocks)
  // (the MAC's walks; m 64: the band pairs (0, 2), (1, 3) alternate per block, C's halves in two L1 regions)
  const dims_3d_t &pa = *(const dims_3d_t *)(ws + {{ C_WALKS }}), &pb = *(const dims_3d_t *)(ws + {{ C_PATS }});
  const dims_3d_t &pl = *(const dims_3d_t *)(ws + {{ C_PATS + 7 }});  // (the MAC's walks: the command's, ``ops.mac_walks``)
  bfloat16 *c = (bfloat16 *)cb;
  float *cf = (float *)cb;  // (the C tile in fp32, rounded to bf16 in place before the epilogue)
  float *rec = (float *)s.s1;
  if (stats == {{ S_GN }} && !(stf & {{ ST_KEEP }})) {{ p }}_zero(s.s1, {{ REC_GN // 4 }});
  for (int t = 0; t < ntile; t++) {
    if (stats == {{ S_LN }}) {{ p }}_zero(s.s1, 2 * m);
    for (int p = 0; p < npass; p++) {
      {{ p }}_bias_f((const bfloat16 *)(ws + {{ C_BIAS }}) + 80 * p, cf, nrb);
      if (ws[{{ C_CONV }}])
        {{ p }}_conv(s, ws, cf, (uint8_t *)tab);
      else
      for (int kt = 0; kt < nkt; kt += kpo) {
        int32_t *a = {{ p }}_take_a(s);
        const int n = nkt - kt < kpo ? nkt - kt : kpo;
        for (int j = 0; j < n; j++) {
          int32_t *b = {{ p }}_take_b(s);
{% if win == "mac" %}
          event0();
{% endif %}
          {{ p }}_ktf({{ p }}_aw(a, 18 * m * j), b, cf, m / 16, pa, pb, pl);
{% if win == "mac" %}
          event1();
{% endif %}
          {{ p }}_give_b(s, b);
        }
        {{ p }}_give_a(s, a);
      }
      int left = m * 80;  // (the residual tile's bf16 values, with the conversion to bf16)
      for (int i = 0; i < nres; i++) {
        int32_t *a = {{ p }}_take_a(s);
        const int n = left < {{ A_OBJ // 2 }} ? left : {{ A_OBJ // 2 }};
        {{ p }}_res_f2b((const bfloat16 *)a, cf + i * {{ A_OBJ // 2 }}, c + i * {{ A_OBJ // 2 }}, n, 32, {{ A_HI // 2 }});
        left -= n;
        {{ p }}_give_a(s, a);
      }
      if (!nres) {{ p }}_res_f2b((const bfloat16 *)(ws + {{ C_ZERO }}), cf, c, m * 80, 0, 0);  // (zeros: the command's)
      if (stats == {{ S_GN }})
        {{ p }}_gnst(c, rec, nrb, ws[{{ C_CH0 }}] + 80 * p, ws[{{ C_GSIZE }}], ws[{{ C_GINV }}], ws[{{ C_G0 }}]);
      else if (stats == {{ S_LN }})
        {{ p }}_lnst(c, rec, nrb);
      int nbytes = m * 160;
      if (om == {{ O_BFP16 }}) {
        {{ p }}_bfp(cb, cb, nrb * 10);
        nbytes = m * 90;
      } else if (om == {{ O_VT }}) {
        {{ p }}_vt(cb, cb, nrb * 10);
        nbytes = m * 90;
      } else if (om == {{ O_GEGLU }}) {
        {{ p }}_geglu(cb, cb, (bfloat16 *)s.s0, (bfloat16 *)s.s1, (const uint32_t *)tab, nrb);  // (scratch: no statistics in a GEGLU op)
        nbytes = m * 45;
      }
      if (om == {{ O_DS }}) {  // (a stride-2 conv's tile: groups 0 and 2, row blocks 0, 1 and 4, 5)
        {{ p }}_out(s, cb, 2560);
        {{ p }}_out(s, cb + 1280, 2560);
      } else
        {{ p }}_out(s, cb, nbytes);
    }
    if (stats == {{ S_LN }}) {{ p }}_out(s, s.s1, 8 * m);
  }
  if (stats == {{ S_GN }} && !(stf & {{ ST_HOLD }})) {{ p }}_out(s, s.s1, {{ REC_GN }});
}

// rstd = 1 / sqrt(max(s2 inv_n - mean^2, 0) + eps), mean = s1 inv_n (16 lanes; the lanes' invsqrt unrolled)
static __attribute__((noinline, minsize)) void {{ p }}_mrstd(const float *__restrict s1, const float *__restrict s2, const int32_t *ws,
                                               float *__restrict mean, float *__restrict rstd) {
  const aie::vector<float, 16> inv_n = aie::broadcast<int32, 16>(ws[{{ C_INVN }}]).template cast_to<float>();
  const aie::vector<float, 16> eps = aie::broadcast<int32, 16>(ws[{{ C_EPS }}]).template cast_to<float>();
  const aie::vector<float, 16> m1 = al::mulff<16>(aie::load_v<16>(s1), inv_n);
  const aie::vector<float, 16> ve = aie::add(aie::max(aie::sub(al::mulff<16>(aie::load_v<16>(s2), inv_n), al::mulff<16>(m1, m1)),
                                                      aie::zeros<float, 16>()), eps);
  aie::vector<float, 16> r;
  AIE_LOOP_UNROLL_FULL
  for (int k = 0; k < 16; k++) r.set(aie::invsqrt(ve[k]), k);
  aie::store_v(mean, m1);
  aie::store_v(rstd, r);
}

// 16 fp32 s, t -> coef [8 ch: s_hi | s_lo][t_hi | t_lo] bf16 (the first 8 lanes)
static inline __attribute__((always_inline)) void {{ p }}_split_st(aie::vector<float, 16> sv, aie::vector<float, 16> tv,
                                                              bfloat16 *cf) {
  aie::vector<bfloat16, 16> sh, sl, th, tl;
  al::split2<16>(sv, sh, sl);
  al::split2<16>(tv, th, tl);
  aie::store_v(cf, aie::concat(sh.extract<8>(0), sl.extract<8>(0)));
  aie::store_v(cf + 16, aie::concat(th.extract<8>(0), tl.extract<8>(0)));
}

// GroupNorm coefficients of n16 x 16 channels (s = gamma rstd, t = beta - mean s): mr = mean [16] | rstd [16] of the
// record window's groups, tw: per 16 channels two words, the indices in it of their first, middle and last channel's
// groups and the lanes of the first and of the last group (``group_words``: groups of 10 channels span three), gb =
// gamma, beta [c16][16 | 16] fp32 -> coef [c8][s_hi 8 | s_lo 8 | t_hi 8 | t_lo 8] (gb and coef may be one buffer: an
// iteration reads its 128 B before it writes them; pipelined)
static __attribute__((noinline)) void {{ p }}_gnc16(const float *__restrict gb, const float *__restrict mr, bfloat16 *__restrict coef,
                                              const int32_t *__restrict tw, int n16) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  AIE_LOOP_MIN_ITERATION_COUNT(5)
  for (int k = 0; k < n16; k++) {
    const unsigned w = *tw++, wm = *tw++, ia = w & 31, ib = (w >> 5) & 31, ic = (w >> 10) & 31;
    const aie::mask<16> ma = aie::mask<16>::from_uint32(wm & 0xFFFF), mc = aie::mask<16>::from_uint32(wm >> 16);
    const aie::vector<float, 16> rs = aie::select(aie::select(aie::broadcast<float, 16>(mr[16 + ib]), aie::broadcast<float, 16>(mr[16 + ic]), mc),
                                                  aie::broadcast<float, 16>(mr[16 + ia]), ma);
    const aie::vector<float, 16> mu = aie::select(aie::select(aie::broadcast<float, 16>(mr[ib]), aie::broadcast<float, 16>(mr[ic]), mc),
                                                  aie::broadcast<float, 16>(mr[ia]), ma);
    const aie::vector<float, 16> sv = al::mulff<16>(rs, aie::load_v<16>(gb));
    const aie::vector<float, 16> tv = aie::sub(aie::load_v<16>(gb + 16), al::mulff<16>(mu, sv));
    aie::vector<bfloat16, 16> sh, sl, th, tl;
    al::split2<16>(sv, sh, sl);
    al::split2<16>(tv, th, tl);
    aie::store_v(coef, aie::concat(sh.extract<8>(0), sl.extract<8>(0)));
    aie::store_v(coef + 16, aie::concat(th.extract<8>(0), tl.extract<8>(0)));
    aie::store_v(coef + 32, aie::concat(sh.extract<8>(1), sl.extract<8>(1)));
    aie::store_v(coef + 48, aie::concat(th.extract<8>(1), tl.extract<8>(1)));
    gb += 32;
    coef += 64;
  }
}

// st [0, 64) += n records of 64 fp32 at r (pipelined)
static __attribute__((noinline)) void {{ p }}_recsum(const float *__restrict r, float *__restrict st, int n) {
  aie::vector<float, 16> s0 = aie::load_v<16>(st), s1 = aie::load_v<16>(st + 16), s2 = aie::load_v<16>(st + 32),
                         s3 = aie::load_v<16>(st + 48);
  for (int k = 0; k < n; k++) {
    s0 = aie::add(s0, aie::load_v<16>(r));
    s1 = aie::add(s1, aie::load_v<16>(r + 16));
    s2 = aie::add(s2, aie::load_v<16>(r + 32));
    s3 = aie::add(s3, aie::load_v<16>(r + 48));
    r += 64;
  }
  aie::store_v(st, s0);
  aie::store_v(st + 16, s1);
  aie::store_v(st + 32, s2);
  aie::store_v(st + 48, s3);
}

// LayerNorm: gamma, beta per channel -> coef [ch / 8][32]
static __attribute__((noinline, minsize)) void {{ p }}_ch_coef(const float *__restrict gam, const float *__restrict bet,
                                                 bfloat16 *__restrict coef, int nch) {
  for (int c8 = 0; c8 < nch; c8 += 8)
    {{ p }}_split_st(aie::concat(aie::load_v<8>(gam + c8), aie::load_v<8>(gam + c8)), aie::concat(aie::load_v<8>(bet + c8), aie::load_v<8>(bet + c8)),
                coef + 32 * (c8 / 8));
}

// LayerNorm per token: a tile's records of the column's 4 cores ([core][S1 m | S2 m], 144 floats apart) -> tokc [rb][a_hi 8 |
// a_lo 8 | b_hi 8 | b_lo 8] (a = rstd, b = -mean rstd); st: scratch (mean 16 | rstd 16)
static __attribute__((noinline, minsize)) void {{ p }}_ln_tok(const float *__restrict ra, const int32_t *ws, float *__restrict st,
                                                bfloat16 *__restrict tokc, int m) {
  for (int k = 0; k < m; k += 16) {
    aie::vector<float, 16> s1 = aie::zeros<float, 16>(), s2 = aie::zeros<float, 16>();
    for (int q = 0; q < 4; q++) {
      const float *r = ra + 144 * q;
      s1 = aie::add(s1, aie::load_v<16>(r + k));
      s2 = aie::add(s2, aie::load_v<16>(r + m + k));
    }
    aie::store_v(st + 32, s1);
    aie::store_v(st + 48, s2);
    {{ p }}_mrstd(st + 32, st + 48, ws, st, st + 16);
    const aie::vector<float, 16> rs = aie::load_v<16>(st + 16);
    const aie::vector<float, 16> bb = aie::sub(aie::zeros<float, 16>(), al::mulff<16>(aie::load_v<16>(st), rs));
    aie::vector<bfloat16, 16> ah, al_, bh, bl;
    al::split2<16>(rs, ah, al_);
    al::split2<16>(bb, bh, bl);
    for (int h = 0; h < 2; h++) {
      bfloat16 *o = tokc + 32 * (k / 8 + h);
      aie::store_v(o, aie::concat(ah.extract<8>(h), al_.extract<8>(h)));
      aie::store_v(o + 16, aie::concat(bh.extract<8>(h), bl.extract<8>(h)));
    }
  }
}

// one row block x (10 sub-blocks [8 tokens][8 ch], bf16) -> its 10 panels pushed at pan: z = x s + t (GroupNorm,
// mode 1 / 2: then SiLU (the table)) or z = (x a + b) s + t (LayerNorm, mode 3: a, b the row block's token coefficients
// tk [a_hi | a_lo | b_hi | b_lo], expanded once); s, t: the channels' cf [cb][s_hi | s_lo | t_hi | t_lo]; zz: z (x
// itself: each read before it is written), tt: scratch (640). GroupNorm 32 lanes a step (64-lane vectors take two x
// registers each: the coefficients' replicas and x spilled, II 26 per 64)
static __attribute__((noinline)) void {{ p }}_apply_rb(const bfloat16 *__restrict x, bfloat16 *__restrict zz, bfloat16 *__restrict tt,
                                                  const bfloat16 *__restrict cf, const bfloat16 *__restrict tk,
                                                  const uint32_t *__restrict tab, bfp16ebs8 *__restrict pan, int mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *z0 = zz, *t0 = tt;
  if (mode != {{ A_LN }}) {
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f), hi = aie::broadcast<bfloat16, 32>(24.0f);
    int cc = 0;
    AIE_LOOP_MIN_ITERATION_COUNT(20)
    for (int i = 0; i < 20; i++) {  // (post-RA pipelined, II 8)
      const aie::vector<bfloat16, 32> xv = aie::load_v<32>(x);
      const aie::accum<accfloat, 32> ta = aie::mac(aie::mul(aie::load_v<8>(cf + 16).grow_replicate<32>(), one),
                                                   aie::load_v<8>(cf + 24).grow_replicate<32>(), one);
      const aie::vector<bfloat16, 32> z = aie::mac(aie::mac(ta, xv, aie::load_v<8>(cf).grow_replicate<32>()), xv,
                                                   aie::load_v<8>(cf + 8).grow_replicate<32>()).template to_vector<bfloat16>();
      aie::store_v(zz, z);
      aie::store_v(tt, aie::min(aie::abs(z), hi));
      x += 32;
      zz += 32;
      tt += 32;
      cf = add_2d_ptr(cf, 32, 1, cc, 0);
    }
  } else {
    const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
    const aie::vector<bfloat16, 64> ah = {{ p }}_tok64(tk), al_ = {{ p }}_tok64(tk + 8);
    const aie::accum<accfloat, 64> bt = aie::mac(aie::mul({{ p }}_tok64(tk + 16), one), {{ p }}_tok64(tk + 24), one);
    AIE_LOOP_MIN_ITERATION_COUNT(10)
    for (int cb = 0; cb < 10; cb++) {  // (pipelined, II 30)
      const aie::vector<bfloat16, 64> xv = aie::load_v<64>(x);
      const aie::vector<bfloat16, 64> v = aie::mac(aie::mac(bt, xv, ah), xv, al_).template to_vector<bfloat16>();
      const aie::accum<accfloat, 64> ta = aie::mac(aie::mul({{ p }}_rep(cf + 16), one), {{ p }}_rep(cf + 24), one);
      aie::store_v(zz, aie::mac(aie::mac(ta, v, {{ p }}_rep(cf)), v, {{ p }}_rep(cf + 8)).template to_vector<bfloat16>());
      x += 64;
      zz += 64;
      cf += 32;
    }
  }
  zz = z0;
  if (mode == {{ A_GNSILU }}) {{ lut }}_tabu(zz, t0, t0, zz, tab);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o(pan);
  AIE_LOOP_MIN_ITERATION_COUNT(10)
  for (int cb = 0; cb < 10; cb++) o.push(aie::accum<accfloat, 64>(aie::load_v<64>(zz + 64 * cb)).template to_vector<bfp16ebs8>());
}

// GroupNorm / LayerNorm apply (the module docstring): the row blocks in place in their A objects (``ops.Apply``: per
// block and tile nrb + 1 slots of 3 per half, the last one the C-tile object's padding). C holds:
// the coefficients (GN: [img][ch / 8][s_hi s_lo t_hi t_lo] from 0; LN: [ch / 8][..] from 0), LN's token coefficients
// [tile][rb][a_hi a_lo b_hi b_lo] (from 8192 B), t (640 at 16640 B), the output chunk's panels (at 17920 B)
static __attribute__((noinline, minsize)) void {{ p }}_apply_run({{ p }}_st &s, int32_t *cb, const int32_t *ws, const uint32_t *tab) {
  const int m = ws[{{ C_M }}], ntile = ws[{{ C_NTILE }}], nblk = ws[{{ C_NBLK }}], mode = ws[{{ C_MODE }}];
  const int nrb = m / 8, nch = 80 * nblk, ln = mode == {{ A_LN }};
  bfloat16 *coef = (bfloat16 *)cb, *tokc = (bfloat16 *)((uint8_t *)cb + 8192);
  bfloat16 *tt = (bfloat16 *)((uint8_t *)cb + 16640);
  int32_t *pan = (int32_t *)((uint8_t *)cb + 17920);
  int32_t *a = nullptr;
  int h = 0;  // (the slot's place in its A object: half h / 3, row block h % 3)
  for (int blk = 0; blk < nblk; blk++)
    for (int t = 0; t < ntile; t++) {
      for (int rb = 0; rb < nrb; rb++) {
{% if win == "apply" %}
        event0();
{% endif %}
        if (h == 0) {
          if (a) {{ p }}_give_a(s, a);
          a = {{ p }}_take_a(s);
        }
        bfloat16 *x = (bfloat16 *)(a + (h < 3 ? 320 * h : {{ (A_OBJ // 2 + A_HI) // 4 }} + 320 * (h - 3)));
        const bfloat16 *cf = coef + 32 * ((ln ? 0 : nch / 8 * (rb & 1)) + 10 * blk);
        {{ p }}_apply_rb(x, x, tt, cf, tokc + 32 * (nrb * t + rb), tab, (bfp16ebs8 *)((uint8_t *)pan + 720 * (rb & 1)), mode);
{% if win == "apply" %}
        event1();
{% endif %}
        if (rb & 1) {{ p }}_out(s, pan, {{ OCH }});  // (two row blocks: 20 panels, an output chunk)
        h = h == 5 ? 0 : h + 1;
      }
      h = h == 5 ? 0 : h + 1;  // (the tile's padding slot)
    }
  if (a) {{ p }}_give_a(s, a);
}

// LayerNorm apply: the coefficients (gamma, beta per channel, the tiles' per-token ones: their records in one A object,
// 2304 B per tile), then the row blocks (``{{ p }}_apply_run``)
static __attribute__((noinline, minsize)) void {{ p }}_apply_ln({{ p }}_st &s, int32_t *cb, const int32_t *ws, const uint32_t *tab) {
  const int nch = 80 * ws[{{ C_NBLK }}], m = ws[{{ C_M }}];
  int32_t *gb = {{ p }}_take_b(s);
  {{ p }}_ch_coef((const float *)gb, (const float *)gb + nch, (bfloat16 *)cb, nch);
  {{ p }}_give_b(s, gb);
  int32_t *ra = {{ p }}_take_a(s);
  for (int t = 0; t < ws[{{ C_NTILE }}]; t++)
    {{ p }}_ln_tok((const float *){{ p }}_aw(ra, 576 * t), ws, (float *)s.s1, (bfloat16 *)((uint8_t *)cb + 8192) + 4 * m * t, m);
  {{ p }}_give_a(s, ra);
  {{ p }}_apply_run(s, cb, ws, tab);
}

// GroupNorm (+ SiLU) apply: gamma, beta (the B slot [c16][gamma 16 | beta 16] fp32, parked in the coefficients'
// second image), then the records (C_NREC in B slots of REC_SLOT, taken one at a time: two B slots, none may be held
// while another cycles) summed into the scratch -> mean / rstd per image (st + 64) -> the coefficients (the second
// image's in place of what it parks), the row blocks
static __attribute__((noinline, minsize)) void {{ p }}_apply_gn({{ p }}_st &s, int32_t *cb, const int32_t *ws, const uint32_t *tab) {
  const int nch = 80 * ws[{{ C_NBLK }}];
  float *st = (float *)s.s1, *gp = (float *)cb + 2 * nch;  // (gp: the second image's coefficients)
{% if win == "gnc" %}
  event0();
{% endif %}
  int32_t *gb = {{ p }}_take_b(s);
  {{ p }}_copy8(gb, (int32_t *)gp, 2 * nch);
  {{ p }}_give_b(s, gb);
  {{ p }}_zero(s.s1, 64);
  for (int left = ws[{{ C_NREC }}]; left > 0; left -= {{ REC_SLOT }}) {
    const float *rb_ = (const float *){{ p }}_take_b(s);
    {{ p }}_recsum(rb_, st, left < {{ REC_SLOT }} ? left : {{ REC_SLOT }});
    {{ p }}_give_b(s, (int32_t *)rb_);
  }
  for (int img = 0; img < 2; img++) {
    {{ p }}_mrstd(st + 32 * img, st + 32 * img + 16, ws, st + 64 + 32 * img, st + 80 + 32 * img);
    {{ p }}_gnc16(gp, st + 64 + 32 * img, (bfloat16 *)cb + 4 * nch * img, ws + {{ C_GTAB }}, nch / 16);
  }
{% if win == "gnc" %}
  event1();
{% endif %}
  {{ p }}_apply_run(s, cb, ws, tab);
}

// 2^y of 32 lanes, A = y + 192 (an fp32 accumulator): the LLM prefill attentions' exp2 (0.59 bf16 ULP; vexp2 of n + 64,
// exact for a non-negative integer, the 2^-64 in the constants; 2^f = 1 + c1 f + f^2 (c2 + c3 f)), y clamped to [-63,
// 63] in bf16 (v = bf16(A) to [129, 255], f to [-1/2, 1/2]: fp32 min / max are emulated and keep the loop from
// pipelining): 2^-63 for masked keys, a score past the reference by 2^63 saturates
inline __attribute__((always_inline)) aie::accum<accfloat, 32> {{ p }}_exp2(aie::accum<accfloat, 32> A) {
  const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
  const aie::vector<bfloat16, 32> v0 = A.template to_vector<bfloat16>();
  const aie::vector<bfloat16, 32> f = aie::min(aie::max(aie::msc(A, v0, one).template to_vector<bfloat16>(),
                                                        aie::broadcast<bfloat16, 32>(-0.5f)), aie::broadcast<bfloat16, 32>(0.5f));
  const aie::vector<bfloat16, 32> v = aie::min(aie::max(v0, aie::broadcast<bfloat16, 32>(129.0f)), aie::broadcast<bfloat16, 32>(255.0f));
  const aie::vector<bfloat16, 32> P = al::exp2_acc<32>(aie::mac(aie::mul(v, one), aie::broadcast<bfloat16, 32>(-128.0f), one));
  const aie::vector<bfloat16, 32> hi = aie::mac(aie::mul(f, aie::broadcast<bfloat16, 32>(0.0625f * 0x1p-64f)),
                                                aie::broadcast<bfloat16, 32>(0.2412109375f * 0x1p-64f), one)
                                           .template to_vector<bfloat16>();
  const aie::vector<bfloat16, 32> pf = aie::mul(P, f).template to_vector<bfloat16>();
  const aie::vector<bfloat16, 32> pf2 = aie::mul(pf, f).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> r = aie::mul(P, aie::broadcast<bfloat16, 32>(0x1p-64f));
  r = aie::mac(r, pf, aie::broadcast<bfloat16, 32>(0.69140625f * 0x1p-64f));
  return aie::mac(r, pf2, hi);
}

// P = 2^(S - m) in place of 4 key panels ([q panel: QS floats apart][key panel 4 (of the group's, 64 floats apart)][64],
// 32 lanes per step: A = S + mb, mb [q panel][8 q][8]); sl, ss: S as distinct restrict parameters (each read before it
// is written: the loop pipelines, II 23; a runtime trip count or walk leaves it unpipelined)
static __attribute__((noinline)) void {{ p }}_pexp(const float *__restrict sl, float *__restrict ss, const float *__restrict mb) {
{% if win == "aexp" %}
  event0();
{% endif %}
  int cl = 0, cs = 0;
  for (int t = 0; t < 16; t++) {
    aie::accum<accfloat, 32> A(aie::load_v<32>(sl));
    A = aie::add(A, aie::load_v<32>(mb + 64 * (t >> 3) + 32 * (t & 1)));
    aie::store_v(ss, {{ p }}_exp2(A).template to_vector<float>());
    sl = add_2d_ptr(sl, {{ QS - 224 }}, 7, cl, 32);
    ss = add_2d_ptr(ss, {{ QS - 224 }}, 7, cs, 32);
  }
{% if win == "aexp" %}
  event1();
{% endif %}
}

// the softmax's rare steps (``{{ p }}_attn_sm``): keys from nk on masked (S = -2^14), the first group's reference max (per
// query the max of its group's scores in bf16: a reference, not the exact max; mb = 192 - it, [q panel][8 q][8])
static __attribute__((noinline, minsize)) void {{ p }}_attn_mm(float *__restrict sm, float *__restrict mb, int first, int nk, int nkp) {
  if (nk < 8 * nkp)
    for (int kp = 0; kp < nkp; kp++) {  // (32 lanes [4 q][8 keys]: keys j < nk - 8 kp kept)
      const int v = nk - 8 * kp;
      const unsigned lo = v >= 8 ? 0xFFu : (v <= 0 ? 0u : (1u << v) - 1);
      const aie::mask<32> mk = aie::mask<32>::from_uint32(lo * 0x01010101u);
      for (int h = 0; h < 4; h++) {  // (q panel h / 2, its half h % 2)
        float *p = sm + {{ QS }} * (h >> 1) + 64 * kp + 32 * (h & 1);
        aie::store_v(p, aie::select(aie::broadcast<float, 32>(-16384.0f), aie::load_v<32>(p), mk));
      }
    }
  if (first)
    for (int qp = 0; qp < 2; qp++) {
      const float *p = sm + {{ QS }} * qp;
      aie::vector<bfloat16, 64> m = aie::accum<accfloat, 64>(aie::load_v<64>(p)).template to_vector<bfloat16>();
      for (int kp = 1; kp < nkp; kp++)
        m = aie::max(m, aie::accum<accfloat, 64>(aie::load_v<64>(p + 64 * kp)).template to_vector<bfloat16>());
      auto [e1, o1] = aie::interleave_unzip(m.template extract<32>(0), m.template extract<32>(1), 1);
      const aie::vector<bfloat16, 32> m1 = aie::max(e1, o1);
      auto [e2, o2] = aie::interleave_unzip(m1.template extract<16>(0), m1.template extract<16>(1), 1);
      const aie::vector<bfloat16, 16> m2 = aie::max(e2, o2);
      auto [e3, o3] = aie::interleave_unzip(m2.template extract<8>(0), m2.template extract<8>(1), 1);
      const aie::vector<float, 8> mx = aie::accum<accfloat, 8>(aie::max(e3, o3)).template to_vector<float>();
      aie::store_v(mb + 64 * qp, al::bcast8(aie::sub(aie::broadcast<float, 8>(192.0f), mx)));
    }
}

// the softmax of a query pair and key group (``K_ATTN``): S [2 q panels, QS floats apart][nkp key panels][64] fp32 (keys
// from nk on: -2^14), the reference max (first group: mb = 192 - max per query, [2][8 q][8]), P = 2^(S - m)
// (``{{ p }}_pexp`` per 4 key panels) -> pp (bfp16 panels [key panel][q panel]: the P V^T MAC's A order), l [2][8 q][8] += the row sums of
// those panels (a MAC with ones: the P that multiplies V)
static __attribute__((noinline)) void {{ p }}_attn_sm(float *__restrict sm, float *__restrict mb, float *__restrict l,
                                                 bfp16ebs8 *__restrict pp, int first, int nk, int nkp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (nk < 8 * nkp || first) {{ p }}_attn_mm(sm, mb, first, nk, nkp);
  for (int k = 0; k < nkp; k += 4) {{ p }}_pexp(sm + 64 * k, sm + 64 * k, mb);
  const al::BV ones = aie::accum<accfloat, 64>(aie::broadcast<bfloat16, 64>(1.0f)).template to_vector<bfp16ebs8>();
  al::MM8 a0(aie::accum<accfloat, 64>(aie::load_v<64>(l))), a1(aie::accum<accfloat, 64>(aie::load_v<64>(l + 64)));
  al::BOut<> po(pp);
  const float *s0 = sm, *s1 = sm + {{ QS }};
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int kp = 0; kp < nkp; kp++) {
    const al::BV p0 = aie::accum<accfloat, 64>(aie::load_v<64>(s0)).template to_vector<bfp16ebs8>();
    const al::BV p1 = aie::accum<accfloat, 64>(aie::load_v<64>(s1)).template to_vector<bfp16ebs8>();
    po.push(p0);
    po.push(p1);
    a0.mac(p0, aie::op_transpose(ones));
    a1.mac(p1, aie::op_transpose(ones));
    s0 += 64;
    s1 += 64;
  }
  aie::store_v(l, a0.template to_vector<float>());
  aie::store_v(l + 64, a1.template to_vector<float>());
}

// a query pair and key group: S := Q K^T (16 queries, 16 nkb keys, k 8 nd; Q [d-block nd][2 panels], K [key block nkb][d-block
// nd][2 panels]) as S [q panel: QS floats apart][key panel 2 nkb][64]: per block of 16 keys four accumulators [q panel][key
// panel] over the d-blocks (post-RA pipelined, II 4)
static __attribute__((noinline)) void {{ p }}_attn_qk(const int32_t *__restrict q, const int32_t *__restrict k, float *__restrict sm,
                                                 int nkb, int nd) {
  using MM = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8>;
  aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> ks((const bfp16ebs8 *)k);
  for (int blk = 0; blk < nkb; blk++) {
    aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a> qs((const bfp16ebs8 *)q);
    aie::block_vector<bfp16ebs8, 64> A0 = qs.pop(), A1 = qs.pop(), B0 = ks.pop(), B1 = ks.pop();
    MM c0, c1, c2, c3;
    c0.mul(A0, aie::op_transpose(B0));
    c1.mul(A0, aie::op_transpose(B1));
    c2.mul(A1, aie::op_transpose(B0));
    c3.mul(A1, aie::op_transpose(B1));
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(9)
    for (int d = 1; d < nd; d++) {
      A0 = qs.pop();
      A1 = qs.pop();
      B0 = ks.pop();
      B1 = ks.pop();
      c0.mac(A0, aie::op_transpose(B0));
      c1.mac(A0, aie::op_transpose(B1));
      c2.mac(A1, aie::op_transpose(B0));
      c3.mac(A1, aie::op_transpose(B1));
    }
    float *o = sm + 128 * blk;
    aie::store_v(o, c0.template to_vector<float>());
    aie::store_v(o + 64, c1.template to_vector<float>());
    aie::store_v(o + {{ QS }}, c2.template to_vector<float>());
    aie::store_v(o + {{ QS + 64 }}, c3.template to_vector<float>());
  }
}

// O += P V^T of d 160 (m 16 npair, n 160, k 32)
static __attribute__((noinline)) void {{ p }}_attn_pv(const bfp16ebs8 *__restrict pp, const int32_t *__restrict v,
                                                 float *__restrict o, int npair) {
  {{ p }}_pvm<32, 160>(pp, v, (const float __aie_dm_resource_c *)o, (float __aie_dm_resource_c *)o, npair);
}

// O / l -> bfp16 panels [qp][db < nd] at out (O's buffer: panel i at 72 i, read at 256 i): O rounded to bf16, times 1 / l
// as bf16 hi + lo (pipelined)
static __attribute__((noinline)) void {{ p }}_attn_fin(const float *__restrict o, const float *__restrict l, int nqp,
                                                  bfp16ebs8 *__restrict out, int nd) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> po(out);
  for (int qp = 0; qp < nqp; qp++) {
    aie::vector<bfloat16, 64> rh, rl;
    al::split2<64>(al::bcast8(aie::inv(aie::filter_even(aie::filter_even(aie::filter_even(aie::load_v<64>(l + 64 * qp)))))), rh, rl);
    AIE_LOOP_MIN_ITERATION_COUNT(10)
    for (int db = 0; db < nd; db++) {
      const aie::vector<bfloat16, 64> ob = aie::accum<accfloat, 64>(aie::load_v<64>(o)).template to_vector<bfloat16>();
      po.push(aie::mac(aie::mul(ob, rh), ob, rl).template to_vector<bfp16ebs8>());
      o += 64;
    }
  }
}

// K_ATTN (the module docstring): O in the C buffer (fp32 [qp][nd][64]), S of a pair in the Q object's halves' tails
// (after the pair, from the next 64-B boundary: one q panel's per half), the pairs' P (d 160: in s1, 576 B each; d 80: pair 0's in s0, pair 1's in s1),
// 192 - m and l ([qp][8 q][8] each) in the workspace after the header
static __attribute__((noinline, minsize)) void {{ p }}_attn({{ p }}_st &s, int32_t *cb, int32_t *ws) {
  const int nunit = ws[{{ C_NUNIT }}], nqp = ws[{{ C_NQP }}], ng = ws[{{ C_NG }}], nkl = ws[{{ C_NK }}];
  const int nd = ws[{{ C_ND }}], nkp = ws[{{ C_NKP }}], d160 = nd == 20;
  const int npair = (nqp + 1) / 2;
  float *o = (float *)cb, *mb = (float *)(ws + 64), *l = mb + 256;
  // (P V^T at d 80: the GEMM MAC of one band, its walks the command's)
  const dims_3d_t &pa = *(const dims_3d_t *)(ws + {{ C_WALKS }}), &pb = *(const dims_3d_t *)(ws + {{ C_PATS }});
  const dims_3d_t &pl = *(const dims_3d_t *)(ws + {{ C_PATS + 7 }});
  for (int u = 0; u < nunit; u++) {
    int32_t *q = {{ p }}_take_a(s);
    {{ p }}_zero(cb, 128 * nd * npair);
    {{ p }}_zero((int32_t *)l, 128 * npair);
    for (int g = 0; g < ng; g++) {  // (K given back before V is taken: the next K arrives during the PVs)
      int32_t *k = {{ p }}_take_b(s);
{% if win == "aqk" %}
      event0();
{% endif %}
      for (int pr = 0; pr < npair; pr++) {
        int32_t *qa = {{ p }}_aw(q, {{ A_OBJ // 8 }} * pr);
        float *sm = (float *)(q + (36 * nd + 15) / 16 * 16);  // (64-B aligned: the 2048-bit stores' unit)
        bfp16ebs8 *pp = (bfp16ebs8 *)(d160 ? s.s1 + {{ 8 * 18 }} * pr : (pr ? s.s1 : s.s0));
        {{ p }}_attn_qk(qa, k, sm, nkp / 2, nd);
        {{ p }}_attn_sm(sm, mb + 128 * pr, l + 128 * pr, pp, g == 0, g == ng - 1 ? nkl : 8 * nkp, nkp);
      }
{% if win == "aqk" %}
      event1();
{% endif %}
      {{ p }}_give_b(s, k);
      int32_t *v = {{ p }}_take_b(s);
{% if win == "apv" %}
      event0();
{% endif %}
      if (d160)
        {{ p }}_attn_pv((const bfp16ebs8 *)s.s1, v, o, npair);
      else
        for (int pr = 0; pr < npair; pr++)
          {{ p }}_ktf(pr ? s.s1 : s.s0, v, o + 1280 * pr, 1, pa, pb, pl);
{% if win == "apv" %}
      event1();
{% endif %}
      {{ p }}_give_b(s, v);
    }
    {{ p }}_give_a(s, q);
{% if win == "afin" %}
    event0();
{% endif %}
    {{ p }}_attn_fin(o, l, nqp, (bfp16ebs8 *)o, nd);
{% if win == "afin" %}
    event1();
{% endif %}
    {{ p }}_out(s, cb, 72 * nd * nqp);
  }
}

// the cores' main loop (shared): commands from the B stream; GEMMs / convs and tables here, the other kinds by the role's
// code through its entry in ext (a role switch rewrites it while the core waits for a command: ``seq.Seq.role``)
__attribute__((noinline, minsize)) void {{ p }}_run(int32_t *r23, int32_t *outr, int32_t *cb, int32_t *ws, int32_t *tab,
                                                const int32_t *ext) {
  {{ p }}_st s{r23 + {{ R23.a0 // 4 }}, r23 + {{ R23.a1 // 4 }}, r23 + {{ R23.b0 // 4 }}, r23 + {{ R23.b1 // 4 }}, outr,
          r23 + {{ R23.s0 // 4 }}, r23 + {{ R23.s1 // 4 }}, 0, 0, 0};
  for (;;) {
    int32_t *b = {{ p }}_take_b(s);
    {{ p }}_copy8(b, ws, {{ WS // 4 }});
    {{ p }}_give_b(s, b);
    const int kind = ws[{{ C_KIND }}];
    if (kind == {{ K_GEMM }}) {
      {{ p }}_gemm(s, cb, ws, tab);
    } else if (kind == {{ K_TABLE }}) {
      int32_t *t = {{ p }}_take_b(s);
      {{ p }}_copy8(t, tab, {{ TAB // 4 }});
      {{ p }}_give_b(s, t);
    } else if (kind != {{ K_NOP }}) {
      ((void (*)({{ p }}_st &, int32_t *, int32_t *, int32_t *))(intptr_t)*ext)(s, cb, ws, tab);
    }
  }
}

// the roles' entries (after every shared definition)
static __attribute__((noinline, minsize)) void {{ p }}_ext_a({{ p }}_st &s, int32_t *cb, int32_t *ws, int32_t *) {
  {{ p }}_attn(s, cb, ws);
}

static __attribute__((noinline, minsize)) void {{ p }}_ext_b({{ p }}_st &s, int32_t *cb, int32_t *ws, int32_t *tab) {
  if (ws[{{ C_MODE }}] == {{ A_LN }})
    {{ p }}_apply_ln(s, cb, ws, (const uint32_t *)tab);
  else
    {{ p }}_apply_gn(s, cb, ws, (const uint32_t *)tab);
}

extern "C" {

{% for role in roles %}
// the start of a core of role {{ role }}
__attribute__((noinline, minsize)) void {{ p }}_main_{{ role }}(int32_t *r23, int32_t *outr, int32_t *cb, int32_t *ws, int32_t *tab,
                                                          int32_t *ext) {
  *ext = (int32_t)(intptr_t)&{{ p }}_ext_{{ role }};
  {{ p }}_run(r23, outr, cb, ws, tab, ext);
}
{% endfor %}

}  // extern "C"
"""


def group_words(ch0: int, nch: int, gsize: int, g0: int) -> np.ndarray:
    """C_GTAB of a GroupNorm apply core (channels [ch0, ch0 + nch) in groups of gsize >= 8, its record window from group
    g0): per 16 channels from ch0 the window indices of their first, middle and last channel's groups (bits 0..4, 5..9,
    10..14), then the lanes of the first group and of the last (masks, bits 0..15, 16..31: the middle group's lanes are
    the rest)."""
    assert gsize >= 8 and nch % 16 == 0
    out = []
    for c in range(ch0, ch0 + nch, 16):
        ga, gm, gc = c // gsize, (c + 8) // gsize, (c + 15) // gsize
        grp = [(c + j) // gsize for j in range(16)]
        assert len(set(grp)) <= 3 and all(0 <= g - g0 < 16 for g in (ga, gm, gc))
        gm = next((g for g in grp if g not in (ga, gc)), ga)
        ma = sum(1 << j for j in range(16) if grp[j] == ga)
        mc = sum(1 << j for j in range(16) if grp[j] == gc and gc != ga)
        out += [(ga - g0) | (gm - g0) << 5 | (gc - g0) << 10, ma | mc << 16]
    return np.array(out, np.uint32).view(np.int32)


def group_inv(gsize: int) -> int:
    """C_GINV: ceil(2^20 / gsize), exact as c // gsize for every channel index c < 4096."""
    gi = -(-(1 << 20) // gsize)
    assert all(c * gi >> 20 == c // gsize for c in range(4096)), gsize
    return gi


@functools.cache
def shift_masks() -> bytes:
    """``v2_shift``'s byte masks (``C_MASKS``): per byte of a 576-byte cycle (8 panels) 0xFF where kept after a shift
    down (lane 0 of each panel cleared), then up (lane 7 cleared)."""
    return bytes(0 if (i % 72 < 9 if d == 0 else i % 72 >= 63) else 0xFF for d in range(2) for i in range(576))


def source(*, win: str = "") -> str:
    from . import design as d
    assert win in ("",) + WINS, win
    return JINJA_ENV.from_string(_SRC).render(
        p=P, roles=ROLES, macf=render_mac(name=f"{P}_macf", k=d.KT, f32=True, nbr_arg=True, pats=True, c2_bank="b"),
        pvm=render_mac(name=f"{P}_pvm", k=32, f32=True, nbr_arg=True), Q=d.Q,
        tabu=act_lut.SILU.render_unrolled(640), lut=act_lut.LUT,
        gelu_tmax=act_lut.GELU.t_max, win=win, CALL=CALL, CROSS=CROSS, A_OBJ=d.A_OBJ, R23=d.R23, A_HI=d.A_HI, QS=(d.A_OBJ // 2 + d.A_HI) // 4, OCH=d.OCH, OSL=d.OSL, NO=d.NO, WS=d.WS, TAB=d.TAB,
        **{k: v for k, v in globals().items() if k[:2] in ("C_", "K_", "O_", "S_", "A_") or k[:3] in ("REC", "ST_")})
