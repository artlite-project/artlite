"""SD 1.5 attention in the overlay's transformer variant (``AttnOverlay``: an ``Overlay`` subclass through its
extension hooks), and ``attn_block``: an attention layer of a transformer block in ``unet.UNetProgram``.

Work unit: a sub-unit = one (image, head) and NQP query panels (8 queries each) whose output is 40 bfp16 panels
(d 40: 64 queries, d 80: 32 queries); a core's C object holds 4 sub-units ([sub-unit][query panel][d-block] panels,
11520 B); key groups of NK = 8 key panels (64 keys) stream through the B ring: K (KSL slots, panels [key panel][d-block])
then V^T (KSL slots, panels [key block][d-block], each 8 dims x 8 keys: the E_BFPT epilogue).

Kernels (one generic MAC for Q K^T and P V):
- ``{p}_amac`` (static): 5 accumulators; per quad q and column j: acc_i = init + sum_k A[q][k][i] B[j][k]^T, A re-read
  for every column (its segment padded to a multiple of 8 panels: the bfp16 stream restart rule), B linear per quad.
  Q K^T: A = Q [quad][d-block][5] (4 query panels + 1 unused), B = K; P V: A = V^T (5 d-blocks of a chunk per key
  block; d 80: the other chunk's 5 popped and dropped), B = P [query panel][key block]: acc = O^T panels.
- ``{p}_asoft`` (static): P = vexp2(S) + vexp2(S + 1/2) / sqrt 2 (bfp16, in place over S); l^T += ones P^T (the
  rounded P, on the MMUL).
- ``{p}_aph`` (static): the phases: 0 a sub-unit's start (Q permuted in place through band, O = l = 0), 1 a key group
  (group 0: m := the max of each query panel's raw scores, then Q K^T again), 2 the sub-unit's end (O^T / l -> O panels).

Numerics (real operands, bfp16 output): the reference m per query panel set once from key group 0
(never raised; the largest later s - m seen: 15.4), 1 / l as one bf16: 1.02-1.06x the exact-exp error (self),
1.08-1.15x (cross, the BOS key alone in its 8-key block).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel

from artlite.operators.design import range32
from artlite.operators.sd15 import overlay as ov
from artlite.operators.sd15.overlay import GemmL, Overlay, OverlayOp, Seq, bfp16_blocks

if TYPE_CHECKING:
    from artlite.operators.sd15.unet import UNetProgram

# ws state words of the attention phases (past the overlay's W_TILE): L1 addresses (the helpers take ws only): Q (the
# sub-unit's A buffer), K, V (ring), band, C, scr; the group counter, the sub-unit index
S_G, S_SUB, S_V = 33, 34, 35  # (S_V, S_V + 1: the V^T parts' slots)
assert S_G > ov.W_TILE and S_V + 2 <= ov.WS
# command words (overlay words an attention op leaves unused; W_NE untouched: lnbegin reads it). Zero in other ops: the
# core program's attention loops then do nothing. LEAD: the B slots before the first group (the ring at an even
# position for KSL 2, the op's slot count a multiple of 3); KSL: ring slots of a group's K (and of its V^T); NQP query
# panels per sub-unit, NKP key panels per group, ND Q K^T d-blocks (cross: + the mask block), NDV V d-blocks, NCH = NDV / 5
A_LEAD, A_NOBJ, A_NSUB, A_NG, A_GSL, A_NQP, A_NKP, A_NH, A_NDV, A_MASK = (
    ov.W_NR, ov.W_NK3, ov.W_ROFF, ov.W_NKG, ov.W_NBP, ov.W_L, ov.W_J, ov.W_R, ov.W_S, ov.W_IM)
A_TSTR = ov.W_INVN  # (Q row panels)
# (GSL: ring slots of a key group = 2 NH: K and V^T of each 40-dim half; NH: halves of the head (d 80: K / V GemmL output
# channels permuted so that a pass holds two heads' halves); MASK: cross attention's mask d-block in half 0)

# workspace (bytes): C: the sub-units' panels at 2880 k, O^T (fp32 panels [query panel][d-block]) at O_OFF; band: S
# (fp32, one quad) / P; scr: l (16 floats per query panel: its 8 queries' sums twice), -m (fp32 panels, row r = -m of
# query r)
O_OFF, U_OFF = 8640, 19968  # (U: the ones panels, 576 B, never sent: in C's tail)
M_OFF = 512

_EXT = r"""
// ---- attention ----
namespace {

#define AW(T, w) ((T)(uintptr_t)ws[w])

// S[nq query panels][8 key panels] (fp32) of one quad and 40-dim part: per query panel its 5 Q panels in registers
// (its natural row: rows tstr panels apart; ``junk``: the second part, 5 panels in), then per key-panel pair 2
// accumulators over the 5 d-blocks (K [8][5] linear, re-read per query panel). init: the query panel's -m panel (first)
// or S.
static __attribute__((noinline)) void {{ p }}_aqk(const bfp16ebs8 *__restrict q, const bfp16ebs8 *__restrict k,
                                                  float *__restrict S, const float *__restrict mref, int tstr,
                                                  int junk, int first, int nq) {
  al::BInR<aie_dm_resource::b> ks(k);
  AIE_LOOP_NO_UNROLL
  for (int qp = 0; qp < nq; qp++) {
    al::BInR<aie_dm_resource::a> qs(q);
    qs.seek(tstr * qp);
    if (junk) {
      (void)qs.pop(); (void)qs.pop(); (void)qs.pop(); (void)qs.pop(); (void)qs.pop();
    }
    const al::BV q0 = qs.pop(), q1 = qs.pop(), q2 = qs.pop(), q3 = qs.pop(), q4 = qs.pop();
    const float *z = first ? mref + 64 * qp : S;
    const int zj = first ? 0 : 128, zb = first ? 0 : 64;
    float *s = S;
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    AIE_LOOP_MAX_ITERATION_COUNT(4)
    for (int kp = 0; kp < 4; kp++) {
      al::MM8 a(aie::accum<accfloat, 64>(aie::load_v<64>(z))), b(aie::accum<accfloat, 64>(aie::load_v<64>(z + zb)));
      a.mac(q0, aie::op_transpose(ks.pop()));
      a.mac(q1, aie::op_transpose(ks.pop()));
      a.mac(q2, aie::op_transpose(ks.pop()));
      a.mac(q3, aie::op_transpose(ks.pop()));
      a.mac(q4, aie::op_transpose(ks.pop()));
      b.mac(q0, aie::op_transpose(ks.pop()));
      b.mac(q1, aie::op_transpose(ks.pop()));
      b.mac(q2, aie::op_transpose(ks.pop()));
      b.mac(q3, aie::op_transpose(ks.pop()));
      b.mac(q4, aie::op_transpose(ks.pop()));
      aie::store_v(s, a.template to_vector<float>());
      aie::store_v(s + 64, b.template to_vector<float>());
      s += 128;
      z += zj;
    }
    ks.seek(-40);
    S += 512;
  }
}

// O^T[query panel j][5 d-blocks of the half] += V^T P^T over the group's 8 key blocks: V^T [8][5] (re-read per query
// panel), P [4][8] linear; O^T panels [query panel][NDV] (fp32, oq = ndv 64 floats per query panel)
static __attribute__((noinline)) void {{ p }}_apv(const bfp16ebs8 *__restrict v, const bfp16ebs8 *__restrict P,
                                                  float *__restrict O, int oq, int nq) {
  al::BInR<aie_dm_resource::b> ps(P);
  al::BInR<aie_dm_resource::a> vs(v);
  AIE_LOOP_NO_UNROLL
  for (int j = 0; j < nq; j++) {
{% for i in range(5) %}
    al::MM8 m{{ i }}(aie::accum<accfloat, 64>(aie::load_v<64>(O + {{ 64 * i }})));
{% endfor %}
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    AIE_LOOP_MAX_ITERATION_COUNT(8)
    for (int kb = 0; kb < 8; kb++) {
      const al::BV b = ps.pop();
{% for i in range(5) %}
      m{{ i }}.mac(vs.pop(), aie::op_transpose(b));
{% endfor %}
    }
    vs.seek(-40);
{% for i in range(5) %}
    aie::store_v(O + {{ 64 * i }}, m{{ i }}.template to_vector<float>());
{% endfor %}
    O += oq;
  }
}

// P = vexp2(y) + vexp2(y + 1/2) / sqrt 2 of y = S (fp32 [4][8], s - m), bfp16 panels in place (panel i at 72 i, read at
// 256 i before)
static inline __attribute__((always_inline)) void {{ p }}_asoft(const float __aie_dm_resource_a *__restrict S, bfp16ebs8 *__restrict P,
                                                    int n) {
  al::BOut<aie_dm_resource::b> po(P);
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
  const aie::vector<bfloat16, 64> half = aie::broadcast<bfloat16, 64>(0.5f);
  const aie::vector<bfloat16, 64> kb = aie::broadcast<bfloat16, 64>(0.70710678f);
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  AIE_LOOP_MAX_ITERATION_COUNT(32)
  for (int i = 0; i < n; i++) {
    const aie::accum<accfloat, 64> A(aie::load_v<64>(S));
    const aie::vector<bfloat16, 64> e1 = al::exp2_acc<64>(A);
    const aie::vector<bfloat16, 64> e2 = al::exp2_acc<64>(aie::mac(A, half, one));
    po.push(aie::mac(aie::mul(e1, one), e2, kb).template to_vector<bfp16ebs8>());
    S += 64;
  }
}

// l[query panel] += P M^T row sums over the 8 key blocks (M: the ones panels, or cross attention's key masks; lanes 0..7
// of the result = the query panel's 8 sums); setm (group 0, m = 0): the -m panels := -floor(log2(the sums)) (their
// exponent bits; row r for query r), l stays 0
static inline __attribute__((always_inline)) void {{ p }}_als(const bfp16ebs8 *__restrict P, const bfp16ebs8 *__restrict M,
                                                  float *__restrict l, float *__restrict mref, int setm, int nq) {
  al::BInR<aie_dm_resource::a> ps(P);
  AIE_LOOP_NO_UNROLL
  for (int qp = 0; qp < nq; qp++) {
    al::BInR<aie_dm_resource::b> ms(M);
    al::MM8 acc(aie::accum<accfloat, 64>(aie::load_v<16>(l).template grow_replicate<64>()));
{% for kb in range(8) %}
    acc.mac(ms.pop(), aie::op_transpose(ps.pop()));
{% endfor %}
    const aie::vector<float, 16> lv = acc.template to_vector<float>().template extract<16>(0);
    if (setm) {
      const aie::vector<int32, 16> e = aie::sub(aie::broadcast<int32, 16>(127), aie::downshift(lv.template cast_to<int32>(), 23));
      const aie::vector<bfloat16, 16> m16 = aie::accum<accfloat, 16>(aie::to_float(e, 0)).template to_vector<bfloat16>();
      aie::store_v(mref, aie::accum<accfloat, 64>(aie::transpose(m16.template extract<8>(0).template grow_replicate<64>(), 8, 8))
                             .template to_vector<float>());
    } else {
      aie::store_v(l, lv);
    }
    l += 16;
    mref += 64;
  }
}

// soft then l (one quad)
static __attribute__((noinline)) void {{ p }}_asl(float *S, const bfp16ebs8 *M, float *l, float *mr, int setm, int nqq) {
  {{ p }}_asoft((const float __aie_dm_resource_a *)S, (bfp16ebs8 *)S, 8 * nqq);
  {{ p }}_als((const bfp16ebs8 *)S, M, l, mr, setm, nqq);
}

}  // namespace

// the entry: one key group (per 40-dim half its K, then its V^T: one ring slot each; cross attention: + a slot of key
// masks) of the current sub-unit. The op's first call: its LEAD B slots skipped (their locks: the core program), the
// ones panels. A sub-unit's first group: O = l = -m = 0 (its Q object, A ping-pong: [query panel][TSTR panels], the
// head's d-blocks first). Its last group: O^T / l -> O panels [query panel][d-block] (bfp16) at 2880 k in C.
__attribute__((noinline, minsize)) void {{ p }}_att_impl(const int32_t *__restrict a0, const int32_t *__restrict a1,
                                                         const int32_t *__restrict ring, int32_t *__restrict band,
                                                         int32_t *__restrict cb, int32_t *__restrict scr,
                                                         int32_t *__restrict ws) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *o = (float *)((uint8_t *)cb + {{ O_OFF }}), *l = (float *)scr, *mr = (float *)scr + {{ M_OFF // 4 }};
  bfp16ebs8 *ones = (bfp16ebs8 *)((uint8_t *)cb + {{ U_OFF }});
  if (ws[{{ A_LEAD }}]) {
    const int pos = ws[{{ W_BPOS }}] + ws[{{ A_LEAD }}];
    ws[{{ W_BPOS }}] = pos >= {{ NQ }} ? pos - {{ NQ }} : pos;
    ws[{{ A_LEAD }}] = 0;
    ws[{{ S_SUB }}] = 0;
    ws[{{ S_G }}] = 0;
    al::BOut<> u(ones);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 8; i++) u.push(aie::accum<accfloat, 64>(aie::broadcast<float, 64>(1.0f)).template to_vector<bfp16ebs8>());
  }
  const int nqp = ws[{{ A_NQP }}], ndv = ws[{{ A_NDV }}], nh = ws[{{ A_NH }}], tstr = ws[{{ A_TSTR }}];
  const int g0 = ws[{{ S_G }}] == 0;
  const bfp16ebs8 *q = (const bfp16ebs8 *)(ws[{{ W_APAR }}] ? a1 : a0);  // (the sub-unit's Q: toggled at its last group)
  if (g0) {
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < nqp * ndv * 2; i++) aie::store_v(o + 32 * i, aie::zeros<float, 32>());
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 20; i++) aie::store_v(l + 32 * i, aie::zeros<float, 32>());  // (l, -m)
  }
  const int nqq = nqp < 4 ? nqp : 4;
  float *S = (float *)band;  // (one quad's S / P)
  const bfp16ebs8 *kp[2], *M = ones;
  for (int h = 0; h < nh; h++) {  // (per part: its K slot, then its V^T slot; then the mask)
    kp[h] = (const bfp16ebs8 *){{ p }}_b(ring, ws, 1);
    ws[{{ S_V }} + h] = (int32_t)(uintptr_t){{ p }}_b(ring, ws, 1);
  }
  if (ws[{{ A_MASK }}]) M = (const bfp16ebs8 *){{ p }}_b(ring, ws, 1);
  for (int qq = 0; qq < nqp; qq += nqq) {  // per quad (qq: its first query panel)
    // S = Q K^T - m, P, l (group 0: first m = 0, then -m := -floor(log2 of the group's sums))
    for (int it = 0; it <= g0; it++) {
      for (int h = 0; h < nh; h++)
        {{ p }}_aqk((const bfp16ebs8 *)((const uint8_t *)q + 72 * tstr * qq), kp[h], S, mr + 64 * qq, tstr, h & 1,
                    h == 0, nqq);
      {{ p }}_asl(S, M, l + 16 * qq, mr + 64 * qq, g0 && it == 0, nqq);
    }
    for (int h = 0; h < nh; h++)  // O^T += V^T P^T
      {{ p }}_apv(AW(const bfp16ebs8 *, {{ S_V }} + h), (const bfp16ebs8 *)S, o + 64 * (ndv * qq + 5 * h), 64 * ndv, nqq);
  }
  if (++ws[{{ S_G }}] < ws[{{ A_NG }}])
    return;
  ws[{{ S_G }}] = 0;
  ws[{{ W_APAR }}] ^= 1;
  const int sub = ws[{{ S_SUB }}];
  ws[{{ S_SUB }}] = sub == 3 ? 0 : sub + 1;
  {
    al::BOut<> out((bfp16ebs8 *)((uint8_t *)cb + 2880 * sub));
    AIE_LOOP_NO_UNROLL
    for (int qp = 0; qp < nqp; qp++) {
      const aie::vector<float, 8> r8 = aie::inv(aie::load_v<8>(l + 16 * qp));
      const aie::vector<bfloat16, 64> r = aie::accum<accfloat, 16>(aie::concat(r8, r8)).template to_vector<bfloat16>().template grow_replicate<64>();
      AIE_LOOP_NO_UNROLL
      for (int db = 0; db < ndv; db++) {
        const aie::vector<bfloat16, 64> x = aie::accum<accfloat, 64>(aie::load_v<64>(o)).template to_vector<bfloat16>();
        out.push(aie::accum<accfloat, 64>(aie::transpose(aie::mul(x, r).template to_vector<bfloat16>(), 8, 8))
                     .template to_vector<bfp16ebs8>());
        o += 64;
      }
    }
  }
  if (sub < 3)
    return;
  // the object's 4 sub-units ([sub][query panel][d-block] panels at C[0 ..]) -> the next GEMM's panel objects
  // [tg 8][q 10][pair 2] (C[0 .. 11520)), through band and the sub-unit's A buffer (its Q is done): d 40: sub (image,
  // head of the pass), query panel = tg, q = 5 head + d-block; d 80: sub (image, query half), tg = 4 half + panel
  {
    uint64_t *c = (uint64_t *)cb;
    const uint64_t *bd = (const uint64_t *)band, *qa = (const uint64_t *)q;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 126; i++) aie::store_v(band + 16 * i, aie::load_v<16>(cb + 16 * i));  // (panels 0 .. 111)
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 54; i++) aie::store_v((int32_t *)q + 16 * i, aie::load_v<16>(cb + 2016 + 16 * i));  // (112 ..)
    const int nqs = nh == 2 ? 4 : 8, nds = nh == 2 ? 10 : 5;  // (a sub-unit's query panels x d-blocks)
    int sb = 0, qp = 0, db = 0;
    AIE_LOOP_NO_UNROLL
    for (int e = 0; e < 160; e++) {  // (e = (sub, query panel, d-block): its panel at 72 e)
      const int tg = qp + (nh == 2 ? 4 * (sb & 1) : 0), qq = db + (nh == 2 ? 0 : 5 * (sb & 1));
      const uint64_t *src = e < 112 ? bd + 9 * e : qa + 9 * (e - 112);
      uint64_t *dst = c + 9 * (20 * tg + 2 * qq + (sb >> 1));
      AIE_LOOP_NO_UNROLL
      for (int w = 0; w < 9; w++) dst[w] = src[w];
      const int dw = db + 1 == nds, qw = dw && qp + 1 == nqs;
      db = dw ? 0 : db + 1;
      qp = qw ? 0 : (dw ? qp + 1 : qp);
      sb = qw ? sb + 1 : sb;
    }
  }
}
"""


class AttnOverlay(Overlay):
    """The overlay with the attention phases: its transformer variant runs them (d 40 at 64x64, d 80 at 32x32);
    ``conv=True``: the regular variant (stub entries, the same core main)."""

    def __init__(self, **kw):
        kw.setdefault("conv", False)
        super().__init__(**kw)

    def ext_source(self) -> str:
        from artlite.operators.base import JINJA_ENV
        g = {k: v for k, v in vars(ov).items() if k.startswith("W_")}
        g.update({k: v for k, v in globals().items() if k[:2] in ("A_", "S_")})
        return JINJA_ENV.from_string(_EXT).render(p=self.p, NQ=ov.NQ, A_OBJ=ov.A_OBJ, O_OFF=O_OFF, U_OFF=U_OFF,
                                                  M_OFF=M_OFF, **g)

    def ext_entries(self):
        return [("att", ("const int32_t *__restrict a0, const int32_t *__restrict a1, const int32_t *__restrict ring, "
                         "int32_t *__restrict band, int32_t *__restrict cb, int32_t *__restrict scr, int32_t *__restrict ws"),
                 "a0, a1, ring, band, cb, scr, ws")]

    def ext_args(self, t, c, r, types):
        return [Kernel(f"{self.p}_att", "kernel.o", [types["A"], types["A"], types["B"], types["band"], types["C"],
                                                      types["S"], types["ws"]])]

    def ext_phases(self, env: dict) -> None:
        att = env["ext"][0]
        ws = env["ws"]
        args = [env[k] for k in ("a0", "a1", "ring", "band", "cb", "scr", "ws")]
        bb, bf = env["bb"], env["bf"]
        for _ in range32(ws[ov.W_ATT]):  # (0 in the overlay's ops: their command words alias ours)
            for _ in range32(ws[A_LEAD]):  # (the B slots before the first group)
                bb.acquire(1)
                bf.release(1)
            for _ in range32(ws[A_NOBJ]):
                env["cf"].acquire(1)
                for _ in range32(ws[A_NSUB]):
                    env["aa"].acquire(1)
                    for _ in range32(ws[A_NG]):
                        for _ in range32(ws[A_GSL]):
                            bb.acquire(1)
                        att(*args)
                        for _ in range32(ws[A_GSL]):
                            bf.release(1)
                    env["af"].release(1)
                env["cc"].release(1)




def _affine_chunks(seq: list[int]) -> list[tuple[int, int]]:
    """Split seq into consecutive pieces that are 2-D affine walks (ov.affine2): [(start, length)]."""
    out, i = [], 0
    while i < len(seq):
        n = len(seq) - i
        while n > 1 and ov.affine2(seq[i:i + n]) is None:
            n -= 1
        out.append((i, n))
        i += n
    return out


class AttnSelf(OverlayOp):
    """Self attention of a level (64x64: C 320, d 40; 32x32: C 640, d 80) over both images: units (image,
    head), 8 heads. Inputs: Q (the Q GemmL's E_BFP objects [P][TT][C_OBJ], q pre-scaled by log2(e) / sqrt(d)), KV (the
    K GemmL's E_BFP and the V GemmL's E_BFPT objects in one tensor [P][2][TT][C_OBJ]), W (``pack``: the command and the
    lead slots), Y: [P][TT][C_OBJ] objects of 4 sub-units' O panels ([sub-unit][query panel][d-block], 72 B each).
    Work: row r of cores takes pass r (d 40) or passes 2r, 2r + 1 (d 80); a round = one object (pass, tile) per core of
    the row (tiles 8 n + c), its 4 sub-units (image i, head of the pass / query half) stream the unit's K / V (all the
    image's tiles: 64-key groups) through the row's B stream, a group = per 40-dim half its K then its V^T (one ring slot
    each). d 80: the K and V GemmLs' output channels permuted (``kv_perm``) so that a pass holds two heads' halves.
    Arguments: Q, KV, W, Y."""

    def __init__(self, ovl: Overlay, *, W: int, C: int):
        super().__init__(ovl)
        assert W in (64, 32) and C // 8 in (40, 80)
        self.W, self.C, self.d = W, C, C // 8
        self.nd = self.d // 8
        self.P, self.TT = C // 80, 2 * W * W // 128
        self.nqp, self.nkp, self.nh = 40 // self.nd, 8, self.nd // 5
        self.ng = self.TT  # (64-key groups: one tile of an image)
        rows = ov.ROWS
        if self.d == 40:  # (row r: pass r; round n: tile 8 n + c)
            self.rounds = [[(r, 8 * n + c) for n in range(8)] for r in range(rows) for c in range(ov.COLS)]
        else:  # (row r: heads 2r, 2r + 1 = Q / O passes; round n: head 2r + n % 2, tile 8 (n // 2) + c)
            self.rounds = [[(2 * r + n % 2, 8 * (n // 2) + c) for n in range(4)] for r in range(rows) for c in range(ov.COLS)]
        self.nobj = len(self.rounds[0])
        self.gsl, self.mask, self.tstr = 2 * self.nh, 0, 64 // self.nqp
        self._lead()

    def _lead(self) -> None:
        # (the lead slots: the op's slot count a multiple of 6 leaves the ring where it was, p0 of the next op)
        slots = 3 + self.nobj * 4 * self.ng * self.gsl
        self.lead = next(k for k in range(1, 7) if (slots + k) % 6 == 0)
        self.slots = slots + self.lead

    def kv_elems(self) -> int:
        return 2 * self.P * self.TT * ov.C_OBJ // 2

    def kv_tap(self, ps: int, sb: int) -> TensorAccessPattern:
        """The B stream of sub-unit sb of pass ps: its unit's key groups (bf16 offset, sizes, strides in KV)."""
        from aie.helpers.taplib import TensorAccessPattern as TAP
        C2, TT = ov.C_OBJ // 2, self.TT
        i, h, _ = self.sub(ps, sb)
        # (the head's halves: passes 2 (h // 2) .. of the permuted KV at d 80 (K0 V0 K1 V1 TT C_OBJ apart); d 40: pass
        # h // 2 (K V))
        off = ((2 * (h // 2) * self.nh) * TT) * C2 + (i * 10 + 5 * (h % 2)) * 36
        return TAP((self.kv_elems(),), off, [self.ng, 2 * self.nh, 8, 180], [C2, TT * C2, 720, 1])

    def job(self, c: int, r: int) -> list[tuple[int, int]]:
        return self.rounds[r * ov.COLS + c]

    def sub(self, ps: int, s: int) -> tuple[int, int, int]:
        """Sub-unit s of pass ps -> (image, head, first q of the head in the pass)."""
        if self.d == 40:
            return s // 2, 2 * ps + s % 2, 5 * (s % 2)
        return s // 2, ps, 0

    def get_arg_spec(self):
        import torch
        bf, I, O = torch.bfloat16, ov.ArgDirection.IN, ov.ArgDirection.OUT
        n = self.P * self.TT * ov.C_OBJ // 2
        return [ov.ArgSpec(I, (n,), bf), ov.ArgSpec(I, (self.kv_elems(),), bf),
                ov.ArgSpec(I, ((3 + self.lead) * ov.Q // 2,), bf), ov.ArgSpec(O, (n,), bf)]

    def pack(self) -> np.ndarray:
        cmd = np.zeros((3 + self.lead) * ov.Q // 4, np.int32)
        cmd[A_LEAD], cmd[A_NOBJ], cmd[A_NSUB], cmd[A_NG], cmd[A_GSL] = self.lead, self.nobj, 4, self.ng, self.gsl
        cmd[A_NQP], cmd[A_NKP], cmd[A_NH], cmd[A_NDV], cmd[A_MASK] = self.nqp, self.nkp, self.nh, self.nd, self.mask
        cmd[A_TSTR] = self.tstr
        cmd[ov.W_ATT] = 1
        return cmd.view(np.uint8)

    def _sequence(self, args: tuple, s: Seq) -> None:
        from aie.dialects._aie_enum_gen import DMAChannelDir  # noqa: F401
        from aie.helpers.taplib import TensorAccessPattern as TAP
        from aie.iron import Acquire, Bd, Release
        Qt, KV, Wt, Y = args
        C2, Q2, COLS, ROWS = ov.C_OBJ // 2, ov.Q // 2, ov.COLS, ov.ROWS
        n_q = self.P * self.TT * C2
        nd, TT = self.nd, self.TT
        qobj = lambda ps, t: (ps * TT + t) * C2
        # (1) per column: staging of the cores' Q blocks (heap slots 2 x C_OBJ per core; 4 A objects of a slot per round)
        slot = lambda k, i: (2 * k + i) * ov.C_OBJ
        # (a slot: 4 sub-units' A objects, each [query panel][TSTR panels] as the shim reads them; the core permutes)
        wtap = lambda off: s.htap(off, 4 * ov.A_OBJ)
        for c in range(COLS):
            t = s.d_in[c].endpoint(s.mems[c]).task(
                *[Bd(s.heap[c], tap=wtap(slot(k, i)), acquires=[Acquire(s.st_free[c, k])],
                     releases=[Release(s.st_full[c, k])]) for i in range(2) for k in range(ROWS)],
                runs=self.nobj // 2, wait=True)
            t.start()
            s.waits.append(t)
            for k in range(ROWS):
                t = s.a[c, k].endpoint(s.mems[c]).task(
                    *[Bd(s.heap[c], tap=s.htap(slot(k, i), 4 * ov.A_OBJ), acquires=[Acquire(s.st_full[c, k])],
                         releases=[Release(s.st_free[c, k])]) for i in range(2)], runs=self.nobj // 2)
                t.start()
                s.tasks.append(t)
        # (2) per column: the drains (the cores' objects in the static order: per round, cores 0..3)
        for c in range(COLS):
            seq = [qobj(*self.job(c, k)[n]) for n in range(self.nobj) for k in range(ROWS)]
            for i0, n in _affine_chunks(seq):
                (a, b), (sa, sb) = ov.affine2(seq[i0:i0 + n]) if n > 1 else ([1, 1], [0, 0])
                run = ov.run_len(C2)
                s.d_out[c].drain(Y, tap=TAP((n_q,), seq[i0], [a, b, C2 // run, run], [sa, sb, run, 1]), wait=True)
        # (3) shim fills, round by round: the Q staging (per column) and the B streams (per row: command + lead, then
        # per sub-unit its unit's K / V groups)
        order = [(("b", r), (s.b[r], Wt, TAP(((3 + self.lead) * Q2,), 0, [1, 1, 1, (3 + self.lead) * Q2], [0, 0, 0, 1])))
                 for r in range(ROWS)]
        for n in range(self.nobj):  # (round-major: a fill awaited only needs earlier rounds' fills)
            for c in range(COLS):
                ps0, t0 = self.job(c, 0)[n]
                ks = qobj(*self.job(c, 1)[n]) - qobj(ps0, t0)
                if nd == 5:  # (sub-unit (i, hh): per query panel 8 panels from the head's first: 5 + 3 unused)
                    tap = TAP((n_q,), qobj(ps0, t0), [ROWS, 4, 8, 288], [ks, 180, 720, 1])
                else:  # (sub-unit (i, query half): per query panel 16 panels: 10 + 6 unused)
                    tap = TAP((n_q,), qobj(ps0, t0), [ROWS, 2, 8, 576], [ks, 360, 720, 1])
                order.append((("q", c), (s.d_in[c], Qt, tap)))
            for r in range(ROWS):
                ps = self.job(0, r)[n][0]
                for sb in range(4):
                    order.append((("b", r), (s.b[r], KV, self.kv_tap(ps, sb))))
        pend: dict = {}
        for key, (ep, src, tap) in order:
            lst = pend.setdefault(key, [])
            if len(lst) == 4:
                t = lst.pop(0)
                t.await_()
                t.free()
            lst.append(ep.fill(src, tap=tap, wait=True, managed=False))
        for lst in pend.values():
            for t in lst:
                t.await_()
                t.free()


class AttnCross(AttnSelf):
    """Cross attention of a level: the queries as in AttnSelf, keys / values the text context (77 tokens per
    image), host-packed (``pack_kv``) per unit (image, head) as 2 groups of 64 key slots (slot 0 the BOS token alone in
    its 8-key block, slots 1-7 BOS copies, 8-83 tokens 1-76, 84-127 copies of token 76; the copies masked: zero V^T
    and zero in the l mask panels, and equal to a real key in their block so they cost no bfp16 precision), each group:
    per half its K and V^T slot, then a slot of 8 mask panels (the l row sums' key masks). Arguments: Q, KV, W, Y."""

    CTX, SLOTS = 77, 128

    def __init__(self, ovl: Overlay, *, W: int, C: int):
        super().__init__(ovl, W=W, C=C)
        self.ng, self.gsl, self.mask = self.SLOTS // 64, 2 * self.nh + 1, 1
        self._lead()

    def kv_elems(self) -> int:
        return 16 * self.ng * self.gsl * ov.Q // 2

    def kv_tap(self, ps: int, sb: int) -> TensorAccessPattern:
        from aie.helpers.taplib import TensorAccessPattern as TAP
        i, h, _ = self.sub(ps, sb)
        n = self.ng * self.gsl * ov.Q // 2
        return TAP((self.kv_elems(),), (8 * i + h) * n, [1, 1, 1, n], [0, 0, 0, 1])

    def slot_map(self) -> np.ndarray:
        """Key slot -> context token."""
        m = np.full(self.SLOTS, self.CTX - 1)
        m[:8] = 0
        m[8:8 + self.CTX - 1] = np.arange(1, self.CTX)
        return m

    def valid(self) -> np.ndarray:
        v = np.zeros(self.SLOTS, bool)
        v[0] = True
        v[8:8 + self.CTX - 1] = True
        return v

    def pack_kv(self, k_ctx: torch.Tensor, v_ctx: torch.Tensor) -> np.ndarray:
        """(2, 77, C) context keys / values -> the KV argument: [image][head][group][part][2880 B]."""
        d, nh, Qb = self.d, self.nh, ov.Q
        sm, va = self.slot_map(), self.valid()
        out = np.zeros((2, 8, self.ng, self.gsl, Qb), np.uint8)
        for i in range(2):
            for h in range(8):
                K = k_ctx[i, sm, h * d:(h + 1) * d].float().numpy()  # (slots, d)
                V = v_ctx[i, sm, h * d:(h + 1) * d].float().numpy() * va[:, None]
                for g in range(self.ng):
                    ks, vs = K[64 * g:64 * g + 64], V[64 * g:64 * g + 64]
                    for hh in range(nh):
                        kh = ks[:, 40 * hh:40 * hh + 40].reshape(8, 8, 5, 8).transpose(0, 2, 1, 3)  # kp db key dim
                        out[i, h, g, 2 * hh, :2880] = bfp16_blocks(np.ascontiguousarray(kh)).reshape(-1)
                        vh = vs[:, 40 * hh:40 * hh + 40].reshape(8, 8, 5, 8).transpose(0, 2, 3, 1)  # kb db dim key
                        out[i, h, g, 2 * hh + 1, :2880] = bfp16_blocks(np.ascontiguousarray(vh)).reshape(-1)
                    mk = np.repeat(va[64 * g:64 * g + 64].reshape(8, 1, 8).astype(np.float32), 8, axis=1)  # kb row key
                    out[i, h, g, 2 * nh, :8 * 72] = bfp16_blocks(mk).reshape(-1)
        return out.reshape(-1)


class GemmLKV(GemmL):
    """The K (kv 0: E_BFP) or V (kv 1: E_BFPT, transposed panels) projection into the attention's KV tensor
    [P][2][TT][C_OBJ] (pass p of K at 2 p, of V at 2 p + 1: give the V op the tensor at TT C_OBJ / 2 elements); d 80:
    the output channels permuted (``kv_perm``). Arguments: A, W, Y (the KV tensor, or its V part)."""

    def __init__(self, ovl: Overlay, *, kv: int, **kw: object):
        super().__init__(ovl, out_bfp16=True, **kw)
        self.kv = kv

    def obj(self, ps: int, t: int) -> int:
        return (2 * ps * self.TT + self.t0 + t) * ov.C_OBJ // 2

    @property
    def y_elems(self) -> int:
        return (2 * self.P - 1) * self.TT * ov.C_OBJ // 2

    def pack(self, w: torch.Tensor, b: torch.Tensor | None = None) -> np.ndarray:
        perm = kv_perm(self.N)
        img = super().pack(w[perm], None if b is None else b[perm])
        if self.kv:  # (each stream's command: the transposed epilogue)
            for j in range(self.kpt):
                o = j * self.part * ov.Q + 4 * ov.W_NE
                img[o:o + 4] = np.array([ov.E_BFPT], np.int32).view(np.uint8)
        return img


def attn_block(up: UNetProgram, p: str, W: int, C: int, nl: str, out: tuple, cross: bool) -> int:
    """An attention layer of a transformer block in the UNet program (``up``, building in the
    transformer variant): Q GemmL (log2 e / sqrt d folded into W_q), self: the K and V GemmLs (one KV tensor), the
    attention op (both 64x64 halves in one op), to_out's GemmL (bias in chunk 0) on the attention's panel objects.
    nl: the LayerNorm output (panel objects [C / 80][TT][C_OBJ]); out: (buf, off) for to_out's K chunks (ysz(C, W)
    each). Cross: K / V from the text context, a per-prompt buffer packed on the host (``up.kv_packs``). Returns the
    chunk count."""
    import math

    from artlite.operators.sd15 import unet as U
    w, d, y = up.w, C // 8, U.ysz(C, W)
    TT = U.tt(W)
    q = up.alloc(p + "aq", y + U.SLACK)
    sc = math.log2(math.e) / math.sqrt(d)
    for k in range(U.nops(W)):
        g = up.get(GemmL, W, K=C, N=C, out_bfp16=True, **up.tk(W, k))
        up.add(g, nl, up.wimg(f"{p}aq.{k}", g.pack(w[p + "to_q.weight"].float() * sc, None)), q)
    if cross:
        op = up.get(AttnCross, W, W=W, C=C)
        kv = up.alloc(p + "akv", op.kv_elems() + U.SLACK)
        wk, wv = w[p + "to_k.weight"].float(), w[p + "to_v.weight"].float()

        def packer(ctx: torch.Tensor, op: AttnSelf = op, wk: torch.Tensor = wk, wv: torch.Tensor = wv) -> torch.Tensor:
            k_, v_ = ctx.float() @ wk.t(), ctx.float() @ wv.t()
            return torch.from_numpy(op.pack_kv(k_, v_)).view(torch.bfloat16)
        up.kv_packs.append((kv, packer))
    else:
        op = up.get(AttnSelf, W, W=W, C=C)
        kv = up.alloc(p + "akv", 2 * y + U.SLACK)
        for k in range(U.nops(W)):
            for j, nm in ((0, "to_k.weight"), (1, "to_v.weight")):
                g = up.get(GemmLKV, W, kv=j, K=C, N=C, **up.tk(W, k))
                up.add(g, nl, up.wimg(f"{p}a{nm[3]}.{k}", g.pack(w[p + nm].float(), None)),
                       (kv, j * TT * ov.C_OBJ // 2))
    o = up.alloc(p + "ao", y + U.SLACK)
    up.add(op, q, kv, up.wimg(p + "aw", op.pack()), o)
    wo, bo = w[p + "to_out.0.weight"].float(), w[p + "to_out.0.bias"].float()
    return up.gemm(p + "ato", W, o, K=C, N=C, wt=wo, b=bo, out=out)


def kv_perm(C: int, heads: int = 8) -> torch.Tensor:
    """d 80: the K / V GemmL output channel order (new channel n <- original kv_perm[n]): pass p holds the halves p % 2 of
    heads 2 (p // 2) and 2 (p // 2) + 1 (40 channels each), the layout of d 40's passes."""
    import torch
    d = C // heads
    if d == 40:
        return torch.arange(C)
    n = torch.arange(C)
    p, hh, dd = n // 80, (n % 80) // 40, n % 40
    return (2 * (p // 2) + hh) * d + (p % 2) * 40 + dd


