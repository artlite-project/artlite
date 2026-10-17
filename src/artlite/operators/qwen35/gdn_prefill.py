"""The Gated DeltaNet layers of a qwen35 prompt chunk of T tokens (the prefill), on the decode overlay's state
(``overlay.py``: GSTATE, the conv state in GHDR) and the gdn_in GEMMs' output G (``prefill.gemm``: column c's slice
[qkv | z | ab] at c pitch). Three operators per layer, the decode's math (``gdn.HELPERS``):

``GdnPrepOperator``: the q / k side, once per k head (two cores per column, the column's k heads): per 8 tokens the
conv of the head's q / k channels (256), SiLU, l2 norms of q and k (bf16 hi + lo), per pair of tokens the dot
products and the MMUL A panels of the scan, and the gates d = expm1(g), beta of the k head's v heads, times a
host-written token mask (d = beta = 0 for padding tokens: the scan leaves the state alone through them), as one
record per pair (``REC_ELEMS`` bf16) in PR.

``GdnScanOperator``: the delta rule. Per column four cores (rows 2 .. 5); a round per v head of the column, core r
taking the head's state quarter r (j-panels 4 r .. 4 r + 3: 16 KB, resident in L1 for the round) over the chunk's
tokens: per 8 tokens the conv and SiLU of its v quarter (32 channels), then per pair of tokens (GdnPrep's records)
the delta rule, exactly (``_pair4``): one pass over a panel gives S^T [k0 q0 k1 q1] on the MMUL (bfp16 hi + lo, 14
bits: as accurate as bf16 hi / lo products), token 1's products from it and delta0;
the update S + D S + k0 e0^T + k1 delta1^T, D = d0 + d1 + d0 d1 (padding tokens: D = 0, e = 0, exactly). (The q / k
work in the scan's cores would repeat in the 4 cores of a head and the rounds of a k head's v heads: 12x on the 27B,
~1100 of ~3000 cycles per token and core.) The conv state: the 3 rows before the chunk in G's row layout (CV: G's
rows before a sub-chunk, or a layer's own rows). Streams per column:

- B (broadcast to the column's cores): per round the k head's records (PR), 4 pairs per object;
- V (split, 512-byte parts): per round the v conv weights, the v conv state (CV, 8 rows read), the state quarters
  (32 parts), per 8 tokens the v quarters (G);
- O (joined, 512-byte parts): per 8 tokens o (the head's 128 values, bf16, into the O buffer at the head's place of the
  ssm_out input), then the state quarters back (GSTATE).

``GdnOutOperator``: one core per column: per 8 tokens and pair of the column's v heads rms(o) ssm_norm silu(z) as the
ssm_out GEMM's bfp16 A panels; first the new conv state: the qkv rows of the last three valid tokens (GC: G's rows of
the whole prompt chunk, a runtime offset: ``conv_offset``) into GHDR.
"""

import math

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...gguf_model import GGUFFile
from ...model_config import QwenConfig
from .. import act_lut
from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS
from ..gemm.gemm_q import chained_fill
from ..kernel_source import kernel_prefix
from ..prefill.gemm import Layout
from ..util import Attr, bf16_hilo, regions
from . import gdn as GD
from .image import gdn_v_heads

D, PANEL = GD.D, GD.PANEL
CONV_OFF = "gdn_conv"  # scratchpad parameter: GC's bf16 offset of the conv state rows (``conv_offset``)
B_ELEMS = 2560  # bf16 of a GdnPrep input object: 8 tokens x [ab slot 64 (alpha, beta bf16) | q 128 | k 128]
PART = 256  # bf16 of a split / joined part (512 B)
N_SPART = 4 * PANEL // (2 * PART)  # parts of a state quarter (32)
REC = 2 * B_ELEMS + 4 * 2 * PART  # bytes of a (column, head) record of P: a GdnPrep input object | 4 v-weight parts
# GdnPrep's record of a token pair (bf16): [k0 hi | k0 lo | q0 hi | q0 lo | the same of token 1 (128 each) | gs (16 fp32,
# from G_OFF) | the MMUL A panels (16 x 72 B, from AP_OFF)]
G_OFF, AP_OFF = 8 * 128, 8 * 128 + 32
REC_ELEMS = AP_OFF + 16 * 36
P_HEAD = 4096  # P: ssm_norm (fp32) and per column [dt 16 | ssm_a 16] fp32 from 512 on, then the records


def scan_params(g: GGUFFile, cfg: QwenConfig, layer: int, rows: np.ndarray) -> np.ndarray:
    """P of a GDN layer (uint8): ssm_norm, per column dt and ssm_a of its v heads, then per column and v head the record
    [q / k conv weights [tap 4][hi, lo][256: q head | k head] bf16 | pad] and per quarter r [v conv weights [tap 4]
    [hi, lo][32]].
    ``rows``: the gdn_in qkv block's rows per column (``Block.rows``: global conv channels)."""
    hc = GD.GdnShape(cfg).hc
    cw = np.asarray(g.tensor(f"blk.{layer}.ssm_conv1d.weight").raw, np.float32)  # (channels, 4)
    dt = np.asarray(g.tensor(f"blk.{layer}.ssm_dt.bias").raw, np.float32).reshape(-1)
    A = np.asarray(g.tensor(f"blk.{layer}.ssm_a").raw, np.float32).reshape(-1)
    norm = np.asarray(g.tensor(f"blk.{layer}.ssm_norm.weight").raw, np.float32).reshape(-1)
    out = np.zeros(P_HEAD + COLS * hc * REC, np.uint8)
    out[:norm.nbytes] = norm.view(np.uint8)
    for c in range(COLS):  # (GdnPrep's gate parameters)
        vh = gdn_v_heads(cfg, c)
        prm = np.zeros(32, np.float32)
        prm[:hc], prm[16:16 + hc] = dt[vh], A[vh]
        out[512 + 128 * c:512 + 128 * (c + 1)] = prm.view(np.uint8)

    def hl(w: np.ndarray) -> np.ndarray:  # (..., ch, 4) -> (..., [tap][hi, lo][ch]) bf16 bits as bytes
        hi, lo = GD.hilo_bits(np.swapaxes(w, -1, -2))
        return np.stack([hi, lo], -2).view(np.uint8).reshape(*w.shape[:-2], -1)

    ch = np.stack([r[r >= 0] for r in rows])  # (COLS, channels): the columns' global conv channels
    h = np.arange(hc)
    kh = (h & 1)[:, None] * D + np.arange(D)  # (hc, D): the head's q / k channels of the column
    qk = ch[:, np.concatenate([kh, 2 * D + kh], 1)]  # (COLS, hc, 2 D)
    v = ch[:, (4 * D + D * h[:, None] + np.arange(D)).reshape(hc, 4, 32)]  # (COLS, hc, quarter, 32)
    rec = out[P_HEAD:].reshape(COLS, hc, REC)
    rec[..., :4096] = hl(cw[qk])
    rec[..., 2 * B_ELEMS:] = hl(cw[v]).reshape(COLS, hc, -1)
    return out


_SCAN = JINJA_ENV.from_string(r"""
// The scan core's workspace (offsets in bytes: GdnScanOperator.workspace())
#define SW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

{{ helpers }}
{{ silu }}
// Two tokens (0, 1) on the core's 4 j-panels of S (each 128 i x 8 j fp32, [ib 16][jj 8][ii 8]), in place. Per panel
// pass 1 on the MMUL, C = A (B = the state block's bfp16 hi and lo): A = the pair's panel of the i-block ([k0 q0 k1
// q1] rows, bfp16 hi | lo: 14 bits), C rows r + r + 4 = S^T k0, S^T q0, S^T k1, S^T q1 (8 j each, into cw: [part
// 4][j 32]). Then for the 32 j at once token 0 (u0 = x0 + d0 x0, delta0 = beta0 (v0 - u0), o0) and token 1 on S1 = S
// + d0 S + k0 delta0^T (S1^T k1 = x1 + d0 x1 + (k0 . k1) delta0, ...). Per panel pass 2a: S + D S + k0 e0^T (D = d0
// + d1 + d0 d1, e0 = delta0 + d1 delta0; bf16 hi / lo products), 2b: + k1 delta1^T. g: d0, beta0, k0 . q0, d1,
// beta1, k1 . q1, k0 . k1, k0 . q1. (variant_search over ~1300 structures: 413
// instead of 698 cycles per token and panel. The token math once for the 4 panels on 32 lanes instead of per panel
// on 16 with 8 used: 2988 -> 2548 cycles per pair.)

// pass 1 of a panel: [x0 | y0 | x1 | y1] (8 j each) -> cw, cw + 32, cw + 64, cw + 96
[[clang::always_inline]] inline void {{ p }}_pass1(const float *__restrict si, const uint8_t *__restrict ap,
                                                   float *__restrict cw)
{
    const al::BV nI = al::neg_eye();
    al::MM8 cm(aie::zeros<accfloat, 64>());
    const float *__restrict ps = si;
    al::BIn<> pa((const bfp16ebs8 *)ap);  // (one stream: a panel is 72 B, a stream per panel would start unaligned)
    // (two blocks per iteration: their split / MMUL chains overlap, II 14 per 2 blocks; one: RecMII 11 per block.
    // Written out: AIE_LOOP_UNROLL in an inlined helper inside a loop broke llvm-aie's debug locations)
    AIE_TRY_INITIATION_INTERVAL(16)
    for (int b = 0; b < 8; b++) {
        al::BV sh, sl;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(ps)), nI, sh, sl);
        const al::BV A = pa.pop();
        cm.mac(A, aie::op_transpose(sh));
        cm.mac(A, aie::op_transpose(sl));
        al::BV th, tl;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(ps + 64)), nI, th, tl);
        const al::BV B = pa.pop();
        cm.mac(B, aie::op_transpose(th));
        cm.mac(B, aie::op_transpose(tl));
        ps += 128;
    }
    const aie::vector<float, 64> cv = cm.to_accum().template to_vector<float>();
    const aie::vector<float, 32> cs = aie::add(cv.template extract<32>(0), cv.template extract<32>(1));
    aie::store_v(cw, cs.template extract<8>(0));
    aie::store_v(cw + 32, cs.template extract<8>(1));
    aie::store_v(cw + 64, cs.template extract<8>(2));
    aie::store_v(cw + 96, cs.template extract<8>(3));
}

// pass 2 of a panel (si / so, si2 / so2: the panel, distinct pointers; eb: e0 hi | lo, delta1 hi | lo of its 8 j)
[[clang::always_inline]] inline void {{ p }}_pass2(const float *__restrict si, float *__restrict so,
                                                   const float *__restrict si2, float *__restrict so2,
                                                   const bfloat16 *__restrict kq0, const bfloat16 *__restrict kq1,
                                                   const bfloat16 *__restrict eb, aie::vector<bfloat16, 64> gh,
                                                   aie::vector<bfloat16, 64> gl)
{
    const aie::vector<bfloat16, 64> eh0 = al::expand8(aie::load_v<8>(eb)), el0 = al::expand8(aie::load_v<8>(eb + 32));
    {
        const float *__restrict ps = si;
        float *__restrict po = so;
        const bfloat16 *__restrict pk = kq0;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int b = 0; b < 16; b++) {
            const aie::vector<float, 64> s = aie::load_v<64>(ps);
            ps += 64;
            aie::vector<bfloat16, 64> sh, sl;
            {{ g }}_split(s, sh, sl);
            const aie::vector<bfloat16, 64> kh_ = {{ g }}_tile8(pk), kl = {{ g }}_tile8(pk + {{ D }});
            pk += 8;
            aie::accum<accfloat, 64> acc(s);
            acc = aie::mac(acc, sh, gh);
            acc = aie::mac(acc, kh_, eh0);
            acc = aie::mac(acc, sl, gh);
            acc = aie::mac(acc, kh_, el0);
            acc = aie::mac(acc, sh, gl);
            acc = aie::mac(acc, kl, eh0);
            aie::store_v(po, acc.template to_vector<float>());
            po += 64;
        }
    }
    const aie::vector<bfloat16, 64> eh1 = al::expand8(aie::load_v<8>(eb + 64)), el1 = al::expand8(aie::load_v<8>(eb + 96));
    {
        const float *__restrict ps = si2;
        float *__restrict po = so2;
        const bfloat16 *__restrict pk = kq1;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int b = 0; b < 16; b++) {
            const aie::vector<float, 64> s = aie::load_v<64>(ps);
            ps += 64;
            const aie::vector<bfloat16, 64> kh_ = {{ g }}_tile8(pk), kl = {{ g }}_tile8(pk + {{ D }});
            pk += 8;
            aie::accum<accfloat, 64> acc(s);
            acc = aie::mac(acc, kh_, eh1);
            acc = aie::mac(acc, kh_, el1);
            acc = aie::mac(acc, kl, eh1);
            aie::store_v(po, acc.template to_vector<float>());
            po += 64;
        }
    }
}

// the pair on the 4 panels at S (si, so, si2, so2: S, distinct pointers): v0, v1 the tokens' v (32 j), o0, o1 their
// outputs (32 fp32 each); cw (128 fp32), eb (128 bf16): scratch
__attribute__((noinline)) void {{ p }}_pair4(const float *__restrict si, float *__restrict so,
                                           const bfloat16 *__restrict kq0, const bfloat16 *__restrict kq1,
                                           const bfloat16 *__restrict v0, const bfloat16 *__restrict v1,
                                           const float *__restrict g, float *__restrict o0, float *__restrict o1,
                                           const uint8_t *__restrict ap, const float *__restrict si2,
                                           float *__restrict so2, float *__restrict cw, bfloat16 *__restrict eb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_LOOP_NO_UNROLL
    for (int pj = 0; pj < 4; pj++)
        {{ p }}_pass1(si + 1024 * pj, ap, cw + 8 * pj);
    const aie::vector<float, 32> x0 = aie::load_v<32>(cw), y0 = aie::load_v<32>(cw + 32);
    const aie::vector<float, 32> x1 = aie::load_v<32>(cw + 64), y1 = aie::load_v<32>(cw + 96);
    const aie::vector<float, 32> d0 = aie::broadcast<float, 32>(g[0]), d1 = aie::broadcast<float, 32>(g[3]);
    const aie::vector<float, 32> u0 = aie::add(x0, al::mulff<32>(d0, x0));
    const aie::vector<float, 32> dl0 = al::mulff<32>(aie::broadcast<float, 32>(g[1]), aie::sub(aie::accum<accfloat, 32>(aie::load_v<32>(v0)).template to_vector<float>(), u0));
    aie::store_v(o0, al::mul_hilo<32, {{ rs[0] }}f, {{ rs[1] }}f>(aie::add(aie::add(y0, al::mulff<32>(d0, y0)), al::mulff<32>(aie::broadcast<float, 32>(g[2]), dl0))));
    const aie::vector<float, 32> s1k = aie::add(aie::add(x1, al::mulff<32>(d0, x1)), al::mulff<32>(aie::broadcast<float, 32>(g[6]), dl0));
    const aie::vector<float, 32> s1q = aie::add(aie::add(y1, al::mulff<32>(d0, y1)), al::mulff<32>(aie::broadcast<float, 32>(g[7]), dl0));
    const aie::vector<float, 32> u1 = aie::add(s1k, al::mulff<32>(d1, s1k));
    const aie::vector<float, 32> dl1 = al::mulff<32>(aie::broadcast<float, 32>(g[4]), aie::sub(aie::accum<accfloat, 32>(aie::load_v<32>(v1)).template to_vector<float>(), u1));
    aie::store_v(o1, al::mul_hilo<32, {{ rs[0] }}f, {{ rs[1] }}f>(aie::add(aie::add(s1q, al::mulff<32>(d1, s1q)), al::mulff<32>(aie::broadcast<float, 32>(g[5]), dl1))));
    const aie::vector<float, 32> e0 = aie::add(dl0, al::mulff<32>(d1, dl0));
    const aie::vector<float, 32> Dv = aie::add(aie::add(d0, d1), al::mulff<32>(d0, d1));
    {
        aie::vector<bfloat16, 32> h_ = aie::accum<accfloat, 32>(e0).template to_vector<bfloat16>();
        aie::store_v(eb, h_);
        aie::store_v(eb + 32, aie::accum<accfloat, 32>(aie::sub(e0, aie::accum<accfloat, 32>(h_).template to_vector<float>())).template to_vector<bfloat16>());
        h_ = aie::accum<accfloat, 32>(dl1).template to_vector<bfloat16>();
        aie::store_v(eb + 64, h_);
        aie::store_v(eb + 96, aie::accum<accfloat, 32>(aie::sub(dl1, aie::accum<accfloat, 32>(h_).template to_vector<float>())).template to_vector<bfloat16>());
    }
    aie::vector<bfloat16, 64> gh, gl;  // D as bf16 hi + lo
    {{ g }}_split(aie::broadcast<float, 64>(Dv[0]), gh, gl);
    AIE_LOOP_NO_UNROLL
    for (int pj = 0; pj < 4; pj++)
        {{ p }}_pass2(si + 1024 * pj, so + 1024 * pj, si2 + 1024 * pj, so2 + 1024 * pj, kq0, kq1, eb + 8 * pj, gh, gl);
}

// y[t][0 .. N) = bf16(sum over the taps of w (hi + lo) x) of 8 tokens t: x rows t .. t + 3 of r (pitch {{ NCH }}: the
// window's 3 rows, then the tokens'), w [tap 4][hi, lo][N]
template <int N>
__attribute__((noinline)) void {{ p }}_conv8(const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                           bfloat16 *__restrict y)
{
    for (int c = 0; c < N; c += 32) {
        aie::vector<bfloat16, 32> wh[4], wl[4];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 4; t++) {
            wh[t] = aie::load_v<32>(w + 2 * N * t + c);
            wl[t] = aie::load_v<32>(w + 2 * N * t + N + c);
        }
        const bfloat16 *__restrict pr = r + c;
        bfloat16 *__restrict py = y + c;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < 8; t++) {
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < 4; k++) {
                const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + {{ NCH }} * k);
                acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
            }
            aie::store_v(py, acc.template to_vector<bfloat16>());
            pr += {{ NCH }};
            py += {{ NCH }};
        }
    }
}

}  // namespace

extern "C" {

// A round's first split parts: k = 0: the v quarter's conv weights; k = 1: its conv state rows ([8][32]: 3 used)
void {{ p }}_vinit(const bfloat16 *__restrict s, int32_t *__restrict ws, int32_t k)
{
    if (k == 0) {  // ([tap][hi, lo][32])
        for (int i = 0; i < 8; i++)
            aie::store_v(SW(bfloat16, {{ w.wc }}) + 32 * i, aie::load_v<32>(s + 32 * i));
        return;
    }
    bfloat16 *__restrict r = SW(bfloat16, {{ w.rows }});
    for (int t = 0; t < 3; t++)
        aie::store_v(r + 32 * t, aie::load_v<32>(s + 32 * t));
}

// Part k of the state quarter in or out
void {{ p }}_sin(const bfloat16 *__restrict s, float *__restrict S, int32_t k)
{
    al::copy<{{ PART // 2 }}, 32>(S + {{ PART // 2 }} * k, (const float *)s);
}

void {{ p }}_sout(const float *__restrict S, bfloat16 *__restrict o, int32_t k)
{
    al::copy<{{ PART // 2 }}, 32>((float *)o, S + {{ PART // 2 }} * k);
}

// 8 tokens of the round's v head h (index h / 2 among its k head's): b = the 4 pairs' records (GdnPrep), s [8][32]
// (v), o [8][32]
void {{ p }}_group(const bfloat16 *__restrict b, const bfloat16 *__restrict s, float *__restrict S,
                   int32_t *__restrict ws, bfloat16 *__restrict o, int32_t h)
{
    const int hh = h >> 1;
{% if "ev_group" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    // the tokens' raw v quarter after the window's 3 rows; conv; silu
    bfloat16 *__restrict r = SW(bfloat16, {{ w.rows }});
    for (int i = 0; i < 8; i++)
        aie::store_v(r + 32 * (3 + i), aie::load_v<32>(s + 32 * i));
    bfloat16 *__restrict yc = SW(bfloat16, {{ w.yc }});
    bfloat16 *__restrict ys = SW(bfloat16, {{ w.ys }});
{% if "conv" not in ablate %}
    {{ p }}_conv8<32>(r, SW(bfloat16, {{ w.wc }}), yc);
{% endif %}
{% if "silu" not in ablate %}
    {{ lut }}_silu_lut<256>(yc, ys);
{% endif %}
    al::copy<96, 32>(r, r + 256);  // (the last 3 tokens: the next group's window)
    float *__restrict ob = SW(float, {{ w.ob }});  // o of the pair [2][32]
    float *__restrict g = SW(float, {{ w.g }});
{% if "pairs" not in ablate %}
    for (int i = 0; i < 8; i += 2) {  // (padding tokens: d = beta = 0, the state stays)
        const bfloat16 *__restrict rec = b + {{ REC_ELEMS }} * (i >> 1);
        const float *__restrict gs = (const float *)(rec + {{ G_OFF }});
        g[0] = gs[hh];
        g[1] = gs[3 + hh];
        g[2] = gs[12];
        g[3] = gs[6 + hh];
        g[4] = gs[9 + hh];
        g[5] = gs[13];
        g[6] = gs[14];
        g[7] = gs[15];
        const bfloat16 *__restrict y0 = ys + 32 * i;
{% if "ev_panel" in ablate %}
        event0();
{% endif %}
{% if "panel" not in ablate %}
        {{ p }}_pair4(S, S, rec, rec + {{ 4 * D }}, y0, y0 + 32, g, ob, ob + 32, (const uint8_t *)(rec + {{ AP_OFF }}), S,
                      S, SW(float, {{ w.cw }}), SW(bfloat16, {{ w.eb }}));
{% endif %}
{% if "ev_panel" in ablate %}
        event1();
{% endif %}
        aie::store_v(o + 32 * i, aie::accum<accfloat, 32>(aie::load_v<32>(ob)).template to_vector<bfloat16>());
        aie::store_v(o + 32 * i + 32, aie::accum<accfloat, 32>(aie::load_v<32>(ob + 32)).template to_vector<bfloat16>());
    }
{% endif %}
{% if "ev_group" in ablate %}
    event1();
{% endif %}
}

}  // extern "C"
""")


_PREP = JINJA_ENV.from_string(r"""
// The prep core's workspace (offsets in bytes: GdnPrepOperator.workspace())
#define SW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

{{ helpers }}
{{ silu }}
// x (rows 0 .. 3 of 8) -> the MMUL A panel [bfp16(x); bfp16(x - bfp16(x))] (attn._ahilo)
[[clang::always_inline]] inline al::BV {{ p }}_ahilo(aie::accum<accfloat, 32> x, const al::BV &nI)
{
    const aie::accum<accfloat, 64> xx = aie::concat(x, x);
    al::MM8 t(xx);
    t.mac(xx.template to_vector<bfp16ebs8>(), aie::op_transpose(nI));
    return aie::concat(x, t.to_accum().template extract<32>(1)).template to_vector<bfp16ebs8>();
}

// The pair's A panels: per i-block [k0 q0 k1 q1] (hi + lo, fp32) -> ap (16 x 72 B)
__attribute__((noinline)) void {{ p }}_apanels(const bfloat16 *__restrict kq0, const bfloat16 *__restrict kq1,
                                             uint8_t *__restrict ap)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::BOut<> po((bfp16ebs8 *)ap);
    for (int b = 0; b < 16; b++) {
        const aie::vector<bfloat16, 32> h = aie::concat(aie::concat(aie::load_v<8>(kq0 + 8 * b), aie::load_v<8>(kq0 + {{ 2 * D }} + 8 * b)),
                                                        aie::concat(aie::load_v<8>(kq1 + 8 * b), aie::load_v<8>(kq1 + {{ 2 * D }} + 8 * b)));
        const aie::vector<bfloat16, 32> l = aie::concat(aie::concat(aie::load_v<8>(kq0 + {{ D }} + 8 * b), aie::load_v<8>(kq0 + {{ 3 * D }} + 8 * b)),
                                                        aie::concat(aie::load_v<8>(kq1 + {{ D }} + 8 * b), aie::load_v<8>(kq1 + {{ 3 * D }} + 8 * b)));
        const aie::vector<float, 32> x = aie::add(aie::accum<accfloat, 32>(h).template to_vector<float>(), aie::accum<accfloat, 32>(l).template to_vector<float>());
        po.push({{ p }}_ahilo(aie::accum<accfloat, 32>(x), nI));
    }
}

// the pair's 4 vectors (k0 = y0 + D, q0 = y0, k1 = y1 + D, q1 = y1: {{ D }} bf16) -> x / sqrt(sum x^2 + eps) as bf16
// hi, lo at r0, r0 + {{ 2 * D }}, r1, r1 + {{ 2 * D }} (lo {{ D }} after hi): one call, the 4 reductions and scalar steps
// overlap (424 cycles; per vector: 4 x 150)
__attribute__((noinline)) void {{ p }}_l2b4(const bfloat16 *__restrict y0, const bfloat16 *__restrict y1,
                                          bfloat16 *__restrict r0, bfloat16 *__restrict r1)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *__restrict x[4] = {y0 + {{ D }}, y0, y1 + {{ D }}, y1};
    bfloat16 *__restrict o[4] = {r0, r0 + {{ 2 * D }}, r1, r1 + {{ 2 * D }}};
    float s[4];
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 4; i++) {
        aie::accum<accfloat, 64> ss = aie::mac_square(aie::mul_square(aie::load_v<64>(x[i])), aie::load_v<64>(x[i] + 64));
        s[i] = aie::reduce_add(ss.template to_vector<float>());
    }
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 4; i++) {
        const aie::vector<float, 64> iv = aie::broadcast<float, 64>(al::inv_rms<1>(s[i], {{ eps }}f));
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 2; k++) {
            aie::vector<bfloat16, 64> a, b;
            {{ g }}_split(al::mulff<64>(aie::accum<accfloat, 64>(aie::load_v<64>(x[i] + 64 * k)).template to_vector<float>(), iv), a, b);
            aie::store_v(o[i] + 64 * k, a);
            aie::store_v(o[i] + {{ D }} + 64 * k, b);
        }
    }
}

// gs[0 .. 3] = k0 . q0, k1 . q1, k0 . k1, k0 . q1 of the pair's [hi | lo] vectors (rec: k0 | q0, k1q: k1 | q1; one
// call: 136 cycles, per product 76)
__attribute__((noinline)) void {{ p }}_dot4(const bfloat16 *__restrict rec, const bfloat16 *__restrict k1q,
                                          float *__restrict gs)
{
    const bfloat16 *__restrict a[4] = {rec, k1q, rec, rec};
    const bfloat16 *__restrict b[4] = {rec + {{ 2 * D }}, k1q + {{ 2 * D }}, k1q, k1q + {{ 2 * D }}};
    aie::accum<accfloat, 64> s[4];
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 4; i++) {
        s[i] = aie::zeros<accfloat, 64>();
        AIE_LOOP_UNROLL_FULL
        for (int j = 0; j < {{ D }}; j += 64) {
            const aie::vector<bfloat16, 64> ah = aie::load_v<64>(a[i] + j), al_ = aie::load_v<64>(a[i] + {{ D }} + j);
            const aie::vector<bfloat16, 64> bh = aie::load_v<64>(b[i] + j), bl = aie::load_v<64>(b[i] + {{ D }} + j);
            s[i] = aie::mac(aie::mac(aie::mac(s[i], ah, bh), ah, bl), al_, bh);
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 4; i++)
        gs[i] = aie::reduce_add(s[i].template to_vector<float>());
}

// y[t][0 .. N) = bf16(sum over the taps of w (hi + lo) x) of 8 tokens t: x rows t .. t + 3 of r (pitch {{ NCH }}: the
// window's 3 rows, then the tokens'), w [tap 4][hi, lo][N]
template <int N>
__attribute__((noinline)) void {{ p }}_conv8(const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                           bfloat16 *__restrict y)
{
    for (int c = 0; c < N; c += 32) {
        aie::vector<bfloat16, 32> wh[4], wl[4];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 4; t++) {
            wh[t] = aie::load_v<32>(w + 2 * N * t + c);
            wl[t] = aie::load_v<32>(w + 2 * N * t + N + c);
        }
        const bfloat16 *__restrict pr = r + c;
        bfloat16 *__restrict py = y + c;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < 8; t++) {
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < 4; k++) {
                const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + {{ NCH }} * k);
                acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
            }
            aie::store_v(py, acc.template to_vector<bfloat16>());
            pr += {{ NCH }};
            py += {{ NCH }};
        }
    }
}

}  // namespace

extern "C" {

// The first objects: k = 0: the q / k conv weights ([tap][hi, lo][256]); k = 1: the conv state rows (10 read, 3
// used); k = 2: the column's dt, ssm_a (fp32 [16] each, at 512 + 128 c bytes of P's head); k = 3: the token mask
// (fp32 [T]: 1 a valid token, 0 padding)
void {{ p }}_pinit(const bfloat16 *__restrict b, int32_t *__restrict ws, int32_t k)
{
    if (k == 0) {
        al::copy<{{ 8 * 256 }}, 64>(SW(bfloat16, {{ w.wc }}), b);
    } else if (k == 1) {
        al::copy<{{ 3 * 256 }}, 64>(SW(bfloat16, {{ w.rows }}), b);
    } else if (k == 2) {
        al::copy<32, 16>(SW(float, {{ w.gp }}), (const float *)b);
    } else {
        al::copy<{{ T }}, 16>(SW(float, {{ w.mask }}), (const float *)b);
    }
}

// 8 tokens 8 g8 .. of the core's k head kh: b [ab slot 8 x 64 (alpha HC | beta HC bf16) | q 8 x 128 | k 8 x 128]
// -> out: the 4 pairs' records (GdnScan's B objects): [k0 hi | k0 lo | q0 hi | q0 lo | the same of token 1 (128 bf16
// each) | gs: d0, beta0, d1, beta1 of the k head's {{ VH }} v heads h = kh + 2 hh (3 slots each), k0 . q0, k1 . q1,
// k0 . k1, k0 . q1 (fp32) | the pair's MMUL A panels (16 x 72 B)]
void {{ p }}_pgroup(const bfloat16 *__restrict b, int32_t *__restrict ws, bfloat16 *__restrict out, int32_t kh,
                    int32_t g8)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict r = SW(bfloat16, {{ w.rows }});
    for (int i = 0; i < 8; i++) {
        bfloat16 *__restrict ri = r + 256 * (3 + i);
        al::copy<128, 64>(ri, b + 512 + 128 * i);
        al::copy<128, 64>(ri + 128, b + 1536 + 128 * i);
    }
    bfloat16 *__restrict yc = SW(bfloat16, {{ w.yc }});
    bfloat16 *__restrict ys = SW(bfloat16, {{ w.ys }});
    {{ p }}_conv8<256>(r, SW(bfloat16, {{ w.wc }}), yc);
    {{ lut }}_silu_lut<{{ 8 * 256 }}>(yc, ys);
    al::copy<{{ 3 * 256 }}, 32>(r, r + {{ 8 * 256 }});  // (the last 3 tokens: the next group's window)
    // the gates of the k head's v heads h = kh, kh + 2, kh + 4, times the mask; two heads per call (lanes [head 2][the
    // 8 tokens]: one call each of softplus, expm1, sigmoid for a pair of heads instead of one per head on 8 lanes)
    const float *__restrict gp = SW(float, {{ w.gp }});
    // (8 lanes, 32 B: a 64-B load at an odd group's 32-B offset would round the address down)
    const aie::vector<float, 8> m8 = aie::load_v<8>(SW(float, {{ w.mask }}) + 8 * g8);
    const aie::vector<float, 16> m = aie::concat(m8, m8);
    float *__restrict gt = SW(float, {{ w.gt }});  // [d 3 | beta 3][16 lanes: the tokens]
    const uint16_t *__restrict ab = (const uint16_t *)b;
{% for hp in range((VH + 1) // 2) %}
{% set h1 = 2 * hp + 1 if 2 * hp + 1 < VH else 2 * hp %}
    {
        const int h0 = kh + {{ 4 * hp }}, h1 = kh + {{ 2 * h1 }};
        aie::vector<int32, 16> ai, bi;
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 8; t++) {
            ai[t] = (int32_t)ab[64 * t + h0] << 16;
            ai[8 + t] = (int32_t)ab[64 * t + h1] << 16;
            bi[t] = (int32_t)ab[64 * t + {{ HC }} + h0] << 16;
            bi[8 + t] = (int32_t)ab[64 * t + {{ HC }} + h1] << 16;
        }
        const aie::vector<float, 16> sp = {{ g }}_softplus(
            aie::add(ai.template cast_to<float>(), aie::concat(aie::broadcast<float, 8>(gp[h0]), aie::broadcast<float, 8>(gp[h1]))));
        const aie::vector<float, 16> d = al::mulff<16>(
            {{ g }}_expm1(al::mulff<16>(sp, aie::concat(aie::broadcast<float, 8>(gp[16 + h0]), aie::broadcast<float, 8>(gp[16 + h1])))), m);
        const aie::vector<float, 16> be = al::mulff<16>({{ g }}_sigmoid(bi.template cast_to<float>()), m);
        aie::store_v(gt + {{ 32 * hp }}, d.template extract<8>(0));
        aie::store_v(gt + {{ 48 + 32 * hp }}, be.template extract<8>(0));
{% if 2 * hp + 1 < VH %}
        aie::store_v(gt + {{ 32 * hp + 16 }}, d.template extract<8>(1));
        aie::store_v(gt + {{ 48 + 32 * hp + 16 }}, be.template extract<8>(1));
{% endif %}
    }
{% endfor %}
    for (int i = 0; i < 8; i += 2) {
        bfloat16 *__restrict rec = out + {{ REC_ELEMS }} * (i >> 1);
        const bfloat16 *__restrict y0 = ys + 256 * i;
        const bfloat16 *__restrict y1 = y0 + 256;
        bfloat16 *__restrict k1q = rec + {{ 4 * D }};
        {{ p }}_l2b4(y0, y1, rec, k1q);
        float *__restrict gs = (float *)(rec + {{ G_OFF }});
        for (int hh = 0; hh < {{ VH }}; hh++) {
            gs[hh] = gt[16 * hh + i];
            gs[3 + hh] = gt[48 + 16 * hh + i];
            gs[6 + hh] = gt[16 * hh + i + 1];
            gs[9 + hh] = gt[48 + 16 * hh + i + 1];
        }
        {{ p }}_dot4(rec, k1q, gs + 12);
        {{ p }}_apanels(rec, k1q, (uint8_t *)(rec + {{ AP_OFF }}));
    }
}

}  // extern "C"
""")


class GdnPrepOperator(ArtcOperator):
    """The q / k side of a GDN layer's chunk, once per k head (module docstring): two cores per column (rows 2, 3: the
    column's k heads), per 8 tokens the conv of the head's q / k channels, SiLU, l2 norms (bf16 hi + lo), per pair the
    dot products and the MMUL A panels, and the gates (d = expm1(g), beta) of the k head's 3 v heads times the token
    mask, as records (``REC_ELEMS`` bf16 per pair) that GdnScan streams. Runtime sequence: G (gdn_in's output,
    ``layout``), CV (``CV_ROWS`` rows of G's layout: the conv state in rows 0 .. 2), P (``scan_params``), M (fp32 per
    token: 1 valid, 0 padding; ``MASK_ELEMS`` bf16 read), PR (the records: [column][k head][pair])."""

    STACK = 5120
    CV_ROWS = 10  # rows of CV read (the q / k conv state: one object of 10 rows x 256)
    MASK_ELEMS = B_ELEMS  # bf16 of the mask argument (an object; T fp32 used)
    pfx = kernel_prefix("gdp")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout):
        assert T % 8 == 0 and T // 8 <= 64 and 4 * (T + 16) <= 2 * self.MASK_ELEMS
        self.T, self.cfg, self.layout = T, cfg, layout
        self.sh = GD.GdnShape(cfg)
        self.hc = self.sh.hc
        assert {"qkv", "z", "ab"} <= set(layout.off) and self.hc % 2 == 0 and self.hc <= 6, \
            "two k heads of up to 3 v heads per column"
        super().__init__()

    def workspace(self) -> dict[str, int]:
        sizes = [("wc", 2 * 8 * 256), ("rows", 2 * 11 * 256), ("yc", 2 * 8 * 256), ("ys", 2 * 8 * 256), ("gp", 4 * 32),
                 ("mask", 4 * (self.T + 16)), ("gt", 4 * 96)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        helpers = JINJA_ENV.from_string(GD.HELPERS).render(p=GD.GDN, eps=repr(self.cfg.rms_norm_eps), ln2=bf16_hilo(math.log(2)),
                                                            D=D)
        return _PREP.render(p=p, g=GD.GDN, helpers=helpers, silu=act_lut.render_silu(), lut=act_lut.LUT, w=Attr(self.workspace()), D=D, NCH=256,
                            HC=self.hc, VH=self.hc // 2, T=self.T, REC_ELEMS=REC_ELEMS, G_OFF=G_OFF, AP_OFF=AP_OFF,
                            eps=repr(self.cfg.rms_norm_eps))

    @property
    def n_pr(self) -> int:
        """bf16 of PR: per column and k head the chunk's pairs' records."""
        return COLS * 2 * self.T // 2 * REC_ELEMS

    def get_arg_spec(self) -> list[ArgSpec]:
        n_p = P_HEAD + COLS * self.hc * REC
        return [ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (n_p // 2,), torch.bfloat16),
                ArgSpec("in", (self.MASK_ELEMS,), torch.bfloat16),
                ArgSpec("out", (self.n_pr,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay = self.pfx, self.T, self.hc, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        b_ty = np.ndarray[(B_ELEMS,), bf]
        r_ty = np.ndarray[(4 * REC_ELEMS,), bf]
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        k_init = Kernel(f"{p}_pinit", "kernel.o", [b_ty, ws_ty, np.int32])
        k_group = Kernel(f"{p}_pgroup", "kernel.o", [b_ty, ws_ty, r_ty, np.int32, np.int32])
        i_f = [[ObjectFifo(b_ty, name=f"gp_i{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        o_f = [[ObjectFifo(r_ty, name=f"gp_o{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        n_g = T // 8

        def core(i_h, o_h, init, group, ws, kh):
            for _ in forever_():
                for k in range_(4):
                    b = i_h.acquire(1)
                    init(b, ws, k)
                    i_h.release(1)
                for i in range_(n_g):
                    b = i_h.acquire(1)
                    o = o_h.acquire(1)
                    group(b, ws, o, kh, i)
                    o_h.release(1)
                    i_h.release(1)

        workers = [Worker(core, [i_f[c][kh].cons(), o_f[c][kh].prod(), k_init, k_group,
                                 Buffer(ws_ty, name=f"gp_ws{c}_{kh}"), kh],
                          tile=Tile(c, 2 + kh), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for kh in range(2)]
        N, pitch = lay.N, lay.pitch
        n_p = P_HEAD + COLS * hc * REC
        n_cv = self.CV_ROWS * N
        n_pr, per = self.n_pr, T // 2 * REC_ELEMS  # (a column's k head's records)

        def seq(G, CV, P, M, PR, ih, oh):
            tg = TaskGroup()
            for c in range(COLS):
                qkv = c * pitch + lay.off["qkv"]
                for kh in range(2):
                    rec = (P_HEAD + (c * hc + kh) * REC) // 2  # (the q / k conv weights of v head kh's record)
                    f = ih[c][kh]
                    f.fill(P, TensorAccessPattern((n_p // 2,), rec, [1, 1, 1, B_ELEMS], [0, 0, 0, 1]), group=tg)
                    f.fill(CV, TensorAccessPattern((n_cv,), qkv + D * kh, [1, self.CV_ROWS, 2, D], [0, N, 2 * D, 1]),
                           group=tg)
                    f.fill(P, TensorAccessPattern((n_p // 2,), (512 + 128 * c) // 2, [1, 1, 1, B_ELEMS], [0, 0, 0, 1]),
                           group=tg)
                    f.fill(M, TensorAccessPattern((self.MASK_ELEMS,), 0, [1, 1, 1, B_ELEMS], [0, 0, 0, 1]), group=tg)
                    chained_fill(f, G, [TensorAccessPattern((T * N,), c * pitch + lay.off["ab"], [n_g, 1, 8, 64],
                                                            [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + D * kh, [n_g, 1, 8, D], [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + 2 * D + D * kh, [n_g, 1, 8, D],
                                                            [8 * N, 0, N, 1])], tg)
                    oh[c][kh].drain(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [1, 1, 1, per], [0, 0, 0, 1]),
                                    group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf], np.ndarray[(n_p // 2,), bf],
                           np.ndarray[(self.MASK_ELEMS,), bf], np.ndarray[(n_pr,), bf],
                           [[f.prod(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(i_f)],
                           [[f.cons(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(o_f)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


class GdnScanOperator(ArtcOperator):
    """The delta rule of a GDN layer over a chunk of T tokens (module docstring), on GdnPrep's records. Runtime
    sequence: PR (GdnPrep's records), G (gdn_in's output, ``layout``: the v rows), CV (``CV_ROWS`` rows of G's layout:
    the conv state in rows 0 .. 2), P (``scan_params``: the v conv weights), GSTATE (the layer's state, per column its
    panels: read and written), O (T x hv x 128 bf16: o of every v head)."""

    STACK = 5120
    CV_ROWS = 8  # rows of CV read (the v quarter's conv state: [8][32])
    pfx = kernel_prefix("gdsc")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, ablate: tuple[str, ...] = ()):
        """``ablate`` (timing experiments, wrong results): kernel parts compiled out ("conv", "silu": of the v quarter,
        "pairs": the token pairs, "panel": the panel calls only); event windows (results unchanged): "ev_group" a group
        call, "ev_panel" a pair's 4 panel calls."""
        assert T % 8 == 0 and T <= 512, "8-token groups, a BD iterates at most 64 times"
        self.T, self.cfg, self.layout = T, cfg, layout
        self.ablate = tuple(ablate)
        self.sh = GD.GdnShape(cfg)
        self.hc, self.ch, self.hv = self.sh.hc, self.sh.ch, cfg.ssm_v_heads
        assert {"qkv", "z", "ab"} <= set(layout.off) and self.hc % 2 == 0 and self.hc <= 6
        super().__init__()

    def workspace(self) -> dict[str, int]:
        sizes = [("wc", 2 * 8 * 32), ("rows", 2 * 11 * 32), ("yc", 2 * 8 * 32), ("ys", 2 * 8 * 32), ("ob", 4 * 64),
                 ("g", 4 * 16), ("cw", 4 * 128), ("eb", 2 * 128)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        helpers = JINJA_ENV.from_string(GD.HELPERS).render(p=GD.GDN, eps=repr(self.cfg.rms_norm_eps), ln2=bf16_hilo(math.log(2)),
                                                            D=D)
        return _SCAN.render(p=p, g=GD.GDN, helpers=helpers, silu=act_lut.render_silu(), lut=act_lut.LUT, w=Attr(self.workspace()), D=D, NCH=32,
                            PART=PART, rs=bf16_hilo(1 / math.sqrt(D)), REC_ELEMS=REC_ELEMS, G_OFF=G_OFF, AP_OFF=AP_OFF,
                            ablate=self.ablate)

    def get_arg_spec(self) -> list[ArgSpec]:
        n_p = P_HEAD + COLS * self.hc * REC
        return [ArgSpec("in", (COLS * 2 * self.T // 2 * REC_ELEMS,), torch.bfloat16),
                ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (n_p // 2,), torch.bfloat16),
                ArgSpec("in", (COLS * self.sh.n_panels * PANEL // 2,), torch.bfloat16),
                ArgSpec("out", (self.T * self.hv * D,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay = self.pfx, self.T, self.hc, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        b_ty = np.ndarray[(4 * REC_ELEMS,), bf]
        part_ty = np.ndarray[(PART,), bf]
        quad_ty = np.ndarray[(4 * PART,), bf]
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        S_ty = np.ndarray[(4 * PANEL // 4,), np.dtype[np.float32]]
        k_vinit = Kernel(f"{p}_vinit", "kernel.o", [part_ty, ws_ty, np.int32])
        k_sin = Kernel(f"{p}_sin", "kernel.o", [part_ty, S_ty, np.int32])
        k_sout = Kernel(f"{p}_sout", "kernel.o", [S_ty, part_ty, np.int32])
        k_group = Kernel(f"{p}_group", "kernel.o", [b_ty, part_ty, S_ty, ws_ty, part_ty, np.int32])

        b3 = [ObjectFifo(b_ty, name=f"gs_b{c}", depth=2) for c in range(COLS)]
        b1 = [b3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"gs_bm{c}") for c in range(COLS)]
        v3 = [ObjectFifo(quad_ty, name=f"gs_v{c}", depth=2) for c in range(COLS)]
        v1 = [v3[c].cons().split([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[part_ty] * 4,
                                 names=[f"gs_vp{c}_{r}" for r in range(4)]) for c in range(COLS)]
        o3 = [ObjectFifo(quad_ty, name=f"gs_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[part_ty] * 4, names=[f"gs_op{c}_{r}" for r in range(4)]) for c in range(COLS)]
        n_g = T // 8

        def core(b_f, v_f, o_f, vinit, sin, sout, group, ws, S):
            for _ in forever_():
                for h in range_(hc):
                    for k in range_(2):
                        v = v_f.acquire(1)
                        vinit(v, ws, k)
                        v_f.release(1)
                    for k in range_(N_SPART):
                        v = v_f.acquire(1)
                        sin(v, S, k)
                        v_f.release(1)
                    for _ in range_(n_g):
                        b = b_f.acquire(1)
                        v = v_f.acquire(1)
                        o = o_f.acquire(1)
                        group(b, v, S, ws, o, h)
                        o_f.release(1)
                        v_f.release(1)
                        b_f.release(1)
                    for k in range_(N_SPART):
                        o = o_f.acquire(1)
                        sout(S, o, k)
                        o_f.release(1)

        workers = []
        for c in range(COLS):
            for r in range(4):
                ws = Buffer(ws_ty, name=f"gs_ws{c}_{r}")  # (every byte written before it is read: no PDI zeroes)
                S = Buffer(S_ty, name=f"gs_S{c}_{r}")
                workers.append(Worker(core, [b1[c].cons(), v1[c][r].cons(), o1[c][r].prod(), k_vinit, k_sin, k_sout,
                                             k_group, ws, S],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))

        N, pitch = lay.N, lay.pitch
        n_p = P_HEAD + COLS * hc * REC
        n_cv = self.CV_ROWS * N
        n_st = COLS * self.sh.n_panels * PANEL // 2
        n_o = T * self.hv * D
        n_pr, per = COLS * 2 * T // 2 * REC_ELEMS, T // 2 * REC_ELEMS

        def seq(PR, G, CV, P, GS, O, bh, vh, oh):
            for h in range(hc):  # (a round on every column at once)
                kh = h & 1
                tg = TaskGroup()
                for c in range(COLS):
                    qkv = c * pitch + lay.off["qkv"]  # (the column's conv channels: rows of CV and G alike)
                    rec = (P_HEAD + (c * hc + h) * REC) // 2
                    st = (c * self.sh.n_panels + 16 * h) * PANEL // 2  # the head's state (4 quarters of 16 KB)
                    bh[c].fill(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [1, 1, 1, per], [0, 0, 0, 1]),
                               group=tg)
                    vh[c].fill(P, TensorAccessPattern((n_p // 2,), rec + B_ELEMS, [1, 1, 1, 4 * PART], [0, 0, 0, 1]),
                               group=tg)
                    vh[c].fill(CV, TensorAccessPattern((n_cv,), qkv + 4 * D + D * h, [1, 4, 8, 32], [0, 32, N, 1]),
                               group=tg)
                    vh[c].fill(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART], [PART, 4 * PANEL // 2, 0, 1]),
                               group=tg)
                    vh[c].fill(G, TensorAccessPattern((T * N,), qkv + 4 * D + D * h, [n_g, 4, 8, 32], [8 * N, 32, N, 1]),
                               group=tg)
                    j = gdn_v_heads(self.cfg, c)[h]
                    oh[c].drain(O, TensorAccessPattern((n_o,), D * j, [n_g, 4, 8, 32], [8 * self.hv * D, 32, self.hv * D, 1]),
                                group=tg)
                    oh[c].drain(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART], [PART, 4 * PANEL // 2, 0, 1]),
                                group=tg, wait=True)
                tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_pr,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf],
                           np.ndarray[(n_p // 2,), bf], np.ndarray[(n_st,), bf], np.ndarray[(n_o,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(v3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_OUT = JINJA_ENV.from_string(r"""
namespace {
{{ silu }}
}  // namespace

extern "C" {

// The first object: ssm_norm (128 fp32)
void {{ p }}_norm(const bfloat16 *__restrict a, float *__restrict ws)
{
    al::copy<{{ D }}, 32>(ws, (const float *)a);
}

// A conv state object (the last valid tokens' qkv rows) -> out
void {{ p }}_copy(const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    al::copy<{{ OBJ }}, 64>(c, b);
}

// 1 / rms of the 16 (token, head) rows of o -> ivt [16][hi 32 | lo 32] (bf16)
[[clang::always_inline]] inline void {{ p }}_rms16(const bfloat16 *__restrict o, bfloat16 *__restrict ivt)
{
    const bfloat16 *__restrict po = o;
    bfloat16 *__restrict pv = ivt;
    AIE_LOOP_NO_UNROLL
    for (int th = 0; th < 16; th++) {
        aie::accum<accfloat, 32> ss = aie::zeros<accfloat, 32>();
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < {{ D }}; i += 32)
            ss = aie::mac_square(ss, aie::load_v<32>(po + i));
        aie::vector<bfloat16, 32> h, l;
        al::bf16_hilo<32>(al::inv_rms<{{ D }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f), h, l);
        aie::store_v(pv, h);
        aie::store_v(pv + 32, l);
        po += {{ D }};
        pv += 64;
    }
}

// y = mulff(mulff(o, 1 / rms), w) silu(z), term by term: o and silu(z) are bf16 (their lo parts 0), the weight's and
// 1 / rms's hi / lo from tables (nwt [4][hi 32 | lo 32]); distinct restrict parameters (the tables and y share a
// workspace: as pointers into it the loop would carry a store -> load dependence, RecMII = its 48-bundle body)
[[clang::always_inline]] inline void {{ p }}_rmsw(const bfloat16 *__restrict o_, const bfloat16 *__restrict zs_,
                                                  const bfloat16 *__restrict ivt_, const bfloat16 *__restrict nwt_,
                                                  float *__restrict y)
{
    const bfloat16 __aie_dm_resource_a *__restrict po = (const bfloat16 __aie_dm_resource_a *)o_;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)zs_;
    const bfloat16 __aie_dm_resource_c *__restrict piv = (const bfloat16 __aie_dm_resource_c *)ivt_;
    const bfloat16 __aie_dm_resource_d *__restrict pn = (const bfloat16 __aie_dm_resource_d *)nwt_;
    float *__restrict py = y;
    dims_2d_t div = dims_2d_from_steps({{ D // 32 }}, 0, 64);  // (a row's 1 / rms for its blocks)
    dims_2d_t dnw = dims_2d_from_steps({{ D // 32 }}, 64, 0);  // (the weight's blocks, again per row)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < {{ 16 * D // 32 }}; j++) {
        const aie::vector<bfloat16, 32> ov = aie::load_v<32>(po), sv = aie::load_v<32>(ps);
        const aie::vector<float, 32> v1 =
            aie::mac(aie::mul(ov, aie::load_v<32>(piv)), ov, aie::load_v<32>(piv + 32)).template to_vector<float>();
        const aie::vector<bfloat16, 32> v1h = aie::accum<accfloat, 32>(v1).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> v1l =
            aie::accum<accfloat, 32>(aie::sub(v1, aie::accum<accfloat, 32>(v1h).template to_vector<float>())).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> wh = aie::load_v<32>(pn), wl = aie::load_v<32>(pn + 32);
        const aie::vector<float, 32> v2 = aie::mac(aie::mac(aie::mul(v1h, wh), v1h, wl), v1l, wh).template to_vector<float>();
        const aie::vector<bfloat16, 32> v2h = aie::accum<accfloat, 32>(v2).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> v2l =
            aie::accum<accfloat, 32>(aie::sub(v2, aie::accum<accfloat, 32>(v2h).template to_vector<float>())).template to_vector<bfloat16>();
        aie::store_v(py, aie::mac(aie::mul(v2h, sv), v2l, sv).template to_vector<float>());
        po += 32;
        ps += 32;
        py += 32;
        piv = add_2d_ptr(piv, div);
        pn = add_2d_ptr(pn, dnw);
    }
}

// 8 tokens of a pair of v heads: o, z [8][2 x 128] -> rms(o) ssm_norm silu(z) as 32 bfp16 panels (8 tokens x 8 dims)
// (4565 cycles: the rows' 1 / rms first, then one pipelined loop, II 14 per 32; per-row loops, unpipelined: 7713)
void {{ p }}_pair(const bfloat16 *__restrict o, const bfloat16 *__restrict z, float *__restrict ws,
                  bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict nw = ws;
    float *__restrict y = ws + {{ D }};  // [8][256]
    bfloat16 *__restrict zs = (bfloat16 *)(ws + {{ D + 8 * 256 }});  // silu(z) [8][256]
    bfloat16 *__restrict ivt = (bfloat16 *)(ws + {{ D + 8 * 256 + 4 * 256 }});  // [16][hi 32 | lo 32]
    bfloat16 *__restrict nwt = ivt + 1024;  // [{{ D // 32 }}][hi 32 | lo 32]
    {{ lut }}_silu_lut<{{ OBJ }}>(z, zs);
    for (int i = 0; i < {{ D // 32 }}; i++) {
        const aie::vector<float, 32> w = aie::load_v<32>(nw + 32 * i);
        const aie::vector<bfloat16, 32> h = aie::accum<accfloat, 32>(w).template to_vector<bfloat16>();
        aie::store_v(nwt + 64 * i, h);
        aie::store_v(nwt + 64 * i + 32,
                     aie::accum<accfloat, 32>(aie::sub(w, aie::accum<accfloat, 32>(h).template to_vector<float>())).template to_vector<bfloat16>());
    }
    {{ p }}_rms16(o, ivt);
    {{ p }}_rmsw(o, zs, ivt, nwt, y);
    al::BOut<> pb((bfp16ebs8 *)out);
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < 32; kb++) {
        const float *__restrict py = y + 8 * kb;
        const aie::vector<float, 64> v = aie::concat(
            aie::concat(aie::concat(aie::load_v<8>(py), aie::load_v<8>(py + 256)),
                        aie::concat(aie::load_v<8>(py + 512), aie::load_v<8>(py + 768))),
            aie::concat(aie::concat(aie::load_v<8>(py + 1024), aie::load_v<8>(py + 1280)),
                        aie::concat(aie::load_v<8>(py + 1536), aie::load_v<8>(py + 1792))));
        pb.push(aie::accum<accfloat, 64>(v).template to_vector<bfp16ebs8>());
    }
}

}  // extern "C"
""")


class GdnOutOperator(ArtcOperator):
    """The gated output of a GDN layer's chunk and its new conv state (module docstring). Runtime sequence: O (the
    scan's), G (z), GC (``gc_rows`` rows of G's layout: the prompt chunk's; the conv state rows from the scratchpad
    parameter ``gdn_conv``: ``conv_offset``), P (ssm_norm first), GHDR (the conv states written), A (the ssm_out
    GEMM's panels, T x hv 128)."""

    STACK = 4096
    OBJ = 2048  # bf16 of an input object: 8 tokens x a pair of heads
    pfx = kernel_prefix("gdo")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, hslot: int, cs_off: int, gc_rows: int = 0):
        assert T % 8 == 0 and T // 8 <= 64
        self.T, self.cfg, self.layout, self.hslot, self.cs_off = T, cfg, layout, hslot, cs_off
        self.gc_rows = gc_rows or T
        self.sh = GD.GdnShape(cfg)
        self.hc, self.ch, self.hv = self.sh.hc, self.sh.ch, cfg.ssm_v_heads
        assert self.hc % 2 == 0 and 3 * self.ch <= 2 * self.OBJ
        assert cs_off + 4 * self.OBJ <= hslot, "the conv-state objects end in the column's slot"
        super().__init__()

    def get_kernel_src(self) -> str:
        p = self.pfx
        return _OUT.render(p=p, silu=act_lut.render_silu(), lut=act_lut.LUT, D=D, OBJ=self.OBJ, eps=repr(self.cfg.rms_norm_eps))

    def conv_offset(self, n: int) -> int:
        """The ``gdn_conv`` parameter of a chunk of n valid tokens: GC's first element of row n - 3."""
        assert 3 <= n <= self.gc_rows, "the conv state: three tokens of the chunk"
        return (n - 3) * self.layout.N

    @property
    def ghdr_bytes(self) -> int:
        return COLS * self.hslot

    def get_arg_spec(self) -> list[ArgSpec]:
        T, hv = self.T, self.hv
        return [ArgSpec("in", (T * hv * D,), torch.bfloat16),
                ArgSpec("in", (T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.gc_rows * self.layout.N,), torch.bfloat16),
                ArgSpec("in", ((P_HEAD + COLS * self.hc * REC) // 2,), torch.bfloat16),
                ArgSpec("out", (self.ghdr_bytes // 2,), torch.bfloat16),
                ArgSpec("out", (T * hv * D * 9 // 16,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay, hv, ch = self.pfx, self.T, self.hc, self.layout, self.hv, self.ch
        bf = np.dtype[bfloat16]
        obj_ty = np.ndarray[(self.OBJ,), bf]
        pan_ty = np.ndarray[(32 * 36,), bf]  # 32 panels of 72 B
        ws_ty = np.ndarray[(D + 8 * 256 + 4 * 256 + 512 + D // 32 * 32,), np.dtype[np.float32]]  # norm | y | silu(z) (bf16) | 1 / rms | norm hi / lo
        k_norm = Kernel(f"{p}_norm", "kernel.o", [obj_ty, ws_ty])
        k_copy = Kernel(f"{p}_copy", "kernel.o", [obj_ty, obj_ty])
        k_pair = Kernel(f"{p}_pair", "kernel.o", [obj_ty, obj_ty, ws_ty, pan_ty])
        conv = ScratchpadParameter(CONV_OFF, np.int32)
        a_f = [ObjectFifo(obj_ty, name=f"go_a{c}", depth=2) for c in range(COLS)]
        b_f = [ObjectFifo(obj_ty, name=f"go_b{c}", depth=2) for c in range(COLS)]
        c_f = [ObjectFifo(pan_ty, name=f"go_c{c}", depth=2) for c in range(COLS)]
        d_f = [ObjectFifo(obj_ty, name=f"go_d{c}", depth=1) for c in range(COLS)]
        n_g, npair = T // 8, hc // 2

        def core(a_h, b_h, c_h, d_h, norm, copy, pair, ws):
            for _ in forever_():
                a = a_h.acquire(1)
                norm(a, ws)
                a_h.release(1)
                for _ in range_(2):
                    b = b_h.acquire(1)
                    d = d_h.acquire(1)
                    copy(b, d)
                    d_h.release(1)
                    b_h.release(1)
                for _ in range_(n_g * npair):
                    a = a_h.acquire(1)
                    b = b_h.acquire(1)
                    c = c_h.acquire(1)
                    pair(a, b, ws, c)
                    c_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        workers = [Worker(core, [a_f[c].cons(), b_f[c].cons(), c_f[c].prod(), d_f[c].prod(), k_norm, k_copy, k_pair,
                                 Buffer(ws_ty, name=f"go_ws{c}")], tile=Tile(c, 2), stack_size=self.STACK,
                          dynamic_objfifo_lowering=True, while_true=False) for c in range(COLS)]
        N, n_o, n_p = lay.N, T * hv * D, (P_HEAD + COLS * hc * REC) // 2
        n_hdr, n_a = self.ghdr_bytes // 2, T * hv * D * 9 // 16
        kb_row = hv * D // 8 * 36  # (bf16 of a token group's panels)

        n_gc = self.gc_rows * N

        def seq(O, G, GC, P, GH, A, ah, bh, ch_, dh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                ah[c].fill(P, TensorAccessPattern((n_p,), 0, [1, 1, 1, self.OBJ], [0, 0, 0, 1]), group=tg)
                ah[c].fill(O, TensorAccessPattern((n_o,), 2 * c * D, [n_g, npair, 8, 2 * D], [8 * hv * D, 16 * D, hv * D, 1]),
                           group=tg)
                qkv = c * lay.pitch + lay.off["qkv"]
                bh[c].fill(GC, TensorAccessPattern((n_gc,), qkv, [1, 1, 3, ch], [0, 0, N, 1]), offset_parameter=conv,
                           group=tg)
                bh[c].fill(GC, TensorAccessPattern((n_gc,), 0, [1, 1, 1, 2 * self.OBJ - 3 * ch], [0, 0, 0, 1]),
                           group=tg)
                bh[c].fill(G, TensorAccessPattern((T * N,), c * lay.pitch + lay.off["z"], [n_g, npair, 8, 2 * D],
                                                  [8 * N, 2 * D, N, 1]), group=tg)
                # (the conv state and the objects' padding after it: the bytes the decode's conv-state drain writes)
                cs = (c * self.hslot + self.cs_off) // 2
                dh[c].drain(GH, TensorAccessPattern((n_hdr,), cs, [1, 1, 1, 2 * self.OBJ], [0, 0, 0, 1]), group=tg)
                ch_[c].drain(A, TensorAccessPattern((n_a,), 2 * c * 16 * 36, [n_g, npair, 1, 32 * 36],
                                                    [kb_row, 16 * 16 * 36, 0, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_o,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_gc,), bf],
                           np.ndarray[(n_p,), bf], np.ndarray[(n_hdr,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a_f)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b_f)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(c_f)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(d_f)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
