"""The Gated DeltaNet layers of a qwen35 prompt chunk of T tokens (the prefill), on the decode overlay's state
(``overlay.py``: GSTATE, the conv state in GHDR) and the gdn_in GEMMs' output G (``prefill.gemm``: column c's slice
[qkv | z | ab] at c pitch). Three operators per layer:

``GdnPrepOperator``: the q / k side, once per k head: per column two cores per k head (a k core and a q core, each
through its half of the head's input objects: the memtile splits them), per 8 tokens the conv of the head's k (or q)
channels (128), SiLU, l2 norm (q times 1 / sqrt(D): the scores' scale), as MMUL panels of the token block (bfp16 hi +
lo of each [token 8][i 8] block, ~14 bits); the q core also the gates of the k head's v heads, d = expm1(g) and beta,
times a host-written token mask (d = beta = 0 for padding tokens: the scan leaves the state alone through them). Per
16-token chunk the memtile joins the two cores' objects into one record ``JOBJ``: [K panels | Q panels | gates] in PR.

``GdnScanOperator``: the delta rule, chunkwise (the WY / UT form): per column four cores (rows 2 .. 5); a round per v
head of the column, core r taking the head's state quarter r (columns 32 r .. 32 r + 31: 16 KB of fp32, resident in L1
for the round). Per chunk of C = 16 tokens from state S0 (u = s + 1 .. t, s = -1: from the chunk's start):

- E[t][s] = prod_u (1 + d_u) - 1 (a scan of products: no cancellation for weak decays, exact zero for none);
- L = beta (1 + E) K K^T (s < t), M = (1 + E) Q K^T (s <= t), T = (I + L)^-1 per 8-token block (row substitution);
- X^T = (K S0)^T, Y^T = (Q S0)^T on the MMUL (S0 as bfp16 hi + lo against the K / Q panels: three products);
- R^T = beta (V^T - Gamma X^T); U^T = R^T T^T by blocks (U_1 = T11 (R_1 - L10 U_0)); O^T = Gamma Y^T + U^T M^T;
- S = S0 + (Gamma_C - 1) S0 + K^T (Gamma_C / Gamma U) (the accumulator starts at S0: exact fp32).

Every product on the MMUL takes both operands as bfp16 hi + lo (three products): o within ~3e-5 of the exact
recurrence (a token-by-token form: ~1.5e-5), 70x below its bf16 rounding. Per column: the conv (per 8 tokens) and SiLU
of the core's v quarter (32 channels), the conv state the 3 rows of G before the chunk (CV: G's rows before a sub-chunk,
or a layer's own rows). Streams per column:

- B (broadcast to the column's cores): per round the k head's records (PR), one per chunk;
- V (split, 1 KB parts): per round the v conv weights, the v conv state (CV, 16 rows read), the state quarters (16
  parts), per chunk the v quarters (G);
- O (joined, 1 KB parts): per chunk o (the head's 128 values, bf16, into the O buffer at the head's place of the
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
from ..gemm.gemm_q import ChainPart, chained_fill
from ..kernel_source import kernel_prefix
from ..prefill.gemm import Layout
from ..util import Attr, bf16_hilo, regions
from . import gdn as GD
from .image import gdn_v_heads

D, PANEL = GD.D, GD.PANEL
CONV_OFF = "gdn_conv"  # scratchpad parameter: GC's bf16 offset of the conv state rows (``conv_offset``)
C = 16  # tokens per chunk of the scan
B_ELEMS = 2560  # bf16 of a GdnPrep input object: 8 tokens x [ab slot 64 (alpha, beta bf16) | q 128 | k 128]
QB_ELEMS = 1536  # (its q core's part: [ab | q]; the k core's: k)
PART = 512  # bf16 of a scan split / joined part (1 KB: a chunk's v or o quarter, 16 tokens x 32)
N_SPART = 4 * PANEL // (2 * PART)  # parts of a state quarter (16)
PAN = 72  # bytes of a bfp16 panel (8 x 8)
KP_BYTES = 16 * 2 * 2 * PAN  # a chunk's K (or Q) panels: [ib 16][tb 2][hi, lo]
PREP_P = 2 * B_ELEMS  # bytes of a (column, k head) record of P: the prep's first object [q part | k part]
V_P = 4 * 2 * PART  # bytes of a (column, v head) record of P: per quarter a part, its v conv weights (512 B) first
P_HEAD = 4096  # P: ssm_norm (fp32) first


def jobj_bytes(vh: int) -> int:
    """Bytes of a chunk's record (GdnPrep's joined object): [K panels | Q panels | per v head of the k head d (16 fp32),
    beta (16 fp32)]."""
    return 2 * KP_BYTES + 128 * vh


def params_bytes(hc: int) -> int:
    """Bytes of P (``scan_params``) for HC v heads per column."""
    return P_HEAD + COLS * (2 * PREP_P + hc * V_P)


def scan_params(g: GGUFFile, cfg: QwenConfig, layer: int, rows: np.ndarray) -> np.ndarray:
    """P of a GDN layer (uint8): ssm_norm; per column and k head the prep's object [q part: q conv weights [tap 4][hi,
    lo][128] bf16, dt and ssm_a of the k head's v heads (fp32 [16] each) | k part: k conv weights]; per column and v
    head per quarter r its v conv weights [tap 4][hi, lo][32].
    ``rows``: the gdn_in qkv block's rows per column (``Block.rows``: global conv channels)."""
    hc = GD.GdnShape(cfg).hc
    cw = np.asarray(g.tensor(f"blk.{layer}.ssm_conv1d.weight").raw, np.float32)  # (channels, 4)
    dt = np.asarray(g.tensor(f"blk.{layer}.ssm_dt.bias").raw, np.float32).reshape(-1)
    A = np.asarray(g.tensor(f"blk.{layer}.ssm_a").raw, np.float32).reshape(-1)
    norm = np.asarray(g.tensor(f"blk.{layer}.ssm_norm.weight").raw, np.float32).reshape(-1)
    out = np.zeros(params_bytes(hc), np.uint8)
    out[:norm.nbytes] = norm.view(np.uint8)

    def hl(w: np.ndarray) -> np.ndarray:  # (..., ch, 4) -> (..., [tap][hi, lo][ch]) bf16 bits as bytes
        hi, lo = GD.hilo_bits(np.swapaxes(w, -1, -2))
        return np.stack([hi, lo], -2).view(np.uint8).reshape(*w.shape[:-2], -1)

    ch = np.stack([r[r >= 0] for r in rows])  # (COLS, channels): the columns' global conv channels
    prep = out[P_HEAD:P_HEAD + COLS * 2 * PREP_P].reshape(COLS, 2, PREP_P)
    for c in range(COLS):
        vh = gdn_v_heads(cfg, c)
        for kh in range(2):
            prep[c, kh, :2048] = hl(cw[ch[c, D * kh + np.arange(D)]])
            heads = vh[kh::2]
            prm = np.zeros(32, np.float32)
            prm[:len(heads)], prm[16:16 + len(heads)] = dt[heads], A[heads]
            prep[c, kh, 2048:2176] = prm.view(np.uint8)
            prep[c, kh, 3072:] = hl(cw[ch[c, 2 * D + D * kh + np.arange(D)]])
    h = np.arange(hc)
    v = ch[:, (4 * D + D * h[:, None] + np.arange(D)).reshape(hc, 4, 32)]  # (COLS, hc, quarter, 32)
    out[P_HEAD + COLS * 2 * PREP_P:].reshape(COLS, hc, 4, 2 * PART)[..., :512] = hl(cw[v])
    return out


_PREP = JINJA_ENV.from_string(r"""
// The prep core's workspace (offsets in bytes: GdnPrepOperator.workspace())
#define SW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

{{ helpers }}
{{ silu }}
// y[t][0 .. 128) = bf16(sum over the taps of w (hi + lo) x) of 8 tokens t: x rows t .. t + 3 of r (pitch 128: the
// window's 3 rows, then the tokens'), w [tap 4][hi, lo][128]
__attribute__((noinline)) void {{ p }}_conv8(const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                           bfloat16 *__restrict y)
{
    for (int c = 0; c < 128; c += 32) {
        aie::vector<bfloat16, 32> wh[4], wl[4];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 4; t++) {
            wh[t] = aie::load_v<32>(w + 256 * t + c);
            wl[t] = aie::load_v<32>(w + 256 * t + 128 + c);
        }
        const bfloat16 *__restrict pr = r + c;
        bfloat16 *__restrict py = y + c;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < 8; t++) {
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < 4; k++) {
                const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + 128 * k);
                acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
            }
            aie::store_v(py, acc.template to_vector<bfloat16>());
            pr += 128;
            py += 128;
        }
    }
}

// The 8 tokens' rows ys [8][128] (bf16) l2-normalized, times sc (q: 1 / sqrt(D), the scores' scale), as fp32 in the
// panel order xf [ib 16][t 8][i 8]: x 1 / sqrt(sum x^2 + eps) as two bf16 products (the scale's hi and lo)
template <bool Q>
__attribute__((noinline)) void {{ p }}_l2rows(const bfloat16 *__restrict ys, float *__restrict xf)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float s[8];
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 8; t++) {
        const aie::accum<accfloat, 64> ss = aie::mac_square(aie::mul_square(aie::load_v<64>(ys + 128 * t)),
                                                            aie::load_v<64>(ys + 128 * t + 64));
        s[t] = aie::reduce_add(ss.template to_vector<float>());
    }
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 8; t++) {
        float iv = al::inv_rms<1>(s[t], {{ eps }}f);
        if constexpr (Q)
            iv = al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(aie::broadcast<float, 16>(iv))[0];
        aie::vector<bfloat16, 64> ih, il;
        al::bf16_hilo<64>(iv, ih, il);
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 2; k++) {
            const aie::vector<bfloat16, 64> x = aie::load_v<64>(ys + 128 * t + 64 * k);
            const aie::vector<float, 64> y = aie::mac(aie::mul(x, ih), x, il).template to_vector<float>();
            AIE_LOOP_UNROLL_FULL
            for (int b = 0; b < 8; b++)
                aie::store_v(xf + 64 * (8 * k + b) + 8 * t, y.template extract<8>(b));
        }
    }
}

// The token block's panels of xf ([ib][t 8][i 8] fp32): per i-block bfp16 hi, lo at out + 288 ib (the block's slot)
__attribute__((noinline)) void {{ p }}_panels(const float *__restrict xf, uint8_t *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    const float *__restrict px = xf;
    uint8_t *__restrict po = out;
    for (int ib = 0; ib < 16; ib++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(px)), nI, h, l);
        al::BOut<> ps((bfp16ebs8 *)po);
        ps.push(h);
        ps.push(l);
        px += 64;
        po += {{ 4 * PAN }};
    }
}

// The 8 tokens' window, conv, SiLU and panels (b: their raw rows [8][128]; the record's slot of the token block)
template <bool Q>
[[clang::always_inline]] inline void {{ p }}_block(const bfloat16 *__restrict b, int32_t *__restrict ws,
                                                   uint8_t *__restrict out)
{
    bfloat16 *__restrict r = SW(bfloat16, {{ w.rows }});
    al::copy<{{ 8 * 128 }}, 64>(r + {{ 3 * 128 }}, b);
    bfloat16 *__restrict yc = SW(bfloat16, {{ w.yc }});
    bfloat16 *__restrict ys = SW(bfloat16, {{ w.ys }});
    {{ p }}_conv8(r, SW(bfloat16, {{ w.wc }}), yc);
    {{ lut }}_silu_lut<{{ 8 * 128 }}>(yc, ys);
    al::copy<{{ 3 * 128 }}, 32>(r, r + {{ 8 * 128 }});  // (the last 3 tokens: the next block's window)
    {{ p }}_l2rows<Q>(ys, SW(float, {{ w.xf }}));
    {{ p }}_panels(SW(float, {{ w.xf }}), out);
}

}  // namespace

extern "C" {

// The first objects (q core: k = 0 the q conv weights ([tap][hi, lo][128]) then the gate parameters (dt, ssm_a of the
// k head's v heads: fp32 [16] each); k = 1 the conv state rows (12 read, 3 used); k = 2 the token mask (fp32 [T]: 1
// a valid token, 0 padding))
void {{ p }}_qinit(const bfloat16 *__restrict b, int32_t *__restrict ws, int32_t k)
{
    if (k == 0) {
        al::copy<{{ 8 * 128 }}, 64>(SW(bfloat16, {{ w.wc }}), b);
        al::copy<32, 16>(SW(float, {{ w.gp }}), (const float *)(b + {{ 8 * 128 }}));
    } else if (k == 1) {
        al::copy<{{ 3 * 128 }}, 64>(SW(bfloat16, {{ w.rows }}), b);
    } else {
        al::copy<{{ T }}, 16>(SW(float, {{ w.mask }}), (const float *)b);
    }
}

// (k core: the k conv weights, the conv state rows; the mask object is not its)
void {{ p }}_kinit(const bfloat16 *__restrict b, int32_t *__restrict ws, int32_t k)
{
    if (k == 0)
        al::copy<{{ 8 * 128 }}, 64>(SW(bfloat16, {{ w.wc }}), b);
    else if (k == 1)
        al::copy<{{ 3 * 128 }}, 64>(SW(bfloat16, {{ w.rows }}), b);
}

// Token block tb of chunk i (k core): b = its k rows [8][128] -> the K panels of the record (out: [ib][tb][hi, lo])
void {{ p }}_kblock(const bfloat16 *__restrict b, int32_t *__restrict ws, bfloat16 *__restrict out, int32_t i,
                    int32_t tb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ p }}_block<false>(b, ws, (uint8_t *)out + {{ 2 * PAN }} * tb);
}

// Token block tb of chunk i (q core): b = [ab slot 8 x 64 (alpha HC | beta HC bf16) | q rows 8 x 128] -> the Q panels
// and the gates of the k head's v heads h = kh + 2 hh: d = expm1(ssm_a softplus(alpha + dt)), beta = sigmoid(beta),
// times the mask, at the record's [hh][d 16 | beta 16] (fp32: the block's 8 tokens at 8 tb); two heads per call
// (lanes [head 2][the 8 tokens])
void {{ p }}_qblock(const bfloat16 *__restrict b, int32_t *__restrict ws, bfloat16 *__restrict out, int32_t kh,
                    int32_t i, int32_t tb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ p }}_block<true>(b + 512, ws, (uint8_t *)out + {{ 2 * PAN }} * tb);
    const float *__restrict gp = SW(float, {{ w.gp }});
    // (8 lanes, 32 B: a 64-B load at an odd block's 32-B offset would round the address down)
    const aie::vector<float, 8> m8 = aie::load_v<8>(SW(float, {{ w.mask }}) + 16 * i + 8 * tb);
    const aie::vector<float, 16> m = aie::concat(m8, m8);
    float *__restrict gt = (float *)((uint8_t *)out + {{ KP_BYTES }}) + 8 * tb;
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
            aie::add(ai.template cast_to<float>(), aie::concat(aie::broadcast<float, 8>(gp[{{ 2 * hp }}]), aie::broadcast<float, 8>(gp[{{ h1 }}]))));
        const aie::vector<float, 16> d = al::mulff<16>(
            {{ g }}_expm1(al::mulff<16>(sp, aie::concat(aie::broadcast<float, 8>(gp[{{ 16 + 2 * hp }}]), aie::broadcast<float, 8>(gp[{{ 16 + h1 }}])))), m);
        const aie::vector<float, 16> be = al::mulff<16>({{ g }}_sigmoid(bi.template cast_to<float>()), m);
        aie::store_v(gt + {{ 64 * hp }}, d.template extract<8>(0));
        aie::store_v(gt + {{ 64 * hp + 16 }}, be.template extract<8>(0));
{% if 2 * hp + 1 < VH %}
        aie::store_v(gt + {{ 64 * hp + 32 }}, d.template extract<8>(1));
        aie::store_v(gt + {{ 64 * hp + 48 }}, be.template extract<8>(1));
{% endif %}
    }
{% endfor %}
}

}  // extern "C"
""")


class GdnPrepOperator(ArtcOperator):
    """The q / k side of a GDN layer's chunk, once per k head (module docstring): per column and k head a k core and
    a q core (rows 2 + 2 kh, 3 + 2 kh), each per 8 tokens the conv of its 128 channels, SiLU, the l2 norm and the
    token block's MMUL panels; the q core also the gates (d = expm1(g), beta) of the k head's v heads times the token
    mask. The memtile splits each input object between the two and joins their parts of a chunk's record (``JOBJ``).
    Runtime sequence: G (gdn_in's output, ``layout``), CV (``CV_ROWS`` rows of G's layout: the conv state in rows 0 ..
    2), P (``scan_params``), M (fp32 per token: 1 valid, 0 padding; ``MASK_ELEMS`` bf16 read), PR (the records:
    [column][k head][chunk])."""

    STACK = 5120
    CV_ROWS = 12  # rows of CV read (the q part's 12 rows, the k part's 8: 3 used)
    MASK_ELEMS = B_ELEMS  # bf16 of the mask argument (an object; T fp32 used)
    pfx = kernel_prefix("gdp")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout):
        assert T % C == 0 and T // 8 <= 64 and 4 * T <= 2 * QB_ELEMS
        self.T, self.cfg, self.layout = T, cfg, layout
        self.sh = GD.GdnShape(cfg)
        self.hc = self.sh.hc
        self.vh = self.hc // 2
        assert {"qkv", "z", "ab"} <= set(layout.off) and self.hc % 2 == 0 and self.hc <= 6, \
            "two k heads of up to 3 v heads per column"
        super().__init__()

    def workspace(self) -> dict[str, int]:
        sizes = [("wc", 2 * 8 * 128), ("rows", 2 * 11 * 128), ("yc", 2 * 8 * 128), ("ys", 2 * 8 * 128),
                 ("xf", 4 * 8 * 128), ("gp", 4 * 32), ("mask", 4 * (self.T + 16))]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        helpers = JINJA_ENV.from_string(GD.HELPERS).render(p=GD.GDN, eps=repr(self.cfg.rms_norm_eps),
                                                           ln2=bf16_hilo(math.log(2)), D=D)
        return _PREP.render(p=self.pfx, g=GD.GDN, helpers=helpers, silu=act_lut.render_silu(), lut=act_lut.LUT,
                            w=Attr(self.workspace()), D=D, HC=self.hc, VH=self.vh, T=self.T, PAN=PAN,
                            KP_BYTES=KP_BYTES, eps=repr(self.cfg.rms_norm_eps), rs=bf16_hilo(1 / math.sqrt(D)))

    @property
    def jobj(self) -> int:
        """bf16 of a chunk's record."""
        return jobj_bytes(self.vh) // 2

    @property
    def n_pr(self) -> int:
        """bf16 of PR: per column and k head the chunk's records."""
        return COLS * 2 * self.T // C * self.jobj

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (params_bytes(self.hc) // 2,), torch.bfloat16),
                ArgSpec("in", (self.MASK_ELEMS,), torch.bfloat16),
                ArgSpec("out", (self.n_pr,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, lay = self.pfx, self.T, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        kq = KP_BYTES // 2
        b_ty, qb_ty, kb_ty = (np.ndarray[(n,), bf] for n in (B_ELEMS, QB_ELEMS, B_ELEMS - QB_ELEMS))
        j_ty, ko_ty, qo_ty = (np.ndarray[(n,), bf] for n in (self.jobj, kq, self.jobj - kq))
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        k_qinit = Kernel(f"{p}_qinit", "kernel.o", [qb_ty, ws_ty, np.int32])
        k_kinit = Kernel(f"{p}_kinit", "kernel.o", [kb_ty, ws_ty, np.int32])
        k_q = Kernel(f"{p}_qblock", "kernel.o", [qb_ty, ws_ty, qo_ty, np.int32, np.int32, np.int32])
        k_k = Kernel(f"{p}_kblock", "kernel.o", [kb_ty, ws_ty, ko_ty, np.int32, np.int32])
        i3 = [[ObjectFifo(b_ty, name=f"gp_i{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        i1 = [[i3[c][kh].cons().split([0, QB_ELEMS], tile=Tile(c, 1), depths=[2, 2], obj_types=[qb_ty, kb_ty],
                                      names=[f"gp_iq{c}_{kh}", f"gp_ik{c}_{kh}"]) for kh in range(2)]
              for c in range(COLS)]
        o3 = [[ObjectFifo(j_ty, name=f"gp_o{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        o1 = [[o3[c][kh].prod().join([0, kq], tile=Tile(c, 1), depths=[2, 2], obj_types=[ko_ty, qo_ty],
                                     names=[f"gp_ok{c}_{kh}", f"gp_oq{c}_{kh}"]) for kh in range(2)]
              for c in range(COLS)]
        n_ch = T // C

        def core_q(i_h, o_h, init, block, ws, kh):
            for _ in forever_():
                for k in range_(3):
                    b = i_h.acquire(1)
                    init(b, ws, k)
                    i_h.release(1)
                for i in range_(n_ch):
                    o = o_h.acquire(1)
                    for tb in range(2):
                        b = i_h.acquire(1)
                        block(b, ws, o, kh, i, tb)
                        i_h.release(1)
                    o_h.release(1)

        def core_k(i_h, o_h, init, block, ws):
            for _ in forever_():
                for k in range_(3):
                    b = i_h.acquire(1)
                    init(b, ws, k)
                    i_h.release(1)
                for i in range_(n_ch):
                    o = o_h.acquire(1)
                    for tb in range(2):
                        b = i_h.acquire(1)
                        block(b, ws, o, i, tb)
                        i_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for kh in range(2):
                workers.append(Worker(core_k, [i1[c][kh][1].cons(), o1[c][kh][0].prod(), k_kinit, k_k,
                                               Buffer(ws_ty, name=f"gp_wk{c}_{kh}")],
                                      tile=Tile(c, 2 + 2 * kh), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
                workers.append(Worker(core_q, [i1[c][kh][0].cons(), o1[c][kh][1].prod(), k_qinit, k_q,
                                               Buffer(ws_ty, name=f"gp_wq{c}_{kh}"), kh],
                                      tile=Tile(c, 3 + 2 * kh), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        N, pitch = lay.N, lay.pitch
        n_p, n_cv = params_bytes(self.hc) // 2, self.CV_ROWS * N
        n_pr, per = self.n_pr, n_ch * self.jobj  # (a column's k head's records)
        n_g = T // 8

        def seq(G, CV, P, M, PR, ih, oh):
            tg = TaskGroup()
            for c in range(COLS):
                qkv = c * pitch + lay.off["qkv"]
                for kh in range(2):
                    f = ih[c][kh]
                    f.fill(P, TensorAccessPattern((n_p,), (P_HEAD + (2 * c + kh) * PREP_P) // 2, [1, 1, 1, B_ELEMS],
                                                  [0, 0, 0, 1]), group=tg)
                    chained_fill(f, CV, [TensorAccessPattern((n_cv,), qkv + D * kh, [1, 1, 12, D], [0, 0, N, 1]),
                                         TensorAccessPattern((n_cv,), qkv + 2 * D + D * kh, [1, 1, 8, D], [0, 0, N, 1])],
                                 tg)
                    f.fill(M, TensorAccessPattern((self.MASK_ELEMS,), 0, [1, 1, 1, B_ELEMS], [0, 0, 0, 1]), group=tg)
                    chained_fill(f, G, [TensorAccessPattern((T * N,), c * pitch + lay.off["ab"], [n_g, 1, 8, 64],
                                                            [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + D * kh, [n_g, 1, 8, D], [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + 2 * D + D * kh, [n_g, 1, 8, D],
                                                            [8 * N, 0, N, 1])], tg)
                    oh[c][kh].drain(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [1, n_ch, self.jobj // 64, 64],
                                                            [0, self.jobj, 64, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf], np.ndarray[(n_p,), bf],
                           np.ndarray[(self.MASK_ELEMS,), bf], np.ndarray[(n_pr,), bf],
                           [[f.prod(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(i3)],
                           [[f.cons(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_SCAN = JINJA_ENV.from_string(r"""
// The scan core's workspace (offsets in bytes: GdnScanOperator.workspace())
#define SW(T, off) ((T *)((uint8_t *)ws + (off)))
#define TR(x) aie::op_transpose(x)

namespace {

{{ silu }}
// y[t][0 .. 32) = bf16(sum over the taps of w (hi + lo) x) of 8 tokens t: x rows t .. t + 3 of r (pitch 32: the
// window's 3 rows, then the tokens'), w [tap 4][hi, lo][32]
__attribute__((noinline)) void {{ p }}_conv8(const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                           bfloat16 *__restrict y)
{
    aie::vector<bfloat16, 32> wh[4], wl[4];
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 4; t++) {
        wh[t] = aie::load_v<32>(w + 64 * t);
        wl[t] = aie::load_v<32>(w + 64 * t + 32);
    }
    const bfloat16 *__restrict pr = r;
    bfloat16 *__restrict py = y;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 8; t++) {
        aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 4; k++) {
            const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + 32 * k);
            acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
        }
        aie::store_v(py, acc.template to_vector<bfloat16>());
        pr += 32;
        py += 32;
    }
}

// lanes [0, N / 2) = a, [N / 2, N) = b (two broadcasts and a select: a concat of half-width broadcasts pushes lane by
// lane)
template <typename T, int N>
[[clang::always_inline]] inline aie::vector<T, N> {{ p }}_bc2(T a, T b)
{
    constexpr unsigned long long hi = ((N == 64 ? 0ull : (1ull << N)) - 1) & ~((1ull << (N / 2)) - 1);
    return aie::select(aie::broadcast<T, N>(a), aie::broadcast<T, N>(b), aie::mask<N>::from_uint64(hi));
}

// lanes 8 r + c = v[r] (each of 8 values over a row of 8 lanes)
[[clang::always_inline]] inline aie::vector<float, 64> {{ p }}_rows8(aie::vector<float, 8> v)
{
    return aie::concat(aie::concat(aie::concat(aie::broadcast<float, 8>(v[0]), aie::broadcast<float, 8>(v[1])),
                                   aie::concat(aie::broadcast<float, 8>(v[2]), aie::broadcast<float, 8>(v[3]))),
                       aie::concat(aie::concat(aie::broadcast<float, 8>(v[4]), aie::broadcast<float, 8>(v[5])),
                                   aie::concat(aie::broadcast<float, 8>(v[6]), aie::broadcast<float, 8>(v[7]))));
}

// an 8 x 8 block of fp32 rows at stride n -> 64 lanes [r][c]
template <int N>
[[clang::always_inline]] inline aie::vector<float, 64> {{ p }}_blk(const float *__restrict p)
{
    return aie::concat(aie::concat(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + N)),
                                   aie::concat(aie::load_v<8>(p + 2 * N), aie::load_v<8>(p + 3 * N))),
                       aie::concat(aie::concat(aie::load_v<8>(p + 4 * N), aie::load_v<8>(p + 5 * N)),
                                   aie::concat(aie::load_v<8>(p + 6 * N), aie::load_v<8>(p + 7 * N))));
}

// x (64 fp32) -> panels hi, lo (bfp16 of the 8 x 8 block, 8 values per block along the lanes' fast axis) pushed at out
[[clang::always_inline]] inline void {{ p }}_push2(aie::accum<accfloat, 64> x, const al::BV &nI, uint8_t *out)
{
    al::BV h, l;
    al::split_hilo(x, nI, h, l);
    al::BOut<> po((bfp16ebs8 *)out);
    po.push(h);
    po.push(l);
}

// The chunk's Gram blocks of its K / Q panels (kp, qp: [ib 16][tb 2][hi, lo]) on the MMUL (three products each):
// g = K K^T 00, 10, 11, Q K^T 00, 10, 11 ([t][s] fp32), and the K^T panels kt ([ib][tb][hi, lo] of the [i 8][t 8]
// blocks: I (K hi + K lo)^T exactly, then split)
__attribute__((noinline)) void {{ p }}_gram(const uint8_t *__restrict kp, const uint8_t *__restrict qp,
                                          float *__restrict g, uint8_t *__restrict kt)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {
        al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>()), c(aie::zeros<accfloat, 64>());
        al::BIn<> ks((const bfp16ebs8 *)kp);
        for (int ib = 0; ib < 16; ib++) {
            const al::BV h0 = ks.pop(), l0 = ks.pop(), h1 = ks.pop(), l1 = ks.pop();
            a.mac(h0, TR(h0));
            b.mac(h1, TR(h0));
            c.mac(h1, TR(h1));
            a.mac(h0, TR(l0));
            b.mac(h1, TR(l0));
            c.mac(h1, TR(l1));
            a.mac(l0, TR(h0));
            b.mac(l1, TR(h0));
            c.mac(l1, TR(h1));
        }
        aie::store_v(g, a.to_accum().template to_vector<float>());
        aie::store_v(g + 64, b.to_accum().template to_vector<float>());
        aie::store_v(g + 128, c.to_accum().template to_vector<float>());
    }
    {
        al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>()), c(aie::zeros<accfloat, 64>());
        al::BIn<> ks((const bfp16ebs8 *)kp);
        al::BIn<> qs((const bfp16ebs8 *)qp);
        for (int ib = 0; ib < 16; ib++) {
            const al::BV h0 = ks.pop(), l0 = ks.pop(), h1 = ks.pop(), l1 = ks.pop();
            const al::BV q0 = qs.pop(), m0 = qs.pop(), q1 = qs.pop(), m1 = qs.pop();
            a.mac(q0, TR(h0));
            b.mac(q1, TR(h0));
            c.mac(q1, TR(h1));
            a.mac(q0, TR(l0));
            b.mac(q1, TR(l0));
            c.mac(q1, TR(l1));
            a.mac(m0, TR(h0));
            b.mac(m1, TR(h0));
            c.mac(m1, TR(h1));
        }
        aie::store_v(g + 192, a.to_accum().template to_vector<float>());
        aie::store_v(g + 256, b.to_accum().template to_vector<float>());
        aie::store_v(g + 320, c.to_accum().template to_vector<float>());
    }
    const al::BV I = al::eye(), nI = al::neg_eye();
    al::BIn<> ks((const bfp16ebs8 *)kp);
    uint8_t *__restrict po = kt;
    for (int ib = 0; ib < 16; ib++) {
        AIE_LOOP_UNROLL_FULL
        for (int tb = 0; tb < 2; tb++) {
            const al::BV h = ks.pop(), l = ks.pop();
            al::MM8 t(aie::zeros<accfloat, 64>());
            t.mac(I, TR(h));
            t.mac(I, TR(l));
            {{ p }}_push2(t.to_accum(), nI, po);
            po += {{ 2 * PAN }};
        }
    }
}

// The decays of the chunk for the v head (gt: d [16], beta [16] fp32). E[t][s] = prod_{u = s + 1 .. t} (1 + d_u) - 1
// by 8 steps on 32 lanes: block 0 (rows 0 .. 7) its columns 0 .. 7 at lanes 0 .. 7, the chunk's start (s = -1) at lane
// 8; block 1 (rows 8 .. 15) its columns 8 .. 15 at lanes 16 .. 23, its start (s = 7) at lane 24 (er: [row k][32]).
// Out: ed = E00, E10, E11 ([t][s] fp32); per token block tb fv [tb][beta | Es = Gamma - 1 | E15 = Gamma_C / Gamma - 1]
// (64 lanes: the 8 tokens' values over the j rows); dc = Gamma_C - 1 as bf16 hi [64], lo [64]
__attribute__((noinline)) void {{ p }}_decays(const float *__restrict gt, float *__restrict er, float *__restrict ed,
                                            float *__restrict fv, bfloat16 *__restrict dc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    // (d as bf16 hi, lo per token: en = e + d + d e as three products into e's accumulator)
    bfloat16 *__restrict dh = (bfloat16 *)(er + 256), *__restrict dl = dh + 16;
    {
        aie::vector<bfloat16, 16> h, l;
        al::split2<16>(aie::load_v<16>(gt), h, l);
        aie::store_v(dh, h);
        aie::store_v(dl, l);
    }
    aie::vector<float, 32> e = aie::zeros<float, 32>();
    const float *__restrict pd = gt;
    const bfloat16 *__restrict ph = dh;
    float *__restrict pe = er;
    unsigned m = 0x100u;  // (lanes s < t and the block's start)
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 8; k++) {
        const aie::vector<bfloat16, 32> bh = {{ p }}_bc2<bfloat16, 32>(ph[0], ph[8]);
        const aie::vector<bfloat16, 32> bl = {{ p }}_bc2<bfloat16, 32>(ph[16], ph[24]);
        aie::vector<bfloat16, 32> eh, el;
        al::split2<32>(e, eh, el);
        aie::accum<accfloat, 32> acc(aie::add(e, {{ p }}_bc2<float, 32>(pd[0], pd[8])));
        acc = aie::mac(aie::mac(aie::mac(acc, bh, eh), bh, el), bl, eh);
        e = aie::select(aie::zeros<float, 32>(), acc.template to_vector<float>(), aie::mask<32>::from_uint32(m | (m << 16)));
        aie::store_v(pe, e);
        pd++;
        ph++;
        pe += 32;
        m = (((m & 0xFFu) << 1) | 1u) | 0x100u;
    }
    // the start column: Es[t] (t < 8: lane 8 of row t; t >= 8: Es[7] (+) F[t], F = lane 24 of row t - 8)
    aie::vector<float, 16> es, f;
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 8; k++) {
        es[k] = er[32 * k + 8];
        f[k] = er[32 * k + 24];
    }
    const aie::vector<float, 8> a7 = aie::broadcast<float, 8>(es[7]);
    const aie::vector<float, 8> f8 = f.template extract<8>(0);
    es.insert(1, aie::add(aie::add(a7, f8), al::mulff<8>(a7, f8)));
    // E10[t][s] = E00[7][s] (+) F[t]; E15[s] = E[15][s]: row 7 of E10 (s < 8), row 7 of block 1 (s >= 8)
    {
        const aie::vector<float, 64> ra = aie::load_v<8>(er + 224).template grow_replicate<64>();
        const aie::vector<float, 64> rb = {{ p }}_rows8(f8);
        const aie::vector<float, 64> e10 = aie::add(aie::add(ra, rb), al::mulff<64>(ra, rb));
        aie::store_v(ed + 64, e10);
        aie::store_v(fv + 128, e10.template extract<8>(7).template grow_replicate<64>());
    }
    aie::store_v(ed, {{ p }}_blk<32>(er));
    aie::store_v(ed + 128, {{ p }}_blk<32>(er + 16));
    aie::store_v(fv + 320, aie::load_v<8>(er + 240).template grow_replicate<64>());
    const aie::vector<float, 16> be = aie::load_v<16>(gt + 16);
    AIE_LOOP_UNROLL_FULL
    for (int tb = 0; tb < 2; tb++) {
        aie::store_v(fv + 192 * tb, be.template extract<8>(tb).template grow_replicate<64>());
        aie::store_v(fv + 192 * tb + 64, es.template extract<8>(tb).template grow_replicate<64>());
    }
    aie::vector<bfloat16, 64> ch, cl;
    al::bf16_hilo<64>(es[15], ch, cl);
    aie::store_v(dc, ch);
    aie::store_v(dc + 64, cl);
}

// L = beta (1 + E) K K^T (s < t), M = (1 + E) Q K^T (s <= t) from the Gram blocks g ([KK 00 | 10 | 11 | QK 00 | 10 | 11])
// and ed ([E00 | E10 | E11]): L00, L11 as bf16 hi, lo (lb: [hi | lo][block 2][t][s]) and the panels (hi, lo) of -L10,
// M00, M10, M11 at pn + 2, 6, 8, 10 panels (rolled over the blocks: program memory)
__attribute__((noinline)) void {{ p }}_lm(const float *__restrict gt, const float *__restrict g,
                                        const float *__restrict ed, bfloat16 *__restrict lb, uint8_t *__restrict pn)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < 3; b++) {  // (block b: 00, 10, 11; its rows' token block b > 0)
        const aie::vector<float, 64> kk = aie::load_v<64>(g + 64 * b), q = aie::load_v<64>(g + 192 + 64 * b);
        const aie::vector<float, 64> e = aie::load_v<64>(ed + 64 * b);
        const aie::vector<float, 64> l = al::mulff<64>({{ p }}_rows8(aie::load_v<8>(gt + 16 + 8 * (b > 0))), aie::add(kk, al::mulff<64>(e, kk)));
        const bool diag = b != 1;
        const aie::mask<64> strict = aie::mask<64>::from_uint64(diag ? 0x{{ strict }}ull : ~0ull);
        const aie::mask<64> lower = aie::mask<64>::from_uint64(diag ? 0x{{ lower }}ull : ~0ull);
        if (diag) {
            aie::vector<bfloat16, 64> h, lo;
            al::split2<64>(aie::select(aie::zeros<float, 64>(), l, strict), h, lo);
            aie::store_v(lb + 32 * b, h);
            aie::store_v(lb + 128 + 32 * b, lo);
        } else
            {{ p }}_push2(aie::accum<accfloat, 64>(aie::sub(aie::zeros<float, 64>(), l)), nI, pn + {{ 2 * PAN }});
        {{ p }}_push2(aie::accum<accfloat, 64>(aie::select(aie::zeros<float, 64>(), aie::add(q, al::mulff<64>(e, q)), lower)), nI,
                      pn + {{ 6 * PAN }} + {{ 2 * PAN }} * b);
    }
}

// T00, T11 = (I + L00)^-1, (I + L11)^-1 by row substitution, r[t] = e_t - sum_{s < t} L[t][s] r[s] on 16 lanes [T00 row |
// T11 row]: per s three bf16 products into the row's accumulator (L from lb, the finished rows as bf16 hi, lo in rb:
// [hi | lo][t][16]; first the unit rows: the rolled loop over every s multiplies the rows s >= t, finite, by L[t][s] =
// 0); tr: the rows (fp32); as panels at pn, pn + 4 panels
__attribute__((noinline)) void {{ p }}_tinv(const bfloat16 *__restrict lb, float *__restrict tr, bfloat16 *__restrict rb,
                                          uint8_t *__restrict pn)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 8; t++) {
        aie::vector<float, 16> x = aie::zeros<float, 16>();
        x[t] = 1.0f;
        x[8 + t] = 1.0f;
        aie::store_v(tr + 16 * t, x);
        aie::store_v(rb + 16 * t, aie::accum<accfloat, 16>(x).template to_vector<bfloat16>());
        aie::store_v(rb + 128 + 16 * t, aie::zeros<bfloat16, 16>());
    }
    AIE_LOOP_NO_UNROLL
    for (int t = 1; t < 8; t++) {
        // (three chains, one per product: a chain of 24 dependent MACs would bind)
        aie::accum<accfloat, 16> a0(aie::load_v<16>(tr + 16 * t)), a1(aie::zeros<accfloat, 16>()), a2(aie::zeros<accfloat, 16>());
        const bfloat16 *__restrict ph = lb + 8 * t;
        const bfloat16 *__restrict qh = rb;
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < 8; s++) {
            const aie::vector<bfloat16, 16> ch = {{ p }}_bc2<bfloat16, 16>(ph[0], ph[64]);
            const aie::vector<bfloat16, 16> cl = {{ p }}_bc2<bfloat16, 16>(ph[128], ph[192]);
            const aie::vector<bfloat16, 16> xh = aie::load_v<16>(qh), xl = aie::load_v<16>(qh + 128);
            a0 = aie::msc(a0, ch, xh);
            a1 = aie::msc(a1, ch, xl);
            a2 = aie::msc(a2, cl, xh);
            ph++;
            qh += 16;
        }
        const aie::accum<accfloat, 16> acc = aie::add(aie::add(a0, a1), a2);
        const aie::vector<float, 16> x = acc.template to_vector<float>();
        const aie::vector<bfloat16, 16> h = acc.template to_vector<bfloat16>();
        aie::store_v(tr + 16 * t, x);
        aie::store_v(rb + 16 * t, h);
        aie::store_v(rb + 128 + 16 * t, aie::accum<accfloat, 16>(aie::sub(x, aie::accum<accfloat, 16>(h).template to_vector<float>())).template to_vector<bfloat16>());
    }
    const al::BV nI = al::neg_eye();
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < 2; b++) {
        const float *__restrict pr = tr + 8 * b;
        aie::vector<float, 64> v = {{ p }}_blk<16>(pr);
        {{ p }}_push2(aie::accum<accfloat, 64>(v), nI, pn + {{ 4 * PAN }} * b);
    }
}

// X^T, Y^T of the chunk on a j-block: S0's 16 [j 8][i 8] blocks (fp32, split hi + lo on the MMUL) against the K / Q
// panels: xy = [X^T tb 0 | X^T tb 1 | Y^T 0 | Y^T 1] ([j][t] fp32)
__attribute__((noinline)) void {{ p }}_xy(const float *__restrict s, const uint8_t *__restrict kp,
                                        const uint8_t *__restrict qp, float *__restrict xy)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::MM8 x0(aie::zeros<accfloat, 64>()), x1(aie::zeros<accfloat, 64>()), y0(aie::zeros<accfloat, 64>()),
        y1(aie::zeros<accfloat, 64>());
    al::BIn<> ks((const bfp16ebs8 *)kp);
    al::BIn<> qs((const bfp16ebs8 *)qp);
    const float *__restrict ps = s;
    AIE_PREPARE_FOR_POSTPIPELINING  // (II 17 with 4 spills; the pre-RA schedule: II 27)
    for (int ib = 0; ib < 16; ib++) {
        al::BV sh, sl;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(ps)), nI, sh, sl);
        ps += 64;
        const al::BV kh0 = ks.pop(), kl0 = ks.pop(), kh1 = ks.pop(), kl1 = ks.pop();
        const al::BV qh0 = qs.pop(), ql0 = qs.pop(), qh1 = qs.pop(), ql1 = qs.pop();
        x0.mac(sh, TR(kh0));
        x1.mac(sh, TR(kh1));
        y0.mac(sh, TR(qh0));
        y1.mac(sh, TR(qh1));
        x0.mac(sh, TR(kl0));
        x1.mac(sh, TR(kl1));
        y0.mac(sh, TR(ql0));
        y1.mac(sh, TR(ql1));
        x0.mac(sl, TR(kh0));
        x1.mac(sl, TR(kh1));
        y0.mac(sl, TR(qh0));
        y1.mac(sl, TR(qh1));
    }
    aie::store_v(xy, x0.to_accum().template to_vector<float>());
    aie::store_v(xy + 64, x1.to_accum().template to_vector<float>());
    aie::store_v(xy + 128, y0.to_accum().template to_vector<float>());
    aie::store_v(xy + 192, y1.to_accum().template to_vector<float>());
}

// S = S0 + dC S0 (bf16 hi / lo products: dc = dC as bf16 hi [64], lo [64]), in place (si, so: the same 64 blocks as
// distinct parameters, each read before it is written)
__attribute__((noinline)) void {{ p }}_decay(const float *__restrict si, float *__restrict so,
                                           const bfloat16 *__restrict dc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 64> ch = aie::load_v<64>(dc), cl = aie::load_v<64>(dc + 64);
    const float __aie_dm_resource_c *__restrict pi = (const float __aie_dm_resource_c *)si;
    float __aie_dm_resource_d *__restrict po = (float __aie_dm_resource_d *)so;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 64; i++) {
        const aie::vector<float, 64> v = aie::load_v<64>(pi);
        pi += 64;
        aie::vector<bfloat16, 64> vh, vl;
        al::split2<64>(v, vh, vl);
        aie::store_v(po, aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 64>(v), vh, ch), vh, cl), vl, ch).template to_vector<float>());
        po += 64;
    }
}

// C += A (hi + lo) B (hi + lo)^T: three products (a, b: two panels each, hi then lo)
[[clang::always_inline]] inline void {{ p }}_mm3(al::MM8 &c, const al::BV &ah, const al::BV &al_, const al::BV &bh,
                                               const al::BV &bl)
{
    c.mac(ah, TR(bh));
    c.mac(ah, TR(bl));
    c.mac(al_, TR(bh));
}

// A j-block of the chunk: R^T = beta (V^T - Gamma X^T); U^T(0) = R^T(0) T00^T; U^T(1) = (R^T(1) - U^T(0) L10^T)
// T11^T; O^T = Gamma Y^T + U^T M^T (bf16, transposed into the chunk's output rows o: the block's 8 columns); U'^T =
// (1 + E15) U^T as panels up ([tb][hi, lo]). xy (256 fp32), vt (128 bf16): the j-block's; pn: [T00 | -L10 | T11 |
// M00 | M10 | M11] (hi, lo each); sw: scratch (R^T(1), U^T(0), U^T(1) fp32, their panels: each value's live range
// through memory, a frame of 64-lane vectors would not fit the stack)
__attribute__((noinline)) void {{ p }}_solve(const float *__restrict xy, const bfloat16 *__restrict vt,
                                           const float *__restrict fv, const uint8_t *__restrict pn,
                                           bfloat16 *__restrict o, uint8_t *__restrict up, float *__restrict sw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    float *__restrict r1 = sw, *__restrict uf0 = sw + 64, *__restrict uf1 = sw + 128;
    uint8_t *__restrict rp = (uint8_t *)(sw + 192), *__restrict u0p = rp + {{ 2 * PAN }}, *__restrict u1p = rp + {{ 4 * PAN }};
    {
        {
            const aie::vector<float, 64> x = aie::load_v<64>(xy);
            const aie::vector<float, 64> v = aie::accum<accfloat, 64>(aie::load_v<64>(vt)).template to_vector<float>();
            {{ p }}_push2(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<64>(fv), aie::sub(aie::sub(v, x), al::mulff<64>(aie::load_v<64>(fv + 64), x)))), nI, rp);
        }
        {
            const aie::vector<float, 64> x = aie::load_v<64>(xy + 64);
            const aie::vector<float, 64> v = aie::accum<accfloat, 64>(aie::load_v<64>(vt + 64)).template to_vector<float>();
            aie::store_v(r1, al::mulff<64>(aie::load_v<64>(fv + 192), aie::sub(aie::sub(v, x), al::mulff<64>(aie::load_v<64>(fv + 256), x))));
        }
        {
            al::BIn<> ps((const bfp16ebs8 *)pn);
            al::BIn<> rs((const bfp16ebs8 *)rp);
            const al::BV rh = rs.pop(), rl = rs.pop(), th = ps.pop(), tl = ps.pop();
            al::MM8 u(aie::zeros<accfloat, 64>());
            {{ p }}_mm3(u, rh, rl, th, tl);
            aie::store_v(uf0, u.to_accum().template to_vector<float>());
            {{ p }}_push2(u.to_accum(), nI, u0p);
        }
        {
            al::BIn<> ps((const bfp16ebs8 *)(pn + {{ 2 * PAN }}));
            al::BIn<> us((const bfp16ebs8 *)u0p);
            const al::BV uh = us.pop(), ul = us.pop(), lh = ps.pop(), ll = ps.pop();
            al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(r1)));
            {{ p }}_mm3(c, uh, ul, lh, ll);
            {{ p }}_push2(c.to_accum(), nI, rp);
        }
        {
            al::BIn<> ps((const bfp16ebs8 *)(pn + {{ 4 * PAN }}));
            al::BIn<> rs((const bfp16ebs8 *)rp);
            const al::BV rh = rs.pop(), rl = rs.pop(), th = ps.pop(), tl = ps.pop();
            al::MM8 u(aie::zeros<accfloat, 64>());
            {{ p }}_mm3(u, rh, rl, th, tl);
            aie::store_v(uf1, u.to_accum().template to_vector<float>());
            {{ p }}_push2(u.to_accum(), nI, u1p);
        }
        {
            const aie::vector<float, 64> y = aie::load_v<64>(xy + 128);
            al::MM8 c(aie::accum<accfloat, 64>(aie::add(y, al::mulff<64>(aie::load_v<64>(fv + 64), y))));
            al::BIn<> ps((const bfp16ebs8 *)(pn + {{ 6 * PAN }}));
            al::BIn<> us((const bfp16ebs8 *)u0p);
            const al::BV uh = us.pop(), ul = us.pop(), mh = ps.pop(), ml = ps.pop();
            {{ p }}_mm3(c, uh, ul, mh, ml);
            const aie::vector<bfloat16, 64> a = aie::transpose(c.to_accum().template to_vector<bfloat16>(), 8, 8);
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < 8; t++)
                aie::store_v(o + 32 * t, a.template extract<8>(t));
        }
        {
            const aie::vector<float, 64> y = aie::load_v<64>(xy + 192);
            al::MM8 c(aie::accum<accfloat, 64>(aie::add(y, al::mulff<64>(aie::load_v<64>(fv + 256), y))));
            al::BIn<> ps((const bfp16ebs8 *)(pn + {{ 8 * PAN }}));
            al::BIn<> us((const bfp16ebs8 *)u0p);  // (U^T(0) then U^T(1): u0p, u1p adjacent)
            const al::BV ah = us.pop(), al_ = us.pop(), mh = ps.pop(), ml = ps.pop();
            {{ p }}_mm3(c, ah, al_, mh, ml);
            const al::BV bh = us.pop(), bl = us.pop(), nh = ps.pop(), nl = ps.pop();
            {{ p }}_mm3(c, bh, bl, nh, nl);
            const aie::vector<bfloat16, 64> a = aie::transpose(c.to_accum().template to_vector<bfloat16>(), 8, 8);
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < 8; t++)
                aie::store_v(o + 256 + 32 * t, a.template extract<8>(t));
        }
        {
            const aie::vector<float, 64> u = aie::load_v<64>(uf0);
            {{ p }}_push2(aie::accum<accfloat, 64>(aie::add(u, al::mulff<64>(aie::load_v<64>(fv + 128), u))), nI, up);
        }
        {
            const aie::vector<float, 64> u = aie::load_v<64>(uf1);
            {{ p }}_push2(aie::accum<accfloat, 64>(aie::add(u, al::mulff<64>(aie::load_v<64>(fv + 320), u))), nI, up + {{ 2 * PAN }});
        }
    }
}

// S += U'^T K^T (the chunk's two token blocks), in place (si / so: the same blocks as distinct parameters, each read
// before it is written); per j-block jb its 16 blocks against the K^T panels kt ([ib][tb][hi, lo]), up: [jb][tb][hi, lo]
__attribute__((noinline)) void {{ p }}_update(const float *__restrict si, float *__restrict so,
                                            const uint8_t *__restrict kt, const uint8_t *__restrict up)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float __aie_dm_resource_c *__restrict pi = (const float __aie_dm_resource_c *)si;
    float __aie_dm_resource_d *__restrict po = (float __aie_dm_resource_d *)so;
    AIE_LOOP_NO_UNROLL
    for (int jb = 0; jb < 4; jb++) {
        // (the U' panels popped inside a loop: popped in the function's first block and held across the inner loop,
        // block vectors exhaust llvm-aie's register allocator)
        al::BIn<> us((const bfp16ebs8 *)(up + {{ 4 * PAN }} * jb));
        const al::BV u0h = us.pop(), u0l = us.pop(), u1h = us.pop(), u1l = us.pop();
        al::BIn<> ks((const bfp16ebs8 *)kt);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int ib = 0; ib < 16; ib++) {
            al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(pi)));
            pi += 64;
            const al::BV k0h = ks.pop(), k0l = ks.pop(), k1h = ks.pop(), k1l = ks.pop();
            {{ p }}_mm3(c, u0h, u0l, k0h, k0l);
            {{ p }}_mm3(c, u1h, u1l, k1h, k1l);
            aie::store_v(po, c.to_accum().template to_vector<float>());
            po += 64;
        }
    }
}

// a token block's v quarter: the window rows (3 before, then the block's raw rows, pitch 32), conv, SiLU -> ys
// [8][32]; the window shifted
__attribute__((noinline)) void {{ p }}_vblock(const bfloat16 *__restrict v, int32_t *__restrict ws,
                                            bfloat16 *__restrict ys)
{
    bfloat16 *__restrict r = SW(bfloat16, {{ w.rows }});
    al::copy<256, 32>(r + 96, v);
    bfloat16 *__restrict yc = SW(bfloat16, {{ w.yc }});
    {{ p }}_conv8(r, SW(bfloat16, {{ w.wc }}), yc);
    {{ lut }}_silu_lut<256>(yc, ys);
    al::copy<96, 32>(r, r + 256);
}

}  // namespace

extern "C" {

// A round's first split parts: k = 0: the v quarter's conv weights; k = 1: its conv state rows ([16][32]: 3 used)
// (minsize: the round's entries are cold)
__attribute__((minsize)) void {{ p }}_vinit(const bfloat16 *__restrict s, int32_t *__restrict ws, int32_t k)
{
    if (k == 0)
        al::copy<256, 32>(SW(bfloat16, {{ w.wc }}), s);
    else
        al::copy<96, 32>(SW(bfloat16, {{ w.rows }}), s);
}

// Part k of the state quarter in or out
__attribute__((minsize)) void {{ p }}_sin(const bfloat16 *__restrict s, float *__restrict S, int32_t k)
{
    al::copy<{{ PART // 2 }}, 32>(S + {{ PART // 2 }} * k, (const float *)s);
}

__attribute__((minsize)) void {{ p }}_sout(const float *__restrict S, bfloat16 *__restrict o, int32_t k)
{
    al::copy<{{ PART // 2 }}, 32>((float *)o, S + {{ PART // 2 }} * k);
}

// A chunk of 16 tokens of the round's v head h (index h / 2 among its k head's): b = the chunk's record (GdnPrep), v = the v quarter's
// raw rows [16][32], S = the state quarter ([jb 4][ib 16][j 8][i 8] fp32, in place), o = the chunk's outputs [16][32]
void {{ p }}_chunk(const bfloat16 *__restrict b, const bfloat16 *__restrict v, float *__restrict S,
                   int32_t *__restrict ws, bfloat16 *__restrict o, int32_t h)
{
{% if "ev_chunk" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const uint8_t *__restrict rec = (const uint8_t *)b;
    bfloat16 *__restrict ys = SW(bfloat16, {{ w.ys }});
    {{ p }}_vblock(v, ws, ys);
    {{ p }}_vblock(v + 256, ws, ys + 256);
    // V^T blocks [jb][tb][j 8][t 8]
    bfloat16 *__restrict vt = SW(bfloat16, {{ w.vt }});
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < 8; b++) {
        const bfloat16 *__restrict py = ys + 256 * (b & 1) + 8 * (b >> 1);
        const aie::vector<bfloat16, 64> x = aie::concat(
            aie::concat(aie::concat(aie::load_v<8>(py), aie::load_v<8>(py + 32)), aie::concat(aie::load_v<8>(py + 64), aie::load_v<8>(py + 96))),
            aie::concat(aie::concat(aie::load_v<8>(py + 128), aie::load_v<8>(py + 160)), aie::concat(aie::load_v<8>(py + 192), aie::load_v<8>(py + 224))));
        aie::store_v(vt + 64 * b, aie::transpose(x, 8, 8));
    }
{% if "chunk" not in ablate %}
    {{ p }}_gram(rec, rec + {{ KP_BYTES }}, SW(float, {{ w.g }}), SW(uint8_t, {{ w.kt }}));
    const float *__restrict gt = (const float *)(rec + {{ 2 * KP_BYTES }}) + 32 * (h >> 1);
    {{ p }}_decays(gt, SW(float, {{ w.er }}), SW(float, {{ w.ed }}), SW(float, {{ w.fv }}), SW(bfloat16, {{ w.dc }}));
    {{ p }}_lm(gt, SW(float, {{ w.g }}), SW(float, {{ w.ed }}), SW(bfloat16, {{ w.lt }}), SW(uint8_t, {{ w.pn }}));
    {{ p }}_tinv(SW(bfloat16, {{ w.lt }}), SW(float, {{ w.tr }}), SW(bfloat16, {{ w.rb }}), SW(uint8_t, {{ w.pn }}));
    AIE_LOOP_NO_UNROLL
    for (int jb = 0; jb < 4; jb++) {  // (S0 read by every j-block before the decay changes it)
        {{ p }}_xy(S + 1024 * jb, rec, rec + {{ KP_BYTES }}, SW(float, {{ w.xy }}));
        {{ p }}_solve(SW(float, {{ w.xy }}), vt + 128 * jb, SW(float, {{ w.fv }}), SW(uint8_t, {{ w.pn }}), o + 8 * jb,
                      SW(uint8_t, {{ w.up }}) + {{ 4 * PAN }} * jb, SW(float, {{ w.sw }}));
    }
    {{ p }}_decay(S, S, SW(bfloat16, {{ w.dc }}));
    {{ p }}_update(S, S, SW(uint8_t, {{ w.kt }}), SW(uint8_t, {{ w.up }}));
{% endif %}
{% if "ev_chunk" in ablate %}
    event1();
{% endif %}
}

}  // extern "C"
""")


class GdnScanOperator(ArtcOperator):
    """The delta rule of a GDN layer over a chunk of T tokens (module docstring), chunkwise on GdnPrep's records: the
    whole prompt chunk in one call (the states read and written once per chunk), its records those of T / ``sub``
    GdnPrep calls of ``sub`` tokens. Runtime sequence: PR (GdnPrep's records, [prep call][column][k head][chunk]), G
    (gdn_in's output, ``layout``: the v rows), CV (``CV_ROWS`` rows of G's layout: the conv state in rows 0 .. 2), P
    (``scan_params``: the v conv weights), GSTATE (the layer's state, per column its panels: read and written), O (T x
    hv x 128 bf16: o of every v head)."""

    STACK = 3072
    CV_ROWS = 16  # rows of CV read (the v quarter's conv state: [16][32])
    RUN = 64 * C  # tokens per BD of the v rows and o (a BD iterates at most 64 times)
    pfx = kernel_prefix("gdsc")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, sub: int = 0, ablate: tuple[str, ...] = ()):
        """``ablate`` (timing experiments, wrong results): "chunk" the chunk's math compiled out (the v conv and SiLU
        kept); "ev_chunk": an event window around a chunk call (results unchanged)."""
        self.sub = sub or T
        assert T % self.sub == 0 and self.sub % C == 0 and T // self.sub <= 64
        assert T <= 2 * self.RUN, "the v rows and o in at most two runs"
        self.T, self.cfg, self.layout = T, cfg, layout
        self.ablate = tuple(ablate)
        self.sh = GD.GdnShape(cfg)
        self.hc, self.ch, self.hv = self.sh.hc, self.sh.ch, cfg.ssm_v_heads
        self.vh = self.hc // 2
        assert {"qkv", "z", "ab"} <= set(layout.off) and self.hc % 2 == 0 and self.hc <= 6
        super().__init__()

    def workspace(self) -> dict[str, int]:
        """(the gates' scratch g .. tr is dead once the T panels are written: xy and sw share its bytes)"""
        common = [("wc", 2 * 4 * 2 * 32), ("rows", 2 * 11 * 32), ("yc", 2 * 8 * 32), ("ys", 2 * 16 * 32),
                  ("vt", 2 * 4 * 2 * 64), ("kt", KP_BYTES), ("fv", 4 * 2 * 3 * 64), ("pn", 12 * PAN), ("dc", 2 * 2 * 64),
                  ("up", 4 * 4 * PAN)]
        gates = [("g", 4 * 6 * 64), ("er", 4 * 8 * 32 + 64), ("ed", 4 * 3 * 64), ("lt", 2 * 4 * 64), ("tr", 4 * 8 * 16),
                 ("rb", 2 * 2 * 8 * 16)]
        jblock = [("xy", 4 * 4 * 64), ("sw", 4 * 192 + 6 * PAN)]
        w = regions(common)
        a, b = regions(gates), regions(jblock)
        out = {k: v for k, v in w.items() if k != "size"}
        out |= {k: w["size"] + v for k, v in a.items() if k != "size"}
        out |= {k: w["size"] + v for k, v in b.items() if k != "size"}
        out["size"] = w["size"] + max(a["size"], b["size"])
        return out

    def get_kernel_src(self) -> str:
        strict = sum(1 << (8 * t + s) for t in range(8) for s in range(t))
        lower = sum(1 << (8 * t + s) for t in range(8) for s in range(t + 1))
        return _SCAN.render(p=self.pfx, silu=act_lut.render_silu(), lut=act_lut.LUT, w=Attr(self.workspace()),
                            PART=PART, PAN=PAN, KP_BYTES=KP_BYTES, strict=f"{strict:016X}", lower=f"{lower:016X}",
                            ablate=self.ablate)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * 2 * self.T // C * jobj_bytes(self.vh) // 2,), torch.bfloat16),
                ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (params_bytes(self.hc) // 2,), torch.bfloat16),
                ArgSpec("in", (COLS * self.sh.n_panels * PANEL // 2,), torch.bfloat16),
                ArgSpec("out", (self.T * self.hv * D,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay = self.pfx, self.T, self.hc, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        jobj = jobj_bytes(self.vh) // 2
        b_ty = np.ndarray[(jobj,), bf]
        part_ty = np.ndarray[(PART,), bf]
        quad_ty = np.ndarray[(4 * PART,), bf]
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        S_ty = np.ndarray[(4 * PANEL // 4,), np.dtype[np.float32]]
        k_vinit = Kernel(f"{p}_vinit", "kernel.o", [part_ty, ws_ty, np.int32])
        k_sin = Kernel(f"{p}_sin", "kernel.o", [part_ty, S_ty, np.int32])
        k_sout = Kernel(f"{p}_sout", "kernel.o", [S_ty, part_ty, np.int32])
        k_chunk = Kernel(f"{p}_chunk", "kernel.o", [b_ty, part_ty, S_ty, ws_ty, part_ty, np.int32])

        b3 = [ObjectFifo(b_ty, name=f"gs_b{c}", depth=2) for c in range(COLS)]
        b1 = [b3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"gs_bm{c}") for c in range(COLS)]
        v3 = [ObjectFifo(quad_ty, name=f"gs_v{c}", depth=2) for c in range(COLS)]
        v1 = [v3[c].cons().split([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[part_ty] * 4,
                                 names=[f"gs_vp{c}_{r}" for r in range(4)]) for c in range(COLS)]
        o3 = [ObjectFifo(quad_ty, name=f"gs_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[part_ty] * 4, names=[f"gs_op{c}_{r}" for r in range(4)]) for c in range(COLS)]
        n_ch = T // C

        def core(b_f, v_f, o_f, vinit, sin, sout, chunk, ws, S):
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
                    for _ in range_(n_ch):
                        b = b_f.acquire(1)
                        v = v_f.acquire(1)
                        o = o_f.acquire(1)
                        chunk(b, v, S, ws, o, h)
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
                                             k_chunk, ws, S],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))

        N, pitch = lay.N, lay.pitch
        n_p = params_bytes(hc) // 2
        n_cv = self.CV_ROWS * N
        n_st = COLS * self.sh.n_panels * PANEL // 2
        n_o = T * self.hv * D
        n_sub, sub_ch = T // self.sub, self.sub // C  # (GdnPrep calls, chunks per call)
        n_pr, per, pr_sub = COLS * 2 * n_ch * jobj, sub_ch * jobj, COLS * 2 * sub_ch * jobj
        runs = [(t0, min(self.RUN, T - t0)) for t0 in range(0, T, self.RUN)]

        def seq(PR, G, CV, P, GS, O, bh, vh, oh):
            for h in range(hc):  # (a round on every column at once)
                kh = h & 1
                tg = TaskGroup()
                for c in range(COLS):
                    qkv = c * pitch + lay.off["qkv"]  # (the column's conv channels: rows of CV and G alike)
                    vp = (P_HEAD + COLS * 2 * PREP_P + (c * hc + h) * V_P) // 2
                    st = (c * self.sh.n_panels + 16 * h) * PANEL // 2  # the head's state (4 quarters of 16 KB)
                    bh[c].fill(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [n_sub, sub_ch, jobj // 64, 64],
                                                       [pr_sub, jobj, 64, 1]), group=tg)
                    chained_fill(vh[c], P, [TensorAccessPattern((n_p,), vp, [1, 1, 4, PART], [0, 0, PART, 1]),
                                            ChainPart(CV, TensorAccessPattern((n_cv,), qkv + 4 * D + D * h, [1, 4, 16, 32],
                                                                              [0, 32, N, 1]))], tg)
                    vh[c].fill(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART], [PART, 4 * PANEL // 2, 0, 1]),
                               group=tg)
                    j = gdn_v_heads(self.cfg, c)[h]
                    for t0, n in runs:
                        vh[c].fill(G, TensorAccessPattern((T * N,), t0 * N + qkv + 4 * D + D * h, [n // C, 4, 16, 32],
                                                          [16 * N, 32, N, 1]), group=tg)
                    for t0, n in runs:
                        oh[c].drain(O, TensorAccessPattern((n_o,), t0 * self.hv * D + D * j, [n // C, 4, 16, 32],
                                                           [16 * self.hv * D, 32, self.hv * D, 1]), group=tg)
                    oh[c].drain(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART], [PART, 4 * PANEL // 2, 0, 1]),
                                group=tg, wait=True)
                tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_pr,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf],
                           np.ndarray[(n_p,), bf], np.ndarray[(n_st,), bf], np.ndarray[(n_o,), bf],
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

// 8 tokens of a pair of v heads: u = [o | z] ([8][2 x 128] each) -> rms(o) ssm_norm silu(z) as 32 bfp16 panels (8
// tokens x 8 dims) (4565 cycles: the rows' 1 / rms first, then one pipelined loop, II 14 per 32; per-row loops,
// unpipelined: 7713)
void {{ p }}_pair(const bfloat16 *__restrict u, float *__restrict ws, bfloat16 *__restrict out)
{
    const bfloat16 *__restrict o = u;
    const bfloat16 *__restrict z = u + {{ OBJ }};
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
    """The gated output of a GDN layer's chunk and its new conv state (module docstring). Per column 4 cores: the units
    (8 tokens x a pair of the column's v heads, pair-major) come as [o | z] objects, four per memtile object split
    between the cores, their panels joined back; the conv state goes from GC through the memtile into GHDR (no core).
    Runtime sequence: O (the scan's), G (z), GC (``gc_rows`` rows of G's layout: the prompt chunk's; the conv state rows
    from the scratchpad parameter ``gdn_conv``: ``conv_offset``), P (ssm_norm first), GHDR (the conv states written), A
    (the ssm_out GEMM's panels, T x hv 128)."""

    STACK = 4096
    OBJ = 2048  # bf16 of an o (or z) unit: 8 tokens x a pair of heads
    pfx = kernel_prefix("gdo")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, hslot: int, cs_off: int, gc_rows: int = 0):
        assert T % 32 == 0 and T // 8 <= 64
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
                ArgSpec("in", (params_bytes(self.hc) // 2,), torch.bfloat16),
                ArgSpec("out", (self.ghdr_bytes // 2,), torch.bfloat16),
                ArgSpec("out", (T * hv * D * 9 // 16,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay, hv, ch = self.pfx, self.T, self.hc, self.layout, self.hv, self.ch
        bf = np.dtype[bfloat16]
        U = 2 * self.OBJ  # (a unit: [o | z])
        obj_ty, u_ty, uq_ty = (np.ndarray[(n,), bf] for n in (self.OBJ, U, 4 * U))
        pan_ty, panq_ty = np.ndarray[(32 * 36,), bf], np.ndarray[(4 * 32 * 36,), bf]  # 32 panels of 72 B
        ws_ty = np.ndarray[(D + 8 * 256 + 4 * 256 + 512 + D // 32 * 32,), np.dtype[np.float32]]  # norm | y | silu(z) (bf16) | 1 / rms | norm hi / lo
        k_norm = Kernel(f"{p}_norm", "kernel.o", [u_ty, ws_ty])
        k_pair = Kernel(f"{p}_pair", "kernel.o", [u_ty, ws_ty, pan_ty])
        conv = ScratchpadParameter(CONV_OFF, np.int32)
        a3 = [ObjectFifo(uq_ty, name=f"go_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([U * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[u_ty] * 4,
                                 names=[f"go_ap{c}_{r}" for r in range(4)]) for c in range(COLS)]
        c3 = [ObjectFifo(panq_ty, name=f"go_c{c}", depth=2) for c in range(COLS)]
        c1 = [c3[c].prod().join([32 * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[pan_ty] * 4, names=[f"go_cp{c}_{r}" for r in range(4)]) for c in range(COLS)]
        d3 = [ObjectFifo(obj_ty, name=f"go_d{c}", depth=2) for c in range(COLS)]  # (the conv state: DDR -> memtile -> DDR)
        d1 = [d3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"go_dm{c}") for c in range(COLS)]
        n_g, npair = T // 8, hc // 2
        per_core = n_g * npair // 4

        def core(a_h, c_h, norm, pair, ws):
            for _ in forever_():
                a = a_h.acquire(1)
                norm(a, ws)
                a_h.release(1)
                for _ in range_(per_core):
                    a = a_h.acquire(1)
                    c = c_h.acquire(1)
                    pair(a, ws, c)
                    c_h.release(1)
                    a_h.release(1)

        workers = [Worker(core, [a1[c][r].cons(), c1[c][r].prod(), k_norm, k_pair, Buffer(ws_ty, name=f"go_ws{c}_{r}")],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        N, n_o, n_p = lay.N, T * hv * D, params_bytes(hc) // 2
        n_hdr, n_a = self.ghdr_bytes // 2, T * hv * D * 9 // 16
        kb_row = hv * D // 8 * 36  # (bf16 of a token group's panels)

        n_gc = self.gc_rows * N

        def seq(O, G, GC, P, GH, A, ah, ch_, dh, d1h):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                # (the norm to each core: P's first unit read four times)
                ah[c].fill(P, TensorAccessPattern((n_p,), 0, [1, 4, U // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                z0 = c * lay.pitch + lay.off["z"]
                for pp in range(npair):  # (the pair's units [o | z] of every token group)
                    chained_fill(ah[c], O, [TensorAccessPattern((n_o,), 2 * c * D + 16 * D * pp, [n_g, 1, 8, 2 * D],
                                                                [8 * hv * D, 0, hv * D, 1]),
                                            ChainPart(G, TensorAccessPattern((T * N,), z0 + 2 * D * pp, [n_g, 1, 8, 2 * D],
                                                                             [8 * N, 0, N, 1]))], tg)
                qkv = c * lay.pitch + lay.off["qkv"]
                dh[c].fill(GC, TensorAccessPattern((n_gc,), qkv, [1, 1, 3, ch], [0, 0, N, 1]), offset_parameter=conv,
                           group=tg)
                dh[c].fill(GC, TensorAccessPattern((n_gc,), 0, [1, 1, 1, 2 * self.OBJ - 3 * ch], [0, 0, 0, 1]),
                           group=tg)
                # (the conv state and the objects' padding after it: the bytes the decode's conv-state drain writes)
                cs = (c * self.hslot + self.cs_off) // 2
                d1h[c].drain(GH, TensorAccessPattern((n_hdr,), cs, [1, 1, 1, 2 * self.OBJ], [0, 0, 0, 1]), group=tg)
                ch_[c].drain(A, TensorAccessPattern((n_a,), 2 * c * 16 * 36, [1, npair, n_g, 32 * 36],
                                                    [0, 16 * 16 * 36, kb_row, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_o,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_gc,), bf],
                           np.ndarray[(n_p,), bf], np.ndarray[(n_hdr,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(c3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(d3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(d1)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
