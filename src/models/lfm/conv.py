"""LFM3's short convolution in the decode overlay: a job of the attention cores (``decode.attn.Attn(conv=)``: the
operator core (c, 5) of every column runs a conv layer's step between two attention layers; its D stream is the
column's only output besides the main core's C, so the conv cannot have a core of its own).

Per conv layer and column c (its 256 channels 256 c ..: ``image.CH``), a step of S <= 4 slots:

- the "in" phase drains its rows (C objects of 64 values per slot: object j of the slots, then j + 1; j 0 .. 3 B, 4 .. 7 C,
  8 .. 11 x) into the column's conv region, [j][slot 4][64] bf16 (``rows_drains``), then a sync;
- the conv region is ``OBJS`` X objects (each ``XB`` bytes: metadata, then ``XPAY`` of payload): the first one's metadata
  ``F_CONV`` and ``M_SKIP`` (the others skip it and the rest), payloads: taps 0 .. 3, [taps 4 .. 6 | history rows 8, 9],
  history rows 0 .. 7, [rows j 0 .. 7], [rows j 8 .. 11 | unused]; a tap is [hi 256 | lo 256] bf16 (the fp32 weight
  as bf16 hi + lo: ``tap_objects``), a history row 256 bf16;
- the core takes the step's S and the conv shift (``M_CSHIFT``: the previous step's accepted tokens) from token object 0
  (``decode.attn``'s F_TOK, words ``M_TOK_S`` / ``M_CSHIFT``), builds the window [history rows cshift .. cshift + 5 | the
  step's S rows of bx = bf16(B x)], y_s = bf16(C_s sum_k (hi_k + lo_k) window[s + k]) (the sum in fp32, k in order, hi
  then lo; times C as bf16 hi + lo of the sum), and emits on D: y as out's packed input ([k block 32][slot SP][8] bf16,
  ``decode.design.packed_slots``), then the new history [window rows 0 .. 5 + S] (``hist_drains``).

The history holds the previous step's 6 base rows and its S rows: the next step's base starts at its accepted count
(lazy commit, as qwen35's GDN conv), so rejected drafts cost nothing and the host patches no addresses per step. A
prefill leaves rows 0 .. 5 = its last 6 rows of bx and the next step's shift 0.
"""

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.design import META, XB, XPAY, packed_slots
from artlite.operators.base import JINJA_ENV

F_CONV = 1 << 8  # M_FLAGS of a conv region's first object (the other cores skip it and M_SKIP objects after it)
M_TOK_S, M_CSHIFT = 45, 46  # token object 0's: the step's slots, the conv shift
CH = 256  # channels per column
WIDTH = 7  # taps (the history: WIDTH - 1 base rows and up to SMAX step rows)
SMAX = 4
HIST = WIDTH - 1 + SMAX  # history rows in the region
ROW = 2 * CH  # bytes of a history row
OBJS = 5  # X objects of a conv region
TAP = 2 * ROW  # bytes of a tap: [hi 256 | lo 256] bf16
RJ = 12  # rows objects (C objects) per slot: B, C, x of 256 channels in 64s
H8 = XPAY + 3 * TAP  # (in the region's payloads, object i's at XPAY i) history rows 8, 9: after taps 4 .. 6
D_BYTES = 128  # a D object (decode.attn's d_len at head_dim 128)
assert 4 * TAP == XPAY and H8 + 2 * ROW == 2 * XPAY and 8 * ROW == XPAY and 8 * SMAX * 128 == XPAY

# state words of the operator core (after decode.attn's A_WORDS)
C_NCV, C_CX, C_CJ, C_S, C_SH = range(5)
C_WORDS = 8


def tap_objects(w: np.ndarray) -> np.ndarray:
    """A layer's conv weights (hidden, WIDTH) fp32 -> per column the region's tap bytes (COLS, 7 TAP): per tap [hi | lo]
    of the column's channels (bf16 hi = bf16(w), lo = bf16(w - hi))."""
    cols = w.shape[0] // CH
    t = np.ascontiguousarray(w.reshape(cols, CH, WIDTH).transpose(0, 2, 1), np.float32)  # (col, tap, ch)
    hi = t.astype(bfloat16)
    lo = (t - hi.astype(np.float32)).astype(bfloat16)
    return np.stack([hi, lo], 2).view(np.uint8).reshape(cols, -1)


def region_bytes() -> int:
    return OBJS * XB


def payload(i: int) -> int:
    """Byte offset of object i's payload in a conv region."""
    return i * XB + 4 * META


def rows_drains(S: int) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """The in phase's C drains into a conv region (offset, bytes, dims): rows j 0 .. 7 into object 3's payload, j 8 ..
    11 into object 4's, each C object (j, slot s) at [j][slot 4][64]."""
    dims = (32, S, 32, 128)
    return [(payload(3), 8 * S * 128, dims), (payload(4), 4 * S * 128, dims)]


def hist_drains(S: int) -> list[tuple[int, int]]:
    """The new history's D drains into a conv region (offset, bytes): rows 0 .. 7 into object 2's payload, the rest
    (rows 8, 9) after the taps in object 1's."""
    n = WIDTH - 1 + S
    out = [(payload(2), min(n, 8) * ROW)]
    if n > 8:
        out.append((payload(1) + 3 * TAP, (n - 8) * ROW))
    return out


def y_bytes(S: int) -> int:
    """Bytes of the column's y in out's packed input."""
    return CH * packed_slots(S) * 2


_SRC = JINJA_ENV.from_string(r"""
// LFM3's conv step on the operator core (models.lfm.conv): workspace regions from {{ base }} on
#define {{ p }}_CW(T, off) ((T *)((uint8_t *)ws + {{ base }} + (off)))
#define {{ p }}_CSW ((int32_t *)((uint8_t *)ws + {{ st }}))
#define {{ p }}_CST(i) {{ p }}_CSW[{{ A_WORDS }} + (i)]

namespace {

__attribute__((noinline, minsize)) void {{ p }}_ccopy(uint8_t *__restrict dst, const uint8_t *__restrict src, int n)
{
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i += 64)
        aie::store_v((bfloat16 *)(dst + i), aie::load_v<32>((const bfloat16 *)(src + i)));
}

// The step: the window [base 6 | bx of S rows] in place of the history, y into its packed rows
__attribute__((noinline, minsize)) void {{ p }}_cstep(int32_t *__restrict ws, const bfloat16 *__restrict xr)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = {{ p }}_CST({{ C_S }}), sh = {{ p }}_CST({{ C_SH }}), sp = S + (S == 3);
    bfloat16 *__restrict h = {{ p }}_CW(bfloat16, {{ c.hist }});
    const bfloat16 *__restrict bc = {{ p }}_CW(bfloat16, {{ c.bc }});
    if (sh)
        {{ p }}_ccopy((uint8_t *)h, (const uint8_t *)(h + {{ CH }} * sh), {{ (WIDTH - 1) * ROW }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ CH // 64 }}; v++) {
            const aie::vector<bfloat16, 64> b = aie::load_v<64>(bc + 64 * (4 * v + s));
            const aie::vector<bfloat16, 64> x = aie::load_v<64>(xr + 64 * (4 * v + s));
            aie::store_v(h + {{ CH }} * ({{ WIDTH - 1 }} + s) + 64 * v, aie::mul(b, x).template to_vector<bfloat16>());
        }
    }
    const bfloat16 *__restrict w = {{ p }}_CW(bfloat16, {{ c.taps }});
    bfloat16 *__restrict y = {{ p }}_CW(bfloat16, {{ c.y }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ CH // 64 }}; v++) {
            aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
            AIE_LOOP_NO_UNROLL
            for (int k = 0; k < {{ WIDTH }}; k++) {
                const aie::vector<bfloat16, 64> r = aie::load_v<64>(h + {{ CH }} * (s + k) + 64 * v);
                acc = aie::mac(acc, aie::load_v<64>(w + {{ 2 * CH }} * k + 64 * v), r);
                acc = aie::mac(acc, aie::load_v<64>(w + {{ 2 * CH }} * k + {{ CH }} + 64 * v), r);
            }
            const aie::vector<bfloat16, 64> ah = acc.template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> al = aie::accum<accfloat, 64>(
                aie::sub(acc.template to_vector<float>(), aie::accum<accfloat, 64>(ah).template to_vector<float>())).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> c = aie::load_v<64>(bc + 64 * (4 * ({{ CH // 64 }} + v) + s));
            const aie::vector<bfloat16, 64> o = aie::mac(aie::mul(ah, c), al, c).template to_vector<bfloat16>();
            AIE_LOOP_UNROLL_FULL
            for (int i = 0; i < 8; i++)  // (k blocks 8 v + i: [k block][slot sp][8]; unrolled: a runtime lane index is a jump table)
                aie::store_v(y + 8 * (sp * (8 * v + i) + s), o.template extract<8>(i));
        }
    }
    {{ p }}_CSW[{{ A_NE }}] = {{ CH * 2 // D_BYTES }} * sp + ({{ WIDTH - 1 }} + S) * {{ ROW // D_BYTES }};
    {{ p }}_CSW[{{ A_EIDX }}] = 0;
}

}  // namespace

extern "C" {

// A conv region's first object (F_CONV): taps 0 .. 3; the rest of the region follows
__attribute__((minsize)) void {{ p }}_cbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_CST({{ C_NCV }}) = 1;
    {{ p }}_CST({{ C_CX }}) = {{ OBJS - 1 }};
    {{ p }}_CST({{ C_CJ }}) = 0;
    {{ p }}_ccopy({{ p }}_CW(uint8_t, {{ c.taps }}), (const uint8_t *)(x + {{ 2 * META }}), {{ XPAY }});
}

// The region's next object: taps 4 .. 6 and history rows 8, 9; history rows 0 .. 7; rows j 0 .. 7 (B, C); rows j
// 8 .. 11 (x): the step
__attribute__((minsize)) void {{ p }}_cobj{{ X }}(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int j = {{ p }}_CST({{ C_CJ }});
    {{ p }}_CST({{ C_CJ }}) = j + 1;
    const uint8_t *__restrict px = (const uint8_t *)(x + {{ 2 * META }});
    if (j == 0) {
        {{ p }}_ccopy({{ p }}_CW(uint8_t, {{ c.taps + XPAY }}), px, {{ 3 * TAP }});
        {{ p }}_ccopy({{ p }}_CW(uint8_t, {{ c.hist + 8 * ROW }}), px + {{ 3 * TAP }}, {{ 2 * ROW }});
    } else if (j == 1) {
        {{ p }}_ccopy({{ p }}_CW(uint8_t, {{ c.hist }}), px, {{ 8 * ROW }});
    } else if (j == 2) {
        {{ p }}_ccopy({{ p }}_CW(uint8_t, {{ c.bc }}), px, {{ XPAY }});
    } else {
        {{ p }}_cstep(ws, (const bfloat16 *)px);
    }
}

// The next D object: y ([k block 32][slot sp][8]), then the new history's rows
__attribute__((minsize)) void {{ p }}_cemit{{ X }}(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    const int e = {{ p }}_CSW[{{ A_EIDX }}], S = {{ p }}_CST({{ C_S }}), ny = {{ CH * 2 // D_BYTES }} * (S + (S == 3));
    {{ p }}_CSW[{{ A_EIDX }}] = e + 1;
    const bfloat16 *__restrict src = e < ny ? {{ p }}_CW(bfloat16, {{ c.y }}) + {{ D_BYTES // 2 }} * e
                                            : {{ p }}_CW(bfloat16, {{ c.hist }}) + {{ D_BYTES // 2 }} * (e - ny);
    aie::store_v(d, aie::load_v<{{ D_BYTES // 2 }}>(src));
}

}  // extern "C"
""")


class ConvJob:
    """The conv step as a job of the attention cores (``decode.attn.Attn(conv=)``): its workspace (``need`` bytes from
    a region of the attention core's scratch on), its kernels (``render``), and the hooks ``decode.attn`` calls:
    ``begin_src`` (abegin: a conv region's first object), ``token_src`` (token object 0: the step's words)."""

    flag = F_CONV
    words = C_WORDS
    d_bytes = D_BYTES

    @staticmethod
    def layout() -> dict[str, int]:
        """Byte offsets of the conv's regions (from its base): taps, the emitted y and history (in a row: cemit), B and C
        rows."""
        out = {"taps": 0, "y": WIDTH * TAP}
        out["hist"] = out["y"] + CH * 2 * SMAX
        out["bc"] = out["hist"] + HIST * ROW
        out["size"] = out["bc"] + XPAY
        return out

    @property
    def need(self) -> int:
        return self.layout()["size"]

    def render(self, p: str, *, base: int, st: int, X: str, consts: dict[str, int]) -> str:
        """(X: the suffix of the worker's kernels when they are page code: ``decode.attn.Attn(pages=)``)"""
        return _SRC.render(p=p, base=base, st=st, X=X, c=self.layout(), CH=CH, WIDTH=WIDTH, ROW=ROW, TAP=TAP, XPAY=XPAY,
                           META=META, OBJS=OBJS, D_BYTES=D_BYTES, C_NCV=C_NCV, C_CX=C_CX, C_CJ=C_CJ, C_S=C_S,
                           C_SH=C_SH, **consts)

    def begin_src(self, p: str) -> str:
        """abegin's call for an F_CONV object (x, ws in scope)."""
        return f"{p}_cbegin(x, ws);"

    def token_src(self, A_WORDS: int) -> str:
        """abegin's lines for token object 0 (m, st in scope): the step's S and conv shift."""
        return (f"st[{A_WORDS + C_S}] = m[{M_TOK_S}];\n"
                f"            st[{A_WORDS + C_SH}] = m[{M_CSHIFT}];")
