"""Elementwise activations through AIE2P's gather LUT (``load_4x64_lo / hi`` = ``vldb.4x64``: four 64-bit table
reads per instruction, ldb slot), bf16 in and out.

SiLU: silu(x) = relu(x) + g(t), g(t) = silu(-t), t = min(|x|, 24) (silu(x) - silu(-x) = x). g is a table of 97
segments of 1/4 over [0, 24], each a quadratic in the offset r from its centre: c0 + c1 r + c2 r^2 (c0 fp32, c1 and c2
bf16; minimax over the segment's bf16 points, in units of the result's bf16 ULP):
    v = bf16(t + 32)              t rounded to the 1/4 grid ([32, 56] is a binade of ULP 1/4)
    r = bf16((32 - v) + t)        exact ((32 - v) first: tiny t stay exact)
    y = bf16(c0 + c1 r + c2 bf16(r^2) + max(x, 0))
    table byte offset = 64 v - 2048 (``bfloat16_to_int(v, 6)``: ``vfloor``)
Max 0.59 bf16 ULP over [-24, inf), 99.84% correctly rounded (below -24: silu(-24), within 1e-9 of the exact value).
The vtanh / vexp2 form: 6.6 ULP; ``kmath.silu``: 1.17 ULP at ~16 cycles per element. 873 cycles per 1024
elements on the NPU (the vtanh / vexp2 form: 2682), in two loops: t into y (II 2), then the table pass in place (II 16
per 32 lanes; bound 9: 8 table reads, the shuffles). Fewer, longer loops ran slower: the constant vectors (1, 24, the
abs mask, 0, the lanes' table pointers) take 5 of the 12 x registers, and the post-pipeliner's allocation fails.

The gather unit (probed on the NPU): lane k of an instruction reads table ab (k % 4 < 2) or
cd, its odd lanes 32 bytes further; offset bits 5:4 select the entry of a 64-byte group, bits 3:0 are ignored. A
table is stored as groups of 4 entries, each group twice (64 bytes per 4 segments), in two copies (ab, cd): 32 bytes
per segment, 3.2 KB for SiLU (cd aliased onto ab: half the memory, 9% slower). Entries: [c2 bf16 | c1 bf16 | c0 fp32]
(``shuffle(.., T16_16x4_lo)`` gives c2 | c1 per 16 lanes, ``T32_16x2_hi`` c0).

Sigmoid (``full``: the function itself over [-16, 16], x + 48 in the binade [32, 64] of ULP 1/4: 129 segments, no
symmetry, x clamped to the range): within 0.6 bf16 ULP, ~1 cycle per element (an fp32 exp2 and reciprocal: ~5).

vexp2 is Schraudolph's exp exactly, 2^floor(y) (1 + the 7-bit truncated fraction of y) of y truncated to bf16 (off by
up to 6.1%; exact at non-negative integers); vtanh returns x below 0.5 and is up to 19 ULP off below 1 (both probed
over every bf16 input).
"""

import functools
import hashlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .base import CACHE_DIR, JINJA_ENV
from .kernel_source import kernel_prefix

LUT = kernel_prefix("lut")  # the prefix of the tables' kernels (``kernel_source``: one copy for all operators)


def _bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32)


def bf16_values() -> np.ndarray:
    """Every finite bf16 value, as float32."""
    x = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().numpy()
    return x[np.isfinite(x)]


def silu64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / (1.0 + np.exp(np.minimum(-x, 700.0)))


def sigmoid64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.minimum(-np.asarray(x, np.float64), 700.0)))


def gelu64(x: np.ndarray) -> np.ndarray:
    """gelu in its tanh form (gelu_pytorch_tanh, ggml_gelu) as x sigmoid(z), z = 2 sqrt(2 / pi) (x + 0.044715 x^3)."""
    x = np.asarray(x, np.float64)
    z = 2.0 * np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return x / (1.0 + np.exp(np.minimum(-z, 700.0)))


def gelu_erf64(x: np.ndarray) -> np.ndarray:
    """gelu in its erf form (x Phi(x): MiniCPM-V's final merger)."""
    import torch
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + torch.erf(torch.from_numpy(x / np.sqrt(2.0))).numpy())


def ulp_bf16(t: np.ndarray) -> np.ndarray:
    """The bf16 ULP at t (normal range)."""
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(t), 2.0 ** -126))) - 7)


def _minimax(cols: list[np.ndarray], f: np.ndarray, w: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Coefficients c of sum c_k col_k minimizing max |w (sum c_k col_k + fixed - f)| (Lawson's iteration)."""
    A = np.stack(cols, 1) * w[:, None]
    b = (f - fixed) * w
    u = np.full(len(f), 1.0 / len(f))
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(300):
        s = np.sqrt(u)
        c = np.linalg.lstsq(A * s[:, None], b * s, rcond=None)[0]
        e = np.abs(A @ c - b)
        if e.max() <= 1e-9 * max(np.abs(b).max(), 1e-30):
            break
        u = u * e
        u /= u.sum()
    return c


def _round_bf16(c: float) -> float:
    return float(_bf16(np.float32([c]))[0])


@dataclass(frozen=True)
class ActTable:
    """An activation f with f(x) - f(-x) = x (x sigmoid of an odd function: SiLU, GELU) as relu(x) + g(t), g(t) =
    f(-t), t = min(|x|, t_max), g a table of segments of ``seg`` over [0, t_max] (t + m in one binade of ULP seg).
    ``full``: f itself over [-t_max, t_max] (x clamped; x + m in the binade [m - t_max, m + t_max]: sigmoid).
    ``floor``: the error that counts in units of the result's ULP is at least this absolute one (GELU's tail, below
    1e-7 at t = 5, decays faster than a quadratic per segment follows; relative accuracy there is irrelevant)."""

    name: str
    f64: Callable[[np.ndarray], np.ndarray]
    seg: float
    m: float
    t_max: float
    floor: float = 0.0
    full: bool = False

    @property
    def base(self) -> float:
        """The reduced argument's binade start (its first segment's)."""
        return self.m - self.t_max if self.full else self.m

    @property
    def segs(self) -> int:
        return int((2 if self.full else 1) * self.t_max / self.seg) + 1

    @property
    def groups(self) -> int:
        return -(-self.segs // 4)

    def _reduce(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kernel's v, r, bf16(r^2) of t (bf16 values in [0, t_max])."""
        v = _bf16(_f32(self.m) + _f32(t))
        r = _bf16(_f32(_f32(self.m) - v) + _f32(t))
        return v, r, _bf16(_f32(r) * _f32(r))

    @functools.cached_property
    def coefficients(self) -> np.ndarray:
        """(segs, 3): c0 (fp32 values), c1, c2 (bf16 values) of g's segments (cached on disk: ~1 s to fit, keyed by the
        table's parameters and the fitting code)."""
        code = [inspect.getsource(f) for f in (self.f64, ActTable._fit, ActTable._reduce, _minimax, _round_bf16, _bf16, _f32,
                                               bf16_values, ulp_bf16)]
        key = hashlib.sha256(repr((self.name, self.seg, self.m, self.t_max, self.floor, self.full, code)).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, "act_lut", f"{self.name}_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        coef = self._fit()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.npy"
        np.save(tmp, coef)
        os.replace(tmp, path)
        return coef

    def _fit(self) -> np.ndarray:
        x = bf16_values()
        if self.full:
            t = x[(np.abs(x) <= self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self._reduce(t)
            f = self.f64(t)
            w = 1.0 / np.maximum(ulp_bf16(f), self.floor)
        else:
            t = -x[(x <= 0) & (x >= -self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self._reduce(t)
            f = self.f64(-t)
            # (the error of y = relu(x) + g on both sides)
            w = 1.0 / np.maximum(np.minimum(ulp_bf16(f), ulp_bf16(self.f64(t))), self.floor)
        s = ((v - self.base) / self.seg).astype(np.int64)
        coef = np.zeros((self.segs, 3))
        for k in range(self.segs):
            sel = (s == k) & ((t != 0) | self.full)
            rk, r2k, fk, wk = r[sel].astype(np.float64), r2[sel].astype(np.float64), f[sel], w[sel]
            ones, zero = np.ones_like(rk), np.zeros_like(fk)
            if k == 0 and not self.full:  # c0 = 0: tiny t stay relative-accurate
                c2 = _round_bf16(_minimax([rk, r2k], fk, wk, zero)[1])
                c1 = _round_bf16(_minimax([rk], fk, wk, c2 * r2k)[0])
                coef[k] = 0.0, c1, c2
                continue
            c2 = _round_bf16(_minimax([ones, rk, r2k], fk, wk, zero)[2])
            c1 = _round_bf16(_minimax([ones, rk], fk, wk, c2 * r2k)[1])
            c0 = _minimax([ones], fk, wk, c1 * rk + c2 * r2k)[0]
            coef[k] = float(np.float32(c0)), c1, c2
        return coef

    def table_words(self) -> np.ndarray:
        """One table copy as uint32 words: entries [c2 | c1 (high half), c0], groups of 4 entries stored twice (the
        gather's odd lanes read 32 bytes further)."""
        coef = self.coefficients

        def bits(c: np.ndarray) -> np.ndarray:
            return torch.from_numpy(_f32(c)).bfloat16().view(torch.int16).numpy().view(np.uint16).astype(np.uint32)

        words = np.zeros((self.groups * 4, 2), np.uint32)
        words[:len(coef), 0] = bits(coef[:, 1]) << 16 | bits(coef[:, 2])
        words[:len(coef), 1] = _f32(coef[:, 0]).view(np.uint32)
        g = words.reshape(self.groups, 8)
        return np.concatenate([g, g], 1).reshape(-1)

    def table_buffer(self) -> np.ndarray:
        """The two table copies (ab, cd) as one uint32 array: a kernel's ``tab`` (``render(table_arg=True)``)."""
        w = self.table_words()
        return np.concatenate([w, w])

    def render(self, *, table_arg: bool = False, gate_up: bool = False, x_step: int = 32) -> str:
        """The C++ of ``{LUT}_<name>_lut<N>(x, y)`` (x and y distinct buffers) and its tables. ``table_arg``: no static
        tables (a core's .rodata lands after the stack, over the buffers a design pins there): ``{LUT}_<name>_tab`` takes
        the tables (``table_buffer``, a worker buffer of the core) as its last parameter. ``gate_up``: only ``_tab``,
        its x the gate halves of a GEMM's gate/up C tile ([gate 64 | up 64] per 128 values); ``x_step``: x's 32 values
        of an iteration are that many elements apart (otherwise)."""
        shift = int(np.log2(16 / self.seg))  # 16 bytes of offset per segment
        return _TABLE.render(p=LUT, name=self.name, words=[f"0x{w:08x}u" for w in self.table_words()], M=repr(self.m),
                             bias=int(self.base * 2 ** shift), shift=shift, t_max=repr(self.t_max), table_arg=table_arg,
                             gate_up=gate_up, x_step=x_step, full=self.full)

    def render_unrolled(self, n: int) -> str:
        """The C++ of ``{LUT}_tabu(x, t, t2, y, tab)``: the table pass over n values (a multiple of 64; t = t2 = min(|x|,
        t_max) as bf16, y may be t; ``tab``: ``table_buffer``), two 32-lane halves per iteration with every load
        before the gathers (the gathers carry no memory operand, so each memory operation orders them: one chain per
        iteration ran at 30 cycles per 32 lanes, two at ~22). A plain function (not a template): a kernel of a
        program-memory-patched design must be emitted where it is defined."""
        assert n % 64 == 0
        shift = int(np.log2(16 / self.seg))
        return _UNROLLED.render(p=LUT, n64=n // 64, bias=int(self.m * 2 ** shift), shift=shift,
                                nw=len(self.table_words()))


_UNROLLED = JINJA_ENV.from_string(r"""
[[clang::always_inline]] inline void {{ p }}_lut16u(v16int32 ptr, v32bfloat16 &q0, v32bfloat16 &q1)
{
    q0 = insert(q0, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 0)));
    q0 = insert(q0, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 0)));
    q1 = insert(q1, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 1)));
    q1 = insert(q1, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 1)));
}

__attribute__((noinline)) void {{ p }}_tabu(const bfloat16 *__restrict x, const bfloat16 *__restrict t_,
                                            const bfloat16 *__restrict t2_, bfloat16 *__restrict y,
                                            const uint32_t *__restrict tab)
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k(aie::broadcast<float, 32>(32.0f));
    const v16int32 base = ::sel(broadcast_s32((long)tab - {{ bias }}), broadcast_s32((long)(tab + {{ nw }}) - {{ bias }}), 0xCCCC);
    const bfloat16 __aie_dm_resource_d *__restrict px = (const bfloat16 __aie_dm_resource_d *)x;
    const bfloat16 __aie_dm_resource_a *__restrict pt = (const bfloat16 __aie_dm_resource_a *)t_;
    const bfloat16 __aie_dm_resource_c *__restrict pt2 = (const bfloat16 __aie_dm_resource_c *)t2_;
    bfloat16 __aie_dm_resource_b *__restrict py = (bfloat16 __aie_dm_resource_b *)y;
    AIE_PREPARE_FOR_POSTPIPELINING AIE_LOOP_REG_REWRITE_SWPAWARE
    for (int j = 0; j < {{ n64 }}; j++) {
{% for u in range(2) %}
        const aie::vector<bfloat16, 32> t{{ u }} = aie::load_v<32>(pt + {{ 32 * u }});
        const aie::vector<bfloat16, 32> t2{{ u }} = aie::load_v<32>(pt2 + {{ 32 * u }});
        const aie::vector<bfloat16, 32> xr{{ u }} = aie::max(aie::load_v<32>(px + {{ 32 * u }}), aie::zeros<bfloat16, 32>());
{% endfor %}
{% for u in range(2) %}
        const aie::vector<bfloat16, 32> v{{ u }} = aie::mac(k, t{{ u }}, one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> r{{ u }} = aie::mac(aie::msc(k, v{{ u }}, one), t2{{ u }}, one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> r2{{ u }} = aie::mul(r{{ u }}, r{{ u }}).template to_vector<bfloat16>();
{% endfor %}
{% for u in range(2) %}
        v32bfloat16 q0{{ u }}, q1{{ u }}, q2{{ u }}, q3{{ u }};
        {{ p }}_lut16u(::add(base, bfloat16_to_int(v{{ u }}.extract<16>(0), {{ shift }})), q0{{ u }}, q1{{ u }});
        {{ p }}_lut16u(::add(base, bfloat16_to_int(v{{ u }}.extract<16>(1), {{ shift }})), q2{{ u }}, q3{{ u }});
{% endfor %}
{% for u in range(2) %}
        aie::accum<accfloat, 32> acc{{ u }};
        {
            const v32bfloat16 s0 = ::shuffle(q0{{ u }}, q1{{ u }}, T16_16x4_lo);
            const v32bfloat16 s1 = ::shuffle(q2{{ u }}, q3{{ u }}, T16_16x4_lo);
            acc{{ u }}.insert(0, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q0{{ u }}, q1{{ u }}, T32_16x2_hi)));
            acc{{ u }}.insert(1, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q2{{ u }}, q3{{ u }}, T32_16x2_hi)));
            acc{{ u }} = aie::mac(acc{{ u }}, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_hi)), r{{ u }});
            acc{{ u }} = aie::mac(acc{{ u }}, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_lo)), r2{{ u }});
            acc{{ u }} = aie::mac(acc{{ u }}, xr{{ u }}, one);
        }
{% endfor %}
{% for u in range(2) %}
        aie::store_v(py + {{ 32 * u }}, acc{{ u }}.template to_vector<bfloat16>());
{% endfor %}
        px += 64;
        pt += 64;
        pt2 += 64;
        py += 64;
    }
}
""")

_TABLE = JINJA_ENV.from_string(r"""
{% if not table_arg %}
alignas(64) static const uint32_t {{ p }}_{{ name }}_ab[{{ words | length }}] = { {{ words | join(", ") }} };
alignas(64) static const uint32_t {{ p }}_{{ name }}_cd[{{ words | length }}] = { {{ words | join(", ") }} };
{% endif %}

// 16 table reads: lanes 0-7 into q0, 8-15 into q1
[[clang::always_inline]] inline void {{ p }}_{{ name }}_lut16(v16int32 ptr, v32bfloat16 &q0, v32bfloat16 &q1)
{
    q0 = insert(q0, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 0)));
    q0 = insert(q0, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 0)));
    q1 = insert(q1, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 1)));
    q1 = insert(q1, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 1)));
}

// y = c0 + c1 r + c2 r^2{{ " + max(x, 0) of t = min(|x|, %d)" % (t_max | float | int) if not full else " of t = x clamped to +-%d" % (t_max | float | int) }} (in t_ and t2_: y, read before it is written){{ "; x: the gate halves of a gate/up C tile" if gate_up }}
template <int N>
__attribute__((noinline)) void {{ p }}_{{ name }}_tab(const bfloat16 *__restrict x, const bfloat16 *__restrict t_,
                                                const bfloat16 *__restrict t2_, bfloat16 *__restrict y{{ ", const uint32_t *__restrict tab" if table_arg }})
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k(aie::broadcast<float, 32>({{ M }}f));
{% if table_arg %}
    const v16int32 base = ::sel(broadcast_s32((long)tab - {{ bias }}), broadcast_s32((long)(tab + {{ words | length }}) - {{ bias }}),
                                0xCCCC);
{% else %}
    const v16int32 base = ::sel(broadcast_s32((long){{ p }}_{{ name }}_ab - {{ bias }}),
                                broadcast_s32((long){{ p }}_{{ name }}_cd - {{ bias }}), 0xCCCC);
{% endif %}
{% if gate_up %}
    int cg = 0;
{% endif %}
    const bfloat16 __aie_dm_resource_d *__restrict px = (const bfloat16 __aie_dm_resource_d *)x;
    const bfloat16 __aie_dm_resource_a *__restrict pt = (const bfloat16 __aie_dm_resource_a *)t_;
    const bfloat16 __aie_dm_resource_c *__restrict pt2 = (const bfloat16 __aie_dm_resource_c *)t2_;
    bfloat16 __aie_dm_resource_b *__restrict py = (bfloat16 __aie_dm_resource_b *)y;
    AIE_PREPARE_FOR_POSTPIPELINING AIE_LOOP_REG_REWRITE_SWPAWARE
    for (int j = 0; j < N / 32; j++) {
        const aie::vector<bfloat16, 32> t = aie::load_v<32>(pt);
        const aie::vector<bfloat16, 32> t2 = aie::load_v<32>(pt2);
        const aie::vector<bfloat16, 32> v = aie::mac(k, t, one).template to_vector<bfloat16>();  // t on the grid
        v32bfloat16 q0, q1, q2, q3;
        {{ p }}_{{ name }}_lut16(::add(base, bfloat16_to_int(v.extract<16>(0), {{ shift }})), q0, q1);
        {{ p }}_{{ name }}_lut16(::add(base, bfloat16_to_int(v.extract<16>(1), {{ shift }})), q2, q3);
        const aie::vector<bfloat16, 32> r = aie::mac(aie::msc(k, v, one), t2, one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> r2 = aie::mul(r, r).template to_vector<bfloat16>();
        const v32bfloat16 s0 = ::shuffle(q0, q1, T16_16x4_lo);  // c2 | c1 of lanes 0-15
        const v32bfloat16 s1 = ::shuffle(q2, q3, T16_16x4_lo);
        aie::accum<accfloat, 32> acc;
        acc.insert(0, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q0, q1, T32_16x2_hi)));  // c0
        acc.insert(1, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q2, q3, T32_16x2_hi)));
        acc = aie::mac(acc, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_hi)), r);
        acc = aie::mac(acc, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_lo)), r2);
{% if not full %}
        acc = aie::mac(acc, aie::max(aie::load_v<32>(px), aie::zeros<bfloat16, 32>()), one);
{% endif %}
        aie::store_v(py, acc.template to_vector<bfloat16>());
{% if gate_up %}
        px = add_2d_ptr(px, 96, 1, cg, 32);  // the pair's second gate half, then the next pair
{% else %}
        px += {{ x_step }};
{% endif %}
        pt += 32;
        pt2 += 32;
        py += 32;
    }
}

{% if not gate_up %}
// {{ name }} of N bf16 (a multiple of 32) into y, through the table (act_lut.py; sets the rounding mode)
template <int N>
__attribute__((noinline)) void {{ p }}_{{ name }}_lut(const bfloat16 *__restrict x, bfloat16 *__restrict y)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {
        const aie::vector<bfloat16, 32> hi = aie::broadcast<bfloat16, 32>({{ t_max }}f);
{% if full %}
        const aie::vector<bfloat16, 32> lo = aie::broadcast<bfloat16, 32>(-{{ t_max }}f);
{% endif %}
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)x;
        bfloat16 __aie_dm_resource_b *__restrict pw = (bfloat16 __aie_dm_resource_b *)y;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < N / 32; j++) {
{% if full %}
            aie::store_v(pw, aie::min(aie::max(aie::load_v<32>(px), lo), hi));
{% else %}
            aie::store_v(pw, aie::min(aie::abs(aie::load_v<32>(px)), hi));
{% endif %}
            px += 32;
            pw += 32;
        }
    }
    {{ p }}_{{ name }}_tab<N>(x, y, y, y);
}
{%- endif %}
""")


SILU = ActTable("silu", silu64, 0.25, 32.0, 24.0)
# GELU: 65 segments of 1/8 over [0, 8] (g(-8) ~ -4e-21): within 0.58 bf16 ULP for |x| < 4, 1.42 in [-6, -4) (absolute
# 1.4e-7), below: within 6e-13; kmath.gelu: 1.7 ULP at ~7 cycles per element
GELU = ActTable("gelu", gelu64, 0.125, 16.0, 8.0)
GELU_ERF = ActTable("gelu_erf", gelu_erf64, 0.125, 16.0, 8.0)
SIGMOID = ActTable("sigmoid", sigmoid64, 0.25, 48.0, 16.0, full=True)


def render_silu() -> str:
    """The C++ of ``{LUT}_silu_lut<N>(x, y)`` (x and y distinct buffers) and its tables."""
    return SILU.render()
