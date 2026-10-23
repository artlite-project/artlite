"""An MoE layer's gathered streams in the prefill: the rows each column's control stream reads at data-dependent
addresses (``ctl``: per group of ``CHAIN`` rows a block that rewrites the chain's row BDs and pushes it; the rows
straight into the column's core, one 4 KB row an object).

- ``GatherNormOperator`` (the experts' inputs, XP): column c's chunks' rows of the residual x (bf16, its tokens per
  ``plan.Plan``) as RMSNorm(x) w panels, chunk k into the column's slot k: the post-attention norm's arithmetic
  (``prefill.norm``: bf16 x, 1 / rms as bf16 hi + lo) on each gathered token group of 8, so the panels equal its own
  (the router's and the shared expert's) for the same token. Only a chunk's groups of valid rows (the last one padded
  with the chunk's first token) are gathered and drained: the rest of its slot keeps finite panels of earlier layers,
  whose outputs are never read. Chunk k's first block pushes its drain (BD set k % 2, released lock ``LOCKS`` k % 2;
  its block waits for chunk k - 2's drain by that lock), the last one with a token the sequence waits for.
- ``CombineOperator`` (the experts' outputs): per token t of the chunk (column c: tokens c T / COLS ..) the delta
  w_s ds_t + sum_i w_i y_(pos[t, i]) as bf16 (the routed experts' rows of Y, gathered, and the shared expert's down
  ds; the weights fp32, as bf16 hi + lo products into an fp32 sum, rounded once), for the residual norm that adds it.
  A row of Y is read as the runs of its chunk slot's pass-major layout (``expert_gemm.ExpertGemmOperator.y_layout``),
  through the column's memtile, which sends it on in output order.
  A token's ds row ends in its weights (``WT_ROW`` fp32 [w_0 .. w_7, w_s, 0 ..]: 64 bytes, a cache line of their own
  past the GEMM's outputs, which the host writes per layer): a column's input is one strided transfer.

Both read their per-layer lists from the host (``plan``): the gather norm the column's row groups (bits 8 .. of the
scratchpad parameter ``expert_gemm.N_PARAM`` c), the combine the weights.
"""

import functools

import numpy as np
import torch
from aie.dialects._aiex_ops_gen import npu_sync as _npu_sync
from aie.dialects.aiex import _as_i32, npu_maskwrite32, npu_write32
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
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ... import txn
from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS, range32, stream_walk
from ..kernel_source import kernel_prefix
from . import ctl
from .expert_gemm import CHUNK, N_PARAM

CHAIN = 8  # rows per block (a token group)
ROW_BDS = (6, 7, 9, 10, 11, 12, 13, 15)  # a chain's row BDs (the sequences' own: 0 .. 2)
BLOCK_BDS = (5, 14)
DRAIN_BDS, LOCKS = (4, 8), (2, 3)  # (the gather norm's per-chunk drains: two sets, a shim lock each)
DRAIN_WORDS = 2 * 6 + 3  # (a chunk's drain BD and push)
WT_ROW = 16  # fp32 per token of the combine's weights

_SRC = JINJA_ENV.from_string(r"""
namespace {
// st: ss[8][32] -> inv_hi[64] | inv_lo[64] at st + 256 (bf16, lane 8 t + e = token t's 1 / rms)
__attribute__((noinline)) void {{ p }}_inv(float *__restrict st)
{
    float *__restrict q = st + 256;  // (fp32 scratch under the output: the tokens' sums folded to 8 lanes, 8 t + e)
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 4; t++) {  // (two tokens: their chunks of 8 summed by two unzips; a 64-lane fp32 vector built
                                   // in registers spills whole accumulators)
        const aie::vector<float, 64> v = aie::load_v<64>(st + 64 * t);
        const auto [a, b] = aie::interleave_unzip(v.template extract<32>(0), v.template extract<32>(1), 8);
        const aie::vector<float, 32> h = aie::add(a, b);
        const auto [c, d] = aie::interleave_unzip(h.template extract<16>(0), h.template extract<16>(1), 8);
        aie::store_v(q + 16 * t, aie::add(c, d));
    }
    aie::vector<bfloat16, 16> hi, lo;
    al::split2<16>(al::inv_rms<{{ E }}, 8>(al::group8<false>(aie::load_v<64>(q)).template grow<16>(), {{ eps }}f), hi, lo);
    aie::store_v((bfloat16 *)(st + 256), al::expand8(hi.template extract<8>(0)));
    aie::store_v((bfloat16 *)(st + 256) + 64, al::expand8(lo.template extract<8>(0)));
}
}  // namespace

extern "C" {

void {{ p }}_w(const bfloat16 *__restrict in, bfloat16 *__restrict w)
{
    al::copy<{{ E }}, 32>(w, in);
}

// a gathered row (token t of the group) -> g[t], ss[t][32 lanes] = its squares
void {{ p }}_row(const bfloat16 *__restrict x, bfloat16 *__restrict g, float *__restrict ss, int32_t t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)x;
    bfloat16 *__restrict pg = g + {{ E }} * t;
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < {{ E // 32 }}; j++) {
        aie::vector<bfloat16, 32> v = aie::load_v<32>(px);
        px += 32;
        aie::store_v(pg, v);
        pg += 32;
        acc = aie::mac_square(acc, v);
    }
    aie::store_v(ss + 32 * t, acc.template to_vector<float>());
}

// pass B, chunk kc: 32 panels (8 tokens x 8 columns) of g * w * (1 / rms) as bfp16ebs8 -> out (``prefill.norm``'s)
void {{ p }}_b(const bfloat16 *__restrict g, const bfloat16 *__restrict w, float *__restrict st,
                 bfloat16 *__restrict out, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict ih = (bfloat16 *)(st + 256);
    bfloat16 *__restrict il = ih + 64;
    if (kc == 0)
        {{ p }}_inv(st);
    const aie::vector<bfloat16, 64> vh = aie::load_v<64>(ih);
    const aie::vector<bfloat16, 64> vl = aie::load_v<64>(il);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> o((bfp16ebs8 *)out);
    const bfloat16 *__restrict pg = g + {{ C }} * kc;
    const bfloat16 *__restrict pw = w + {{ C }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ C // 8 }}; kb++) {
        aie::vector<bfloat16, 64> xp = aie::concat(
            aie::concat(aie::load_v<8>(pg), aie::load_v<8>(pg + {{ E }})),
            aie::concat(aie::load_v<8>(pg + {{ 2 * E }}), aie::load_v<8>(pg + {{ 3 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 4 * E }}), aie::load_v<8>(pg + {{ 5 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 6 * E }}), aie::load_v<8>(pg + {{ 7 * E }})));
        aie::vector<bfloat16, 64> xw = aie::mul(xp, aie::load_v<8>(pw).template grow_replicate<64>())
                                           .template to_vector<bfloat16>();
        o.push(aie::mac(aie::mul(xw, vh), xw, vl).template to_vector<bfp16ebs8>());
        pg += 8;
        pw += 8;
    }
}

// the combine: acc (fp32 [E]) = w_s ds (the object: ds as {{ COLS }} blocks of {{ E // COLS }}, {{ DP }} apart, then
// the {{ WT }} fp32 weights); then acc += w_i y (slot i)
void {{ p }}_cs(const bfloat16 *__restrict obj, float *__restrict acc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float w = ((const float *)(obj + {{ COLS * DP }}))[8];
    const bfloat16 wh = (bfloat16)w;
    const bfloat16 wl = (bfloat16)(w - (float)wh);
    const aie::vector<bfloat16, 32> h = aie::broadcast<bfloat16, 32>(wh), l = aie::broadcast<bfloat16, 32>(wl);
    float *__restrict pa = acc;
    for (int b = 0; b < {{ COLS }}; b++) {
        const bfloat16 *__restrict pd = obj + {{ DP }} * b;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ E // COLS // 32 }}; j++) {
            const aie::vector<bfloat16, 32> d = aie::load_v<32>(pd);
            pd += 32;
            aie::store_v(pa, aie::mac(aie::mul(d, h), d, l).template to_vector<float>());
            pa += 32;
        }
    }
}

void {{ p }}_cy(const bfloat16 *__restrict y, const bfloat16 *__restrict obj, float *__restrict acc, int32_t i)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float w = ((const float *)(obj + {{ COLS * DP }}))[i];
    const bfloat16 wh = (bfloat16)w;
    const bfloat16 wl = (bfloat16)(w - (float)wh);
    const aie::vector<bfloat16, 32> h = aie::broadcast<bfloat16, 32>(wh), l = aie::broadcast<bfloat16, 32>(wl);
    const bfloat16 *__restrict py = y;
    float *__restrict pa = acc;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < {{ E // 32 }}; j++) {
        const aie::vector<bfloat16, 32> v = aie::load_v<32>(py);
        py += 32;
        aie::accum<accfloat, 32> a(aie::load_v<32>(pa));
        aie::store_v(pa, aie::mac(aie::mac(a, v, h), v, l).template to_vector<float>());
        pa += 32;
    }
}

// acc -> bf16 (one rounding)
void {{ p }}_co(const float *__restrict acc, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int j = 0; j < {{ E // 32 }}; j++)
        aie::store_v(out + 32 * j, aie::accum<accfloat, 32>(aie::load_v<32>(acc + 32 * j)).template to_vector<bfloat16>());
}

}  // extern "C"
""")


class _Gathered(ArtcOperator):
    """The common part: E (the hidden size), the row source rows' bytes, the control stream's BDs."""

    p = kernel_prefix("mg")

    def __init__(self, *, E: int, eps: float = 1e-6, ds_pitch: int = 0):
        self.E, self.eps, self.ds_pitch = E, eps, ds_pitch
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, E=self.E, C=256, eps=repr(self.eps), WT=WT_ROW, COLS=COLS,
                           DP=self.ds_pitch or self.E // COLS)

    @property
    def row_n(self) -> int:
        """bf16 of a gathered row."""
        return self.E

    @property
    def row_words(self) -> int:
        return self.row_n // 2

    @property
    def b0_words(self) -> int:
        """Block 0: one row object."""
        return self.row_words

    @functools.cached_property
    def _templates(self) -> list[ctl.Template]:
        """The blocks (``ctl.template``; bases: a chain's 8 rows): the first writes the chain's row BDs whole, the
        later ones their address words; both push the chain."""
        first: list[int] = []
        slots = []
        for j in range(CHAIN):
            slots.append((len(first) + 3, len(first) + 4, j, 0))
            first += ctl.bd_packets(ROW_BDS[j], self.row_bd_words(0, j))
        first += ctl.push(txn.MM2S, 1, ROW_BDS[0])
        later: list[int] = []
        pk = 1 << 30 | ctl.XID << 19
        for j in range(CHAIN):
            later += [ctl._HDR, ctl._ctl_header(txn.SHIM_BD0 + txn.BD_BYTES * ROW_BDS[j] + 4, 2), 0, pk]
        later += ctl.push(txn.MM2S, 1, ROW_BDS[0])
        return [ctl.template(first, slots), ctl.template(later, [(4 * j + 2, 4 * j + 3, j, 0) for j in range(CHAIN)])]

    def row_tap(self) -> TensorAccessPattern:
        """A gathered row from its address: one run."""
        return TensorAccessPattern((1, self.E), 0, [1, 1, 1, self.E], [0, 0, 0, 1])

    def row_bd_words(self, address: int, j: int) -> list[int]:
        """A row BD's words: one row at ``address``, chained to the next row BD of its chain (the last: none)."""
        return ctl.bd_words(address, self.row_tap(), packet=ctl.XID, next_bd=ROW_BDS[j + 1] if j + 1 < CHAIN else None)

    def stream(self, rows: np.ndarray, *, cb: int, out: np.ndarray | None = None) -> np.ndarray:
        """The control stream (uint32 words, at device address ``cb``; into ``out``, else a new array) gathering the
        rows at device addresses ``rows`` (in order, a multiple of CHAIN): block 1 writes the chain's row BDs whole,
        block k + 1 their address words for rows CHAIN k .. and pushes the chain (``ctl.assemble``)."""
        rows = np.asarray(rows, np.int64).reshape(-1, CHAIN)
        blocks = np.ones(rows.shape[0], np.int32)
        blocks[0] = 0
        out = np.empty(self.cb_words, np.uint32) if out is None else out
        return out[:ctl.assemble(out, self._templates, blocks, rows, base=cb, b0_words=self.b0_words, bds=BLOCK_BDS)]

    @staticmethod
    def stream_words(rows: int) -> int:
        """Words of a stream of ``rows`` rows at most (``stream``)."""
        blocks = -(-rows // CHAIN)
        return 2 * CHAIN * 6 + blocks * (CHAIN * 4 + 3 + ctl.NEXT_WORDS)

    def _route(self) -> None:
        for c in range(COLS):
            for reg, val in ctl.ROUTE:
                npu_write32(reg, val, column=c, row=0)


class GatherNormOperator(_Gathered):
    """XP of an MoE layer: per column its chunks' gathered rows normed (module docstring). ``cap``: chunks per column
    at most. Arguments: the control streams (per column ``cb_words``), the norm weight (E bf16), XP (out)."""

    def __init__(self, *, E: int, eps: float, cap: int):
        self.cap = cap
        super().__init__(E=E, eps=eps)

    @property
    def slot(self) -> int:
        """bf16 units of a chunk's panels."""
        return CHUNK * self.E * 9 // 16

    @property
    def group(self) -> int:
        """bf16 units of a token group's panels."""
        return 8 * self.E * 9 // 16

    @property
    def cb_words(self) -> int:
        return -(-(self.b0_words + self.stream_words(self.cap * CHUNK) + self.cap * DRAIN_WORDS) // 16) * 16

    @functools.cached_property
    def _drain_templates(self) -> dict[tuple[bool, int, bool], ctl.Template]:
        """(first block of the stream, drain set, the last chunk's) -> a chunk's first block (bases: its 8 rows, its
        slot's address, its drain's words): the drain BD whole and its push, the rows' BDs (the stream's first block:
        whole) and the chain's push."""
        out = {}
        for first in (False, True):
            for parity in (0, 1):
                for last in (False, True):
                    words: list[int] = []
                    slots = []
                    rel = ctl.bd_words(0, TensorAccessPattern((1, 2), 0, [1, 1, 1, 2], [0, 0, 0, 1]),
                                       release=(LOCKS[parity], 1))
                    slots += [(len(words) + 3, len(words) + 4, CHAIN, 0), (len(words) + 2, -1, CHAIN + 1, 0)]
                    words += ctl.bd_packets(DRAIN_BDS[parity], rel)
                    words += ctl.push(txn.S2MM, 0, DRAIN_BDS[parity], token=last)
                    base_t, base_s = self._templates[0 if first else 1]
                    off = len(words)
                    words += base_t.tolist()
                    slots += [(lo + off, hi + off, b, o) for lo, hi, b, o in base_s.tolist()]
                    out[first, parity, last] = ctl.template(words, slots)
        return out

    def stream(self, rows: np.ndarray, sizes: np.ndarray, *, cb: int, xp: int,
               out: np.ndarray | None = None) -> np.ndarray:
        """The control stream (uint32 words at device address ``cb``; into ``out``, else a new array) of a column's
        chunks: rows (n, CHUNK) the device addresses of their rows (padding: the chunk's first), sizes (n) their valid
        rows, the column's XP slots from device address xp: per chunk its groups of valid rows (``ctl.assemble``; its
        first block also pushes its drain)."""
        out = np.empty(self.cb_words, np.uint32) if out is None else out
        return self.streams([rows], [sizes], cb=[cb], xp=[xp], out=[out])[0]

    def streams(self, rows: list[np.ndarray], sizes: list[np.ndarray], *, cb: list[int], xp: list[int],
                out: list[np.ndarray]) -> list[np.ndarray]:
        """``stream`` of several columns (their blocks computed together): per column its rows, sizes, addresses and
        output array; returns each column's words."""
        n = np.array([len(s) for s in sizes])
        assert ((1 <= n) & (n <= self.cap)).all()
        g = -(-np.concatenate(sizes).astype(np.int64) // CHAIN)  # (groups per chunk)
        assert (g >= 1).all()
        col = np.repeat(np.arange(len(n)), n)  # (per chunk: its column, its index there)
        kk = np.arange(g.size) - np.repeat(np.cumsum(n) - n, n)
        k = np.repeat(np.arange(g.size), g)  # (per block its chunk, its group in the chunk)
        first = np.cumsum(g) - g
        j = np.arange(k.size) - first[k]
        bases = np.empty((k.size, CHAIN + 2), np.int64)
        bases[:, :CHAIN] = np.concatenate([np.asarray(r, np.int64) for r in rows]).reshape(g.size, -1, CHAIN)[k, j]
        bases[:, CHAIN] = (np.asarray(xp, np.int64)[col] + 2 * self.slot * kk)[k]
        bases[:, CHAIN + 1] = g[k] * self.group // 2  # (the drain's words)
        ts = [self._templates[1], *self._drain_templates.values()]  # (the drain templates by (first, parity, last))
        blocks = np.zeros(k.size, np.int32)
        blocks[first] = 1 + 4 * (kk == 0) + 2 * (kk % 2) + (kk == n[col] - 1)
        lk, start = np.zeros(k.size, np.int64), np.zeros(k.size, np.int64)
        lk[first], start[first] = np.array(LOCKS)[kk % 2], 1
        ends = np.cumsum(np.bincount(col[k], minlength=len(n)))
        res = []
        for c, (a, b) in enumerate(zip([0, *ends[:-1]], ends)):
            w = ctl.assemble(out[c], ts, blocks[a:b], bases[a:b], base=cb[c], b0_words=self.b0_words, bds=BLOCK_BDS,
                             acquire=(lk[a:b], start[a:b]))
            res.append(out[c][:w])
        return res

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * self.cb_words * 2,), torch.bfloat16), ArgSpec("in", (self.E,), torch.bfloat16),
                ArgSpec("out", (COLS * self.cap * self.slot,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        E, p, cap = self.E, self.p, self.cap
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(E,), bf]
        grp_ty = np.ndarray[(8 * E,), bf]
        pan_ty = np.ndarray[(256 // 8 * 36,), bf]  # (a chunk of 256 columns: 32 panels)
        st_ty = np.ndarray[(256 + 64,), np.dtype[np.float32]]
        k_w = Kernel(f"{p}_w", "kernel.o", [row_ty, row_ty])
        k_row = Kernel(f"{p}_row", "kernel.o", [row_ty, grp_ty, st_ty, np.int32])
        k_b = Kernel(f"{p}_b", "kernel.o", [grp_ty, row_ty, st_ty, pan_ty, np.int32])
        pg = [ScratchpadParameter(f"{N_PARAM}{c}", np.int32) for c in range(COLS)]
        xf = [ObjectFifo(row_ty, name=f"mgx{c}", depth=2, packet=True, packet_id=ctl.XID) for c in range(COLS)]
        wf = [ObjectFifo(row_ty, name=f"mgw{c}", depth=1) for c in range(COLS)]
        pf = [ObjectFifo(pan_ty, name=f"mgp{c}", depth=2) for c in range(COLS)]

        def core(x_f, w_f, p_f, kw, krow, kb, wb, st, g, prm):
            w = w_f.acquire(1)
            kw(w, wb)
            w_f.release(1)
            n = prm.read() // 256  # (the column's row groups: ``N_PARAM`` from bit 8)
            for _ in range32(n):
                for t in range_(8):
                    x = x_f.acquire(1)
                    krow(x, g, st, t)
                    x_f.release(1)
                for kc in range_(E // 256):
                    o = p_f.acquire(1)
                    kb(g, wb, st, o, kc)
                    p_f.release(1)

        workers = [Worker(core, [xf[c].cons(), wf[c].cons(), pf[c].prod(), k_w, k_row, k_b,
                                 Buffer(row_ty, name=f"mg_w{c}"), Buffer(st_ty, name=f"mg_st{c}"),
                                 Buffer(grp_ty, name=f"mg_g{c}"), pg[c]],
                          tile=Tile(c, 2), stack_size=1024, while_true=False, dynamic_objfifo_lowering=True)
                   for c in range(COLS)]
        n_cb, n_xp = COLS * self.cb_words * 2, COLS * cap * self.slot

        def seq(CB, W, XP, xh, wh, ph):
            sync_parameters()
            self._route()
            for c in range(COLS):  # (the drains' locks free, their tokens to the controller)
                for lk in LOCKS:
                    npu_write32(txn.SHIM_LOCK0 + 16 * lk, 1, column=c, row=0)
                npu_maskwrite32(txn.CTRL[txn.S2MM][0], txn.CONTROLLER_ID << 8, 0x1F00, column=c, row=0)
            tg = TaskGroup()
            for c in range(COLS):
                wh[c].fill(W, TensorAccessPattern((E,), 0, [1, 1, E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                xh[c].fill(CB, TensorAccessPattern((n_cb,), 2 * c * self.cb_words, [1, 1, 1, 2 * self.b0_words],
                                                   [0, 0, 0, 1]), packet=(0, ctl.CID), group=tg)
            tg.finish()
            for c in range(COLS):  # (the last chunk's drain: its token)
                _npu_sync(_as_i32(c), _as_i32(0), _as_i32(txn.S2MM), _as_i32(0), _as_i32(1), _as_i32(1))

        rt = Runtime(seq, [np.ndarray[(n_cb,), bf], np.ndarray[(E,), bf], np.ndarray[(n_xp,), bf],
                           [xf[c].prod(tile=Tile(c, 0), channel=1) for c in range(COLS)],
                           [wf[c].prod(tile=Tile(c, 0), channel=0) for c in range(COLS)],
                           [pf[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


def ds_row(E: int, ds_pitch: int) -> int:
    """bf16 per token of the shared down's output rows: COLS blocks of ds_pitch, then the weights (``WT_ROW`` fp32)."""
    return COLS * ds_pitch + 2 * WT_ROW


class CombineOperator(_Gathered):
    """The delta of an MoE layer's T tokens (module docstring): arguments the control streams (per column
    ``cb_words``), DS (per token its shared-down row and weights: ``ds_row``), the output (T rows of E bf16). Column c
    takes tokens c T / COLS .. ; ``ds_pitch``: the shared down's column pitch; ``y_layout``: Y's chunk slots
    (``ExpertGemmOperator.y_layout``)."""

    def __init__(self, *, E: int, T: int, ds_pitch: int, y_layout: tuple[int, int, int, int]):
        assert T % COLS == 0 and ds_pitch % 32 == 0
        self.T, self.y_layout = T, tuple(y_layout)
        passes, vp, tiles, run = self.y_layout
        assert passes * vp * tiles * run >= E and run % 2 == 0
        super().__init__(E=E, ds_pitch=ds_pitch)

    @property
    def row_n(self) -> int:
        passes, vp, tiles, run = self.y_layout
        return passes * vp * tiles * run

    def row_tap(self) -> TensorAccessPattern:
        """A row of Y from its address (``ExpertGemmOperator.y_offset``): passes x vp runs of tiles x run values, a
        chunk's tokens apart."""
        passes, vp, tiles, run = self.y_layout
        return TensorAccessPattern((1, 1 << 30), 0, [1, 1, passes * vp, tiles * run], [0, 0, CHUNK * tiles * run, 1])

    @property
    def cb_words(self) -> int:
        return -(-(self.b0_words + self.stream_words(self.T // COLS * 8)) // 16) * 16

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * self.cb_words * 2,), torch.bfloat16),
                ArgSpec("in", (self.T * ds_row(self.E, self.ds_pitch),), torch.bfloat16),
                ArgSpec("out", (self.T * self.E,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        E, p, T, dp = self.E, self.p, self.T, self.ds_pitch
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(self.row_n,), bf]
        dr = ds_row(E, dp)
        obj_ty = np.ndarray[(dr,), bf]  # (a token's ds row and weights)
        acc_ty = np.ndarray[(E,), np.dtype[np.float32]]
        k_cs = Kernel(f"{p}_cs", "kernel.o", [obj_ty, acc_ty])
        k_cy = Kernel(f"{p}_cy", "kernel.o", [row_ty, obj_ty, acc_ty, np.int32])
        out_ty = np.ndarray[(E,), bf]
        k_co = Kernel(f"{p}_co", "kernel.o", [acc_ty, out_ty])
        # (Y's rows: shim -> memtile -> the core in output order: per pass, tile and C object its run)
        passes, vp, tiles, run = self.y_layout
        order = stream_walk(self.row_n, [(passes, vp * tiles * run), (tiles, run), (vp, tiles * run), (run, 1)])
        y3 = [ObjectFifo(row_ty, name=f"mcy{c}", depth=2, packet=True, packet_id=ctl.XID) for c in range(COLS)]
        yf = [y3[c].cons().forward(obj_type=row_ty, name=f"mcy1{c}", depth=2, tile=Tile(c, 1), to_stream=order)
              for c in range(COLS)]
        df = [ObjectFifo(obj_ty, name=f"mcd{c}", depth=2) for c in range(COLS)]
        of = [ObjectFifo(out_ty, name=f"mco{c}", depth=2) for c in range(COLS)]
        tpc = T // COLS

        def core(y_f, d_f, o_f, cs, cy, co, acc):
            for _ in range_(tpc):
                d = d_f.acquire(1)
                cs(d, acc)
                for i in range_(8):
                    y = y_f.acquire(1)
                    cy(y, d, acc, i)
                    y_f.release(1)
                d_f.release(1)
                o = o_f.acquire(1)
                co(acc, o)
                o_f.release(1)

        workers = [Worker(core, [yf[c].cons(), df[c].cons(), of[c].prod(), k_cs, k_cy, k_co,
                                 Buffer(acc_ty, name=f"mc_acc{c}")],
                          tile=Tile(c, 2), stack_size=1024, while_true=False, dynamic_objfifo_lowering=True)
                   for c in range(COLS)]
        n_cb, n_ds, n_o = COLS * self.cb_words * 2, T * dr, T * E

        def seq(CB, DS, O, yh, dh, oh):
            self._route()
            tg = TaskGroup()
            for c in range(COLS):
                t0 = c * tpc
                yh[c].fill(CB, TensorAccessPattern((n_cb,), 2 * c * self.cb_words, [1, 1, 1, 2 * self.b0_words],
                                                   [0, 0, 0, 1]), packet=(0, ctl.CID), group=tg)
                dh[c].fill(DS, TensorAccessPattern((n_ds,), t0 * dr, [1, 1, 1, tpc * dr], [0, 0, 0, 1]), group=tg)
                oh[c].drain(O, TensorAccessPattern((n_o,), t0 * E, [1, 1, tpc, E], [0, 0, E, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_cb,), bf], np.ndarray[(n_ds,), bf], np.ndarray[(n_o,), bf],
                           [y3[c].prod(tile=Tile(c, 0), channel=1) for c in range(COLS)],
                           [df[c].prod(tile=Tile(c, 0), channel=0) for c in range(COLS)],
                           [of[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
