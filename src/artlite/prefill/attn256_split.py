"""The prefill attention of head_dim 256 split between core pairs: the qwen35 models with a q head per column and two KV
heads (MiniCPM-V 1B), causal. A chunk of T tokens (a multiple of 96) at p0 from ``attn256.AttnPrepOperator``'s Q slots
(``qg`` = ``QG``: passes of 96 queries) and the cache it wrote, to the attn_out GEMM's A, as ``PrefillAttnOperator``.

Why split: a core of ``PrefillAttnOperator`` keeps Q, the scores, o and both cache streams' objects in its L1, which
holds two row panels (16 queries); every 64-query pass streams the history, and at long context the passes' DDR reads
and each core's two input channels bound it (~1170 cycles per 32-key group). Here pair k of column c (head c) splits a
group's work: A (row 3 + 2 k) the scores and the softmax's first pass, B (row 2 + 2 k) its second pass, the row sums and
P V, the two through shared memory. A keeps Q and no o, B o and no Q, each takes one half of the cache (A the K records,
B the V records) on its two input channels: a pair holds six row panels (48 queries), panel 5's o in A's memory (``o5``):
96-query passes, the history streamed 2/3 as often.

Streams per KV head h (columns 4 h .. 4 h + 3), each from one shim channel through its memtile, broadcast to the head's
8 A or 8 B cores, objects of 4608 B, depth 1 at the cores:
- K0 / K1 (columns 4 h, 4 h + 1): a pass's Q slots, d-blocks 0 .. 15 / 16 .. 31 (24 objects each: core j's row panels
  2 i, 2 i + 1, the slots as the prep wrote them, [d-block][hi | lo]: A keeps the hi panels: single Q; a shim BD's
  innermost dimension holds at most half a slot), then per group record pair 0 / 1 of its K object ([d-block 32][record
  2]: the decode's layout, as it lies);
- V0 / V1 (columns 4 h + 2, 4 h + 3): per group d-blocks 0 .. 15 / 16 .. 31 of its V object, the memtile interleaving
  the record pairs per d-block ([d-block 16][record 4]); then the pass's gate slots' halves (24 objects each: two slots
  of a core's panels 2 i, 2 i + 1, interleaved the same way).

Per group A: the scores of its six panels in three tiles of two per record pair (single Q panels against two K panels
per d-block: 4 pops, 4 MMULs), the diagonal groups' masks, pass 1 into the A -> B object (P = 2^(round(s - m) + 64), f),
the lazy maximum's raise (alpha into the object, flagged; pass 1 again). B: the rescale when flagged, pass 2 (P as bfp16
panels), l += P E^T on the MMUL (E a panel of 1 / 8), o += P V over V0's then V1's object (three tiles of two panels).
At a pass's end B writes each panel's o / l sigmoid(gate) (the gates read in place from the gate objects) as the attn_out
GEMM's A panels, the column's two B cores joined in its memtile.

L1 by 16 KB region (``l1_layout``), A: 0 the stack, K0's object, an A -> B object; 1 K1's, the other A -> B object, m, b;
2 q; 3 the scores, o5. B: 0 the stack, V0's and V1's objects, the P panels, l, the output object; 1 o4; 2 o0, o2; 3 o1,
o3. QK streams K (0, 1) against q (2); P V V (0) against two o in two regions.
"""

import numpy as np
from aie.dialects.aiex import npu_write32
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

from artlite.decode.attn256 import AttnConfig
from artlite.npu_regs import window_counter_writes
from artlite.operators import templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.bfp16 import panel_elems
from artlite.operators.chain import ChainPart, chained_fill
from artlite.operators.design import COLS, MAX_BD_ITER, range32, stream_walk
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.l1_layout import REGION, pin_l1_buffers
from artlite.operators.util import Attr, arg_elems, mask64

from .attn256 import A_SET, GROUP, HD, ND, OBJ_ELEMS, QA_N, QA_OFF, SLOT

QG = 3  # token groups per core-equivalent of a pass (the prep's ``qg``): 4 x 3 x 8 = 96 queries per column
R = 6  # row panels per pair
PASS = 96  # queries per pass and column
SL = SLOT // 2  # bf16 of a slot (4608 B: a record pair, a Q slot, a gate slot)
HALF = SL // 2  # bf16 of half a slot
PA = 3  # row panels whose pass 2 runs on A (the pair's balance)
AB_PP = 2 * 2 * 256 * (R - PA) + 4 * 64 * R + 64  # the A -> B object's P panels of row panels 0 .. PA - 1 (bytes in)
AB_BYTES = -(-(AB_PP + PA * 4 * 72) // 64) * 64  # the A -> B object: B's panels' P, f (bf16), alpha (fp32), the flag,
# A's P panels
PF_BYTES = 2 * 2 * 256 * PA  # A's pass 1 output of its pass-2 panels
O_BYTES = 2048 * 4  # a row panel's o
STACK = 2048


def applies(cfg: AttnConfig) -> bool:
    """Whether a model's attention fits the split design: a q head per column, two KV heads, causal."""
    return (cfg.head_dim == HD and cfg.num_key_value_heads == 2 and cfg.num_attention_heads == COLS
            and getattr(cfg, "causal", True) and not getattr(cfg, "moe", False))


def v_walk() -> TensorAccessPattern:
    """The V streams' memtile walk of an object ([record pair 2][d-block 16][144 B], two runs of 2304 B as the shim
    reads them) -> [d-block 16][record pair 2][144 B]: P V pops a d-block's four records in a row."""
    return stream_walk(SL, [(ND // 2, 72), (2, HALF), (72, 1)])


_SRC = templating.load("artlite/prefill/attn256_split.cc.jinja")


class SplitAttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``AttnPrepOperator(qg=QG)``). Runtime sequence: QS (the prep's, passes of 96), KV (one layer's cache), A (T x heads
    x 256 as bfp16 panels [T / 8][heads x 32]). Parameters ``QA_N`` = p0 / 32 + 1, ``QA_OFF`` (T's): the bf16 offset
    of the chunk's second group (``params``, as ``PrefillAttnOperator``'s)."""

    pfx = kernel_prefix("qas")

    def __init__(self, *, T: int, cfg: AttnConfig, head_bytes: int, ablate: tuple[str, ...] = (), stall: int = 26,
                 mem_events: tuple[int, ...] = ()):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_aqk",
        "ev_asoft", "ev_bsoft", "ev_bpv": an event window around the kernel, every core's counters programmed
        (``npu_regs.window_counter_writes``; counter 1 the event ``stall``, default the lock stall)."""
        assert applies(cfg)
        assert T % PASS == 0, T
        self.T, self.cfg, self.head_bytes, self.ablate, self.stall = T, cfg, head_bytes, tuple(ablate), stall
        self.mem_events = tuple(mem_events)  # (with "ev_": the memory module's counters, e.g. bank conflicts 77 .. 84)
        self.kvh, self.nph, self.H = 2, COLS // 2, COLS
        self.causal, self.qg = True, QG  # (as ``PrefillAttnOperator``'s, for the time model)
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.pfx, wa=Attr(WA), wb=Attr(WB), R=R, PA=PA, AB_PP=AB_PP, ND=ND, HD=HD,
                           OB=[WB[f"o{h}"] for h in range(4)],
                           O5=A_NORTH + A_L1["o5"], tri=f"{mask64(lambda q, k: k > q):016X}", ablate=self.ablate)

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (COLS * T // 8 * 2 * SL,), bfloat16),
                ArgSpec("in", (self.kvh * self.head_bytes // 2,), bfloat16),
                ArgSpec("out", (panel_elems(T * self.H * HD),), bfloat16)]

    def params(self, p0: int, n: int = 0) -> dict[str, int]:
        """(n: the chunk's valid tokens, as ``PrefillAttnOperator.params``'s: unused)"""
        assert p0 % 32 == 0
        return {QA_N: p0 // 32 + 1, f"{QA_OFF}{self.T}": GROUP // 2 * (p0 // 32 + 1)}

    def get_mlir_artifact(self) -> Module:
        p, T, H, nph, kvh = self.pfx, self.T, self.H, self.nph, self.kvh
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        h_ty = np.ndarray[(SL,), bf]  # (a stream's object: 4608 B)
        ab_ty = np.ndarray[(AB_BYTES // 4,), f32]
        o_ty = np.ndarray[(A_SET,), bf]  # (32 A panels)
        oj_ty = np.ndarray[(2 * A_SET,), bf]  # (the column's two B cores' panels, joined)
        wa_ty = np.ndarray[(WA["size"] // 4,), f32]
        mb_ty = np.ndarray[(2 * R * 64,), f32]
        o5_ty = np.ndarray[(2048,), f32]
        wb_ty = np.ndarray[(WB["size"] // 4,), f32]
        lp_ty = np.ndarray[(LP_BYTES // 4,), f32]
        k_ainit = Kernel(f"{p}_ainit", "kernel.o", [mb_ty, o5_ty])
        k_aqsel = Kernel(f"{p}_aqsel", "kernel.o", [h_ty, h_ty, wa_ty, np.int32, np.int32])
        k_aqk = [Kernel(f"{p}_aqk{i}", "kernel.o", [h_ty, wa_ty, mb_ty]) for i in range(2)]
        pf_ty = np.ndarray[(PF_BYTES // 2,), bf]
        k_asoft = Kernel(f"{p}_asoft", "kernel.o", [wa_ty, mb_ty, ab_ty, pf_ty, np.int32, np.int32, np.int32])
        k_binit = Kernel(f"{p}_binit", "kernel.o", [wb_ty, lp_ty, o5_ty])
        k_bsoft = Kernel(f"{p}_bsoft", "kernel.o", [ab_ty, wb_ty, lp_ty, o5_ty])
        k_bpv = Kernel(f"{p}_bpv", "kernel.o", [h_ty, wb_ty, lp_ty, o5_ty, np.int32])
        k_bfin = Kernel(f"{p}_bfin", "kernel.o", [h_ty, h_ty, wb_ty, lp_ty, o5_ty, np.int32, np.int32])
        k_bout = Kernel(f"{p}_bout", "kernel.o", [wb_ty, o5_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(f"{QA_OFF}{T}", np.int32)
        n_pass = T // PASS
        n_q = 2 * nph * R // 2  # (a pass's Q objects per K stream, and gate objects per V stream: 8 cores x 3)

        # streams per KV head h: kind 0 / 1 (K0, K1: the A cores), 2 / 3 (V0, V1: the B cores), shim and memtile of
        # column 4 h + kind
        names = ("qs_ka", "qs_kb", "qs_va", "qs_vb")
        l3, l1 = {}, {}
        for h in range(kvh):
            for kind in range(4):
                l3[h, kind] = ObjectFifo(h_ty, name=f"qs_s{h}{kind}", depth=2)
                l1[h, kind] = l3[h, kind].cons().forward(tile=Tile(nph * h + kind, 1), obj_type=h_ty, depth=2,
                                                         name=f"{names[kind]}{h}",
                                                         to_stream=v_walk() if kind >= 2 else None)
        ab = {(c, k): ObjectFifo(ab_ty, name=f"qs_ab{c}{k}", depth=2) for c in range(COLS) for k in range(2)}
        out3 = [ObjectFifo(oj_ty, name=f"qs_out{c}", depth=2) for c in range(COLS)]
        out1 = [out3[c].prod().join([A_SET * k for k in range(2)], tile=Tile(c, 1), depths=[1, 1], obj_types=[o_ty] * 2,
                                    names=[f"qs_ob{c}{k}" for k in range(2)]) for c in range(COLS)]

        def a_core(k0, k1, abp, ainit, aqsel, aqk0, aqk1, asoft, wa, mb, o5r, pf, n_p, k, me):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q):
                        x = k0.acquire(1)
                        y = k1.acquire(1)
                        aqsel(x, y, wa, i, me)
                        k0.release(1)
                        k1.release(1)
                    ainit(mb, o5r)
                    n = n_p.read() + b * QG + (QG - 1)
                    for g in range32(n):
                        x = k0.acquire(1)
                        aqk0(x, wa, mb)
                        k0.release(1)
                        y = k1.acquire(1)
                        aqk1(y, wa, mb)
                        k1.release(1)
                        o = abp.acquire(1)
                        asoft(wa, mb, o, pf, n - 1 - g, k, g)
                        abp.release(1)

        def b_core(v0, v1, abc, outp, binit, bsoft, bpv, bfin, bout, wb, lp, o4, n_p, me):
            for _ in forever_():
                for b in range_(n_pass):
                    binit(wb, lp, o4)
                    n = n_p.read() + b * QG + (QG - 1)
                    for g in range32(n):
                        x = abc.acquire(1)
                        bsoft(x, wb, lp, o4)
                        abc.release(1)
                        x = v0.acquire(1)  # (each half released after its P V: the other's refill overlaps)
                        bpv(x, wb, lp, o4, 0)
                        v0.release(1)
                        y = v1.acquire(1)
                        bpv(y, wb, lp, o4, 1)
                        v1.release(1)
                    for i in range_(n_q):  # (the gate objects; the core's own: its panels' outputs into their o)
                        x = v0.acquire(1)
                        y = v1.acquire(1)
                        bfin(x, y, wb, lp, o4, i, me)
                        v0.release(1)
                        v1.release(1)
                    for h in range(R):
                        ob = outp.acquire(1)
                        bout(wb, o4, h, ob)
                        outp.release(1)

        workers = []
        for c in range(COLS):
            h = c // nph
            for k in range(2):
                me = 2 * (c % nph) + k  # (the pair's index among the KV head's 8)
                workers.append(Worker(a_core, [l1[h, 0].cons(depth=1), l1[h, 1].cons(depth=1), ab[c, k].prod(),
                                               k_ainit, k_aqsel, k_aqk[0], k_aqk[1], k_asoft,
                                               Buffer(wa_ty, name=f"qs_wa{c}{k}"), Buffer(mb_ty, name=f"qs_mb{c}{k}"),
                                               Buffer(o5_ty, name=f"qs_o5{c}{k}"), Buffer(pf_ty, name=f"qs_pf{c}{k}"),
                                               p_n, k, me],
                                      tile=Tile(c, 3 + 2 * k), stack_size=STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
                workers.append(Worker(b_core, [l1[h, 2].cons(depth=1), l1[h, 3].cons(depth=1), ab[c, k].cons(),
                                               out1[c][k].prod(), k_binit, k_bsoft, k_bpv, k_bfin, k_bout,
                                               Buffer(wb_ty, name=f"qs_wb{c}{k}"), Buffer(lp_ty, name=f"qs_lp{c}{k}"),
                                               Buffer(o5_ty, name=f"qs_o4{c}{k}"), p_n, me],
                                      tile=Tile(c, 2 + 2 * k), stack_size=STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = (arg_elems(self, i) for i in range(3))
        hb = self.head_bytes // 2
        blk = 4 * QG * 2 * SL  # (bf16 of a column's slots in a pass: 12 token groups x [Q | gate])
        core_st = blk // 2  # (a pair's six token groups)

        def seq(QS, KV, A, hs, oh):
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, stalls)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in window_counter_writes(stall=self.stall, memory_events=self.mem_events):
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):  # (per pass the column's 12 token groups' panels: [panel i][pair k], a k's 6 H A_SET apart)
                for b0 in range(0, n_pass, MAX_BD_ITER):
                    oh[c].drain(A, TensorAccessPattern((n_a,), c * A_SET + b0 * 12 * H * A_SET,
                                                       [min(MAX_BD_ITER, n_pass - b0), R, 2, A_SET],
                                                       [12 * H * A_SET, H * A_SET, R * H * A_SET, 1]), group=tg, wait=True)
            inflight = []
            for b in range(n_pass):
                tp = TaskGroup()
                more = QG * b + QG - 1  # (the chunk's groups after its first through the pass's)
                for (h, kind), f in zip(((h, kind) for h in range(kvh) for kind in range(4)), hs, strict=True):
                    q0 = (b * COLS + nph * h) * blk  # (the pass's slots of the head's columns)
                    k0 = h * hb + (kind % 2) * (SL if kind < 2 else HALF) + (OBJ_ELEMS if kind >= 2 else 0)
                    if kind < 2:  # K0 / K1: half kind of every Q slot (24 objects: two panels each), then record pair kind
                        parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0 + kind * HALF, [1, 2 * nph, R, HALF],
                                                                   [0, core_st, 2 * SL, 1])),
                                 ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 2, 2, HALF], [0, GROUP // 2, HALF, 1]),
                                           length_parameter=p_n, length_unit=SL)]
                        if more:
                            parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, more, 2, HALF],
                                                                           [0, GROUP // 2, HALF, 1]), offset_parameter=p_off))
                    else:  # V0 / V1: d-blocks 16 (kind - 2) .. of both record pairs, then the gate slots' halves
                        parts = [ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 2, 2, HALF], [0, GROUP // 2, SL, 1]),
                                           length_parameter=p_n, length_unit=SL)]
                        if more:
                            parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, more, 2, HALF],
                                                                           [0, GROUP // 2, SL, 1]), offset_parameter=p_off))
                        parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + SL + (kind - 2) * HALF,
                                                                       [1, 2 * nph, R, HALF], [0, core_st, 2 * SL, 1])))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) >= 2:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [l3[h, kind].prod(tile=Tile(nph * h + kind, 0)) for h in range(kvh) for kind in range(4)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(out3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for key, v in FIFO_L1.items() if name.startswith(key)), None),
                       buffer=lambda name: next((v for key, v in BUF_L1.items() if name.startswith(key)), None))
        return module


# A's L1 (byte addresses): region 0 the stack, K0's and K1's objects, m / b; 1 the A -> B objects (B reads them there),
# pf;
# 2 q; 3 the scores, o5 (B's panel 5, at B's north window)
A_L1 = {"k0": STACK, "k1": STACK + SLOT, "mb": STACK + 2 * SLOT, "ab0": REGION, "ab1": REGION + AB_BYTES,
        "pf": REGION + 2 * AB_BYTES, "wa": 2 * REGION, "o5": 3 * REGION + R * 1024}
WA = {"q": 0, "sc": REGION, "size": REGION + R * 1024}
A_NORTH = 0x60000  # (a core's view of its north neighbour's memory)
# B's L1: region 0 the stack, V0's and V1's objects, the output object; 1 o4, l and the P panels;
# 2 and 3 o0 .. o3
LP_BYTES = R * 256 + R * 4 * 72
B_L1 = {"v0": STACK, "v1": STACK + SLOT, "out": STACK + 2 * SLOT, "o4": REGION, "lp": REGION + O_BYTES,
        "wb": 2 * REGION}
WB = {"o0": 0, "o2": O_BYTES, "o1": REGION, "o3": REGION + O_BYTES, "size": 2 * REGION, "l": 0, "pw": R * 256}
FIFO_L1 = {"qs_ka": [A_L1["k0"]], "qs_kb": [A_L1["k1"]], "qs_ab": [A_L1["ab0"], A_L1["ab1"]], "qs_va": [B_L1["v0"]],
           "qs_vb": [B_L1["v1"]], "qs_ob": [B_L1["out"]]}
BUF_L1 = {"qs_wa": A_L1["wa"], "qs_mb": A_L1["mb"], "qs_o5": A_L1["o5"], "qs_pf": A_L1["pf"], "qs_wb": B_L1["wb"], "qs_lp": B_L1["lp"],
          "qs_o4": B_L1["o4"]}
