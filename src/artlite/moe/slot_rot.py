"""An MoE layer's experts' h rotated in the prefill (PARO experts, ``artlite.paro``): the down projections take their
input rotated by the downs' shared rotation (``decode.rotated``; the decode's: ``hrot``), and the experts' gate / up
GEMM (``expert_gemm``, its SiLU epilogue) writes h as bfp16 panels into its chunk slots (H: per slot [token group
CHUNK / 8][K / 8 panels]). ``SlotRotOperator`` rotates each chunk's items in place before the downs read them: an item is
``TOK`` tokens (4 token groups) of the chunk's valid rows (the last item's groups past them rotated too: the slot's, never
read), ``prefill.rotate``'s kernel at its full width (the tokens in the vector lanes: the panels in, u = x w' fp32, the
rounds, the panels out). Per column its 4 cores (rows 2 .. 5) take one rotation group each (K = 512: 4 groups of 128):
an item's fill walks it core-major ([core][token group][16 panels]), the column's memtile splits it into the cores' parts
and joins their outputs, which the drain writes back the same way. The rotation objects ([group][``OBJ``]: one object
of an item's size) come first on the same channel: core m keeps group m's.

The work is data-dependent (``moe.plan``: a chunk's valid rows): a column's control stream (``ctl``) pushes per chunk the
drain of its items (a BD of set k % 2 repeated per item, each releasing lock ``LOCKS`` k % 2; the block waits for chunk
k - 2's by that lock, the last chunk's with a token) and their fill (on the packet-switched channel); the cores run the
column's items (the scratchpad parameter ``expert_gemm.N_PARAM`` c's field from bit ``N_ITEMS``: ``items``)."""

import functools

import numpy as np
from aie.dialects._aiex_ops_gen import npu_sync as _npu_sync
from aie.dialects.aiex import _as_i32, npu_maskwrite32, npu_write32
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Packet,
    Port,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Transport,
    Worker,
    sync_parameters,
)
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import txn
from ..operators import templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, range32
from ..operators.kernel_source import kernel_prefix
from ..prefill.rotate import GROUP, OBJ, row_floats
from . import ctl
from .expert_gemm import CHUNK, N_ITEMS, N_PARAM

TOK = row_floats(False)  # tokens of an item (4 token groups: the rotation kernel's full width)
GROUPS = 4  # rotation groups of h (K 512), a core each
BLOCK_BDS, FILL_BD = (5, 14), 6
DRAIN_BDS, LOCKS = (4, 8), (2, 3)  # (per chunk: its drain, two sets, a shim lock each)

_SRC = templating.load("artlite/prefill/rotate.cc.jinja")


class SlotRotOperator(ArtcOperator):
    """The experts' h of an MoE layer rotated in place in its chunk slots (module docstring): K the experts' width
    (4 groups), ``cap`` chunks per column at most. Arguments: the control streams (per column ``cb_words``), the
    rotation objects (``paro.TanForm.pair_objects(TOK)``), H in and H out (one buffer: in place)."""

    p = kernel_prefix("msr")
    STACK = 1024

    def __init__(self, *, K: int, cap: int, rounds: int = 8):
        assert K == GROUPS * GROUP and 512 * rounds + 4 * GROUP == OBJ, (K, rounds)
        self.K, self.cap, self.rounds = K, cap, rounds
        assert self.item == GROUPS * OBJ and CHUNK % TOK == 0, "the rotation objects: one item-sized object"
        super().__init__()

    def get_kernel_src(self) -> str:
        p = self.p
        return _SRC.render(p=p, R=self.rounds, TOK=TOK, gu=False, xb=False, gurows=False, rows_out=False, hilo=False) + f"""
extern "C" {{
// a core's group of an item (its words: the design's FIFOs are int32)
void {p}_rotw(const int32_t *__restrict x, const int32_t *__restrict ro, int32_t *__restrict o, float *__restrict ua,
              float *__restrict ub)
{{
    {p}_rot((const uint8_t *)x, (const uint8_t *)ro, (uint8_t *)o, ua, ub);
}}

// the core's rotation object (the first object's part) kept
void {p}_ro(const int32_t *__restrict x, int32_t *__restrict ro)
{{
    al::copy<{OBJ // 4}, 16>(ro, x);
}}
}}
"""

    @property
    def item(self) -> int:
        """Bytes of an item's panels (``TOK`` tokens of K)."""
        return TOK * self.K * 9 // 8

    @property
    def slot(self) -> int:
        """bf16 units of a chunk slot (``expert_gemm.ExpertGemmOperator.c_slot`` of "gu")."""
        return CHUNK * self.K * 9 // 16

    @property
    def b0_words(self) -> int:
        """Block 0: one object of the packet-switched channel (an item)."""
        return self.item // 4

    @functools.cached_property
    def cb_words(self) -> int:
        """32-bit words of a column's control stream at most."""
        per = 2 * 2 * 6 + 2 * 3 + ctl.NEXT_WORDS
        return -(-(self.b0_words + self.cap * per) // 16) * 16

    @staticmethod
    def items_of(sizes: np.ndarray) -> np.ndarray:
        """Items of chunks of ``sizes`` valid rows."""
        return -(-np.asarray(sizes, np.int64) // TOK)

    def items(self, sizes: list[np.ndarray]) -> list[int]:
        """Per column the items of its chunks (sizes[c]: their valid rows): ``N_PARAM`` c's field from bit ``N_ITEMS``."""
        return [int(self.items_of(s).sum()) for s in sizes]

    @functools.cached_property
    def _templates(self) -> list[ctl.Template]:
        """Per (items - 1, drain set, the last chunk's) a chunk's block (base 0: its slot's address): its drain's BD whole
        and push, then its fill's (the channel's packets), each repeated per item: [core][token group][16 panels]."""
        q = self.item // GROUPS // (TOK // 8) // 2  # (bf16 units of a core's part of a token group: 16 panels)
        out = []
        for items in range(1, CHUNK // TOK + 1):
            tap = TensorAccessPattern((1, 2), 0, [items, GROUPS, TOK // 8, q], [self.item // 2, q, GROUPS * q, 1])
            for parity in (0, 1):
                for last in (False, True):
                    words: list[int] = []
                    slots = [(len(words) + 3, len(words) + 4, 0, 0)]
                    words += ctl.bd_packets(DRAIN_BDS[parity], ctl.bd_words(0, tap, release=(LOCKS[parity], 1)))
                    words += ctl.push(txn.S2MM, 0, DRAIN_BDS[parity], repeat=items, token=last)
                    slots.append((len(words) + 3, len(words) + 4, 0, 0))
                    words += ctl.bd_packets(FILL_BD, ctl.bd_words(0, tap, packet=ctl.XID))
                    words += ctl.push(txn.MM2S, 1, FILL_BD, repeat=items)
                    out.append(ctl.template(words, slots))
        return out

    def streams(self, sizes: list[np.ndarray], *, cb: list[int], h: list[int], out: list[np.ndarray]) -> list[np.ndarray]:
        """Per column the control stream (uint32 words at device address cb[c], into out[c]) of its chunks: sizes[c]
        their valid rows, its H slots from device address h[c]: per chunk its items' drain and fill."""
        res = []
        for c in range(len(sizes)):
            n = len(sizes[c])
            assert 1 <= n <= self.cap
            k = np.arange(n)
            items = self.items_of(sizes[c])
            assert (items >= 1).all()
            bases = (h[c] + 2 * self.slot * k)[:, None]
            blocks = (4 * (items - 1) + 2 * (k % 2) + (k == n - 1)).astype(np.int32)
            locks = np.array(LOCKS, np.int64)[k % 2]
            # (chunk k - 2's drain released its set's lock once per item; chunks 0, 1 the initial 1)
            val = np.where(k >= 2, np.roll(items, 2), 1)
            w = ctl.assemble(out[c], self._templates, blocks, bases, base=cb[c], b0_words=self.b0_words, bds=BLOCK_BDS,
                             acquire=(locks, val))
            assert w <= self.cb_words, (w, self.cb_words)
            res.append(out[c][:w])
        return res

    def get_arg_spec(self) -> list[ArgSpec]:
        n_h = COLS * self.cap * self.slot
        return [ArgSpec("in", (COLS * self.cb_words * 2,), bfloat16), ArgSpec("in", (GROUPS * OBJ // 2,), bfloat16),
                ArgSpec("in", (n_h,), bfloat16), ArgSpec("out", (n_h,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, i32, f32 = self.p, np.dtype[np.int32], np.dtype[np.float32]
        iw = self.item // 4  # (an item's words)
        pw = iw // GROUPS  # (a core's: its group's 16 panels of the item's token groups)
        it_ty, part_ty = np.ndarray[(iw,), i32], np.ndarray[(pw,), i32]
        u_ty = np.ndarray[(GROUP * TOK,), f32]
        k_rot = Kernel(f"{p}_rotw", "kernel.o", [part_ty, part_ty, part_ty, u_ty, u_ty])
        k_ro = Kernel(f"{p}_ro", "kernel.o", [part_ty, part_ty])
        pg = [ScratchpadParameter(f"{N_PARAM}{c}", np.int32) for c in range(COLS)]
        xf = [ObjectFifo(it_ty, name=f"msx{c}", depth=2, transport=Transport.dma(Packet(ctl.XID))) for c in range(COLS)]
        of = [ObjectFifo(it_ty, name=f"mso{c}", depth=2) for c in range(COLS)]
        offs = [pw * m for m in range(GROUPS)]
        xs = [xf[c].cons().split(offs, tile=Tile(c, 1), obj_types=[part_ty] * GROUPS,
                                 names=[f"msx{c}_{m}" for m in range(GROUPS)], depths=[2] * GROUPS) for c in range(COLS)]
        os_ = [of[c].prod().join(offs, tile=Tile(c, 1), obj_types=[part_ty] * GROUPS,
                                 names=[f"mso{c}_{m}" for m in range(GROUPS)], depths=[2] * GROUPS) for c in range(COLS)]

        def core(x_f, o_f, rot, kro, ro, ua, ub, prm):
            x = x_f.acquire(1)  # (the first object: the rotation objects)
            kro(x, ro)
            x_f.release(1)
            n = prm.read() // (1 << N_ITEMS)  # (the column's items: ``N_PARAM``'s)
            for _ in range32(n):
                x, o = x_f.acquire(1), o_f.acquire(1)
                rot(x, ro, o, ua, ub)
                o_f.release(1)
                x_f.release(1)

        workers = [Worker(core, [xs[c][m].cons(), os_[c][m].prod(), k_rot, k_ro, Buffer(part_ty, name=f"ms_ro{c}_{m}"),
                                 Buffer(u_ty, name=f"ms_ua{c}_{m}"), Buffer(u_ty, name=f"ms_ub{c}_{m}"), pg[c]],
                          tile=Tile(c, 2 + m), stack_size=self.STACK, while_true=False, dynamic_objfifo_lowering=True)
                   for c in range(COLS) for m in range(GROUPS)]
        n_cb, n_ro, n_h = COLS * self.cb_words * 2, GROUPS * OBJ // 2, COLS * self.cap * self.slot
        bf = np.dtype[bfloat16]

        def seq(CB, RO, H, HO, xh, oh):
            sync_parameters()
            for c in range(COLS):
                for reg, val in ctl.ROUTE:
                    npu_write32(reg, val, column=c, row=0)
                for lk in LOCKS:  # (the drains' locks free, their tokens to the controller)
                    npu_write32(txn.SHIM_LOCK0 + 16 * lk, 1, column=c, row=0)
                npu_maskwrite32(txn.CTRL[txn.S2MM][0], txn.CONTROLLER_ID << 8, 0x1F00, column=c, row=0)
            tg = TaskGroup()
            for c in range(COLS):  # (the rotation objects, then block 0 of the control stream: one channel, in order)
                xh[c].fill(RO, TensorAccessPattern((n_ro,), 0, [1, 1, n_ro // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                xh[c].fill(CB, TensorAccessPattern((n_cb,), 2 * c * self.cb_words, [1, 1, 1, 2 * self.b0_words],
                                                   [0, 0, 0, 1]), packet=(0, ctl.CID), group=tg)
            tg.finish()
            for c in range(COLS):  # (the last chunk's drain: its token)
                _npu_sync(_as_i32(c), _as_i32(0), _as_i32(txn.S2MM), _as_i32(0), _as_i32(1), _as_i32(1))

        rt = Runtime(seq, [np.ndarray[(n_cb,), bf], np.ndarray[(n_ro,), bf], np.ndarray[(n_h,), bf],
                           np.ndarray[(n_h,), bf],
                           [xf[c].prod(tile=Tile(c, 0), port=Port.dma(1)) for c in range(COLS)],
                           [of[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
