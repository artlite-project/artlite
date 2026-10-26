"""The vision overlay's static design: compiled once, run by host-built transactions (``seq``) for every op of a
network (all of the UNet's levels, the VAE decoder's, the ViT's).

Array roles (column c, core row r):
- column c holds a token tile's activations in its memtile: every channel of its tokens;
- core (c, r) computes channel slice r of every layer for column c's tokens;
- row r's B stream (shim ``B_SHIM[r]`` MM2S 1 -> cores (0..7, r) S2MM 1, broadcast): commands, the weights of the
  row's channel slice, statistics, attention keys / values: each weight byte is read once per token tile;
- A: memtile c MM2S r -> core (c, r) S2MM 0 (the memtile tensors, ``A_OBJ`` objects);
- O: core (c, r) MM2S 0 -> memtile c S2MM r (``OCH`` chunks);
- DDR: shim c MM2S 0 -> memtile c S2MM 4 (fills), memtile c MM2S 4 -> shim c S2MM 0 (drains);
- the gate: memtile c MM2S 5 -> its own S2MM 5 (a static landing chain): the op handover's lock task (``seq``);
- pages: core (c, r) MM2S 1 -> its own TileControl (``paging.loop_flow``): a static chain over the two B slots streams
  a page that came in the B stream into the core's program memory (``core``'s roles).
Memtile channels and shim channels run host tasks only; the cores' channels are static chains (``layout``).

Core L1 (pinned, ``L1``): the stack, the activation table, the C tile (m tokens x 80 channels, 8 x 8 sub-blocks
[rb][cb][64]: fp32 at m = 64, its halves in regions 0 and 1, the MAC alternating between them so that a block's C
stores and the next block's C loads never meet in one region), the output ring, the command workspace (header + the
op's biases), and regions 2, 3 (``R23``): each A slot's halves (one per region: two chained BDs), B slot k and
scratch buffer k in region 2 + k. An m-64 A object's k-tile j is in region 2 + j; with an op's command in B slot 0 (the
host's NOP commands, ``seq``) its k-tile i is in B slot (i + 1) % 2, so the MAC never reads A and B from one region.

The cores' program (``core``): a C++ main reading commands from the B stream (lock intrinsics on the pinned core locks:
0 / 1 A free / full, 2 / 3 B, 4 / 5 the output ring; 6 between an A slot's two BDs; 7 / 8 a page's slots sent / done,
9 the pager's wake). What does not fit 16 KB of program memory together is split into roles (``core.ROLES``), pages of
code the cores load themselves (``paging``): the host puts a role's page into a row's B stream before the first op that
needs it there (``seq``), so every row runs every role.
"""

from typing import ClassVar

import numpy as np
import torch
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.iron import (
    Acquire,
    Bd,
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
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import txn
from ..operators import paging
from ..operators.base import ArgDirection, ArgSpec, ArtcOperator
from ..operators.design import COLS, ROWS
from . import core

KT = 64  # k of a k-tile
A_OBJ = 128 * KT * 9 // 8  # 9216 B: an A object (a k-tile of 128 tokens, two of 64, eight of 16)
Q = 80 * KT * 9 // 8  # 5760 B: a B slot (a k-tile of 80 output channels)
OCH, NO = 1440, 2  # output chunks: bfp16 tiles fill whole chunks, bf16 tiles are padded to them
OSL = 1472  # (an output slot: OCH rounded up to 64 B, the vector stores' alignment)
C_BYTES = 128 * 80 * 2
WS = 64 * 4 + 16 * 80 * 2 + 96 * 4  # the command header, up to 16 passes' biases, the attention's key masks
SCR = 1408  # two scratch buffers (statistics records, GEGLU's): the rest of regions 2 and 3
TAB = 3200  # an activation table (two copies of kmath.ActTable.table_words)
CBUF = 4 * 1152  # a conv's lane-shifted groups (``core``: over the table: the host reloads it after a conv)
TABB = max(TAB, CBUF)  # (the buffer of both: nothing else may be placed in a conv's part)
STACK = 1280
HEAP = 510 * 1024  # memtile bytes for tensors
GATE_SCRATCH = HEAP  # (memtile: the gate tasks' landing words)
B_SHIM = (0, 2, 4, 6)
L_SEND, L_DONE, L_WAKE = 7, 8, 9  # (core locks of the pages: ``paging``)
# pinned L1 offsets (16 KB regions: 0 stack + tab + C's first half, 1 C's second half + out + ws, 2 and 3 the stream
# buffers ``R23``: per region the A slots' halves, a B slot, a scratch buffer)
L1 = {"tab": STACK, "c": 0x4000 - C_BYTES // 2, "out": 0x4000 + C_BYTES // 2, "ws": 0x4000 + C_BYTES // 2 + NO * OSL,
      "r23": 0x8000, "ext": 0x4000 - C_BYTES // 2 - 64}  # (ext: the pager's state words, ``core``)
# (byte offsets in regions 2, 3: A slot k's first half at k A_OBJ / 2, its second half 16 KB on; B slot k and scratch
# buffer k in region 2 + k)
R23 = {"a0": 0, "a1": A_OBJ // 2, "b0": A_OBJ, "b1": 0x4000 + A_OBJ, "s0": A_OBJ + Q, "s1": 0x4000 + A_OBJ + Q}
A_HI = 0x4000 - A_OBJ // 2  # (the gap between an A slot's halves: object byte o >= A_OBJ / 2 at o + A_HI)


def a_byte(o: int) -> int:
    """Byte o of an A object in its slot (``R23``: the halves 16 KB apart)."""
    return o + (A_HI if o >= A_OBJ // 2 else 0)
assert L1["tab"] + TABB <= L1["ext"] and L1["ws"] + WS <= 0x8000 and R23["s1"] + SCR <= 0x8000
assert all(v % 64 == 0 for v in R23.values())
assert all(v % 64 == 0 for v in L1.values())

# memtile BDs: per channel number 0..4 a pool of 8 BDs of its parity (even channels 0-23, odd 24-47) that its two directions
# share (the sequencer takes the least recently used); odd 40: the gate's static landing chain; 41-47 the gate tasks (one
# BD per op, rotating)
GATE_BDS = tuple(range(41, 48))


def mem_bd_pool(ch: int) -> tuple[int, ...]:
    """The BDs of memtile channels S2MM / MM2S ``ch`` (0..4)."""
    assert 0 <= ch < 5
    base = 24 * (ch % 2) + 8 * (ch // 2)
    return tuple(range(base, base + 8))


class VisionDesign(ArtcOperator):
    """The static design. Its runtime sequence is a probe (never run: the transactions are built on the host) whose
    tasks make aiecc route the memtiles' task-complete tokens; ``check_endpoints`` checks the shim channels."""

    ENDPOINTS: ClassVar[dict[str, tuple[int, int]]] = {"D": (txn.MM2S, 0), "B": (txn.MM2S, 1), "O": (txn.S2MM, 0)}

    def __init__(self, *, win: str = ""):
        """win: a region of the core program between event0 / event1 (counter windows: ``core.WINS``)."""
        super().__init__()
        self.win = win

    def get_kernel_src(self) -> str:
        return core.source(win=self.win)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec(ArgDirection.IN, (4096,), torch.bfloat16), ArgSpec(ArgDirection.OUT, (4096,), torch.bfloat16)]

    def get_mlir_artifact(self):
        i32 = np.dtype[np.int32]

        def ty(n: int) -> type[np.ndarray]:
            return np.ndarray[(n // 4,), i32]
        sig = [ty(0x8000), ty(NO * OSL), ty(C_BYTES), ty(WS), ty(TABB), ty(16)]
        k_main = Kernel(f"{core.P}_main", "kernel.o", sig)
        mems = [Tile(c, 1) for c in range(COLS)]
        shims = [Tile(c, 0) for c in range(COLS)]
        cores = [[Tile(c, r + 2) for r in range(ROWS)] for c in range(COLS)]
        heap = [Buffer(type=ty(HEAP), tile=mems[c], name=f"sd15_heap_{c}", address=0) for c in range(COLS)]
        d_in = [Flow(shims[c], mems[c], src_channel=0, dst_channel=4, name=f"sd15_din{c}") for c in range(COLS)]
        d_out = [Flow(mems[c], shims[c], src_channel=4, dst_channel=0, name=f"sd15_dout{c}") for c in range(COLS)]
        b = [Flow(shims[B_SHIM[r]], [cores[c][r] for c in range(COLS)], src_channel=1, name=f"sd15_b{r}")
             for r in range(ROWS)]
        a = {(c, r): Flow(mems[c], cores[c][r], src_channel=r, dst_channel=0, name=f"sd15_a{c}_{r}")
             for c in range(COLS) for r in range(ROWS)}
        o = {(c, r): Flow(cores[c][r], mems[c], src_channel=0, dst_channel=r, name=f"sd15_o{c}_{r}")
             for c in range(COLS) for r in range(ROWS)}
        gate = [Flow(mems[c], mems[c], src_channel=5, dst_channel=5, name=f"sd15_gate{c}") for c in range(COLS)]
        loops = [paging.loop_flow(cores[c][r]) for c in range(COLS) for r in range(ROWS)]
        flows = d_in + d_out + b + list(a.values()) + list(o.values()) + gate + loops
        dmas, workers = [], []
        for c in range(COLS):  # (the gate tasks' words land in a scratch buffer: a static chain, no locks)
            gs = Buffer(type=ty(64), tile=mems[c], name=f"sd15_gs_{c}", address=GATE_SCRATCH)
            dmas.append(TileDma(mems[c], [DmaChannel(DMAChannelDir.S2MM, 5, [Bd(gs, tap=TAP((16,), 0, [4], [1]),
                                                                                bd_id=40)])]))
        for c in range(COLS):
            for r in range(ROWS):
                t = cores[c][r]
                bufs = {n: Buffer(type=ty(sz), tile=t, name=f"sd15_{n}_{c}_{r}", address=L1[n])
                        for n, sz in (("r23", 0x8000), ("out", NO * OSL), ("c", C_BYTES), ("ws", WS), ("tab", TABB),
                                      ("ext", 16))}
                r23 = bufs["r23"]

                def part(off: int, n: int) -> TAP:
                    return TAP((0x2000,), off // 4, [n // 4], [1])
                af, aa = Lock(t, lock_id=0, init=2, name=f"sd15_af_{c}_{r}"), Lock(t, lock_id=1, init=0, name=f"sd15_aa_{c}_{r}")
                bf, bb = Lock(t, lock_id=2, init=2, name=f"sd15_bf_{c}_{r}"), Lock(t, lock_id=3, init=0, name=f"sd15_bb_{c}_{r}")
                of, oo = Lock(t, lock_id=4, init=NO, name=f"sd15_of_{c}_{r}"), Lock(t, lock_id=5, init=0, name=f"sd15_oo_{c}_{r}")
                ah = Lock(t, lock_id=6, init=0, name=f"sd15_ah_{c}_{r}")  # (an A slot's first half written)
                ps = Lock(t, lock_id=L_SEND, init=0, name=f"sd15_ps_{c}_{r}")
                pd = Lock(t, lock_id=L_DONE, init=0, name=f"sd15_pd_{c}_{r}")
                Lock(t, lock_id=L_WAKE, init=0, name=f"sd15_pw_{c}_{r}")
                dmas.append(TileDma(t, [
                    DmaChannel(DMAChannelDir.S2MM, a[c, r].endpoint(t),
                               [Bd(r23, tap=part(R23[n] + h * 0x4000, A_OBJ // 2), acquires=[Acquire(ah if h else af)],
                                   releases=[Release(aa if h else ah)]) for n in ("a0", "a1") for h in range(2)]),
                    DmaChannel(DMAChannelDir.S2MM, b[r].endpoint(t),
                               [Bd(r23, tap=part(R23[n], Q), acquires=[Acquire(bf)], releases=[Release(bb)]) for n in ("b0", "b1")]),
                    DmaChannel(DMAChannelDir.MM2S, o[c, r].endpoint(t),
                               [Bd(bufs["out"], tap=TAP((NO * OSL // 4,), k * OSL // 4, [OCH // 4], [1]),
                                   acquires=[Acquire(oo)], releases=[Release(of)]) for k in range(NO)]),
                    DmaChannel(DMAChannelDir.MM2S, 1,  # (a page's slots, held by the core: ``paging.loop_flow``)
                               [Bd(r23, tap=part(R23[n], Q), acquires=[Acquire(ps)], releases=[Release(pd)]) for n in ("b0", "b1")]),
                ]))

                def core_fn(rb, ou, cb, w, tb, ex, km):
                    km(rb, ou, cb, w, tb, ex)
                workers.append(Worker(core_fn, [*(bufs[n] for n in ("r23", "out", "c", "ws", "tab", "ext")),
                                                k_main], tile=t, stack_size=STACK, while_true=False))
        ty_arg = np.ndarray[(4096,), np.dtype[bfloat16]]

        def seq(X, Y):  # (the probe: awaited tasks on the memtile channels the transactions await; never run)
            tasks = []
            for c in range(COLS):
                eps = [d_in[c].endpoint(mems[c]), d_out[c].endpoint(mems[c])] + [o[c, r].endpoint(mems[c])
                                                                                  for r in range(ROWS)]
                for ep in eps:
                    t = ep.task(Bd(heap[c], tap=TAP((HEAP // 4,), 0, [256], [1])), wait=True)
                    t.start()
                    tasks.append(t)
                d_in[c].fill(X, tap=TAP((4096,), 0, [1, 1, 1, 512], [0, 0, 0, 1]))
                d_out[c].drain(Y, tap=TAP((4096,), 0, [1, 1, 1, 512], [0, 0, 0, 1]), wait=True)
            for r in range(ROWS):
                b[r].fill(X, tap=TAP((4096,), 0, [1, 1, 1, 512], [0, 0, 0, 1]))
            for t in tasks:
                t.await_()
                t.free()
        rt = Runtime(seq, [ty_arg, ty_arg])
        for f in flows:
            rt.add_flow(f)
        for d in dmas:
            rt.add_tile_dma(d)
        return Program(NPU2(), rt, workers=workers).resolve_program()

    def check_endpoints(self, insts: np.ndarray) -> None:
        """The probe's memtile token controllers are ``txn.MEM_CONTROLLER_ID`` (the transactions assume it)."""
        for name, f in txn.decode(insts):
            if name == "maskwrite32" and f["mask"] in (0x1F00, 0xFF00) and (f["addr"] >> 20) & 31 == 1:
                assert (f["value"] >> 8) & 0xFF == txn.MEM_CONTROLLER_ID, f
