"""NPU transactions (the ``insts.bin`` format) built on the host.

aiecc lowers an operator's runtime sequence to a transaction: a list of register
writes and waits that the NPU firmware executes for one dispatch. A design whose
cores interpret their inputs (the Gemma decode overlay) is sequenced at run time
instead: the host writes the transaction of every dispatch itself - which shim DMA
reads and writes to issue, in which order, how long - so one compiled design runs
any number of phases, tokens or variants without a recompile or a second hardware
context.

:class:`Txn` encodes the operations (word layout: mlir-aie ``aie/Runtime/TxnEncoding.h``).
The shim DMA helpers write a buffer descriptor with the final device address (no
firmware address patch) and push it on a channel's task queue; ``sync`` waits for
the task-complete tokens of channels whose tasks were pushed with ``token=True``.

Rules of the shim DMA (AM029, NOC module):
- 16 BDs per shim tile, shared by its 2 MM2S (DDR -> array) and 2 S2MM channels; a
  BD may be rewritten once the task that used it is complete;
- a channel queues at most 4 tasks (a push on a full queue blocks the firmware);
- a BD's AxQoS field must stay 0 (a nonzero value hangs the NPU until reboot);
- a BD may walk DDR in three dimensions (``dims``): gathering a stream from strips
  of several buffers (MM2S) or scattering one over them (S2MM).

Memory-tile DMA (AM029 MEMORY_TILE_MODULE; the words aiecc writes for its runtime tasks):
- 48 BDs of 8 words at 0xA0000 + 32 bd, by channel parity (even channels BDs 0-23, odd 24-47; a next-BD id is
  absolute); 6 S2MM and 6 MM2S channels, control at 0xA0600 + 8 c (S2MM) / 0xA0630 + 8 c (MM2S), the start queue at
  + 4 (``mem_push``: BD | (repeat - 1) << 16 | token << 31);
- a BD's base is a word address in the tile's 1.5 MB view: the west neighbour's memory from 0, its own from
  ``MEM_OWN``, the east neighbour's from ``MEM_EAST`` (channels 0-3 only); lock ids likewise: west 0-63, own 64-127,
  east 128-191 (``mem_lock``); 4 dimensions (d0..d2 with wraps, d3 a stride) and an iteration;
- a BD's lock acquire is "greater or equal" (subtracting the value) or "equal"; a release needs no acquire (aiecc's
  verifier wants both or neither; the hardware does not);
- an S2MM channel ignores TLAST by default: a BD's length decides, whatever the sender's BD sizes;
- every memory tile's task-complete tokens go to controller ``MEM_CONTROLLER_ID`` (as aiecc routes them).
"""

import numpy as np

DDR_AIE_ADDR_OFFSET = 0x80000000  # device address of a host buffer = its paddr + this

# opcodes
OP_WRITE, OP_BLOCKWRITE, OP_MASKWRITE, OP_MASKPOLL, OP_TCT, OP_DDR_PATCH = 0, 1, 3, 4, 0x80, 0x81
OP_LOAD_PDI, OP_CREATE_SCRATCHPAD = 8, 10
# words of the fixed-size operations (a blockwrite and the custom operations, 0x80 on, carry their byte size)
_OP_WORDS = {OP_WRITE: 6, OP_MASKWRITE: 7, OP_MASKPOLL: 7, 5: 1, 6: 1, 7: 7, OP_LOAD_PDI: 4, OP_CREATE_SCRATCHPAD: 4,
             12: 3}

# shim tile DMA registers (row 0)
SHIM_BD0 = 0x1D000
BD_BYTES = 0x20
S2MM, MM2S = 0, 1  # directions (the sync operation's encoding)
CTRL = {S2MM: (0x1D200, 0x1D208), MM2S: (0x1D210, 0x1D218)}
QUEUE = {S2MM: (0x1D204, 0x1D20C), MM2S: (0x1D214, 0x1D21C)}
QUEUE_STATUS = {S2MM: (0x1D220, 0x1D224), MM2S: (0x1D228, 0x1D22C)}  # (Task_Queue_Size: bits 22:20)
SHIM_BDS = 16
QUEUE_DEPTH = 4  # tasks a channel queues
CONTROLLER_ID = 15  # the firmware's task-complete-token controller (as aiecc sets it)

# memory tile (row 1)
MEM_BD0, MEM_S2MM_CTRL, MEM_MM2S_CTRL, MEM_LOCK0 = 0xA0000, 0xA0600, 0xA0630, 0xC0000
MEM_WEST, MEM_OWN, MEM_EAST = 0, 0x20000, 0x40000  # word-address bases of the three memories a BD can reach
MEM_BDS, MEM_CONTROLLER_ID = 48, 26
CORE_LOCK0 = 0x1F000  # a core tile's lock values (16 B apart)

_BD_BURST_512B = 3 << 30  # BD word 4: AXI burst length 32 beats
_BD_AXCACHE = 2 << 24  # BD word 5 (AxQoS stays 0)
_BD_VALID = 1 << 25  # BD word 7


def tile_address(col: int, row: int, reg: int) -> int:
    """Absolute address of a tile register (NPU2: column << 25 | row << 20 | offset)."""
    return (col << 25) | (row << 20) | reg


class Txn:
    """A transaction under construction; ``words()`` returns it with its header."""

    def __init__(self, *, dev_gen: int = 4, rows: int = 6, cols: int = 8, mem_rows: int = 1):
        self._header = ((rows << 24) | (dev_gen << 16) | (1 << 8), (mem_rows << 8) | cols)
        self._ops: list[list[int]] = []
        self.n_ops = 0
        self.n_words = 4  # words so far, with the header

    def _add(self, words: list[int]) -> int:
        """Append an operation; returns its word index in ``words()``."""
        at = self.n_words
        self._ops.append(words)
        self.n_ops += 1
        self.n_words += len(words)
        return at

    def write32(self, addr: int, value: int) -> int:
        return self._add([OP_WRITE, 0, addr, 0, value, 24])

    def maskwrite32(self, addr: int, value: int, mask: int) -> int:
        return self._add([OP_MASKWRITE, 0, addr, 0, value, mask, 28])

    def blockwrite(self, addr: int, data: list[int]) -> int:
        return self._add([OP_BLOCKWRITE, 0, addr, 4 * (4 + len(data)), *data])

    def maskpoll32(self, addr: int, value: int, mask: int) -> int:
        """Wait until the register's bits ``mask`` hold ``value``."""
        return self._add([OP_MASKPOLL, 0, addr, 0, value, mask, 28])

    def sync(self, *, col: int, direction: int, channel: int, ncol: int = 1, row: int = 0, nrow: int = 1) -> None:
        """Wait for the task-complete tokens of ``channel`` (``direction`` S2MM / MM2S) of the
        tiles in columns col .. col + ncol - 1, rows row .. row + nrow - 1."""
        self._add([OP_TCT, 16, direction | (row << 8) | (col << 16), (nrow << 8) | (ncol << 16) | (channel << 24)])

    # ------------------------------------------------------------------
    # Shim DMA
    # ------------------------------------------------------------------

    def token_controller(self, col: int, direction: int, channel: int) -> None:
        """Route the channel's task-complete tokens to the firmware (once per channel and
        dispatch, before its first task with a token)."""
        self.maskwrite32(tile_address(col, 0, CTRL[direction][channel]), CONTROLLER_ID << 8, 0x1F00)

    def shim_bd(self, col: int, bd: int, *, address: int, nbytes: int,
                dims: tuple[int, int, int, int] | None = None, iteration: tuple[int, int] | None = None,
                next_bd: int | None = None) -> int:
        """Write shim BD ``bd`` of column ``col``: ``nbytes`` bytes from the device address
        ``address`` (``device_address`` of a host buffer) on, contiguous - or, with ``dims`` = (n0,
        n1, s1, s2) in 32-bit words, word number i0 + n0 * (i1 + n1 * i2) of the stream at address
        + 4 * (i0 + i1 * s1 + i2 * s2) (i0 < n0, i1 < n1): strips of n0 words, n1 of them s1
        apart, such groups s2 apart. ``iteration`` = (step, count) in 32-bit words: execution i of a task
        that repeats the BD (``shim_push(repeat=)``) starts at address + 4 * step * (i % count). Returns the
        word index of the BD in ``words()``: [length in words, address low, address high] may be rewritten
        there (``bd_words``) before a later dispatch of the same transaction. next_bd: the task continues with that BD
        (a chain: one queue entry)."""
        assert 0 <= bd < SHIM_BDS
        d0 = d1 = d2 = 0
        if dims is not None:
            n0, n1, s1, s2 = dims
            assert 0 < n0 < 1024 and 0 < n1 < 1024 and 0 < s1 <= 1 << 20 and 0 < s2 <= 1 << 20, dims
            assert nbytes % (4 * n0 * n1) == 0, (nbytes, dims)
            d0, d1, d2 = n0 << 20, (n1 << 20) | (s1 - 1), s2 - 1  # (size, step - 1); d0 steps by single words
        it = 0
        if iteration is not None:  # BD word 6: Iteration_Current 0, Iteration_Wrap - 1, Iteration_Stepsize - 1
            step, count = iteration
            assert 0 < step <= 1 << 20 and 0 < count <= 64, iteration
            it = ((count - 1) << 20) | (step - 1)
        nxt = 0 if next_bd is None else next_bd << 27 | 1 << 26  # (BD word 7: Next_BD, Use_Next_BD)
        assert next_bd is None or 0 <= next_bd < SHIM_BDS
        return self.blockwrite(tile_address(col, 0, SHIM_BD0 + BD_BYTES * bd),
                               [*bd_words(address, nbytes), d0, _BD_BURST_512B | d1, _BD_AXCACHE | d2, it,
                                _BD_VALID | nxt]) + 4

    def shim_push(self, col: int, direction: int, channel: int, bd: int, *, token: bool = False,
                  repeat: int = 1) -> None:
        """Queue BD ``bd`` as a task of the channel (``token``: a task-complete token for ``sync``)."""
        assert 1 <= repeat <= 256
        self.write32(tile_address(col, 0, QUEUE[direction][channel]), (int(token) << 31) | ((repeat - 1) << 16) | bd)

    def set_token(self, op: int) -> None:
        """Give the queue push of operation number ``op`` (a ``shim_push``) a task-complete token."""
        words = self._ops[op]
        assert words[0] == OP_WRITE, words
        words[4] |= 1 << 31

    def shim_task(self, col: int, direction: int, channel: int, bd: int, *, address: int, nbytes: int,
                  token: bool = False, dims: tuple[int, int, int, int] | None = None, repeat: int = 1,
                  iteration: tuple[int, int] | None = None) -> int:
        """BD + push: read (MM2S) or write (S2MM) ``nbytes`` at ``address`` through the channel,
        ``repeat`` times. Returns the BD's word index (see ``shim_bd``)."""
        at = self.shim_bd(col, bd, address=address, nbytes=nbytes, dims=dims, iteration=iteration)
        self.shim_push(col, direction, channel, bd, token=token, repeat=repeat)
        return at

    # ------------------------------------------------------------------
    # Memory tile DMA, locks
    # ------------------------------------------------------------------

    def mem_bd(self, col: int, bd: int, words: list[int]) -> int:
        """Write memory-tile BD ``bd`` of column ``col`` (``mem_bd_words``)."""
        assert 0 <= bd < MEM_BDS
        return self.blockwrite(tile_address(col, 1, MEM_BD0 + 32 * bd), words)

    def mem_push(self, col: int, direction: int, channel: int, bd: int, *, repeat: int = 1, token: bool = False) -> int:
        """Queue memory-tile BD ``bd`` as a task of the channel (``repeat`` runs); returns the operation's number (for
        ``set_token``)."""
        assert 1 <= repeat <= 256 and 0 <= bd < MEM_BDS and 0 <= channel < 6 and bd // 24 == channel % 2, (bd, channel)
        reg = (MEM_S2MM_CTRL if direction == S2MM else MEM_MM2S_CTRL) + 8 * channel + 4
        n = self.n_ops
        self.write32(tile_address(col, 1, reg), (int(token) << 31) | ((repeat - 1) << 16) | bd)
        return n

    def mem_token_controller(self, col: int, direction: int, channel: int) -> None:
        """Route a memory-tile channel's task-complete tokens to the firmware (once per channel and dispatch)."""
        reg = (MEM_S2MM_CTRL if direction == S2MM else MEM_MM2S_CTRL) + 8 * channel
        self.maskwrite32(tile_address(col, 1, reg), MEM_CONTROLLER_ID << 8, 0xFF00)

    def lock_set(self, col: int, row: int, lock: int, value: int) -> None:
        """Set a lock's value (row 1: a memory tile's lock 0-63; rows 2-5: a core tile's 0-15)."""
        self.write32(tile_address(col, row, (MEM_LOCK0 if row == 1 else CORE_LOCK0) + 16 * lock), value)

    def words(self) -> np.ndarray:
        body = [w for op in self._ops for w in op]
        return np.array([*self._header, self.n_ops, 4 * (4 + len(body)), *body], np.uint32)


def mem_lock(lock: int, base: int = MEM_OWN) -> int:
    """Memory-tile lock ``lock`` (0-63) of the tile ``base`` names (``MEM_WEST`` / ``MEM_OWN`` / ``MEM_EAST``) as its
    BDs number it."""
    assert 0 <= lock < 64
    return lock + {MEM_WEST: 0, MEM_OWN: 64, MEM_EAST: 128}[base]


def _s7(v: int) -> int:
    assert -64 <= v < 64, v
    return v & 0x7F


def mem_bd_words(*, addr: int, nwords: int, dims: list[tuple[int, int]] | None = None, base: int = MEM_OWN,
                 iteration: tuple[int, int] | None = None, acquire: tuple[int, int, bool] | None = None,
                 release: tuple[int, int] | None = None, next_bd: int | None = None) -> list[int]:
    """A memory-tile BD's 8 words: ``nwords`` 32-bit words from byte ``addr`` of the memory ``base`` names; ``dims``
    innermost first as (size, stride) in words, d0..d2 with sizes (< 1024) and an optional (0, stride) d3; iteration
    (step words, count <= 64): execution i of a repeated task starts at step * (i % count); acquire (lock, value,
    greater or equal: else equal), release (lock, value); next_bd: chain (the task runs it after this one)."""
    assert addr % 4 == 0 and 0 <= addr < 1 << 19 and 0 < nwords < 1 << 17, (addr, nwords)
    w = [nwords, base + addr // 4, 0, 0, 0, 0, 0, 0]
    if next_bd is not None:
        assert 0 <= next_bd < MEM_BDS
        w[1] |= 1 << 19 | next_bd << 20
    for i, (size, stride) in enumerate(dims or []):
        assert 0 < stride <= 1 << 17, (i, size, stride)
        if i < 3:
            assert 0 < size < 1024, (i, size)
            w[2 + i] = (size << 17) | (stride - 1)
        else:
            assert i == 3 and size == 0, dims
            w[5] = stride - 1
    if iteration is not None and iteration[1] > 1:  # (an iteration of one step: none)
        step, count = iteration
        assert 0 < step <= 1 << 17 and count <= 64, iteration
        w[6] = ((count - 1) << 17) | (step - 1)
    w7 = 1 << 31
    if acquire is not None:
        lk, v, ge = acquire
        w7 |= lk | _s7(-v if ge else v) << 8 | 1 << 15
    if release is not None:
        lk, v = release
        w7 |= lk << 16 | _s7(v) << 24
    w[7] = w7
    return w


def bd_words(address: int, nbytes: int) -> list[int]:
    """The first three words of a shim BD: length in 32-bit words, address low, address high."""
    assert nbytes % 4 == 0 and nbytes > 0 and address % 4 == 0 and address < 1 << 48 and nbytes // 4 < 1 << 32, (
        address, nbytes)
    return [nbytes // 4, address & 0xFFFFFFFF, address >> 32]


def bd_offsets(nbytes: int, dims: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Host model of a shim BD (``Txn.shim_bd``): the byte offsets from its address of the 32-bit
    words of its stream, in stream order (layout tests: numbered data through a gather or a
    scatter)."""
    i = np.arange(nbytes // 4)
    if dims is None:
        return 4 * i
    n0, n1, s1, s2 = dims
    return 4 * (i % n0 + i // n0 % n1 * s1 + i // (n0 * n1) * s2)


def device_address(buffer: object, offset: int = 0) -> int:
    """The address the shim DMA uses for byte ``offset`` of a host buffer (an ``artlite``
    Buffer or Tensor)."""
    raw = getattr(buffer, "_raw", buffer)
    return raw.paddr + offset + DDR_AIE_ADDR_OFFSET


def op_offsets(body: np.ndarray) -> list[tuple[int, int]]:
    """(word offset, opcode) of each operation of a transaction body (its words after the header)."""
    m = memoryview(np.ascontiguousarray(body, np.uint32)).cast("B").cast("I")
    out, p = [], 0
    while p < len(m):
        op = m[p] & 0xFF
        out.append((p, op))
        p += m[p + 3] >> 2 if op == OP_BLOCKWRITE else m[p + 1] >> 2 if op >= 0x80 else _OP_WORDS[op]
    assert p == len(m), (p, len(m))
    return out


def decode(words: np.ndarray) -> list[tuple[str, dict[str, int]]]:
    """The operations of a transaction (``insts.bin`` words), for inspection and tests."""
    w = np.asarray(words, np.uint32)
    n_ops, size = int(w[2]), int(w[3])
    assert size == 4 * w.size, (size, w.size)
    out: list[tuple[str, dict[str, int]]] = []
    p = 4
    for _ in range(n_ops):
        op = int(w[p]) & 0xFF
        if op == OP_WRITE:
            out.append(("write32", {"addr": int(w[p + 2]), "value": int(w[p + 4])}))
            p += 6
        elif op == OP_MASKWRITE:
            out.append(("maskwrite32", {"addr": int(w[p + 2]), "value": int(w[p + 4]), "mask": int(w[p + 5])}))
            p += 7
        elif op == OP_MASKPOLL:
            out.append(("maskpoll32", {"addr": int(w[p + 2]), "value": int(w[p + 4]), "mask": int(w[p + 5])}))
            p += 7
        elif op == OP_BLOCKWRITE:
            n = int(w[p + 3]) // 4
            out.append(("blockwrite", {"addr": int(w[p + 2]), "data": [int(x) for x in w[p + 4:p + n]]}))
            p += n
        elif op == OP_TCT:
            a, b = int(w[p + 2]), int(w[p + 3])
            out.append(("sync", {"direction": a & 0xFF, "row": (a >> 8) & 0xFF, "col": (a >> 16) & 0xFF,
                                 "nrow": (b >> 8) & 0xFF, "ncol": (b >> 16) & 0xFF, "channel": b >> 24}))
            p += 4
        elif op == OP_DDR_PATCH:
            out.append(("patch", {"addr": int(w[p + 6]), "arg": int(w[p + 8]),
                                  "offset": int(w[p + 10]) | (int(w[p + 11]) << 32)}))
            p += 12
        else:
            raise ValueError(f"unknown transaction opcode {op:#x} at word {p}")
    assert p == w.size, (p, w.size)
    return out
