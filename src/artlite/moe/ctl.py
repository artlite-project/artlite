"""Control streams of the MoE prefill's work-list operators (``expert_gemm``, ``gather``): a column's shim streams
control packets from DDR into its own TileControl (the decode's mechanism, ``route``; ``aie2p-coding`` section 8),
here host-written per layer, so that one static operator runs any number of chunks, each at data-dependent addresses.

A column's packet-switched MM2S channel carries its control blocks (stream id ``CID``, routed to the shim's TileControl:
``control_route``) between its data (id ``XID``). Block 0 is the runtime sequence's own fill of the control argument:
its BD stamps the first stream header (the block's words begin with its first packet's control header) and IRON fills
move whole objects of the channel's ObjectFifo (block 0 padded by idempotent rewrites, ``padding``). Block 0 starts
block 1; block k >= 1 writes the BDs of its tasks (whole: two packets each, ``bd_packets``; or their address words),
pushes them, then writes block k + 1's BD (no packet header: its words carry their stream headers) and pushes it
(``assemble``). The channel's FIFO order makes block k + 1 run after the tasks block k queued on it.

BD words from a 4-dimensional tap [iteration, d2, d1, d0] (bf16 units, d0 contiguous: ``bd_words``), as the runtime
sequence's BDs encode them (``txn.Txn.shim_bd``). A layer's streams are assembled in C++ (``assemble``) from block
templates the operators keep."""

import functools

import numpy as np
from aie.helpers.taplib import TensorAccessPattern

from .. import _native, txn
from ..operators.fused.ctrlpkt import _parity, _slot

CID, XID = 9, 3  # stream ids: control packets, the channel's data (as the decode overlay's: ``route``)
# the control route on a shim's MM2S1 (stream-switch slave South 7; the configuration's: slot 0, the data's id -> arbiter
# 0 -> North; its control-packet overlay: arbiter 5): slot 1 (CID) -> arbiter 1 -> master Tile_Ctrl, packet mode, the
# header kept (AM029 PL_MODULE Stream_Switch_*)
SW_SOUTH7_SLOT1, SW_MASTER_TILE_CTRL, CTL_ARB = 0x3F294, 0x3F000, 1
ROUTE = ((SW_SOUTH7_SLOT1, _slot(CID, CTL_ARB)), (SW_MASTER_TILE_CTRL, 1 << 31 | 1 << 30 | 1 << 3 | CTL_ARB))
NEXT_WORDS = 15  # (a block's tail: the next block's BD, whole, and its push)


def ctl(addr: int, data: list[int]) -> list[int]:
    """One control packet: stream header, control header (a write of 1-4 words at the shim's register addr), data."""
    assert 1 <= len(data) <= 4
    return [_parity(CID), _parity(((len(data) - 1) << 20) | (addr & 0xFFFFF)), *data]


def bd_packets(bd: int, words: list[int]) -> list[int]:
    """The two packets that write shim BD ``bd`` whole."""
    assert 0 <= bd < txn.SHIM_BDS and len(words) == 8
    reg = txn.SHIM_BD0 + txn.BD_BYTES * bd
    return ctl(reg, words[:4]) + ctl(reg + 16, words[4:])


def push(direction: int, channel: int, bd: int, *, repeat: int = 1, token: bool = False) -> list[int]:
    """The packet that queues BD ``bd`` on a channel of the shim."""
    assert 1 <= repeat <= 256
    return ctl(txn.QUEUE[direction][channel], [int(token) << 31 | (repeat - 1) << 16 | bd])


def padding(reg: int, words: list[int], n: int) -> list[int]:
    """n words of control packets (3 .. 6 words each) rewriting ``words`` (at least 4) at reg: idempotent."""
    q, r = divmod(n, 6)
    sizes = [6] * q if r == 0 else [6] * q + [r] if r >= 3 else [6] * (q - 1) + [4, 2 + r]  # (r 1, 2: 4 + 3, 4 + 4)
    assert sum(sizes) == n and all(3 <= s <= 6 for s in sizes), n
    out: list[int] = []
    for s in sizes:
        out += ctl(reg, words[:s - 2])
    return out


def bd_words(address: int, tap: TensorAccessPattern, *, packet: int | None = None, next_bd: int | None = None,
             acquire: tuple[int, int] | None = None, release: tuple[int, int] | None = None) -> list[int]:
    """A shim BD's 8 words for ``tap`` (sizes [iteration, d2, d1, d0], strides [step, s2, s1, 1] in bf16 units) of the
    buffer at device address ``address``: the transfer d2 x d1 x d0 from address + 2 offset, repeated by its task with
    the address advancing ``step`` per execution (Iteration_Wrap = the iteration size). packet / next_bd / acquire /
    release: as ``txn.Txn.shim_bd``."""
    (it, d2, d1, d0), (s_it, s2, s1, s0) = tap.sizes, tap.strides
    assert s0 == 1 and d0 % 2 == 0 and (d1 == 1 or s1 % 2 == 0) and (d2 == 1 or s2 % 2 == 0), (tap.sizes, tap.strides)
    nbytes = 2 * d2 * d1 * d0
    w = txn.bd_words(address + 2 * tap.offset, nbytes)
    dims = [0, 0, 0]
    if d1 > 1 or d2 > 1:
        n0, n1 = d0 // 2, d1
        stride1 = s1 // 2 if d1 > 1 else n0
        stride2 = s2 // 2 if d2 > 1 else n0 * n1
        assert 0 < n0 < 1024 and 0 < n1 < 1024 and 0 < stride1 <= 1 << 20 and 0 < stride2 <= 1 << 20, tap
        dims = [n0 << 20, (n1 << 20) | (stride1 - 1), stride2 - 1]
    iteration = 0
    if it > 1:
        assert s_it > 0 and s_it % 2 == 0 and s_it // 2 <= 1 << 20 and it <= 64, tap
        iteration = ((it - 1) << 20) | (s_it // 2 - 1)
    if packet is not None:
        assert 0 <= packet < 32
        w[2] |= 1 << 30 | packet << 19
    w7 = 1 << 25 | (0 if next_bd is None else next_bd << 27 | 1 << 26) | txn.shim_lock_bits(acquire, release)
    return [*w, dims[0], 3 << 30 | dims[1], 2 << 24 | dims[2], iteration, w7]


# ----------------------------------------------------------------------
# Streams (a layer's host step: thousands of blocks)
# ----------------------------------------------------------------------

_HDR = _parity(CID)


def _ctl_header(addr: int, n: int) -> int:
    return _parity(((n - 1) << 20) | (addr & 0xFFFFF))


@functools.lru_cache(maxsize=16)
def _tails(bds: tuple[int, int], channel: int) -> np.ndarray:
    """The two tails (2 x NEXT_WORDS words): writing a next block's BD bds[i] whole (its length, address and lock bits
    zero: ``_native.ctl_assemble`` fills them) and pushing it on MM2S ``channel``."""
    out = np.zeros((2, NEXT_WORDS), np.uint32)
    for i, bd in enumerate(bds):
        reg = (txn.SHIM_BD0 + txn.BD_BYTES * bd) & 0xFFFFF
        out[i, 0] = out[i, 6] = out[i, 12] = _HDR
        out[i, 1] = _parity((3 << 20) | reg)
        out[i, 7] = _parity((3 << 20) | (reg + 16))
        out[i, 8], out[i, 9], out[i, 11] = 3 << 30, 2 << 24, 1 << 25
        out[i, 13] = _ctl_header(txn.QUEUE[txn.MM2S][channel], 1)
        out[i, 14] = bd
    return out.reshape(-1)


@functools.lru_cache(maxsize=256)
def _block0(b1: int, size1: int, b0_words: int, bd: int, channel: int,
            acquire: tuple[int, int] | None = None) -> np.ndarray:
    """Block 0 (b0_words, its first stream header the fill's): block 1's BD (at b1, size1 words) and its push, padded
    (the same per column stream address and block 1's size: cached)."""
    w = bd_words(b1, TensorAccessPattern((1, 2 * size1), 0, [1, 1, 1, 2 * size1], [0, 0, 0, 1]), acquire=acquire)
    tail = bd_packets(bd, w) + push(txn.MM2S, channel, bd)
    b = tail[:-3] + padding(txn.SHIM_BD0 + txn.BD_BYTES * bd, w, b0_words + 1 - len(tail)) + tail[-3:]
    return np.array(b[1:], np.uint32)


Template = tuple[np.ndarray, np.ndarray]  # (a block body's words: uint32; its address slots (S, 4) int64)


def template(words: list[int] | np.ndarray, slots: list[tuple[int, int, int, int]]) -> Template:
    """A block body with slots (lo word, hi word, base index, offset): the value bases[base index] + offset goes into
    word lo (low 32 bits) and word hi (ORed: its high bits; an address), or, hi -1, into word lo alone; the slots'
    words zero here."""
    w = np.array(words, np.uint32)
    for lo, hi, _, _ in slots:
        w[lo] = 0
        if hi >= 0:
            w[hi] &= 0xFFFF0000
    return w, np.array(slots, np.int64).reshape(-1, 4)


def assemble(out: np.ndarray, templates: list[Template], blocks: np.ndarray, bases: np.ndarray, *, base: int,
             b0_words: int, bds: tuple[int, int], channel: int = 1,
             acquire: tuple[np.ndarray, np.ndarray] | None = None) -> int:
    """A stream from device address ``base`` into out (uint32): block 0, then blocks 1 .. of templates[blocks[i]],
    patched from bases[i] (uint64, (n, B)), each with its tail that writes and pushes the next (module docstring;
    ``_native.ctl_assemble``). ``acquire``: (locks, values) per block 1 .. (int arrays; value 0: none), the lock its
    BD acquires. Returns the stream's words."""
    blocks = np.asarray(blocks, np.int32)
    n = blocks.size
    assert n >= 1
    first = templates[blocks[0]][0].size + (NEXT_WORDS if n > 1 else 0)
    bits, a0 = None, None
    if acquire is not None:  # (shim_lock_bits: enable, -value, lock)
        lk, v = (np.asarray(x, np.int64) for x in acquire)
        bits = np.where(v > 0, 1 << 12 | ((-v) & 0x7F) << 5 | lk, 0).astype(np.uint32)
        a0 = (int(lk[0]), int(v[0])) if v[0] else None
    b0 = _block0(base + 4 * b0_words, first, b0_words, bds[1], channel, a0)
    return _native.ctl_assemble(out, base, b0, templates, blocks, np.ascontiguousarray(bases, np.uint64),
                                _tails(tuple(bds), channel), bits)
