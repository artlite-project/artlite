"""What the models' prefills share (``ChunkPrefill``; the models': ``qwen35.prefill``, ``gemma4.prefill``,
``llama.prefill``): a prompt chunk of T tokens (multiples of ``SUB`` up to ``MAX_T``) through the layers on the decode
overlay's buffers, so that the decode continues from the caches and states the chunk leaves.

A chunk runs layer-major in S = T / SUB sub-chunks: each operator of a layer runs over the sub-chunks back to back on
buffer slices SUB tokens apart (consecutive runs of one operator share its configuration: a ConfigureOp per layer and
operator, not per sub-chunk). The control code is assembled on the host (``fused.Library``): one ELF holds each
distinct operator design once (it compiles like a single layer, whatever the model's depth and the chunk sizes), a
chunk's program lists them per layer and sub-chunk with each item's buffer slices (the model's ``_runlist``), and runs
in pieces of whole layers (dispatches of one HW context), each well within the driver's hang detection (2 s:
``PIECE_BUDGET_S``, by the model's time estimate at the chunk's position, ``layer_seconds``).

Chunks start at multiples of ``group`` positions (whole cache groups: ``GROUP``, or a model's own, Llama's 64 / 128);
``plan`` covers a prompt with full chunks of the largest size that fits, then the tail with the sizes of the least
estimated time, but a prompt's last few tokens (a decode step's SLOTS at most) for one decode step. The padding tokens
after a chunk's valid ones write cache records that later tokens overwrite (masked until then).

Packed chunks (a readout model's: several inputs, segments, in one chunk; ``pack_sizes``) run their own programs,
keyed ``("pack", T)``: the model's runlist with the operators that keep the segments apart (``_runlist_of(T,
packed=True)``), the same items per layer; a recurrent model's segments after a cached prefix run tree chunks
(``tree_sizes``, ``("tree", T)``, ``_runlist_of(T, tree=True)``: each segment continues the prefix's states); a model with
a second packed attention design runs it as ``("ring", T)`` (``ring_sizes``, ``_runlist_of(T, packed=True, ring=True)``). ``pack``
plans a request's inputs: first-fit-decreasing into chunks of each packed size in turn (``packed_sizes``; segments
rounded up to ``seg_unit`` tokens; a chunk's fixed cost, its ConfigureOps and every weight's pass, shared by its inputs),
the plan of the least estimated time (``packed_seconds``) against each input run alone (``chunk_seconds`` of its
``plan``): an input longer than ``span``, a chunk that would not be faster than its inputs alone, run alone.

A model's prefill supplies its runlist per sub-chunk count (``_runlist``, built with ``_sl`` / ``_tok``; chunk sizes below
SUB, ``SMALL``: one sub-chunk of operators of their own size, ``_runlist_of``), the bytes per token of its own buffers
(``rb``), their static contents (``_bind``), its time model (``layer_seconds``, ``layer_configs``, ``layer_entries`` or
its own ``_layer_starts``;
``_fits_budget``; the sub-chunks S a fraction for a small size) and the layers whose caches a chunk writes
(``kv_layers``).

The dense GEMMs are paced designs (``gemm.pacing``): every chunk's parameters carry the machine's ``PACE``
word (``pace.resolve``: before the first chunk, or at ``prepare``; 0 where nothing throttles them).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .. import _native, parallel_compile
from ..gemm.pacing import PACE, PACE_IDLE, pace_levels
from ..operators.base import ArtcOperator
from ..operators.fused import Library
from ..operators.util import arg_bytes, rup
from . import gemm as PG
from . import pace as PACING

PIECE_BUDGET_S = 0.8  # estimated seconds per dispatch (the hang detection: 2 s; DDR contention adds up to 40%)
PEER_PIECE_S = 0.05  # the same while another process uses the NPU: its requests wait for a piece (``Library.run``),
# and two processes that both keep the NPU busy get about half of it each (Qwen3-Embedding-0.6B's batches against Llama
# 3.2 3B's decode: 1.9x / 2.1x their time alone; at 0.1 s 1.5x / 2.8x)
CONFIG_S = 0.12e-3  # a ConfigureOp (~30-40 us alone: no array reset but at a piece's first, ``fused.program``; the
# cores' programs by control packets: Llama's fitted 0.111 ms, Qwen's 0.085 ms in their own models; Gemma's model
# 1.5-1.6x its chunks)
GROUP = 32  # positions a chunk starts at a multiple of (whole cache groups; a model's default ``group``)


def norm_blocks(T: int, t: int, sizes: Sequence[int]) -> list[tuple[int, int]]:
    """A chunk of T tokens' norm blocks (first token, tokens): the largest of ``sizes`` (descending) that fit, then
    sub-chunks of t tokens."""
    out, a = [], 0
    while a < T:
        n = next((m for m in sizes if T - a >= m), t)
        out.append((a, n))
        a += n
    return out


@dataclass
class Packing:
    """A request's inputs as packed chunks (``bins``: per chunk its size and the inputs' indices) and inputs run alone
    (``alone``); ``seconds`` its estimate, ``sequential`` every input's alone."""

    bins: list[tuple[int, list[int]]] = field(default_factory=list)
    alone: list[int] = field(default_factory=list)
    seconds: float = 0.0
    sequential: float = 0.0


class ChunkPrefill:
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (those within the overlay's caches: a
    larger chunk's transfers leave them). ``compile()`` builds the library ELF (cached; no device needed), ``bind()``
    (after the overlay's ``load``) allocates the prefill's buffers on the overlay's device; ``plan`` splits a prompt
    into chunks, the model's ``run`` runs one."""

    SUB: ClassVar[int]  # tokens per sub-chunk (the operators' T)
    SMALL: ClassVar[tuple[int, ...]] = ()  # chunk sizes below SUB (one sub-chunk of operators of their own T)
    MAX_T: ClassVar[int]  # the largest chunk
    MIN_TOKENS: ClassVar[int] = 1  # valid tokens of a chunk at least
    NAME: ClassVar[str]  # the library ELF's
    SHARED_KV_RANGE: ClassVar[bool] = False  # the chunk sizes' attention offsets are one runtime parameter
    KV_PAD: ClassVar[int]  # positions the overlay's KV arguments extend past the caches (its ``KV_PAD``)
    CFG_S: ClassVar[float]  # the time model's constants (``_layer_static``): a ConfigureOp,
    WEIGHT_S: ClassVar[float]  # a byte of a layer's weights per sub-chunk

    def __init__(self, overlay: Any, sizes: tuple[int, ...]):
        assert all((t % self.SUB == 0 or t in self.SMALL) and t <= self.MAX_T for t in sizes)
        self.ov, self.sizes = overlay, tuple(sorted(t for t in sizes if t <= overlay.max_positions))
        assert self.sizes, f"no chunk size within {overlay.max_positions} positions"
        # (a chunk at the caches' end slices a cache up to its size - SUB positions past them: the NPU writes outside the
        # KV buffer without the overlay's padding. Where the sizes share one offset parameter (the attention's P_FO:
        # one range, the largest chunk's) a smaller chunk's offsets take that range: the bound is the largest size less
        # the smallest; qwen35's attention names its offset parameter per size)
        low = self.sizes[0] if self.SHARED_KV_RANGE else self.SUB
        assert self.sizes[-1] - low <= self.KV_PAD, (self.sizes[-1], low, self.KV_PAD)
        self.cfg, self.img = overlay.cfg, overlay.image
        self.H = self.cfg.hidden_size
        self.group = GROUP
        self.rb: dict[str, int] = {}  # (bytes per token of the token buffers)
        self.pack_sizes: tuple[int, ...] = ()  # (the packed chunks' sizes: the model's, a readout model's)
        self.tree_sizes: tuple[int, ...] = ()  # (a recurrent model's packed chunks after a prefix)
        self.ring_sizes: tuple[int, ...] = ()  # (packed chunks of a second attention design: the model's ``("ring", T)``)
        self.lead = 0  # (the tokens of a prompt a packed chunk may lead with, which every segment sees: ``run_packed``)
        self.span = 0  # (a packed chunk's longest segment)
        self._est: dict[tuple, Any] = {}  # (the planner's estimates by (kind, tokens or size, position); pieces' splits)
        self._last_pack: tuple[tuple, Packing] | None = None  # (``pack``'s last request and plan)
        self._items: dict[int | tuple[str, int], list] = {}
        self._static_s: dict[tuple[int, float], float] = {}  # (``_layer_static`` by layer and sub-chunks)
        self.dispatches = 0  # (the last chunk's)
        self.bound = False
        self.pace: int | None = None  # (the PACE word, once resolved: ``pace_ready``)

    def _library(self) -> None:
        """The runlists of the chunk sizes, the prefill's own buffers, the library of their operators."""
        self.runlists: dict[int | tuple[str, int], list[tuple]] = {T: self._runlist_of(T) for T in self.sizes}
        self.runlists |= {("pack", T): self._runlist_of(T, packed=True) for T in self.pack_sizes}
        self.runlists |= {("tree", T): self._runlist_of(T, tree=True) for T in self.tree_sizes}
        self.runlists |= {("ring", T): self._runlist_of(T, packed=True, ring=True) for T in self.ring_sizes}
        self._sizes()
        self.lib = Library(self.NAME, (op for rl in self.runlists.values() for op, *_ in rl))
        self.paced = any(getattr(op, "pace", None) is not None for rl in self.runlists.values() for op, *_ in rl)

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _runlist(self, S: int) -> list[tuple]:
        """The runlist of a chunk of S sub-chunks: per layer each operator over the sub-chunks, the items (operator,
        its arguments' buffer slices "NAME[a:b]")."""
        raise NotImplementedError

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False) -> list[tuple]:
        """The runlist of a chunk of size T (a ``SMALL`` size: the model's own; ``packed``, ``tree``: a packed or tree
        chunk's)."""
        assert not packed and not tree, f"{type(self).__name__} has no packed chunks"
        return self._runlist(T // self.SUB)

    def subs(self, T: int) -> int | float:
        """The sub-chunks of a chunk of size T for the time model (a ``SMALL`` size: the fraction T / SUB)."""
        return T // self.SUB if T >= self.SUB else T / self.SUB

    @staticmethod
    def _sl(buf: str, off: int, op: ArtcOperator, *, i: int) -> str:
        """The slice of buffer ``buf`` from byte off as long as the operator's argument i."""
        return f"{buf}[{off}:{off + arg_bytes(op, i)}]"

    def _tok(self, buf: str, j: int, op: ArtcOperator, *, i: int, row: int = 0, extra: int = 0) -> str:
        """Sub-chunk j's slice of a token buffer (``row``: bytes per token, default ``rb``'s; ``extra``: bytes past the
        sub-chunk's first token) for the operator's argument i."""
        return self._sl(buf, j * self.SUB * (row or self.rb[buf]) + extra, op, i=i)

    @staticmethod
    def _w(g: PG.Gemm) -> str:
        """A GEMM's weights: its tiles in the overlay's image."""
        return f"W[{g.w_off}:{g.w_off + g.op.w_bytes}]"

    def _layer_starts(self, T: int) -> list[int]:
        """The program item where each layer of a chunk of size T starts, and the program's end (``layer_entries``,
        after the first norm's sub-chunks)."""
        n = max(1, T // self.SUB)
        starts = [n]
        for L in self.layers:
            starts.append(starts[-1] + self.layer_entries(L, n))
        return starts

    def _splits(self, T: int, p0: int, budget: float = PIECE_BUDGET_S) -> list[int]:
        """The program items (``_runlist``) that start a piece: whole layers, each piece under ``budget`` seconds at p0,
        the pace's slowdown included (``pace_factor``; ``Library.run`` also cuts where a piece's control code would
        exceed its BOs)."""
        S, starts, f = self.subs(T), self._layer_starts(T), self.pace_factor()
        splits, t = [], 0.0
        for k, L in enumerate(self.layers):
            s = self.layer_seconds(L, p0, S) * f
            if k and t + s > budget:
                splits.append(starts[k])
                t = 0.0
            t += s
        assert starts[-1] == len(self.runlists[T]), (starts[-1], len(self.runlists[T]))
        return splits

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _sizes(self) -> None:
        """The sizes of the prefill's own buffers (``own``: the largest reference of any size's runlist, then the
        model's ``_own_sizes``), in 4 KB."""
        own: dict[str, int] = {}
        ovb = {"W", *self.ov.sizes, *self._external()}
        for rl in self.runlists.values():
            for op, *bufs in rl:
                for i, b in enumerate(bufs):
                    base = b.split("[")[0]
                    if base in ovb:
                        continue
                    end = int(b[b.index(":") + 1:-1]) if "[" in b else arg_bytes(op, i)
                    own[base] = max(own.get(base, 0), end)
        self._own_sizes(own)
        self.own = {n: rup(v, 4096) for n, v in own.items()}

    def _own_sizes(self, own: dict[str, int]) -> None:
        """The model's changes to the sizes of its own buffers (in place)."""

    def _external(self) -> set[str]:
        """Names of buffers the items address beyond the overlay's and the prefill's own (``_buffers``)."""
        return set()

    def _buffers(self) -> dict[str, Any]:
        """The device buffers the items address by name: the overlay's (the prefill's own among them)."""
        return self.ov.dev.buffers

    def _stage_px(self, x: np.ndarray, T: int, dtype: type, *, zero: np.ndarray | None = None) -> None:
        """The chunk's input: x (n, hidden) as the first rows of the residual stream PX in ``dtype`` (``bfloat16`` or
        ``np.float32``), the rows to T zero (and those at the indices ``zero``), flushed to the device."""
        dev, H, n = self.ov.dev, self.H, x.shape[0]
        size = np.dtype(dtype).itemsize
        px = dev.view("PX")[:size * T * H].view(dtype).reshape(T, H)
        px[:n] = np.asarray(x, np.float32)
        px[n:] = 0
        if zero is not None:
            px[zero] = 0
        dev.flush("PX", 0, size * T * H)

    def _px_rows(self, i: int, n: int, dtype: type) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) fp32 of the residual stream PX the last chunk left (``dtype``: ``bfloat16`` or
        ``np.float32``, as ``_stage_px``)."""
        dev, H = self.ov.dev, self.H
        size = np.dtype(dtype).itemsize
        dev.fetch("PX", size * i * H, size * n * H)
        rows = dev.view("PX")[size * i * H:size * (i + n) * H].copy()
        return rows.view(dtype).astype(np.float32).reshape(n, H)

    def bind(self) -> ChunkPrefill:
        """Allocate the prefill's buffers on the (loaded) overlay's device, write the static ones."""
        assert self.ov.dev is not None, "load the overlay first"
        if self.bound:
            return self
        for name, n in self.own.items():
            self.ov.dev.alloc(name, n)
        self._bind()
        self.bound = True
        return self

    def _bind(self) -> None:
        """The model's static buffers (``bind``)."""

    def _program(self, key: int | tuple[str, int]) -> list[tuple[int, list[int]]]:
        """The items of a chunk's program (``key``: its size, or ("pack" / "tree", size)): (library entry, device
        addresses of its arguments' slices)."""
        if key not in self._items:
            self._items[key] = self.lib.items(self.runlists[key], self._buffers())
        return self._items[key]

    # ------------------------------------------------------------------
    # Compile, run
    # ------------------------------------------------------------------

    def compile(self) -> ChunkPrefill:
        """Build the library and load it (its configurations' control packets: seconds of host work for a model's
        designs, e.g. in the thread that runs while the overlay's weights pack, not at the first chunk)."""
        parallel_compile.compile_all([self.lib])
        self.lib.load()
        return self

    def prepare(self, p0: int, n: int) -> None:
        """Before a prompt of n tokens at p0 is timed: the library's hardware context and piece BOs, the items of the
        plan's chunks (else the first chunk's)."""
        self.lib.prepare()
        for T in set(self.plan(p0, n)):
            self._program(T)
        self.pace_ready()

    def pace_factor(self) -> float:
        """At most how much slower than its estimate a layer runs at the resolved pace (1 / its deepest level's duty: a
        level forced by ``ARTLITE_PACE`` on a machine that does not throttle; 1 before it is resolved)."""
        return 1 + max((PACE_IDLE[v] for v in pace_levels(self.pace or 0).values()), default=0) / 16

    def pace_ready(self) -> int:
        """The PACE word (``pace.resolve``: calibrated once per machine and library, a fraction of a second where
        nothing throttles, then cached), resolved at the first call; 0 for a library without paced designs."""
        if self.pace is None:
            self.pace = PACING.resolve(self) if self.paced else 0
        return self.pace

    def fits(self, T: int, p0: int) -> bool:
        """Whether a chunk of size T may start at p0: inside the caches, within the budget (``_fits_budget``)."""
        if p0 % self.group or p0 + T > self.ov.max_positions:
            return False
        key = ("fits", T, p0, self.pace)
        if key not in self._est:
            self._est[key] = self._fits_budget(T, p0)
        return self._est[key]

    def _fits_budget(self, T: int, p0: int) -> bool:
        """Whether a chunk of size T at p0 runs in pieces within PIECE_BUDGET_S (one layer each at least)."""
        return max(self.layer_seconds(L, p0, self.subs(T)) for L in self.layers) * self.pace_factor() <= PIECE_BUDGET_S

    def plan(self, p0: int, n: int, *, steps: bool = True) -> list[int]:
        """The chunk sizes (in order) that run the first tokens of n from position p0: full chunks of the largest size
        that fits, then the tail as the sizes of the least estimated time; tokens left (the overlay's SLOTS at most:
        one decode step, which streams the weights once as a chunk does, without its ConfigureOps and the LM head's
        dispatch, unless ``steps`` is False: a readout, which needs no LM head; fewer than ``MIN_TOKENS``; or no size
        fits) are for decode steps."""
        out: list[int] = []
        while n >= (max(self.MIN_TOKENS, self.ov.SLOTS + 1) if steps else self.MIN_TOKENS):
            first = next((T for T in self.sizes if self.fits(T, p0)), 0)
            if not first:
                break
            ok = [first, *(T for T in self.sizes if T > first and self.fits(T, p0))] if n > first else [first]
            if n >= ok[-1] or n <= first:  # (or the smallest size holds them all: the cheapest cover)
                T = ok[-1]
            else:  # (the tail in units of the sizes' gcd: the cheapest cover)
                cost = {T: self.chunk_seconds(T, p0) for T in ok}
                unit = math.gcd(*ok)
                u = -(-n // unit)
                best = [(0.0, 0)] + [(math.inf, 0)] * u
                for k in range(1, u + 1):
                    best[k] = min((cost[T] + best[max(0, k - T // unit)][0], T) for T in ok)
                T = best[u][1]
            m = min(T, n)
            out.append(T)
            p0, n = p0 + m, n - m
        return out

    def alone_seconds(self, n: int, p0: int) -> float:
        """Estimated time of an input of n tokens at p0 alone: its chunks (``plan`` of a readout)."""
        key = ("alone", n, p0)
        if key not in self._est:
            out, p = 0.0, p0
            for T in self.plan(p0, n, steps=False):
                out += self.chunk_seconds(T, p)
                p += T
            self._est[key] = out
        return self._est[key]

    def packed_sizes(self, p0: int) -> tuple[int, ...]:
        """The packed chunks' sizes after p0 cached positions (this default: ``pack_sizes`` at any position)."""
        return self.pack_sizes

    def seg_unit(self, p0: int) -> int:
        """The multiple of tokens a packed chunk's segments start at after p0 cached positions."""
        return 1

    def pack(self, lengths: Sequence[int], p0: int) -> Packing:
        """How inputs of ``lengths`` tokens after p0 (the caches' prefix every one of them continues) run (module
        docstring): packed chunks of their indices in order and inputs alone (a plan per request: the last one kept)."""
        key = (tuple(lengths), p0, self.pace)
        if self._last_pack is not None and self._last_pack[0] == key:
            return self._last_pack[1]
        out = self._pack(lengths, p0)
        self._last_pack = (key, out)
        return out

    def _pack(self, lengths: Sequence[int], p0: int) -> Packing:
        alone = [self.alone_seconds(n, p0) for n in lengths]
        if p0 % self.group or not self.packed_sizes(p0):
            return Packing(alone=list(range(len(lengths))), seconds=sum(alone), sequential=sum(alone))
        sizes = tuple(T for T in self.packed_sizes(p0) if self.fits(T, p0))
        u = self.seg_unit(p0)
        lengths = [-(-n // u) * u for n in lengths]  # (each segment's tokens and the padding to the next one's start)

        def best(idx: list[int], caps: tuple[int, ...]) -> tuple[float, list[tuple[int, list[int]]]]:
            """The least estimated time of the inputs idx (packable) in chunks of the sizes caps, and their chunks:
            first-fit-decreasing into chunks of each size, each chunk's inputs instead alone or in smaller ones."""
            out = (sum(alone[i] for i in idx), [])
            for cap in caps:
                bins: list[list[int]] = []
                fill: list[int] = []
                for i in sorted(idx, key=lambda i: -lengths[i]):
                    k = next((k for k, f in enumerate(fill) if f + lengths[i] <= cap), len(bins))
                    if k == len(bins):
                        bins.append([])
                        fill.append(0)
                    bins[k].append(i)
                    fill[k] += lengths[i]
                t, plan = 0.0, []
                for b, f in zip(bins, fill, strict=True):
                    T = next(T for T in sizes if T >= f)
                    if ("packed", T, p0) not in self._est:
                        self._est["packed", T, p0] = self.packed_seconds(T, p0)
                    options = [(sum(alone[i] for i in b), [])]
                    if len(b) > 1 and f >= self.MIN_TOKENS:
                        options.append((self._est["packed", T, p0], [(T, sorted(b))]))
                    if len(b) > 1 and T > sizes[0]:
                        options.append(best(b, tuple(c for c in sizes if c < T)))
                    tb, pb = min(options, key=lambda o: o[0])
                    t += tb
                    plan += pb
                if t < out[0]:
                    out = (t, plan)
            return out

        packable = [i for i, n in enumerate(lengths) if 0 < n <= self.span]
        t, bins = best(packable, sizes)
        packed = {i for _, b in bins for i in b}
        rest = [i for i in range(len(lengths)) if i not in packed]
        return Packing(bins=bins, alone=rest, seconds=t + sum(alone[i] for i in rest if i not in packable),
                       sequential=sum(alone))

    def _size(self, n: int, p0: int, T: int) -> int:
        """The size of a chunk of n tokens at p0 (T, or the smallest that holds them)."""
        T = T or next(t for t in self.sizes if t >= n)
        assert self.bound, "bind() first"
        assert T in self.runlists and 0 < n <= T and self.fits(T, p0), (T, p0, n)
        return T

    def _launch(self, T: int, p0: int, params: dict[str, int], hooks: dict[int, Callable[[], None]] | None = None, *,
                key: int | tuple[str, int] | None = None) -> None:
        """Dispatch a chunk's program (its inputs written; ``key``: a packed or tree chunk's, ``_program``): the
        scratchpad ``params``, the pieces of ``_splits`` (and one from each item index of ``hooks`` on, its host
        function called before it); the cache layers' rings then hold no rows of the chunk's last record
        (``kv_layers``: record mode)."""
        self.lib.set_params({**params, PACE: self.pace_ready()} if self.paced else params)
        self.ov.ready()
        budget = PEER_PIECE_S if _native.turn_peers() else PIECE_BUDGET_S  # (another process's requests wait less)
        cut = ("splits", T, p0, self.pace, budget)  # (a packed or tree chunk's: its size's, the same items per layer)
        if cut not in self._est:
            self._est[cut] = self._splits(T, p0, budget)
        self.dispatches = self.lib.run(self._program(T if key is None else key),
                                       splits=[*self._est[cut], *(hooks or {})],
                                       bufs=list(self._buffers().values()), hooks=hooks)
        self.ov.vring.from_prefill.update(self.kv_layers)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def layer_seconds(self, L: int, p0: int, S: int = 1) -> float:
        """Estimated time of layer L of a chunk of S sub-chunks at position p0."""
        raise NotImplementedError

    def _phases(self, L: int) -> tuple[str, ...]:
        """The GEMM phases (image regions) of layer L."""
        raise NotImplementedError

    def _layer_static(self, L: int, S: float) -> float:
        """``layer_seconds`` without the attention's: the GEMMs by the bytes of their weights (the norms and the prep
        included), ``CFG_S`` per ConfigureOp (cached: the planner asks for every layer and size)."""
        key = (L, S)
        if key not in self._static_s:
            wbytes = sum(self.img.region(L, name).nbytes for name in self._phases(L))
            self._static_s[key] = S * self.WEIGHT_S * wbytes + self.CFG_S * self.layer_configs(L)
        return self._static_s[key]

    def layer_configs(self, L: int) -> int:
        """ConfigureOps of layer L (its distinct operators)."""
        raise NotImplementedError

    def layer_entries(self, L: int, S: int) -> int:
        """Runlist entries of layer L (``_runlist``) in a chunk of S sub-chunks."""
        raise NotImplementedError

    def chunk_seconds(self, T: int, p0: int) -> float:
        """Estimated time of a chunk of size T at p0."""
        key = ("chunk", T, p0)
        if key not in self._est:
            self._est[key] = sum(self.layer_seconds(L, p0, self.subs(T)) for L in self.layers)
        return self._est[key]

    def packed_seconds(self, T: int, p0: int) -> float:
        """Estimated time of a packed chunk of size T at p0."""
        raise NotImplementedError
