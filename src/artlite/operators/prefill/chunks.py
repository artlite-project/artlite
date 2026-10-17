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
estimated time. The padding tokens after a chunk's valid ones write cache records that later tokens overwrite (masked
until then).

A model's prefill supplies its runlist per sub-chunk count (``_runlist``, built with ``_sl`` / ``_tok``; chunk sizes below
SUB, ``SMALL``: one sub-chunk of operators of their own size, ``_runlist_of``), the bytes per token of its own buffers
(``rb``), their static contents (``_bind``), its time model (``layer_seconds``, ``layer_configs``, ``layer_entries`` or
its own ``_layer_starts``;
``_fits_budget``; the sub-chunks S a fraction for a small size) and the layers whose caches a chunk writes
(``kv_layers``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

from ... import parallel_compile
from ..base import ArtcOperator
from ..fused import Library
from ..util import arg_bytes, rup
from . import gemm as PG

PIECE_BUDGET_S = 0.8  # estimated seconds per dispatch (the hang detection: 2 s; DDR contention adds up to 40%)
CONFIG_S = 0.15e-3  # a ConfigureOp (~80-90 us alone: the PDI load's reset ~40 us, the rest's operations; the cores'
# programs by control packets: Llama's fitted 0.143 ms, Qwen's 0.075 ms in its own model; Gemma's ~40% above its chunks)
GROUP = 32  # positions a chunk starts at a multiple of (whole cache groups; a model's default ``group``)


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

    def __init__(self, overlay: Any, sizes: tuple[int, ...]):
        assert all((t % self.SUB == 0 or t in self.SMALL) and t <= self.MAX_T for t in sizes)
        self.ov, self.sizes = overlay, tuple(sorted(t for t in sizes if t <= overlay.max_positions))
        assert self.sizes, f"no chunk size within {overlay.max_positions} positions"
        self.cfg, self.img = overlay.cfg, overlay.image
        self.H = self.cfg.hidden_size
        self.group = GROUP
        self.rb: dict[str, int] = {}  # (bytes per token of the token buffers)
        self._items: dict[int, list] = {}
        self.dispatches = 0  # (the last chunk's)
        self.bound = False

    def _library(self) -> None:
        """The runlists of the chunk sizes, the prefill's own buffers, the library of their operators."""
        self.runlists = {T: self._runlist_of(T) for T in self.sizes}
        self._sizes()
        self.lib = Library(self.NAME, (op for rl in self.runlists.values() for op, *_ in rl))

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _runlist(self, S: int) -> list[tuple]:
        """The runlist of a chunk of S sub-chunks: per layer each operator over the sub-chunks, the items (operator,
        its arguments' buffer slices "NAME[a:b]")."""
        raise NotImplementedError

    def _runlist_of(self, T: int) -> list[tuple]:
        """The runlist of a chunk of size T (a ``SMALL`` size: the model's own)."""
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

    def _splits(self, T: int, p0: int) -> list[int]:
        """The program items (``_runlist``) that start a piece: whole layers, each piece under PIECE_BUDGET_S at p0
        (``Library.run`` also cuts where a piece's control code would exceed its BOs)."""
        S, starts = self.subs(T), self._layer_starts(T)
        splits, t = [], 0.0
        for k, L in enumerate(self.layers):
            s = self.layer_seconds(L, p0, S)
            if k and t + s > PIECE_BUDGET_S:
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
        ovb = {"W", *self.ov.sizes}
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

    def _program(self, T: int) -> list[tuple[int, list[int]]]:
        """The items of a chunk's program: (library entry, device addresses of its arguments' slices)."""
        if T not in self._items:
            self._items[T] = self.lib.items(self.runlists[T], self.ov.dev.buffers)
        return self._items[T]

    # ------------------------------------------------------------------
    # Compile, run
    # ------------------------------------------------------------------

    def compile(self) -> ChunkPrefill:
        parallel_compile.compile_all([self.lib])
        return self

    def fits(self, T: int, p0: int) -> bool:
        """Whether a chunk of size T may start at p0: inside the caches, within the budget (``_fits_budget``)."""
        return p0 % self.group == 0 and p0 + T <= self.ov.max_positions and self._fits_budget(T, p0)

    def _fits_budget(self, T: int, p0: int) -> bool:
        """Whether a chunk of size T at p0 runs in pieces within PIECE_BUDGET_S (one layer each at least)."""
        return max(self.layer_seconds(L, p0, self.subs(T)) for L in self.layers) <= PIECE_BUDGET_S

    def plan(self, p0: int, n: int) -> list[int]:
        """The chunk sizes (in order) that run the first tokens of n from position p0: full chunks of the largest size
        that fits, then the tail as the sizes of the least estimated time; tokens left (fewer than ``MIN_TOKENS``, or
        no size fits) are for decode steps."""
        out: list[int] = []
        while n >= self.MIN_TOKENS:
            ok = [T for T in self.sizes if self.fits(T, p0)]
            if not ok:
                break
            if n >= ok[-1]:
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
            if m < self.MIN_TOKENS:
                break
            out.append(T)
            p0, n = p0 + m, n - m
        return out

    def _size(self, n: int, p0: int, T: int) -> int:
        """The size of a chunk of n tokens at p0 (T, or the smallest that holds them)."""
        T = T or next(t for t in self.sizes if t >= n)
        assert self.bound, "bind() first"
        assert T in self.runlists and 0 < n <= T and self.fits(T, p0), (T, p0, n)
        return T

    def _launch(self, T: int, p0: int, params: dict[str, int], hooks: dict[int, Callable[[], None]] | None = None
                ) -> None:
        """Dispatch a chunk's program (its inputs written): the scratchpad ``params``, the pieces of ``_splits`` (and
        one from each item index of ``hooks`` on, its host function called before it); the cache layers' rings then
        hold no rows of the chunk's last record (``kv_layers``: record mode)."""
        self.lib.set_params(params)
        self.ov.ready()
        self.dispatches = self.lib.run(self._program(T), splits=[*self._splits(T, p0), *(hooks or {})],
                                       bufs=list(self.ov.dev.buffers.values()), hooks=hooks)
        self.ov.vring.from_prefill.update(self.kv_layers)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def layer_seconds(self, L: int, p0: int, S: int = 1) -> float:
        """Estimated time of layer L of a chunk of S sub-chunks at position p0."""
        raise NotImplementedError

    def layer_configs(self, L: int) -> int:
        """ConfigureOps of layer L (its distinct operators)."""
        raise NotImplementedError

    def layer_entries(self, L: int, S: int) -> int:
        """Runlist entries of layer L (``_runlist``) in a chunk of S sub-chunks."""
        raise NotImplementedError

    def chunk_seconds(self, T: int, p0: int) -> float:
        """Estimated time of a chunk of size T at p0."""
        return sum(self.layer_seconds(L, p0, self.subs(T)) for L in self.layers)
