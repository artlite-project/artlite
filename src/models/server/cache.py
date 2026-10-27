"""The server's prompt cache (``PromptCache``): the sequences whose KV records and states it keeps - the one the device
holds and others in host RAM - and what a prompt continues from.

An entry (``Entry``) is a sequence: the tokens of its positions, its checkpoints (``NpuModel.checkpoint``: the model's
state at a position besides the KV records before it) and the host copies of its KV records
(``NpuModel.read_records``, in segments: each save copies the positions after the last one's). The
device holds one entry (``active``). A prompt continues from the entry that keeps the most of it: the device's end
when the prompt extends it, else a checkpoint at or below their common prefix, or (a model that rewinds: ``recurrent``
False) the multiple of ``group`` below it; another entry's records are written to the device from where the device
holds the same tokens, then its checkpoint is restored. The prompt runs the rest.

Saving: before the device's sequence loses more than ``MIN_SAVE`` positions that RAM does not hold, they are copied
(their records, a checkpoint at its end), so that a conversation another one interrupted continues where it was:
agents interleave several (Claude Code's subagents, opencode's title requests, a web UI's chats).

Checkpoints of a recurrent model (Qwen's Gated DeltaNet states cannot go back: a prompt keeps a prefix only where its
state was saved), each where the prompt stops between two parts of it (at the cache group below: ``marks``): where
its last message begins (``Model.turn``: the answer's opener; the next request re-renders the answer, often in other
tokens - a template that drops its thought, a client's own echo of a tool call, a newline before the end token -, and
a regenerated answer runs little more than the opener), where the conversation's first user message begins
(``Model.user_turn``: the system prompt and tools, which the client's next sessions share) and at a saved sequence's
end; at most ``checkpoints`` per sequence (the oldest go first, the lowest stays). A model that rewinds takes one at a
saved sequence's end only (exact there, inside a cache group).

Memory: the saved records and every checkpoint (Qwen3.6-35B-A3B: 17.6 KB of records per position, 80 MB per
checkpoint) within ``ram`` bytes (< 0: no limit; 0: nothing saved, the device's checkpoints only): the least recently
used entry is cut back to its first checkpoint (the prefix later sessions share), then dropped; the device's own
checkpoints go last.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..common import Checkpoint, NpuModel
from ..sequence import Sequence

log = logging.getLogger("artlite.server")

MIN_SAVE = 512  # positions of the device's sequence a prompt may drop without saving them


def common_prefix(a: list[int] | Sequence, b: list[int] | Sequence) -> int:
    """The length of the longest common prefix of two prompts' tokens (ids; a sequence's given rows by their bytes)."""
    if isinstance(a, Sequence) or isinstance(b, Sequence):
        a, b = (x.cache_keys() if isinstance(x, Sequence) else x for x in (a, b))
        return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    n = min(len(a), len(b))
    d = np.flatnonzero(np.asarray(a[:n], np.int64) != np.asarray(b[:n], np.int64))
    return int(d[0]) if len(d) else n


def _cut(segments: list[list], n: int) -> list[list]:
    """Records segments of positions 0 .. n - 1 alone."""
    out = [s for s in segments if s[0].start < n]
    if out and out[-1][0].end > n:
        out[-1] = [r.cut(n) for r in out[-1]]
    return out


@dataclass(eq=False)
class Entry:
    """A cached sequence: ``tokens`` at its positions, ``checkpoints`` (position -> state, oldest first),
    ``segments``: host copies of its records (``NpuModel.read_records``' of
    positions start .. end - 1, in order) of which positions 0 .. ``valid`` - 1 are its tokens' (the device's entry
    holds the rest on the device)."""

    tokens: list[int] | Sequence = field(default_factory=list)
    checkpoints: dict[int, Checkpoint] = field(default_factory=dict)
    segments: list[list] = field(default_factory=list)
    valid: int = 0

    def cut(self, n: int) -> None:
        """Keep positions 0 .. n - 1."""
        self.tokens = self.tokens[:n]
        self.checkpoints = {p: c for p, c in self.checkpoints.items() if p <= n}
        self.valid = min(self.valid, n)

    def branch(self, n: int) -> "Entry":
        """Another entry of positions 0 .. n - 1 (sharing the checkpoints and records)."""
        out = Entry(self.tokens, dict(self.checkpoints), list(self.segments), self.valid)
        out.cut(n)
        return out


class PromptCache:
    """The prompt cache of ``npu`` (module docstring): ``ram`` bytes of host memory (< 0: no limit), ``checkpoints`` per
    sequence at most; ``turn``, ``user_turn``: the tokens that begin a message, a user message (a recurrent model's
    checkpoints where the last message and the first user message begin; None: none)."""

    def __init__(self, npu: NpuModel, *, ram: int = 8192 << 20, checkpoints: int = 32, turn: list[int] | None = None,
                 user_turn: list[int] | None = None):
        self.npu, self.ram, self.max_checkpoints = npu, ram, checkpoints
        self.turn, self.user_turn = turn, user_turn
        self.active = Entry()
        self.entries: list[Entry] = []  # (the others, saved: least recently used first)

    @property
    def tokens(self) -> list[int] | Sequence:
        """The tokens at the device's cached positions."""
        return self.active.tokens

    def nbytes(self) -> int:
        """Host memory the cache holds (records and checkpoints shared by entries counted once)."""
        seen: dict[int, int] = {}
        for e in (self.active, *self.entries):
            for seg in e.segments:
                seen.update((id(r), r.nbytes) for r in seg)
            seen.update((id(c), c.nbytes) for c in e.checkpoints.values())
        return sum(seen.values())

    # ------------------------------------------------------------------
    # A prompt
    # ------------------------------------------------------------------

    def _keeps(self, e: Entry, c: int, n: int) -> int:
        """The positions of entry e a prompt of n tokens that shares its first c with it keeps."""
        live = e is self.active
        if live and len(e.tokens) <= c < n:
            return len(e.tokens)
        k = max((p for p in e.checkpoints if p <= min(c, n - 1)), default=0)
        if not self.npu.recurrent:
            m = min(c, n - 1, len(e.tokens) if live else e.valid)
            k = max(k, m - m % self.npu.group)
        return k

    def prepare(self, ids: list[int] | Sequence, *, enabled: bool = True) -> tuple[int, str]:
        """Put the device where ``ids`` continue from: the positions kept (the device holds ``ids[:kept]``, fewer than
        all) and how (for the log)."""
        npu, act = self.npu, self.active
        best, kept = act, 0
        if enabled:
            for e in (act, *reversed(self.entries)):
                k = self._keeps(e, common_prefix(e.tokens, ids), len(ids))
                if k > kept:
                    best, kept = e, k
        how = ""
        if best is act:
            if kept < len(act.tokens):  # (in place: a checkpoint or a rewind)
                if self._freeze(kept) is act:
                    self.active = act.branch(kept)
                else:
                    act.cut(kept)
                cp = self.active.checkpoints.get(kept)
                if cp is not None and (npu.recurrent or kept % npu.group):
                    npu.restore(kept, cp)
                    how = f"checkpoint at {kept}"
                elif kept:
                    kept = npu.rewind(kept)
                else:
                    npu.restore(0)
        else:  # (another entry: its records from where the device holds the same tokens, then its state)
            t0 = time.perf_counter()
            d = min(common_prefix(act.tokens, best.tokens), kept)
            self._freeze(0)
            for seg in best.segments:
                s0, s1 = max(d, seg[0].start), min(kept, seg[0].end)
                if s0 < s1:
                    npu.write_records(seg, s0, s1)
            npu.restore(kept, best.checkpoints.get(kept))
            self.entries.remove(best)
            if len(best.tokens) - kept > MIN_SAVE:  # (the rest of it stays saved)
                self.entries.append(best)
                self.active = best.branch(kept)
            else:
                best.cut(kept)
                self.active = best
            how = (f"a {len(best.tokens)}-token sequence from RAM ({kept - d} positions' records in "
                   f"{time.perf_counter() - t0:.2f} s)")
        self._fit()
        return kept, how

    def _freeze(self, n: int) -> Entry | None:
        """Before the device keeps only positions 0 .. n - 1 of its sequence: a copy of it stays in RAM (returned) when
        RAM would hold more than MIN_SAVE positions past n - the device's sequence itself, the records RAM lacks
        saved when more than MIN_SAVE, else what RAM holds of it."""
        act = self.active
        if len(act.tokens) - max(n, act.valid) > MIN_SAVE and self._save():
            out = act
        elif act.valid - n > MIN_SAVE:
            out = act.branch(act.valid)
        else:
            return None
        self.entries.append(out)
        return out

    def _save(self) -> bool:
        """The device's sequence into RAM: the records RAM lacks and a checkpoint at its end (False: saving is off)."""
        if self.ram == 0:
            return False
        npu, act = self.npu, self.active
        t0 = time.perf_counter()
        n = len(act.tokens)
        seg = npu.read_records(act.valid, n)  # (it supersedes the records past ``valid`` of the segments before)
        act.segments = [s for s in act.segments if s[0].start < act.valid] + [seg]
        act.valid = n
        if n not in act.checkpoints:
            self._add(act, npu.checkpoint())
        log.info("prompt cache: a %d-token sequence saved (%.2f GB of records in %.2f s)", n,
                 sum(r.nbytes for r in seg) / 1e9, time.perf_counter() - t0)
        return True

    # ------------------------------------------------------------------
    # During and after a prompt
    # ------------------------------------------------------------------

    def marks(self, ids: list[int] | Sequence, kept: int) -> list[int]:
        """The positions inside a prompt (after ``kept``, before its end) where it stops for a checkpoint: where the
        first user message and the last message begin (a recurrent model's), rounded down to the model's ``group``
        (the prompt's next part starts where a prefill chunk can: else its first tokens up to the next group run as
        decode steps, ~0.5 s on the 35B-A3B) unless that is not after ``kept`` (the tokens up to it then run as decode
        steps anyway)."""
        if not self.npu.recurrent or self.max_checkpoints <= 0:
            return []
        t = np.asarray(ids.ids if isinstance(ids, Sequence) else ids, np.int64)
        out = set()
        for pattern, last in ((self.user_turn, False), (self.turn, True)):
            if pattern and len(t) >= len(pattern):
                win = np.lib.stride_tricks.sliding_window_view(t, len(pattern))
                hits = np.flatnonzero((win == np.asarray(pattern)).all(1))
                out |= {int(hits[-1 if last else 0])} if len(hits) else set()
        g = self.npu.group
        return sorted({m - m % g if m - m % g > kept else m for m in out if kept < m < len(ids)})

    def checkpoint(self, tokens: list[int] | Sequence) -> None:
        """The device holds ``tokens`` (a prompt's first part): a recurrent model's checkpoint there."""
        self.active.tokens = tokens
        if not self.npu.recurrent or self.max_checkpoints <= 0:
            return
        assert self.npu.position == len(tokens), (self.npu.position, len(tokens))
        self._add(self.active, self.npu.checkpoint())
        self._fit()

    def ran(self, tokens: list[int] | Sequence) -> None:
        """After a job: the tokens at the device's cached positions."""
        act = self.active
        if (c := common_prefix(act.tokens, tokens)) < len(act.tokens):
            act.cut(c)
        act.tokens = tokens

    def clear(self) -> None:
        """The device's state is unknown (a failure): its sequence keeps what RAM holds of it."""
        act = self.active
        if act.valid > MIN_SAVE:
            self.entries.append(act.branch(act.valid))
        self.active = Entry()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _covered(self, e: Entry, n: int) -> bool:
        """Whether another entry holds e's first n positions and a checkpoint there."""
        return any(f is not e and n in f.checkpoints and f.valid >= n and common_prefix(f.tokens, e.tokens) >= n
                   for f in (*self.entries, self.active))

    def _add(self, e: Entry, cp: Checkpoint) -> None:
        """A checkpoint of entry e (beyond ``checkpoints`` of them: its oldest but the lowest go)."""
        e.checkpoints[cp.position] = cp
        while len(e.checkpoints) > max(self.max_checkpoints, 1):
            low = min(e.checkpoints)
            del e.checkpoints[next(p for p in e.checkpoints if p != low)]

    def _fit(self) -> None:
        """Within ``ram`` bytes: the least recently used entries cut back to their first checkpoint, then dropped; then
        the device's records past what they hold, its own checkpoints, oldest first (its lowest last). (``ram`` 0:
        nothing saved, the device's checkpoints by their count alone.)"""
        if self.ram <= 0:
            return
        while self.nbytes() > self.ram and self.entries:
            e = self.entries.pop(0)
            low = min(e.checkpoints, default=0)
            if self.npu.recurrent and len(e.checkpoints) > 1 and low and not self._covered(e, low):
                e.cut(low)
                e.segments = _cut(e.segments, low)
                self.entries.append(e)
                log.info("prompt cache: a sequence cut back to its first %d tokens", low)
            else:
                log.info("prompt cache: a %d-token sequence dropped", len(e.tokens))
        act = self.active
        if self.nbytes() > self.ram:
            act.segments = _cut(act.segments, act.valid)
        while self.nbytes() > self.ram and len(act.checkpoints) > 1:
            low = min(act.checkpoints)
            del act.checkpoints[next(p for p in act.checkpoints if p != low)]
