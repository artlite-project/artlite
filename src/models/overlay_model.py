"""A model on a decode overlay with a chunk prefill (``OverlayModel``): its steps, prompts in prefill chunks and decode
steps, the output rows of prompts (``Readout``) and the packed chunks of several inputs, and the accounting of its
dispatches. Its generation in rounds of drafts: ``speculative.SpeculativeMixin``; its lanes, where its overlay holds
several sequences: ``lanes``."""

import contextlib
import copy
import dataclasses
import functools
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypeVar

import numpy as np

from artlite.decode.model import HostRecords, nbytes_of
from artlite.decode.screen import Screened, ScreenedRow

from .lanes import Request, Rounds, alone, kept
from .npu_model import Checkpoint, NpuModel
from .sampling import Penalties, top_logits
from .sequence import Sequence as Tokens
from .speculative import SpeculativeMixin

R = TypeVar("R")


class Readout:
    """The output rows of a prompt's tokens at indices ``at`` (in order, repeats allowed), gathered as its chunks and
    steps run (``OverlayModel.prompt``)."""

    def __init__(self, at: np.ndarray, hidden: int):
        self.at = at
        self.rows = np.zeros((len(at), hidden), np.float32)

    def take(self, i: int, n: int, rows: Callable[[int, int], np.ndarray]) -> None:
        """Tokens i .. i + n - 1 ran: ``rows(j, k)`` (k, hidden) are the outputs of k of them from their j-th on."""
        sel = np.flatnonzero((self.at >= i) & (self.at < i + n))
        if sel.size:
            lo, hi = int(self.at[sel].min()) - i, int(self.at[sel].max()) - i + 1
            self.rows[sel] = rows(lo, hi - lo)[self.at[sel] - i - lo]


class OverlayModel(SpeculativeMixin, NpuModel):
    """A model on its decode overlay (``overlay``: an ``artlite.decode.model.ModelOverlay``) with a chunk
    prefill (``prefill``: an ``artlite.prefill.chunks.ChunkPrefill``, or None: prompts as decode steps).

    ``steps`` runs up to ``overlay.SLOTS`` tokens per dispatch; ``prompt`` runs a prompt's tokens in prefill chunks
    where one can start (a whole cache group: the prefill's ``plan``), as decode steps before them and after, each
    chunk's inputs prepared on a worker thread while the previous chunk runs; ``forward`` is a prompt with its last
    token's logits (a decode step's, or of a chunk's last token: ``_chunk_logits``). A model supplies a step's inputs
    (``_step_inputs``: the overlay step's arguments before the position), a chunk's (``_chunk_inputs``: the prefill
    run's), the logits of a chunk's last token and the soft cap of its final logits (``softcap``).

    ``generate`` (``SpeculativeMixin``) runs a generation in rounds of a step of the sampled token and its drafts.

    Lanes (a model whose overlay holds several sequences, ``_open_lanes``; module ``lanes``): ``lanes[s]`` is the model
    for sequence s, lane 0 the model object itself and the others shallow copies of it: one device, prefill and set of
    options (``LANE_SHARED``), a sequence's state, counters, drafter calibration and planner each (``LANE_OWN``).
    Each lane is driven by one thread at a time. Its generation's rounds share their dispatches with the other lanes'
    (``_joined``); a lane's other calls on the device (a prompt, the state copies, single steps) hold it alone
    (``lanes.alone``): a prompt lets the others in between its chunks, and the lanes' round between a chunk's pieces
    (the prefill's ``guest``). A model whose round starts with dispatches of its own (a drafter's steps on the
    sequence's caches) runs them inside ``_before``, gives a joint step its per-slot keywords (``_step_kw``) and takes
    what else it reads of a step through ``_tapped`` / ``_took``. After a failed dispatch a lane's calls raise
    ``lanes.DeviceFailed`` until its ``recover``."""

    overlay: Any
    _w_out: np.ndarray  # the output norm's weights (hidden,) fp32 (a readout's norm: ``outputs``)
    slot = 0  # the slot of the last token whose logits a dispatch returned (a step's last; 0 after a chunk's)
    softcap = 0.0  # the model's logits of the overlay's l: softcap tanh(l / softcap) (0: l)
    seq = 0  # the lane's sequence on the overlay's device
    last_step = False  # the prompt's last token as a decode step (a prefill without the LM head: no ``_chunk_logits``)
    # a block drafter's (one dispatch a round whatever number of its drafts find a slot: DSpark, DFlash2): how many
    # times what the block costs the lanes of a shared round its drafts must be kept (``_block_pays``). Measured on
    # Spark-X with two lanes and a block every round, where the two are about equal (1.0 - 1.2): -1.5% (1.7B) to +7%
    # (4B) against no block; with four lanes (0.3 - 0.5): -30%. None: a drafter whose cost follows its drafts
    BLOCK_MARGIN: ClassVar[float | None] = None
    _lanes: "tuple[OverlayModel, ...] | None" = None  # (``lanes``; None: the model alone)
    _rounds: Rounds | None = None  # several lanes: what joins their rounds and gives one the device (``lanes.Rounds``)
    _round: Request | None = None  # (the lane's last round, where it asked ``_rounds`` for it: ``_overhead``)
    # a model's attributes by what its lanes do with them (``_open_lanes`` refuses one on neither list): one object or
    # value for all of them (the device's side, the model's options) ...
    LANE_SHARED: ClassVar[tuple[str, ...]] = ("overlay", "prefill", "cfg", "eos_token_ids", "generates", "_w_out", "_pool",
                                              "_lanes", "_rounds", "DRAFTS", "lookup", "context", "softcap", "drafter")
    # ... or each lane's own, from the class's defaults on (a sequence's state, its generations' counters, the drafter's
    # calibration, the planner)
    LANE_OWN: ClassVar[tuple[str, ...]] = (
        "seq", "position", "slot", "carry", "rounds", "_rounds_of", "_round", "_offer", "planner", "t_npu", "t_host",
        "dispatches",
        "drafted", "accepted", "drafted_ctx", "accepted_ctx", "forced", "blocks", "t_draft", "_t_row", "_kept_at",
        "_chain_at", "_chains")

    def cap(self, logits: np.ndarray | Screened | ScreenedRow) -> np.ndarray:
        """The model's logits (fp32) of the overlay's (bf16 or fp32): the soft cap, if any. A screened head's
        (``artlite.decode.screen``): its largest logits exact, the others screened (``dense``; every one exact: the
        logits as an array first)."""
        if isinstance(logits, (Screened, ScreenedRow)):
            logits = logits.dense()
        logits = np.asarray(logits, np.float32)
        return np.tanh(logits / self.softcap) * np.float32(self.softcap) if self.softcap else logits

    @functools.cached_property
    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(1)  # (a prompt chunk's inputs while the previous chunk runs)

    def _open(self, *, threads: int) -> None:
        """Load the overlay: its weights pack (``threads``) while the prefill's designs compile, then the drafters' (``_load_
        drafters``), the steps' transactions (``_prepare_steps``), the prefill's buffers (``bind``: after the overlay's
        device is loaded) and the wait for the weights' thread (``ready``)."""
        with ThreadPoolExecutor(1) as pool:
            build = pool.submit(self.prefill.compile) if self.prefill is not None else None  # (while the weights are packed)
            self.overlay.load(threads=threads, also=self._also())
            self._load_drafters(threads)
            self._prepare_steps()
            if build is not None:
                build.result()  # (its error, if any)
                self.prefill.bind()
        self.overlay.ready()  # (the weights' thread packs meanwhile)

    @property
    def lanes(self) -> "tuple[OverlayModel, ...]":
        """The model per sequence of its overlay (class docstring): lane 0 is the model object."""
        return self._lanes or (self,)

    @property
    def failed(self) -> bool:  # type: ignore[override]
        """(several lanes: a dispatch failed, under any lane's work, and this lane has not started over)"""
        return self._rounds is not None and self._rounds.down(self.seq)

    def device(self) -> contextlib.AbstractContextManager[None]:
        """(several lanes: ``lanes.Rounds.alone``, the lane's sequence selected)"""
        return contextlib.nullcontext() if self._rounds is None else self._rounds.alone(self.seq)

    def _open_lanes(self) -> None:
        """The lanes of a model whose overlay holds several sequences (``lanes``), once it is loaded and its
        attributes are set: copies of the model, each attribute shared or a lane's own (``LANE_SHARED``, ``LANE_OWN``;
        TypeError: one that is on neither list)."""
        if self.overlay.sequences == 1:
            return
        self._pool  # noqa: B018 (one for the lanes: made before the copies)
        self._rounds = Rounds(self.overlay, self._tapped)
        if self.prefill is not None:  # (the lanes' steps between a chunk's pieces)
            self.prefill.guest = self._rounds
        if unknown := sorted(set(vars(self)) - set(self.LANE_SHARED) - set(self.LANE_OWN)):
            raise TypeError(f"{type(self).__name__}: several sequences, and {', '.join(unknown)} neither shared by its "
                            f"lanes nor a lane's own (LANE_SHARED, LANE_OWN)")
        lanes = [self]
        for s in range(1, self.overlay.sequences):
            lane = copy.copy(self)
            for name in self.LANE_OWN:  # (the class's: counters of none, a planner made at its first use)
                vars(lane).pop(name, None)
            lane.seq, lane.position = s, 0
            lanes.append(lane)
        for lane in lanes:
            lane._lanes = tuple(lanes)

    @property
    def budget(self) -> int | None:  # type: ignore[override]
        """(several lanes: the slots a round of the lane can get, ``lanes.Rounds.budget``)"""
        return None if self._rounds is None else self._rounds.budget(self.seq)

    @property
    def share(self) -> int | None:  # type: ignore[override]
        """(several lanes: ``lanes.Rounds.share``; a block drafter's: no slot for its guesses in a round whose block
        does not pay, ``_block_pays``)"""
        if self._rounds is None:
            return None
        s = self._rounds.share(self.seq)
        return 1 if s > 1 and self.BLOCK_MARGIN is not None and not self._block_pays(s - 1) else s

    def _block_pays(self, k: int) -> bool:
        """(a block drafter's: ``BLOCK_MARGIN``) Whether a block for k drafts pays in the lane's next round. Its
        dispatch is one of the device's for every lane that asks: n lanes that draft make a round's time step + n
        blocks for n (1 + g) tokens, g the drafts a block gets kept, so a block pays where g step > n block (alone:
        about always; Spark-X 1.7B with 2 lanes: 3 drafts, 1.1 kept, for a block of half a step: not; 4 lanes: a
        draft each, never). From the lanes' measured rounds (``_kept_at``, a block's seconds ``_t_row``); before
        any: its first blocks measure."""
        r = self._rounds
        n = r.asking(self.seq)
        if n == 1:
            return True
        cal = next((x for x in (self, *self.lanes) if x._t_row and x._kept_at), None)
        if cal is None or not r.step_s:
            return True
        return sum(cal._kept_at[:k]) * r.step_s > self.BLOCK_MARGIN * n * cal._t_row

    @property
    def _drafting(self) -> bool:
        """(several lanes: not while another lane has the device alone, a prompt: ``_before`` would refuse)"""
        return super()._drafting and (self._rounds is None or not self._rounds.held())

    def _before(self) -> contextlib.AbstractContextManager[bool]:
        """A context for a round's dispatches of the lane's own before its step (a drafter's): True inside where they
        may run (several lanes: with the device, ``lanes.Rounds.before``), False: the round goes without them."""
        return contextlib.nullcontext(True) if self._rounds is None else self._rounds.before(self.seq)

    def _step_kw(self, tokens: list[int]) -> dict[str, Any]:
        """The overlay step's keywords that are per slot, for the tokens (None: not for these): a joint step takes one
        where each of its lanes has it."""
        return {}

    def _tapped(self, S: int) -> tuple:
        """What else than the logits the model reads of a step of S slots, right after it: arrays of a row per slot
        (copies) or None (``lanes.Rounds``' ``tap``: whichever lane's thread ran the step)."""
        return ()

    def _took(self, tokens: list[int], taken: tuple) -> None:
        """The lane's rows of ``_tapped`` after its round's step of ``tokens``."""

    def _total(self, name: str) -> float:
        """(of every lane)"""
        return sum(getattr(lane, name) for lane in self.lanes)

    def idle(self) -> None:
        """(the other lanes' rounds no longer wait for this lane's, until its next one)"""
        if self._rounds is not None:
            self._rounds.idle(self.seq)

    def _abandon(self) -> None:
        super()._abandon()
        self.idle()

    def recover(self) -> None:
        """(a lane: its own sequence, its calls run again; the other lanes' once they recover)"""
        if self._rounds is not None:
            self._rounds.recover(self.seq)
        self.reset()

    def _also(self) -> dict[str, int]:
        """The buffers on the overlay's device besides its own, allocated with them (``ModelOverlay.load``): the
        prefill's."""
        return dict(self.prefill.own) if self.prefill is not None else {}

    def _load_drafters(self, threads: int) -> None:
        """The drafters' weights and static objects on the loaded overlay's device."""

    def _prepare_steps(self) -> None:
        """The steps' transactions before the first prompt: of S = 1 .. SLOTS tokens (a prompt's, its last token's and a
        generated one's), without and with logits."""
        for S in range(1, self.overlay.SLOTS + 1) if self.causal else ():
            self.overlay.prepare(S, logits=False)
            self.overlay.prepare(S, logits=True)

    @alone
    def prepare(self, prompt_len: int) -> None:
        """(the overlay's and the prefill's hardware contexts, the prefill's piece BOs and its chunks' items: else the
        first prompt of a process pays them, ~50 ms, most of it the first context waking the NPU)"""
        self.overlay.dev.open()
        if self.prefill is not None:
            self.prefill.prepare(self.position, prompt_len)

    def _step_inputs(self, tokens: list[int]) -> tuple:
        """The overlay step's arguments before the position for the tokens (their embeddings, ...)."""
        raise NotImplementedError

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """The prefill run's arguments before the position for a chunk's tokens (``first``: the prompt's first chunk,
        whose inputs nothing prepared ahead)."""
        raise NotImplementedError

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """The overlay's logits (vocab,) of the last chunk's last token (``token``, the chunk's n-th; positions up to
        it cached), in the dispatches of ``_dispatch``."""
        raise NotImplementedError

    def _dispatch(self, run: Callable[[], R]) -> R:
        """A dispatch (``run``), timed and counted."""
        t0 = time.perf_counter()
        out = run()
        self.t_npu += time.perf_counter() - t0
        self.dispatches += 1
        return out

    def _step_inputs_of(self, tokens: list[int]) -> tuple:
        """``_step_inputs`` of a step that fits (ValueError), timed."""
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        if not self.causal:
            raise ValueError("a non-causal model reads whole prompts (readouts), it runs no decode steps")
        if len(tokens) > self.overlay.SLOTS:
            raise ValueError(f"a step of {len(tokens)} tokens: the overlay runs {self.overlay.SLOTS} at most")
        t0 = time.perf_counter()
        args = self._step_inputs(tokens)
        self.t_host += time.perf_counter() - t0
        return args

    @alone
    def steps(self, tokens: list[int], *, logits: bool = True, **step: Any) -> np.ndarray | None:
        """Run up to ``overlay.SLOTS`` tokens at the next positions in one dispatch: the overlay's logits (n, vocab)
        bf16 after each of them (a view of the NPU's buffer, valid until the next dispatch; a model of several lanes:
        copies), or None (``logits=False``: prompt tokens); ``step``: the overlay step's other keywords."""
        args = self._step_inputs_of(tokens)
        out = self._dispatch(lambda: self.overlay.step(*args, self.position, logits=logits, **step))
        self.position += len(tokens)
        if logits:
            self.slot = len(tokens) - 1
        return out if out is None or self._rounds is None else kept(out)  # (another lane's dispatch may come next)

    def _joined(self, tokens: list[int]) -> np.ndarray | Screened:
        """A generation round's step of the tokens at the next positions (``SpeculativeMixin._verify``): ``steps``; a
        model of several lanes: in one dispatch with the other lanes' rounds (``lanes.Rounds.join``: it waits for
        them), its rows copies, valid until its next round: of the first of the tokens where the step had no room for
        the drafts the round offered it (``_offer``). The inputs are made and checked first: their errors are this
        lane's alone."""
        rounds = self._rounds
        if rounds is None:
            return self.steps(tokens)
        offer, self._offer = self._offer or (), None  # (this round's)
        if rounds.mine():  # (the caller holds the device: no round could run)
            self._round = None
            return self.steps(tokens)
        args = self._step_inputs_of(tokens)
        r = self._round = rounds.join(self.seq, args, self.position, len(tokens) - len(offer), offer,
                                      self._step_kw(tokens))
        self.t_npu += r.wall
        self.dispatches += 1
        self.position += r.n
        self.slot = r.n - 1
        self._took(tokens[:r.n], r.taken)
        return r.rows

    def _overhead(self, wall: float) -> float | None:
        """(a round that asked the lanes' ``Rounds`` for its step: what it waited for the round and for the device;
        None where the step ran other lanes' slots too)"""
        r = self._round
        if r is None:
            return super()._overhead(wall)
        return wall - r.seconds if r.parts == 1 else None

    def _generating(self) -> None:
        """NotImplementedError for a readout model (``generates``)."""
        if not self.generates:
            raise NotImplementedError(f"{type(self).__name__}: a readout model's (embeddings, rerank, decisions: no LM "
                                      f"head on the NPU), no text generation: its rows through outputs / readouts")

    def step(self, token: int, *, logits: bool = True) -> np.ndarray | None:
        """Run ``token`` at the next position: the logits (vocab,) fp32 of the next token, or None (``logits=False``)."""
        if logits:
            self._generating()
        out = self.steps([token], logits=logits)
        return None if out is None else self.cap(out[0])

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(from the overlay's bf16 logits: the soft cap on the candidates alone)"""
        self._generating()
        raw = self.steps([token])[0]
        return top_logits(raw, k, self.softcap) if penalties is None else penalties.top(raw, k, self.softcap)

    def stale(self) -> None:
        """Another hardware context ran on the device (a prefill chunk, an image encoder): the overlay's state on the
        array comes back before its next dispatch (``OverlayDevice.set_prologue``)."""
        self.overlay.dev.stale = True

    def _prefill_run(self, args: tuple, T: int, n: int, sets: Sequence[int] | None = None) -> None:
        """A prompt chunk of n tokens at the next position through the prefill's chunk of T (its inputs ``args``;
        ``sets``: a non-causal prompt's, ``outputs``)."""
        self.prefill.run(*args, self.position, T, **({"sets": sets} if sets is not None else {}))
        self.stale()

    def prompt_seconds(self, n: int, position: int = 0) -> float | None:
        """(its prefill chunks' estimates, ``ChunkPrefill.chunk_seconds``; the tokens left for decode steps not
        counted)"""
        if self.prefill is None:
            return None
        if not self.causal:
            T = self._chunk(n, position)
            return self.prefill.chunk_seconds(T, position) if T else None
        p, out = position, 0.0
        for T in self.prefill.plan(position, n, steps=False):
            out += self.prefill.chunk_seconds(T, p)
            p += T
        return out

    def _chunk(self, left: int, position: int | None = None, *, steps: bool = True) -> int:
        """The size of the next prefill chunk at ``position`` (default: the next) when ``left`` tokens remain (0: the
        next token runs as a decode step; ``steps`` False: a readout, as few tokens as possible as decode steps)."""
        p = self.position if position is None else position
        if not self.causal:  # (the whole prompt: the smallest size that holds it)
            ok = self.prefill is not None and left >= self.prefill.MIN_TOKENS
            return next((T for T in self.prefill.sizes if T >= left and self.prefill.fits(T, p)), 0) if ok else 0
        plan = self.prefill.plan(p, left, steps=steps) if self.prefill is not None else []
        return plan[0] if plan else 0

    @property
    def group(self) -> int:
        """(a whole cache group of the prefill, or 8 positions without one: a step at a multiple of 8 opens a new V
        record, so the ring holds what it reads; a prompt from a group runs in chunks at once, while one inside a
        group runs as decode steps up to the next)"""
        return self.prefill.group if self.prefill is not None else 8

    @alone
    def rewind(self, n: int) -> int:
        """(down to a multiple of ``group``; a recurrent model's: all cached positions or none, its states hold the whole
        sequence and cannot go back: a ``checkpoint`` keeps a position)"""
        self._abandon()
        if self.recurrent:
            if n < self.position:
                self.reset()
            return self.position
        keep = min(n, self.position)
        self.position = keep - keep % self.group
        return self.position

    def _overlays(self) -> list[Any]:
        """The overlays that hold the sequence's caches and states (the model's, a drafter's)."""
        return [self.overlay]

    def _state(self) -> dict[str, Any]:
        """The model's own part of a checkpoint (its overlays' are theirs: ``ModelOverlay.snapshot``)."""
        return {}

    def _set_state(self, state: dict[str, Any]) -> None:
        """Back to ``_state``'s."""

    @alone
    def checkpoint(self) -> Checkpoint:
        self._abandon()
        state = ([ov.snapshot(self.position) for ov in self._overlays()], self._state())
        return Checkpoint(self.position, state, nbytes_of(state))

    @alone
    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        self._abandon()
        if checkpoint is None:
            if position % self.group or (self.recurrent and position):
                raise ValueError(f"position {position}: a restore without a checkpoint keeps a multiple of "
                                 f"{self.group}{' (0: the model is recurrent)' if self.recurrent else ''}")
            if self.recurrent:
                self.reset()
            self.position = position
            return
        assert checkpoint.position == position, (checkpoint.position, position)
        snapshots, state = checkpoint.state
        for ov, snap in zip(self._overlays(), snapshots, strict=True):
            ov.restore(snap)
        self._set_state(state)
        self.position = position

    @alone
    def read_records(self, start: int, end: int) -> list[HostRecords]:
        return [ov.read_records(start, end) for ov in self._overlays()]

    @alone
    def write_records(self, records: list[HostRecords], start: int, end: int) -> None:
        for ov, r in zip(self._overlays(), records, strict=True):
            ov.write_records(r, start, end)

    @alone
    def prompt(self, tokens: list[int], *, logits: bool = False, progress: Callable[[int], None] | None = None,
               readout: Readout | None = None, sets: Sequence[int] | None = None) -> np.ndarray | None:
        """Run tokens after the cached positions (prefill chunks, else decode steps); ``logits``: return the last
        token's (vocab,) fp32; ``progress``: as ``forward``'s, after each chunk and each group of steps; ``readout``:
        gathers output rows (``outputs``: every token a chunk can take in chunks); ``sets``: a non-causal prompt's
        (``outputs``)."""
        if logits:
            self._generating()
        self._abandon()
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        if sets is not None and (self.causal or len(sets) < 1 or sets[-1] != len(tokens) or list(sets) != sorted(sets)
                                 or sets[0] < 1):
            raise ValueError(f"sets {list(sets)}: a non-causal prompt's increasing bounds, the last its length")
        if not self.causal and (self.position or not self._chunk(len(tokens))):
            sizes = self.prefill.sizes if self.prefill is not None else (0,)
            raise ValueError(f"a non-causal model reads a whole prompt in one chunk from position 0 "
                             f"({self.prefill.MIN_TOKENS if self.prefill else 1} to {sizes[-1]} tokens): "
                             f"{len(tokens)} tokens at position {self.position}")
        steps = readout is None
        out, i, nxt = None, 0, None
        end = len(tokens) - (1 if steps and self.last_step else 0)  # (the tokens a chunk may take)
        while i < len(tokens):
            T = self._chunk(end - i, steps=steps) if end > i else 0
            if not T:  # (decode steps up to the next position where a chunk can start)
                n = 1
                while n < min(self.overlay.SLOTS, len(tokens) - i) and \
                        not (end > i + n and self._chunk(end - i - n, self.position + n, steps=steps)):
                    n += 1
                last = logits and i + n == len(tokens)
                o = self.steps(tokens[i:i + n], logits=last or not steps)  # (a readout's rows: the LM head's input)
                out = self.cap(o[-1]) if last else None
                if readout is not None:
                    readout.take(i, n, lambda j, k: np.stack([self.overlay.hidden(-1, s) for s in range(j, j + k)]))
                i += n
                if progress is not None:
                    progress(i)
                continue
            n = min(T, end - i)
            t0 = time.perf_counter()
            args = nxt.result() if nxt is not None else self._chunk_inputs(tokens[i:i + n], first=True)
            t1 = time.perf_counter()
            m = self._chunk(end - i - n, self.position + n, steps=steps) if end > i + n else 0  # (the next chunk: its
            # inputs meanwhile)
            nxt = self._pool.submit(lambda t: self._chunk_inputs(t, first=False),
                                    tokens[i + n:min(i + n + m, end)]) if m else None
            self._prefill_run(args, T, n, sets)
            self.t_host += t1 - t0
            self.t_npu += time.perf_counter() - t1
            self.dispatches += self.prefill.dispatches
            if readout is not None:
                readout.take(i, n, self.prefill.hidden)
            if logits and i + n == len(tokens):
                out = self.cap(self._chunk_logits(tokens[-1], n))
                self.slot = 0
            self.position += n
            i += n
            if progress is not None:
                progress(i)
            if self._rounds is not None and i < len(tokens):  # (the other lanes' turn, between two chunks)
                self._rounds.breathe(self.seq)
        return out

    def forward(self, tokens: list[int], *, progress: Callable[[int], None] | None = None) -> np.ndarray:
        """Run the tokens after the cached positions: the logits (vocab,) of the last."""
        return self.prompt(tokens, logits=True, progress=progress)

    def outputs(self, tokens: list[int] | Tokens, at: Sequence[int] | None = None, *, norm: bool = True,
                progress: Callable[[int], None] | None = None, sets: Sequence[int] | None = None) -> np.ndarray:
        """(every token a chunk can take in chunks, the last one padded: a chunk's rows need no LM head, which a decode
        step's do; the output norm on the host: ``_w_out``)"""
        n = len(tokens)
        idx = np.arange(n) if at is None else np.asarray(at, np.int64).reshape(-1)
        if n == 0 or ((idx < -n) | (idx >= n)).any():
            raise IndexError(f"readout rows {idx.tolist()} of {n} tokens")
        readout = Readout(idx % n, self.cfg.hidden_size)
        self.prompt(tokens, progress=progress, readout=readout, sets=sets)
        return self._normed(readout.rows) if norm else readout.rows

    def _normed(self, x: np.ndarray) -> np.ndarray:
        """Output rows (n, hidden) through the output norm (the LM head's input), in place."""
        x *= np.asarray(self._w_out, np.float32) / np.sqrt((x * x).mean(-1, keepdims=True) + self.cfg.rms_norm_eps)
        return x

    @property
    def packs(self) -> bool:
        """(the prefill runs packed chunks: a readout model's)"""
        return self.prefill is not None and bool(self.prefill.pack_sizes)

    @property
    def lead_tokens(self) -> int:  # type: ignore[override]
        """(the prefill's packed chunks' lead: ``ChunkPrefill.lead``)"""
        return self.prefill.lead if self.packs else 0

    def readout_seconds(self, lengths: list[int], position: int = 0) -> float | None:
        """(the packing's estimate, ``ChunkPrefill.pack``)"""
        if self.packs and self.prefill.packed_sizes(position):
            return self.prefill.pack(lengths, position).seconds
        return super().readout_seconds(lengths, position)

    @alone
    def readouts(self, inputs: list[list[int] | Tokens], at: list[np.ndarray], *, norm: bool = True,
                 lead: list[int] | None = None, sets: list[Sequence[int] | None] | None = None) -> list[np.ndarray]:
        """(the prefill's packed chunks where its plan packs inputs, ``ChunkPrefill.pack``: every one of them continues
        the cached prefix; a recurrent model's after a prefix are tree chunks, which leave its states as the prefix left
        them; inputs alone one after another from the prefix, a checkpoint's where needed; a lead of ``lead_tokens``
        inside the first packed chunk where it has room and no input runs alone, else alone first)"""
        if not self.packs or sets is not None:
            return super().readouts(inputs, at, norm=norm, lead=lead, sets=sets)
        self._abandon()
        at = [np.asarray(r, np.int64).reshape(-1) % len(ids) for ids, r in zip(inputs, at, strict=True)]
        lead = list(lead or [])
        p = self.position + len(lead)  # (the prefix's end: the inputs' history)
        lengths = [len(ids) for ids in inputs]
        # (a lead planned as one more input: its chunk first, the lead its first segment, the inputs alone after the
        # packed chunks; a lead alone runs first)
        n, inside = len(inputs), False
        if lead and len(lead) == self.lead_tokens:
            plan = self.prefill.pack([*lengths, len(lead)], p)
            j = next((k for k, (_, b) in enumerate(plan.bins) if n in b), None)
            if j is not None:
                (T0, b0), rest = plan.bins[j], plan.bins[:j] + plan.bins[j + 1:]
                plan = dataclasses.replace(plan, bins=[(T0, [i for i in b0 if i != n]), *rest])
                inside = True
            else:
                plan = dataclasses.replace(plan, alone=[i for i in plan.alone if i != n])
        else:
            plan = self.prefill.pack(lengths, p)
        if lead and not inside:
            self.outputs(lead, [-1])
        out: list[np.ndarray] = [np.zeros(0)] * len(inputs)

        def run_alone() -> None:
            """The inputs alone, one after another from the prefix (a checkpoint's where needed), back at it after."""
            cp = self.checkpoint() if p and (self.recurrent or p % self.group) else None
            for k, i in enumerate(plan.alone):
                if k:
                    self.restore(p, cp)
                out[i] = self.outputs(inputs[i], at[i], norm=norm)
            self.restore(p, cp)  # (back at the prefix, which the packed chunks continue)

        if plan.alone and not inside:
            run_alone()
        bins, u = plan.bins, self.prefill.seg_unit(p)

        def chunk_of(idx: list[int], lead: list[int]) -> tuple:
            """A packed chunk's inputs, positions (per token; (T, H, W) where an input is a ``sequence.Sequence``),
            segment starts (-1: the padding tokens between two segments, up to the next multiple of ``seg_unit``: the
            prefill zeroes their rows) and the segments' offsets (``lead``: its first tokens, a segment from p -
            len(lead))."""
            toks: list[int] | Tokens = list(lead)
            rope = any(isinstance(inputs[i], Tokens) for i in idx)
            text = (lambda n, p0: p0 + np.repeat(np.arange(n)[:, None], 3, 1)) if rope else \
                (lambda n, p0: p0 + np.arange(n))
            pos, lo, offs = [text(len(lead), p - len(lead))], [np.zeros(len(lead), np.int64)], []
            for k, i in enumerate(idx):
                ids = inputs[i]
                gap = -len(ids) % u if k + 1 < len(idx) else 0
                offs.append(len(toks))
                toks = toks + ids
                for _ in range(gap):
                    toks = toks + ids[-1:]
                pos += [p + ids.pos if isinstance(ids, Tokens) else text(len(ids), p),
                        np.full((gap, 3) if rope else gap, p)]
                lo += [np.full(len(ids), offs[-1]), np.full(gap, -1)]
            return self._chunk_inputs(toks, first=False), np.concatenate(pos), np.concatenate(lo), offs

        nxt = self._pool.submit(chunk_of, bins[0][1], lead if inside else []) if bins else None
        for k, (T, idx) in enumerate(bins):
            t0 = time.perf_counter()
            args, pos, lo, offs = nxt.result()
            t1 = time.perf_counter()
            nxt = self._pool.submit(chunk_of, bins[k + 1][1], []) if k + 1 < len(bins) else None  # (meanwhile)
            if k == 0 and inside:  # (the lead's records a sequence: the prefix then ends at p)
                self.prefill.run_packed(*args, p - len(lead), pos, lo, T, lead=len(lead))
                self.stale()
                self.restore(p)
            else:
                self._packed_run(args, T, pos, lo)
            self.t_host += t1 - t0
            self.t_npu += time.perf_counter() - t1
            self.dispatches += self.prefill.dispatches
            for i, o in zip(idx, offs, strict=True):
                r0, r1 = int(at[i].min()), int(at[i].max()) + 1
                x = self.prefill.hidden(int(o) + r0, r1 - r0)[at[i] - r0]
                out[i] = self._normed(x) if norm else x
        if bins and self.recurrent and not p:
            self.restore(0)  # (the packed chunks' states: none of a sequence)
        if plan.alone and inside:
            run_alone()
        return out

    def _packed_run(self, args: tuple, T: int, pos: np.ndarray, lo: np.ndarray) -> None:
        """A packed chunk of size T at the next position (``ChunkPrefill.run_packed``'s: its inputs ``args``, its tokens'
        positions ``pos``, segment starts ``lo``)."""
        self.prefill.run_packed(*args, self.position, pos, lo, T)
        self.stale()

    def forced_logits(self, tokens: list[int], first: int = 0, *, batch: int = 1) -> np.ndarray:
        """Teacher-forced logits (len(tokens) - first, vocab) fp32 of tokens[first:], from position 0, ``batch`` tokens
        per dispatch (quality tooling)."""
        self._generating()
        self.reset()
        self.prompt(tokens[:first])
        # (a screened head's rows as arrays: every logit exact)
        return np.concatenate([self.cap(np.asarray(self.steps(tokens[i:i + batch])))
                               for i in range(first, len(tokens), batch)])
