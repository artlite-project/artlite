"""A model on the NPU as the tools and the server drive it (``NpuModel``: a prompt's logits, steps, the generation loop,
a sequence's state as host copies: ``Checkpoint`` and the KV records) and the default KV cache capacity."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .sampling import LogitsLike, Mask, Penalties, Probs, Sampling, top_logits
from .sequence import Sequence as Tokens

if TYPE_CHECKING:
    from artlite.decode.model import HostRecords


def auto_capacity(tokens: int) -> int:
    """KV cache positions for a session of ``tokens`` (prompt and generated): a power of two from 4096 (one decode
    design per size), with room for a prefill's padded last chunk."""
    need, cap = max(tokens, 1) + 256, 4096
    while cap < need:
        cap *= 2
    return cap


@dataclass(eq=False)
class Checkpoint:
    """A sequence's state at ``position`` as host copies (``NpuModel.checkpoint``): what continuing from there needs
    besides the KV caches' records of the positions before it; ``nbytes`` of host memory."""

    position: int
    state: Any
    nbytes: int


class NpuModel:
    """A model on the NPU. ``forward`` runs a prompt after the cached positions and returns its last token's logits,
    ``step`` / ``top_k`` run one token; ``generate`` samples from them. ``t_npu``: seconds in dispatches, ``t_host``:
    host work on their inputs, ``dispatches``: their count.

    A sequence's state as host copies (a prompt cache's): ``checkpoint`` / ``restore`` (the state at a position besides
    the KV records), ``read_records`` / ``write_records`` (the records of positions)."""

    cfg: Any  # the model file's configuration (``artlite.model_config``: ``hidden_size``, ``rms_norm_eps``, ...)
    capacity: int  # KV cache positions
    position: int  # cached positions
    eos_token_ids: tuple[int, ...] | set[int]
    prefill: Any = None  # the chunk prefill (an ``artlite.prefill.chunks.ChunkPrefill``), or None: prompts as steps
    t_npu = t_host = 0.0
    dispatches = 0
    drafted = accepted = 0  # draft tokens run and accepted (of every drafter; 0: a model without drafts)
    recurrent = False  # positions are kept only where a checkpoint was taken (states that hold the whole sequence)
    group = 1  # ``rewind`` and a ``restore`` without a checkpoint land on its multiples
    causal = True  # False: a prompt's every token sees all of them (vllm-d3's decision models): one chunk from position
    # 0, no decode steps (``readouts`` only)
    DRAFTS = 0  # draft tokens per step at most (0: steps of one token)

    def prepare(self, prompt_len: int) -> None:
        """Compile and open what a prompt of ``prompt_len`` tokens needs (before it is timed)."""

    def describe(self) -> str:
        """What the model holds in memory (weights, caches)."""
        raise NotImplementedError

    def stats(self) -> str:
        """More of its generation's statistics for the report (e.g. draft tokens), or ""."""
        return ""

    def reset(self) -> None:
        """Forget the cached positions: a new sequence."""
        raise NotImplementedError

    def rewind(self, n: int) -> int:
        """Keep at most the first ``n`` cached positions (a prompt cache: a new prompt continues from the prefix it
        shares with the cached tokens) and return how many it kept; this default keeps none (``reset``)."""
        self.reset()
        return 0

    def forward(self, tokens: list[int], *, progress: Callable[[int], None] | None = None) -> np.ndarray:
        """Run tokens after the cached positions: the logits (vocab,) fp32 of the last. ``progress``: called with the
        number of tokens done as they run (an exception it raises stops the prompt between dispatches, the positions
        run so far cached)."""
        raise NotImplementedError

    def prompt(self, tokens: list[int], *, logits: bool = False, progress: Callable[[int], None] | None = None
               ) -> np.ndarray | None:
        """Run tokens after the cached positions (``forward``); ``logits``: return the last one's."""
        out = self.forward(tokens, progress=progress)
        return out if logits else None

    def outputs(self, tokens: list[int] | Tokens, at: Sequence[int] | None = None, *, norm: bool = True,
                progress: Callable[[int], None] | None = None, sets: Sequence[int] | None = None) -> np.ndarray:
        """Run tokens (ids, or a ``sequence.Sequence``: images' rows among them) after the cached positions
        (``prompt``) and return the last layer's outputs (len(at), hidden) fp32 of the tokens at indices ``at``
        (negative: from the end; None: every token), output-normed (``norm``: the LM head's input; else the residual
        stream): a readout's rows (embeddings, a classifier's or the LM head's rows on the host: ``models.readouts``).
        ``sets`` (a non-causal model whose prefill reads sets: LFM3's d1): the prompt as a state and questions after it,
        bounds (S, e1, .., en): the state's tokens 0 .. S - 1 see every token on the non-causal layers, question k's
        (e(k-1) .. ek - 1, e0 = S) the state's and their own; None: every token sees every token."""
        raise NotImplementedError(f"{type(self).__name__} has no readouts")

    packs = False  # whether ``readouts`` packs inputs into prefill chunks (a readout model's prefill)
    lead_tokens = 0  # the length of a ``readouts`` lead that may run inside its first packed chunk (0: none)

    def readouts(self, inputs: list[list[int] | Tokens], at: list[np.ndarray], *, norm: bool = True,
                 lead: list[int] | None = None, sets: list[Sequence[int] | None] | None = None) -> list[np.ndarray]:
        """Several inputs' output rows (each as ``outputs`` returns them): each input's tokens after the cached positions
        (a prefix every one of them continues; ``lead``: tokens of it not cached yet, run first; ``sets``: each one's
        as ``outputs`` takes them); the model is left at the prefix's end. This default runs them one after another
        from a checkpoint there."""
        if lead:
            self.outputs(lead, [-1])
        p = self.position
        cp = self.checkpoint() if p else None
        out = []
        for k, (ids, rows) in enumerate(zip(inputs, at, strict=True)):
            self.restore(p, cp)
            out.append(self.outputs(ids, rows, norm=norm, **({"sets": sets[k]} if sets is not None else {})))
        self.restore(p, cp)
        return out

    def readout_seconds(self, lengths: list[int], position: int = 0) -> float | None:
        """Estimated seconds of ``readouts`` of inputs of ``lengths`` tokens after ``position`` (None: no estimate); this
        default: each alone (``prompt_seconds``)."""
        t = [self.prompt_seconds(n, position) for n in lengths]
        return None if None in t else sum(t)

    def prompt_seconds(self, n: int, position: int = 0) -> float | None:
        """Estimated seconds of a prompt of n tokens from ``position`` (None: no estimate)."""
        return None

    def checkpoint(self) -> Checkpoint:
        """The state at the current position that continuing from it needs besides the KV caches' records of the
        positions before it (any position: inside a cache group, after a step whose drafts were rejected)."""
        raise NotImplementedError

    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        """Continue from ``position``: the caches hold the records of the positions before it (this sequence's, or
        ``write_records``'), ``checkpoint`` (taken there) the rest of the state; without one, a position a rewind
        lands on (a multiple of ``group``; of a recurrent model: 0, a new sequence)."""
        raise NotImplementedError

    def read_records(self, start: int, end: int) -> "list[HostRecords]":
        """Host copies of the KV caches' records of positions start .. end - 1 (whole groups; one per cache: the
        model's, a drafter's)."""
        raise NotImplementedError

    def write_records(self, records: "list[HostRecords]", start: int, end: int) -> None:
        """The caches' records of positions start .. end - 1 from ``read_records``' copies (which hold them)."""
        raise NotImplementedError

    def step(self, token: int) -> np.ndarray:
        """Run ``token`` at the next position: the logits (vocab,) of the next token."""
        raise NotImplementedError

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run ``token`` at the next position: the ``k`` largest logits of the next token (fp32, descending) and their
        token ids (``penalties``: of the penalized logits)."""
        logits = self.step(token)
        return top_logits(logits, k) if penalties is None else penalties.top(logits, k)

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None, probs: Probs | None = None, *, drafts: int | None = None) -> Iterator[int]:
        """Sample up to ``max_tokens`` tokens (< 0: no limit) after the cached positions; ``logits``: of the last
        cached one; ``prompt``: the prompt's tokens (the penalties' first); ``mask``: the tokens each may be (told
        each token as it is sampled); ``probs``: told each choice before its token is yielded (``logprobs``);
        ``drafts``: draft tokens per step at most in this generation, up to ``DRAFTS`` (None: ``DRAFTS``). Ends after
        an end-of-sequence token or when the cache is full."""
        if max_tokens == 0:
            return
        penalties = Penalties(sampling, prompt, mask)

        def choose(row: LogitsLike) -> int:
            cand = penalties.top(row, sampling.k)
            token = sampling.pick(*cand)
            if probs is not None:
                probs(row, token, 0.0, cand)
            penalties.accept(token)
            return token

        token = choose(logits)
        n = 1
        yield token
        while n != max_tokens and token not in self.eos_token_ids and self.position < self.capacity:
            if probs is None:
                token = sampling.pick(*self.top_k(token, sampling.k, penalties))
                penalties.accept(token)
            else:  # (the whole row)
                token = choose(self.step(token))
            n += 1
            yield token
