"""The host side every model's decode overlay shares (``ModelOverlay``; the models': ``qwen35.overlay``,
``gemma4.overlay``): the weight image's buffer (a ``WeightBuffer``'s adopted, or packed at ``load``), the buffers'
layout (``_take``), the residual-slice objects (F_LOADR), the phases' K-group input regions (``InputRegion``) and their
metadata objects, a K group's input fills, the drains of a column's outputs into a phase's x and of the LM head's into
LOGITS, a column-major image's weight runs, the transaction of a step per slot count, first layer and pattern of
sub-steps (built at its first use or ahead: ``prepare``) and its vectorized BD patches (``_bd_words``), a step's x and
logits, the LM head alone (``logits_of``); a sequence's state as host copies (a prompt cache's: ``read_records`` /
``write_records``, the KV caches' records of positions; ``snapshot`` / ``restore``, the rest).

A model's overlay supplies its config and image classes (``CONFIG``, ``IMAGE``), its design (``op``), its phases and
their flags (``_flags``; the metadata words of its own: ``_meta``), its static objects (``_load``), the layout of its
buffers (``_plan``: ``_layout``'s buffers, ``loadr``, the phases' input regions, its attention's; then ``_sizes``), a
step's transaction (``_build``), its sub-steps (``subs``, repeating every ``period`` positions), per-step patches
(``_set_position``) and ring bookkeeping (``_ran``), the LM head's phase alone (``_lm_phase``), its caches' rows
(``cache_rows``) and rings (``rings``).
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from ..formats import has_tile_format, tile_format
from ..gguf_model import GGUFFile
from ..operators.design import COLS
from ..operators.util import rup
from ..overlay import OverlayDevice, Sequence, Transaction, WeightBuffer
from . import phases as mp
from .design import (
    BC_COL,
    BC_ID,
    C_LEN,
    META,
    XB,
    input_fill,
    interleave,
    packed_slots,
    post_input_fill,
    post_region_bytes,
    region_bytes,
)
from .image import TileImage
from .vring import VRing


def paged_formats(gguf: GGUFFile, fmts: set[str], *, wide: bool = False) -> tuple[str, ...]:
    """The formats whose kernels are code pages (``DecodeOverlayOperator(paged=)``): the main cores' program memory holds
    the kernels of three formats; of four (Llama's Q3_K_M: Q3_K, Q4_K, Q5_K, Q6_K; Qwen3.6-35B-A3B's Q4_K, Q6_K,
    BFP16 with its drafter's Q4_0) the one of the most bytes in the file stays and the others share a page slot (two would leave the slot less than a page: the pager and the paged
    dispatch take ~0.5 KB); of five or more (Occamy's Q3_K, IQ3_XXS, Q4_K, Q5_K, Q6_K, Q8_0, BFP16), or of four in an
    MoE design with tiles of more than 16 rows (``wide``: Occamy's attention layers alone), all are pages: their dispatch
    and an MoE's wide frame (gate / up windows) leave no room for a resident one next to the slot. A page loads
    when a tile of another paged format comes (Q3_K_M: Q5_K and Q4_K in the first layers, Q4_K for the rest, the LM
    head's Q6_K, a few per step). By the whole file: the same choice for any layers and vocabulary the overlay runs."""
    if len(fmts) < 4:
        return ()
    if len(fmts) > 4 or wide:
        return tuple(sorted(fmts))
    nbytes = dict.fromkeys(fmts, 0)
    for _, qtype, _, n in gguf.tensor_info.values():
        if has_tile_format(qtype) and tile_format(qtype) in nbytes:
            nbytes[tile_format(qtype)] += n
    return tuple(sorted(sorted(fmts, key=lambda f: -nbytes[f])[1:]))



@dataclass
class InputRegion:
    """A K group's input region in ACT: [metadata object | x] (``design.region_bytes``; a header's:
    ``design.post_region_bytes``), its input elements k0 .. k0 + K."""

    off: int
    k0: int
    K: int
    post: bool = False


def host_bytes(shape: tuple[int, ...]) -> np.ndarray:
    """An uninitialized uint8 host array in 4 KB pages: numpy asks for huge pages for large arrays, and when RAM runs
    short of free pages (a model's file in the page cache) the kernel compacts memory for them first - a copy of
    600 MB then takes ~1 s instead of ~0.15 s."""
    return torch.empty(shape, dtype=torch.uint8).numpy()


@dataclass(eq=False)
class Records:
    """Host copies of an overlay's KV records of positions start .. end - 1 (``ModelOverlay.read_records``): per cache
    kind (``ModelOverlay.cache_rows``) its positions per group and the groups that hold those positions, (rows, groups,
    bytes per group) uint8 from group start // positions per group on."""

    start: int
    end: int
    parts: list[tuple[int, np.ndarray]]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for _, a in self.parts)

    def cut(self, end: int) -> "Records":
        """The records of positions start .. end - 1 alone (end <= self.end; copies: the rest is freed)."""
        assert self.start <= end <= self.end
        parts = []
        for pg, a in self.parts:
            b = host_bytes((a.shape[0], -(-end // pg) - self.start // pg, a.shape[2]))
            b[:] = a[:, :b.shape[1]]
            parts.append((pg, b))
        return Records(self.start, end, parts)


class ModelOverlay:
    """A GGUF model on its decode overlay (module docstring). ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``weights``: the image's buffer allocated ahead (its image's
    layers and vocabulary), ``image``: an image laid out already (a drafter's view of its model's image: ``w_of``)."""

    CONFIG: ClassVar[Any]  # the model's config class (``from_gguf``)
    IMAGE: ClassVar[type[TileImage]]  # its image class: (gguf, cfg, layers=, vocab=)
    # (the next step's position patched into a second transaction while a step runs: ``_run``; the model's
    # ``_set_position`` takes ``ahead`` and its transactions their buffer, ``_transaction``)
    AHEAD: ClassVar[bool] = False
    rings = (0, 0)  # (byte offset, bytes) in KV of the attention layers' rings (``vring``), set by ``_plan``

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None, vocab: int | None, weights: WeightBuffer | None,
                 image: TileImage | None = None):
        self.gguf = gguf
        self.cfg = cfg = weights.image.cfg if weights is not None else self.CONFIG.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.weights = weights
        self.w_of: ModelOverlay | None = None  # (the overlay whose weight buffer ``load`` adopts: ``image``'s)
        if weights is not None:
            assert weights.image.layers == self.layers and vocab in (None, weights.image.vocab)
            self.image = weights.image
        else:
            self.image = image or self.IMAGE(gguf, cfg, layers=self.layers, vocab=vocab)
        self.H, self.vocab, self.T = cfg.hidden_size, self.image.vocab, self.image.tile_bytes
        assert self.vocab % (COLS * C_LEN) == 0, "the logits drain whole C objects per column"
        self.start = self.layers[0]  # (the layer whose input is x: ``_mode``)
        self.inp: dict[tuple[int, str], list[InputRegion]] = {}  # phase -> its K groups' input regions
        self.dev: OverlayDevice | None = None
        self.vring = VRing()  # (the attention layers' rings: which open records they hold, which caches a prefill wrote)
        self._layout: dict[str, int] = {}  # (buffer -> bytes so far: ``_take``)
        self._steps: dict[tuple, Any] = {}  # (S, logits, start, sub-steps) -> the step's transaction
        self._metas_s: tuple[int, int] = (0, 0)  # (the S and start the phases' metadata objects were written for)
        self._meta_bytes: dict[tuple[int, int], tuple] = {}  # (S, start) -> the objects' ACT bytes (``_metas``)
        self._last_s = 1
        self._tr_lm: Transaction | None = None
        self._ahead: tuple[tuple, Any] | None = None  # ((the step's key, position), its transaction) patched ahead

    @property
    def SLOTS(self) -> int:
        """Token slots of a step at most (the design's)."""
        return self.op.slots

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _flags(self, L: int, name: str) -> int:
        """The flags of a phase (``decode.design``'s F_*)."""
        raise NotImplementedError

    def _take(self, buf: str, n: int) -> int:
        """n bytes (64-aligned) at the end of buffer ``buf``'s layout (a key of ``_layout``): their offset."""
        off = self._layout[buf]
        self._layout[buf] = off + rup(n)
        return off

    def _add_inputs(self, phase: tuple[int, str], *, post: bool = False, extra: int = 0) -> None:
        """The input regions of a phase's K groups (``phases.group_chunks``) in ACT; ``extra``: bytes per input element
        past a region's x (e.g. a sink of padding outputs)."""
        r = self.image.region(*phase)
        regs = []
        for k0, g in mp.group_chunks(r):
            K = g * r.kc
            n = (post_region_bytes(K) if post else region_bytes(K, self.SLOTS)) + extra * K
            regs.append(InputRegion(self._take("ACT", n), k0 * r.kc, K, post))
        self.inp[phase] = regs

    def _sizes(self) -> None:
        """The buffers' sizes (``sizes``): the layout's (64 bytes at least), LOGITS [slot][vocab] bf16."""
        self.sizes = {k: max(v, 64) for k, v in self._layout.items()} | {"LOGITS": 2 * self.SLOTS * self.vocab}

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, *, threads: int = 0, release: bool = True) -> None:
        """Compile the design, allocate the buffers, pack the weight image (or adopt the ``WeightBuffer``'s), write the
        static objects."""
        img = self.image
        if self.weights is not None or self.w_of is not None:
            device, buf = self.weights.buffer() if self.w_of is None else (self.w_of.dev.device, self.w_of.dev.buffers["W"])
            self.dev = dev = OverlayDevice(self.op, device=device)
            dev.adopt("W", buf)
        else:
            self.dev = dev = OverlayDevice(self.op)
            if img.pack(dev.alloc("W", img.nbytes), threads=threads, release=release):
                dev.flush("W")
        for name, n in self.sizes.items():
            dev.alloc(name, n)
        act = dev.view("ACT")
        for c in range(COLS):
            for s in range(self.SLOTS):
                act[self.loadr + (c * self.SLOTS + s) * XB:][:4 * META] = mp.loadr_meta(s).view(np.uint8)
        self._load()
        dev.flush("ACT")
        if (store := self.op.page_store()) is not None:  # (paged formats: their pages loadable before every dispatch)
            dev.set_prologue(self.op.page_prologue(store))
        self.reset()

    def _load(self) -> None:
        """The model's static objects (ACT is flushed after)."""

    def reset(self) -> None:
        """A new sequence: the KV caches zeroed."""
        self.dev.view("KV")[:] = 0
        self.dev.flush("KV")
        self.vring.clear()

    def ready(self) -> None:
        """Until the weight image is packed (``WeightBuffer``)."""
        if self.weights is not None:
            self.weights.wait()
        if self.w_of is not None:
            self.w_of.ready()

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta(self, L: int, name: str, rg: InputRegion, m: np.ndarray) -> np.ndarray:
        """The metadata object of a K group (``phases.groups``' words, the model's own added)."""
        return m

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        """The K groups of a phase for S slots (``phases.groups``)."""
        return mp.groups(self.image.region(L, name), self._flags(L, name), S, slots=self.SLOTS)

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """The phases' metadata objects for S slots: (ACT offset, bytes) each."""
        return [(rg.off, self._meta(L, name, rg, gr.meta).view(np.uint8))
                for (L, name), regs in self.inp.items() for rg, gr in zip(regs, self._groups(L, name, S))]

    def _metas(self, S: int) -> None:
        """The phases' metadata objects for S slots (from layer ``start`` on): a generation with drafts changes S every
        few steps: after another S of the same start only the bytes that differ, a few words per phase, are written and
        flushed; the objects' span takes ~6 ms to flush."""
        key = (S, self.start)
        idx, val = self._meta_of(S)
        old = self._meta_bytes.get(self._metas_s)
        if old is not None and self._metas_s[1] == self.start:
            ch = val != old[1]
            self.dev.view("ACT")[idx[ch]] = val[ch]
            self.dev.flush_lines("ACT", idx[ch])
        else:
            self.dev.view("ACT")[idx] = val
            self.dev.flush("ACT", int(idx[0]), int(idx[-1]) + 1 - int(idx[0]))
        self._metas_s = key

    def _meta_of(self, S: int) -> tuple[np.ndarray, np.ndarray]:
        """``_meta_writes``' bytes (ACT offsets, values) for S slots, kept per S and start."""
        key = (S, self.start)
        if (w := self._meta_bytes.get(key)) is None:
            writes = self._meta_writes(S)[::-1]  # (a byte written twice: the last write's)
            idx, first = np.unique(np.concatenate([o + np.arange(b.size) for o, b in writes]), return_index=True)
            val = np.concatenate([b for _, b in writes])[first]
            w = self._meta_bytes[key] = (idx, val)
        return w

    def _sp(self, S: int, phase: tuple[int, str] | None = None) -> int:
        """The slots of an input k block in a step of S slots (``design.packed_slots`` in a packed design) of a phase's
        input."""
        return packed_slots(S) if self.op.packed else self.SLOTS

    def _weight_run(self, seq: Sequence, phases: list[tuple[int, str]], *, image: TileImage | None = None,
                    buf: str = "W", blocks: slice = slice(None)) -> None:
        """The A fills of consecutive GEMV phases of a column-major image (default: the model's, in W): per column one
        run of it (their norm-weight data regions and tiles; one per layout block, ``TileImage.runs``; ``blocks``: of
        these runs only those).

        After an attention's cache fills a run's first block goes before the sync of the attention's inputs (A keeps
        streaming it while the attention waits for them), its later ones after the attention's X objects: a column's
        cache range and two tail objects hold three of A's 4 queued tasks, and a fifth task waits for the oldest, the
        cache's, which the core consumes only with those X objects (once the range outgrows the buffers on its way: a
        hang). Where A has 3 BDs for the host (an MoE design's), the whole run goes after the X objects."""
        img = image or self.image
        first = phases[0]
        norm = (first[0], first[1] + "_norm")
        r0 = img.region(*norm) if norm in img._index else img.region(*first)
        for r, n in img.runs(r0, img.region(*phases[-1]))[blocks]:
            for c in range(COLS):
                seq.fill("A", c, self.dev.address(buf, r.offset + c * r.col_stride), n)

    def _inputs(self, seq: Sequence, rg: InputRegion, S: int, phase: tuple[int, str] | None = None) -> None:
        """The X fills of a K group's input (``design.input_fill``, a header's ``post_input_fill``) of a phase."""
        sp = self._sp(S, phase)
        nbytes, dims = post_input_fill(rg.K, sp) if rg.post else input_fill(rg.K, sp, self.SLOTS)
        if self.op.bcast and phase != self._input(self.start):  # (a step's first phase: ``design``'s "Broadcast inputs")
            seq.fill("X", BC_COL, self.dev.address("ACT", rg.off), nbytes, dims, packet=BC_ID)
            return
        for c in range(COLS):
            seq.fill("X", c, self.dev.address("ACT", rg.off), nbytes, dims)

    def _values_to(self, seq: Sequence, c: int, phase: tuple[int, str], *, e0: int, n: int, S: int,
                   stream: str = "C", whole: bool = False) -> None:
        """n output values per slot of column c (objects (j, slot) of C_LEN values) -> the input elements e0 .. e0 + n
        of a phase's plain x ([k block][slot sp][8]). whole: the outputs as x lies (a residual of a design of 8 slots)."""
        sp = self._sp(S, phase)
        for rg in self.inp[phase]:
            a, b = max(e0, rg.k0), min(e0 + n, rg.k0 + rg.K)
            if whole:
                if a < b:
                    seq.drain(stream, c, self.dev.address("ACT", rg.off + XB + 2 * sp * (a - rg.k0)), 2 * sp * (b - a))
                continue
            j = a
            while j < b:  # (repeated BDs: at most 64 iterations)
                m = min(b - j, 64 * C_LEN)
                assert (j - rg.k0) % C_LEN == 0 and m % C_LEN == 0, (j, rg.k0, m)
                nobj = m // C_LEN
                seq.drain(stream, c, self.dev.address("ACT", rg.off + XB + 2 * sp * (j - rg.k0)), S * 2 * C_LEN,
                          (4, 8, 4 * sp, 4), nobj, (4 * sp * 8, nobj))
                j += m

    def _logits_out(self, seq: Sequence, c: int, S: int) -> None:
        """Column c's LM head outputs (vocab / 8 per slot) -> LOGITS [slot][vocab] bf16."""
        per = self.vocab // COLS
        seq.drain("C", c, self.dev.address("LOGITS", 2 * per * c), S * 2 * per,
                  (C_LEN // 2, S, self.vocab // 2, C_LEN // 2))

    def _build(self, S: int, lm: bool, subs: tuple) -> Any:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's sub-steps): an
        object with the ``Transaction`` ``tr`` (and what ``_set_position`` patches)."""
        raise NotImplementedError

    def subs(self, p: int, S: int) -> tuple:
        """The attention's sub-steps of a step at p .. p + S - 1 (the key of its transaction)."""
        raise NotImplementedError

    def passes(self, p: int, S: int) -> int:
        """The passes over a KV cache of a step at p .. p + S - 1."""
        return len(self.subs(p, S))

    def _set_position(self, st: Any, p: int, ahead: bool = False) -> bool:
        """A step's position-dependent objects and transfers (the patches of its transaction ``st``). ``ahead``: while
        the previous step runs (``AHEAD``), only objects of st's own buffer: False when the step needs more (it is
        patched again before it runs)."""
        raise NotImplementedError

    def _ran(self, p: int, S: int) -> None:
        """After a step of S slots at p: the rings it wrote (``vring``)."""

    def _ring_state(self) -> tuple:
        """What ``_set_position`` reads besides the position (a step patched ahead runs only if it is unchanged: a
        prefill, a reset or a rewind in between patch it again)."""
        return (frozenset(self.vring.from_prefill),
                tuple(sorted((L, tuple(sorted(d.items()))) for L, d in self.vring.slots.items())))

    def _transaction(self, S: int, logits: bool, subs: tuple, b: int = 0) -> Any:
        """The transaction of a step (built at its first use); ``b``: which of its two buffers (``AHEAD``: the next
        step's is patched while one runs; the builder reads ``_b``)."""
        key = (S, logits, self._variant(), subs, b)
        if key not in self._steps:
            self._b = b
            self._steps[key] = st = self._build(S, logits, subs)
            st.b = b
        return self._steps[key]

    def _variant(self) -> tuple:
        """What besides S, logits and the sub-steps selects a step's transaction: its first layer (``_mode``; a model's
        own variants after it)."""
        return (self.start,)

    def _mode(self, start: int | None) -> int:
        """Set the step's first layer (None: the first of ``layers``)."""
        start = self.layers[0] if start is None else start
        assert start in self.layers, start
        self.start = start
        return start

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """Build the transactions of a step of S slots (from layer ``start`` on: one per pattern of sub-steps over the
        positions, ``period``) and its metadata ahead (a step builds them at its first use)."""
        start = self._mode(start)
        self._meta_of(S)
        for subs in {self.subs(p, S) for p in range(self.period)}:
            for b in range(2 if self.AHEAD else 1):
                self._transaction(S, logits, subs, b)

    @staticmethod
    def _bd_words(words: np.ndarray, at: np.ndarray, address: np.ndarray, nbytes: np.ndarray) -> None:
        """``txn.bd_words`` of many BDs at once: length in words, address low, high at words[at ..]."""
        assert np.all(nbytes > 0) and np.all(nbytes % 4 == 0)
        words[at] = nbytes // 4
        words[at + 1] = address & 0xFFFFFFFF
        words[at + 2] = address >> 32

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _write_loadr(self, x: np.ndarray) -> None:
        """The slots' residual slices (F_LOADR payloads, fp32) of x (S, hidden) fp32."""
        S, h8 = x.shape[0], self.H // COLS
        lr = self.dev.view("ACT")[self.loadr:self.loadr + COLS * self.SLOTS * XB].reshape(COLS, self.SLOTS, XB)
        lr[:, :S, 4 * META:4 * META + 4 * h8] = x.reshape(S, COLS, h8).transpose(1, 0, 2).view(np.uint8).reshape(COLS, S, -1)
        self.dev.flush("ACT", self.loadr, COLS * self.SLOTS * XB)

    def _write_input(self, rg: InputRegion, xi: np.ndarray) -> None:
        """A plain input's x (``design.interleave``, as bytes) -> the x of its (first) K group's region."""
        self.dev.view("ACT")[rg.off + XB:rg.off + XB + xi.size] = xi
        self.dev.flush("ACT", rg.off + XB, xi.size)

    def _logits(self, S: int) -> torch.Tensor:
        """The last step's logits (S, vocab) bf16 (a view of the NPU's buffer, valid until the next dispatch): the
        LM rows of the vocabulary (an image's rows padded to whole C objects per column are cut off)."""
        self.dev.fetch("LOGITS", 0, 2 * S * self.vocab)
        x = torch.from_numpy(self.dev.view("LOGITS")[:2 * S * self.vocab]).view(torch.bfloat16).reshape(S, self.vocab)
        return x[:, :min(self.vocab, self.cfg.vocab_size)]

    def _run(self, S: int, position: int, logits: bool) -> torch.Tensor | None:
        """Dispatch the step of S slots at ``position`` (its inputs written): its transaction (built at its first use),
        patched (or patched ahead, while the previous step ran: ``AHEAD``); then, while it runs, the next step's (S slots
        at position + S, the same logits: a decode's next step) into the other buffer; the logits (``_logits``) or
        None."""
        subs = self.subs(position, S)
        key = (S, logits, self._variant(), subs)
        if self._ahead is not None and self._ahead[0] == (key, position, self._ring_state()):
            st = self._ahead[1]
        else:
            st = self._transaction(S, logits, subs)
            self._set_position(st, position)
        self._ahead = None
        self.ready()
        self.dev.submit(st.tr)
        try:
            self._ran(position, S)
            p = position + S
            if self.AHEAD and p + S <= self.max_positions:
                nxt = self._transaction(S, logits, self.subs(p, S), 1 - st.b)
                if self._set_position(nxt, p, ahead=True):
                    self._ahead = (((S, logits, self._variant(), self.subs(p, S)), p, self._ring_state()), nxt)
        finally:
            self.dev.wait()
        self._last_s = S
        return self._logits(S) if logits else None

    def _lm_phase(self, seq: Sequence) -> None:
        """The LM head's phase alone for one slot (``logits_of``): its weights, input fills, logits drains."""
        raise NotImplementedError

    def logits_of(self, x: torch.Tensor) -> torch.Tensor:
        """The LM head alone (one dispatch): the logits (vocab) bf16 of a last layer's output x (hidden), e.g. a
        prompt's last token from the prefill (a view of the NPU's buffer, valid until the next dispatch)."""
        rg = self.inp[(-1, "lm")][0]
        assert not rg.post, "the LM head reads x (not a header's residual parts)"
        if self._metas_s != (1, self.start):
            self._metas(1)
        if self._tr_lm is None:
            seq = Sequence(self.op, rolling=True, inputs=("X",))
            self._lm_phase(seq)
            seq.sync(fills=True)
            self._tr_lm = Transaction(self.dev, seq.words())
        self._write_input(rg, interleave(x.reshape(1, -1).float().numpy(), self._sp(1)).view(np.uint8).reshape(-1))
        self.ready()
        self.dev.dispatch(self._tr_lm)
        return self._logits(1)[0]

    # ------------------------------------------------------------------
    # A sequence's state
    # ------------------------------------------------------------------

    def cache_rows(self) -> list[tuple[np.ndarray, int, int]]:
        """The KV caches in KV, per kind: each row's byte offset (a KV head's cache of a layer), the bytes and the
        positions of a group (a row's groups follow each other from its offset)."""
        return []

    def read_records(self, start: int, end: int) -> Records:
        """Host copies of the caches' records of positions start .. end - 1 (the groups that hold them)."""
        dev, kv, parts = self.dev, self.dev.view("KV"), []
        for offs, gb, pg in self.cache_rows():
            g0, g1 = start // pg, -(-end // pg)
            a = host_bytes((len(offs), g1 - g0, gb))
            for i, o in enumerate(offs):
                dev.fetch("KV", o + g0 * gb, (g1 - g0) * gb)
                a[i] = kv[o + g0 * gb:o + g1 * gb].reshape(g1 - g0, gb)
            parts.append((pg, a))
        return Records(start, end, parts)

    def write_records(self, records: Records, start: int, end: int) -> None:
        """The caches' records of positions start .. end - 1 from ``records`` (which hold them; whole groups, so the
        positions around them in their groups too)."""
        assert records.start <= start and end <= records.end
        if end <= start:
            return
        dev, kv = self.dev, self.dev.view("KV")
        for (offs, gb, pg), (_, a) in zip(self.cache_rows(), records.parts, strict=True):
            g0, g1, r0 = start // pg, -(-end // pg), records.start // pg
            for i, o in enumerate(offs):
                kv[o + g0 * gb:o + g1 * gb] = a[i, g0 - r0:g1 - r0].reshape(-1)
                dev.flush("KV", o + g0 * gb, (g1 - g0) * gb)
        self._ahead = None

    def snapshot(self, position: int) -> dict[str, Any]:
        """What continuing a sequence from ``position`` needs of the overlay besides its caches' records before it: the
        open records' rings (their bytes, ``vring``), the records of the groups that hold positions position - SLOTS ..
        position (the open records: after a prefill their V panels are the exact values the rings hold otherwise,
        and the steps re-encode them; a drafter's rows that are not records yet) and the last step's slot count
        (``restore``)."""
        o, n = self.rings
        self.dev.fetch("KV", o, n)
        lo, hi = max(position - self.SLOTS, 0), min(position + 1, self.max_positions)
        rings = host_bytes((n,))
        rings[:] = self.dev.view("KV")[o:o + n]
        return {"rings": rings, "vring": self.vring.copy(), "last_s": self._last_s,
                "open": (lo, hi, self.read_records(lo, hi))}

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Back to a ``snapshot`` (the caches hold the records of the positions before it)."""
        o, n = self.rings
        self.dev.view("KV")[o:o + n] = snapshot["rings"]
        self.dev.flush("KV", o, n)
        lo, hi, records = snapshot["open"]
        self.write_records(records, lo, hi)
        self.vring.assign(snapshot["vring"])
        self._last_s, self._ahead = snapshot["last_s"], None
