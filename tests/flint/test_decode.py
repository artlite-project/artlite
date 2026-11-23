"""The Flint decode overlay (models/flint: ``FlintOverlay``) on the NPU:

- slices of layers with a truncated vocabulary against the numpy reference (``ref.layer``, fp32 weights): layers 0 .. 5
  (dense full-attention without positional encoding, MoE sliding-window, MoE full) over 600 positions (past the window
  of 513) in steps of 1 .. 4 tokens, K cached as hi + lo blocks, and an MoE full layer (5) with single K blocks: the
  logits, the residual after the last layer's attention;
- steps of S slots give each slot the logits of a step of one, bit for bit (the attention's sub-steps, the MoE's partials
  summed on the slots' grid whatever the union of the step's experts), also where a step continues from fewer slots than
  the last one ran (rejected drafts);
- several sequences on one device, alternated and in joint steps: every slot the logits of its sequence alone, bit for
  bit, with linear caches and with the sliding-window caches as rings, one sequence across the wrap;
- a dense layer's gate / up as Q8_0 tiles of 8 rows (the design's gate / up windows).
"""

import numpy as np
import pytest

from artlite.decode.protocol import SLOTS
from artlite.decode.step import Part
from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig, flint_gguf
from models.flint.overlay import FlintOverlay
from models.tokenizer import BpeTokenizer

from ..common import essay
from ..model_files import require
from ..numerics import rel
from . import ref

VOCAB = 1024
STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)
TOL = 0.03  # (relative rms error of the logits: bfp16 x panels, bf16 sums, the caches' blocks; ~1% here; a position whose
# routing flips a near-tied expert up to ~6%: the 95th percentile and the median are held)


@pytest.fixture(scope="module")
def flint() -> tuple[GGUFFile, FlintConfig]:
    path = flint_gguf()
    require(path)
    g = GGUFFile(path)
    return g, FlintConfig.from_gguf(g)


def _ids(g: GGUFFile, n: int) -> list[int]:
    return BpeTokenizer(g).encode(essay("apple.txt"))[:n]


def _run(ov: FlintOverlay, x: np.ndarray, steps: tuple[int, ...]) -> np.ndarray:
    """The logits (fp32) of every position of x, in steps of the given sizes (repeated) from position 0."""
    out = np.zeros((x.shape[0], ov.vocab), np.float32)
    p, i = 0, 0
    while p < x.shape[0]:
        n = min(steps[i % len(steps)], x.shape[0] - p)
        i += 1
        out[p:p + n] = np.asarray(ov.step(x[p:p + n], p), np.float32)
        p += n
    return out


@pytest.mark.parametrize(("layers", "n", "kv_k", "steps"),
                         [([5], 40, "single", STEPS), ([0, 1, 2, 3, 4, 5], 600, "hilo", STEPS)],
                         ids=["moe-full-single", "six"])
def test_flint_overlay_layers(flint: tuple[GGUFFile, FlintConfig], layers: list[int], n: int, kv_k: str,
                              steps: tuple[int, ...]) -> None:
    g, cfg = flint
    ids = _ids(g, n)
    x = g.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)
    h, pos = x.copy(), np.arange(n)
    for L in layers:
        out = ref.layer(g, cfg, L, h, pos, q8=False)
        ffn_in, h = out["ffn_inp"], out["l_out"]
    _, want = ref.head_logits(g, cfg, h, np.arange(VOCAB), q8=False)
    ov = FlintOverlay(g, layers=layers, vocab=VOCAB, max_positions=1024, kv_k=kv_k)
    ov.load()
    got = _run(ov, x, steps)
    errs = np.array([rel(got[p], want[p]) for p in range(n)])
    agree = float(np.mean(got.argmax(1) == want.argmax(1)))
    p0 = n - _sizes(n, steps)[-1]  # (the last step's first slot: its residual after the last layer's attention)
    ffn = rel(ov.ffn_in(layers[-1]), ffn_in[p0])
    print(f"logits median {np.median(errs):.4f}, p95 {np.percentile(errs, 95):.4f}, max {errs.max():.4f}, argmax {agree:.3f}, "
          f"ffn_in {ffn:.4f}")
    assert np.median(errs) < TOL / 2 and np.percentile(errs, 95) < TOL and ffn < TOL and agree > 0.9, (errs, agree, ffn)


def _sizes(n: int, steps: tuple[int, ...]) -> list[int]:
    """The step sizes ``_run`` takes over n positions."""
    out, p, i = [], 0, 0
    while p < n:
        out.append(min(steps[i % len(steps)], n - p))
        p += out[-1]
        i += 1
    return out


def test_flint_overlay_steps(flint: tuple[GGUFFile, FlintConfig]) -> None:
    """Steps of 1 .. 4 slots give the logits of steps of one, bit for bit (a dense and an MoE layer); so do steps that
    continue from fewer slots than the last one ran (a generation's rejected drafts: the slots past the accepted ones hold
    other tokens)."""
    g, _ = flint
    ids = _ids(g, 60)
    x = g.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)
    ov = FlintOverlay(g, layers=[1, 2], vocab=VOCAB, max_positions=512)
    ov.load()
    one = _run(ov, x, (1,))
    ov.reset()
    assert np.array_equal(_run(ov, x, STEPS), one)
    ov.reset()
    rng = np.random.default_rng(0)
    p = 0
    while p < len(ids):
        n = min(int(rng.integers(1, 5)), len(ids) - p)
        acc = int(rng.integers(1, n + 1))
        xs = x[p:p + n].copy()
        xs[acc:] = g.rows("token_embd.weight", rng.integers(0, 150000, n - acc)).astype(np.float32)
        got = np.asarray(ov.step(xs, p), np.float32)
        assert np.array_equal(got[:acc], one[p:p + acc]), p
        p += acc


@pytest.mark.parametrize(("layers", "wrap", "lengths"), [([1, 2], 0, (72, 72, 72)), ([2, 3], 9, (640, 150, 40))],
                         ids=["linear", "ring"])
def test_flint_overlay_sequences(flint: tuple[GGUFFile, FlintConfig], layers: list[int], wrap: int,
                                 lengths: tuple[int, ...]) -> None:
    """Three sequences on one device (``FlintOverlay(sequences=3)``: texts of their own, at positions of their own) give,
    slot by slot, the logits of each sequence alone in steps of one, bit for bit: alternated (``select``: a step of 1 ..
    4 slots of one, then of the next) and in joint steps (``step``'s parts: slots dealt at random, a sequence's second
    slot too, one sequence joining later; an MoE layer runs the union of the slots' experts). "ring": the
    sliding-window caches as rings of 9 groups (576 positions), the first sequence across the wrap while the others
    are before it."""
    g, _ = flint
    nseq = len(lengths)
    ids = _ids(g, sum(lengths))
    offs = np.cumsum([0, *lengths])
    xs = [g.rows("token_embd.weight", np.asarray(ids[offs[s]:offs[s + 1]])).astype(np.float32) for s in range(nseq)]
    ov = FlintOverlay(g, layers=layers, vocab=VOCAB, max_positions=1024, wrap=wrap, sequences=nseq)
    ov.load()
    want = []
    for s in range(nseq):  # (alone, in steps of one, as sequence 0)
        ov.select(0)
        ov.reset()
        want.append(_run(ov, xs[s], (1,)))
    assert not np.array_equal(want[0][:32], want[1][:32]), "the sequences' texts differ"

    def check(got: np.ndarray, at: list[tuple[int, int]], what: str) -> None:
        bad = [(s, p) for row, (s, p) in zip(np.asarray(got, np.float32), at, strict=True)
               if not np.array_equal(row, want[s][p])]
        assert not bad, f"{what}: (sequence, position) {bad[:8]} differ from the sequence alone"

    def fresh() -> None:
        for s in range(nseq):
            ov.select(s)
            ov.reset()

    fresh()  # alternated: a step of one sequence, then of the next
    pos, i = [0] * nseq, 0
    while any(pos[s] < lengths[s] for s in range(nseq)):
        s = i % nseq
        n = min(STEPS[(i // nseq + s) % len(STEPS)], lengths[s] - pos[s])
        i += 1
        if n:
            ov.select(s)
            check(ov.step(xs[s][pos[s]:pos[s] + n], pos[s]), [(s, pos[s] + k) for k in range(n)], "alternated")
            pos[s] += n

    fresh()  # joint steps
    rng = np.random.default_rng(0)
    pos, steps, patterns = [0] * nseq, 0, set()
    while any(pos[s] < lengths[s] for s in range(nseq)):
        take = [[s, 1] for s in range(nseq) if pos[s] < lengths[s] and not (s == nseq - 1 and steps < 12)]
        spare = SLOTS - len(take)
        for t in take:  # (more slots of its own)
            while spare and rng.random() < 0.4 and pos[t[0]] + t[1] < lengths[t[0]]:
                t[1] += 1
                spare -= 1
        parts = tuple(Part(s, pos[s], n) for s, n in take)
        patterns.add(ov._join(parts))
        got = ov.step(np.concatenate([xs[s][pos[s]:pos[s] + n] for s, n in take]), parts)
        check(got, [(s, pos[s] + k) for s, n in take for k in range(n)], f"step {steps}")
        for s, n in take:
            pos[s] += n
        steps += 1
    assert max(len(u) for u in patterns) >= 3, "a step of three sequences"


def test_flint_overlay_q8_gate_up(flint: tuple[GGUFFile, FlintConfig], monkeypatch: pytest.MonkeyPatch) -> None:
    """A dense layer whose gate / up are Q8_0 tiles of 8 rows (another file's formats: unsloth's UD-Q4_K_XL; here the
    official weights requantized in memory, the overlay's format check lifted): the design's gate / up windows
    (``gu_win``: an 8-row tile holds only gate or only up rows; without them a step never ended) give the reference's
    logits on the same weights, in steps of 1 .. 4 over 200 positions."""
    from gguf.quants import Q8_0

    import models.flint.overlay as FO
    from artlite.gguf_model import QTensor
    from models.flint.image import FlintImage

    g, cfg = flint
    names = ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight")
    own = {}
    for n in names:
        t = g.tensor(n)
        w = np.asarray(t.dequantize(), np.float32).reshape(t.rows, t.cols)
        own[n] = QTensor(name=n, qtype="Q8_0", rows=t.rows, cols=t.cols, raw=Q8_0.quantize(w).reshape(t.rows, -1))

    class Image(FlintImage):
        def _layer(self, L: int) -> None:
            self._own.update({n: t for n, t in own.items() if n.startswith(f"blk.{L}.")})
            super()._layer(L)

    class G8:  # (the file with the requantized tensors: the reference's)
        def __getattr__(self, k: str) -> object:
            return getattr(g, k)

        def tensor(self, n: str) -> QTensor:
            return own.get(n) or g.tensor(n)

        def rows(self, n: str, idx: object, cols: int = 0, **kw: object) -> np.ndarray:
            return np.asarray(own[n].dequantize(), np.float32)[np.asarray(idx)] if n in own else g.rows(n, idx, cols, **kw)

    monkeypatch.setattr(FO, "FORMATS", FO.FORMATS | {"Q8_0"})
    monkeypatch.setattr(FO.FlintOverlay, "IMAGE", Image)
    ov = FO.FlintOverlay(g, layers=[0], vocab=VOCAB, max_positions=512)
    assert ov.image.gu_rows() == {8} and ov.op.gu_win
    ov.load()
    n = 200
    x = g.rows("token_embd.weight", np.asarray(_ids(g, n))).astype(np.float32)
    h = ref.layer(G8(), cfg, 0, x, np.arange(n), q8=False)["l_out"]
    _, want = ref.head_logits(G8(), cfg, h, np.arange(VOCAB), q8=False)
    got = _run(ov, x, STEPS)
    errs = np.array([rel(got[p], want[p]) for p in range(n)])
    assert np.median(errs) < TOL / 2 and np.percentile(errs, 95) < TOL, errs
