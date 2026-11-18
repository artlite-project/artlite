"""The Llama decode overlay (models/llama: ``LlamaOverlay``) on 2-layer slices of GGUF files with a truncated
vocabulary, against the overlay's arithmetic on the host (ref.LlamaNpuRef), in both K formats of the caches:

- a text from position 0 in steps of 1 .. 4 tokens (record pairs, records and cache groups crossed at every alignment;
  a step of S slots gives each the logits of a step of one): logits, attention outputs, the LM head's input, then the
  caches and the open record's ring;
- steps at positions 960 .. 1039 after a cache filled with random K, V (record, group and object boundaries deep in
  the cache), against the reference attending to the cache as stored;
- three sequences on one device, alternated and in joint steps: every slot the logits of its sequence alone, bit for bit.

Files: 1B Q4_K_M (head_dim 64, 4 query heads per KV head; layer 0's V / down Q6_K), 1B Q5_K_M (13-row Q5_K tiles), 1B
Q6_K (12-row tiles of 10112 B), 3B Q4_K_M (head_dim 128, 3 query heads per KV head), 8B Q4_K_S (hidden 4096, Q4_K, Q5_K
and Q6_K tiles; down in 7 K groups), 1B BF16 (8-row BFP16 tiles of 9216 B), 1B Q4_0 (Q4_0 of 16 rows, layer 0's down
Q4_1 of 14); Qwen3 0.6B Q4_1 (16 rows, 10240 B; 2 query heads per KV head, q / k norms with weights per dim:
layer 0's k_norm up to 96.5), Qwen3-Embedding-0.6B Q8_0 (8-row Q8_0 tiles of 9216 B) and Qwen3-Embedding-4B Q4_K_M
(hidden 2560: half regions, ``llama.image``; ffn 9728 in K chunks padded with zero super-blocks; 4 query heads per KV
head at head_dim 128), whose half regions also against its padded image (``test_llama_decode_tails``).
"""


import time

import numpy as np
import pytest
import torch

from artlite.decode import kv
from artlite.decode.kv import KV_K
from artlite.decode.protocol import SLOTS
from artlite.decode.step import Part
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import llama_gguf
from artlite.operators.design import COLS
from models.llama.image import LlamaImage
from models.llama.overlay import LlamaOverlay

from ..common import as_torch, essay, to_torch
from ..model_files import llama_path
from ..numerics import rel_finite
from . import ref

VOCAB = 2048
# (GGUF quantization, model size, layers)
CASES = {"1B_Q4_K_M": ("Q4_K_M", "1B", [0, 2]), "1B_Q5_K_M": ("Q5_K_M", "1B", [0, 3]),
         "1B_Q6_K": ("Q6_K", "1B", [0, 1]), "3B_Q4_K_M": ("Q4_K_M", "3B", [0, 3]),
         "8B_Q4_K_S": ("Q4_K_S", "8B", [0, 4]), "1B_BF16": ("BF16", "1B", [0, 1]), "1B_Q4_0": ("Q4_0", "1B", [0, 2]),
         "3B_Q2_K": ("Q2_K", "3B", [0, 3]), "3B_Q3_K_M": ("Q3_K_M", "3B", [0, 3]),
         "1B_Q3_K_L": ("Q3_K_L", "1B", [0, 2]), "qwen3_0.6B_Q4_1": ("Q4_1", "qwen3:0.6B", [0, 1]),
         "qwen3_emb_Q8_0": ("Q8_0", "qwen3:embedding-0.6B", [0, 1]),
         "qwen3_emb4b_Q4_K_M": ("Q4_K_M", "qwen3:embedding-4B", [0, 2])}
STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)
# relative rms error: the reference leaves out the GEMV's panels (14 bits) and its 32-k sums rounded to bf16, which move
# a projection's rows by ~1% (the NPU's o projection is bit-exact to ``mmv.ref_rows``); up to 4.2% here
TOL = 0.05


def _gguf(quant: str, size: str) -> GGUFFile:
    return GGUFFile(llama_path(quant, size))


def _tokens(n: int) -> list[int]:
    from models.llama.tokenizer import load_tokenizer

    return load_tokenizer(llama_gguf("Q4_K_M", "1B")).encode(essay("addiction.txt"))[:n]


# (Qwen3: hi + lo K only; with single K blocks the roundings of its K, whose dims span ~1e5 within a block, follow the
# GEMV's noise, and its large scores turn them into ~30% of a layer's attention: the KL test measures that mode)
# (the formats' slices run in single K; the hilo ones are the Q4_K_M files', where the head layouts differ, and Qwen3's)
STEP_CASES = [(c, k) for c in CASES for k in KV_K
              if (k == "single" and not c.startswith("qwen3"))
              or (k == "hilo" and (c in ("1B_Q4_K_M", "3B_Q4_K_M") or c.startswith("qwen3")))]


@pytest.mark.parametrize(("case", "kv_k"), STEP_CASES, ids=[f"{c}-{k}" for c, k in STEP_CASES])
def test_llama_decode_steps(case: str, kv_k: str) -> None:
    quant, size, layers = CASES[case]
    g = _gguf(quant, size)
    ids = _tokens(136)  # (crosses a cache group: 64 positions at head_dim 128, 128 at 64)
    model = ref.LlamaNpuRef(g, layers, VOCAB, kp=KV_K[kv_k])
    trace: dict[str, torch.Tensor] = {}
    want = model.run(ids, 0, trace)
    x = model.embed(ids)
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=256, kv_k=kv_k)
    ov.load()
    worst: dict[str, float] = {}

    def keep(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
        worst[name] = max(worst.get(name, 0.0), rel_finite(as_torch(got), as_torch(want)))

    p, i = 0, 0
    while p < len(ids):
        n = min(STEPS[i % len(STEPS)], len(ids) - p)
        i += 1
        logits = to_torch(ov.step(x[p:p + n], p)).float()
        for s in range(n):
            keep("logits", logits[s], want[p + s])
            keep("lm_in", ov.hidden(-1, s), trace["r"][p + s])
            for L in layers:
                keep(f"attn.{L}", ov.attn_out(L, s), trace[f"attn.{L}"][p + s])
        p += n
    r0 = (len(ids) - 1) // 8 * 8  # (the last record: its values in the ring)
    ov.dev.fetch("KV")
    for L, cache in model.caches.items():  # the caches as the last token left them
        k, v = ov.read_kv(L, range(len(ids)))
        keep(f"K.{L}", k, cache.K)
        keep(f"V.{L}", v, cache.V)
        rows = [ring_rows(ov.dev.view("KV")[ov.ring(L, c) + (r0 // 8 & 1) * ov.T:][:ov.T], ov.hd) for c in range(COLS)]
        keep(f"ring.{L}", torch.from_numpy(np.stack(rows)[:, :len(ids) - r0]), cache.rows[:, r0:])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < TOL for v in worst.values()), worst


@pytest.mark.parametrize("pairs", [False, True], ids=["halves", "pairs"])
def test_llama_decode_tails(pairs: bool) -> None:
    """Qwen3-Embedding-4B's half regions (``llama.image``: inputs of the hidden size 2560) on the design of 4 slots:
    without pairs the tiles and every sum's order are the padded image's (its last half chunk's zero super-blocks
    times the helper's copy of x), so steps of 1 .. 4 slots give its logits and LM-head inputs bit for bit; with pairs
    (gate / up's and the LM head's row tiles: a pair's second sums its tail first) steps of 1 .. 4 slots give those of
    steps of one, bit for bit (``test_llama_decode_steps``: within the reference's noise)."""
    quant, size, layers = CASES["qwen3_emb4b_Q4_K_M"]
    g = _gguf(quant, size)
    ids = _tokens(136)
    x = ref.LlamaNpuRef(g, layers, VOCAB, kp=KV_K["hilo"]).embed(ids)

    def run(image: LlamaImage, steps: tuple[int, ...]) -> list[np.ndarray]:
        ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=256, kv_k="hilo", image=image)
        assert ov.op.tails == any(r.half for r in image.regions)
        assert ov.op.tail_pairs == any(b.pt for r in image.regions for b in r.blocks)
        ov.load()
        out, p, i = [], 0, 0
        while p < len(ids):
            n = min(steps[i % len(steps)], len(ids) - p)
            i += 1
            lg = np.asarray(ov.step(x[p:p + n], p)).view(np.uint16)
            out += [np.concatenate([lg[s], ov.hidden(-1, s).view(np.uint32)], dtype=np.int64) for s in range(n)]
            p += n
        ov.dev = None
        return out

    img = LlamaImage(g, layers=layers, vocab=VOCAB, pairs=pairs)
    assert {(r.name, b.name) for r in img.regions for b in r.blocks if b.pt} == \
        ({("gate_up", "gu"), ("lm", "lm")} if pairs else set())
    want = run(img, (1,)) if pairs else run(LlamaImage(g, layers=layers, vocab=VOCAB, tails=False), STEPS)
    got = run(img, STEPS)
    bad = [p for p in range(len(ids)) if not np.array_equal(want[p], got[p])]
    assert not bad, f"{len(bad)} of {len(ids)} positions differ: {bad[:8]}"


@pytest.mark.parametrize(("case", "kv_k"), [("1B_Q4_K_M", "single"), ("3B_Q4_K_M", "hilo"), ("qwen3_0.6B_Q4_1", "hilo")],
                         ids=lambda v: v)
def test_llama_decode_sequences(case: str, kv_k: str) -> None:
    """Three sequences on one device (``LlamaOverlay(sequences=3)``: texts of their own, at positions of their own)
    give, slot by slot, the logits and LM-head inputs of each sequence alone in steps of one, bit for bit:

    - alternated (``select``: a step of 1 .. 4 slots of one, then of the next: no copies between them);
    - in joint steps (``step``'s parts: one sequence 5 positions behind, one joining at the 20th step, slots dealt at
      random, a sequence's second slot too: sub-steps of every pattern up to a sequence per slot);

    and each sequence's caches and open record's ring are then those of the sequence alone."""
    quant, size, layers = CASES[case]
    g = _gguf(quant, size)
    N, NSEQ = 136, 3  # (crosses a cache group: 64 positions at head_dim 128, 128 at 64)
    ids = _tokens(NSEQ * N)
    xs = [np.asarray(g.rows("token_embd.weight", ids[s * N:(s + 1) * N]), np.float32) for s in range(NSEQ)]
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=256, kv_k=kv_k, sequences=NSEQ)
    ov.load()

    def slots(n: int) -> list[np.ndarray]:  # (per slot of the last step of n slots: its logits and LM-head input)
        lg = np.asarray(ov._npu_logits(n)).view(np.uint16)
        return [np.concatenate([lg[s], ov.hidden(-1, s).view(np.uint32)], dtype=np.int64) for s in range(n)]

    def caches(s: int) -> list[np.ndarray]:  # (sequence s's caches as stored, its open record's rows)
        ov.select(s)
        ov.dev.fetch("KV")
        out = [a for L in layers for a in ov.read_kv(L, range(N))]
        r0 = (N - 1) // 8
        rows = N - 8 * r0
        return out + [ring_rows(ov.dev.view("KV")[ov.ring(L, c) + (r0 & 1) * ov.T:][:ov.T], ov.hd)[:rows]
                      for L in layers for c in range(COLS)]

    want, kept = [], []
    for s in range(NSEQ):  # (alone, in steps of one, as sequence 0)
        ov.select(0)
        ov.reset()
        row = []
        for p in range(N):
            ov.step(xs[s][p:p + 1], p)
            row.append(slots(1)[0])
        want.append(row)
        kept.append(caches(0))
    assert not any(np.array_equal(a, b) for a, b in zip(want[0], want[1])), "the sequences' texts differ"

    def check(got: list[np.ndarray], at: list[tuple[int, int]], what: str) -> None:
        bad = [(s, p) for g_, (s, p) in zip(got, at, strict=True) if not np.array_equal(g_, want[s][p])]
        assert not bad, f"{what}: (sequence, position) {bad[:8]} differ from the sequence alone"

    def fresh() -> None:
        for s in range(NSEQ):
            ov.select(s)
            ov.reset()

    fresh()  # alternated: a step of one sequence, then of the next
    pos, i = [0] * NSEQ, 0
    while min(pos) < N:
        s = i % NSEQ
        n = min(STEPS[(i // NSEQ + s) % len(STEPS)], N - pos[s])
        i += 1
        if n:
            ov.select(s)
            ov.step(xs[s][pos[s]:pos[s] + n], pos[s])
            check(slots(n), [(s, pos[s] + k) for k in range(n)], "alternated")
            pos[s] += n
    for s in range(NSEQ):
        assert all(np.array_equal(a, b) for a, b in zip(caches(s), kept[s], strict=True)), f"alternated: caches of {s}"

    fresh()  # joint steps
    rng = np.random.default_rng(0)
    pos, steps, patterns = [0] * NSEQ, 0, set()
    while min(pos) < N:
        take = [[s, 1] for s in range(NSEQ) if pos[s] < N and not (s == 1 and steps < 5) and not (s == 2 and steps < 20)]
        spare = SLOTS - len(take)
        for t in take:  # (a second slot of its own)
            if spare and rng.random() < 0.4 and pos[t[0]] + t[1] < N:
                t[1] += 1
                spare -= 1
        rng.shuffle(take)
        parts = tuple(Part(s, pos[s], n) for s, n in take)
        patterns.add(ov._join(parts))
        ov.step(np.concatenate([xs[s][pos[s]:pos[s] + n] for s, n in take]), parts)
        check(slots(sum(n for _, n in take)), [(s, pos[s] + k) for s, n in take for k in range(n)], f"step {steps}")
        for s, n in take:
            pos[s] += n
        steps += 1
    assert max(len(u) for u in patterns) == SLOTS, "a step of a sequence per slot"
    for s in range(NSEQ):
        assert all(np.array_equal(a, b) for a, b in zip(caches(s), kept[s], strict=True)), f"joint steps: caches of {s}"


def test_llama_decode_after_suspend() -> None:
    """A paged overlay's step after the NPU powered down (Linux: the driver's runtime suspend, 5 s after the device's
    last job) gives the logits it gave before: the array lost the formats' code pages, and the overlay's prologue loads
    them again (``OverlayDevice.set_prologue``). Skipped where the device does not suspend (``power/control`` on,
    another process on the NPU, no such files)."""
    pm = "/sys/class/accel/accel0/device/power/"

    def read(name: str) -> str:
        try:
            with open(pm + name) as f:
                return f.read().strip()
        except OSError:
            return ""

    if read("control") != "auto":
        pytest.skip(f"the NPU does not runtime-suspend here ({pm}control: {read('control') or 'no such file'})")
    quant, size, layers = CASES["1B_Q4_K_M"]
    g = _gguf(quant, size)
    x = ref.LlamaNpuRef(g, layers, VOCAB).embed(_tokens(4))
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=256)
    assert ov.op.paged, "the slice's formats are code pages"
    ov.load()
    before = np.array(ov.step(x, 0))
    end = time.monotonic() + int(read("autosuspend_delay_ms") or 5000) / 1e3 + 3
    while read("runtime_status") != "suspended":
        if time.monotonic() > end:
            pytest.skip("the NPU stayed awake (another process's work)")
        time.sleep(0.1)
    assert np.array_equal(np.array(ov.step(x, 0)), before)


@pytest.mark.parametrize(("case", "kv_k"), [("1B_Q4_K_M", "single"), ("3B_Q4_K_M", "hilo"), ("8B_Q4_K_S", "single")],
                         ids=lambda v: v)
def test_llama_decode_deep(case: str, kv_k: str) -> None:
    """Steps at 960 .. 1039 (one token each, and steps of 4 from 1008) after a cache of 960 random positions."""
    quant, size, layers = CASES[case]
    g = _gguf(quant, size)
    P0, N = 960, 80
    model = ref.LlamaNpuRef(g, layers, VOCAB, kp=KV_K[kv_k])
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=1152, kv_k=kv_k)
    ov.load()
    gen = torch.Generator().manual_seed(7)
    nkv, hd = ov.cfg.num_key_value_heads, ov.hd
    kvb = ov.dev.view("KV")
    for L in layers:  # (K, V as a model's: K of rotated rows, larger than V)
        K = (torch.randn(nkv, P0, hd, generator=gen) * 2).bfloat16().float()
        V = torch.randn(nkv, P0, hd, generator=gen).bfloat16().float()
        pad = (0, -P0 % ov.pg)  # (whole groups; the steps write the positions from P0 on)
        for c in range(nkv):
            grp = kv.kv_groups(np.pad(K[c].numpy(), (pad, (0, 0))), np.pad(V[c].numpy(), (pad, (0, 0))), ov.T, ov.kp)
            kvb[ov.kv_base(L, c):][:grp.size] = grp.reshape(-1)
    ov.dev.flush("KV")
    for L in layers:  # (the reference attends to the cache as stored)
        k, v = (torch.from_numpy(a) for a in ov.read_kv(L, range(P0)))
        cache = model.caches[L]
        cache.K, cache.V, cache.rows = k, v, v.clone()
    ids = _tokens(N)
    want = model.run(ids, P0)
    x = model.embed(ids)
    worst = 0.0
    p = P0
    while p < P0 + N:
        n = 4 if p >= 1008 else 1
        logits = to_torch(ov.step(x[p - P0:p - P0 + n], p)).float()
        for s in range(n):
            worst = max(worst, rel_finite(logits[s], want[p - P0 + s]))
        p += n
    for L in layers:
        k, v = ov.read_kv(L, range(P0, P0 + N))
        cache = model.caches[L]
        worst = max(worst, rel_finite(torch.from_numpy(k), cache.K[:, P0:]), rel_finite(torch.from_numpy(v), cache.V[:, P0:]))
    print(f"worst relative error {worst:.4f}")
    assert worst < TOL, worst
