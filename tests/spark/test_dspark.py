"""The DSpark drafter on the NPU (``models.spark.dspark.DSparkOverlay``) against the fp32 drafter (``dspark_cpu``) on the
same taps: the model's layers ``taps`` alone (a slice: the taps are what those layers read in it), decode steps of an
essay's tokens, after each the drafter's injection of the step's rows (the taps read back from the model's ACT feed the
CPU's); the rings' records of the window (K, V per layer and KV head) and a block's logits over the head's rows and its
output-normed rows. The drafter's records of a prompt's chunks (``SparkPrefill(dspark=)``) against the injections of the
same tokens as decode steps. The whole model (slow): greedy text with drafts the text without; teacher-forced drafts
against the fp32 drafter's on the model's own taps."""

import numpy as np
import pytest
import torch

from artlite.decode import attn256 as AT
from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import DSparkConfig, spark_dspark_of
from artlite.operators import bfp16
from models.sampling import Sampling
from models.spark.dspark import DSparkOverlay, needs
from models.spark.overlay import SparkOverlay
from models.spark.prefill import SparkPrefill
from models.spark.spark import NpuSpark, Spark

from ..model_files import require
from ..numerics import rel_finite
from . import cpu, ref
from .dspark_cpu import DSparkCpu

VOCAB = 512
PREFILL_SIZES = (64, 128, 256, 512)  # (the prefill tests' library: one per size for both)


def _records(d: DSparkOverlay, L: int, lo: int, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drafter layer L's K, V per KV head of positions lo .. n - 1 from its rings: (K, V) each (n - lo, 256)."""
    nr, T = d.nr, d.T
    g0, g1 = lo // 32, -(-n // 32)
    d.dev.fetch("KV")
    kv = d.dev.view("KV")
    out = []
    for h in range(d.ash.kvh):
        base = d.kv_off[L] + h * nr * AT.OBJS * T
        grp = np.stack([kv[base + (g % nr) * AT.OBJS * T:][:AT.OBJS * T].reshape(AT.OBJS, T) for g in range(g0, g1)])
        K, V = KV.kv_from_groups(grp, AT.HD, dmajor=True)
        out.append((np.asarray(K, np.float32)[lo - 32 * g0:n - 32 * g0], np.asarray(V, np.float32)[lo - 32 * g0:n - 32 * g0]))
    return out


def _slice(size: str) -> tuple[SparkOverlay, DSparkOverlay]:
    g = ref.gguf(size)
    dpath = spark_dspark_of(ref.path(size))
    require(dpath)
    dg = GGUFFile(dpath)
    tgt = SparkOverlay(g, layers=sorted(DSparkConfig.from_gguf(dg).taps), vocab=VOCAB, max_positions=4096, **needs(dg))
    tgt.load()
    d = DSparkOverlay(tgt, dg)
    d.load()
    return tgt, d


@pytest.mark.parametrize("size", ["1.7B", "4B"])
def test_dspark_rounds(size: str) -> None:
    """Steps of 8, 3 and 1 slots over 700 positions (the rings' 20 groups wrap), each followed by the injection of its
    rows, blocks at 300 and 700 (the block's records then overwritten by the next injection): the window's records, the
    block's logits and h against the CPU's; the drafter's state at 300 restored after the rest (its rings wrapped
    since): the block there bit for bit."""
    g = ref.gguf(size)
    dpath = spark_dspark_of(ref.path(size))
    taps = DSparkConfig.from_gguf(GGUFFile(dpath)).taps
    tgt, d = _slice(size)
    emb = cpu.SparkCpu(g, layers=[]).emb
    ids = ref.corpus_ids(720, size)
    x = emb[torch.tensor(ids)]
    head_ids = d.table.ids
    w = np.asarray(g.tensor(tgt.image.lm).dequantize(), np.float32)
    ref_d = DSparkCpu(dpath, emb=emb, head_w=torch.from_numpy(w[head_ids]))
    worst: dict[str, float] = {}
    p = 0
    blocks = {300, 700}
    sizes = [8, 3, 1]
    i = 0
    while p < 700:
        S = sizes[i % 3] if p + sizes[i % 3] <= 700 else 700 - p
        i += 1
        tgt.step(x[p:p + S].numpy(), p, logits=False)
        t = torch.from_numpy(np.stack([[tgt.hidden(L, s) for L in taps] for s in range(S)]))
        ref_d.inject(t, p)
        P = p + S
        blk = P in blocks
        anchor = ids[P] if P < len(ids) else 0
        xb = d.block_rows(emb[anchor].numpy(), emb[ref_d.mask_id].numpy()) if blk else None
        d.run(S, p, xb, P)
        p = P
        if blk:
            want_l, want_h = ref_d.block_rows(anchor, P)
            got_l = torch.from_numpy(d.logits().astype(np.float32))
            if P == 300:  # (a sequence's state there: ``snapshot``, restored after the rest)
                snap, again = d.snapshot(P), (xb, got_l.clone())
            worst[f"logits@{P}"] = rel_finite(got_l, want_l)
            worst[f"h@{P}"] = rel_finite(torch.from_numpy(d.h()), want_h)
            agree = float((got_l.argmax(-1) == want_l.argmax(-1)).double().mean())
            worst[f"argmax@{P}"] = 1 - agree
            for L in range(d.cfg.layers):
                lo = max(0, P - 520) // 32 * 32
                for h, (k, v) in enumerate(_records(d, L, lo, P)):
                    worst[f"K{L}.{h}@{P}"] = rel_finite(k, ref_d.K[L][h, lo:P])
                    worst[f"V{L}.{h}@{P}"] = rel_finite(v, ref_d.V[L][h, lo:P])
    d.restore(snap)
    d.run(0, 0, again[0], 300)
    assert torch.equal(torch.from_numpy(d.logits().astype(np.float32)), again[1]), "the block after a restore"
    print({k: round(v, 4) for k, v in worst.items()})
    # (the records within 1.7%, h within 2.6% and the logits within 3.7% (the 4B's: bfp16 GEMV inputs through 5 layers;
    # the model's own decode step against its fp32 model: up to 7.8% a row); a block's argmax at most one row off)
    bound = {"K": 0.02, "V": 0.02, "h": 0.035, "l": 0.045, "a": 0.15}
    assert all(v < bound[k[0]] for k, v in worst.items()), worst


@pytest.mark.parametrize(("size", "n"), [("4B", 600), ("1.7B", 1100)])
def test_dspark_prefill(size: str, n: int) -> None:
    """A prompt of n tokens through the prefill's chunks (512s, then a 64; 1100: two chunks of 512, the second's rows
    past the rings' 20 groups of the first) and the rest as decode steps with injections, against all of it as steps of
    8 with injections: the drafter's records of the window before n within 3%, a block at n within 3% (logits), its
    argmax per row."""
    tgt, d = _slice(size)
    x = cpu.SparkCpu(ref.gguf(size), layers=[]).emb[torch.tensor(ref.corpus_ids(n + 1, size))].numpy()
    mask = x[0]  # (any row: both paths the same block)

    def steps(p: int) -> None:
        while p < n:
            S = min(8, n - p)
            tgt.step(x[p:p + S], p, logits=False)
            d.run(S, p, None, 0)
            p += S

    def block() -> tuple[np.ndarray, list]:
        d.run(0, 0, d.block_rows(x[n], mask), n)
        lo = max(0, n - 520) // 32 * 32
        return d.logits().astype(np.float32), [_records(d, L, lo, n) for L in range(d.cfg.layers)]

    steps(0)
    want_l, want = block()
    tgt.reset()
    d.reset()
    pf = SparkPrefill(tgt, sizes=PREFILL_SIZES, dspark=d).compile().bind()
    p = 0
    for T in pf.plan(0, n):
        m = min(T, n - p)
        pf.run(x[p:p + m], p, T)
        p += m
    steps(p)
    got_l, got = block()
    worst = {"logits": rel_finite(got_l, want_l), "argmax": float((got_l.argmax(-1) != want_l.argmax(-1)).mean())}
    for L in range(d.cfg.layers):
        for h, ((k, v), (wk, wv)) in enumerate(zip(got[L], want[L], strict=True)):
            worst[f"K{L}.{h}"], worst[f"V{L}.{h}"] = rel_finite(k, wk), rel_finite(v, wv)
    print({k: round(v, 4) for k, v in worst.items()})
    # (the two paths' taps differ by the model's prefill against its decode: 3-3.7% the records, 3.4-4.6% the block's
    # logits over prompts of 300 .. 900 (4B); the drafter's part alone: ``test_dspark_prefill_rows``)
    assert all(v < {"argmax": 0.15, "logits": 0.05}.get(k, 0.045) for k, v in worst.items()), worst


@pytest.mark.parametrize(("size", "p0", "T"), [("4B", 64, 256)])
def test_dspark_prefill_rows(size: str, p0: int, T: int) -> None:
    """The drafter's records of a chunk (the staging DS) against the fp32 drafter on the chunk's own tap panels (FT:
    the taps normed by fc_norm, as the fc GEMMs read them): fc, enc.output_norm, per layer K (rotated) and V within
    1.9% (bfp16 g panels and GEMM outputs: the injections' own error, ``test_dspark_rounds``)."""
    tgt, d = _slice(size)
    x = cpu.SparkCpu(ref.gguf(size), layers=[]).emb[torch.tensor(ref.corpus_ids(p0 + T, size))].numpy()
    pf = SparkPrefill(tgt, sizes=PREFILL_SIZES, dspark=d).compile().bind()
    if p0:
        pf.run(x[:p0], 0, p0)
    pf.run(x[p0:], p0, T)
    dev, H, kmax = tgt.dev, tgt.H, pf.kmax
    n = len(d.cfg.taps)
    rb = pf.rb["PXN"]
    dev.fetch("FT")
    ft = dev.view("FT")
    taps = np.stack([bfp16.from_panels(ft[i * pf.ft_rows * rb:][:T * rb], T, kmax)[:, :H] for i in range(n)], 1)
    R = DSparkCpu(spark_dspark_of(ref.path(size)), emb=None, head_w=None)
    g = R.rms(R.mm(torch.from_numpy(taps.astype(np.float32)).reshape(T, -1), R.fc), R.enc_norm)
    pos = torch.arange(p0, p0 + T)
    gb, kvh, hb = AT.OBJS * tgt.T, tgt.ash.kvh, pf.dstage * AT.OBJS * tgt.T
    dev.fetch("DS")
    stg = dev.view("DS")
    worst = {}
    for L, lw in enumerate(R.layers):
        k = R.rope(R.mm(g, lw["k"]).reshape(T, R.nkv, R.hd).transpose(0, 1), pos)
        v = R.mm(g, lw["v"]).reshape(T, R.nkv, R.hd).transpose(0, 1)
        for h in range(kvh):
            a = (L * kvh + h) * hb
            grp = np.stack([stg[a + i * gb:][:gb].reshape(AT.OBJS, tgt.T) for i in range(T // 32)])
            K, V = KV.kv_from_groups(grp, AT.HD, dmajor=True)
            worst[f"K{L}.{h}"], worst[f"V{L}.{h}"] = rel_finite(np.asarray(K, np.float32), k[h]), \
                rel_finite(np.asarray(V, np.float32), v[h])
    print({k: round(v, 4) for k, v in worst.items()})
    assert max(worst.values()) < 0.021, worst


def test_dspark_sequences() -> None:
    """The drafter of a model of three sequences (``DSparkOverlay`` on ``SparkOverlay(sequences=3)``: rings of each,
    selected with the model's; an injection's taps from its staging region, ``run``'s ``rows``): per sequence the
    model's steps (of one sequence, and of the three: a sequence's rows at its slots of the step), the injection of
    its rows and blocks give the blocks' logits and rows (bf16 bits) of the drafter of a model of one sequence over
    the same steps, its taps read in place."""
    from artlite.decode.step import Part

    size, nseq, rounds = "1.7B", 3, 24
    g = ref.gguf(size)
    dg = GGUFFile(spark_dspark_of(ref.path(size)))
    taps = DSparkConfig.from_gguf(dg).taps
    rng = np.random.default_rng(11)
    sizes = rng.integers(1, 9, (nseq, rounds))  # (a sequence's slots of each round's step: 1 .. 8 in a separate step,
    sizes[:, 1::2] = rng.integers(1, 3, (nseq, rounds // 2))  # 1 .. 2 of a step of the three)
    ids = [ref.corpus_ids(200 + 40 * s, size)[40 * s:] for s in range(nseq)]
    emb = [g.rows("token_embd.weight", t) for t in ids]
    mask = g.rows("token_embd.weight", [DSparkConfig.from_gguf(dg).mask_token])[0]

    def block(d: DSparkOverlay, s: int, P: int) -> np.ndarray:
        return d.block_rows(emb[s][P], mask)

    tgt, d = _slice(size)
    want: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for s in range(nseq):  # (alone: a model of one sequence, the taps in place)
        tgt.reset()
        d.reset()
        p, out = 0, []
        for i in range(rounds):
            S = int(sizes[s, i])
            tgt.step(emb[s][p:p + S], p, logits=False)
            d.run(S, p, block(d, s, p + S) if i % 3 == 2 else None, p + S)
            p += S
            if i % 3 == 2:
                out.append((d.logits().view(np.uint16).copy(), d.h().copy()))
        want.append(out)
    d.dev = tgt.dev = None
    tgt = SparkOverlay(g, layers=sorted(taps), vocab=VOCAB, max_positions=4096, sequences=nseq, **needs(dg))
    tgt.load()
    d = DSparkOverlay(tgt, dg)
    d.load()
    assert d.sequences == nseq
    pos, got = [0] * nseq, [[] for _ in range(nseq)]
    for i in range(rounds):
        n = [int(sizes[s, i]) for s in range(nseq)]
        if i % 2:  # (a step of the three; else a step of each)
            parts = tuple(Part(s, pos[s], n[s]) for s in range(nseq))
            tgt.step(np.concatenate([emb[s][pos[s]:pos[s] + n[s]] for s in range(nseq)]), parts, logits=False)
            rows = np.split(tgt.tap_rows(taps, sum(n)), np.cumsum(n)[:-1])
        else:
            rows = []
            for s in range(nseq):
                tgt.select(s)
                tgt.step(emb[s][pos[s]:pos[s] + n[s]], pos[s], logits=False)
                rows.append(tgt.tap_rows(taps, n[s]))
        for s in (2, 0, 1):  # (the drafter's rounds: a sequence's rows from the staging region)
            tgt.select(s)
            assert d.seq == s
            P = pos[s] + n[s]
            d.run(n[s], pos[s], block(d, s, P) if i % 3 == 2 else None, P, rows=rows[s])
            if i % 3 == 2:
                got[s].append((d.logits().view(np.uint16).copy(), d.h().copy()))
            pos[s] = P
    for s in range(nseq):
        for j, ((gl, gh), (wl, wh)) in enumerate(zip(got[s], want[s], strict=True)):
            assert np.array_equal(gl, wl) and np.array_equal(gh, wh), f"sequence {s}, block {j}"
    d.dev = tgt.dev = None


CODE = "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring and two tests."


@pytest.mark.slow
def test_dspark_model() -> None:
    """The whole model on a code prompt, greedy: the text with DSpark rounds the text without drafts (200 tokens), more
    than 3 tokens a round; the text teacher-forced: at each position the drafts against the fp32 drafter's (the fork's
    whole head and Markov bias) on the taps the model's step left: its first draft equal on 90% of positions at least,
    the walk's tokens per round within 5%."""
    size = "4B"
    path = ref.path(size)
    dpath = spark_dspark_of(path)
    require(dpath)
    m = NpuSpark(GGUFFile(path), capacity=1024, dspark=dpath)
    m.lookup = False
    ids = Spark(path).chat([{"role": "user", "content": CODE}])
    texts, rounds = [], []
    for drafts in (0, None):
        m.reset()
        logits = m.forward(ids)
        m.blocks = m.accepted = 0
        texts.append(list(m.generate(logits, 200, Sampling(greedy=True), ids, drafts=drafts)))
        rounds.append(m.blocks)
    assert texts[0] == texts[1]
    tpr = len(texts[1]) / max(rounds[1], 1)
    seq = ids + texts[1]
    taps: list[np.ndarray] = []
    run = m.steps

    def steps(tokens: list[int], **kw: object) -> np.ndarray | None:  # (the taps each step leaves)
        out = run(tokens, **kw)
        taps.append(np.stack([m.overlay.hidden(L, 0) for L in m.dspark.cfg.taps]))
        return out

    m.steps = steps
    npu = [d for d, _ in m.forced_drafts(seq, len(ids))]
    m.steps = run
    g = ref.gguf(size)
    R = DSparkCpu(dpath, emb=cpu.SparkCpu(g, layers=[]).emb,
                  head_w=torch.from_numpy(np.asarray(g.tensor(m.overlay.image.lm).dequantize(), np.float32)))
    # (the CPU's taps: the prompt's from a decode-step run of it, then the forced steps')
    m.reset()
    pre: list[np.ndarray] = []
    for p in range(0, len(ids), 8):
        m.overlay.step(m._rows(ids[p:p + 8]), p, logits=False)
        S = min(8, len(ids) - p)
        pre += [np.stack([m.overlay.hidden(L, s) for L in m.dspark.cfg.taps]) for s in range(S)]
    allt = torch.from_numpy(np.stack(pre + taps[:-1]))
    cpu_d, eq = [], []
    first = len(ids)
    for i, d in enumerate(npu):  # (the CPU's: the whole head, the Markov bias over the whole vocabulary: the fork's)
        P = first + i
        if R.K[0].shape[1] < P:
            R.inject(allt[R.K[0].shape[1]:P], R.K[0].shape[1])
        c, _, _ = R.draft(seq[P], P)
        cpu_d.append(c)
        eq.append(bool(d) and d[0] == c[0])

    def walk(ds: list[list[int]]) -> float:
        i, n = 0, 0
        while i < len(ds):
            k = next((j for j, (a, b) in enumerate(zip(ds[i], seq[first + i + 1:])) if a != b), len(ds[i]))
            i, n = i + k + 1, n + 1
        return len(ds) / n

    print(f"{size}: {tpr:.2f} tokens a round; forced: first draft equal {np.mean(eq):.3f}, tokens per round NPU "
          f"{walk(npu):.3f} CPU {walk(cpu_d):.3f}")
    assert tpr > 3 and np.mean(eq) >= 0.9 and abs(walk(npu) / walk(cpu_d) - 1) < 0.05
