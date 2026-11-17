"""The DSpark drafter on the NPU (``models.spark.dspark.DSparkOverlay``) against the fp32 drafter (``dspark_cpu``) on the
same taps: the model's layers ``taps`` alone (a slice: the taps are what those layers read in it), decode steps of an
essay's tokens, after each the drafter's injection of the step's rows (the taps read back from the model's ACT feed the
CPU's); the rings' records of the window (K, V per layer and KV head) and a block's logits over the head's rows and its
output-normed rows."""

import numpy as np
import pytest
import torch

from artlite.decode import attn256 as AT
from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import DSparkConfig, spark_dspark_of
from models.spark.dspark import DSparkOverlay, needs
from models.spark.overlay import SparkOverlay

from ..model_files import require
from ..numerics import rel_finite
from . import cpu, ref
from .dspark_cpu import DSparkCpu

VOCAB = 512


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


@pytest.mark.parametrize("size", ["1.7B", "4B"])
def test_dspark_rounds(size: str) -> None:
    """Steps of 8, 3 and 1 slots over 700 positions (the rings' 20 groups wrap), each followed by the injection of its
    rows, blocks at 300 and 700 (the block's records then overwritten by the next injection): the window's records, the
    block's logits and h against the CPU's."""
    g = ref.gguf(size)
    dpath = spark_dspark_of(ref.path(size))
    require(dpath)
    dg = GGUFFile(dpath)
    taps = DSparkConfig.from_gguf(dg).taps
    tgt = SparkOverlay(g, layers=sorted(taps), vocab=VOCAB, max_positions=1024, **needs(dg))
    tgt.load()
    d = DSparkOverlay(tgt, dg)
    d.load()
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
            worst[f"logits@{P}"] = rel_finite(got_l, want_l)
            worst[f"h@{P}"] = rel_finite(torch.from_numpy(d.h()), want_h)
            agree = float((got_l.argmax(-1) == want_l.argmax(-1)).double().mean())
            worst[f"argmax@{P}"] = 1 - agree
            for L in range(d.cfg.layers):
                lo = max(0, P - 520) // 32 * 32
                for h, (k, v) in enumerate(_records(d, L, lo, P)):
                    worst[f"K{L}.{h}@{P}"] = rel_finite(k, ref_d.K[L][h, lo:P])
                    worst[f"V{L}.{h}@{P}"] = rel_finite(v, ref_d.V[L][h, lo:P])
    print({k: round(v, 4) for k, v in worst.items()})
    # (the records within 1.7%, h within 2.6% and the logits within 3.7% (the 4B's: bfp16 GEMV inputs through 5 layers;
    # the model's own decode step against its fp32 model: up to 7.8% a row); a block's argmax at most one row off)
    bound = {"K": 0.02, "V": 0.02, "h": 0.035, "l": 0.045, "a": 0.15}
    assert all(v < bound[k[0]] for k, v in worst.items()), worst
