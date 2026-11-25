"""Inkling 2's prefill on the NPU (``models/inkling2/prefill.py``) against the CPU model (``ref.Model``): a slice of
visits (the attention parts: ``ffn=False``), the residual it leaves, the visits' K / V records, the post convs' history
windows and the K / V raw rings the decode continues from."""

import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, inkling_gguf
from artlite.operators.design import COLS
from artlite.prefill.attn.ink import RING_BYTES
from models.inkling2.overlay import InklingOverlay
from models.inkling2.prefill import InklingPrefill

from ..model_files import require
from ..numerics import rel
from .ref import Model, VisitCache, rms


@pytest.fixture(scope="module")
def ink() -> tuple[GGUFFile, InklingConfig]:
    path = inkling_gguf()
    require(path)
    g = GGUFFile(path)
    return g, InklingConfig.from_gguf(g)


def _caches(ov: InklingOverlay, v: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Visit v's K, V (n, KV heads, hd) of its caches' whole records below n."""
    dev, hd = ov.dev, ov.hd
    ks, vs = [], []
    for h in range(ov.kv_heads):
        b = ov.kv_base(v, h)
        ng = -(-n // ov.pg)
        dev.fetch("KV", b, ng * ov.G * ov.T)
        groups = dev.view("KV")[b:b + ng * ov.G * ov.T].reshape(ng, ov.G, ov.T)
        k, vv = KV.kv_from_groups(groups, hd)
        ks.append(k[:n // 8 * 8])
        vs.append(vv[:n // 8 * 8])
    return np.stack(ks, 1), np.stack(vs, 1)


def _ring(ov: InklingOverlay, v: int) -> np.ndarray:
    """Visit v's raw ring (slot, K / V, KV head x hd) fp32: position p's raw rows at slot p % 8."""
    dev = ov.dev
    dev.fetch("KV", ov.raw[v], COLS * RING_BYTES)
    r = dev.view("KV")[ov.raw[v]:ov.raw[v] + COLS * RING_BYTES].reshape(COLS, RING_BYTES)[:, :8192]
    return r.copy().view(np.float32).reshape(COLS, 8, 2, -1).transpose(1, 2, 0, 3).reshape(8, 2, -1)


@pytest.mark.parametrize("visits, chunks", [(6, (200,)), (6, (512, 188)), (6, (2000,)), (48, (200,))], ids=str)
def test_visits(ink: tuple[GGUFFile, InklingConfig], visits: int, chunks: tuple[int, ...]) -> None:
    """The first ``visits`` visits without their FFNs (6: the prelude; 48: the passes' messages and controllers too), a
    prompt in chunks of the given tokens from position 0 (a chunk of two sub-chunks; one after a history; one of 2048:
    its prep in four parts, the attention's 64 passes). (With the
    FFNs the routing's near ties make the paths diverge: ``test_moe.test_moe_block`` checks the block on the NPU's
    routing.)"""
    g, cfg = ink
    n = sum(chunks)
    ov = InklingOverlay(g, visits=list(range(visits)), max_positions=max(1024, -(-n // 1024) * 1024), ffn=False)
    ov.load()
    pf = InklingPrefill(ov, sizes=tuple(sorted({256, *(-(-c // 256) * 256 for c in chunks)})), ffn=False).compile().bind()
    toks = np.random.default_rng(3).integers(0, cfg.vocab_size, n)
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * tn
    m = Model(g, cfg)
    caches: dict[int, VisitCache] = {}

    def run(vis: object, cur: torch.Tensor, blocks: list[torch.Tensor], messages: torch.Tensor | None) -> torch.Tensor:
        if vis.index >= visits:
            return cur
        return m.visit(vis, cur, blocks, caches.setdefault(vis.index, VisitCache()), 0, messages, ffn=False)

    with torch.no_grad():
        if visits == len(cfg.visits):
            cur, _ = m.walk(x, run)
        else:  # (the prelude: no pass restarts)
            cur, blocks = x, []
            for v in ov.layers:
                cur = run(cfg.visits[v], cur, blocks, None)
    ov.reset()
    p = 0
    for c in chunks:
        pf.run(x[p:p + c].numpy(), p)
        p += c
    last = ov.layers[-1]
    err = rel(pf.state(pf.walk[last].new or pf.walk[last].cur, 0, chunks[-1]), cur[n - chunks[-1]:].numpy())  # (the last chunk's)
    print(f"cur after visit {last}: {err:.2e}")
    errs = [err]
    nk = n // 8 * 8
    for v in ov.layers:
        c = caches[v]
        k, vv = _caches(ov, v, n)
        ring = _ring(ov, v)[np.arange(n - 3, n) % 8]
        e = (rel(k, c.k[:nk].numpy()), rel(vv, c.v[:nk].numpy()), rel(ring[:, 0], c.hist["k"].numpy()),
             rel(ring[:, 1], c.hist["v"].numpy()), rel(ov.history(v, 0, 3), c.hist["attn"].numpy()))
        print(f"visit {v}: K {e[0]:.2e}, V {e[1]:.2e}, the raw rings' {e[2]:.2e} {e[3]:.2e}, the post conv's window "
              f"{e[4]:.2e}")
        errs += e
    # (the prefill's GEMMs on bfp16 panels: ~1.35 2^-9 sqrt(K / 64) of a row, 1.8% at K 3072; the windows' 3 rows of o
    # up to 4.4e-2 after 48 visits)
    assert max(errs) < 5e-2, errs
