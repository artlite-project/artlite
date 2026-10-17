"""The Gemma 4 prefill on the NPU, operator by operator, on the GGUF file's decode image (operators/gemma4/image.py):
the attention prep and flash attention, the per-layer input operator, a slice's caches after the prefill program (the
GEMMs: test_prefill_gemm.py, the norm: test_prefill_norm.py).
"""

import os

import numpy as np
import pytest
import torch

import artlite
from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig, gemma_gguf
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.gemma4.attn_prefill import Kind
from artlite.operators.gemma4.image import GemmaImage
from artlite.operators.gemma4.prefill import plan_gemm

from .common import essay

LAYERS = (0, 5)  # a sliding-window and a full-attention layer
_image: dict[str, tuple] = {}


def _img() -> tuple[GGUFFile, GemmaImage, np.ndarray]:
    path = gemma_gguf("Q4_0")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    if not _image:
        g = GGUFFile(path)
        img = GemmaImage(g, GemmaConfig.from_gguf(g), layers=list(LAYERS), vocab=COLS * 64)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W, release=False)
        _image["x"] = (g, img, W)
    return _image["x"]


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm())


# (head_dim, chunk position, tokens, K panels per record)
def _q_values(slots: np.ndarray, kind: Kind) -> np.ndarray:
    """Q panel slots (..., qp) uint8 -> (..., 8 queries, hd) the values they hold (hi + lo)."""
    lead = slots.shape[:-1]
    if kind.swa:
        p = slots.reshape(*lead, kind.nd, 2, 8, 9)  # [db][hi / lo][query][block]
    else:  # (per half of the d-blocks: hi panels, then lo)
        p = np.swapaxes(slots.reshape(*lead, 2, 2, kind.nd // 2, 8, 9), -4, -3).reshape(*lead, kind.nd, 2, 8, 9)
    v = bfp16.block_values(p[..., 0, :, :]) + bfp16.block_values(p[..., 1, :, :])  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, kind.hd)


ATTN = [(256, 0, 64, 2), (256, 576, 64, 2), (512, 0, 64, 2), (512, 96, 64, 2), (256, 32, 256, 2), (256, 64, 288, 2),
        (512, 96, 288, 2), (256, 576, 64, 1), (512, 96, 64, 1), (256, 64, 288, 1), (512, 96, 288, 1)]


@pytest.mark.parametrize(("hd", "t0", "T", "kp"), ATTN,
                         ids=[f"hd{hd}-p{t0}-T{T}{'' if kp == 2 else '-single'}" for hd, t0, T, kp in ATTN])
def test_attn_prefill(hd: int, t0: int, T: int, kp: int) -> None:
    """GemmaPrepOperator then GemmaAttnOperator (one fused runlist) on random projections of a chunk at t0 after a
    cache the host filled, in both K formats: the Q panels and the K / V records against their fp64 model (Q, K hi + lo
    within the panels' ~14 bits, single K and V within bfp16), the output panels against the exact attention (causal; head_dim 256 within the window)
    of the cores' own operands (their Q panels, the cache they read): within 1.1 x the error of the exact output
    rounded to bfp16 (the output panels; P as hi + lo panels)."""
    from artlite.operators import FusedOperator
    from artlite.operators.gemma4 import attn_prefill as AP
    from artlite.operators.gemma4.overlay import GUARD
    from artlite.operators.overlay import kv as KV

    _, img, _ = _img()
    cfg = img.cfg
    L = 0 if hd == 256 else 5
    kind = AP.kind_of(cfg, hd, kp)
    max_pos = 1024
    lay = plan_gemm(img, img.region(L, "qkv"), T).layout
    guard = GUARD if kind.swa else 0
    n_groups = guard + max_pos // kind.pg
    kvb = n_groups * kind.group
    prep = AP.GemmaPrepOperator(T=T, cfg=cfg, hd=hd, layout=lay, kv_bytes=kvb, cs_positions=max_pos, kp=kp)
    attn = AP.GemmaAttnOperator(T=T, cfg=cfg, hd=hd, kv_bytes=kvb, kp=kp)
    fused = FusedOperator(name=f"g4_attn{hd}_{kp}", runlist=[(prep, "G", "CS", "PRM", "QS", "KV"), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", "PRM"], output_args=["A"]).compile()
    gen = torch.Generator().manual_seed(t0 + hd)
    c_score = 1.7 / np.log(2)
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(AP.cs_table(cfg, kind, max_pos).view(np.uint8).reshape(-1))
    fused.get_buffer("PRM").view(torch.uint8)[:] = torch.from_numpy(AP.prm_object(kind, c_score).view(np.uint8))
    kv = fused.get_buffer("KV").view(torch.uint8).numpy().reshape(COLS // 4, n_groups, kp + 1, AP.OBJ)
    kv[:] = 0
    hist = t0 // kind.pg  # (the cache before the chunk: random normed K, V)
    for h in range(COLS // 4):
        k_h = torch.nn.functional.normalize(torch.randn(t0, hd, generator=gen), dim=-1).numpy() * hd ** 0.5
        v_h = torch.randn(t0, hd, generator=gen).numpy()
        if t0:
            kv[h, guard:guard + hist] = KV.kv_groups(k_h, v_h, AP.OBJ, kp)
    q = torch.randn(T, COLS, hd, generator=gen) * 3
    k = torch.randn(T, COLS // 4, hd, generator=gen) * 2
    v = torch.randn(T, COLS // 4, hd, generator=gen)
    G = np.zeros((T, lay.N), np.float32)
    for c in range(COLS):
        h, j = divmod(c, 4)
        G[:, c * lay.pitch + lay.off["q"] + np.arange(hd)] = q[:, c]
        src = (k if j < 2 else v)[:, h, (j % 2) * hd // 2:(j % 2 + 1) * hd // 2]
        G[:, c * lay.pitch + lay.off["kv"] + np.arange(hd // 2)] = src
    fused.get_buffer("G")[:] = torch.from_numpy(G.reshape(-1)).bfloat16()
    fused.sync_inputs()
    fused.set_params(prep.params(t0, guard) | attn.params(t0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    def nr(x: torch.Tensor, rot: bool) -> np.ndarray:  # x / rms(x), rotated (fp64), x (T, heads, hd) bf16 values
        x = x.bfloat16().double().numpy()
        y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + cfg.rms_norm_eps)
        if rot:
            cs = AP.cs_table(cfg, kind, max_pos)[t0:t0 + T].astype(np.float64)
            c_, s_ = cs[:, None, :kind.rot], cs[:, None, kind.rot:]
            a, b = y[..., :kind.rot].copy(), y[..., hd // 2:hd // 2 + kind.rot].copy()
            y[..., :kind.rot], y[..., hd // 2:hd // 2 + kind.rot] = a * c_ - b * s_, a * s_ + b * c_
        return y

    slots = fused.get_buffer("QS").view(torch.uint8).numpy()[:T // 8 * COLS * kind.qp]
    if not kind.swa:  # (pass-major: [pass][KV head][token group][column of the head])
        slots = slots.reshape(T // 32, COLS // 4, 4, 4, kind.qp).transpose(0, 2, 1, 3, 4)
    slots = slots.reshape(T // 8, COLS, kind.qp).transpose(1, 0, 2)  # (the attention's QS: whole passes, padded)
    qv = _q_values(slots, kind).reshape(COLS, T, hd)  # [head][token]
    assert _rel(torch.from_numpy(qv), torch.from_numpy(nr(q, True).transpose(1, 0, 2) * c_score)) < 2e-4
    K, V = {}, {}
    n = (t0 + T) // kind.pg
    for h in range(COLS // 4):
        K[h], V[h] = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[h, guard:guard + n], hd))
        assert _rel(torch.from_numpy(K[h][t0:]), torch.from_numpy(nr(k, True)[:, h])) < (2e-4 if kp == 2 else 1e-2), h
        assert _rel(torch.from_numpy(V[h][t0:]), torch.from_numpy(nr(v, False)[:, h])) < 1e-2, h
    A = bfp16.from_panels(fused.get_buffer("A")[:T * COLS * hd * 9 // 16].clone(), T, COLS * hd).numpy()
    A = A.reshape(T, COLS, hd)
    want = np.zeros((T, COLS, hd))
    pos = np.arange(t0, t0 + T)[:, None]
    keys = np.arange(t0 + T)[None]
    masked = (keys > pos) | ((pos - keys >= cfg.sliding_window) if kind.swa else False)
    for c in range(COLS):
        sc = qv[c].astype(np.float64) @ K[c // 4].T  # (log2 units)
        sc[masked] = -np.inf
        pr = np.exp2(sc - sc.max(-1, keepdims=True))
        want[:, c] = (pr @ V[c // 4]) / pr.sum(-1, keepdims=True)
    err = _rel(torch.from_numpy(A), torch.from_numpy(want))
    eq = _rel(torch.from_numpy(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape)),
              torch.from_numpy(want))
    print(f"attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e})")
    assert err < 1.1 * eq


def _ple_reference(y: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
    """h (T, 8 SLOT) fp32 before the panels: y (T, 8 SLOT), pli (T, 8 PER)."""
    from artlite.operators.gemma4 import prefill_ple as PL

    T = y.shape[0]
    g = torch.nn.functional.gelu(y.bfloat16().float(), approximate="tanh").bfloat16().float()
    h = torch.zeros(T, COLS, PL.SLOT)
    h[:, :, :PL.PER] = g.view(T, COLS, PL.SLOT)[:, :, :PL.PER] * pli.bfloat16().float().view(T, COLS, PL.PER)
    return h.reshape(T, -1)


def test_ple() -> None:
    """PleOperator against its model (gelu through the table: within 1 bf16 ULP; the panels' bfp16 rounding)."""
    from artlite.operators.gemma4.prefill_ple import PleOperator

    T = 256
    op = PleOperator(T=T).compile()
    dev = op.device
    g = torch.Generator().manual_seed(5)
    y = (torch.randn(T, COLS * 64, generator=g) * 2).bfloat16()
    pli = (torch.randn(T, COLS * 32, generator=g)).bfloat16()
    A = artlite.from_torch(torch.zeros(T * COLS * 64 * 9 // 16, dtype=torch.bfloat16), dev)
    op.dispatch(artlite.from_torch(y.clone(), dev), artlite.from_torch(pli.clone(), dev), A)
    A.sync_from_device()
    got = bfp16.from_panels(A.to_torch(), T, COLS * 64).double()
    want = _ple_reference(y, pli).double()
    eq = _rel(torch.from_numpy(bfp16.values(want.float().numpy())).double(), want)
    err = _rel(got, want)
    print(f"ple {err:.2e} (bfp16 {eq:.2e})")
    assert err < 1.2 * eq



@pytest.mark.slow
@pytest.mark.parametrize(("n", "p0"), [(300, 0), (300, 64)], ids=["two-chunks", "after-decode"])
def test_prefill_caches(n: int, p0: int) -> None:
    """The prefill program (GemmaPrefill) on a slice of 6 layers (5 sliding-window, 1 full) leaves the caches the
    decode path leaves for the same essay tokens, up to the rounding of the two paths' GEMM inputs (single bfp16
    panels here, hi / lo there): p0 tokens as decode steps first (a prefill continued after a decode), then n in
    chunks of 256 and the rest."""
    from artlite.operators.gemma4.overlay import GemmaOverlay
    from artlite.operators.gemma4.prefill import GemmaPrefill
    from models.gemma4.tokenizer import GemmaTokenizer

    from .gemma_cpu import GemmaCpu

    path = gemma_gguf("Q4_0")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    layers = list(range(6))
    ids = GemmaTokenizer(g).encode(essay("apple.txt"), add_bos=True)[:p0 + n]
    host = GemmaCpu(g)
    x = host.embed(ids)
    pli = host.per_layer_inputs(ids, x, bf16=True)
    x = x.bfloat16().float()  # (the app's inputs: bf16)
    pli = pli.bfloat16()
    ov = GemmaOverlay(g, layers=layers, vocab=COLS * 64, max_positions=1024)
    ov.load()
    for t in range(0, p0 + n, 4):  # the decode path
        s = min(4, p0 + n - t)
        ov.step(x[t:t + s], pli[t:t + s], t, logits=False)
    want = {L: ov.read_kv(L, range(p0 + n)) for L in layers}
    ov.reset()
    for t in range(0, p0, 4):
        ov.step(x[t:t + 4], pli[t:t + 4], t, logits=False)
    pf = GemmaPrefill(ov).compile().bind()
    t = p0
    for T in pf.plan(p0, n):
        m = min(T, p0 + n - t)
        pf.run(x[t:t + m], pli[t:t + m], t, T)
        t += m
    worst = {}
    for L in layers:
        k, v = ov.read_kv(L, range(p0 + n))
        worst[f"K.{L}"], worst[f"V.{L}"] = _rel(k, want[L][0]), _rel(v, want[L][1])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < 0.05 for v in worst.values()), worst
