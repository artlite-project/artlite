"""The attention prefill operators (models/qwen35/attn_prefill.py) on the NPU, on an attention layer's own inputs
(llama.cpp's prompt-mode projections of an essay laid out as the attn_in GEMMs write them), two chunks: the second
at position T, with padding tokens after its valid ones."""

import os

import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_gguf
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.prefill import gemm as PG
from models.qwen35 import attn as AT
from models.qwen35 import attn_prefill as AP
from models.qwen35.image import QwenImage

from ..common import essay
from . import ref

MAX_POS = 1024


def _bf(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


def _slot_order(slots: np.ndarray, two: bool) -> np.ndarray:
    """QS slots (..., SLOT) uint8 as stored ([half 2][value 2][d-block 16][72 B], two streams [quarter 4][value 2]
    [d-block 8][72 B]: ``AP.slot_walk``) -> [d-block 32][the two values' 144 B]."""
    lead, n = slots.shape[:-1], 4 if two else 2
    return np.ascontiguousarray(slots.reshape(*lead, n, 2, AP.ND // n, 72).swapaxes(-3, -2)).reshape(*lead, AP.ND, 144)


def _q_panels(slots: np.ndarray, two: bool) -> np.ndarray:
    """Q slots (..., SLOT) uint8 -> (..., 8 queries, 256) the Q values they hold (hi + lo)."""
    lead = slots.shape[:-1]
    p = _slot_order(slots, two).reshape(*lead, AP.ND, 2, 8, 9)  # [db][hi / lo][query][block]
    v = bfp16.block_values(p[..., 0, :, :]) + bfp16.block_values(p[..., 1, :, :])  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, AP.HD)


def _gate_sigmoid(slots: np.ndarray, two: bool) -> np.ndarray:
    """sigmoid(gate) slots (..., SLOT) uint8 -> (..., 8, 256) (fp32 of the bf16 stored: per d-block [query][8])."""
    lead = slots.shape[:-1]
    g = _slot_order(slots, two).view(np.uint16).reshape(*lead, AP.ND, 72)[..., :64].astype(np.uint32) << 16
    g = g.view(np.float32).reshape(*lead, AP.ND, 8, 8)  # [db][query][8]
    return np.moveaxis(g, -3, -2).reshape(*lead, 8, AP.HD)


def _qs_slots(fused: FusedOperator, T: int, qc: int, qg: int) -> np.ndarray:
    """QS (pass-major: [32 QG-token pass][column][token group 4 QG][Q | gate][head][SLOT]) as [column][token group][Q |
    gate][head][SLOT] uint8."""
    qs = fused.get_buffer("QS").view(torch.uint8).numpy().reshape(T // (32 * qg), COLS, 4 * qg, 2, qc, AP.SLOT)
    return np.ascontiguousarray(qs.swapaxes(0, 1)).reshape(COLS, T // 8, 2, qc, AP.SLOT)


class _Setup:
    """The model, the layer, two chunks of essay tokens and llama.cpp's tensors of them."""

    def __init__(self, size: str, T: int, n2: int):
        path = qwen_gguf("Q4_K_M", size)
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")
        self.g = g = GGUFFile(path)
        self.cfg = cfg = QwenConfig.from_gguf(g)
        self.L = L = next(i for i, a in enumerate(cfg.attention) if a)
        img = QwenImage(g, cfg, layers=[L], vocab=COLS * 64)
        self.lay = PG.plan_phase(img, img.region(L, "attn_in"), T, pitch=PG.phase_pitch(img, "attn_in", T)).layout
        self.ash = AT.AttnShape(cfg, MAX_POS, a_bytes=img.tile_bytes)
        self.T, self.n2, self.qc = T, n2, cfg.num_attention_heads // COLS
        self.kvh = cfg.num_key_value_heads
        self.per = self.kvh * AT.HD // COLS  # (K / V dims of a column)
        self.tokens = ref.llama_tokenize(path, essay("apple.txt"))[:T + n2]
        n = len(self.tokens)
        G_ = ref.golden(path, self.tokens, layers=(L,), names="Qcur_full|Kcur|Vcur|Qcur|gate_sigmoid|attn_gated",
                        logits="none")
        self.qfull = _bf(np.asarray(G_["Qcur_full", L], np.float32).reshape(n, -1))
        self.k = _bf(np.asarray(G_["Kcur", L], np.float32).reshape(n, -1))
        self.v = _bf(np.asarray(G_["Vcur", L], np.float32).reshape(n, -1))
        self.gold = G_
        self.qn = np.asarray(g.tensor(f"blk.{L}.attn_q_norm.weight").raw, np.float32).reshape(-1)
        self.kn = np.asarray(g.tensor(f"blk.{L}.attn_k_norm.weight").raw, np.float32).reshape(-1)

    def G(self, t0: int, n: int) -> np.ndarray:
        """attn_in's output of tokens t0 .. t0 + n - 1 (rows n .. T - 1 zero)."""
        lay, qc = self.lay, self.qc
        G = np.zeros((self.T, lay.N), np.float32)
        for c in range(COLS):
            G[:n, c * lay.pitch + lay.off["q"] + np.arange(2 * qc * AT.HD)] = \
                self.qfull[t0:t0 + n, 2 * qc * AT.HD * c:2 * qc * AT.HD * (c + 1)]
            e = self.per
            G[:n, c * lay.pitch + lay.off["k"] + np.arange(e)] = self.k[t0:t0 + n, e * c:e * (c + 1)]
            G[:n, c * lay.pitch + lay.off["v"] + np.arange(e)] = self.v[t0:t0 + n, e * c:e * (c + 1)]
        return G

    def nr(self, x: np.ndarray, w: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """rms(x) w, rotated (fp64): x (n, heads, 256) at positions pos."""
        x = np.asarray(x, np.float64)
        y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + self.cfg.rms_norm_eps) * np.asarray(w, np.float64)
        th = self.cfg.rope_theta(list(pos)).astype(np.float64)
        return ref.rope_partial(y, th).astype(np.float64)


def _prep_fused(s: _Setup) -> tuple[AP.AttnPrepOperator, FusedOperator]:
    op = AP.AttnPrepOperator(T=s.T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    fused = FusedOperator(name="attn_prep", runlist=[(op, "G", "CS", "NRM", "QS", "KV")], input_args=["G", "CS", "NRM"],
                          output_args=["QS"]).compile()  # (KV: scratch, zeroed by the host: flushed with the inputs)
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(AP.cs_table(s.cfg, MAX_POS).view(np.uint8).reshape(-1))
    nrm = np.concatenate([s.qn, s.kn]).astype(np.float32)
    fused.get_buffer("NRM").view(torch.uint8)[:] = torch.from_numpy(nrm.view(np.uint8))
    fused.get_buffer("KV").zero_()
    return op, fused


def _run(fused: FusedOperator, params: dict[str, int], G: np.ndarray) -> None:
    fused.get_buffer("G")[:] = torch.from_numpy(G.reshape(-1)).bfloat16()
    fused.sync_inputs()
    fused.set_params(params)
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()


@pytest.mark.parametrize("size", ["9B", "1B", "35B-A3B"])
def test_attn_prep(size: str) -> None:
    T, n2 = 64, 45
    s = _Setup(size, T, n2)
    op, fused = _prep_fused(s)
    qc, hd = s.qc, AT.HD
    for t0, n in [(0, T), (T, n2)]:
        _run(fused, op.params(t0), s.G(t0, n))
        pos = np.arange(t0, t0 + n)
        slots = _qs_slots(fused, T, qc, op.qg)
        q = _q_panels(slots[:, :, 0], op.two).transpose(0, 2, 1, 3, 4).reshape(COLS * qc, T, hd)[:, :n]  # [head][token]
        sg = _gate_sigmoid(slots[:, :, 1], op.two).transpose(0, 2, 1, 3, 4).reshape(COLS * qc, T, hd)[:, :n]
        qg = s.qfull[t0:t0 + n].reshape(n, -1, 2, hd)
        q_want = s.nr(qg[:, :, 0], s.qn, pos).transpose(1, 0, 2) * np.log2(np.e) / 16
        assert ref.rel_err(q, q_want) < 2e-4, t0  # (hi + lo panels: ~14 bits below a block's maximum)
        q_gold = np.asarray(s.gold["Qcur", s.L], np.float32).reshape(-1, COLS * qc, hd)[t0:t0 + n].transpose(1, 0, 2)
        assert ref.rel_err(q / (np.log2(np.e) / 16), q_gold) < 1e-2, t0
        sg_want = 1 / (1 + np.exp(-np.asarray(qg[:, :, 1], np.float64))).transpose(1, 0, 2)
        assert np.abs(sg - sg_want).max() < 2 ** -8, t0
    # the cache: both chunks' positions of every KV head (K normed, rotated; V as stored)
    kv = fused.get_buffer("KV").view(torch.uint8).numpy().reshape(s.kvh, -1, 3, 9280)
    pos = np.arange(T + n2)
    kr = s.k[:T + n2].reshape(T + n2, s.kvh, hd)
    k_want = s.nr(kr, s.kn, pos)
    for h in range(s.kvh):
        K, V = KV.kv_from_groups(kv[h, :(T + n2 + 31) // 32], AT.HD)
        assert ref.rel_err(K[:T + n2], k_want[:, h]) < 2e-4, h
        assert ref.rel_err(V[:T + n2], s.v[:T + n2, 256 * h:256 * (h + 1)]) < 1e-2, h
        k_gold = np.asarray(s.gold["Kcur.1", s.L], np.float32).reshape(-1, s.kvh, hd)[:T + n2, h]
        assert ref.rel_err(K[:T + n2], k_gold) < 1e-2, h


@pytest.mark.parametrize(("size", "T"), [("9B", 64), ("27B", 64), ("1B", 64), ("1B", 192), ("35B-A3B", 64)])
def test_attn_prefill(size: str, T: int) -> None:
    """AttnPrep then PrefillAttn (one fused runlist) over two chunks: the output panels against the exact attention
    of the cores' own operands (their Q panels, the cache they wrote, their sigmoid(gate): within 0.7%, the bfp16
    output and P panels) and llama.cpp's attn_gated (~1% rms). The 1B: passes of 64 queries (one head per column)."""
    from artlite.operators import bfp16

    n2 = 45
    s = _Setup(size, T, n2)
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes)
    fused = FusedOperator(name="attn_prefill", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(AP.cs_table(s.cfg, MAX_POS).view(np.uint8).reshape(-1))
    fused.get_buffer("NRM").view(torch.uint8)[:] = torch.from_numpy(np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8))
    fused.get_buffer("KV").zero_()
    H, qc, hd = s.cfg.num_attention_heads, s.qc, AT.HD
    for t0, n in [(0, T), (T, n2)]:
        _run(fused, prep.params(t0) | attn.params(t0), s.G(t0, n))
        A = bfp16.from_panels(fused.get_buffer("A").clone(), T, H * hd).numpy().reshape(T, H, hd)[:n]
        slots = _qs_slots(fused, T, qc, prep.qg)
        q = _q_panels(slots[:, :, 0], prep.two).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n] * (16 / np.log2(np.e))
        sg = _gate_sigmoid(slots[:, :, 1], prep.two).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n]
        kv = fused.get_buffer("KV").view(torch.uint8).numpy().reshape(s.kvh, -1, 3, 9280)
        want = np.zeros((n, H, hd))
        for h in range(s.kvh):
            K, V = (np.asarray(x, np.float64)[:t0 + n] for x in KV.kv_from_groups(kv[h, :(t0 + n + 31) // 32], AT.HD))
            for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
                sc = q[g].astype(np.float64) @ K.T / 16
                sc[np.arange(t0, t0 + n)[:, None] < np.arange(t0 + n)[None]] = -np.inf
                pr = np.exp(sc - sc.max(-1, keepdims=True))
                want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g]
        # (1B: a peakier softmax, its largest P 0.45 on average (9B 0.28): single V rows carry more of an output, the
        # largest error ~2 ULPs of a bfp16 block; the rms error as the 9B's)
        assert ref.rel_err(A, want) < (1.5e-2 if size == "1B" else 1e-2), t0
        assert np.linalg.norm(A - want) / np.linalg.norm(want) < 1.2e-2, t0
        gold = np.asarray(s.gold["attn_gated", s.L], np.float32).reshape(-1, H, hd)[t0:t0 + n]
        assert ref.rel_err(A, gold) < 3e-2, t0
