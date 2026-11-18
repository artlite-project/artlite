"""The attention prefill operators (artlite/prefill/attn256.py) on the NPU, on an attention layer's own inputs
(llama.cpp's prompt-mode projections of an essay laid out as the attn_in GEMMs write them), two chunks: the second
at position T, with padding tokens after its valid ones; non-causal (d3-lite's), one chunk with padding tokens."""


import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode import attn256 as AT
from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, decision_gguf, qwen_gguf
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.prefill import attn256 as AP
from artlite.prefill import gemm as PG
from models.qwen35.image import QwenImage

from ..common import essay
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..numerics import bf16_round, rel_max
from ..qwen35 import ref

MAX_POS = 1024


def _fill_cs(fused: FusedOperator, cfg: QwenConfig, positions: np.ndarray | None = None) -> None:
    """The CS buffer: the cos / sin rows of every text position (MAX_POS), the last len(positions) rows those of
    ``positions`` (a chunk of packed segments' own positions: ``prep.params(p0, MAX_POS - len(positions))``)."""
    cs = AP.cs_table(cfg, MAX_POS)
    if positions is not None:
        cs[MAX_POS - len(positions):] = AP.cs_rows(cfg, positions)
    fused.get_buffer("CS").view(np.uint8)[:] = cs.view(np.uint8).reshape(-1)


def _q_panels(slots: np.ndarray) -> np.ndarray:
    """Q slots (..., SLOT) uint8 ([d-block 32][hi | lo panel]) -> (..., 8 queries, 256) the Q values they hold (hi +
    lo)."""
    lead = slots.shape[:-1]
    p = slots.reshape(*lead, AP.ND, 2, 8, 9)  # [db][hi / lo][query][block]
    v = bfp16.block_values(p[..., 0, :, :]) + bfp16.block_values(p[..., 1, :, :])  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, AP.HD)


def _gate_sigmoid(slots: np.ndarray) -> np.ndarray:
    """sigmoid(gate) slots (..., SLOT) uint8 -> (..., 8, 256) (fp32 of the bf16 stored: per d-block [query][8])."""
    lead = slots.shape[:-1]
    g = np.ascontiguousarray(slots).view(np.uint16).reshape(*lead, AP.ND, 72)[..., :64].astype(np.uint32) << 16
    g = g.view(np.float32).reshape(*lead, AP.ND, 8, 8)  # [db][query][8]
    return np.moveaxis(g, -3, -2).reshape(*lead, 8, AP.HD)


def _qs_slots(fused: FusedOperator, T: int, qc: int, qg: int) -> np.ndarray:
    """QS (pass-major: [32 QG-token pass][column][token group 4 QG][Q | gate][head][SLOT]) as [column][token group][Q |
    gate][head][SLOT] uint8."""
    qs = fused.get_buffer("QS").view(np.uint8).reshape(T // (32 * qg), COLS, 4 * qg, 2, qc, AP.SLOT)
    return np.ascontiguousarray(qs.swapaxes(0, 1)).reshape(COLS, T // 8, 2, qc, AP.SLOT)


class _Setup:
    """The model, the layer, two chunks of essay tokens (T, n2; or ``tokens`` of them) and llama.cpp's tensors of
    them."""

    def __init__(self, size: str, T: int, n2: int, tokens: int = 0):
        path = decision_gguf(size) if size.startswith("d3") else qwen_gguf("Q4_K_M", size)
        require(path)
        self.g = g = GGUFFile(path)
        self.cfg = cfg = QwenConfig.from_gguf(g)
        self.L = L = next(i for i, a in enumerate(cfg.attention) if a)
        img = QwenImage(g, cfg, layers=[L], vocab=COLS * 64)
        self.lay = PG.plan_phase(img, img.region(L, "attn_in"), T, pitch=PG.phase_pitch(img, "attn_in", T)).layout
        self.ash = AT.AttnShape(cfg, MAX_POS, a_bytes=img.tile_bytes)
        self.T, self.n2, self.qc = T, n2, cfg.num_attention_heads // COLS
        self.kvh = cfg.num_key_value_heads
        self.per = self.kvh * AT.HD // COLS  # (K / V dims of a column)
        self.tokens = llama_tokenize(path, essay("apple.txt"))[:tokens or T + n2]
        n = len(self.tokens)
        G_ = golden(path, self.tokens, layers=(L,), names="Qcur_full|Kcur|Vcur|Qcur|gate_sigmoid|attn_gated",
                        logits="none")
        self.qfull = bf16_round(np.asarray(G_["Qcur_full", L], np.float32).reshape(n, -1))
        self.k = bf16_round(np.asarray(G_["Kcur", L], np.float32).reshape(n, -1))
        self.v = bf16_round(np.asarray(G_["Vcur", L], np.float32).reshape(n, -1))
        self.gold = G_
        self.qn = np.asarray(g.tensor(f"blk.{L}.attn_q_norm.weight").raw, np.float32).reshape(-1)
        self.kn = np.asarray(g.tensor(f"blk.{L}.attn_k_norm.weight").raw, np.float32).reshape(-1)

    def G(self, t0: int, n: int, rows: np.ndarray | None = None) -> np.ndarray:
        """attn_in's output of tokens t0 .. t0 + n - 1 in rows 0 .. n - 1 (or ``rows``; the others zero)."""
        lay, qc = self.lay, self.qc
        G = np.zeros((self.T, lay.N), np.float32)
        r = np.arange(n) if rows is None else rows
        for c in range(COLS):
            G[r[:, None], c * lay.pitch + lay.off["q"] + np.arange(2 * qc * AT.HD)] = \
                self.qfull[t0:t0 + n, 2 * qc * AT.HD * c:2 * qc * AT.HD * (c + 1)]
            e = self.per
            G[r[:, None], c * lay.pitch + lay.off["k"] + np.arange(e)] = self.k[t0:t0 + n, e * c:e * (c + 1)]
            G[r[:, None], c * lay.pitch + lay.off["v"] + np.arange(e)] = self.v[t0:t0 + n, e * c:e * (c + 1)]
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
    _fill_cs(fused, s.cfg)
    nrm = np.concatenate([s.qn, s.kn]).astype(np.float32)
    fused.get_buffer("NRM").view(np.uint8)[:] = nrm.view(np.uint8)
    fused.get_buffer("KV")[:] = 0
    return op, fused


def _run(fused: FusedOperator, params: dict[str, int], G: np.ndarray) -> None:
    fused.get_buffer("G")[:] = G.reshape(-1).astype(bfloat16)
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
        q = _q_panels(slots[:, :, 0]).transpose(0, 2, 1, 3, 4).reshape(COLS * qc, T, hd)[:, :n]  # [head][token]
        sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(COLS * qc, T, hd)[:, :n]
        qg = s.qfull[t0:t0 + n].reshape(n, -1, 2, hd)
        q_want = s.nr(qg[:, :, 0], s.qn, pos).transpose(1, 0, 2) * np.log2(np.e) / 16
        assert rel_max(q, q_want) < 2e-4, t0  # (hi + lo panels: ~14 bits below a block's maximum)
        q_gold = np.asarray(s.gold["Qcur", s.L], np.float32).reshape(-1, COLS * qc, hd)[t0:t0 + n].transpose(1, 0, 2)
        assert rel_max(q / (np.log2(np.e) / 16), q_gold) < 1e-2, t0
        sg_want = 1 / (1 + np.exp(-np.asarray(qg[:, :, 1], np.float64))).transpose(1, 0, 2)
        assert np.abs(sg - sg_want).max() < 2 ** -8, t0
    # the cache: both chunks' positions of every KV head (K normed, rotated; V as stored)
    kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
    pos = np.arange(T + n2)
    kr = s.k[:T + n2].reshape(T + n2, s.kvh, hd)
    k_want = s.nr(kr, s.kn, pos)
    for h in range(s.kvh):
        K, V = KV.kv_from_groups(kv[h, :(T + n2 + 31) // 32], AT.HD, dmajor=True)
        assert rel_max(K[:T + n2], k_want[:, h]) < 1e-2, h  # (single bfp16: 8 bits below a block's maximum)
        assert rel_max(V[:T + n2], s.v[:T + n2, 256 * h:256 * (h + 1)]) < 1e-2, h
        k_gold = np.asarray(s.gold["Kcur.1", s.L], np.float32).reshape(-1, s.kvh, hd)[:T + n2, h]
        assert rel_max(K[:T + n2], k_gold) < 1e-2, h


@pytest.mark.parametrize(("size", "T", "pp"), [("9B", 64, 1), ("27B", 64, 1), ("1B", 64, 1), ("1B", 192, 1),
                                              ("35B-A3B", 64, 1), ("9B", 64, 2), ("1B", 192, 2)])
def test_attn_prefill(size: str, T: int, pp: int) -> None:
    """AttnPrep then PrefillAttn (one fused runlist) over two chunks: the output panels against the exact attention
    of the cores' own operands (their Q panels, the cache they wrote, their sigmoid(gate): within 0.7%, the bfp16
    output and P panels; hi + lo P, ``pp`` 2: the output's rounding alone) and llama.cpp's attn_gated (~1% rms). The
    1B: passes of 64 queries (one head per column)."""
    from artlite.operators import bfp16

    n2 = 45
    s = _Setup(size, T, n2)
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, pp=pp)
    fused = FusedOperator(name=f"attn_prefill{pp}", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    _fill_cs(fused, s.cfg)
    fused.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    fused.get_buffer("KV")[:] = 0
    H, qc, hd = s.cfg.num_attention_heads, s.qc, AT.HD
    for t0, n in [(0, T), (T, n2)]:
        _run(fused, prep.params(t0) | attn.params(t0), s.G(t0, n))
        A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:n]
        slots = _qs_slots(fused, T, qc, prep.qg)
        q = _q_panels(slots[:, :, 0]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n] * (16 / np.log2(np.e))
        sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n]
        kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
        want = np.zeros((n, H, hd))
        for h in range(s.kvh):
            K, V = (np.asarray(x, np.float64)[:t0 + n] for x in KV.kv_from_groups(kv[h, :(t0 + n + 31) // 32], AT.HD, dmajor=True))
            for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
                sc = q[g].astype(np.float64) @ K.T / 16
                sc[np.arange(t0, t0 + n)[:, None] < np.arange(t0 + n)[None]] = -np.inf
                pr = np.exp(sc - sc.max(-1, keepdims=True))
                want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g]
        # (1B: a peakier softmax, its largest P 0.45 on average (9B 0.28): single V rows carry more of an output, the
        # largest error ~2 ULPs of a bfp16 block; the rms error as the 9B's)
        eq = rel_max(bfp16.values(want.reshape(n, -1).astype(np.float32)).reshape(want.shape), want)
        print(f"{size} T {T} pp {pp} chunk at {t0}: {rel_max(A, want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
        assert rel_max(A, want) < (1.5e-2 if size == "1B" else 1e-2), t0
        assert np.linalg.norm(A - want) / np.linalg.norm(want) < 1.2e-2, t0
        gold = np.asarray(s.gold["attn_gated", s.L], np.float32).reshape(-1, H, hd)[t0:t0 + n]
        assert rel_max(A, gold) < 3e-2, t0


def test_attn_split() -> None:
    """The split attention (``attn256_split``: a pair of cores per six row panels, 96-query passes) after AttnPrep with
    its passes, over two chunks: the output panels against the exact attention of the cores' own operands (the Q slots'
    hi panels: single Q, the cache, sigmoid(gate): within the tolerances of ``test_attn_prefill``) and llama.cpp's
    attn_gated."""
    from artlite.prefill import attn256_split as SP

    T, n2 = 192, 45
    s = _Setup("1B", T, n2)
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS, qg=SP.QG)
    attn = SP.SplitAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes)
    fused = FusedOperator(name="attn_split", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    _fill_cs(fused, s.cfg)
    fused.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    fused.get_buffer("KV")[:] = 0
    H, hd = s.cfg.num_attention_heads, AT.HD
    for t0, n in [(0, T), (T, n2)]:
        _run(fused, prep.params(t0) | attn.params(t0), s.G(t0, n))
        A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:n]
        slots = _qs_slots(fused, T, 1, SP.QG)
        hi = slots[:, :, 0].reshape(COLS, T // 8, 1, AP.ND, 2, 8, 9)[..., 0, :, :]  # (the hi panels: single Q)
        q = np.moveaxis(bfp16.block_values(hi), -3, -2).reshape(COLS, T // 8, 1, 8, hd)
        q = q.transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n] * (16 / np.log2(np.e))
        sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n]
        kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
        want = np.zeros((n, H, hd))
        for h in range(s.kvh):
            K, V = (np.asarray(x, np.float64)[:t0 + n] for x in KV.kv_from_groups(kv[h, :(t0 + n + 31) // 32], AT.HD, dmajor=True))
            for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
                sc = q[g].astype(np.float64) @ K.T / 16
                sc[np.arange(t0, t0 + n)[:, None] < np.arange(t0 + n)[None]] = -np.inf
                pr = np.exp(sc - sc.max(-1, keepdims=True))
                want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g]
        eq = rel_max(bfp16.values(want.reshape(n, -1).astype(np.float32)).reshape(want.shape), want)
        print(f"split T {T} chunk at {t0}: {rel_max(A, want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
        assert rel_max(A, want) < 1.5e-2, t0
        assert np.linalg.norm(A - want) / np.linalg.norm(want) < 1.2e-2, t0
        gold = np.asarray(s.gold["attn_gated", s.L], np.float32).reshape(-1, H, hd)[t0:t0 + n]
        assert rel_max(A, gold) < 3e-2, t0


@pytest.mark.parametrize(("size", "T", "n", "pp"), [("d3-lite", 128, 99, 1), ("d3-lite", 96, 70, 1), ("d3-lite", 128, 99, 2),
                                                   ("d3-lite", 96, 70, 2)])
def test_attn_prefill_noncausal(size: str, T: int, n: int, pp: int) -> None:
    """A non-causal chunk (vllm-d3's prompt: every query sees every key) of n valid tokens at position 0: against the
    exact attention of the cores' own operands and llama.cpp's attn_gated of the whole prompt. T 128: passes of 64
    queries on two streams; T 96: of 32 on one. ``pp`` 2: hi + lo P."""
    s = _Setup(size, T, 0, tokens=n)
    assert not s.cfg.causal
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, causal=False, pp=pp)
    assert attn.two == (T % 64 == 0)
    fused = FusedOperator(name=f"attn_prefill_nc{pp}", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    _fill_cs(fused, s.cfg)
    fused.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    fused.get_buffer("KV")[:] = 0
    H, qc, hd = s.cfg.num_attention_heads, s.qc, AT.HD
    _run(fused, prep.params(0) | attn.params(0, n), s.G(0, n))
    A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:n]
    slots = _qs_slots(fused, T, qc, prep.qg)
    q = _q_panels(slots[:, :, 0]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n] * (16 / np.log2(np.e))
    sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n]
    kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
    want = np.zeros((n, H, hd))
    for h in range(s.kvh):
        K, V = (np.asarray(x, np.float64)[:n] for x in KV.kv_from_groups(kv[h, :(n + 31) // 32], AT.HD, dmajor=True))
        for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
            sc = q[g].astype(np.float64) @ K.T / 16
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g]
    eq = rel_max(bfp16.values(want.reshape(n, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"{size} T {T} n {n} pp {pp}: {rel_max(A, want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
    assert rel_max(A, want) < 1.5e-2
    assert np.linalg.norm(A - want) / np.linalg.norm(want) < 1.2e-2
    gold = np.asarray(s.gold["attn_gated", s.L], np.float32).reshape(-1, H, hd)[:n]
    assert rel_max(A, gold) < 3e-2


@pytest.mark.parametrize(("size", "T", "causal"), [("d3-lite", 96, True), ("d3-lite", 128, True), ("d3-lite", 128, False),
                                                   ("d3-lite", 96, False)])
def test_attn_prefill_packed(size: str, T: int, causal: bool) -> None:
    """A packed chunk at position 0 (``PrefillAttnOperator(pack=True)``): segments of 32, 37, 3 and 20 tokens (from 0,
    32, 69, 72: inside a group and a record), padding after them, each token's rotary position from its segment's start
    (cos / sin rows written past the table's text rows); every row against the exact attention of the cores' own
    operands over its segment's keys (up to itself, or all of them non-causal); the first segment bit-identical to its
    tokens alone (the unpacked design: the same keys, the masked ones adding exact zeros). At 128 tokens two streams,
    at 96 (32-query passes) one."""
    lengths = [32, 37, 3, 20]
    n = sum(lengths)
    s = _Setup(size, T, 0, tokens=n)
    starts = np.cumsum([0, *lengths])[:-1]
    lo = np.arange(T)
    hi = np.arange(T)
    pos = np.zeros(T, np.int64)
    for a, m in zip(starts, lengths, strict=True):
        lo[a:a + m] = a
        hi[a:a + m] = a + m - 1 if not causal else np.arange(a, a + m)
        pos[a:a + m] = np.arange(m)
    R = MAX_POS - T  # (the chunk's rows of the cos / sin table)
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, causal=causal, pp=2, pack=True)
    fused = FusedOperator(name=f"attn_prefill_packed{int(causal)}",
                          runlist=[(prep, "G", "CS", "NRM", "QS", "KV"), (attn, "QS", "KV", "A", "SEG")],
                          input_args=["G", "CS", "NRM", "SEG"], output_args=["A"]).compile()
    _fill_cs(fused, s.cfg, pos)
    fused.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    fused.get_buffer("SEG").view(np.uint8)[:] = attn.seg(lo, hi)
    fused.get_buffer("KV")[:] = 0
    H, qc, hd = s.cfg.num_attention_heads, s.qc, AT.HD
    _run(fused, prep.params(0, R) | attn.params(0, n), s.G(0, n))
    A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:n]
    slots = _qs_slots(fused, T, qc, prep.qg)
    q = _q_panels(slots[:, :, 0]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n] * (16 / np.log2(np.e))
    sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)[:, :n]
    kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
    want = np.zeros((n, H, hd))
    keys = np.arange(n)
    seen = (keys[None] >= lo[:n, None]) & (keys[None] <= hi[:n, None])
    for h in range(s.kvh):
        K, V = (np.asarray(x, np.float64)[:n] for x in KV.kv_from_groups(kv[h, :(n + 31) // 32], AT.HD, dmajor=True))
        for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
            sc = q[g].astype(np.float64) @ K.T / 16
            sc[~seen] = -np.inf
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g]
    eq = rel_max(bfp16.values(want.reshape(n, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"{size} T {T} causal {causal}: {rel_max(A, want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
    assert rel_max(A, want) < 1.1 * eq
    # the first segment alone (the unpacked design, its 32 tokens at position 0)
    attn1 = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, causal=causal, pp=2)
    alone = FusedOperator(name=f"attn_prefill_alone{int(causal)}", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"),
                                                                             (attn1, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    _fill_cs(alone, s.cfg)
    alone.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    alone.get_buffer("KV")[:] = 0
    _run(alone, prep.params(0) | attn1.params(0, lengths[0]), s.G(0, lengths[0]))
    A1 = bfp16.from_panels(alone.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:lengths[0]]
    bad = np.flatnonzero((A1 != A[:lengths[0]]).any((1, 2)))
    assert bad.size == 0, f"rows {bad.tolist()} differ from the segment alone"


@pytest.mark.parametrize(("size", "T"), [("d3-lite", 96), ("d3-lite", 128)])
def test_attn_prefill_tree(size: str, T: int) -> None:
    """A packed chunk after a history (``PrefillAttnOperator(pack=True)`` at p0 = T: a prefix every segment continues,
    the tree packing of a recurrent model's questions): segments of 32, 37, 3 and 20 tokens at 16-token starts (0, 32,
    80, 96), padding between them, each token's rotary position T + its distance from its segment's start; every row
    against the exact attention of the cores' own operands over the history and its segment's keys up to itself, and
    every segment bit-identical to its tokens alone after the same history (the unpacked design at p0 = T: the same
    keys, the masked ones adding exact zeros, the same V blocks). At 128 tokens two streams, at 96 one (segments of 13,
    37, 3 and 10 tokens from 0, 16, 64, 80)."""
    lengths = [32, 37, 3, 20] if T == 128 else [13, 37, 3, 10]
    starts = np.cumsum([0, *(-(-m // 16) * 16 for m in lengths)])[:-1]
    n = sum(lengths)
    s = _Setup(size, T, 0, tokens=T + n)
    H, qc, hd = s.cfg.num_attention_heads, s.qc, AT.HD
    prep = AP.AttnPrepOperator(T=T, cfg=s.cfg, layout=s.lay, head_bytes=s.ash.head_bytes, cs_positions=MAX_POS)
    attn1 = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, pp=2)
    alone = FusedOperator(name="attn_prefill_tree_alone", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"),
                                                                   (attn1, "QS", "KV", "A")],
                          input_args=["G", "CS", "NRM"], output_args=["A"]).compile()
    for f in (alone,):
        _fill_cs(f, s.cfg)
        f.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
        f.get_buffer("KV")[:] = 0
    _run(alone, prep.params(0) | attn1.params(0), s.G(0, T))  # (the history: the essay's first T tokens)
    hist = alone.get_buffer("KV").copy()
    # the packed chunk at p0 = T: segment k's tokens T + sum(lengths[:k]) .. in rows starts[k] ..
    lo, hi, pos = np.arange(T), np.arange(T), np.full(T, T, np.int64)
    G = np.zeros((T, s.lay.N), np.float32)
    t0 = T + np.cumsum([0, *lengths])[:-1]
    for a, m, t in zip(starts, lengths, t0, strict=True):
        lo[a:a + m], hi[a:a + m], pos[a:a + m] = a, np.arange(a, a + m), T + np.arange(m)
        G += s.G(int(t), m, np.arange(a, a + m))
    R = MAX_POS - T  # (the chunk's rows of the cos / sin table)
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.ash.head_bytes, pp=2, pack=True)
    fused = FusedOperator(name="attn_prefill_tree", runlist=[(prep, "G", "CS", "NRM", "QS", "KV"),
                                                             (attn, "QS", "KV", "A", "SEG")],
                          input_args=["G", "CS", "NRM", "SEG"], output_args=["A"]).compile()
    _fill_cs(fused, s.cfg, pos)
    fused.get_buffer("NRM").view(np.uint8)[:] = np.concatenate([s.qn, s.kn]).astype(np.float32).view(np.uint8)
    fused.get_buffer("SEG").view(np.uint8)[:] = attn.seg(lo, hi, T)
    fused.get_buffer("KV")[:] = hist
    fused.get_buffer("G")[:] = G.reshape(-1).astype(bfloat16)
    fused.sync_inputs()
    fused.set_params(prep.params(T, R) | attn.params(T))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()
    A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)
    slots = _qs_slots(fused, T, qc, prep.qg)
    q = _q_panels(slots[:, :, 0]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd) * (16 / np.log2(np.e))
    sg = _gate_sigmoid(slots[:, :, 1]).transpose(0, 2, 1, 3, 4).reshape(H, T, hd)
    kv = fused.get_buffer("KV").view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
    rows = np.concatenate([np.arange(a, a + m) for a, m in zip(starts, lengths, strict=True)])
    keys = np.arange(2 * T)
    seen = (keys[None] < T) | ((keys[None] >= T + lo[rows, None]) & (keys[None] <= T + hi[rows, None]))
    want = np.zeros((len(rows), H, hd))
    for h in range(s.kvh):
        K, V = (np.asarray(x, np.float64)[:2 * T] for x in KV.kv_from_groups(kv[h, :2 * T // 32], AT.HD, dmajor=True))
        for g in range(H // s.kvh * h, H // s.kvh * (h + 1)):
            sc = q[g][rows].astype(np.float64) @ K.T / 16
            sc[~seen] = -np.inf
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            want[:, g] = (pr @ V) / pr.sum(-1, keepdims=True) * sg[g][rows]
    eq = rel_max(bfp16.values(want.reshape(len(rows), -1).astype(np.float32)).reshape(want.shape), want)
    print(f"{size} T {T}: {rel_max(A[rows], want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
    assert rel_max(A[rows], want) < 1.1 * eq
    for a, m, t in zip(starts, lengths, t0, strict=True):  # (each segment alone after the history)
        alone.get_buffer("KV")[:] = hist
        _run(alone, prep.params(T) | attn1.params(T), s.G(int(t), m))
        A1 = bfp16.from_panels(alone.get_buffer("A"), T, H * hd).reshape(T, H, hd)[:m]
        bad = np.flatnonzero((A1 != A[a:a + m]).any((1, 2)))
        print(f"segment {a} + {m}: {m - bad.size} of {m} rows bit-identical to alone")
        assert bad.size == 0, f"segment {a}: rows {bad.tolist()} differ from the segment alone"


# ----------------------------------------------------------------------
# Spark-X's attention: no norms, a gate per head, sliding-window layers on rings
# ----------------------------------------------------------------------

SPARK_POS = 4096  # (the CS tables' rows)


class _Spark:
    """A Spark-X layer of a kind (full: layer 3, swa: layer 0) at T tokens: its attn_in layout, the attention's shape
    (a sliding-window layer's caches: a ring of ``ring`` groups) and random attn_in outputs."""

    def __init__(self, size: str, kind: str, T: int, ring: int = 82):
        from artlite.model_config import SparkConfig, spark_gguf
        from models.spark.decode import attn_spec
        from models.spark.image import SparkImage

        path = spark_gguf(size)
        require(path)
        g = GGUFFile(path)
        self.cfg = cfg = SparkConfig.from_gguf(g)
        self.kind, self.T = kind, T
        self.L = L = cfg.layers(kind)[0]
        img = SparkImage(g, cfg, layers=[L], vocab=COLS * 64)
        self.lay = PG.plan_phase(img, img.region(L, "attn_in"), T, pitch=PG.phase_pitch(img, "attn_in", T)).layout
        self.spec = attn_spec(cfg)
        self.ring = ring if kind == "swa" else 0
        self.ash = AT.AttnShape(cfg, SPARK_POS, a_bytes=img.tile_bytes, slots=8, spec=self.spec)
        self.head_bytes = self.ring * AT.OBJS * img.tile_bytes if self.ring else self.ash.head_bytes
        self.qc, self.kvh = cfg.num_attention_heads // COLS, cfg.num_key_value_heads
        self.per = self.kvh * AT.HD // COLS
        rng = np.random.default_rng(T + len(kind))
        n = SPARK_POS
        self.q = bf16_round(rng.standard_normal((n, cfg.num_attention_heads, AT.HD)).astype(np.float32))
        self.gate = bf16_round(rng.standard_normal((n, cfg.num_attention_heads)).astype(np.float32) * 2)
        self.k = bf16_round(rng.standard_normal((n, self.kvh, AT.HD)).astype(np.float32))
        self.v = bf16_round(rng.standard_normal((n, self.kvh, AT.HD)).astype(np.float32))

    def G(self, t0: int) -> np.ndarray:
        """attn_in's output of positions t0 .. t0 + T - 1 (rows 0 ..)."""
        lay, qc, hd, T = self.lay, self.qc, AT.HD, self.T
        G = np.zeros((T, lay.N), np.float32)
        k = self.k[t0:t0 + T].reshape(T, -1)
        v = self.v[t0:t0 + T].reshape(T, -1)
        for c in range(COLS):
            o = c * lay.pitch
            G[:, o + lay.off["q"]:o + lay.off["q"] + qc * hd] = self.q[t0:t0 + T, qc * c:qc * (c + 1)].reshape(T, -1)
            G[:, o + lay.off["g"]:o + lay.off["g"] + qc] = self.gate[t0:t0 + T, qc * c:qc * (c + 1)]
            G[:, o + lay.off["k"]:o + lay.off["k"] + self.per] = k[:, self.per * c:self.per * (c + 1)]
            G[:, o + lay.off["v"]:o + lay.off["v"] + self.per] = v[:, self.per * c:self.per * (c + 1)]
        return G

    def rope(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """x (n, heads, 256) rotated as the kind rotates (fp64; a sliding-window layer's cos / sin as its bf16 table)."""
        th = np.asarray(self.cfg.rope_theta(list(pos), self.kind), np.float64)[:, None]
        r = th.shape[-1]
        c, sn = np.cos(th), np.sin(th)
        if self.kind == "swa":
            c, sn = (bf16_round(v.astype(np.float32)).astype(np.float64) for v in (c, sn))
        y = np.asarray(x, np.float64).copy()
        a, b = y[..., :r].copy(), y[..., r:2 * r].copy()
        y[..., :r], y[..., r:2 * r] = a * c - b * sn, a * sn + b * c
        return y

    @property
    def hist(self) -> int:
        """(a sliding-window layer) the staging's groups of the window's history."""
        return self.cfg.sliding_window // 32 if self.ring else 0

    def prep(self) -> tuple[AP.AttnPrepOperator, str]:
        """The prep and its KV argument's name (a sliding-window layer's: the staging S)."""
        hb = (self.hist + self.T // 32) * AP.GROUP if self.ring else self.head_bytes
        op = AP.AttnPrepOperator(T=self.T, cfg=self.cfg, layout=self.lay, head_bytes=hb, cs_positions=SPARK_POS,
                                 spec=self.spec, kind=self.kind, hist=self.hist)
        return op, "S" if self.ring else "KV"

    def records(self, kv: np.ndarray, h: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """KV head h's K, V of positions 0 .. n - 1 from the cache (a ring: group g at slot g % ring)."""
        groups = -(-n // 32)
        slots = [g % self.ring for g in range(groups)] if self.ring else list(range(groups))
        K, V = KV.kv_from_groups(kv[h, slots], AT.HD, dmajor=True)
        return np.asarray(K, np.float64)[:n], np.asarray(V, np.float64)[:n]


def _spark_slots(fused: FusedOperator, T: int, qc: int, qg: int) -> tuple[np.ndarray, np.ndarray]:
    """A gate per head's QS (pass-major [pass][column][token group 4 QG][head][SLOT], Q slots alone) -> the Q values
    (heads, T, 256: the single panels) and the gates (heads, T, 256: the slot's 8 sigmoid(gate) values over every dim)."""
    qs = fused.get_buffer("QS").view(np.uint8).reshape(T // (32 * qg), COLS, 4 * qg, qc, AP.SLOT)
    sl = np.ascontiguousarray(qs.swapaxes(0, 1)).reshape(COLS, T // 8, qc, AP.SLOT)
    p = sl.reshape(COLS, T // 8, qc, AP.ND, 2, 8, 9)  # [db][hi / lo][query][block]
    q = np.moveaxis(bfp16.block_values(p[..., 0, :, :]), -3, -2).reshape(COLS, T // 8, qc, 8, AP.HD)
    g = np.ascontiguousarray(sl[..., 2 * AP.GATE_AT:2 * AP.GATE_AT + 16]).view(np.uint16).astype(np.uint32) << 16
    g = np.broadcast_to(g.view(np.float32)[..., None], (COLS, T // 8, qc, 8, AP.HD))
    heads = COLS * qc
    return (q.transpose(0, 2, 1, 3, 4).reshape(heads, T, AP.HD), g.transpose(0, 2, 1, 3, 4).reshape(heads, T, AP.HD))


def _spark_fused(s: _Spark, runlist: list, out: str) -> FusedOperator:
    fused = FusedOperator(name=f"spark_attn_{s.kind}_{len(runlist)}", runlist=runlist, input_args=["G", "CS"],
                          output_args=[out]).compile()
    fused.get_buffer("CS").view(np.uint8)[:] = AP.cs_table(s.cfg, SPARK_POS, s.kind).view(np.uint8).reshape(-1)
    for name in ("KV", "S"):
        if name in {a for run in runlist for a in run[1:]}:
            fused.get_buffer(name)[:] = 0
    return fused


@pytest.mark.parametrize(("size", "kind"), [("1.7B", "swa"), ("1.7B", "full"), ("4B", "swa"), ("4B", "full")])
def test_spark_prep(size: str, kind: str) -> None:
    """Spark's prep: Q slots (q log2(e) / 16 rotated as the kind: 128 pairs or the first 32; single panels; d-block 0's lo
    panel the 8 tokens' sigmoid(gate): one per head and token), the records (a full layer's in the cache at the
    chunk's groups, a sliding-window layer's in the staging after the window's history), chunks at 0, 64, 2560, 2592."""
    T = 64
    s = _Spark(size, kind, T)
    op, kv_arg = s.prep()
    fused = _spark_fused(s, [(op, "G", "CS", "QS", kv_arg)], "QS")
    qc = s.qc
    for t0 in (0, 64, 2560, 2592):
        _run(fused, op.params(t0), s.G(t0))
        pos = np.arange(t0, t0 + T)
        q, sg = _spark_slots(fused, T, qc, op.qg)
        q_want = s.rope(s.q[t0:t0 + T], pos).transpose(1, 0, 2) * np.log2(np.e) / 16
        assert rel_max(q, q_want) < 1e-2, t0  # (single bfp16 panels: 8 bits below a block's maximum)
        sg_want = 1 / (1 + np.exp(-s.gate[t0:t0 + T].astype(np.float64))).T[:, :, None]
        assert np.abs(sg - sg_want).max() < 2 ** -8, t0
        kv = fused.get_buffer(kv_arg).view(np.uint8).reshape(s.kvh, -1, AT.OBJS, 9280)
        k_want = s.rope(s.k[:t0 + T], np.arange(t0 + T))
        for h in range(s.kvh):
            if s.ring:  # (the staging: the chunk's records after the window's history)
                K, V = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[h, s.hist:], AT.HD, dmajor=True))
                K, V = np.concatenate([np.zeros((t0, AT.HD)), K]), np.concatenate([np.zeros((t0, AT.HD)), V])
            else:
                K, V = s.records(kv, h, t0 + T)
            assert rel_max(K[t0:], k_want[t0:, h]) < 1e-2, (t0, h)
            assert rel_max(V[t0:], s.v[t0:t0 + T, h]) < 1e-2, (t0, h)


SPARK_ATTN = [("1.7B", "swa", 64, 0), ("1.7B", "swa", 128, 64), ("1.7B", "swa", 128, 3840), ("1.7B", "swa", 64, 2592),
              ("4B", "swa", 64, 0), ("4B", "swa", 64, 640), ("4B", "swa", 128, 2560), ("1.7B", "full", 64, 64),
              ("4B", "full", 64, 1024)]


@pytest.mark.parametrize(("size", "kind", "T", "p0"), SPARK_ATTN, ids=[f"{s}-{k}-T{t}-p{p}" for s, k, t, p in SPARK_ATTN])
def test_spark_attention(size: str, kind: str, T: int, p0: int) -> None:
    """Spark's prep then attention at p0 after a history in the cache (a sliding-window layer's ring: its last 81
    groups): the output panels against the exact attention of the cores' own operands (Q panels, the cache, the gate
    slots), a sliding-window row's keys q - 511 .. q (the window's history from the ring into the staging, the chunk's
    records from the staging into the ring: across its end at 2592 / 2560), a full row's 0 .. q. The 1.7B: passes of 64
    queries on two streams; the 4B: of 32."""
    from artlite.operators import bfp16

    s = _Spark(size, kind, T)
    prep, kv_arg = s.prep()
    swa = kind == "swa"
    attn = AP.PrefillAttnOperator(T=T, cfg=s.cfg, head_bytes=s.head_bytes, window=s.cfg.sliding_window if swa else 0,
                                  ring=s.ring, spec=s.spec)
    fused = _spark_fused(s, [(prep, "G", "CS", "QS", kv_arg), (attn, "QS", "KV", "A", *(["S"] if swa else []))], "A")
    H, qc, hd, kvh = s.cfg.num_attention_heads, s.qc, AT.HD, s.kvh
    kv = fused.get_buffer("KV").view(np.uint8).reshape(kvh, -1, AT.OBJS, 9280)
    if p0:  # (the history as records, rotated as the kind rotates; a ring: its last groups at their slots)
        g0 = max(0, p0 // 32 - (s.ring - 1)) if swa else 0
        k_rot = s.rope(s.k[:p0], np.arange(p0)).astype(np.float32)
        for h in range(kvh):
            grp = KV.kv_groups(k_rot[32 * g0:, h], s.v[32 * g0:p0, h], 9280, 1, dmajor=True)
            for i in range(grp.shape[0]):
                kv[h, (g0 + i) % s.ring if swa else g0 + i] = grp[i].reshape(AT.OBJS, 9280)
    _run(fused, prep.params(p0) | attn.params(p0), s.G(p0))
    A = bfp16.from_panels(fused.get_buffer("A"), T, H * hd).reshape(T, H, hd)
    q, sg = _spark_slots(fused, T, qc, prep.qg)
    q = q * (16 / np.log2(np.e))
    kv = fused.get_buffer("KV").view(np.uint8).reshape(kvh, -1, AT.OBJS, 9280)
    pos, keys = np.arange(p0, p0 + T)[:, None], np.arange(p0 + T)[None]
    seen = (keys <= pos) & ((pos - keys < s.cfg.sliding_window) if swa else True)
    want = np.zeros((T, H, hd))
    for h in range(kvh):
        K, V = s.records(kv, h, p0 + T)
        lo = max(0, p0 - s.cfg.sliding_window) if swa else 0  # (keys a ring still holds)
        for g in range(H // kvh * h, H // kvh * (h + 1)):
            sc = q[g].astype(np.float64) @ K[lo:].T / 16
            sc[~seen[:, lo:]] = -np.inf
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            want[:, g] = (pr @ V[lo:]) / pr.sum(-1, keepdims=True) * sg[g]
    eq = rel_max(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"{size} {kind} T {T} p0 {p0}: {rel_max(A, want):.2e} (the exact output rounded to bfp16 {eq:.2e})")
    assert rel_max(A, want) < 1.5e-2 and np.linalg.norm(A - want) / np.linalg.norm(want) < 1.2e-2
