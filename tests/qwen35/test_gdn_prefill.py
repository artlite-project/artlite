"""The GDN prefill operators (models/qwen35/gdn_prefill.py) on the NPU, on a 9B GDN layer's own inputs (llama.cpp's
prompt-mode tensors of an essay laid out as the gdn_in GEMMs write them): GdnPrep and GdnScan, two chunks, the
second continuing the first's state and conv state with padding tokens after its valid ones (an odd count: masked);
o and the state against the cores' arithmetic (``ref.core_gdn_scan``: the exact SiLU, the cores' table SiLU is
within 0.59 bf16 ULP) and llama.cpp's."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.operators import FusedOperator
from artlite.operators.design import COLS
from artlite.prefill import gemm as PG
from models.qwen35 import gdn as GD
from models.qwen35 import gdn_prefill as GP
from models.qwen35.image import QwenImage, gdn_v_heads

from ..common import essay
from . import ref

HSLOT, CS_OFF = 16384, 896  # (header slots as the overlay's: the conv state 4 META + 640 bytes in)


def _bf(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


def _bits(x: np.ndarray) -> np.ndarray:
    """bf16 bits (uint16) of values."""
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().view(torch.int16).numpy().view(np.uint16)


@pytest.mark.parametrize("size", ["9B", "1B", "occamy"])
@pytest.mark.parametrize("T", [64])
def test_gdn_scan(T: int, size: str) -> None:
    """(Occamy: its F32 alpha in the ab slot as fp32, ``gdn_prefill.ab_f32``)"""
    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    L = cfg.gdn_layers[0]
    img = QwenImage(g, cfg, layers=[L], vocab=COLS * 64)
    region = img.region(L, "gdn_in")
    lay = PG.plan_phase(img, region, T, pitch=PG.phase_pitch(img, "gdn_in", T)).layout
    blk = region.block("qkv")
    n2 = T - 5  # (the second chunk's valid tokens)
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:T + n2]
    G_ = ref.golden(path, tokens, layers=(L,), names="linear_attn_qkv_mixed|alpha|beta|attn_output|new_state",
                    logits="none")
    qkv = _bf(np.asarray(G_["linear_attn_qkv_mixed", L], np.float32).reshape(len(tokens), -1))
    alpha = np.asarray(G_["alpha", L], np.float32).reshape(len(tokens), -1)
    alpha = alpha if GP.ab_f32(lay) else _bf(alpha)  # (the ab slot's values: fp32 or bf16)
    beta = _bf(np.asarray(G_["beta", L], np.float32).reshape(len(tokens), -1))
    op = GP.GdnScanOperator(T=T, cfg=cfg, layout=lay)
    prep = GP.GdnPrepOperator(T=T, cfg=cfg, layout=lay)
    fused = FusedOperator(name="gdn_scan", runlist=[(prep, "G", "CV", "P", "M", "PR"),
                                                    (op, "PR", "G", "CV", "P", "GS", "O")],
                          input_args=["G", "P", "M", "CV"], output_args=["O"]).compile()  # (GS: scratch, written and read)
    sh = op.sh
    P = GP.scan_params(g, cfg, L, blk.rows)
    cw = np.asarray(g.tensor(f"blk.{L}.ssm_conv1d.weight").raw, np.float32)
    dt = np.asarray(g.tensor(f"blk.{L}.ssm_dt.bias").raw, np.float32).reshape(-1)
    A = np.asarray(g.tensor(f"blk.{L}.ssm_a").raw, np.float32).reshape(-1)
    for name in ("G", "P", "M", "CV", "GS"):
        fused.get_buffer(name).zero_()
    fused.get_buffer("P").view(torch.uint8)[:P.size] = torch.from_numpy(P)
    st = [np.zeros((sh.hc, 128, 128), np.float32) for _ in range(COLS)]
    cs = [np.zeros((3, sh.ch), np.float32) for _ in range(COLS)]
    cv = fused.get_buffer("CV").view(torch.int16).numpy().reshape(-1, lay.N)
    mask = fused.get_buffer("M").view(torch.float32).numpy()
    s_err = []
    for chunk, (t0, n) in enumerate([(0, T), (T, n2)]):
        final: list[np.ndarray] = []
        Gb = np.zeros((T, lay.N), np.uint16)  # (bf16 bits)
        for c in range(COLS):
            rows = blk.rows[c][blk.rows[c] >= 0]
            vh = gdn_v_heads(cfg, c)
            Gb[:n, c * lay.pitch + lay.off["qkv"] + np.arange(rows.size)] = _bits(qkv[t0:t0 + n][:, rows])
            if GP.ab_f32(lay):  # (alpha fp32: two bf16 units each, beta after them)
                a32 = np.ascontiguousarray(alpha[t0:t0 + n][:, vh], np.float32).view(np.uint16)
                Gb[:n, c * lay.pitch + lay.off["alpha"] + np.arange(a32.shape[1])] = a32
                Gb[:n, c * lay.pitch + lay.off["beta"] + np.arange(len(vh))] = _bits(beta[t0:t0 + n][:, vh])
            else:
                ab = np.concatenate([alpha[t0:t0 + n][:, vh], beta[t0:t0 + n][:, vh]], 1)
                Gb[:n, c * lay.pitch + lay.off["ab"] + np.arange(ab.shape[1])] = _bits(ab)
            o = c * lay.pitch + lay.off["qkv"]  # (this chunk's conv state: the reference's, in G's row layout)
            cv[:3, o:o + sh.ch] = torch.from_numpy(cs[c]).bfloat16().view(torch.int16).numpy()
        fused.get_buffer("G").view(torch.int16)[:] = torch.from_numpy(Gb.reshape(-1).view(np.int16))
        mask[:T] = np.arange(T) < n
        fused.sync_inputs()
        fused()
        fused.sync_outputs()
        fused.sync_scratch_from_device()
        O = fused.get_buffer("O").float().numpy().reshape(T, cfg.ssm_v_heads, 128)
        S = fused.get_buffer("GS").view(torch.float32).numpy().reshape(COLS, -1).copy()
        o_want = np.zeros_like(O)
        for c in range(COLS):
            rows = blk.rows[c][blk.rows[c] >= 0]
            vh = gdn_v_heads(cfg, c)
            ab = np.concatenate([alpha[t0:t0 + n][:, vh], beta[t0:t0 + n][:, vh]], 1)
            ab = np.concatenate([ab, np.zeros((T - n, ab.shape[1]), np.float32)])
            x = np.concatenate([qkv[t0:t0 + n][:, rows], np.zeros((T - n, rows.size), np.float32)])
            o_c, st[c] = ref.core_gdn_scan(qkv=x, ab=ab, conv_state=cs[c], conv_w=cw[rows], dt=dt[vh], ssm_a=A[vh],
                                           state=st[c], n=n, eps=cfg.rms_norm_eps)
            cs[c] = np.concatenate([cs[c], x[:n]])[-3:]
            o_want[:, vh] = o_c
            got_s = GD.state_from_panels(S[c].reshape(-1, 1024), sh.hc)
            s_err.append(ref.rel_err(got_s, st[c]))
            assert s_err[-1] < 8e-3, (chunk, c)
            final.append(got_s)
        assert ref.rel_err(O[:n], o_want[:n]) < 1e-2, chunk
        gold = np.asarray(G_["attn_output", L], np.float32).reshape(len(tokens), cfg.ssm_v_heads, 128)[t0:t0 + n]
        assert ref.rel_err(O[:n], gold) < 2e-2, chunk
    # the final state: as close to llama.cpp's as the exact arithmetic's (bf16 inputs on both; e_npu, e_ref: largest)
    gold = np.asarray(G_["new_state", L], np.float32).reshape(-1, 128, 128)
    g8 = np.concatenate([gold[gdn_v_heads(cfg, c)] for c in range(COLS)])
    e_npu, e_ref = ref.rel_err(np.concatenate(final), g8), ref.rel_err(np.concatenate(st), g8)
    r_npu = np.linalg.norm(np.concatenate(final) - g8) / np.linalg.norm(g8)
    r_ref = np.linalg.norm(np.concatenate(st) - g8) / np.linalg.norm(g8)
    print(f"{size}: states vs the cores' exact arithmetic max {max(s_err):.5f}; final state vs llama.cpp {e_npu:.5f}, the "
          f"exact arithmetic's {e_ref:.5f}; rms {r_npu:.6f} / {r_ref:.6f}")
    # (rms: the largest of 2M values moves with any rounding near it, 1B 1.09x the exact arithmetic's, 9B 1.00x)
    assert r_npu < 1.05 * r_ref  # (measured 9B 1.002, 1B 1.004)


@pytest.mark.parametrize("size", ["9B", "1B"])
def test_gdn_out(size: str) -> None:
    """GdnOut on random o and z of the model's shapes: the panels against rms(o) ssm_norm silu(z) in bfp16 blocks;
    the conv state = G's qkv rows of the last three valid tokens, bit for bit."""
    from artlite.operators import bfp16

    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    L, T, n = cfg.gdn_layers[0], 64, 50
    img = QwenImage(g, cfg, layers=[L], vocab=COLS * 64)
    region = img.region(L, "gdn_in")
    lay = PG.plan_phase(img, region, T, pitch=PG.phase_pitch(img, "gdn_in", T)).layout
    op = GP.GdnOutOperator(T=T, cfg=cfg, layout=lay, hslot=HSLOT, cs_off=CS_OFF)
    fused = FusedOperator(name="gdn_out", runlist=[(op, "O", "G", "G", "P", "GH", "A")], input_args=["O", "G", "P"],
                          output_args=["A"]).compile()
    gen = torch.Generator().manual_seed(1)
    hv = cfg.ssm_v_heads
    O = (torch.randn(T, hv * 128, generator=gen) * 0.01).bfloat16()
    G = (torch.randn(T, lay.N, generator=gen)).bfloat16()
    P = GP.scan_params(g, cfg, L, region.block("qkv").rows)
    fused.get_buffer("O")[:] = O.reshape(-1)
    fused.get_buffer("G")[:] = G.reshape(-1)
    fused.get_buffer("P").view(torch.uint8)[:P.size] = torch.from_numpy(P)
    fused.get_buffer("GH").zero_()
    fused.sync_inputs()
    fused.set_params({GP.CONV_OFF: op.conv_offset(n)})
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()
    A = bfp16.from_panels(fused.get_buffer("A").clone(), T, hv * 128).numpy().reshape(T, hv, 128)
    norm = np.asarray(g.tensor(f"blk.{L}.ssm_norm.weight").raw, np.float32).reshape(-1)
    for c in range(COLS):
        vh = gdn_v_heads(cfg, c)
        z = G[:, c * lay.pitch + lay.off["z"]:][:, :len(vh) * 128].float().numpy().reshape(T, -1, 128)
        want = ref.core_gdn_out(O.float().numpy().reshape(T, hv, 128)[:, vh], z, norm, cfg.rms_norm_eps).reshape(T, -1)
        want = bfp16.from_panels(bfp16.to_panels(torch.from_numpy(want)), T, want.shape[1]).numpy()  # (its blocks)
        assert ref.rel_err(A[:, vh].reshape(T, -1), want) < 1.2e-2, c  # (rel_err: max-based; a flip of the
        # 7-bit mantissa in the block of the largest values is ~1 / 100)
        gh = fused.get_buffer("GH").view(torch.int16).numpy()
        o = (c * HSLOT + CS_OFF) // 2
        rows = G[n - 3:n, c * lay.pitch + lay.off["qkv"]:][:, :op.ch].view(torch.int16).numpy().reshape(-1)
        assert np.array_equal(gh[o:o + 3 * op.ch], rows), c


@pytest.mark.parametrize("size", ["1B", "d3-lite"])
def test_gdn_scan_packed(size: str) -> None:
    """GdnPrep and GdnScan packed (``pack``): a 128-token chunk of segments (from position 0) that start inside a
    16-token scan chunk and an 8-token conv block, one of 3 tokens, padding after them, the layer's state garbage before
    it: each segment's o against its tokens alone (the unpacked designs from a zero state and conv state) and the
    cores' arithmetic of them alone (``ref.core_gdn_scan``): within the reference's tolerance as close as alone (no leak
    of a segment's state into the next); one that starts at a scan chunk bit-identical up to the chunk the next one
    starts inside (whose reset changes that chunk's arithmetic)."""
    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    L, T = cfg.gdn_layers[0], 128
    img = QwenImage(g, cfg, layers=[L], vocab=COLS * 64)
    region = img.region(L, "gdn_in")
    lay = PG.plan_phase(img, region, T, pitch=PG.phase_pitch(img, "gdn_in", T)).layout
    blk = region.block("qkv")
    lengths = [37, 8, 3, 50, 20]  # (starts 0, 37, 45, 48, 98; padding from 118)
    starts = np.cumsum([0, *lengths])[:-1]
    n = int(sum(lengths))
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = ref.golden(path, tokens, layers=(L,), names="linear_attn_qkv_mixed|alpha|beta", logits="none")
    qkv = _bf(np.asarray(G_["linear_attn_qkv_mixed", L], np.float32).reshape(n, -1))
    alpha = np.asarray(G_["alpha", L], np.float32).reshape(n, -1)
    alpha = alpha if GP.ab_f32(lay) else _bf(alpha)
    beta = _bf(np.asarray(G_["beta", L], np.float32).reshape(n, -1))
    sh = GD.GdnShape(cfg)
    P = GP.scan_params(g, cfg, L, blk.rows)
    cw = np.asarray(g.tensor(f"blk.{L}.ssm_conv1d.weight").raw, np.float32)
    dt = np.asarray(g.tensor(f"blk.{L}.ssm_dt.bias").raw, np.float32).reshape(-1)
    A = np.asarray(g.tensor(f"blk.{L}.ssm_a").raw, np.float32).reshape(-1)

    def run(pack: bool, sel: np.ndarray, lo: np.ndarray | None) -> np.ndarray:
        prep = GP.GdnPrepOperator(T=T, cfg=cfg, layout=lay, pack=pack)
        scan = GP.GdnScanOperator(T=T, cfg=cfg, layout=lay, pack=pack)
        fused = FusedOperator(name=f"gdn_scan{'_packed' if pack else ''}",
                              runlist=[(prep, "G", "CV", "P", "M", "PR"), (scan, "PR", "G", "CV", "P", "GS", "O")],
                              input_args=["G", "P", "M", "CV"], output_args=["O"]).compile()
        for name in ("G", "P", "M", "CV"):
            fused.get_buffer(name).zero_()
        gs = fused.get_buffer("GS").view(torch.float32)  # (the state before: garbage packed, zero alone)
        gs[:] = torch.randn(gs.numel(), generator=torch.Generator().manual_seed(3)) if pack else 0
        fused.get_buffer("P").view(torch.uint8)[:P.size] = torch.from_numpy(P)
        Gb = np.zeros((T, lay.N), np.uint16)
        for c in range(COLS):
            rows = blk.rows[c][blk.rows[c] >= 0]
            vh = gdn_v_heads(cfg, c)
            Gb[:len(sel), c * lay.pitch + lay.off["qkv"] + np.arange(rows.size)] = _bits(qkv[sel][:, rows])
            if GP.ab_f32(lay):
                a32 = np.ascontiguousarray(alpha[sel][:, vh], np.float32).view(np.uint16)
                Gb[:len(sel), c * lay.pitch + lay.off["alpha"] + np.arange(a32.shape[1])] = a32
                Gb[:len(sel), c * lay.pitch + lay.off["beta"] + np.arange(len(vh))] = _bits(beta[sel][:, vh])
            else:
                ab = np.concatenate([alpha[sel][:, vh], beta[sel][:, vh]], 1)
                Gb[:len(sel), c * lay.pitch + lay.off["ab"] + np.arange(ab.shape[1])] = _bits(ab)
        fused.get_buffer("G").view(torch.int16)[:] = torch.from_numpy(Gb.reshape(-1).view(np.int16))
        m = GP.GdnPrepOperator.mask(len(sel), T, lo, T)
        fused.get_buffer("M").view(torch.float32)[:m.size] = torch.from_numpy(m)
        fused.sync_inputs()
        fused()
        fused.sync_outputs()
        return fused.get_buffer("O").float().numpy().reshape(T, cfg.ssm_v_heads, 128)

    O = run(True, np.arange(n), np.repeat(starts, lengths))
    for a, m in zip(starts, lengths, strict=True):
        sel = np.arange(a, a + m)
        alone = run(False, sel, None)[:m]
        want = np.zeros((T, cfg.ssm_v_heads, 128), np.float32)
        for c in range(COLS):
            rows = blk.rows[c][blk.rows[c] >= 0]
            vh = gdn_v_heads(cfg, c)
            ab = np.concatenate([np.concatenate([alpha[sel][:, vh], beta[sel][:, vh]], 1),
                                 np.zeros((T - m, 2 * len(vh)), np.float32)])
            x = np.concatenate([qkv[sel][:, rows], np.zeros((T - m, rows.size), np.float32)])
            want[:, vh] = ref.core_gdn_scan(qkv=x, ab=ab, conv_state=np.zeros((3, sh.ch), np.float32), conv_w=cw[rows],
                                            dt=dt[vh], ssm_a=A[vh], state=np.zeros((sh.hc, 128, 128), np.float32),
                                            n=m, eps=cfg.rms_norm_eps)[0]
        got = O[a:a + m]
        e_alone, e_ref = ref.rel_err(got, alone), ref.rel_err(got, want[:m])
        diff = np.flatnonzero((got != alone).any((1, 2)))
        print(f"segment {a} + {m}: vs alone {e_alone:.2e} ({m - diff.size} of {m} rows bit-identical; first differing "
              f"{diff[:4].tolist()}), vs the cores' arithmetic {e_ref:.2e} (alone {ref.rel_err(alone, want[:m]):.2e})")
        assert e_ref < 1.05 * ref.rel_err(alone, want[:m]) + 1e-4, a
        if a % GP.C == 0:
            k = m if (a + m) % GP.C == 0 else (a + m) // GP.C * GP.C - a  # (the rows before the next start's chunk)
            assert (got[:k] == alone[:k]).all(), (a, np.flatnonzero((got[:k] != alone[:k]).any((1, 2))).tolist())
