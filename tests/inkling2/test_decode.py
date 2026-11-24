"""Inkling 2's decode overlay on the NPU, piece by piece (``models/inkling2/decode.py``): a visit's GEMV phases on
fp32 state inputs (``decode.protocol.F_STATES``: the helper mixes a k range's states into the main core's res, then
builds the x panels) against numpy's products of the file's dequantized weights."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.decode import phases as mp
from artlite.decode.protocol import (
    B_F32,
    B_PAIR,
    C_LEN,
    DIMS_OBJ,
    F_COEF,
    F_NORM,
    F_STATES,
    M_CSET,
    M_NMIX,
    M_NQ,
    M_NR,
    M_NX,
    META,
    XB,
    XPAY,
    packed_slots,
)
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, Visit, inkling_gguf
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction
from models.inkling2.decode import InklingDecodeOperator
from models.inkling2.image import InklingImage
from models.inkling2.overlay import InklingOverlay, psums
from models.tokenizer import BpeTokenizer

from ..model_files import require
from ..numerics import rel
from .ref import Model, VisitCache, rms

PAD = 512  # (bytes before the state array: a state object's fill reads the XB bytes that end with its payload)


@pytest.fixture(scope="module")
def ink() -> tuple[GGUFFile, InklingConfig]:
    path = inkling_gguf()
    require(path)
    g = GGUFFile(path)
    return g, InklingConfig.from_gguf(g)


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32)


@pytest.mark.parametrize("S", [1, 4])
def test_qkv_states(ink: tuple[GGUFFile, InklingConfig], S: int) -> None:
    """Layer 0's q, k, v phases from the normed embeddings of S tokens as one fp32 state (coefficient 1, F_NORM with
    ``attn_norm``): every output row (q, gate, r, the sink's fp32 pair sums, k, v) against numpy."""
    g, cfg = ink
    H, eps = cfg.hidden_size, cfg.rms_norm_eps
    img = InklingImage(g, cfg, layers=[0], vocab=COLS * 64)
    op = InklingDecodeOperator(hidden=H, rows={"Q8_0": 8, "BFP16": 8}, tile_bytes=img.tile_bytes, eps=eps)
    dev = OverlayDevice(op)
    if (store := op.page_store()) is not None:  # (the formats' pages loadable: ``decode.model.ModelOverlay.load``)
        dev.set_prologue(op.page_prologue(store))
    sp = packed_slots(S)
    state = -(-H * sp * 4 // XPAY) * XPAY  # (a state's bytes: the hidden's whole k ranges, ``protocol.F_STATES``)
    K = state // (4 * sp)
    n_obj = state // XPAY
    phases = ("q", "k", "v")
    # (C objects per slot and column: the valid rows' outputs, a pair's sum as fp32 in two bf16 slots)
    out_obj = {n: -(-sum(mp.valid_rows(img.region(0, n))) // C_LEN) for n in phases}
    meta_off, st_off = 0, len(phases) * XB + PAD
    out_off = st_off + state
    out_bytes = {n: COLS * out_obj[n] * S * 2 * C_LEN for n in phases}
    act_bytes = out_off + sum(out_bytes.values())
    dev.alloc_all({"W": img.nbytes, "ACT": act_bytes})
    img.pack(dev.view("W"))
    dev.flush("W")
    act = dev.view("ACT")
    act[:] = 0
    tokens = [9707, 11, 1063, 2201][:S]
    emb = g.rows("token_embd.weight", np.asarray(tokens))
    x = _rms(emb, eps) * g.tensor("token_embd_norm.weight").dequantize().reshape(-1)  # (S, H) fp32
    xs = np.zeros((K // 8, sp, 8), np.float32)
    xs[:H // 8, :S] = x.reshape(S, H // 8, 8).transpose(1, 0, 2)
    act[st_off:st_off + state] = xs.reshape(-1).view(np.uint8)
    for i, name in enumerate(phases):
        gr = mp.groups(img.region(0, name), F_NORM, S, block_flags=dict.fromkeys(("sink", "kd", "vd"), B_PAIR | B_F32))
        assert len(gr) == 1
        m = gr[0].meta.copy()
        m[0] |= F_STATES
        m[M_NX], m[M_NR], m[M_CSET] = n_obj, 1, -1
        act[meta_off + i * XB:][:4 * META] = m.view(np.uint8)
    dev.flush("ACT")
    seq = Sequence(op, inputs=("X",))
    o = out_off
    for i, name in enumerate(phases):
        r0, r = img.region(0, name + "_norm"), img.region(0, name)
        for rr, n in img.runs(r0, r):
            for c in range(COLS):
                seq.fill("A", c, dev.address("W", rr.offset + c * rr.col_stride), n)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", meta_off + i * XB), XB)
            seq.fill("X", c, dev.address("ACT", st_off - (XB - XPAY)), n_obj * XB, DIMS_OBJ)
        per = out_obj[name] * S * 2 * C_LEN
        for c in range(COLS):
            seq.drain("C", c, dev.address("ACT", o + c * per), per)
        o += COLS * per
        seq.sync()
    seq.sync(fills=True)
    dev.dispatch(Transaction(dev, seq.words()))
    dev.fetch("ACT")
    act = dev.view("ACT")
    o = out_off
    for name in phases:
        r = img.region(0, name)
        per = out_obj[name] * S * 2 * C_LEN
        w = g.tensor("blk.0.attn_norm.weight").dequantize().reshape(-1)
        xn = _rms(x, eps) * w
        for c in range(COLS):
            v = act[o + c * per:o + (c + 1) * per].view(bfloat16)
            v = v.reshape(out_obj[name], S, C_LEN).transpose(1, 0, 2).reshape(S, -1)  # ([slot][outputs])
            at = 0
            for b in r.blocks:
                rows = b.rows[c][b.rows[c] >= 0]
                if b.pairs:  # (planes: the pairs' sums as fp32, two bf16 slots each)
                    got = v[:, at:at + len(rows)].astype(bfloat16).copy().view(np.float32)
                    t = img.tensor(b.tensors[0])
                    wr = np.asarray(t.raw, np.float32).reshape(t.rows, t.cols)[rows[::2] // 2]
                else:
                    got = v[:, at:at + len(rows)].astype(np.float32)
                    wr = g.rows(b.tensors[0], rows)
                at += len(rows)
                want = xn @ wr.T
                err = rel(got, want)
                assert err < (2e-3 if b.pairs else 8e-3), (name, b.name, c, err)
        o += COLS * per


@pytest.mark.parametrize("S", [1, 4])
def test_mix(ink: tuple[GGUFFile, InklingConfig], S: int) -> None:
    """A full layer's (5) attention inputs as depth mixes of 3 states (read as 4: the fourth's coefficient 0): the q
    phase's partial sums object (F_COEF) gives the helper the three queries' coefficient sets, the q, k and v phases mix
    the states by sets 0, 1, 2 (F_STATES), each normed by its own ``attn_*_inp_norm``: the outputs against numpy's mixes
    (``InklingConfig``'s depth mix) and products."""
    g, cfg = ink
    H, eps, L = cfg.hidden_size, cfg.rms_norm_eps, 5
    img = InklingImage(g, cfg, layers=[L], vocab=COLS * 64)
    op = InklingDecodeOperator(hidden=H, rows={"Q8_0": 8, "BFP16": 8}, tile_bytes=img.tile_bytes, eps=eps)
    dev = OverlayDevice(op)
    if (store := op.page_store()) is not None:  # (the formats' pages loadable: ``decode.model.ModelOverlay.load``)
        dev.set_prologue(op.page_prologue(store))
    sp = packed_slots(S)
    n, nr = 3, 4
    ST = -(-H * sp * 4 // XPAY) * XPAY  # (a state's bytes in the state array: the hidden's whole k ranges)
    K = ST // (4 * sp)
    ranges = ST // XPAY
    phases = ("q", "k", "v")
    out_obj = {m: -(-sum(mp.valid_rows(img.region(L, m))) // C_LEN) for m in phases}
    meta_off, ps_off, st_off = 0, len(phases) * XB, len(phases) * XB + XB + PAD
    out_off = st_off + nr * ST
    act_bytes = out_off + sum(COLS * out_obj[m] * S * 2 * C_LEN for m in phases)
    dev.alloc_all({"W": img.nbytes, "ACT": act_bytes})
    img.pack(dev.view("W"))
    dev.flush("W")
    act = dev.view("ACT")
    act[:] = 0
    rng = np.random.default_rng(1)
    toks = rng.integers(0, cfg.vocab_size, (n, S))
    tn = g.tensor("token_embd_norm.weight").dequantize().reshape(-1)
    states = np.stack([_rms(g.rows("token_embd.weight", toks[b]), eps) * tn * (1.0 + 2.0 * b) for b in range(n)])
    for b in range(n):  # ([state][k block][slot sp][8] fp32)
        xs = np.zeros((K // 8, sp, 8), np.float32)
        xs[:H // 8, :S] = states[b].reshape(S, H // 8, 8).transpose(1, 0, 2)
        act[st_off + b * ST:st_off + (b + 1) * ST] = xs.reshape(-1).view(np.uint8)
    queries = np.asarray(g.tensor(f"blk.{L}.attn_res_query.weight").dequantize(), np.float32).reshape(3, H)
    act[ps_off + XB - XPAY:ps_off + XB] = psums(states, queries).reshape(-1).view(np.uint8)
    for i, name in enumerate(phases):
        gr = mp.groups(img.region(L, name), F_NORM, S, block_flags=dict.fromkeys(("sink", "kd", "vd"), B_PAIR | B_F32))
        m = gr[0].meta.copy()
        m[0] |= F_STATES | (F_COEF if i == 0 else 0)
        m[M_NX], m[M_NR], m[M_CSET], m[M_NQ], m[M_NMIX] = ranges * nr, nr, i, 3, n
        act[meta_off + i * XB:][:4 * META] = m.view(np.uint8)
    dev.flush("ACT")
    seq = Sequence(op, inputs=("X",))
    o = out_off
    for i, name in enumerate(phases):
        r0, r = img.region(L, name + "_norm"), img.region(L, name)
        for rr, nb in img.runs(r0, r):
            for c in range(COLS):
                seq.fill("A", c, dev.address("W", rr.offset + c * rr.col_stride), nb)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", meta_off + i * XB), XB)
            if i == 0:
                seq.fill("X", c, dev.address("ACT", ps_off), XB)
            seq.fill("X", c, dev.address("ACT", st_off - (XB - XPAY)), nr * XB, (XB // 8, 2, XB // 8, ST // 4),
                     repeat=ranges, iteration=(XPAY // 4, ranges))
        per = out_obj[name] * S * 2 * C_LEN
        for c in range(COLS):
            seq.drain("C", c, dev.address("ACT", o + c * per), per)
        o += COLS * per
        seq.sync()
    seq.sync(fills=True)
    dev.dispatch(Transaction(dev, seq.words()))
    dev.fetch("ACT")
    act = dev.view("ACT")
    r_b = np.sqrt((states.astype(np.float64) ** 2).mean(-1) + InklingConfig.MIX_EPS)  # (n, S)
    o = out_off
    for i, name in enumerate(phases):
        sc = np.einsum("bsh,h->bs", states / r_b[..., None], queries[i]) / np.sqrt(H)
        w = np.exp(sc - sc.max(0))
        w /= w.sum(0)
        mix = np.einsum("bs,bsh->sh", w, states / r_b[..., None]).astype(np.float32)
        xn = _rms(mix, eps) * g.tensor(f"blk.{L}.attn_{name}_inp_norm.weight").dequantize().reshape(-1)
        r = img.region(L, name)
        per = out_obj[name] * S * 2 * C_LEN
        for c in (0, 3, 7):
            v = act[o + c * per:o + (c + 1) * per].view(bfloat16)
            v = v.reshape(out_obj[name], S, C_LEN).transpose(1, 0, 2).reshape(S, -1)
            at = 0
            for b in r.blocks:
                rows = b.rows[c][b.rows[c] >= 0]
                if b.pairs:
                    got = v[:, at:at + len(rows)].astype(bfloat16).copy().view(np.float32)
                    t = img.tensor(b.tensors[0])
                    wr = np.asarray(t.raw, np.float32).reshape(t.rows, t.cols)[rows[::2] // 2]
                else:
                    got = v[:, at:at + len(rows)].astype(np.float32)
                    wr = g.rows(b.tensors[0], rows)
                at += len(rows)
                err = rel(got, xn @ wr.T)
                assert err < (2e-3 if b.pairs else 8e-3), (name, b.name, c, err)
        o += COLS * per


def _attention_ref(ref: "Model", v: Visit, x: torch.Tensor, rows: int) -> torch.Tensor:
    """The reference's attention output (the gated o before W_o) of visit v for inputs x (T, H) at positions 0 .. T - 1
    (the visit's three inputs alike: one state; ``ref.Model.visit``'s steps), the last ``rows`` positions'."""
    c, b, L = ref.cfg, f"blk.{v.layer}.", v.layer
    T, eps, hd = x.shape[0], c.rms_norm_eps, c.head_dim
    swa = v.kind == "swa"
    norm = (lambda n: ref.vec(b + "attn_norm.weight")) if swa else (lambda n: ref.vec(b + f"attn_{n}_inp_norm.weight"))
    xq, xk, xv = (rms(x, eps) * norm(n) for n in "qkv")
    q, kraw, vraw = ref.mm(xq, b + "attn_q.weight"), ref.mm(xk, b + "attn_k.weight"), ref.mm(xv, b + "attn_v.weight")
    kdyn = None if swa else ref.mm(xk, b + "attn_k_conv_proj.weight")
    vdyn = None if swa else ref.mm(xv, b + "attn_v_conv_proj.weight")
    k, _ = ref.conv(kraw, ref.w(b + "attn_k_conv.weight"), None, kdyn)
    vv, _ = ref.conv(vraw, ref.w(b + "attn_v_conv.weight"), None, vdyn)
    r = ref.mm(xq, b + "attn_rel.weight")
    nh, kvh = c.num_attention_heads, c.num_key_value_heads
    q = ref.clip(q.reshape(T, nh, hd), ref.w(b + "attn_q_clip.weight"), c.qk_clip)
    k = ref.clip(k.reshape(T, kvh, hd), None, c.qk_clip)
    vv = rms(vv.reshape(T, kvh, hd), eps)
    r = ref.clip(r.reshape(T, nh, c.rel_length), ref.w(b + "attn_rel_clip.weight"), c.rel_clip)
    if not swa:
        tau = torch.from_numpy(c.tau(range(T)))[:, None, None]
        q, r = q * tau, r * tau
    o = ref.attention(L, q, k, vv, r, ref.mm(xq, b + "attn_sink_proj.weight"), 0)
    o = o * torch.sigmoid(ref.mm(xq, b + "attn_gate.weight"))
    return o[T - rows:]


@pytest.mark.parametrize("v", [0, 5])
def test_attention(ink: tuple[GGUFFile, InklingConfig], v: int) -> None:
    """A visit's q, k, v phases and its attention (``Attn(ink=True)``) over 80 steps of one token, then two of 4 and
    one of 3 (crossing groups and records): o (gated) of every step against the reference's attention from the same
    inputs (fp32 activations; a sliding-window visit and a full one: tau, the taps' deltas)."""
    g, cfg = ink
    ov = InklingOverlay(g, visits=[v], max_positions=1024, ffn=False)
    ov.load()
    ref = Model(g, cfg, q8=False)
    rng = np.random.default_rng(3)
    toks = rng.integers(0, cfg.vocab_size, 91)
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * tn
    vis = cfg.visits[v]
    with torch.no_grad():
        want = _attention_ref(ref, vis, x, 91).numpy()
    errs = []
    p = 0
    for S in [1] * 80 + [4, 4, 3]:
        ov.step(np.stack([x[p:p + S].numpy()] * (ov.n0 + 1)), p)  # (a mix of equal states: x / rms(x))
        got = ov.attn_out(v, S)
        errs.append(rel(got, want[p:p + S]))
        p += S
    assert max(errs) < 2e-2, (np.argmax(errs), max(errs), np.median(errs))


@pytest.mark.parametrize("v", [0, 5])
def test_o(ink: tuple[GGUFFile, InklingConfig], v: int) -> None:
    """A visit's o phase with the residual stream's step (F_INK) over steps of 1, then 4, 4, 3 slots: W_o's rows, the
    post conv over the visit's history window, cur (visit 0: a block visit, its output a new state; visit 5: added to cur,
    the states x and 2 x), the partial sums of the FFN's mix, the history window, against the reference."""
    g, cfg = ink
    ov = InklingOverlay(g, visits=[v], max_positions=1024, ffn=False)
    ov.load()
    ref = Model(g, cfg, q8=False)
    rng = np.random.default_rng(5)
    T = 31
    toks = rng.integers(0, cfg.vocab_size, T)
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * tn
    vis, L = cfg.visits[v], cfg.visits[v].layer
    b = f"blk.{L}."
    with torch.no_grad():
        y = ref.mm(_attention_ref(ref, vis, x, T), b + "attn_output.weight")
        out, _ = ref.conv(y, ref.w(b + "attn_post_conv.weight"), None)
    states = np.stack([x.numpy(), 2 * x.numpy()])[:ov.n0 + 1]  # (T, H) each
    cur = out.numpy() if vis.block else states[-1] + out.numpy()
    qf = np.asarray(g.tensor(b + "ffn_res_query.weight").dequantize(), np.float32).reshape(1, -1)
    n1 = ov.n0 + int(vis.block)
    errs, perr = [], []
    p = 0
    for S in [1] * 20 + [4, 4, 3]:
        ov.step(states[:, p:p + S], p)
        got = ov.state(n1, S)
        errs.append(rel(got, cur[p:p + S]))
        mixed = np.concatenate([states[:n1, p:p + S], cur[None, p:p + S]])  # (states 0 .. n1 - 1, cur)
        want = psums(mixed, qf)
        perr.append(rel(ov.psums(v, "f")[:, :S], want[:, :S]))
        p += S
    assert max(errs) < 1e-2, (np.argmax(errs), max(errs), np.median(errs))
    assert max(perr) < 2e-2, (np.argmax(perr), max(perr), np.median(perr))
    hist = ov.history(v, 0, 3 + 3)  # (positions p - 3 .. p + 2 of the last step at p = T - 3)
    assert rel(hist, y[T - 6:].numpy()) < 1e-2


@pytest.mark.parametrize("v", [0, 5])
def test_ffn(ink: tuple[GGUFFile, InklingConfig], v: int) -> None:
    """A whole visit with its FFN (the MoE: the router on the FFN's mix, the routing by sigmoid + bias, the units' situ
    and downs, the combine and its residual step) over steps of 1, then 4, 4, 3 slots: the visit's output (cur) and the
    next visit's attention-mix partial sums against the reference's ``visit`` from the same states (fp32; the NPU's
    activations bf16), but where the routing's 8th and 9th keys tie within 5e-3 (a near-tie may choose the other
    expert)."""
    g, cfg = ink
    ov = InklingOverlay(g, visits=[v], max_positions=1024)
    ov.load()
    ref = Model(g, cfg, q8=False)
    ref.record = {"ffn_moe_logits"}
    rng = np.random.default_rng(7)
    T = 31
    toks = rng.integers(0, cfg.vocab_size, T)
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * tn
    vis = cfg.visits[v]
    x2 = rms(torch.from_numpy(g.rows("token_embd.weight", rng.integers(0, cfg.vocab_size, T))), cfg.rms_norm_eps) * tn
    states = np.stack([x.numpy(), 3 * x2.numpy()])[:ov.n0 + 1]  # (distinct states: the mixes' coefficients matter)
    blocks = [torch.from_numpy(b) for b in states[:-1]]
    with torch.no_grad():
        want = ref.visit(vis, torch.from_numpy(states[-1]), blocks, VisitCache(), 0).numpy()
    n1 = ov.n0 + int(vis.block)
    La, nq = ov.queries_after(v)
    qn = np.asarray(g.tensor(f"blk.{La}.attn_res_query.weight").dequantize(), np.float32).reshape(nq, -1)
    # (the routing's keys: where the 8th and 9th are within 5e-3, the logits' ~0.3% from the attention's bf16 outputs may
    # pick the other expert)
    bias = np.asarray(g.tensor(f"blk.{vis.layer}.exp_probs_b.bias").dequantize(), np.float32).reshape(-1)
    keys = np.sort(1 / (1 + np.exp(-ref.out["ffn_moe_logits", vis.slot].numpy().astype(np.float64))) + bias, 1)[:, ::-1]
    tie = keys[:, cfg.n_experts_used - 1] - keys[:, cfg.n_experts_used] < 5e-3
    errs, perr = np.zeros(T), np.zeros(T)
    p = 0
    for S in [1] * 20 + [4, 4, 3]:
        ov.step(states[:, p:p + S], p)
        got = ov.state(n1, S)
        errs[p:p + S] = [rel(got[i], want[p + i]) for i in range(S)]
        mixed = np.concatenate([states[:n1, p:p + S], want[None, p:p + S]])
        ps, pw = ov.psums(v + 1, "a"), psums(mixed, qn)
        perr[p:p + S] = [rel(ps[:, i], pw[:, i]) for i in range(S)]
        p += S
    assert errs[~tie].max() < 3e-2 and np.median(errs) < 1e-2, (np.argmax(errs * ~tie), errs[~tie].max(), np.median(errs))
    assert perr[~tie].max() < 4e-2 and np.median(perr) < 2e-2, (perr[~tie].max(), np.median(perr))


def test_prelude(ink: tuple[GGUFFile, InklingConfig]) -> None:
    """The prelude's six visits in one step (the states, the mixes' partial sums, the block state of visit 0 chained on
    the NPU) over steps of 1, then 4, 3 slots: the prelude's output and the next visit's attention-mix partial sums
    against the reference's visits, but where a visit's routing ties (as ``test_ffn``)."""
    g, cfg = ink
    ov = InklingOverlay(g, visits=list(range(6)), max_positions=1024)
    ov.load()
    ref = Model(g, cfg, q8=False)
    ref.record = {"ffn_moe_logits"}
    rng = np.random.default_rng(11)
    T = 15
    toks = rng.integers(0, cfg.vocab_size, T)
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * tn
    cur, blocks = x, []
    tie = np.zeros(T, bool)
    with torch.no_grad():
        for v in range(6):
            vis = cfg.visits[v]
            cur = ref.visit(vis, cur, blocks, VisitCache(), 0)
            bias = np.asarray(g.tensor(f"blk.{vis.layer}.exp_probs_b.bias").dequantize(), np.float32).reshape(-1)
            lg = ref.out["ffn_moe_logits", vis.slot].numpy().astype(np.float64)
            keys = np.sort(1 / (1 + np.exp(-lg)) + bias, 1)[:, ::-1]
            tie |= keys[:, cfg.n_experts_used - 1] - keys[:, cfg.n_experts_used] < 5e-3
    want = cur.numpy()
    qn = np.asarray(g.tensor("blk.6.attn_res_query.weight").dequantize(), np.float32).reshape(1, -1)
    errs, perr = np.zeros(T), np.zeros(T)
    p = 0
    for S in [1] * 8 + [4, 3]:
        ov.step(x.numpy()[None, p:p + S], p)
        got = ov.state(1, S)
        errs[p:p + S] = [rel(got[i], want[p + i]) for i in range(S)]
        mixed = np.stack([x.numpy()[p:p + S], want[p:p + S]])
        ps, pw = ov.psums(6, "a"), psums(mixed, qn)
        perr[p:p + S] = [rel(ps[:, i], pw[:, i]) for i in range(S)]
        p += S
    assert errs[~tie].max() < 3e-2 and np.median(errs) < 1.5e-2, (errs, tie)
    assert perr[~tie].max() < 4e-2, (perr, tie)


@pytest.mark.parametrize("v", [18, 23])
def test_loop(ink: tuple[GGUFFile, InklingConfig], v: int) -> None:
    """A loop visit after the first pass (its streams' controllers: the down's halves over the mix and the bank's
    message, the blends; visit 23 also copies its output into its pass's bank) from distinct states and a bank, over steps
    of 1, then 4, 3 slots: its output against the reference's ``visit`` with the messages, but where the routing ties."""
    g, cfg = ink
    ov = InklingOverlay(g, visits=[v], max_positions=1024)
    ov.load()
    ref = Model(g, cfg, q8=False)
    ref.record = {"ffn_moe_logits"}
    rng = np.random.default_rng(13)
    T = 15
    tn = torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))

    def emb(scale: float) -> np.ndarray:
        x = rms(torch.from_numpy(g.rows("token_embd.weight", rng.integers(0, cfg.vocab_size, T))), cfg.rms_norm_eps)
        return (x * tn).numpy() * scale

    states = np.stack([emb(1.0), emb(3.0), emb(2.0), emb(5.0)])[:ov.n0 + 1]
    bank = np.stack([emb(4.0), emb(7.0)])
    vis = cfg.visits[v]
    blocks = [torch.from_numpy(b) for b in states[:-1]]
    with torch.no_grad():
        msgs = ref.mix([torch.from_numpy(bank[0])], torch.from_numpy(bank[1]), ref.msg_queries(vis.layer))
        want = ref.visit(vis, torch.from_numpy(states[-1]), blocks, VisitCache(), 0, msgs).numpy()
    bias = np.asarray(g.tensor(f"blk.{vis.layer}.exp_probs_b.bias").dequantize(), np.float32).reshape(-1)
    keys = np.sort(1 / (1 + np.exp(-ref.out["ffn_moe_logits", vis.slot].numpy().astype(np.float64))) + bias, 1)[:, ::-1]
    tie = keys[:, cfg.n_experts_used - 1] - keys[:, cfg.n_experts_used] < 5e-3
    n1 = ov.n0 + int(vis.block)
    errs = np.zeros(T)
    p = 0
    for S in [1] * 8 + [4, 3]:
        ov.step(states[:, p:p + S], p, bank[:, p:p + S])
        got = ov.state(n1, S)
        errs[p:p + S] = [rel(got[i], want[p + i]) for i in range(S)]
        p += S
    assert errs[~tie].max() < 3e-2 and np.median(errs) < 1e-2, (errs, tie)


@pytest.mark.slow
def test_model(ink: tuple[GGUFFile, InklingConfig]) -> None:
    """The whole model's decode, a token per step (the 48 visits, the output's mix and norm, the screened head),
    teacher-forced over a short text: each step's exact top token and the reference's top 5 logits (softcapped) against
    the reference model's."""
    g, cfg = ink
    ids = BpeTokenizer(g).encode("The quick brown fox jumps over the lazy dog. The capital of France is")
    T = len(ids)
    ov = InklingOverlay(g, max_positions=1024)
    ov.load()
    ref = Model(g, cfg, q8=False)
    with torch.no_grad():
        want = ref.forward(ids).numpy()
    tn = np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1)
    x = _rms(g.rows("token_embd.weight", np.asarray(ids)), cfg.rms_norm_eps) * tn
    cap = cfg.softcap
    agree, diffs = 0, []
    for p in range(T):
        row = ov.step(x[None, p:p + 1], p, logits=True)[0]
        top = np.argsort(-want[p])[:5]
        got = np.tanh(row.at(top) / cap) * cap
        diffs.append(float(np.abs(got - want[p, top]).max()))
        agree += int(row.top(1)[1][0] == top[0])
    assert agree >= T - 1 and np.median(diffs) < 0.5, (agree, T, diffs)
