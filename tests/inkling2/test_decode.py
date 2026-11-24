"""Inkling 2's decode overlay on the NPU, piece by piece (``models/inkling2/decode.py``): a visit's GEMV phases on
fp32 state inputs (``decode.protocol.F_STATES``: the helper mixes a k range's states into the main core's res, then
builds the x panels) against numpy's products of the file's dequantized weights."""

import numpy as np
import pytest
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
from artlite.model_config import InklingConfig, inkling_gguf
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction
from models.inkling2.decode import InklingDecodeOperator
from models.inkling2.image import InklingImage

from ..model_files import require
from ..numerics import rel

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
    sp = packed_slots(S)
    K = img.region(0, "q").K  # (3072: the hidden padded to whole chunks)
    state = K * sp * 4
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


def _psums(states: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The partial sums object's payload a producer writes ((COLS, 4 slots, 32) fp32: per column the states' sums of
    squares over its channels, lanes 0 .. 7, their dots with the queries, lanes 8 + 8 j + b) of states (n, S, H) and
    queries (J, H)."""
    n, S, H = states.shape
    h8 = H // COLS
    out = np.zeros((COLS, 4, 32), np.float32)
    for c in range(COLS):
        x = states[..., h8 * c:h8 * (c + 1)].astype(np.float32)
        q = queries[:, h8 * c:h8 * (c + 1)]
        out[c, :S, :n] = (x * x).sum(-1).T
        for j in range(len(q)):
            out[c, :S, 8 + 8 * j:8 + 8 * j + n] = (x @ q[j]).T
    return out


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
    sp = packed_slots(S)
    K = img.region(L, "q").K
    n, nr = 3, 4
    ST = K * sp * 4  # (a state's bytes in the state array)
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
    act[ps_off + XB - XPAY:ps_off + XB] = _psums(states, queries).reshape(-1).view(np.uint8)
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
