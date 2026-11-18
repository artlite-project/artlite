"""The attention core of the gated attention of head_dim 256 (``decode.attn256``) on the NPU on random inputs: in the qwen35
models' decode design (``models.qwen35.decode``) against ``tests/qwen35/ref.py``'s ``core_attention``; with Spark-X's
spec (no norms, a gate per head, sliding-window layers, one pass) against the fp64 attention of what each slot sees."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode import attn256 as A1
from artlite.decode import kv as KV
from artlite.decode import protocol as PR
from artlite.decode.attn256_design import Attn256DecodeOperator
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, SparkConfig, spark_gguf
from artlite.operators.bfp16 import block_bytes, block_values
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction
from models.qwen35 import decode as MD

from ..model_files import require, size_gguf
from ..numerics import bf16_round, from_bf16_bits, rel_max
from ..qwen35 import ref


def _enc(rows: np.ndarray, n: int) -> np.ndarray:
    """A V record's values as its panels hold them: rows (8, 256) of its positions, those from n on zero, encoded as
    blocks over the positions."""
    r = np.asarray(rows, np.float32).copy()
    r[n:] = 0
    v = block_values(block_bytes(r.reshape(8, A1.ND, 8).transpose(1, 2, 0)))  # [db][i][position]
    return np.ascontiguousarray(v.transpose(2, 0, 1)).reshape(8, A1.HD)


@pytest.mark.parametrize(("size", "p", "S", "split", "rec"), [
    ("9B", 38, 2, False, False), ("9B", 5, 1, False, True), ("9B", 150, 4, True, True), ("9B", 1001, 3, True, False),
    ("27B", 0, 1, False, False), ("27B", 36, 4, False, True), ("27B", 150, 2, True, True), ("27B", 1000, 4, True, False),
    ("1B", 38, 2, False, False), ("1B", 150, 2, True, True), ("1B", 1001, 4, True, False), ("1B", 129, 1, True, False),
    ("35B-A3B", 38, 2, False, False), ("35B-A3B", 1001, 2, True, True), ("35B-A3B", 768, 1, True, False),
    ("35B-A3B", 150, 4, 2, False), ("35B-A3B", 1001, 3, 2, True),
    ("3.8-27B", 62, 5, False, False), ("3.8-27B", 160, 8, False, True), ("3.8-27B", 1001, 7, False, False)])
def test_attention(size: str, p: int, S: int, split: int, rec: bool) -> None:
    _attention(size, p, S, split, rec)


@pytest.mark.parametrize(("size", "p", "S", "split"), [("35B-A3B", 150, 2, True)])
def test_attention_later_keys(size: str, p: int, S: int, split: int) -> None:
    """Keys past the first group far above it, as a chat's text above its system prompt: the softmax follows them."""
    _attention(size, p, S, split, False, far=30.0, tol=0.1)


def _attention(size: str, p: int, S: int, split: int, rec: bool, far: float = 1.0, tol: float = 3e-2) -> None:
    """A sub-step of S slots at positions p .. p + S - 1 over a cache of p positions on random inputs: per column and
    slot o within bfp16 / bf16 rounding of ref.core_attention over what the slot sees (closed records as the cache
    holds them, the open ones encoded from their exact rows up to the slot's position: the ring slot's rows before p,
    with ``rec`` the V panels' values a prefill left, as ``AttentionSteps._rings_from_prefill`` puts them in the ring, and
    the new ones); the relayed records (a chain's first column: the K records of the new positions; its last: their
    records' V panels up to the last one and rows). split: the chain of a
    KV head's columns (9B / 27B 2, 1B 4) divides the cache's groups, the partials merged west to east, the last column
    emitting o of the chain's heads (35B-A3B: chains of 4 columns of 2 heads, split sub-steps of up to two slots; split 2:
    of 3 or 4 in two rounds, one per KV head, both chains on its cache, each with half its heads, on the first chain's
    streams shared with the columns 4 away; its own chain's records). The
    3.8-27B on a design of 8 slots (one pass): each core over the whole cache with its own heads, on its chain's first
    column's stream, a step across a cache group too; o an octet's 8 rows."""
    path = size_gguf(size)
    require(path)
    cfg = QwenConfig.from_gguf(GGUFFile(path))
    g = p // 32
    one = size == "3.8-27B"
    assert one or ((p + S - 1) // 32 == g and A1.AttnShape.splits(g) == bool(split))
    rounds = split == A1.SPLIT_ROUND
    P = 32 * ((p + S - 1) // 32 + 1)  # (cache positions: whole groups)
    slots = 8 if one else PR.SLOTS
    ash = A1.AttnShape(cfg, P, slots=slots)
    op = MD.QwenDecodeOperator(hidden=cfg.hidden_size, rows={"Q4_K": 16}, eps=cfg.rms_norm_eps, attn=ash, slots=slots)
    dev = OverlayDevice(op)
    rng = np.random.default_rng(3)
    qc, HD, T, nph = ash.qc, A1.HD, op.tile_bytes, ash.nph
    kvh = cfg.num_key_value_heads
    rp, rl = p // 8, (p + S - 1) // 8
    # the cache: positions 0 .. p - 1 of every KV head (K as stored: normed, rotated; V bf16), the rest of its groups
    # zero; per head its ring (record rp's rows in slot rp & 1)
    kc = rng.standard_normal((kvh, P, HD)).astype(np.float32)
    vc = bf16_round(rng.standard_normal((kvh, P, HD)).astype(np.float32))
    kc[:, p:] = vc[:, p:] = 0
    kc[:, 32:] *= far
    groups = [KV.kv_groups(kc[h], vc[h], T, 1, dmajor=True) for h in range(kvh)]
    kc_s = [KV.kv_from_groups(gr, HD, dmajor=True)[0] for gr in groups]  # (as stored)
    vc_s = [KV.kv_from_groups(gr, HD, dmajor=True)[1] for gr in groups]
    hb = groups[0].nbytes
    rb = 2 * T  # (a head's ring)
    kv_bytes = kvh * (hb + rb) + T
    dev.alloc("KV", kv_bytes)
    kvv = dev.view("KV")
    for h in range(kvh):
        kvv[h * hb:(h + 1) * hb] = groups[h].reshape(-1)
        rows = (vc_s[h] if rec else vc[h])[8 * rp:8 * rp + 8].reshape(8, A1.ND, 8).transpose(1, 0, 2)  # ([d-block][position][8])
        o = kvh * hb + h * rb + (rp & 1) * T
        kvv[o:o + A1.ROWS_B] = interleave_bits(rows.reshape(-1))
    dev.flush("KV")
    qg = bf16_round(rng.standard_normal((COLS, S, qc, 2 * HD)).astype(np.float32))  # [q | gate] per head
    kv = bf16_round(rng.standard_normal((S, kvh, 2 * HD)).astype(np.float32))  # [K | V] per KV head
    qn = (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32)
    kn = (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32)
    cs = [tuple(v[0] for v in A1.rope_cs(cfg, [p + s])) for s in range(S)]
    G = (p + S - 1) // 32 + 1
    b = ash.bounds(g) if split else None  # (split: chain core k's attention groups b[k] .. b[k + 1] - 1)
    o_k = A1.OBJS  # (a group's cache objects)
    nt = ash.records(S)
    rnds = (A1.SPLIT_ROUND, A1.SPLIT_ROUND | A1.SPLIT_KV) if rounds else (int(split),)  # (the sub-step's rounds)
    # X: per column [F_ATT | q objects SLOTS], then the raw K, V objects (shared), then per column a second round's F_ATT
    per = (1 + slots) * PR.XB
    act = dev.alloc("ACT", COLS * per + S * PR.XB + COLS * PR.XB)

    def fatt(c: int, u: int) -> int:
        return c * per if u == 0 else COLS * per + S * PR.XB + c * PR.XB

    for c in range(COLS):
        k = c % nph
        tl = nt if not split or k == nph - 1 else 0
        for u, sp in enumerate(rnds):
            m = np.zeros(PR.META, np.int32)
            sw = ash.round_word(sp, c)
            m[PR.M_FLAGS], m[PR.M_NS], m[PR.M_SKIP] = A1.F_ATT, S, (3 if rounds else nph + 1 if split else 2) * S
            m[PR.M_NREL], m[A1.M_POS], m[A1.M_SPLIT] = ash.n_rel(c, sw, S), p, sw
            m[A1.M_NTAIL] = tl
            if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                m[PR.M_NASKIP], m[A1.M_G0], m[A1.M_GATT] = o_k * G + tl + (2 if k == 0 and not one else 0), 0, 1 << 20
            elif k == 0:  # (then group g's K and V objects)
                m[PR.M_NASKIP], m[A1.M_G0], m[A1.M_GATT], m[A1.M_GJ] = o_k * b[1] + 2, 0, b[1], g
            else:
                m[PR.M_NASKIP], m[A1.M_G0], m[A1.M_GATT] = o_k * (b[k + 1] - b[k]) + tl, b[k], 1 << 20
            if one:  # (on the first column's stream; the rows' masks and K lanes)
                m[A1.M_ANA], m[PR.M_NASKIP] = m[PR.M_NASKIP], 0 if k else m[PR.M_NASKIP]
                m[A1.M_BRANCH], w, m[A1.M_ALANE] = A1.step_words(p, S)
                m[A1.M_AMASK:A1.M_AMASK + 4] = w.view(np.int32)
            o = fatt(c, u)
            act[o:o + 4 * PR.META] = m.view(np.uint8)
            nrm = np.zeros(2 * HD + 64 * ash.smax, np.float32)
            nrm[:HD], nrm[HD:2 * HD] = qn * np.float32(np.log2(np.e) / 16), kn
            for s in range(S):
                nrm[2 * HD + 64 * s:2 * HD + 64 * s + 64] = np.concatenate(cs[s])
            act[o + 4 * PR.META:o + 4 * PR.META + nrm.nbytes] = nrm.view(np.uint8)
        for s in range(S):
            q0 = c * per + (1 + s) * PR.XB + 4 * PR.META
            act[q0:q0 + 2 * qc * 2 * HD] = interleave_bits(qg[c, s].reshape(-1))
    for s in range(S):
        k0 = COLS * per + s * PR.XB + 4 * PR.META
        act[k0:k0 + 2 * kvh * 2 * HD] = interleave_bits(kv[s].reshape(-1))
    dev.flush("ACT")
    ob = nph * qc * 32 * (8 if one else S) * 16  # (o of a chain's heads)
    nr = 2 * (A1.REC_V + A1.ROWS_B)  # (the relayed records' bytes at most)
    dev.alloc("OUT", len(rnds) * COLS * ob)
    dev.alloc("REC", COLS * nr)
    seq = Sequence(op)
    def cache(u: int) -> None:
        kh = (rnds[u] & A1.SPLIT_KV) >> 2
        for c in range(0, COLS, nph if one else 1):
            k, role = c % nph, ash.role(c)
            if rounds and c >= nph:  # (the first chain's streams, for the column 4 away too)
                continue
            pk = {"packet": PR.AB_ID + ash.a_bcast.index((c, c + nph))} if rounds else {}
            h = kh if rounds else c // nph
            base, ring = dev.address("KV", h * hb), dev.address("KV", kvh * hb + h * rb)
            if not split:
                seq.fill("A", c, base, o_k * G * T)
            else:
                seq.fill("A", c, base + o_k * b[k] * T, o_k * (b[k + 1] - b[k]) * T, **pk)
                if k == 0:
                    seq.fill("A", c, base + o_k * g * T, 2 * T, **pk)
            if not split or role == A1.ROLE_V:  # (the tail objects)
                for t in range(nt):
                    seq.fill("A", c, ring + ((rl if t else rp) & 1) * T, T, **pk)
            if not split and k == 0 and not one:
                seq.fill("A", c, base + o_k * g * T, 2 * T)

    def inputs(u: int) -> None:
        kh = (rnds[u] & A1.SPLIT_KV) >> 2
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", fatt(c, u)), PR.XB)
            if split:  # (the q objects of the chain's columns in order; a round: its half of the KV head's)
                q0, nc = (kh * nph + 2 * (c // nph), 2) if rounds else (c - c % nph, nph)
                for c1 in range(q0, q0 + nc):
                    seq.fill("X", c, dev.address("ACT", c1 * per + PR.XB), S * PR.XB)
            else:
                seq.fill("X", c, dev.address("ACT", c * per + PR.XB), S * PR.XB)
            seq.fill("X", c, dev.address("ACT", COLS * per), S * PR.XB)

    def outputs(u: int) -> None:
        for c in range(COLS):
            role = A1.ROLE_O if ash.round_word(rnds[u], c) & A1.SPLIT_NOREC else ash.role(c)
            no = (ob // 2 if rounds else ob) if split and c % nph == nph - 1 else 0 if split else ob // nph
            if no:
                seq.drain("C", c, dev.address("OUT", (u * COLS + c) * ob), no)
            rb_c = nt * {A1.ROLE_K: (1 if one else 2) * A1.REC_K, A1.ROLE_V: A1.REC_V + A1.ROWS_B, A1.ROLE_O: 0}[role]
            if rb_c:
                seq.drain("C", c, dev.address("REC", c * nr), rb_c)

    # (as the overlay: a second round on the same streams, its inputs and cache objects after the first's, then the
    # outputs)
    cache(0)
    inputs(0)
    if rounds:
        inputs(1)
        cache(1)
    for u in range(len(rnds)):
        outputs(u)
    seq.sync(fills=True)
    dev.dispatch(Transaction(dev, seq.words()))
    dev.fetch("OUT")
    dev.fetch("REC")
    outs = dev.view("OUT").view(np.uint16).reshape(len(rnds), COLS, -1)
    recs = dev.view("REC").reshape(COLS, -1)
    for c in range(COLS):
        h, k = c // nph, c % nph
        if rounds:  # (round h: the chain of its half of the heads, its last column)
            o_c = outs[h, nph * (k // 2) + nph - 1, :ob // 4].reshape(2 * qc, 32, S, 8)[qc * (k & 1):qc * ((k & 1) + 1)]
        else:
            o_c = outs[0, c - k + nph - 1].reshape(nph * qc, 32, S, 8)[qc * k:qc * (k + 1)] if split else \
                outs[0, c, :ob // (2 * nph)].reshape(qc, 32, 8 if one else S, 8)
        # the open records' rows: the ring's before p (rec: the panels' values), then the new ones
        rows = {r: (vc_s[h] if rec else vc[h])[8 * r:8 * r + 8].copy() for r in {rp, rl}}
        for s in range(S):
            rows[(p + s) // 8][(p + s) % 8] = kv[s, h, HD:]
        kn_all = []
        for s in range(S):
            q = p + s
            v_seen = vc_s[h][:q].copy()
            for r in sorted(rows):  # (an open record up to q: encoded from its rows)
                if 8 * r <= q:
                    v_seen = np.concatenate([v_seen[:8 * r], _enc(rows[r], min(q + 1 - 8 * r, 8))])[:q + 1]
            k_prev = np.concatenate([kc_s[h][:p], *[x[None] for x in kn_all]]) if kn_all else kc_s[h][:p]
            want, knew = ref.core_attention(qg[c, s], kv[s, h, :HD], v_seen[q], k_prev, v_seen[:q], q_norm=qn,
                                            k_norm=kn, cs=cs[s], eps=cfg.rms_norm_eps)
            kn_all.append(knew)
            got = from_bf16_bits(o_c[:, :, s].reshape(qc, HD))
            assert rel_max(got, want) < tol, (c, s, rel_max(got, want))
        recs_c = [rp, rl][:nt]
        if ash.role(c) == A1.ROLE_K:  # the K records of positions p .. p + S - 1 (as the cache holds them now)
            k_all = np.concatenate([kc_s[h][:p], np.stack(kn_all)])
            for i, r in enumerate(recs_c):
                kg = np.zeros((P // 32, 2, T), np.uint8)
                if one:  # (a record: into its pair)
                    b_ = recs[c, i * A1.REC_K:(i + 1) * A1.REC_K].reshape(A1.ND, 72)
                    kg[r // 4, 0, (r % 4) // 2 * 2 * A1.REC_K:][:2 * A1.REC_K].reshape(A1.ND, 2, 72)[:, r % 2] = b_
                else:  # (its pair)
                    kg[r // 4, 0, (r % 4) // 2 * 2 * A1.REC_K:][:2 * A1.REC_K] = recs[c, 2 * i * A1.REC_K:2 * (i + 1) * A1.REC_K]
                got_k = KV.kv_from_groups(kg, HD, dmajor=True)[0][8 * r:min(8 * r + 8, p + S)]
                assert rel_max(got_k, k_all[8 * r:min(8 * r + 8, p + S)]) < 1e-2, (c, r)
        elif ash.role(c) == A1.ROLE_V:  # per record its V panels up to the last new position and its rows
            for i, r in enumerate(recs_c):
                b_ = recs[c, i * (A1.REC_V + A1.ROWS_B):(i + 1) * (A1.REC_V + A1.ROWS_B)]
                vg = np.zeros((P // 32, 2, T), np.uint8)
                vg[r // 4, 1, (r % 4) * A1.REC_V:(r % 4 + 1) * A1.REC_V] = b_[:A1.REC_V]
                n = min(p + S - 8 * r, 8)
                assert np.array_equal(KV.kv_from_groups(vg, HD)[1][8 * r:8 * r + 8], _enc(rows[r], n)), (c, r)
                got_rows = ring_rows(b_[A1.REC_V:], HD)
                assert np.array_equal(got_rows[:n], rows[r][:n]), (c, r)


def interleave_bits(v: np.ndarray) -> np.ndarray:
    """bf16 values -> their bytes."""
    return np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32).__rshift__(16).astype(np.uint16).view(np.uint8)


# ----------------------------------------------------------------------
# Spark-X's spec: no q / k norms, a gate per head, sliding-window layers
# ----------------------------------------------------------------------

SPARK = A1.AttnSpec(norms=False, gate="head", window=512, branch=False)


class _SparkDesign(Attn256DecodeOperator):
    PREFIX = "spkt"


def _spark_tables(cfg: SparkConfig, kind: str, positions: list[int]) -> tuple[np.ndarray, np.ndarray, bytes]:
    """The slots' angles as the core takes them (a full layer's 32 pairs fp32, a sliding-window layer's 128 bf16) and
    the F_ATT payload's bytes: cos, sin (S, rot) of the values the core multiplies by."""
    th = cfg.rope_theta(positions, kind).astype(np.float64)
    c, s = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    if kind == "swa":
        c, s = bf16_round(c), bf16_round(s)
        payload = np.concatenate([c, s], 1).astype(bfloat16).tobytes()
    else:
        payload = np.concatenate([c, s], 1).astype(np.float32).tobytes()
    return c, s, payload


def _rot(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """NeoX pairs (i, i + rot) of x (..., 256) turned by cos c, sin s (rot,), fp64."""
    r = len(c)
    y = np.asarray(x, np.float64).copy()
    a, b = y[..., :r].copy(), y[..., r:2 * r].copy()
    y[..., :r], y[..., r:2 * r] = a * c - b * s, a * s + b * c
    return y


def _spark(size: str, kind: str, p: int, S: int, *, ring: int = 0, seed: int = 5, block: bool = False) -> dict:
    """One step of S slots at p of a Spark layer of ``kind`` on random inputs, one pass (``SPARK``): the cache of the
    positions before p (K as stored, V bf16) in groups, a ring of ``ring`` groups (0: linear; a sliding-window layer's
    window that crosses its end: two fills), streamed from the window's first group (``window_words``); per column the
    o of its heads' slots and its emitted records. ``block``: a drafter's block (every row the window before p and the
    step whole)."""
    path = spark_gguf(size)
    require(path)
    cfg = SparkConfig.from_gguf(GGUFFile(path))
    G = (p + S - 1) // 32 + 1
    P = 32 * G
    ash = A1.AttnShape(cfg, P, slots=8, spec=SPARK)
    op = _SparkDesign(attn=ash, hidden=cfg.hidden_size, rows={"Q4_K": 16}, eps=cfg.rms_norm_eps, slots=8, packed=True,
                      bcast=True)
    dev = OverlayDevice(op)
    rng = np.random.default_rng(seed)
    qc, HD, T, nph, kvh = ash.qc, A1.HD, op.tile_bytes, ash.nph, cfg.num_key_value_heads
    rp, rl = p // 8, (p + S - 1) // 8
    swa = kind == "swa"
    g0, wrec, wm, late = A1.window_words(p, cfg.sliding_window, block=block) if swa else (
        0, -1, np.zeros(4, np.uint32), np.zeros(2, np.uint32))
    P8 = 32 * ((p + 7) // 32 + 1)  # (drawn for 8 slots: a step of any S sees the same cache)
    kc = rng.standard_normal((kvh, P8, HD)).astype(np.float32)[:, :P]
    vc = bf16_round(rng.standard_normal((kvh, P8, HD)).astype(np.float32))[:, :P]
    kc[:, p:] = vc[:, p:] = 0
    groups = [KV.kv_groups(kc[h], vc[h], T, 1, dmajor=True) for h in range(kvh)]  # (G, 2, T) each
    kc_s = [KV.kv_from_groups(gr, HD, dmajor=True)[0] for gr in groups]
    vc_s = [KV.kv_from_groups(gr, HD, dmajor=True)[1] for gr in groups]
    NR = ring or G  # (groups of a head's cache: group g at g % NR)
    assert G - g0 <= NR
    hb, rb = NR * 2 * T, 2 * T
    dev.alloc("KV", kvh * (hb + rb) + T)
    kvv = dev.view("KV")
    for h in range(kvh):
        for g in range(g0, G):
            kvv[h * hb + (g % NR) * 2 * T:][:2 * T] = groups[h][g].reshape(-1)
        rows = vc[h][8 * rp:8 * rp + 8].reshape(8, A1.ND, 8).transpose(1, 0, 2)
        o = kvh * hb + h * rb + (rp & 1) * T
        kvv[o:o + A1.ROWS_B] = interleave_bits(rows.reshape(-1))
    dev.flush("KV")
    # (drawn for 8 slots: a slot's inputs are the same in steps of any S)
    q = bf16_round(rng.standard_normal((COLS, 8, qc, HD)).astype(np.float32))[:, :S]
    gate = bf16_round(rng.standard_normal((COLS, 8, qc)).astype(np.float32))[:, :S]
    kv = bf16_round(rng.standard_normal((8, kvh, 2 * HD)).astype(np.float32))[:S]  # [K | V] per KV head
    c_, s_, payload = _spark_tables(cfg, kind, list(range(p, p + S)))
    nt = ash.records(S)
    per = 9 * PR.XB  # (per column: F_ATT, then its 8 q objects)
    act = dev.alloc("ACT", COLS * per + S * PR.XB)
    word, amask, lanes = A1.step_words(p, S, block=block)
    ana = A1.OBJS * (G - g0) + nt
    for c in range(COLS):
        m = np.zeros(PR.META, np.int32)
        m[PR.M_FLAGS], m[PR.M_NS], m[PR.M_SKIP], m[PR.M_NREL] = A1.F_ATT, S, 2 * S, ash.n_rel(c, 0, S)
        m[A1.M_POS], m[A1.M_SPLIT], m[A1.M_NTAIL], m[A1.M_G0], m[A1.M_GATT], m[A1.M_GJ] = p, 0, nt, g0, 1 << 20, 0
        m[PR.M_NASKIP], m[A1.M_ANA] = 0 if c % nph else ana, ana
        m[A1.M_BRANCH], m[A1.M_ALANE] = word, lanes
        m[A1.M_AMASK:A1.M_AMASK + 4] = amask.view(np.int32)
        m[A1.M_KIND], m[A1.M_WREC] = A1.KIND_SWA if swa else A1.KIND_FULL, wrec
        m[A1.M_WMASK:A1.M_WMASK + 4] = wm.view(np.int32)
        m[A1.M_WLATE:A1.M_WLATE + 2] = late.view(np.int32)
        m[A1.M_WHOLE] = int(block)
        o = c * per
        act[o:o + 4 * PR.META] = m.view(np.uint8)
        act[o + 4 * PR.META:o + 4 * PR.META + len(payload)] = np.frombuffer(payload, np.uint8)
        for s in range(S):
            q0 = o + (1 + s) * PR.XB + 4 * PR.META
            act[q0:q0 + 2 * qc * HD] = interleave_bits(q[c, s].reshape(-1))
            act[q0 + 2 * qc * HD:q0 + 2 * qc * HD + 2 * qc] = interleave_bits(gate[c, s])
    for s in range(S):
        k0 = COLS * per + s * PR.XB + 4 * PR.META
        act[k0:k0 + 2 * kvh * 2 * HD] = interleave_bits(kv[s].reshape(-1))
    dev.flush("ACT")
    ob = qc * 32 * 8 * 16  # (o of a column's heads: [head][k block 32][slot 8][8] bf16)
    nr = 2 * (A1.REC_V + A1.ROWS_B)
    dev.alloc("OUT", COLS * ob)
    dev.alloc("REC", COLS * nr)
    seq = Sequence(op)
    for c in range(0, COLS, nph):  # (one pass: a chain's first column's stream reaches its cores)
        h = c // nph
        base, rg = dev.address("KV", h * hb), dev.address("KV", kvh * hb + h * rb)
        a, b = g0 % NR, (G - 1) % NR + 1
        if b > a:
            seq.fill("A", c, base + a * 2 * T, (b - a) * 2 * T)
        else:  # (the window wraps the ring)
            seq.fill("A", c, base + a * 2 * T, (NR - a) * 2 * T)
            seq.fill("A", c, base, b * 2 * T)
        for t in range(nt):
            seq.fill("A", c, rg + ((rl if t else rp) & 1) * T, T)
    for c in range(COLS):
        seq.fill("X", c, dev.address("ACT", c * per), PR.XB)
        seq.fill("X", c, dev.address("ACT", c * per + PR.XB), S * PR.XB)
        seq.fill("X", c, dev.address("ACT", COLS * per), S * PR.XB)
    for c in range(COLS):
        seq.drain("C", c, dev.address("OUT", c * ob), ob)
        role = ash.role(c)
        n = nt * {A1.ROLE_K: A1.REC_K, A1.ROLE_V: A1.REC_V + A1.ROWS_B, A1.ROLE_O: 0}[role]
        if n:
            seq.drain("C", c, dev.address("REC", c * nr), n)
    seq.sync(fills=True)
    dev.dispatch(Transaction(dev, seq.words()))
    dev.fetch("OUT")
    dev.fetch("REC")
    outs = dev.view("OUT").view(np.uint16).reshape(COLS, qc, 32, 8, 8)
    o = np.stack([from_bf16_bits(outs[:, :, :, s].reshape(COLS, qc, HD)) for s in range(S)], 1)  # (COLS, S, qc, 256)
    return {"cfg": cfg, "ash": ash, "o": o, "recs": dev.view("REC").reshape(COLS, -1).copy(), "q": q, "gate": gate,
            "kv": kv, "kc_s": kc_s, "vc_s": vc_s, "vc": vc, "cs": (c_, s_), "swa": swa, "block": block}


def _spark_ref(r: dict, p: int, S: int, c: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """fp64: o * sigmoid(gate) (qc, 256) of column c's heads at slot s (what the slot sees: the cache before p as stored,
    the step's open records encoded from their rows up to its position, the step's earlier slots' K as the core
    computed them; a block's: the window before p, the step's records and K whole), and the slot's K."""
    cfg, ash, HD = r["cfg"], r["ash"], A1.HD
    h = c // ash.nph
    cs = (r["cs"][0][s].astype(np.float64), r["cs"][1][s].astype(np.float64))
    pos = p + S - 1 if r["block"] else p + s
    lo = max(0, (p - cfg.sliding_window) if r["block"] else pos - cfg.sliding_window + 1) if r["swa"] else 0
    rows = {x: r["vc"][h][8 * x:8 * x + 8].copy() for x in {p // 8, (p + S - 1) // 8}}
    for t in range(S):
        rows[(p + t) // 8][(p + t) % 8] = r["kv"][t, h, HD:]
    v_seen = r["vc_s"][h][:pos].copy()
    for x in sorted(rows):
        if 8 * x <= pos:
            v_seen = np.concatenate([v_seen[:8 * x], _enc(rows[x], min(pos + 1 - 8 * x, 8))])[:pos + 1]
    knew = [_rot(r["kv"][t, h, :HD], r["cs"][0][t].astype(np.float64), r["cs"][1][t].astype(np.float64))
            for t in range(pos - p + 1)]
    K = np.concatenate([np.asarray(r["kc_s"][h][:p], np.float64), np.stack(knew)])[lo:pos + 1]
    V = np.asarray(v_seen, np.float64)[lo:pos + 1]
    qr = _rot(r["q"][c, s], *cs)
    sc = qr @ K.T / 16
    pr = np.exp(sc - sc.max(-1, keepdims=True))
    o = (pr @ V) / pr.sum(-1, keepdims=True)
    return o / (1 + np.exp(-np.asarray(r["gate"][c, s], np.float64)))[:, None], knew[s]


@pytest.mark.parametrize(("size", "kind", "p", "S", "ring", "block"),
                         [("4B", "swa", 0, 1, 0, False), ("4B", "swa", 512, 8, 0, False),
                          ("4B", "swa", 1000, 8, 0, False), ("4B", "full", 1001, 8, 0, False),
                          ("4B", "full", 30, 4, 0, False), ("1.7B", "full", 700, 5, 0, False),
                          ("4B", "swa", 1000, 8, 20, False), ("1.7B", "swa", 1500, 7, 18, False),
                          ("1.7B", "swa", 505, 8, 0, False),
                          ("4B", "swa", 300, 7, 0, True), ("4B", "swa", 543, 7, 20, True),
                          ("1.7B", "swa", 540, 7, 0, True)])
def test_spark_attention(size: str, kind: str, p: int, S: int, ring: int, block: bool) -> None:
    """Spark's attention (``SPARK``): per column and slot o within bfp16 / bf16 rounding of the fp64 attention over what the
    slot sees: a sliding-window layer's keys p + s - 511 .. p + s (its window's first group streamed first, the keys
    before a row's window masked: rows whose window starts in the second group too), a full layer's all; the rotation of
    the kind (128 bf16 pairs, 32 fp32); one gate per head; the relayed K records of the step's positions; a ring whose
    window crosses its end read in two fills; a step whose later slots' windows start past position 0 (505); a drafter's
    block (``block``: every row the keys p - 512 .. p + S - 1, windows from a group's last key (543: 31 + 512) and of a
    record past the group's first (540), the step across two records)."""
    r = _spark(size, kind, p, S, ring=ring, block=block)
    ash, HD = r["ash"], A1.HD
    for c in range(COLS):
        for s in range(S):
            want, _ = _spark_ref(r, p, S, c, s)
            assert rel_max(r["o"][c, s], want) < 3e-2, (c, s, rel_max(r["o"][c, s], want))
    rp, rl = p // 8, (p + S - 1) // 8
    for c in range(0, COLS, ash.nph):  # (a chain's first column: the K records of the step's first and last positions)
        h = c // ash.nph
        knew = np.stack([_spark_ref(r, p, S, c, s)[1] for s in range(S)])
        k_all = np.concatenate([r["kc_s"][h][:p], knew])
        for i, x in enumerate([rp, rl]):
            kg = np.zeros(((p + S - 1) // 32 + 1, 2, A1.OBJ_BYTES), np.uint8)
            b_ = r["recs"][c, i * A1.REC_K:(i + 1) * A1.REC_K].reshape(A1.ND, 72)
            kg[x // 4, 0, (x % 4) // 2 * 2 * A1.REC_K:][:2 * A1.REC_K].reshape(A1.ND, 2, 72)[:, x % 2] = b_
            got = KV.kv_from_groups(kg, HD, dmajor=True)[0][8 * x:min(8 * x + 8, p + S)]
            assert rel_max(got, k_all[8 * x:min(8 * x + 8, p + S)]) < 1e-2, (c, x)


@pytest.mark.parametrize(("size", "kind", "p"), [("4B", "swa", 1000), ("1.7B", "full", 700)])
def test_spark_slot_alone(size: str, kind: str, p: int) -> None:
    """A step's first slot gets the numbers of a step of one at its position (bit for bit): its row depends on no later
    slot's (later slots in its record or the next, a window's masks of rows past it)."""
    a, b = _spark(size, kind, p, 8), _spark(size, kind, p, 1)
    assert np.array_equal(a["o"][:, 0], b["o"][:, 0])
