"""The attention core of the gated attention of head_dim 256 (``decode.attn256``) on the NPU, in the qwen35 models'
decode design (``models.qwen35.decode``) on random inputs, against ``tests/qwen35/ref.py``'s ``core_attention``."""

import numpy as np
import pytest

from artlite.decode import attn256 as A1
from artlite.decode import kv as KV
from artlite.decode import protocol as PR
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
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


@pytest.mark.parametrize(("size", "p", "S", "split", "rec"), [("9B", 38, 2, False, False), ("27B", 39, 2, False, False),
                                                             ("27B", 0, 1, False, False), ("9B", 5, 1, False, True),
                                                             ("27B", 150, 2, True, True), ("9B", 151, 2, True, False),
                                                             ("1B", 38, 2, False, False), ("1B", 5, 1, False, True),
                                                             ("1B", 150, 2, True, True), ("1B", 1001, 2, True, False),
                                                             ("1B", 129, 1, True, False), ("35B-A3B", 38, 2, False, False),
                                                             ("35B-A3B", 150, 1, True, True),
                                                             ("35B-A3B", 1001, 1, True, False), ("35B-A3B", 1001, 2, True, True),
                                                             ("9B", 150, 4, True, True), ("9B", 1001, 3, True, False),
                                                             ("1B", 1001, 4, True, False), ("9B", 37, 4, False, False),
                                                             ("27B", 36, 4, False, True), ("35B-A3B", 768, 1, True, True),
                                                             ("35B-A3B", 768, 1, True, False), ("35B-A3B", 384, 2, True, False),
                                                             ("27B", 150, 4, True, False), ("27B", 1001, 3, True, True),
                                                             ("27B", 1000, 4, True, False), ("3.8-27B", 39, 2, False, False),
                                                             ("3.8-27B", 1001, 7, False, False), ("3.8-27B", 160, 8, False, True),
                                                             ("3.8-27B", 1000, 5, False, True), ("3.8-27B", 62, 5, False, False),
                                                             ("3.8-27B", 94, 8, False, True), ("3.8-27B", 30, 2, False, False),
                                                             ("35B-A3B", 1001, 4, 2, False), ("35B-A3B", 1001, 3, 2, True),
                                                             ("35B-A3B", 150, 4, 2, False), ("35B-A3B", 384, 3, 2, False)])
def test_attention(size: str, p: int, S: int, split: int, rec: bool) -> None:
    _attention(size, p, S, split, rec)


@pytest.mark.parametrize(("size", "p", "S", "split"), [("35B-A3B", 90, 2, False), ("35B-A3B", 150, 2, True)])
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
