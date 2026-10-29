"""The qwen35 multi-slot decode overlay (models/qwen35/decode.py on decode/design.py) on the NPU, phase by phase on real weights, for S
token slots per dispatch: the GEMV phases (normed input projections, output projections into the residual slices in
K groups, gate / up with silu(gate) up, down, the LM head) against the cores' numerics (``mmv.ref_rows``) and
llama.cpp's projections of the same inputs (golden tensors of a decode run: the slots take the last S tokens)."""

import os

import numpy as np
import pytest
import torch

from artlite.decode import design as OD
from artlite.decode import kv as KV
from artlite.decode import phases as mp
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.operators import kmath
from artlite.operators.bfp16 import block_bytes, block_values
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction
from models.qwen35 import decode as MD
from models.qwen35.image import QwenImage

from ..common import essay
from ..decode import mmv
from ..operators import kmath_sim
from . import ref

T = 6  # tokens of the golden decode run; slot s takes token T - S + s
NAMES = ("attn_norm|linear_attn_qkv_mixed|z|alpha|beta|final_output|attn_residual|ffn_swiglu|l_out|Qcur_full|Kcur|"
         "Vcur|attn_gated|model.input_embed")


class _Phases:
    """One phase per dispatch on the overlay's main and helper cores (inputs shared by the columns)."""

    def __init__(self, path: str, layers: list[int]):
        self.path = path
        self.g = GGUFFile(path)
        self.cfg = cfg = ref.QwenConfig.from_gguf(self.g)
        self.H = cfg.hidden_size
        self.img = img = QwenImage(self.g, cfg, layers=layers, vocab=COLS * 1024)
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        self.op = MD.QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in fmts}, tile_bytes=img.tile_bytes,
                                         eps=cfg.rms_norm_eps)
        self.dev = dev = OverlayDevice(self.op)
        img.pack(dev.alloc("W", img.nbytes), release=False)
        dev.flush("W")
        self.act = 4 << 20
        dev.alloc("ACT", self.act)
        dev.alloc("NORM", 2 * img.tile_bytes)
        self.ob = 1 << 16  # bytes of a column's outputs
        dev.alloc("OUT", COLS * self.ob)

    def norm_w(self, name: str) -> np.ndarray:
        return np.asarray(self.g.tensor(name).raw, np.float32).reshape(-1).copy()

    def run(self, region: object, x: np.ndarray, flags: int, *, norm: str | None = None,
            resid: np.ndarray | None = None, f32_ab: bool = False) -> np.ndarray:
        """x (S, K) bf16 values; resid (S, H) fp32. Returns the outputs (COLS, S, objects x C_LEN) as raw bf16 bits."""
        dev, op, S = self.dev, self.op, x.shape[0]
        groups = mp.groups(region, flags, S, f32_ab=f32_ab)
        act = dev.view("ACT")
        act[:] = 0
        h8 = self.H // COLS
        inputs, off = [], 0
        for gr in groups:  # per group its region [metadata object | x interleaved]
            act[off:off + 4 * OD.META] = gr.meta.view(np.uint8)
            xi = OD.interleave(np.asarray(x)[:, gr.k0 * 1024:gr.k0 * 1024 + gr.K])
            act[off + OD.XB:off + OD.region_bytes(gr.K)] = xi.view(np.uint8).reshape(-1)
            inputs.append((off, gr.K))
            off += -(-OD.region_bytes(gr.K) // 64) * 64
        loadr = off
        if resid is not None:  # per column and slot an F_LOADR object
            for c in range(COLS):
                for s in range(S):
                    o = loadr + (c * OD.SLOTS + s) * OD.XB
                    act[o:o + 4 * OD.META] = mp.loadr_meta(s).view(np.uint8)
                    act[o + 4 * OD.META:o + 4 * OD.META + 4 * h8] = \
                        np.asarray(resid[s, h8 * c:h8 * (c + 1)], np.float32).view(np.uint8)
        dev.flush("ACT")
        n_norm = 0
        if norm:
            nv = dev.view("NORM")
            nv[:] = 0
            objs = OD.norm_objects(self.norm_w(norm), op.tile_bytes)
            nv[:objs.nbytes] = objs.view(np.uint8).reshape(-1)
            n_norm = len(objs)
            dev.flush("NORM")
        n_obj = mmv.out_objects(region, flags, self.H, f32_ab=f32_ab)
        assert OD.SLOTS * n_obj * 2 * OD.C_LEN <= self.ob
        seq = Sequence(op, rolling=True)
        for c in range(COLS):  # drains [slot][objects]: object j of every slot, then j + 1
            seq.drain("C", c, dev.address("OUT", c * self.ob), S * n_obj * 2 * OD.C_LEN,
                      (OD.C_LEN // 2, S, n_obj * OD.C_LEN // 2, OD.C_LEN // 2))
        for c in range(COLS):
            if resid is not None:
                for s in range(S):
                    seq.fill("X", c, dev.address("ACT", loadr + (c * OD.SLOTS + s) * OD.XB), OD.XB)
            for i, gr in enumerate(groups):
                nbytes, dims = OD.input_fill(gr.K)
                seq.fill("X", c, dev.address("ACT", inputs[i][0]), nbytes, dims)
                if n_norm and i == 0:
                    seq.fill("A", c, dev.address("NORM"), n_norm * op.tile_bytes)
                for o, n, dims_, rep, it in mp.a_reads(region, c, gr):
                    seq.fill("A", c, dev.address("W", o), n, dims_, rep, it)
        seq.sync(fills=True)
        dev.dispatch(Transaction(dev, seq.words()))
        dev.fetch("OUT")
        out = dev.view("OUT").view(np.uint16).reshape(COLS, -1)
        return out[:, :S * n_obj * OD.C_LEN].reshape(COLS, S, -1).copy()


@pytest.fixture(scope="module", params=["9B", "1B"])
def setup(request: pytest.FixtureRequest) -> tuple[_Phases, ref.Golden, int, int]:
    path = ref.qwen_gguf("Q4_K_M", request.param)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    cfg = ref.QwenConfig.from_gguf(GGUFFile(path))
    Lg, La = cfg.gdn_layers[0], cfg.attention_layers[0]
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:T]
    G = ref.golden(path, tokens, ubatch=1, layers=("g", Lg, La - 1, La, cfg.num_hidden_layers - 1), names=NAMES,
                   logits="f32")
    return _Phases(path, [Lg, La]), G, Lg, La


def _t(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, np.float32)


def _bf16(u: np.ndarray) -> np.ndarray:
    """Raw bf16 bits -> fp32 values."""
    return (np.asarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _cols(region: object, values: np.ndarray, c: int, block: int = 0) -> np.ndarray:
    rows = region.blocks[block].rows[c]
    return _t(np.asarray(values, np.float32)[..., rows[rows >= 0]])


class _Check:
    """Every column and slot against the cores' numerics (bf16 rounding of the outputs; tol: max error / max value),
    all together against llama.cpp (its Q8_K activations: ~1-3%)."""

    def __init__(self, tol: float = 4e-3) -> None:
        self.tol = tol
        self.got: list[np.ndarray] = []
        self.gold: list[np.ndarray] = []

    def __call__(self, got: np.ndarray, kern: np.ndarray, gold: np.ndarray) -> None:
        assert got.shape == kern.shape, (got.shape, kern.shape)
        top = float(np.abs(kern).max())
        ulp = 2.0 ** (np.floor(np.log2(top)) - 7)  # (a rounding flip of the largest value: one bf16 ULP)
        err = float(np.abs(np.asarray(got, np.float64) - kern).max())
        assert err <= max(self.tol * top, ulp), ref.rel_err(got, kern)
        self.got.append(got.reshape(-1))
        self.gold.append(np.asarray(gold).reshape(-1))

    def done(self, tol: float = 5e-2) -> None:
        assert ref.rel_err(np.concatenate(self.got), np.concatenate(self.gold)) < tol


@pytest.mark.parametrize("S", [1, 4])
@pytest.mark.parametrize("phase", ["gdn_in", "gdn_out", "gate_up", "down", "attn_in", "attn_out", "lm"])
def test_phase(setup: tuple, phase: str, S: int) -> None:
    P, G, Lg, La = setup
    cfg, img, eps, H = P.cfg, P.img, P.cfg.rms_norm_eps, P.H
    t = np.arange(T - S, T)  # the slots' tokens
    h8 = H // COLS
    # (gate_up rounds the gate to bf16 before the SiLU: rows within the MMUL's ~1e-3 of a rounding midpoint flip, a
    # bf16 ULP of the gate moves the output by up to 2 of its ULPs)
    check = _Check(1e-2 if phase == "gate_up" else 4e-3)
    if phase in ("gdn_in", "attn_in", "lm"):
        L = {"gdn_in": Lg, "attn_in": La, "lm": -1}[phase]
        r = img.region(L, phase)
        x = {"gdn_in": _t(G["model.input_embed"][t]), "attn_in": _t(G["l_out", La - 1][t]),
             "lm": _t(G["l_out", cfg.num_hidden_layers - 1][t])}[phase]
        norm = "output_norm.weight" if L < 0 else f"blk.{L}.attn_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), eps)
        f32_ab = phase == "gdn_in"
        out = P.run(r, x, OD.F_NORM, norm=norm, f32_ab=f32_ab)
        if phase == "gdn_in":
            golds = [G["linear_attn_qkv_mixed", L][t], G["z", L][t],
                     np.concatenate([G["alpha", L][t], np.asarray(G["beta", L][t]).reshape(S, -1)], -1)]
        elif phase == "attn_in":
            golds = [G["Qcur_full", L][t], G["Kcur", L][t], G["Vcur", L][t]]
        else:
            golds = [G.logits[t]]
        for c in range(COLS):
            rows = mmv.ref_rows(img, r, c, xn)
            kern, got, k0 = [], [], 0
            for b, y in zip(r.blocks, rows):
                y = y * inv[:, None]
                n = y.shape[1]
                if f32_ab and b.name == "ab":  # (fp32: pairs of bf16 slots)
                    got.append(np.ascontiguousarray(out[c, :, k0:k0 + 2 * n]).view(np.float32))
                    kern.append(y.astype(np.float32))
                    k0 += 2 * n
                else:
                    got.append(_bf16(out[c, :, k0:k0 + n]))
                    kern.append(mmv.bf(y))
                    k0 += n
            gold = np.concatenate([_cols(r, np.asarray(v).reshape(S, -1), c, i) for i, v in enumerate(golds)], -1)
            check(np.concatenate(got, -1), np.concatenate(kern, -1), gold)
    elif phase in ("gdn_out", "attn_out", "down"):
        L = Lg if phase in ("gdn_out", "down") else La
        r = img.region(L, phase)
        x, resid, gold = {
            "gdn_out": (G["final_output", Lg][t], G["model.input_embed"][t], G["attn_residual", Lg][t]),
            "attn_out": (G["attn_gated", La][t], G["l_out", La - 1][t], G["attn_residual", La][t]),
            "down": (G["ffn_swiglu", Lg][t], G["attn_residual", Lg][t], G["l_out", Lg][t])}[phase]
        x, resid, gold = _t(x).reshape(S, -1), _t(resid).reshape(S, -1), _t(gold).reshape(S, -1)
        x = np.pad(x, ((0, 0), (0, r.K - x.shape[1])))  # (a K padded to whole chunks: zero inputs)
        out = P.run(r, x, OD.F_RESID | OD.F_EMITR, resid=resid)
        xb = mmv.bf(x)
        for c in range(COLS):
            kern = mmv.bf(resid[:, h8 * c:h8 * (c + 1)] + mmv.ref_rows(img, r, c, xb)[0])
            check(_bf16(out[c, :, :h8]), kern, gold[:, h8 * c:h8 * (c + 1)])
    else:  # gate_up
        r = img.region(Lg, phase)
        x = _t(G["attn_residual", Lg][t]).reshape(S, -1)
        norm = f"blk.{Lg}.post_attention_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), eps)
        out = P.run(r, x, OD.F_NORM | OD.F_GU, norm=norm)
        ffn = cfg.intermediate_size // COLS
        sw = _t(G["ffn_swiglu", Lg][t]).reshape(S, -1)
        for c in range(COLS):
            y = mmv.ref_rows(img, r, c, xn)[0].reshape(S, -1, 2, 8) * inv[:, None, None, None]
            kern = mmv.bf(kmath_sim.emulate(kmath.act(kmath.SILU), mmv.bf(y[:, :, 0])).astype(np.float64)
                          * y[:, :, 1]).reshape(S, -1)
            check(_bf16(out[c, :, :ffn]), kern, sw[:, ffn * c:ffn * (c + 1)])
    check.done()


@pytest.fixture(scope="module")
def setup27() -> tuple[_Phases, int, int]:
    path = ref.qwen_gguf("Q4_K_M", "27B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    cfg = ref.QwenConfig.from_gguf(GGUFFile(path))
    Lg, La = cfg.gdn_layers[0], cfg.attention_layers[0]
    return _Phases(path, [Lg, La]), Lg, La


@pytest.mark.parametrize("phase", ["gdn_in", "attn_in", "attn_out", "down"])
def test_phase_27b(setup27: tuple, phase: str) -> None:
    """The 27B's K groups (attn_out: 6 chunks as 3 + 3, down: 17 chunk-major as 5 + 5 + 5 + 2; gdn_in: Q6_K qkv) on
    random inputs, against the cores' numerics."""
    P, Lg, La = setup27
    S, H = 3, P.H
    h8 = H // COLS
    rng = np.random.default_rng(1)
    L = La if phase.startswith("attn") else Lg
    r = P.img.region(L, phase)
    x = mmv.bf(rng.standard_normal((S, r.n_kc * 1024)).astype(np.float32))
    if phase in ("gdn_in", "attn_in"):
        norm = f"blk.{L}.attn_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), P.cfg.rms_norm_eps)
        out = P.run(r, x, OD.F_NORM, norm=norm)
        for c in range(COLS):
            kern = np.concatenate([mmv.bf(y * inv[:, None]) for y in mmv.ref_rows(P.img, r, c, xn)], -1)
            got = _bf16(out[c, :, :kern.shape[1]])
            assert ref.rel_err(got, kern) < 4e-3, (c, ref.rel_err(got, kern))
        return
    resid = rng.standard_normal((S, H)).astype(np.float32)
    out = P.run(r, x, OD.F_RESID | OD.F_EMITR, resid=resid)
    for c in range(COLS):
        kern = mmv.bf(resid[:, h8 * c:h8 * (c + 1)] + mmv.ref_rows(P.img, r, c, x)[0])
        assert ref.rel_err(_bf16(out[c, :, :h8]), kern) < 4e-3, (c, ref.rel_err(_bf16(out[c, :, :h8]), kern))


# ----------------------------------------------------------------------
# The GDN core (models/qwen35/gdn.py)
# ----------------------------------------------------------------------


class _Gdn:
    """A GDN layer's step of S slots on all columns: its gdn_in phase to C (the exact message values), then with F_MSG
    to the GDN cores with their X objects (conv, K, deltas, state panels) and D drains."""

    def __init__(self, path: str, layer: int):
        from models.qwen35 import gdn as GD
        from models.qwen35.image import gdn_v_heads

        self.MG = GD
        self.g = g = GGUFFile(path)
        self.cfg = cfg = ref.QwenConfig.from_gguf(g)
        self.H, self.L = cfg.hidden_size, layer
        img = QwenImage(g, cfg, layers=[layer], vocab=COLS * 64)
        self.sh = sh = GD.GdnShape(cfg)
        self.op = op = MD.QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in {b.fmt for r in img.regions
                                                                                        for b in r.blocks}},
                                              tile_bytes=img.tile_bytes, eps=cfg.rms_norm_eps, gdn=sh)
        self.dev = dev = OverlayDevice(op)
        img.pack(dev.alloc("W", img.nbytes), release=False)
        dev.flush("W")
        self.r = r = img.region(layer, "gdn_in")
        dev.alloc("ACT", OD.region_bytes(self.H))
        nw = np.asarray(g.tensor(f"blk.{layer}.attn_norm.weight").raw, np.float32).reshape(-1)
        objs = OD.norm_objects(nw, img.tile_bytes)
        self.n_norm = len(objs)
        dev.alloc("NORM", objs.nbytes)[:] = objs.view(np.uint8).reshape(-1)
        dev.flush("NORM")
        self.n_c = mmv.out_objects(r, OD.F_NORM, self.H, f32_ab=True)
        self.ob = OD.SLOTS * self.n_c * 2 * OD.C_LEN
        dev.alloc("OUT", COLS * self.ob)
        self.npair = sh.hc // 2
        self.gcol = 256 + 4096 * (5 + 3 * self.npair)  # a column's GDN objects [header | payloads]
        gobj = dev.alloc("GOBJ", COLS * self.gcol)
        self.ps = sh.hc * 16 * GD.PANEL  # a column's state panels
        dev.alloc("GSTATE", 256 + COLS * self.ps)
        dev.alloc("GO", COLS * self.npair * GD.D_BYTES)
        blk = r.block("qkv")
        cw = np.asarray(g.tensor(f"blk.{layer}.ssm_conv1d.weight").raw, np.float32)
        dt = np.asarray(g.tensor(f"blk.{layer}.ssm_dt.bias").raw, np.float32).reshape(-1)
        A = np.asarray(g.tensor(f"blk.{layer}.ssm_a").raw, np.float32).reshape(-1)
        norm = np.asarray(g.tensor(f"blk.{layer}.ssm_norm.weight").raw, np.float32).reshape(-1)
        self.vh, self.params = [], []
        meta = np.zeros(OD.META, np.int32)
        meta[OD.M_FLAGS], meta[OD.M_SKIP] = GD.F_GDN, sh.n_objects - 1
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = blk.rows[c][blk.rows[c] >= 0]
            o = c * self.gcol
            gobj[o:o + 4 * OD.META] = meta.view(np.uint8)
            for b, pay in enumerate(GD.conv_payloads(sh, cw[rows], dt[vh], A[vh], norm)):
                k = b if b < 4 else 5 + 3 * ((b - 4) // 2) + (b - 4) % 2
                gobj[self.pay(c, k):self.pay(c, k) + sh.xpay] = pay
            self.vh.append(vh)
            self.params.append({"conv_w": cw[rows], "dt": dt[vh], "ssm_a": A[vh], "norm": norm})
        dev.flush("GOBJ")

    def pay(self, c: int, k: int) -> int:
        """Byte offset of column c's GDN object k's payload in GOBJ (k: q / k blocks 0 .. 3, K 4, pair p 5 + 3 p ..)."""
        return c * self.gcol + 256 + 4096 * k

    def _transaction(self, S: int, msg: bool) -> Transaction:
        dev, op, r = self.dev, self.op, self.r
        MG = self.MG
        groups = mp.groups(r, OD.F_NORM | (OD.F_MSG if msg else 0), S, f32_ab=True)
        assert len(groups) == 1
        seq = Sequence(op, rolling=True)
        for c in range(COLS):
            seq.fill("A", c, dev.address("NORM"), self.n_norm * op.tile_bytes)
            for o, n, dims, rep, it in mp.a_reads(r, c, groups[0]):
                seq.fill("A", c, dev.address("W", o), n, dims, rep, it)
        dims4 = (OD.XB // 8, 2, OD.XB // 8, OD.XPAY // 4)
        # (in rounds over the columns, each round's drains before the next round's fills: a fifth task on a channel
        # waits for its oldest, which must not need a later round)
        for c in range(COLS):
            nbytes, dims = OD.input_fill(self.H)
            seq.fill("X", c, dev.address("ACT"), nbytes, dims)
            if msg:
                seq.fill("X", c, dev.address("GOBJ", c * self.gcol), 5 * OD.XB, dims4)
        for c in range(COLS):
            if not msg:
                seq.drain("C", c, dev.address("OUT", c * self.ob), S * self.n_c * 2 * OD.C_LEN,
                          (OD.C_LEN // 2, S, self.n_c * OD.C_LEN // 2, OD.C_LEN // 2))
                continue
            seq.drain("D", c, dev.address("GOBJ", self.pay(c, 0) + 2048), 2 * MG.D_BYTES, (512, 2, 1024, 2048))
            seq.drain("D", c, dev.address("GOBJ", self.pay(c, 4)), MG.D_BYTES)
        for p in range(self.npair if msg else 0):
            for c in range(COLS):
                seq.fill("X", c, dev.address("GOBJ", self.pay(c, 5 + 3 * p) - 256), 3 * OD.XB, dims4)
                seq.fill("X", c, dev.address("GSTATE", c * self.ps + 32 * p * MG.PANEL), 32 * OD.XB, dims4)
            for c in range(COLS):
                seq.drain("D", c, dev.address("GOBJ", self.pay(c, 5 + 3 * p) + 2048), MG.D_BYTES, (512, 2, 1024, 1024))
                seq.drain("D", c, dev.address("GSTATE", 256 + c * self.ps + 32 * p * MG.PANEL), 32 * MG.PANEL)
                seq.drain("D", c, dev.address("GOBJ", self.pay(c, 7 + 3 * p)), MG.D_BYTES)
                seq.drain("D", c, dev.address("GO", (c * self.npair + p) * MG.D_BYTES), MG.D_BYTES)
        seq.sync(fills=True)
        return Transaction(dev, seq.words())

    def step(self, x: np.ndarray, *, nrep: int = 0, cshift: int = 0, commit: bool = False
             ) -> tuple[list[list[dict]], np.ndarray]:
        """x (S, hidden): the slots' gdn_in inputs. Returns (per column per slot the message as the GDN core got it,
        the gated outputs (S, v heads of the columns in column order, 128))."""
        dev, hc, ch, S = self.dev, self.sh.hc, self.sh.ch, x.shape[0]
        out = []
        for msg in (False, True):
            gr = mp.groups(self.r, OD.F_NORM | (OD.F_MSG if msg else 0), S, f32_ab=True)[0]
            gr.meta[MD.M_NREP], gr.meta[MD.M_CSHIFT], gr.meta[MD.M_COMMIT] = nrep, cshift, int(commit)
            act = dev.view("ACT")
            act[:] = 0
            act[:4 * OD.META] = gr.meta.view(np.uint8)
            act[OD.XB:] = OD.interleave(x).view(np.uint8).reshape(-1)
            dev.flush("ACT")
            dev.dispatch(self._transaction(S, msg))
            if not msg:
                dev.fetch("OUT")
                raw = dev.view("OUT").view(np.uint16).reshape(COLS, -1)[:, :S * self.n_c * OD.C_LEN].reshape(COLS, S, -1)
                msgs = []
                for c in range(COLS):
                    m = []
                    for s in range(S):
                        v = _bf16(raw[c, s])
                        ab = np.ascontiguousarray(raw[c, s, ch + 128 * hc:ch + 128 * hc + 4 * hc]).view(np.float32)
                        m.append({"x": v[:ch], "z": v[ch:ch + 128 * hc].reshape(hc, 128), "alpha": ab[:hc],
                                  "beta": ab[hc:]})
                    msgs.append(m)
        dev.fetch("GO")
        go = dev.view("GO").view(np.uint16).reshape(COLS, self.npair, -1)[:, :, :2 * 16 * OD.SLOTS * 8]
        for c in range(COLS):
            for p in range(self.npair):
                o = _bf16(go[c, p]).reshape(2, 16, OD.SLOTS, 8)[:, :, :S].transpose(2, 0, 1, 3).reshape(S, 2, 128)
                out.append(o)
        return msgs, np.stack(out, 1).reshape(S, -1, 128)

    def states(self) -> list[np.ndarray]:
        from models.qwen35 import gdn as G1

        self.dev.fetch("GSTATE")
        v = self.dev.view("GSTATE")[256:].reshape(COLS, -1)
        return [G1.state_from_panels(v[c].view(np.float32), self.sh.hc) for c in range(COLS)]


def _gdn_steps(path: str, Lg: int, x: np.ndarray, gold: np.ndarray | None) -> None:
    h = _Gdn(path, Lg)
    hc, eps = h.sh.hc, h.cfg.rms_norm_eps
    vh = np.concatenate(h.vh)
    st = [np.zeros((hc, 128, 128), np.float32) for _ in range(COLS)]
    cs = [np.zeros((3, h.sh.ch), np.float32) for _ in range(COLS)]

    def ref_tokens(msgs: list, n: int, st: list, cs: list) -> tuple[np.ndarray, list, list]:
        """The first n slots of a step in sequence from (st, cs): (outputs, states, conv states after them)."""
        outs = np.zeros((n, len(vh), 128), np.float32)
        st, cs = list(st), list(cs)
        for c in range(COLS):
            for s in range(n):
                o, cs[c], st[c] = ref.core_gdn_step(**msgs[c][s], conv_state=cs[c], state=st[c], eps=eps, **h.params[c])
                outs[s, hc * c:hc * (c + 1)] = o
        return outs, st, cs

    steps = (([0, 1, 2], 0, 0, False, 2), ([2, 3], 2, 2, False, 2), ([4], 2, 2, True, 1))
    for toks, nrep, cshift, commit, acc in steps:
        msgs, out = h.step(x[toks], nrep=nrep, cshift=cshift, commit=commit)
        want, st_after, _ = ref_tokens(msgs, len(toks), st, cs)
        assert ref.rel_err(out, want) < 1e-2, (toks, ref.rel_err(out, want))
        if gold is not None:
            g = np.asarray(gold[toks]).reshape(len(toks), -1, 128)[:, vh]
            assert ref.rel_err(out, g) < 2e-2, (toks, ref.rel_err(out, g))
        written = h.states()
        if commit:  # the state after the step's tokens
            exp = st_after
        else:  # the base: the previous step's accepted tokens replayed (= the state before this step's tokens)
            exp = st
        if np.any(exp[0]):
            assert max(ref.rel_err(written[c], exp[c]) for c in range(COLS)) < 2e-3, toks
        else:
            assert not any(np.any(w) for w in written), toks
        # the next step starts after this step's first acc tokens
        _, st, cs = ref_tokens(msgs, acc, st, cs)


def test_gdn_steps(setup: tuple) -> None:
    """Layer 0 (GDN) over tokens 0 .. 4 in three steps with the lazy commit: tokens 0 .. 2 (2 accepted), then 2, 3
    (replaying 0, 1 first; both accepted), then 4 committing (replaying 2, 3): every output within bf16 rounding of the
    cores' arithmetic in sequence (ref.core_gdn_step), the written states (the base after the replay, then
    after token 4) within 2e-3; the outputs within llama.cpp's noise of its final_output."""
    P, G_, Lg, _ = setup
    _gdn_steps(P.path, Lg, _t(G_["model.input_embed"][:5]), G_["final_output", Lg])


def test_gdn_steps_27b() -> None:
    """The same steps on the 27B's first GDN layer (6 v heads per column: 3 pairs) on random inputs."""
    path = ref.qwen_gguf("Q4_K_M", "27B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    cfg = ref.QwenConfig.from_gguf(GGUFFile(path))
    x = mmv.bf(np.random.default_rng(2).standard_normal((5, cfg.hidden_size)).astype(np.float32) * 0.02)
    _gdn_steps(path, cfg.gdn_layers[0], x, None)


# ----------------------------------------------------------------------
# The attention core (models/qwen35/attn.py)
# ----------------------------------------------------------------------


def _enc(rows: np.ndarray, n: int) -> np.ndarray:
    """A V record's values as its panels hold them: rows (8, 256) of its positions, those from n on zero, encoded as
    blocks over the positions."""
    from models.qwen35 import attn as A1

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
                                                             ("27B", 1000, 4, True, False)])
def test_attention(size: str, p: int, S: int, split: bool, rec: bool) -> None:
    """A sub-step of S slots at positions p .. p + S - 1 over a cache of p positions on random inputs: per column and
    slot o within bfp16 / bf16 rounding of ref.core_attention over what the slot sees (closed records as the cache
    holds them, the open ones encoded from their exact rows up to its position: the ring slot's rows before p, or with
    ``rec`` the V panels as a prefill left them, and the new ones); the relayed records (a chain's first column: the K
    records of the new positions; its last: their records' V panels up to the last one and rows). split: the chain of a
    KV head's columns (9B / 27B 2, 1B 4) divides the cache's groups, the partials merged west to east, the last column
    emitting o of the chain's heads (35B-A3B: chains of 4 columns of 2 heads, split sub-steps of up to two slots)."""
    from models.qwen35 import attn as A1

    path = ref.qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    cfg = ref.QwenConfig.from_gguf(GGUFFile(path))
    g = p // 32
    assert (p + S - 1) // 32 == g and A1.AttnShape.splits(g) == split
    P = 32 * (g + 1)  # (cache positions: whole groups)
    ash = A1.AttnShape(cfg, P)
    op = MD.QwenDecodeOperator(hidden=cfg.hidden_size, rows={"Q4_K": 16}, eps=cfg.rms_norm_eps, attn=ash)
    dev = OverlayDevice(op)
    rng = np.random.default_rng(3)
    qc, HD, T, nph = ash.qc, A1.HD, op.tile_bytes, ash.nph
    kvh = cfg.num_key_value_heads
    rp, rl = p // 8, (p + S - 1) // 8
    # the cache: positions 0 .. p - 1 of every KV head (K as stored: normed, rotated; V bf16), the rest of its groups
    # zero; per head its ring (record rp's rows in slot rp & 1)
    kc = rng.standard_normal((kvh, P, HD)).astype(np.float32)
    vc = mmv.bf(rng.standard_normal((kvh, P, HD)).astype(np.float32))
    kc[:, p:] = vc[:, p:] = 0
    groups = [KV.kv_groups(kc[h], vc[h], T) for h in range(kvh)]
    hb = groups[0].nbytes
    rb = 2 * T  # (a head's ring)
    kv_bytes = kvh * (hb + rb) + T
    dev.alloc("KV", kv_bytes)
    kvv = dev.view("KV")
    for h in range(kvh):
        kvv[h * hb:(h + 1) * hb] = groups[h].reshape(-1)
        rows = vc[h, 8 * rp:8 * rp + 8].reshape(8, A1.ND, 8).transpose(1, 0, 2)  # ([d-block][position][8])
        o = kvh * hb + h * rb + (rp & 1) * T
        kvv[o:o + A1.ROWS_B] = interleave_bits(rows.reshape(-1))
    dev.flush("KV")
    kc_s = [KV.kv_from_groups(gr, HD)[0] for gr in groups]  # (as stored)
    vc_s = [KV.kv_from_groups(gr, HD)[1] for gr in groups]
    qg = mmv.bf(rng.standard_normal((COLS, S, qc, 2 * HD)).astype(np.float32))  # [q | gate] per head
    kv = mmv.bf(rng.standard_normal((S, kvh, 2 * HD)).astype(np.float32))  # [K | V] per KV head
    qn = (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32)
    kn = (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32)
    cs = [A1.rope_cs(cfg, p + s) for s in range(S)]
    G = g + 1
    b = ash.bounds(g) if split else None  # (split: chain core k's attention groups b[k] .. b[k + 1] - 1)
    nt = A1.TAILS
    # X: per column [F_ATT | q objects SLOTS], then the raw K, V objects (shared)
    per = (1 + OD.SLOTS) * OD.XB
    act = dev.alloc("ACT", COLS * per + S * OD.XB)
    for c in range(COLS):
        k, role = c % nph, ash.role(c)
        tl = nt if not split or role == A1.ROLE_V else 0
        m = np.zeros(OD.META, np.int32)
        m[OD.M_FLAGS], m[OD.M_NS], m[OD.M_SKIP] = A1.F_ATT, S, (nph + 1 if split else 2) * S
        m[OD.M_NREL], m[MD.M_POS], m[MD.M_SPLIT] = ash.n_rel(c, split, S), p, int(split)
        m[MD.M_NTAIL], m[MD.M_ATREC] = tl, int(rec)
        if not split:  # (a chain's first column: then group g's K objects again)
            m[OD.M_NASKIP], m[MD.M_G0], m[MD.M_GATT] = 3 * G + tl + (2 if k == 0 else 0), 0, 1 << 20
        elif k == 0:  # (then group g's K objects)
            m[OD.M_NASKIP], m[MD.M_G0], m[MD.M_GATT], m[MD.M_GJ] = 3 * b[1] + 2, 0, b[1], g
        else:
            m[OD.M_NASKIP], m[MD.M_G0], m[MD.M_GATT] = 3 * (b[k + 1] - b[k]) + tl, b[k], 1 << 20
        o = c * per
        act[o:o + 4 * OD.META] = m.view(np.uint8)
        nrm = np.zeros(2 * HD + 64 * A1.SMAX, np.float32)
        nrm[:HD], nrm[HD:2 * HD] = qn, kn
        for s in range(S):
            nrm[2 * HD + 64 * s:2 * HD + 64 * s + 64] = np.concatenate(cs[s])
        act[o + 4 * OD.META:o + 4 * OD.META + nrm.nbytes] = nrm.view(np.uint8)
        for s in range(S):
            q0 = o + (1 + s) * OD.XB + 4 * OD.META
            act[q0:q0 + 2 * qc * 2 * HD] = interleave_bits(qg[c, s].reshape(-1))
    for s in range(S):
        k0 = COLS * per + s * OD.XB + 4 * OD.META
        act[k0:k0 + 2 * kvh * 2 * HD] = interleave_bits(kv[s].reshape(-1))
    dev.flush("ACT")
    ob = nph * qc * 32 * S * 16  # (o of a chain's heads)
    nr = 2 * (A1.REC_V + A1.ROWS_B)  # (the relayed records' bytes at most)
    dev.alloc("OUT", COLS * ob)
    dev.alloc("REC", COLS * nr)
    seq = Sequence(op, rolling=True)
    for c in range(COLS):
        k, role = c % nph, ash.role(c)
        base, ring = dev.address("KV", c // nph * hb), dev.address("KV", kvh * hb + c // nph * rb)
        if not split:
            seq.fill("A", c, base, 3 * G * T)
        else:
            seq.fill("A", c, base + 3 * b[k] * T, 3 * (b[k + 1] - b[k]) * T)
            if k == 0:
                seq.fill("A", c, base + 3 * g * T, 2 * T)
        if not split or role == A1.ROLE_V:  # (the tail objects)
            for t in range(nt):
                r = rl if t else rp
                v0 = base + 3 * (rp // 4) * T + 2 * T + rp % 4 * A1.REC_V
                seq.fill("A", c, v0 if rec and t == 0 else ring + (r & 1) * T, T)
        if not split and k == 0:
            seq.fill("A", c, base + 3 * g * T, 2 * T)
    for c in range(COLS):
        seq.fill("X", c, dev.address("ACT", c * per), OD.XB)
        if split:  # (the q objects of the chain's columns in order)
            for c1 in range(c - c % nph, c - c % nph + nph):
                seq.fill("X", c, dev.address("ACT", c1 * per + OD.XB), S * OD.XB)
        else:
            seq.fill("X", c, dev.address("ACT", c * per + OD.XB), S * OD.XB)
        seq.fill("X", c, dev.address("ACT", COLS * per), S * OD.XB)
    for c in range(COLS):
        role = ash.role(c)
        no = (ob if role == A1.ROLE_V else 0) if split else ob // nph
        if no:
            seq.drain("C", c, dev.address("OUT", c * ob), no)
        rb_c = {A1.ROLE_K: 2 * A1.REC_K, A1.ROLE_V: 2 * (A1.REC_V + A1.ROWS_B), A1.ROLE_O: 0}[role]
        if rb_c:
            seq.drain("C", c, dev.address("REC", c * nr), rb_c)
    seq.sync(fills=True)
    dev.dispatch(Transaction(dev, seq.words()))
    dev.fetch("OUT")
    dev.fetch("REC")
    outs = dev.view("OUT").view(np.uint16).reshape(COLS, -1)
    recs = dev.view("REC").reshape(COLS, -1)
    for c in range(COLS):
        h, k = c // nph, c % nph
        o_c = outs[c - k + nph - 1].reshape(nph * qc, 32, S, 8)[qc * k:qc * (k + 1)] if split else \
            outs[c, :ob // (2 * nph)].reshape(qc, 32, S, 8)
        # the open records' rows: the ring's (or the panels' in record mode) before p, then the new ones
        rows = {r: (vc_s[h] if rec else vc[h])[8 * r:8 * r + 8].copy() for r in {rp, rl}}
        for s in range(S):
            rows[(p + s) // 8][(p + s) % 8] = kv[s, h, HD:]
        kn_all = []
        for s in range(S):
            q = p + s
            v_seen = vc_s[h][:q].copy()
            for r in sorted(rows):  # (an open record up to q: encoded from its rows)
                if 8 * r <= q:
                    v_seen = np.concatenate([v_seen[:8 * r], _enc(rows[r], min(q - 8 * r + 1, 8))])[:q + 1]
            k_prev = np.concatenate([kc_s[h][:p], *[x[None] for x in kn_all]]) if kn_all else kc_s[h][:p]
            want, knew = ref.core_attention(qg[c, s], kv[s, h, :HD], v_seen[q], k_prev, v_seen[:q], q_norm=qn,
                                            k_norm=kn, cs=cs[s], eps=cfg.rms_norm_eps)
            kn_all.append(knew)
            got = _bf16(o_c[:, :, s].reshape(qc, HD))
            assert ref.rel_err(got, want) < 3e-2, (c, s, ref.rel_err(got, want))
        recs_c = [rp, rl]
        if ash.role(c) == A1.ROLE_K:  # the K records of positions p .. p + S - 1 (as the cache holds them now)
            k_all = np.concatenate([kc_s[h][:p], np.stack(kn_all)])
            for i, r in enumerate(recs_c):
                kg = np.zeros((P // 32, 3, T), np.uint8)
                kg[r // 4, (r % 4) // 2, (r % 2) * A1.REC_K:(r % 2 + 1) * A1.REC_K] = recs[c, i * A1.REC_K:(i + 1) * A1.REC_K]
                got_k = KV.kv_from_groups(kg, HD)[0][8 * r:min(8 * r + 8, p + S)]
                assert ref.rel_err(got_k, k_all[8 * r:min(8 * r + 8, p + S)]) < 1e-2, (c, r)
        elif ash.role(c) == A1.ROLE_V:  # per record its V panels up to the last new position and its rows
            for i, r in enumerate(recs_c):
                b_ = recs[c, i * (A1.REC_V + A1.ROWS_B):(i + 1) * (A1.REC_V + A1.ROWS_B)]
                vg = np.zeros((P // 32, 3, T), np.uint8)
                vg[r // 4, 2, (r % 4) * A1.REC_V:(r % 4 + 1) * A1.REC_V] = b_[:A1.REC_V]
                n = min(p + S - 8 * r, 8)
                assert np.array_equal(KV.kv_from_groups(vg, HD)[1][8 * r:8 * r + 8], _enc(rows[r], n)), (c, r)
                got_rows = ring_rows(b_[A1.REC_V:], HD)
                assert np.array_equal(got_rows[:n], rows[r][:n]), (c, r)


def interleave_bits(v: np.ndarray) -> np.ndarray:
    """bf16 values -> their bytes."""
    return np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32).__rshift__(16).astype(np.uint16).view(np.uint8)


# ----------------------------------------------------------------------
# The overlay (models/qwen35/overlay.py)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("size", ["9B", "1B", "35B-A3B", "occamy"])
@pytest.mark.parametrize("mode", ["s1", "s3", "lazy", "split"])
def test_overlay_slice(size: str, mode: str) -> None:
    """The first layers up to the first full-attention layer over 34 positions (a second cache group), by steps of one
    token (s1), three committed tokens (s3), or four tokens of which two are accepted (lazy: the next step replays
    them); over 104 positions by steps of two (split: from 96 positions on, the attention's chains divide their
    caches: 9B pairs, 1B chains of 4): the residual stream after every layer and accepted token within llama.cpp's noise
    (its Q8_K activations) of its decode. The 1B's noise is ~3x the 9B's: its layer 0 against float64 math on the same
    weights is off by 1.9% on average (6% at most; the NPU's: 0.8%). 35B-A3B (every layer an MoE): where a token's
    routing differs from llama.cpp's (a near tie of its 8th and 9th logits) its layer output moves by 10-40%; none here
    (0.5-0.9% on average, 3.4% at most). Occamy (its Q3_K experts in these layers) within the 35B-A3B's bounds: its
    routing flips move a layer's output by up to ~2% (``test_moe_slice``)."""
    from models.qwen35.overlay import QwenOverlay

    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    La = ref.QwenConfig.from_gguf(GGUFFile(path)).attention_layers[0]
    n, layers = 104 if mode == "split" else 34, list(range(La + 1))
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = ref.golden(path, tokens, ubatch=1, layers=("g", *layers), names="l_out|model.input_embed", logits="none")
    ov = QwenOverlay(GGUFFile(path), layers=layers, vocab=COLS * 64, max_positions=128)
    ov.load(release=False)
    errs = np.zeros((n, len(layers)))
    t = 0
    while t < n:
        S, commit, acc = {"s1": (1, True, 1), "s3": (min(3, n - t), True, min(3, n - t)),
                          "lazy": (min(4, n - t), False, min(2, n - t)), "split": (2, True, 2)}[mode]
        x = ov.gguf.rows("token_embd.weight", tokens[t:t + S])
        if t == 0:
            assert np.array_equal(x[0].numpy(), np.asarray(G_["model.input_embed"][0], np.float32))
        ov.step(x, t, logits=False, commit=commit)
        for s in range(acc):
            for i, L in enumerate(layers):
                got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1, slot=s).numpy()
                errs[t + s, i] = ref.rel_err(got, np.asarray(G_["l_out", L][t + s], np.float32))
        ov.accept(acc)
        t += acc
    assert any(split for key in ov._steps for _, _, split in key[3]) == (mode == "split")
    top, mean = {"9B": (4e-2, 1.5e-2), "1B": (1.2e-1, 4e-2), "35B-A3B": (5e-2, 1.5e-2), "occamy": (5e-2, 1.5e-2)}[size]
    assert errs.max() < top and errs.mean(0).max() < mean, (errs.max(), errs.mean(0))


@pytest.mark.parametrize("first", [0, 10])
def test_moe_slice(first: int) -> None:
    """Occamy's layers first .. first + 3 (0-3: Q3_K experts, 10-13: IQ3_XXS; 3 and 11 attention layers) over 16
    positions, fed llama.cpp's residual stream before them token by token: each layer's routing (the top 8 of the route
    object's logits) llama.cpp's wherever no pair of an expert in and one out of its top 8 lies within their logits'
    errors (the NPU's against llama.cpp's; a nearer tie flips on either side's rounding: here a few per slice), and the
    residual after every layer within llama.cpp's noise of its decode, until a flip in the token's layers so far (one
    moves the layer's output by up to ~2% and the next layers' by up to ~5%)."""
    from models.qwen35.overlay import QwenOverlay

    path = ref.size_gguf("occamy")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    layers, n = list(range(first, first + 4)), 16
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = ref.golden(path, tokens, ubatch=1, layers=(("g",) if first == 0 else (first - 1,)) + tuple(layers),
                    names="l_out|model.input_embed|ffn_moe_logits|ffn_moe_topk", logits="none")
    ov = QwenOverlay(GGUFFile(path), layers=layers, vocab=COLS * 64, max_positions=64)
    ov.load(release=False)
    E, k = ov.cfg.n_experts, ov.cfg.n_experts_used
    errs = np.zeros((n, len(layers)))
    clean = np.ones((n, len(layers)), bool)  # (no flip in the token's layers so far)
    checked = 0
    for t in range(n):
        x = G_["model.input_embed"][t] if first == 0 else G_["l_out", first - 1][t]
        ov.step(torch.from_numpy(np.asarray(x, np.float32).copy()), t, logits=False)
        for i, L in enumerate(layers):
            got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1).numpy()
            errs[t, i] = ref.rel_err(got, np.asarray(G_["l_out", L][t], np.float32))
            o = ov.xch[L] + 4 * OD.META  # (the route object's payload: [slot][expert] fp32)
            ov.dev.fetch("ACT", o, 4 * E)
            lg = ov.dev.view("ACT")[o:o + 4 * E].view(np.float32)
            want = np.asarray(G_["ffn_moe_logits", L][t], np.float32)
            top = np.asarray(G_["ffn_moe_topk", L][t])
            ins = np.isin(np.arange(E), top)
            e = np.abs(lg - want)
            flip = set(np.argsort(-lg)[:k].tolist()) != set(top.tolist())
            if (want[ins][:, None] - want[~ins][None, :] > e[ins][:, None] + e[~ins][None, :]).all():
                checked += 1
                assert not flip, (t, L)
            clean[t, i:] &= not flip
    mean = np.array([errs[clean[:, i], i].mean() for i in range(len(layers))])
    print(f"layers {layers}: mean error per layer {errs.mean(0)} (tokens without a flip so far: {mean}), max "
          f"{errs.max():.3f} ({errs[clean].max():.3f}); routing checked in {checked} of {n * len(layers)}, flips "
          f"{int((~clean[:, 1:] & clean[:, :-1]).sum() + (~clean[:, 0]).sum())}")
    assert errs[clean].max() < 4e-2 and mean.max() < 1.5e-2 and errs.max() < 1e-1, (errs.max(), mean)
    assert checked >= n * len(layers) // 3


@pytest.mark.parametrize("size", ["35B-A3B", "occamy"])
def test_lm_head(size: str) -> None:
    """The LM head alone (``logits_of``: the output norm, every vocabulary row) on llama.cpp's last residual of T
    tokens: the logits within llama.cpp's own error (its Q8_K activations) of its logits, the same argmax."""
    from models.qwen35.overlay import QwenOverlay

    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    L = ref.QwenConfig.from_gguf(g).num_hidden_layers - 1
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:T]
    G_ = ref.golden(path, tokens, ubatch=1, layers=(L,), names="l_out", logits="f32")
    ov = QwenOverlay(g, layers=[L], max_positions=64)
    ov.load(release=False)
    for t in range(T):
        got = ov.logits_of(torch.from_numpy(np.asarray(G_["l_out", L][t], np.float32).copy())).float().numpy()
        want = np.asarray(G_.logits[t], np.float32)
        assert got.shape == want.shape and int(got.argmax()) == int(want.argmax()), t
        assert ref.rel_err(got, want) < 2e-2, (t, ref.rel_err(got, want))


# ----------------------------------------------------------------------
# An MoE layer (decode/design.py's "MoE", qwen35/overlay.py)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("size", "L"), [("35B-A3B", 0), ("occamy", 0), ("occamy", 10)])
@pytest.mark.parametrize("S", [1, 2, 3, 4])
def test_moe_layer(size: str, L: int, S: int) -> None:
    """A layer L (GDN, then its MoE) of Qwen3.6-35B-A3B or Occamy (layer 0: Q3_K experts, 10: IQ3_XXS) as a slice of
    its own in one step of S slots: on the NPU's own input to the MoE (moe_in's x), the router's logits (the route
    object's payload) within 3e-5 of fp32 math (the norm weight's bf16 rounding corrected in the router's rows:
    ``image.QwenImage.pack``), the slots' top 8 the oracle's (``moe_route``); every column's control packets and expert
    object as the host's routing of the NPU's logits makes them (``moe.route``: the union dealt as whole experts and
    halves; the items' weights to 1e-5: the cores' fp32 to ~2^-16); each slot's layer output within 1% of the oracle's
    (``moe_ffn`` + ``shared_expert`` on x; Occamy's layer 10: 1.5%, its shared expert's output 2.6x the routed experts'
    and x small: 1.1% from the oracle, 0.8% from one with the cores' bf16 gate and h = silu(gate) up)."""
    from artlite.moe import route as MR
    from models.qwen35.overlay import CP_BYTES, QwenOverlay

    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    ov = QwenOverlay(g, layers=[L], vocab=COLS * 64, max_positions=128)
    ov.load(release=False)
    cfg, H = ov.cfg, ov.H
    x = g.rows("token_embd.weight", ref.llama_tokenize(path, essay("apple.txt"))[:S])
    ov.step(x, 0, logits=False)
    dev = ov.dev
    dev.fetch("ACT")
    act = dev.view("ACT")
    off = ov.inp[(L, "moe_in")][0].off + OD.XB
    xin = _bf16(act[off:off + 8 * H].view(np.uint16).reshape(H // 8, OD.SLOTS, 8)[:, :S].transpose(1, 0, 2).reshape(S, H))
    b = f"blk.{L}."
    xn = ref.rms_norm(xin, np.asarray(g.tensor(b + "post_attention_norm.weight").raw, np.float32).reshape(-1),
                      cfg.rms_norm_eps)
    W = np.asarray(g.tensor(b + "ffn_gate_inp.weight").raw, np.float32).reshape(cfg.n_experts, H)
    lg_ref = xn.astype(np.float64) @ W.T.astype(np.float64)
    o = ov.xch[L] + 4 * OD.META
    lg = act[o:o + 4096].view(np.float32).reshape(OD.SLOTS, -1)[:S]
    assert ref.rel_err(lg, lg_ref) < 3e-5, ref.rel_err(lg, lg_ref)
    ids, w, items = MR.route(lg, cfg.n_experts_used, COLS)
    rids, rw = ref.moe_route(lg_ref.astype(np.float32), cfg.n_experts_used)
    assert np.array_equal(ids, rids)
    gu = ov.image.region(L, "exp_gu")
    for c in range(COLS):
        cp = act[ov.cp[L] + c * (CP_BYTES + OD.XB):][:CP_BYTES].view(np.uint32)
        assert np.array_equal(cp[:MR.PACKET_WORDS], MR.packet_words(items[c], base=dev.address("W", gu.offset),
                                                                     stride=gu.col_stride)), c
        em = cp[MR.RBUF_META:MR.RBUF_META + OD.META].view(np.int32)
        dn = ov.image.region(L, "exp_down")
        nh = sum(h1 - h0 + 1 for _, h0, h1 in items[c])
        assert (em[OD.M_NITEM], em[OD.M_NT]) == (len(items[c]), nh * (gu.n_rt * gu.n_kc + dn.n_rt * dn.n_kc)), c
        for i, (e, h0, h1) in enumerate(items[c]):
            it = em[OD.M_ITEM + OD.ITEM_WORDS * i:][:OD.ITEM_WORDS]
            assert it[0] == e | h0 << 8 | h1 << 9
            assert np.abs(it[1:1 + S].view(np.float32) - MR.item_weights(ids, w, e)).max() < 1e-5, (c, i)
    ex = ref.experts(g, L)
    shared = ref.shared_expert(xn, *(g.tensor(b + t).dequantize().numpy() for t in
                                     ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")),
                               np.asarray(g.tensor(b + "ffn_gate_inp_shexp.weight").raw, np.float32).reshape(-1))
    want = xin + ref.moe_ffn(xn, rids, rw, ex) + shared
    for s in range(S):
        got = ov.hidden(-1, slot=s).numpy()
        assert ref.rel_err(got, want[s]) < (1.5e-2 if (size, L) == ("occamy", 10) else 1e-2), (s, ref.rel_err(got, want[s]))


# ----------------------------------------------------------------------
# Host layouts
# ----------------------------------------------------------------------


def test_state_panels() -> None:
    from models.qwen35 import gdn as G

    m = np.random.default_rng(0).standard_normal((3, 128, 128)).astype(np.float32)
    p = G.state_panels(m)
    assert p.shape == (48, 1024)
    # panel 16 h + jp, sub-panel ib: [jj][ii] = S[i = 8 ib + ii][j = 8 jp + jj] = m[h][j][i]
    assert p[16 * 2 + 5, 64 * 3 + 8 * 4 + 6] == m[2, 8 * 5 + 4, 8 * 3 + 6]
    assert np.array_equal(G.state_from_panels(p, 3), m)
