"""The qwen35 multi-slot decode overlay (models/qwen35/decode.py on decode/design.py) on the NPU, phase by phase on real weights, for S
token slots per dispatch: the GEMV phases (normed input projections, output projections into the residual slices in
K groups, gate / up with silu(gate) up, down, the LM head) against the cores' numerics (``mmv.ref_rows``) and
llama.cpp's projections of the same inputs (golden tensors of a decode run: the slots take the last S tokens)."""


import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.decode import phases as mp
from artlite.decode import protocol as PR
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_gguf
from artlite.operators import kmath
from artlite.operators.bfp16 import block_bytes, block_values
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction
from models.qwen35 import decode as MD
from models.qwen35 import words as W
from models.qwen35.image import QwenImage

from .. import np_ops
from ..common import essay
from ..decode import mmv
from ..goldens.text import Golden, golden
from ..llama_cpp import llama_tokenize
from ..model_files import require, size_gguf
from ..moe import ref as moe_ref
from ..numerics import bf16_round, from_bf16_bits, rel_max
from ..operators import kmath_sim
from . import ref

T = 6  # tokens of the golden decode run; slot s takes token T - S + s
NAMES = ("attn_norm|linear_attn_qkv_mixed|z|alpha|beta|final_output|attn_residual|ffn_swiglu|l_out|Qcur_full|Kcur|"
         "Vcur|attn_gated|model.input_embed")


class _Phases:
    """One phase per dispatch on the overlay's main and helper cores (inputs shared by the columns)."""

    def __init__(self, path: str, layers: list[int], slots: int = PR.SLOTS):
        self.path, self.slots = path, slots
        self.g = GGUFFile(path)
        self.cfg = cfg = QwenConfig.from_gguf(self.g)
        self.H = cfg.hidden_size
        self.img = img = QwenImage(self.g, cfg, layers=layers, vocab=COLS * 1024)
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        self.op = MD.QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in fmts}, tile_bytes=img.tile_bytes,
                                         eps=cfg.rms_norm_eps, slots=slots)
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
        dev, op, S, slots = self.dev, self.op, x.shape[0], self.slots
        groups = mp.groups(region, flags, S, f32_ab=f32_ab, slots=slots)
        sp = PR.packed_slots(S) if op.packed else slots  # (a packed design: the step's slots per k block)
        for gr in groups:
            gr.meta[PR.M_SP] = sp
        act = dev.view("ACT")
        act[:] = 0
        h8 = self.H // COLS
        inputs, off = [], 0
        for gr in groups:  # per group its region [metadata object | x interleaved]
            act[off:off + 4 * PR.META] = gr.meta.view(np.uint8)
            xi = PR.interleave(np.asarray(x)[:, gr.k0 * 1024:gr.k0 * 1024 + gr.K], sp)
            act[off + PR.XB:off + PR.XB + xi.nbytes] = xi.view(np.uint8).reshape(-1)
            inputs.append((off, gr.K))
            off += -(-PR.region_bytes(gr.K, slots) // 64) * 64
        loadr = off
        if resid is not None:  # per column and slot an F_LOADR object
            for c in range(COLS):
                for s in range(S):
                    o = loadr + (c * slots + s) * PR.XB
                    act[o:o + 4 * PR.META] = mp.loadr_meta(s).view(np.uint8)
                    act[o + 4 * PR.META:o + 4 * PR.META + 4 * h8] = \
                        np.asarray(resid[s, h8 * c:h8 * (c + 1)], np.float32).view(np.uint8)
        dev.flush("ACT")
        n_norm = 0
        if norm:
            nv = dev.view("NORM")
            nv[:] = 0
            objs = PR.norm_objects(self.norm_w(norm), op.tile_bytes)
            nv[:objs.nbytes] = objs.view(np.uint8).reshape(-1)
            n_norm = len(objs)
            dev.flush("NORM")
        n_obj = mmv.out_objects(region, flags, self.H, f32_ab=f32_ab)
        assert slots * n_obj * 2 * PR.C_LEN <= self.ob
        seq = Sequence(op)
        whole = slots == 8 and flags & PR.F_EMITR  # (the residual as the next input's x lies: [k block][slot sp][8])
        for c in range(COLS):  # drains [slot][objects]: object j of every slot, then j + 1
            if whole:
                seq.drain("C", c, dev.address("OUT", c * self.ob), sp * n_obj * 2 * PR.C_LEN)
            else:
                seq.drain("C", c, dev.address("OUT", c * self.ob), S * n_obj * 2 * PR.C_LEN,
                          (PR.C_LEN // 2, S, n_obj * PR.C_LEN // 2, PR.C_LEN // 2))
        for c in range(COLS):
            if resid is not None:
                for s in range(S):
                    seq.fill("X", c, dev.address("ACT", loadr + (c * slots + s) * PR.XB), PR.XB)
            for i, gr in enumerate(groups):
                nbytes, dims = PR.input_fill(gr.K, sp, slots)
                seq.fill("X", c, dev.address("ACT", inputs[i][0]), nbytes, dims)
                if n_norm and i == 0:
                    seq.fill("A", c, dev.address("NORM"), n_norm * op.tile_bytes)
                for o, n, dims_, rep, it in mp.a_reads(region, c, gr):
                    seq.fill("A", c, dev.address("W", o), n, dims_, rep, it)
        seq.sync(fills=True)
        dev.dispatch(Transaction(dev, seq.words()))
        dev.fetch("OUT")
        out = dev.view("OUT").view(np.uint16).reshape(COLS, -1)
        if whole:
            return out[:, :sp * n_obj * PR.C_LEN].reshape(COLS, -1, sp, 8)[:, :, :S].transpose(0, 2, 1, 3).reshape(COLS, S, -1).copy()
        return out[:, :S * n_obj * PR.C_LEN].reshape(COLS, S, -1).copy()


@pytest.fixture(scope="module", params=["9B", "1B"])
def setup(request: pytest.FixtureRequest) -> tuple[_Phases, Golden, int, int]:
    path = qwen_gguf("Q4_K_M", request.param)
    require(path)
    cfg = QwenConfig.from_gguf(GGUFFile(path))
    Lg, La = cfg.gdn_layers[0], cfg.attention_layers[0]
    tokens = llama_tokenize(path, essay("apple.txt"))[:T]
    G = golden(path, tokens, ubatch=1, layers=("g", Lg, La - 1, La, cfg.num_hidden_layers - 1), names=NAMES,
                   logits="f32")
    return _Phases(path, [Lg, La]), G, Lg, La


def _t(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, np.float32)


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
        assert err <= max(self.tol * top, ulp), rel_max(got, kern)
        self.got.append(got.reshape(-1))
        self.gold.append(np.asarray(gold).reshape(-1))

    def done(self, tol: float = 5e-2) -> None:
        assert rel_max(np.concatenate(self.got), np.concatenate(self.gold)) < tol


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
        out = P.run(r, x, PR.F_NORM, norm=norm, f32_ab=f32_ab)
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
                    got.append(from_bf16_bits(out[c, :, k0:k0 + n]))
                    kern.append(bf16_round(y))
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
        out = P.run(r, x, PR.F_RESID | PR.F_EMITR, resid=resid)
        xb = bf16_round(x)
        for c in range(COLS):
            kern = bf16_round(resid[:, h8 * c:h8 * (c + 1)] + mmv.ref_rows(img, r, c, xb)[0])
            check(from_bf16_bits(out[c, :, :h8]), kern, gold[:, h8 * c:h8 * (c + 1)])
    else:  # gate_up
        r = img.region(Lg, phase)
        x = _t(G["attn_residual", Lg][t]).reshape(S, -1)
        norm = f"blk.{Lg}.post_attention_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), eps)
        out = P.run(r, x, PR.F_NORM | PR.F_GU, norm=norm)
        ffn = cfg.intermediate_size // COLS
        sw = _t(G["ffn_swiglu", Lg][t]).reshape(S, -1)
        for c in range(COLS):
            y = mmv.ref_rows(img, r, c, xn)[0].reshape(S, -1, 2, 8) * inv[:, None, None, None]
            kern = bf16_round(kmath_sim.emulate(kmath.act(kmath.SILU), bf16_round(y[:, :, 0])).astype(np.float64)
                          * y[:, :, 1]).reshape(S, -1)
            check(from_bf16_bits(out[c, :, :ffn]), kern, sw[:, ffn * c:ffn * (c + 1)])
    check.done()


@pytest.fixture(scope="module", params=[4, 8])
def setup27(request: pytest.FixtureRequest) -> tuple[_Phases, int, int]:
    path = qwen_gguf("Q4_K_M", "27B")
    require(path)
    cfg = QwenConfig.from_gguf(GGUFFile(path))
    Lg, La = cfg.gdn_layers[0], cfg.attention_layers[0]
    return _Phases(path, [Lg, La], request.param), Lg, La


@pytest.mark.parametrize("S", [1, 3, 8])
@pytest.mark.parametrize("phase", ["gdn_in", "attn_in", "attn_out", "down", "gate_up"])
def test_phase_27b(setup27: tuple, phase: str, S: int) -> None:
    """The 27B's K groups (attn_out: 6 chunks as 3 + 3, down: 17 chunk-major as 5 + 5 + 5 + 2; gdn_in: Q6_K qkv) on
    random inputs, against the cores' numerics: on the 4-slot design and on the 8-slot one (x as single bfp16 rows,
    ``decode.gemv``; its inputs packed: 1, 4, 8 slots per k block)."""
    P, Lg, La = setup27
    H, slots = P.H, P.slots
    if S > slots:
        pytest.skip(f"a design of {slots} slots")
    h8 = H // COLS
    rng = np.random.default_rng(1)
    L = La if phase.startswith("attn") else Lg
    r = P.img.region(L, phase)
    x = bf16_round(rng.standard_normal((S, r.n_kc * 1024)).astype(np.float32))
    if phase in ("gdn_in", "attn_in"):
        norm = f"blk.{L}.attn_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), P.cfg.rms_norm_eps)
        out = P.run(r, x, PR.F_NORM, norm=norm)
        for c in range(COLS):
            kern = np.concatenate([bf16_round(y * inv[:, None]) for y in mmv.ref_rows(P.img, r, c, xn, slots)], -1)
            got = from_bf16_bits(out[c, :, :kern.shape[1]])
            assert rel_max(got, kern) < 4e-3, (c, rel_max(got, kern))
        return
    if phase == "gate_up":
        norm = f"blk.{L}.post_attention_norm.weight"
        xn, inv = mmv.norm_input(x, P.norm_w(norm), P.cfg.rms_norm_eps)
        out = P.run(r, x, PR.F_NORM | PR.F_GU, norm=norm)
        ffn = P.cfg.intermediate_size // COLS
        for c in range(COLS):
            y = mmv.ref_rows(P.img, r, c, xn, slots)[0].reshape(S, -1, 2, 8) * inv[:, None, None, None]
            kern = bf16_round(kmath_sim.emulate(kmath.act(kmath.SILU), bf16_round(y[:, :, 0])).astype(np.float64)
                          * y[:, :, 1]).reshape(S, -1)
            got = from_bf16_bits(out[c, :, :ffn])
            assert rel_max(got, kern) < 1e-2, (c, rel_max(got, kern))
        return
    resid = rng.standard_normal((S, H)).astype(np.float32)
    out = P.run(r, x, PR.F_RESID | PR.F_EMITR, resid=resid)
    for c in range(COLS):
        kern = bf16_round(resid[:, h8 * c:h8 * (c + 1)] + mmv.ref_rows(P.img, r, c, x, slots)[0])
        assert rel_max(from_bf16_bits(out[c, :, :h8]), kern) < 4e-3, (c, rel_max(from_bf16_bits(out[c, :, :h8]), kern))


# ----------------------------------------------------------------------
# The GDN core (models/qwen35/gdn.py)
# ----------------------------------------------------------------------


class _Gdn:
    """A GDN layer's step of S slots on all columns: its gdn_in phase to C (the exact message values), then with F_MSG
    to the GDN cores with their X objects (conv, K, deltas, state panels; the model's core: ``overlay.chunk_of``) and D
    drains."""

    def __init__(self, path: str, layer: int):
        from models.qwen35 import gdn as GD
        from models.qwen35.image import gdn_v_heads
        from models.qwen35.overlay import chunk_of

        self.MG = GD
        self.g = g = GGUFFile(path)
        self.cfg = cfg = QwenConfig.from_gguf(g)
        self.H, self.L = cfg.hidden_size, layer
        img = QwenImage(g, cfg, layers=[layer], vocab=COLS * 64)
        self.sh = sh = GD.GdnShape(cfg, chunk=chunk_of(cfg))
        self.op = op = MD.QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in {b.fmt for r in img.regions
                                                                                        for b in r.blocks}},
                                              tile_bytes=img.tile_bytes, eps=cfg.rms_norm_eps, gdn=sh)
        self.dev = dev = OverlayDevice(op)
        img.pack(dev.alloc("W", img.nbytes), release=False)
        dev.flush("W")
        self.r = r = img.region(layer, "gdn_in")
        dev.alloc("ACT", PR.region_bytes(self.H))
        nw = np.asarray(g.tensor(f"blk.{layer}.attn_norm.weight").raw, np.float32).reshape(-1)
        objs = PR.norm_objects(nw, img.tile_bytes)
        self.n_norm = len(objs)
        dev.alloc("NORM", objs.nbytes)[:] = objs.view(np.uint8).reshape(-1)
        dev.flush("NORM")
        self.n_c = mmv.out_objects(r, PR.F_NORM, self.H, f32_ab=True)
        self.ob = PR.SLOTS * self.n_c * 2 * PR.C_LEN
        dev.alloc("OUT", COLS * self.ob)
        self.npair = sh.hc // 2
        self.gcol = 256 + 4096 * (sh.n_slots if sh.chunk else 5 + 3 * self.npair)  # a column's GDN objects [header | payloads]
        gobj = dev.alloc("GOBJ", COLS * self.gcol)
        self.ps = sh.hc * 16 * GD.PANEL  # a column's state panels
        self.half = COLS * self.ps if sh.chunk else 0  # (a chunkwise core's: two halves, ``GdnCursor.half``)
        dev.alloc("GSTATE", 256 + COLS * self.ps + self.half)
        dev.alloc("GO", COLS * sh.hc * GD.D_BYTES // 2)  # (per v head 2 KB; a chunkwise core's: its 1 KB, 1 KB of padding)
        blk = r.block("qkv")
        cw = np.asarray(g.tensor(f"blk.{layer}.ssm_conv1d.weight").raw, np.float32)
        dt = np.asarray(g.tensor(f"blk.{layer}.ssm_dt.bias").raw, np.float32).reshape(-1)
        A = np.asarray(g.tensor(f"blk.{layer}.ssm_a").raw, np.float32).reshape(-1)
        norm = np.asarray(g.tensor(f"blk.{layer}.ssm_norm.weight").raw, np.float32).reshape(-1)
        self.vh, self.params = [], []
        meta = np.zeros(PR.META, np.int32)
        meta[PR.M_FLAGS], meta[PR.M_SKIP] = W.F_GDN, sh.n_objects - 1
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = blk.rows[c][blk.rows[c] >= 0]
            o = c * self.gcol
            gobj[o:o + 4 * PR.META] = meta.view(np.uint8)
            if sh.chunk:
                h_ = self.pay(c, 0) + GD.P_PRM
                gobj[h_:h_ + 640] = GD.head_payload(dt[vh], A[vh], norm)
                for b, w in enumerate(GD.conv_weights(cw[rows])):
                    gobj[self.pay(c, 0) + sh.weights_at(b):self.pay(c, 0) + sh.weights_at(b) + w.size] = w
            else:
                for b, pay in enumerate(GD.conv_payloads(sh, cw[rows], dt[vh], A[vh], norm)):
                    k = b if b < 4 else 5 + 3 * ((b - 4) // 2) + (b - 4) % 2
                    gobj[self.pay(c, k):self.pay(c, k) + sh.xpay] = pay
            self.vh.append(vh)
            self.params.append({"conv_w": cw[rows], "dt": dt[vh], "ssm_a": A[vh], "norm": norm})
        dev.flush("GOBJ")
        self.gobj0 = gobj.copy()

    def reset(self) -> None:
        """A new sequence: the objects as loaded (no conv history), zero states."""
        self.dev.view("GOBJ")[:] = self.gobj0
        self.dev.view("GSTATE")[:] = 0
        self.dev.flush("GOBJ")
        self.dev.flush("GSTATE")

    def pay(self, c: int, k: int) -> int:
        """Byte offset of column c's GDN object k's payload in GOBJ (k: q / k blocks 0 .. 3, K 4, pair p 5 + 3 p ..; a
        chunkwise core's: its payload slot)."""
        return c * self.gcol + 256 + 4096 * k

    def _chunk_units(self, seq: Sequence, h: int) -> None:
        """A chunkwise core's objects and drains (``gdn_steps.GdnTransfers``; its state from half h, into the other), its outputs
        into GO."""
        dev, MG, sh = self.dev, self.MG, self.sh
        dims4 = (PR.XB // 8, 2, PR.XB // 8, PR.XPAY // 4)
        units = [u for kh in range(2) for u in [("k", kh)] + [("v", h) for h in range(kh, sh.hc, 2)]]

        def fills(kind: str, i: int, c: int) -> None:
            if kind == "k":
                seq.fill("X", c, dev.address("GOBJ", self.pay(c, sh.kh_slot(i)) - 256), MG.KH_SPAN4 * PR.XB, dims4)
                return
            seq.fill("X", c, dev.address("GOBJ", self.pay(c, sh.v_slot(i)) - 256), MG.V_SPAN4 * PR.XB, dims4)
            seq.fill("X", c, dev.address("GSTATE", h * self.half + c * self.ps + 16 * i * MG.PANEL), 16 * PR.XB, dims4)

        def drains(kind: str, i: int, c: int) -> None:
            if kind == "k":
                seq.drain("D", c, dev.address("GOBJ", self.pay(c, sh.kh_slot(i))), 2 * MG.D_BYTES, (512, 2, 512, 2048))
                return
            seq.drain("D", c, dev.address("GSTATE", 256 + (1 - h) * self.half + c * self.ps + 16 * i * MG.PANEL), 16 * MG.PANEL)
            seq.drain("D", c, dev.address("GO", (c * sh.hc + i) * MG.D_BYTES // 2), MG.D_BYTES // 2)
            seq.drain("D", c, dev.address("GOBJ", self.pay(c, sh.v_slot(i)) + 2048), 3 * MG.D_BYTES // 2)

        for n in range(len(units) + 1):
            if n < len(units):
                for c in range(COLS):
                    fills(*units[n], c)
            if n:
                for c in range(COLS):
                    drains(*units[n - 1], c)

    def _transaction(self, S: int, msg: bool, h: int = 0) -> Transaction:
        dev, op, r = self.dev, self.op, self.r
        MG = self.MG
        groups = mp.groups(r, PR.F_NORM | (PR.F_MSG if msg else 0), S, f32_ab=True)
        assert len(groups) == 1
        seq = Sequence(op)
        for c in range(COLS):
            seq.fill("A", c, dev.address("NORM"), self.n_norm * op.tile_bytes)
            for o, n, dims, rep, it in mp.a_reads(r, c, groups[0]):
                seq.fill("A", c, dev.address("W", o), n, dims, rep, it)
        dims4 = (PR.XB // 8, 2, PR.XB // 8, PR.XPAY // 4)
        # (in rounds over the columns, each round's drains before the next round's fills: a fifth task on a channel
        # waits for its oldest, which must not need a later round)
        for c in range(COLS):
            nbytes, dims = PR.input_fill(self.H)
            seq.fill("X", c, dev.address("ACT"), nbytes, dims)
            if msg and not self.sh.chunk:
                seq.fill("X", c, dev.address("GOBJ", c * self.gcol), 5 * PR.XB, dims4)
        if msg and self.sh.chunk:
            self._chunk_units(seq, h)
            seq.sync(fills=True)
            return Transaction(dev, seq.words())
        for c in range(COLS):
            if not msg:
                seq.drain("C", c, dev.address("OUT", c * self.ob), S * self.n_c * 2 * PR.C_LEN,
                          (PR.C_LEN // 2, S, self.n_c * PR.C_LEN // 2, PR.C_LEN // 2))
                continue
            seq.drain("D", c, dev.address("GOBJ", self.pay(c, 0) + 2048), 2 * MG.D_BYTES, (512, 2, 1024, 2048))
            seq.drain("D", c, dev.address("GOBJ", self.pay(c, 4)), MG.D_BYTES)
        for p in range(self.npair if msg else 0):
            for c in range(COLS):
                seq.fill("X", c, dev.address("GOBJ", self.pay(c, 5 + 3 * p) - 256), 3 * PR.XB, dims4)
                seq.fill("X", c, dev.address("GSTATE", c * self.ps + 32 * p * MG.PANEL), 32 * PR.XB, dims4)
            for c in range(COLS):
                seq.drain("D", c, dev.address("GOBJ", self.pay(c, 5 + 3 * p) + 2048), MG.D_BYTES, (512, 2, 1024, 1024))
                seq.drain("D", c, dev.address("GSTATE", 256 + c * self.ps + 32 * p * MG.PANEL), 32 * MG.PANEL)
                seq.drain("D", c, dev.address("GOBJ", self.pay(c, 7 + 3 * p)), MG.D_BYTES)
                seq.drain("D", c, dev.address("GO", (c * self.npair + p) * MG.D_BYTES), MG.D_BYTES)
        seq.sync(fills=True)
        return Transaction(dev, seq.words())

    def step(self, x: np.ndarray, *, nrep: int = 0, cshift: int = 0, commit: bool = False, kshift: int = 0, half: int = 0
             ) -> tuple[list[list[dict]], np.ndarray]:
        """x (S, hidden): the slots' gdn_in inputs (the words: ``GdnTransfers.set_words``'s; a chunkwise core's state from
        ``half``). Returns (per column per slot the message as the GDN core got it, the gated outputs (S, v heads of the
        columns in column order, 128))."""
        dev, hc, ch, S = self.dev, self.sh.hc, self.sh.ch, x.shape[0]
        out = []
        for msg in (False, True):
            gr = mp.groups(self.r, PR.F_NORM | (PR.F_MSG if msg else 0), S, f32_ab=True)[0]
            gr.meta[W.M_NREP], gr.meta[W.M_CSHIFT], gr.meta[W.M_COMMIT] = nrep, cshift, int(commit)
            gr.meta[W.M_KSHIFT] = kshift
            act = dev.view("ACT")
            act[:] = 0
            act[:4 * PR.META] = gr.meta.view(np.uint8)
            act[PR.XB:] = PR.interleave(x).view(np.uint8).reshape(-1)
            dev.flush("ACT")
            dev.dispatch(self._transaction(S, msg, half))
            if not msg:
                dev.fetch("OUT")
                raw = dev.view("OUT").view(np.uint16).reshape(COLS, -1)[:, :S * self.n_c * PR.C_LEN].reshape(COLS, S, -1)
                msgs = []
                for c in range(COLS):
                    m = []
                    for s in range(S):
                        v = from_bf16_bits(raw[c, s])
                        ab = np.ascontiguousarray(raw[c, s, ch + 128 * hc:ch + 128 * hc + 4 * hc]).view(np.float32)
                        m.append({"x": v[:ch], "z": v[ch:ch + 128 * hc].reshape(hc, 128), "alpha": ab[:hc],
                                  "beta": ab[hc:]})
                    msgs.append(m)
        dev.fetch("GO")
        if self.sh.chunk:
            go = dev.view("GO").view(np.uint16).reshape(COLS, hc, -1)[:, :, :16 * PR.SLOTS * 8]
            for c in range(COLS):
                for h in range(hc):
                    out.append(from_bf16_bits(go[c, h]).reshape(16, PR.SLOTS, 8)[:, :S].transpose(1, 0, 2).reshape(S, 128))
            return msgs, np.stack(out, 1)
        go = dev.view("GO").view(np.uint16).reshape(COLS, self.npair, -1)[:, :, :2 * 16 * PR.SLOTS * 8]
        for c in range(COLS):
            for p in range(self.npair):
                o = from_bf16_bits(go[c, p]).reshape(2, 16, PR.SLOTS, 8)[:, :, :S].transpose(2, 0, 1, 3).reshape(S, 2, 128)
                out.append(o)
        return msgs, np.stack(out, 1).reshape(S, -1, 128)

    def states(self, half: int = 0) -> list[np.ndarray]:
        self.dev.fetch("GSTATE")
        v = self.dev.view("GSTATE")[256 + half * self.half:256 + half * self.half + COLS * self.ps].reshape(COLS, -1)
        return [ref.state_from_panels(v[c].view(np.float32), self.sh.hc) for c in range(COLS)]


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
        assert rel_max(out, want) < 1e-2, (toks, rel_max(out, want))
        if gold is not None:
            g = np.asarray(gold[toks]).reshape(len(toks), -1, 128)[:, vh]
            assert rel_max(out, g) < 2e-2, (toks, rel_max(out, g))
        written = h.states()
        if commit:  # the state after the step's tokens
            exp = st_after
        else:  # the base: the previous step's accepted tokens replayed (= the state before this step's tokens)
            exp = st
        if np.any(exp[0]):
            assert max(rel_max(written[c], exp[c]) for c in range(COLS)) < 2e-3, toks
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


class _Blocks:
    """The host's words of a chunkwise core of 4 (``GdnCursor``'s ``advance``, ``half``): blocks of 8 positions from 0."""

    def __init__(self):
        self.pos = self.cshift = self.kshift = 0

    def words(self) -> dict[str, int]:
        return {"nrep": self.pos % 8, "cshift": self.cshift, "kshift": self.kshift, "half": self.pos // 8 & 1}

    def accept(self, m: int) -> None:
        self.kshift = 8 if self.pos % 8 + m >= 8 else 0
        self.cshift = m
        self.pos += m


def _gdn_blocks(path: str, Lg: int, x: np.ndarray, junk: np.ndarray) -> None:
    """A chunkwise core of 4 (blocks of 8 positions, ``gdn``'s "Chunkwise") over the rows x in steps of one: each
    output within bf16 rounding of the cores' arithmetic in sequence (ref.core_gdn_step), the state at each block's end in
    the other half within 2e-3; then in steps of up to 4 slots whose drafts past the accepted ones are junk rows (steps
    inside a block, past its end with all or part of its lanes accepted, ending at its end): every accepted slot's
    outputs those of the steps of one, bit for bit."""
    h = _Gdn(path, Lg)
    hc, eps, N = h.sh.hc, h.cfg.rms_norm_eps, x.shape[0]
    st = [np.zeros((hc, 128, 128), np.float32) for _ in range(COLS)]
    cs = [np.zeros((3, h.sh.ch), np.float32) for _ in range(COLS)]
    w, ones = _Blocks(), []
    for p in range(N):
        msgs, out = h.step(x[p:p + 1], **w.words())
        want = np.zeros((1, COLS * hc, 128), np.float32)
        for c in range(COLS):
            o, cs[c], st[c] = ref.core_gdn_step(**msgs[c][0], conv_state=cs[c], state=st[c], eps=eps, **h.params[c])
            want[0, hc * c:hc * (c + 1)] = o
        assert rel_max(out, want) < 1e-2, (p, rel_max(out, want))
        ones.append(out[0].copy())
        if p % 8 == 7:  # (the block's end: its state in the other half)
            got = h.states(1 - w.words()["half"])
            assert max(rel_max(got[c], st[c]) for c in range(COLS)) < 2e-3, p
        w.accept(1)
    h.reset()
    w, j = _Blocks(), 0
    for S, m in ((4, 2), (4, 3), (4, 2), (4, 4), (3, 1), (4, 4), (1, 1), (4, 3)):
        p = w.pos
        rows = np.concatenate([x[p:p + m], junk[j:j + S - m]])
        j += S - m
        _, out = h.step(rows, **w.words())
        for s in range(m):
            assert np.array_equal(out[s].view(np.uint32), ones[p + s].view(np.uint32)), (p, S, m, s)
        w.accept(m)
    assert w.pos == N


def test_gdn_steps_a3b() -> None:
    """``_gdn_blocks`` on the 35B-A3B's first GDN layer (the chunkwise core of 4 slots, 4 v heads per column), random
    inputs."""
    path = size_gguf("35B-A3B")
    require(path)
    cfg = QwenConfig.from_gguf(GGUFFile(path))
    rng = np.random.default_rng(3)
    x = bf16_round(rng.standard_normal((20, cfg.hidden_size)).astype(np.float32) * 0.02)
    junk = bf16_round(rng.standard_normal((12, cfg.hidden_size)).astype(np.float32) * 0.02)
    _gdn_blocks(path, cfg.gdn_layers[0], x, junk)


def test_gdn_steps_27b() -> None:
    """The same steps on the 27B's first GDN layer (6 v heads per column: 3 pairs) on random inputs."""
    path = qwen_gguf("Q4_K_M", "27B")
    require(path)
    cfg = QwenConfig.from_gguf(GGUFFile(path))
    x = bf16_round(np.random.default_rng(2).standard_normal((5, cfg.hidden_size)).astype(np.float32) * 0.02)
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
    from models.qwen35 import attn as A1

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
            m[PR.M_FLAGS], m[PR.M_NS], m[PR.M_SKIP] = W.F_ATT, S, (3 if rounds else nph + 1 if split else 2) * S
            m[PR.M_NREL], m[W.M_POS], m[W.M_SPLIT] = ash.n_rel(c, sw, S), p, sw
            m[W.M_NTAIL] = tl
            if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                m[PR.M_NASKIP], m[W.M_G0], m[W.M_GATT] = o_k * G + tl + (2 if k == 0 and not one else 0), 0, 1 << 20
            elif k == 0:  # (then group g's K and V objects)
                m[PR.M_NASKIP], m[W.M_G0], m[W.M_GATT], m[W.M_GJ] = o_k * b[1] + 2, 0, b[1], g
            else:
                m[PR.M_NASKIP], m[W.M_G0], m[W.M_GATT] = o_k * (b[k + 1] - b[k]) + tl, b[k], 1 << 20
            if one:  # (on the first column's stream; the rows' masks and K lanes)
                m[W.M_ANA], m[PR.M_NASKIP] = m[PR.M_NASKIP], 0 if k else m[PR.M_NASKIP]
                m[W.M_BRANCH], w, m[W.M_ALANE] = A1.step_words(p, S)
                m[W.M_AMASK:W.M_AMASK + 4] = w.view(np.int32)
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
# The overlay (models/qwen35/overlay.py)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("size", ["9B", "1B", "35B-A3B", "occamy", "3.8-27B", "swift"])
@pytest.mark.parametrize("mode", ["s1", "s3", "lazy", "split", "split4", "s7", "lazy8"])
def test_overlay_slice(size: str, mode: str) -> None:
    """The first layers up to the first full-attention layer over 34 positions (a second cache group), by steps of one
    token (s1), three committed tokens (s3), or four tokens of which two are accepted (lazy: the next step replays
    them); over 104 positions by steps of two (split: from 96 positions on, the attention's chains divide their
    caches: 9B pairs, 1B chains of 4) or of four (split4: the 35B-A3B's rounds, a KV head each): the residual stream after every
    layer and accepted token within llama.cpp's noise
    (its Q8_K activations) of its decode. The 1B's noise is ~3x the 9B's: its layer 0 against float64 math on the same
    weights is off by 1.9% on average (6% at most; the NPU's: 0.8%). 35B-A3B (every layer an MoE): where a token's
    routing differs from llama.cpp's (a near tie of its 8th and 9th logits) its layer output moves by 10-40%; none here
    (0.5-0.9% on average, 3.4% at most). Occamy (its Q3_K experts in these layers) within the 35B-A3B's bounds: its
    routing flips move a layer's output by up to ~2% (``test_moe_slice``). Qwen3.8-27B on its design of 8 slots (x as
    single bfp16 rows, ``decode.gemv``) also by steps of seven committed tokens (s7) and of eight tokens of which three
    are accepted (lazy8)."""
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf(size)
    require(path)
    if mode in ("s7", "lazy8") and size not in ("3.8-27B", "swift"):
        pytest.skip("steps of more than 4 tokens: designs of 8 slots")
    La = QwenConfig.from_gguf(GGUFFile(path)).attention_layers[0]
    n, layers = 104 if mode.startswith("split") else 34, list(range(La + 1))
    tokens = llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = golden(path, tokens, ubatch=1, layers=("g", *layers), names="l_out|model.input_embed", logits="none")
    ov = QwenOverlay(GGUFFile(path), layers=layers, vocab=COLS * 64, max_positions=128)
    ov.load(release=False)
    errs = np.zeros((n, len(layers)))
    t = 0
    while t < n:
        S, commit, acc = {"s1": (1, True, 1), "s3": (min(3, n - t), True, min(3, n - t)),
                          "lazy": (min(4, n - t), False, min(2, n - t)), "split": (2, True, 2),
                          "split4": (min(4, n - t), True, min(4, n - t)),
                          "s7": (min(7, n - t), True, min(7, n - t)), "lazy8": (min(8, n - t), False, min(3, n - t))}[mode]
        x = ov.gguf.rows("token_embd.weight", tokens[t:t + S])
        if t == 0:
            assert np.array_equal(x[0], np.asarray(G_["model.input_embed"][0], np.float32))
        ov.step(x, t, logits=False, commit=commit)
        for s in range(acc):
            for i, L in enumerate(layers):
                got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1, slot=s)
                errs[t + s, i] = rel_max(got, np.asarray(G_["l_out", L][t + s], np.float32))
        ov.accept(acc)
        t += acc
    assert any(split for key in ov._steps for _, _, split in key[3]) == (mode.startswith("split") and not ov.ash.one)
    top, mean = {"9B": (4e-2, 1.5e-2), "1B": (1.2e-1, 4e-2), "35B-A3B": (5e-2, 1.5e-2), "occamy": (5e-2, 1.5e-2),
                 "3.8-27B": (5.6e-2, 2.1e-2), "swift": (5.6e-2, 2.1e-2)}[size]
    assert errs.max() < top and errs.mean(0).max() < mean, (errs.max(), errs.mean(0))


@pytest.mark.parametrize("size", ["1B", "9B", "35B-A3B", "occamy", "3.8-27B"])
def test_drafts_exact(size: str) -> None:
    """A step of S slots gives each kept slot the numbers of a step of one, bit for bit, so that drafts never change a
    greedy text: the first layers up to the first full-attention layer over 120 random tokens (the attention's chains
    divide the caches from 96 positions on), by steps of one and by a generation's rounds: a token and up to SLOTS - 1
    drafts, junk (rejected: the round keeps the token) or the next tokens (the round keeps some of them); every kept
    slot's logits (bf16 bits) those of the step of one."""
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf(size)
    require(path)
    g = GGUFFile(path)
    layers = list(range(QwenConfig.from_gguf(g).attention_layers[0] + 1))
    P = 120
    rng = np.random.default_rng(5)
    tokens = [int(t) for t in rng.integers(0, 50000, P)]

    def run(drafts: bool) -> dict[int, np.ndarray]:
        ov = QwenOverlay(g, layers=layers, vocab=COLS * 64, max_positions=P + 64)
        ov.load(release=False)
        out, t = {}, 0
        while t < P:
            S = int(rng.integers(1, ov.SLOTS + 1)) if drafts else 1
            junk = bool(rng.random() < 0.5)
            ids = [tokens[t], *([int(x) for x in rng.integers(0, 50000, S - 1)] if junk else tokens[t + 1:t + S])]
            lg = ov.step(g.rows("token_embd.weight", ids), t, commit=len(ids) == 1)
            m = 1 if junk or len(ids) == 1 else int(rng.integers(1, len(ids) + 1))
            for s in range(m):
                out[t + s] = np.asarray(lg[s]).view(np.uint16).copy()
            if len(ids) > 1:
                ov.accept(m)
            t += m
        ov.dev = None
        return out

    one, rounds = run(False), run(True)
    bad = [q for q in sorted(one) if not np.array_equal(one[q], rounds[q])]
    assert not bad, f"{len(bad)} of {P} positions differ: {bad[:8]}"


@pytest.mark.parametrize("first", [0, 10])
def test_moe_slice(first: int) -> None:
    """Occamy's layers first .. first + 3 (0-3: Q3_K experts, 10-13: IQ3_XXS; 3 and 11 attention layers) over 16
    positions, fed llama.cpp's residual stream before them token by token: each layer's routing (the top 8 of the route
    object's logits) llama.cpp's wherever no pair of an expert in and one out of its top 8 lies within their logits'
    errors (the NPU's against llama.cpp's; a nearer tie flips on either side's rounding: here a few per slice), and the
    residual after every layer within llama.cpp's noise of its decode, until a flip in the token's layers so far (one
    moves the layer's output by up to ~2% and the next layers' by up to ~5%)."""
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf("occamy")
    require(path)
    layers, n = list(range(first, first + 4)), 16
    tokens = llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = golden(path, tokens, ubatch=1, layers=(("g",) if first == 0 else (first - 1,)) + tuple(layers),
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
            got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1)
            errs[t, i] = rel_max(got, np.asarray(G_["l_out", L][t], np.float32))
            o = ov.xch[L] + 4 * PR.META  # (the route object's payload: [slot][expert] fp32)
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


@pytest.mark.parametrize("first", [16, 48])
def test_dense_slice(first: int) -> None:
    """Swift 1.5's layers first .. first + 3 (16-19: IQ3_XXS gate / up; 48-51: IQ3_S; IQ4_XS downs in both, 19 and 51
    attention layers) over 16 positions, fed llama.cpp's residual stream before them token by token: the residual after
    every layer within the 27B's bounds of ``test_overlay_slice``."""
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf("swift")
    require(path)
    layers, n = list(range(first, first + 4)), 16
    tokens = llama_tokenize(path, essay("apple.txt"))[:n]
    G_ = golden(path, tokens, ubatch=1, layers=(first - 1, *layers), names="l_out", logits="none")
    ov = QwenOverlay(GGUFFile(path), layers=layers, vocab=COLS * 64, max_positions=64)
    ov.load(release=False)
    errs = np.zeros((n, len(layers)))
    for t in range(n):
        ov.step(torch.from_numpy(np.asarray(G_["l_out", first - 1][t], np.float32).copy()), t, logits=False)
        for i, L in enumerate(layers):
            got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1)
            errs[t, i] = rel_max(got, np.asarray(G_["l_out", L][t], np.float32))
    print(f"layers {layers}: mean error per layer {errs.mean(0)}, max {errs.max():.4f}")
    assert errs.max() < 5.6e-2 and errs.mean(0).max() < 2.1e-2, (errs.max(), errs.mean(0))


@pytest.mark.parametrize("size", ["35B-A3B", "occamy", "swift"])
def test_lm_head(size: str) -> None:
    """The LM head alone (``logits_of``: the output norm, every vocabulary row) on llama.cpp's last residual of T
    tokens: the logits within llama.cpp's own error (its Q8_K activations) of its logits, the same argmax."""
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf(size)
    require(path)
    g = GGUFFile(path)
    L = QwenConfig.from_gguf(g).num_hidden_layers - 1
    tokens = llama_tokenize(path, essay("apple.txt"))[:T]
    G_ = golden(path, tokens, ubatch=1, layers=(L,), names="l_out", logits="f32")
    ov = QwenOverlay(g, layers=[L], max_positions=64)
    ov.load(release=False)
    for t in range(T):
        got = ov.logits_of(np.asarray(G_["l_out", L][t], np.float32)).astype(np.float32)
        want = np.asarray(G_.logits[t], np.float32)
        assert got.shape == want.shape and int(got.argmax()) == int(want.argmax()), t
        assert rel_max(got, want) < 2e-2, (t, rel_max(got, want))


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
    from models.qwen35.moe_steps import CP_BYTES
    from models.qwen35.overlay import QwenOverlay

    path = size_gguf(size)
    require(path)
    g = GGUFFile(path)
    ov = QwenOverlay(g, layers=[L], vocab=COLS * 64, max_positions=128)
    ov.load(release=False)
    cfg, H = ov.cfg, ov.H
    x = g.rows("token_embd.weight", llama_tokenize(path, essay("apple.txt"))[:S])
    ov.step(x, 0, logits=False)
    dev = ov.dev
    dev.fetch("ACT")
    act = dev.view("ACT")
    off = ov.inp[(L, "moe_in")][0].off + PR.XB
    xin = from_bf16_bits(act[off:off + 8 * H].view(np.uint16).reshape(H // 8, PR.SLOTS, 8)[:, :S].transpose(1, 0, 2).reshape(S, H))
    b = f"blk.{L}."
    xn = np_ops.rms_norm(xin, np.asarray(g.tensor(b + "post_attention_norm.weight").raw, np.float32).reshape(-1),
                      cfg.rms_norm_eps)
    W = np.asarray(g.tensor(b + "ffn_gate_inp.weight").raw, np.float32).reshape(cfg.n_experts, H)
    lg_ref = xn.astype(np.float64) @ W.T.astype(np.float64)
    o = ov.xch[L] + 4 * PR.META
    lg = act[o:o + 4096].view(np.float32).reshape(PR.SLOTS, -1)[:S]
    assert rel_max(lg, lg_ref) < 3e-5, rel_max(lg, lg_ref)
    ids, w, items = MR.route(lg, cfg.n_experts_used, COLS)
    rids, rw = moe_ref.moe_route(lg_ref.astype(np.float32), cfg.n_experts_used)
    assert np.array_equal(ids, rids)
    gu = ov.image.region(L, "exp_gu")
    for c in range(COLS):
        cp = act[ov.cp[L] + c * (CP_BYTES + PR.XB):][:CP_BYTES].view(np.uint32)
        assert np.array_equal(cp[:MR.PACKET_WORDS], MR.packet_words(items[c], base=dev.address("W", gu.offset), stride=gu.col_stride,
                                                                     a_packet=ov.op.PACKET.get("A"))), c
        em = cp[MR.RBUF_META:MR.RBUF_META + PR.META].view(np.int32)
        dn = ov.image.region(L, "exp_down")
        nh = sum(h1 - h0 + 1 for _, h0, h1 in items[c])
        assert (em[PR.M_NITEM], em[PR.M_NT]) == (len(items[c]), nh * (gu.n_rt * gu.n_kc + dn.n_rt * dn.n_kc)), c
        for i, (e, h0, h1) in enumerate(items[c]):
            it = em[PR.M_ITEM + PR.ITEM_WORDS * i:][:PR.ITEM_WORDS]
            assert it[0] == e | h0 << 8 | h1 << 9
            assert np.abs(it[1:1 + S].view(np.float32) - MR.item_weights(ids, w, e)).max() < 1e-5, (c, i)
    ex = moe_ref.experts(g, L)
    shared = moe_ref.shared_expert(xn, *(g.tensor(b + t).dequantize() for t in
                                     ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")),
                               np.asarray(g.tensor(b + "ffn_gate_inp_shexp.weight").raw, np.float32).reshape(-1))
    want = xin + moe_ref.moe_ffn(xn, rids, rw, ex) + shared
    for s in range(S):
        got = ov.hidden(-1, slot=s)
        assert rel_max(got, want[s]) < (1.5e-2 if (size, L) == ("occamy", 10) else 1e-2), (s, rel_max(got, want[s]))


# ----------------------------------------------------------------------
# Host layouts
# ----------------------------------------------------------------------


def test_state_panels() -> None:
    m = np.random.default_rng(0).standard_normal((3, 128, 128)).astype(np.float32)
    p = ref.state_panels(m)
    assert p.shape == (48, 1024)
    # panel 16 h + jp, sub-panel ib: [jj][ii] = S[i = 8 ib + ii][j = 8 jp + jj] = m[h][j][i]
    assert p[16 * 2 + 5, 64 * 3 + 8 * 4 + 6] == m[2, 8 * 5 + 4, 8 * 3 + 6]
    assert np.array_equal(ref.state_from_panels(p, 3), m)
