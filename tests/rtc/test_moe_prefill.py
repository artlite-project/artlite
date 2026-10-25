"""An MoE layer's prefill operators (operators/moe: the router, the shared expert, the experts' work lists, the
combine) on the decode images (``qwen35.image.QwenImage``, layers packed in host memory, each GEMM's W argument copied
to the device) of Qwen3.6-35B-A3B Q4_K_M (layers 3, 5: Q6_K / Q4_K downs) and Occamy 1.0 APEX (layers 3, 13: Q3_K /
IQ3_XXS experts, a Q5_K shared expert), against fp64 models on the operators' own inputs."""

import importlib.metadata
import os
import pathlib
import platform
import subprocess

import numpy as np
import pytest
import torch

import artlite
from artlite.gguf_model import GGUFFile
from artlite.model_config import occamy_gguf
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.moe import gemm as MG
from artlite.operators.prefill.norm import PrefillNormOperator

from . import qwen35_ref as ref

# (model, layer): the layers each image holds
MODELS = {"35B-A3B": (lambda: ref.qwen_gguf("Q4_K_M", "35B-A3B"), (3, 5)), "occamy": (occamy_gguf, (3, 13))}
LAYERS = [(m, L) for m, (_, ls) in MODELS.items() for L in ls]
_state: dict[str, object] = {}


def _image(model: str = "35B-A3B") -> tuple[GGUFFile, object, np.ndarray]:
    """A model's image of its two layers, packed (cached per module)."""
    path, layers = MODELS[model][0](), MODELS[model][1]
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    if _state.get("model") != model:
        from artlite.operators.qwen35.image import QwenImage

        g = GGUFFile(path)
        img = QwenImage(g, layers=list(layers), vocab=COLS * 64)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W, release=False)
        _state.update(model=model, g=g, img=img, W=W)
    return _state["g"], _state["img"], _state["W"]


def _w_arg(op: object, W: np.ndarray, w_off: int, dev: object) -> object:
    """A GEMM's W argument from the image (zeros outside it)."""
    w = np.zeros(op.w_bytes, np.uint8)
    a, b = max(w_off, 0), min(w_off + op.w_bytes, W.size)
    w[a - w_off:b - w_off] = W[a:b]
    return artlite.from_torch(torch.from_numpy(w).view(torch.bfloat16), dev)


def _f32(g: GGUFFile, name: str) -> np.ndarray:
    t = g.tensor(name)
    return np.asarray(t.dequantize().numpy(), np.float64).reshape(t.rows, t.cols)


@pytest.mark.parametrize(("model", "L"), LAYERS)
def test_router(model: str, L: int) -> None:
    """The post-attention norm with lo panels (``PrefillNormOperator(lo=True)``), then the router GEMM
    (``moe.gemm.router``: BFP16 hi / lo rows against x hi + lo, fp32 pairs): the logits within 5e-5 of fp64 math on the
    bf16 residual (rms, the fp32 norm weight, the F32 rows; x hi + lo carries ~2^-16 of each block's maximum: 2.6e-5, the
    logits 3.1-4.2e-5), the shared gate's within 8e-5 (one row: 4.6-5.8e-5); the top 8 the oracle's where the 8th / 9th
    gap exceeds 1e-3."""
    g, img, W = _image(model)
    cfg = img.cfg
    T, H, E = 256, cfg.hidden_size, cfg.n_experts
    region = img.region(L, "moe_in")
    gm = MG.router(img, region, T)
    norm = PrefillNormOperator(T=T, E=H, eps=cfg.rms_norm_eps, lo=True)
    norm.compile(), gm.op.compile()
    dev = norm.device
    rng = np.random.default_rng(L)
    x = torch.from_numpy((rng.standard_normal((T, H)) * 2).astype(np.float32)).bfloat16()
    d = torch.zeros(T, H).bfloat16()
    wn = np.asarray(g.tensor(f"blk.{L}.post_attention_norm.weight").dequantize().numpy(), np.float64).reshape(-1)
    xb = artlite.from_torch(x.clone().flatten(), dev)
    xn = artlite.empty((2 * T * H * 9 // 16,), dtype=torch.bfloat16, device=dev)
    norm.dispatch(xb, artlite.from_torch(d.flatten(), dev), artlite.from_torch(torch.from_numpy(wn).float().bfloat16(), dev),
                  xb, xn)
    C = artlite.empty((T, gm.op.N), dtype=torch.bfloat16, device=dev)
    gm.op.dispatch(xn, _w_arg(gm.op, W, gm.w_off, dev), C)
    C.sync_from_device()
    c = C.to_torch().view(torch.float32).numpy().reshape(T, COLS, MG.router_pitch(region))
    got = c[:, :, :E // COLS].reshape(T, E)
    got_sg = c[:, :, router_gate(region)]
    xd = x.double().numpy()
    xr = xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + cfg.rms_norm_eps) * wn
    want = xr @ _f32(g, f"blk.{L}.ffn_gate_inp.weight").T
    want_sg = xr @ _f32(g, f"blk.{L}.ffn_gate_inp_shexp.weight")[0]
    err, err_sg = ref.rel_err(got, want), ref.rel_err(got_sg, np.repeat(want_sg[:, None], COLS, 1))
    print(f"layer {L}: logits rel-rms {err:.2e}, shared gate {err_sg:.2e}")
    assert err < 5e-5 and err_sg < 8e-5
    ids, _ = ref.moe_route(got.astype(np.float32), cfg.n_experts_used)
    rids, _ = ref.moe_route(want.astype(np.float32), cfg.n_experts_used)
    srt = np.sort(want, 1)[:, ::-1]
    clear = srt[:, cfg.n_experts_used - 1] - srt[:, cfg.n_experts_used] > 1e-3
    assert (np.sort(ids, 1) == np.sort(rids, 1)).all(1)[clear].all()


def router_gate(region: object) -> int:
    """The shared gate's logit in a column's router output (the second pass's first pair)."""
    return MG.PASS // 2


def _normed(model: str, L: int, T: int) -> tuple[object, object, np.ndarray, np.ndarray]:
    """The post-attention norm (lo panels) of random bf16 x on the device: (its device, xn [hi | lo] panels, x fp64,
    x / rms(x) w fp64)."""
    g, img, _ = _image(model)
    cfg = img.cfg
    H = cfg.hidden_size
    norm = PrefillNormOperator(T=T, E=H, eps=cfg.rms_norm_eps, lo=True).compile()
    dev = norm.device
    rng = np.random.default_rng(100 + L)
    x = torch.from_numpy((rng.standard_normal((T, H)) * 2).astype(np.float32)).bfloat16()
    wn = np.asarray(g.tensor(f"blk.{L}.post_attention_norm.weight").dequantize().numpy(), np.float64).reshape(-1)
    xb = artlite.from_torch(x.clone().flatten(), dev)
    xn = artlite.empty((2 * T * H * 9 // 16,), dtype=torch.bfloat16, device=dev)
    norm.dispatch(xb, artlite.from_torch(torch.zeros(T * H).bfloat16(), dev),
                  artlite.from_torch(torch.from_numpy(wn).float().bfloat16(), dev), xb, xn)
    xd = x.double().numpy()
    return dev, xn, xd, xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + cfg.rms_norm_eps) * wn


SHARED = [("35B-A3B", 3, 256), ("35B-A3B", 5, 256), ("35B-A3B", 3, 128), ("35B-A3B", 5, 64), ("occamy", 3, 128),
          ("occamy", 13, 64)]


@pytest.mark.parametrize(("model", "L", "T"), SHARED)
def test_shared_gate_up(model: str, L: int, T: int) -> None:
    """The shared expert's gate / up (``moe.gemm.shared_gate_up``: Q4_K, Occamy's Q5_K planar SiLU groups) on the
    norm's hi panels: silu(gate) up within 2.5e-2 rms of fp64 on those panels' values (the bfp16 weights, the bf16 C tile
    over 32 k-tiles, the bfp16 output panels: ~2.0e-2 in both formats), its largest error within 4.5e-2 of the largest
    value (Occamy's layer 3: 4.1e-2 at that value itself)."""
    g, img, W = _image(model)
    cfg = img.cfg
    H, F = cfg.hidden_size, cfg.shared_ffn
    gm = MG.shared_gate_up(img, img.region(L, "moe_in"), T, width=F)
    gm.op.compile()
    dev, xn, _, _ = _normed(model, L, T)
    out = artlite.empty((T * F * 9 // 16,), dtype=torch.bfloat16, device=dev)
    gm.op.dispatch(xn, _w_arg(gm.op, W, gm.w_off, dev), out)
    out.sync_from_device()
    got = bfp16.from_panels(out.to_torch(), T, F).double().numpy()
    xn.sync_from_device()
    xh = bfp16.from_panels(xn.to_torch(), T, 2 * H).double().numpy()[:, :H]
    b = f"blk.{L}."
    gate, up = xh @ _f32(g, b + "ffn_gate_shexp.weight").T, xh @ _f32(g, b + "ffn_up_shexp.weight").T
    want = gate / (1 + np.exp(-gate)) * up
    err, rms = ref.rel_err(got, want), np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean())
    print(f"layer {L} T {T}: shared silu(gate) up err {err:.2e}, rms {rms:.2e}")
    assert err < 4.5e-2 and rms < 2.5e-2


@pytest.mark.parametrize(("model", "L", "T"), SHARED)
def test_shared_down(model: str, L: int, T: int) -> None:
    """The shared expert's down (``moe.gemm.row_group_gemm``: row groups of 2, GemmQd; Q6_K / Q4_K in the 35B's layers
    3 / 5, Occamy's Q5_K of 13 rows) on random bfp16 panels of the shared width: every column's outputs at its pitch
    within 1.7e-2 of the exact product on the panels' values (the largest error over the largest value: ~0.9e-2 for
    Q6_K, 1.4-1.5e-2 for Q4_K)."""
    g, img, W = _image(model)
    cfg = img.cfg
    H, F = cfg.hidden_size, cfg.shared_ffn
    region = img.region(L, "sdown")
    pitch = MG.shared_pitch(img, [L])
    gm = MG.row_group_gemm(region, T, pitch=pitch)
    gm.op.compile()
    dev = gm.op.device
    A = (torch.randn(T, F, generator=torch.Generator().manual_seed(L)) * 0.5).bfloat16()
    pan = bfp16.to_panels(A)
    Aq = bfp16.from_panels(pan, T, F).double().numpy()
    C = artlite.from_torch(torch.zeros(gm.op.M, gm.op.N, dtype=torch.bfloat16), dev)
    gm.op.dispatch(artlite.from_torch(torch.cat([pan, torch.zeros(gm.op.M * F * 9 // 16 - pan.numel(),
                                                                   dtype=pan.dtype)]), dev),
                   _w_arg(gm.op, W, gm.w_off, dev), C)
    C.sync_from_device()
    c = C.to_torch().float().numpy()[:T].reshape(T, COLS, pitch)[:, :, :H // COLS].reshape(T, H)
    want = Aq @ _f32(g, f"blk.{L}.ffn_down_shexp.weight").T
    err = ref.rel_err(c, want)
    print(f"layer {L} ({region.blocks[0].fmt}) T {T}: shared down err {err:.2e}")
    assert err < 1.7e-2


EG_CAP = 3  # (test_expert_gemms: chunks per column at most)
EG_UNSET = 0x7FC1  # (what no drain writes)
EG_TOL = (3.5e-2, 1.35 * 2 ** -9 * (2048 / 64) ** 0.5)  # (H, the down's rows)


def _eg_lists(seed: int, n_experts: int) -> tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
    """test_expert_gemms' work lists: per column 1 .. EG_CAP chunks (column 0 EG_CAP) of random experts, rows 64 ..
    active or not (column 1's only through its last chunk)."""
    rng = np.random.default_rng(seed)
    ns = rng.integers(1, EG_CAP + 1, size=COLS)
    ns[0] = EG_CAP
    ex = [[int(e) for e in rng.integers(0, n_experts, size=int(n))] for n in ns]
    act = [rng.random(int(n)) < 0.5 for n in ns]
    act[1][:] = False
    return ns, ex, act


class _ExpertGemms:
    """A layer's expert GEMMs as test_expert_gemms runs them: the gate / up and the down in one library, the layer's
    image on the device, the A slots' panels from ``seed``."""

    def __init__(self, model: str, L: int, seed: int):
        from artlite.operators.fused import Library
        from artlite.operators.moe.expert_gemm import CHUNK, ExpertGemmOperator

        self.g, self.img, W = _image(model)
        cfg = self.img.cfg
        self.L, self.H, self.F = L, cfg.hidden_size, cfg.expert_ffn
        self.gu_r, self.dn_r = self.img.region(L, "exp_gu"), self.img.region(L, "exp_down")
        self.expert = ref.experts(self.g, L)
        self.gu = ExpertGemmOperator.of("gu", self.gu_r, cap=EG_CAP, out=self.F)
        self.dn = ExpertGemmOperator.of("down", self.dn_r, cap=EG_CAP, out=self.H)
        self.lib = Library(f"t_moe_eg_{model}{L}", [self.gu, self.dn]).compile().load()
        self.Wd = artlite.empty((self.img.nbytes // 2,), dtype=torch.bfloat16, device=self.lib.device)
        self.Wd._raw.map()[:] = W
        self.Wd._raw.sync_to_device()
        A = (torch.randn(COLS, EG_CAP, CHUNK, self.H, generator=torch.Generator().manual_seed(seed)) * 0.5).bfloat16()
        self.pan = torch.stack([torch.stack([bfp16.to_panels(A[c, k]) for k in range(EG_CAP)]) for c in range(COLS)])

    def run(self, ns: np.ndarray, ex: list[list[int]], act: list[np.ndarray],
            h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One program over the columns' work lists (ns chunks, their experts and active flags): H's slots (COLS,
        EG_CAP, c_slot) and the down's rows (COLS, EG_CAP, CHUNK, hidden) as bf16 bits (int16; EG_UNSET where no drain
        wrote). h: H's slots to run the down alone on."""
        from artlite.operators.moe.expert_gemm import N_PARAM
        from artlite.txn import device_address

        lib, gu, dn, dev = self.lib, self.gu, self.dn, self.lib.device
        ctx = lib._ctx
        XP = artlite.from_torch(self.pan.reshape(-1).contiguous(), dev)
        h0 = (torch.full((COLS * EG_CAP * gu.c_slot,), EG_UNSET, dtype=torch.int16) if h is None
              else torch.from_numpy(np.ascontiguousarray(h).reshape(-1)))
        Hb = artlite.from_torch(h0.view(torch.bfloat16), dev)
        Yb = artlite.from_torch(torch.full((COLS * EG_CAP * dn.c_slot,), EG_UNSET, dtype=torch.int16)
                                .view(torch.bfloat16), dev)
        cbs = {}
        for op, name, region, a_in, out in ((gu, "gu", self.gu_r, XP, Hb), (dn, "dn", self.dn_r, Hb, Yb)):
            cb = artlite.from_torch(torch.zeros(COLS * op.cb_words * 2, dtype=torch.bfloat16), dev)
            cbv = cb._raw.map().view(np.uint32)
            for c in range(COLS):
                op.stream(ex[c], act[c], cb=device_address(cb, 4 * c * op.cb_words, ctx),
                          slab=device_address(self.Wd, region.offset, ctx), slab_stride=region.col_stride,
                          c_address=device_address(out, 2 * c * EG_CAP * op.c_slot, ctx),
                          a_address=device_address(a_in, 2 * c * EG_CAP * op.a_slot, ctx),
                          out=cbv[c * op.cb_words:(c + 1) * op.cb_words])
            cb._raw.sync_to_device()
            cbs[name] = cb
        lib.set_params({f"{N_PARAM}{c}": int(ns[c]) for c in range(COLS)})
        a = {k: device_address(b, 0, ctx) for k, b in (("gu", cbs["gu"]), ("dn", cbs["dn"]), ("xp", XP), ("h", Hb),
                                                       ("y", Yb))}
        items = ([] if h is not None else [(0, [a["gu"], a["xp"], a["h"]])]) + [(1, [a["dn"], a["h"], a["y"]])]
        lib.run(items, bufs=[b._raw for b in (self.Wd, cbs["gu"], cbs["dn"], XP, Hb, Yb)])
        Hb.sync_from_device()
        Yb.sync_from_device()
        hr = Hb.to_torch().view(torch.int16).numpy().reshape(COLS, EG_CAP, -1).copy()
        yr = dn.y_rows(Yb.to_torch().view(torch.int16).numpy().reshape(COLS, EG_CAP, -1))[..., :self.H].copy()
        return hr, yr

    def want_h(self, w: tuple[np.ndarray, ...], c: int, k: int) -> np.ndarray:
        """silu(gate) up of an expert's matrices w (``ref.experts``) on slot (c, k)'s A panels, fp64 (CHUNK, F)."""
        from artlite.operators.moe.expert_gemm import CHUNK

        gate, up = (np.asarray(m, np.float64) for m in w[:2])
        aq = bfp16.from_panels(self.pan[c, k], CHUNK, self.H).double().numpy()
        hg, hu = aq @ gate.T, aq @ up.T
        return hg / (1 + np.exp(-hg)) * hu

    def check(self, ns: np.ndarray, ex: list[list[int]], act: list[np.ndarray], hr: np.ndarray,
              yr: np.ndarray) -> list[dict]:
        """Per chunk: H's error against fp64 on its A panels, the down's against the exact product of the NPU's H (not
        finite: inf), whether the drains wrote exactly the rows that move, H's largest exponent byte on them, the first
        non-finite H value's (row, column), and grids of the 64-row strips x 8 column groups of H and of the down ('.'
        within the bound, 'x' beyond it, '!' not finite or beyond 1e30, '?' its reference not finite, '-' rows that do
        not move)."""
        from artlite.operators.moe.expert_gemm import CHUNK, M

        F = self.F
        out = []
        for c in range(COLS):
            for k in range(int(ns[c])):
                rows = CHUNK if act[c][k] or k == ns[c] - 1 else M
                w = self.expert(ex[c][k])
                down = np.asarray(w[2], np.float64)
                want_h = self.want_h(w, c, k)[:rows]
                n = rows // 8 * F // 8 * 36  # (H: the rows' token groups' panels first)
                with np.errstate(all="ignore"):
                    got_h = bfp16.from_panels(torch.from_numpy(hr[c, k]).view(torch.bfloat16), CHUNK, F).double().numpy()[:rows]
                    got_y = torch.from_numpy(yr[c, k, :rows]).view(torch.bfloat16).double().numpy()
                    want_y = got_h @ down.T
                    eh, ey = ref.rel_err(got_h, want_h), ref.rel_err(got_y, want_y)
                bad = np.argwhere(~np.isfinite(got_h) | (np.abs(got_h) > 1e30))
                ehdr = hr[c, k].view(np.uint8)[:rows * F * 9 // 8].reshape(rows // 8, F // 8, 8, 9)[..., 0]
                out.append({"c": c, "k": k, "e": ex[c][k], "rows": rows,
                            "err_h": eh if np.isfinite(eh) else np.inf, "err_y": ey if np.isfinite(ey) else np.inf,
                            "drains": bool((hr[c, k, :n] != EG_UNSET).any() and (hr[c, k, n:] == EG_UNSET).all()
                                           and (yr[c, k, rows:] == EG_UNSET).all()
                                           and not (yr[c, k, :rows] == EG_UNSET).all(axis=1).any()),
                            "emax": int(ehdr.max()), "bad_at": tuple(int(x) for x in bad[0]) if len(bad) else None,
                            "grid_h": _eg_grid(got_h, want_h, EG_TOL[0]), "grid_y": _eg_grid(got_y, want_y, EG_TOL[1])})
        return out


def _eg_grid(got: np.ndarray, want: np.ndarray, tol: float) -> str:
    """A chunk's 64-row strips x 8 column groups: '.' within tol of the chunk's largest reference value, 'x' beyond,
    '!' not finite or beyond 1e30, '?' the reference not finite, '-' rows that do not move."""
    w = got.shape[1] // 8
    with np.errstate(all="ignore"):
        scale = max(float(np.nan_to_num(np.abs(want), posinf=0).max()), 1e-30)
        strips = []
        for r in range(3):
            if 64 * r >= got.shape[0]:
                strips.append("-" * 8)
                continue
            s = ""
            for j in range(8):
                gb, wb = got[64 * r:64 * r + 64, j * w:j * w + w], want[64 * r:64 * r + 64, j * w:j * w + w]
                s += ("!" if not np.isfinite(gb).all() or np.abs(gb).max() > 1e30 else "?" if not np.isfinite(wb).all()
                      else "x" if np.abs(gb - wb).max() / scale > tol else ".")
            strips.append(s)
    return " ".join(strips)


def _eg_ok(res: list[dict]) -> bool:
    return all(r["drains"] and r["err_h"] < EG_TOL[0] and r["err_y"] < EG_TOL[1] for r in res)


def _eg_report(title: str, res: list[dict]) -> None:
    eh, ey = max(r["err_h"] for r in res), max(r["err_y"] for r in res)
    print(f"{title}: H err max {eh:.2e}, down err max {ey:.2e}; "
          f"{sum(not _eg_ok([r]) for r in res)} of {len(res)} chunks fail")
    for r in res:
        print(f"  col {r['c']} chunk {r['k']} expert {r['e']:3d} rows {r['rows']:3d}: H {r['err_h']:.1e} "
              f"[{r['grid_h']}] E max 0x{r['emax']:02x}  down {r['err_y']:.1e} [{r['grid_y']}]"
              + ("" if r["drains"] else "  (the drains wrote other rows)"))


def _eg_diagnose(model: str, eg: _ExpertGemms, ns: np.ndarray, ex: list[list[int]], act: list[np.ndarray],
                 hr: np.ndarray, yr: np.ndarray, res: list[dict]) -> None:
    """What a failing test_expert_gemms prints to locate the fault: the machine and the file; every chunk of the run;
    its first non-finite H panel's bytes; the same program again (do the outputs repeat?); every chunk active; one
    chunk per column; the down alone on H made on the host; the same work lists on the model's other layers (other
    tile formats)."""
    path = MODELS[model][0]()
    fw = pathlib.Path("/sys/class/accel/accel0/device/fw_version")
    print(f"\n--- test_expert_gemms[{model}-{eg.L}] diagnostics: {eg.gu.fmt} gate / up, {eg.dn.fmt} down; "
          f"{os.path.basename(path)} ({os.path.getsize(path)} bytes)")
    print(f"{platform.system()} {platform.release()}; amdxdna {artlite.Device().driver_version() or '?'} "
          f"({'xdna-driver' if os.path.exists('/sys/module/amdxdna/version') else 'in-kernel'}), firmware "
          f"{fw.read_text().strip() if fw.exists() else '?'}")
    from models.common import stale_toolchain

    head = subprocess.run(["git", "-C", os.path.dirname(__file__), "describe", "--always", "--dirty"],
                          capture_output=True, text=True, check=False).stdout.strip()
    print(f"checkout {head or '?'}; toolchain: mlir_aie {importlib.metadata.version('mlir_aie')}, llvm_aie "
          f"{importlib.metadata.version('llvm_aie')}" + (f"; {stale}" if (stale := stale_toolchain()) else ""))
    print("grids: rows 0-63 | 64-127 | 128-191 x 8 column groups; '.' ok, 'x' wrong, '!' not finite, '?' reference "
          "not finite, '-' rows that do not move")
    _eg_report("the test's work lists", res)
    for r in res:
        if r["bad_at"] is not None:
            t, f = r["bad_at"]
            off = (t // 8 * (eg.F // 8) + f // 8) * 72  # (the panel of 8 rows x 8 columns: 8 blocks [E, m0 .. m7])
            p = hr[r["c"], r["k"]].view(np.uint8)[off:off + 72]
            print(f"first non-finite H: col {r['c']} chunk {r['k']} row {t} column {f}; its panel's blocks: "
                  + " ".join(p[9 * i:9 * i + 9].tobytes().hex() for i in range(8)))
            break
    hr2, yr2 = eg.run(ns, ex, act)
    print(f"the same program again: {int((hr2 != hr).sum())} of {hr.size} H values and {int((yr2 != yr).sum())} of "
          f"{yr.size} down values differ")
    _eg_report("again", eg.check(ns, ex, act, hr2, yr2))
    every = [np.ones(int(n), bool) for n in ns]
    _eg_report("every chunk active", eg.check(ns, ex, every, *eg.run(ns, ex, every)))
    one, ex1, act1 = np.ones(COLS, np.int64), [e[:1] for e in ex], [np.ones(1, bool)] * COLS
    _eg_report("one chunk per column", eg.check(one, ex1, act1, *eg.run(one, ex1, act1)))
    from artlite.operators.moe.expert_gemm import CHUNK, M

    h = np.full_like(hr, EG_UNSET)
    for c in range(COLS):
        for k in range(int(ns[c])):
            rows = CHUNK if act[c][k] or k == ns[c] - 1 else M
            p = bfp16.to_panels(torch.from_numpy(eg.want_h(eg.expert(ex[c][k]), c, k)[:rows].astype(np.float32)))
            h[c, k, :p.numel()] = p.view(torch.int16).numpy()
    _eg_report("the down alone on H made on the host", eg.check(ns, ex, act, *eg.run(ns, ex, act, h=h)))
    for L2 in MODELS[model][1]:
        if L2 != eg.L:
            eg2 = _ExpertGemms(model, L2, eg.L)
            _eg_report(f"layer {L2} ({eg2.gu.fmt} / {eg2.dn.fmt}) on these work lists",
                       eg2.check(ns, ex, act, *eg2.run(ns, ex, act)))


@pytest.mark.parametrize(("model", "L"), LAYERS)
def test_expert_gemms(model: str, L: int) -> None:
    """The experts' gate / up and down (``moe.expert_gemm``: per column a work list of 192-token chunks, each of a
    random expert, rows 64 .. 191 active or not; the weights, A strips and drains pushed by the column's control stream)
    in one program: per chunk H = silu(gate) up within 3.5e-2 of fp64 on the chunk's A panels, the down within 3.4e-2
    of the exact product of the NPU's H (both halves; the 35B's Q4_K gate / up, Q6_K / Q4_K downs; Occamy's Q3_K and
    IQ3_XXS), on the rows that move; an inactive chunk's rows 64 .. (but the column's last chunk's) untouched; columns of
    1 .. EG_CAP chunks. A failure prints what locates it (``_eg_diagnose``)."""
    eg = _ExpertGemms(model, L, L)
    ns, ex, act = _eg_lists(L, eg.img.cfg.n_experts)
    hr, yr = eg.run(ns, ex, act)
    res = eg.check(ns, ex, act, hr, yr)
    print(f"layer {L}: chunks {[int(n) for n in ns]}; H err max {max(r['err_h'] for r in res):.2e}, down err max "
          f"{max(r['err_y'] for r in res):.2e}")
    if not _eg_ok(res):
        _eg_diagnose(model, eg, ns, ex, act, hr, yr, res)
    assert _eg_ok(res)


@pytest.mark.parametrize(("model", "L"), LAYERS)
def test_moe_block(model: str, L: int) -> None:
    """A whole MoE block of 256 tokens (random bf16 residual) in two programs around the host's routing (``plan``):
    the norm with lo panels, the router, the shared expert (gate / up, down: the prefill's whole-chunk designs); then the
    gathered norm (XP), the experts' gate / up and down (work lists), the combine and the residual norm. The routing from the NPU's logits equals the
    oracle's top 8 where the gaps are clear; the delta (routed + shared, ``moe_ffn`` + ``shared_expert`` on the routing
    the NPU's logits give) within 2.5e-2 of fp64 (the experts' bfp16 inputs and weights, bf16 H and y); the residual
    x + delta exactly the bf16 sum."""
    from artlite.operators.fused import Library
    from artlite.operators.moe import plan as MP
    from artlite.operators.moe.expert_gemm import N_PARAM, ExpertGemmOperator, M
    from artlite.operators.moe.gather import CHAIN, CombineOperator, GatherNormOperator, ds_row
    from artlite.txn import device_address

    g, img, W = _image(model)
    cfg = img.cfg
    T, H, F, Fs, E, K = 256, cfg.hidden_size, cfg.expert_ffn, cfg.shared_ffn, cfg.n_experts, cfg.n_experts_used
    eps, cap = cfg.rms_norm_eps, 34
    mi, sd = img.region(L, "moe_in"), img.region(L, "sdown")
    pitch = MG.shared_pitch(img, [L])
    norm_lo = PrefillNormOperator(T=T, E=H, eps=eps, lo=True)
    router, sgu = MG.router(img, mi, T, whole=True), MG.shared_gate_up(img, mi, T, width=Fs, whole=True)
    sdown = MG.row_group_gemm(sd, T, pitch=pitch, row=ds_row(H, pitch), whole=True)
    gnorm = GatherNormOperator(E=H, eps=eps, cap=cap)
    gu = ExpertGemmOperator.of("gu", img.region(L, "exp_gu"), cap=cap, out=F)
    dn = ExpertGemmOperator.of("down", img.region(L, "exp_down"), cap=cap, out=H)
    comb = CombineOperator(E=H, T=T, ds_pitch=pitch, y_layout=dn.y_layout)
    norm = PrefillNormOperator(T=T, E=H, eps=eps)
    ops = [norm_lo, router.op, sgu.op, sdown.op, gnorm, gu, dn, comb, norm]
    lib = Library(f"t_moe_block_{model}{L}", ops).compile().load()
    dev = lib.device
    rng = np.random.default_rng(10 + L)
    x = torch.from_numpy((rng.standard_normal((T, H)) * 2).astype(np.float32)).bfloat16()
    wn = np.asarray(g.tensor(f"blk.{L}.post_attention_norm.weight").dequantize().numpy(), np.float64).reshape(-1)
    bufs = {}

    def buf(name: str, n: int, init: torch.Tensor | None = None) -> object:
        bufs[name] = artlite.from_torch(init if init is not None else torch.zeros(n, dtype=torch.bfloat16), dev)
        return bufs[name]

    buf("PX", 0, x.clone().flatten())
    buf("D", T * H)
    buf("WN", 0, torch.from_numpy(wn).float().bfloat16())
    buf("XN2", 2 * T * H * 9 // 16)
    buf("LG", T * router.op.N)
    buf("HS", T * Fs * 9 // 16)
    buf("DS", sdown.op.M * sdown.op.N)
    Wd = buf("W", img.nbytes // 2)
    Wd._raw.map()[:] = W
    Wd._raw.sync_to_device()
    for name, op in (("CBG", gnorm), ("CBU", gu), ("CBD", dn), ("CBC", comb)):
        buf(name, COLS * op.cb_words * 2)
    buf("XP", COLS * cap * gnorm.slot)
    buf("H", COLS * cap * gu.c_slot)
    buf("Y", COLS * cap * dn.c_slot)
    buf("DL", T * H)
    buf("XN", T * H * 9 // 16)
    a = {k: device_address(b, 0, lib._ctx) for k, b in bufs.items()}
    raw = [b._raw for b in bufs.values()]
    lib.set_params({MG.CHUNK_PARAM: T // MG.CHUNK_UNIT})
    lib.run([(0, [a["PX"], a["D"], a["WN"], a["PX"], a["XN2"]]),
             (1, [a["XN2"], a["W"] + router.w_off, a["LG"]]),
             (2, [a["XN2"], a["W"] + sgu.w_off, a["HS"]]),
             (3, [a["HS"], a["W"] + sdown.w_off, a["DS"]])], bufs=raw)
    for k in ("LG", "PX"):
        bufs[k].sync_from_device()
    lg = bufs["LG"].to_torch().view(torch.float32).numpy().reshape(T, COLS, MG.router_pitch(mi))
    logits, sg = lg[:, :, :E // COLS].reshape(T, E), lg[:, 0, MG.PASS // 2]
    ids, w = MP.route(logits, K)
    plan = MP.Plan.of(ids, cap=cap, E=E)
    counts = plan.counts
    # the control streams, the weights, the counts
    px_rows = a["PX"] + 2 * H * np.arange(T)
    for name, op, region in (("CBU", gu, img.region(L, "exp_gu")), ("CBD", dn, img.region(L, "exp_down"))):
        v = bufs[name]._raw.map().view(np.uint32)
        a_in, out = (a["XP"], a["H"]) if name == "CBU" else (a["H"], a["Y"])
        for c in range(COLS):
            op.stream(plan.experts[c], plan.sizes[c] > M, cb=a[name] + 4 * c * op.cb_words,
                      slab=a["W"] + region.offset, slab_stride=region.col_stride,
                      c_address=out + 2 * c * cap * op.c_slot, a_address=a_in + 2 * c * cap * op.a_slot,
                      out=v[c * op.cb_words:(c + 1) * op.cb_words])
    v = bufs["CBG"]._raw.map().view(np.uint32)
    for c in range(COLS):
        words = gnorm.stream(px_rows[plan.rows[c]], plan.sizes[c], cb=a["CBG"] + 4 * c * gnorm.cb_words,
                             xp=a["XP"] + 2 * c * cap * gnorm.slot)
        v[c * gnorm.cb_words:c * gnorm.cb_words + words.size] = words
    v = bufs["CBC"]._raw.map().view(np.uint32)
    y_rows = a["Y"] + 2 * dn.y_offset(plan.pos)
    tpc = T // COLS
    for c in range(COLS):
        words = comb.stream(y_rows[c * tpc:(c + 1) * tpc].reshape(-1), cb=a["CBC"] + 4 * c * comb.cb_words)
        v[c * comb.cb_words:c * comb.cb_words + words.size] = words
    wt = bufs["DS"]._raw.map().view(np.float32).reshape(sdown.op.M, -1)[:T, COLS * pitch // 2:]  # (a row's weights)
    wt[:, :K] = w
    wt[:, K] = 1 / (1 + np.exp(-sg.astype(np.float64)))
    for k in ("CBG", "CBU", "CBD", "CBC", "DS"):
        bufs[k]._raw.sync_to_device()
    groups = [int((-(-plan.sizes[c] // CHAIN)).sum()) for c in range(COLS)]
    lib.set_params({f"{N_PARAM}{c}": counts[c] | groups[c] << 8 for c in range(COLS)})
    lib.run([(4, [a["CBG"], a["WN"], a["XP"]]), (5, [a["CBU"], a["XP"], a["H"]]), (6, [a["CBD"], a["H"], a["Y"]]),
             (7, [a["CBC"], a["DS"], a["DL"]]), (8, [a["PX"], a["DL"], a["WN"], a["PX"], a["XN"]])], bufs=raw)
    for k in ("DL", "PX"):
        bufs[k].sync_from_device()
    got = bufs["DL"].to_torch().double().numpy().reshape(T, H)
    # the oracle on the NPU's routing
    xd = x.double().numpy()
    xr = xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + eps) * wn
    want_lg = xr @ _f32(g, f"blk.{L}.ffn_gate_inp.weight").T
    rids, _ = ref.moe_route(want_lg.astype(np.float32), K)
    srt = np.sort(want_lg, 1)[:, ::-1]
    clear = srt[:, K - 1] - srt[:, K] > 1e-3
    assert (np.sort(ids, 1) == np.sort(rids, 1)).all(1)[clear].all()
    b = f"blk.{L}."
    shared = ref.shared_expert(xr, *(g.tensor(b + t).dequantize().numpy() for t in
                                     ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")),
                               _f32(g, b + "ffn_gate_inp_shexp.weight")[0])
    want = ref.moe_ffn(xr, ids, w, ref.experts(g, L)) + shared
    err, rms = ref.rel_err(got, want), np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean())
    print(f"layer {L}: chunks per column {counts}; delta err max {err:.2e}, rms {rms:.2e}")
    assert err < 4e-2 and rms < 3e-2
    assert torch.equal(bufs["PX"].to_torch().view(T, H), (x.float() + torch.from_numpy(got).float()).bfloat16())
