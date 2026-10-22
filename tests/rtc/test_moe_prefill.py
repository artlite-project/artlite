"""An MoE layer's prefill operators (operators/moe: the router, the shared expert, the experts' work lists, the
combine) on Qwen3.6-35B-A3B's decode image (``qwen35.image.QwenImage``, layers packed in host memory, each GEMM's W
argument copied to the device), against fp64 models on the operators' own inputs."""

import os

import numpy as np
import pytest
import torch

import artlite
from artlite.gguf_model import GGUFFile
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.moe import gemm as MG
from artlite.operators.prefill.norm import PrefillNormOperator

from . import qwen35_ref as ref

PATH = ref.qwen_gguf("Q4_K_M", "35B-A3B")
_state: dict[str, object] = {}


def _image(layers: tuple[int, ...] = (3, 5)) -> tuple[GGUFFile, object, np.ndarray]:
    """The 35B's image of the layers (an attention layer with Q6_K downs, a GDN layer with Q4_K), packed (cached per module)."""
    if not os.path.exists(PATH):
        pytest.skip(f"{PATH} missing")
    if _state.get("layers") != layers:
        from artlite.operators.qwen35.image import QwenImage

        g = GGUFFile(PATH)
        img = QwenImage(g, layers=list(layers), vocab=COLS * 64)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W, release=False)
        _state.update(layers=layers, g=g, img=img, W=W)
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


@pytest.mark.parametrize("L", [3, 5])
def test_router(L: int) -> None:
    """The post-attention norm with lo panels (``PrefillNormOperator(lo=True)``), then the router GEMM
    (``moe.gemm.router``: BFP16 hi / lo rows against x hi + lo, fp32 pairs): the logits within 5e-5 of fp64 math on the
    bf16 residual (rms, the fp32 norm weight, the F32 rows; x hi + lo carries ~2^-16 of each block's maximum: 2.6e-5, the
    logits 3.1-4.2e-5), the shared gate's within 8e-5 (one row: 4.6-5.8e-5); the top 8 the oracle's where the 8th / 9th
    gap exceeds 1e-3."""
    g, img, W = _image()
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


def _normed(L: int, T: int) -> tuple[object, object, np.ndarray, np.ndarray]:
    """The post-attention norm (lo panels) of random bf16 x on the device: (its device, xn [hi | lo] panels, x fp64,
    x / rms(x) w fp64)."""
    g, img, _ = _image()
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


@pytest.mark.parametrize(("L", "T"), [(3, 256), (5, 256), (3, 128), (5, 64)])
def test_shared_gate_up(L: int, T: int) -> None:
    """The shared expert's gate / up (``moe.gemm.shared_gate_up``, Q4_K) on the norm's hi panels: silu(gate) up within
    3.5e-2 of fp64 on those panels' values (the bfp16 weights and the bfp16 output panels; as the dense gate/up)."""
    g, img, W = _image()
    cfg = img.cfg
    H, F = cfg.hidden_size, cfg.shared_ffn
    gm = MG.shared_gate_up(img, img.region(L, "moe_in"), T, width=F)
    gm.op.compile()
    dev, xn, _, _ = _normed(L, T)
    out = artlite.empty((T * F * 9 // 16,), dtype=torch.bfloat16, device=dev)
    gm.op.dispatch(xn, _w_arg(gm.op, W, gm.w_off, dev), out)
    out.sync_from_device()
    got = bfp16.from_panels(out.to_torch(), T, F).double().numpy()
    xn.sync_from_device()
    xh = bfp16.from_panels(xn.to_torch(), T, 2 * H).double().numpy()[:, :H]
    b = f"blk.{L}."
    gate, up = xh @ _f32(g, b + "ffn_gate_shexp.weight").T, xh @ _f32(g, b + "ffn_up_shexp.weight").T
    want = gate / (1 + np.exp(-gate)) * up
    err = ref.rel_err(got, want)
    print(f"layer {L} T {T}: shared silu(gate) up err {err:.2e}")
    assert err < 3.5e-2


@pytest.mark.parametrize(("L", "T"), [(3, 256), (5, 256), (3, 128), (5, 64)])
def test_shared_down(L: int, T: int) -> None:
    """The shared expert's down (``moe.gemm.row_group_gemm``: row groups of 2, GemmQd; Q6_K in layer 3, Q4_K in 5) on
    random bfp16 panels of the shared width: every column's outputs at its pitch within 1.7e-2 of the exact product
    on the panels' values (the largest error over the largest value: ~0.9e-2 for Q6_K, 1.4-1.5e-2 for Q4_K)."""
    g, img, W = _image()
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


@pytest.mark.parametrize("L", [3, 5])
def test_expert_gemms(L: int) -> None:
    """The experts' gate / up and down (``moe.expert_gemm``: per column a work list of 192-token chunks, each of a
    random expert, rows 64 .. 191 active or not; the weights, A strips and drains pushed by the column's control stream)
    in one program: per chunk H = silu(gate) up within 3.5e-2 of fp64 on the chunk's A panels, the down within 3.4e-2
    of the exact product of the NPU's H (both halves; Q6_K in layer 3, Q4_K in 5), on the rows that move; an inactive
    chunk's rows 64 .. (but the column's last chunk's) untouched; columns of 1 .. cap chunks."""
    from artlite.operators.fused import Library
    from artlite.operators.moe.expert_gemm import CHUNK, N_PARAM, ExpertGemmOperator, M
    from artlite.txn import device_address

    g, img, W = _image()
    cfg = img.cfg
    H, F, cap = cfg.hidden_size, cfg.expert_ffn, 3
    gu_r, dn_r = img.region(L, "exp_gu"), img.region(L, "exp_down")
    gu = ExpertGemmOperator.of("gu", gu_r, cap=cap, out=F)
    dn = ExpertGemmOperator.of("down", dn_r, cap=cap, out=H)
    lib = Library(f"t_moe_eg{L}", [gu, dn]).compile().load()
    dev = lib.device
    rng = np.random.default_rng(L)
    ns = rng.integers(1, cap + 1, size=COLS)
    ns[0] = cap
    ex = [list(rng.integers(0, cfg.n_experts, size=int(n))) for n in ns]
    act = [rng.random(int(n)) < 0.5 for n in ns]
    act[1][:] = False  # (a column whose only rows 64 .. are its last chunk's)
    Wd = artlite.empty((img.nbytes // 2,), dtype=torch.bfloat16, device=dev)
    Wd._raw.map()[:] = W
    Wd._raw.sync_to_device()
    A = (torch.randn(COLS, cap, CHUNK, H, generator=torch.Generator().manual_seed(L)) * 0.5).bfloat16()
    pan = torch.stack([torch.stack([bfp16.to_panels(A[c, k]) for k in range(cap)]) for c in range(COLS)])
    XP = artlite.from_torch(pan.reshape(-1).contiguous(), dev)
    UNSET = 0x7FC1  # (what no drain writes)
    Hb = artlite.from_torch(torch.full((COLS * cap * gu.c_slot,), UNSET, dtype=torch.int16).view(torch.bfloat16), dev)
    Yb = artlite.from_torch(torch.full((COLS * cap * dn.c_slot,), UNSET, dtype=torch.int16).view(torch.bfloat16), dev)
    cbs = {}
    for op, name, region, a_in, out in ((gu, "gu", gu_r, XP, Hb), (dn, "dn", dn_r, Hb, Yb)):
        cb = artlite.from_torch(torch.zeros(COLS * op.cb_words * 2, dtype=torch.bfloat16), dev)
        cbv = cb._raw.map().view(np.uint32)
        for c in range(COLS):
            op.stream(ex[c], act[c], cb=device_address(cb, 4 * c * op.cb_words, lib._ctx),
                      slab=device_address(Wd, region.offset, lib._ctx), slab_stride=region.col_stride,
                      c_address=device_address(out, 2 * c * cap * op.c_slot, lib._ctx),
                      a_address=device_address(a_in, 2 * c * cap * op.a_slot, lib._ctx),
                      out=cbv[c * op.cb_words:(c + 1) * op.cb_words])
        cb._raw.sync_to_device()
        cbs[name] = cb
    lib.set_params({f"{N_PARAM}{c}": int(ns[c]) for c in range(COLS)})
    a = {k: device_address(b, 0, lib._ctx) for k, b in (("gu", cbs["gu"]), ("dn", cbs["dn"]), ("xp", XP), ("h", Hb),
                                                        ("y", Yb))}
    lib.run([(0, [a["gu"], a["xp"], a["h"]]), (1, [a["dn"], a["h"], a["y"]])],
            bufs=[b._raw for b in (Wd, cbs["gu"], cbs["dn"], XP, Hb, Yb)])
    Hb.sync_from_device()
    Yb.sync_from_device()
    hv = Hb.to_torch().view(COLS, cap, -1)
    hr = hv.view(torch.int16).numpy()
    yr = dn.y_rows(Yb.to_torch().view(torch.int16).numpy().reshape(COLS, cap, -1))[..., :H]
    yv = torch.from_numpy(yr.copy()).view(torch.bfloat16)
    ex_w = ref.experts(g, L)
    errs_h, errs_y = [], []
    for c in range(COLS):
        for k in range(int(ns[c])):
            rows = CHUNK if act[c][k] or k == ns[c] - 1 else M
            gate, up, down = (np.asarray(m, np.float64) for m in ex_w(int(ex[c][k])))
            aq = bfp16.from_panels(pan[c, k], CHUNK, H).double().numpy()[:rows]
            hg, hu = aq @ gate.T, aq @ up.T
            g = rows // 8 * F // 8 * 36  # (H: the rows' token groups' panels first)
            assert (hr[c, k, :g] != UNSET).any() and (hr[c, k, g:] == UNSET).all(), (c, k, rows)
            assert (yr[c, k, rows:] == UNSET).all() and not (yr[c, k, :rows] == UNSET).all(axis=1).any(), (c, k)
            got_h = bfp16.from_panels(hv[c, k], CHUNK, F).double().numpy()[:rows]
            errs_h.append(ref.rel_err(got_h, hg / (1 + np.exp(-hg)) * hu))
            errs_y.append(ref.rel_err(yv[c, k, :rows].double().numpy(), got_h @ down.T))
    print(f"layer {L}: chunks {list(ns)}; H err max {max(errs_h):.2e}, down err max {max(errs_y):.2e}")
    assert max(errs_h) < 3.5e-2 and max(errs_y) < 1.35 * 2 ** -9 * (2048 / 64) ** 0.5


@pytest.mark.parametrize("L", [3, 5])
def test_moe_block(L: int) -> None:
    """A whole MoE block of 256 tokens (random bf16 residual) in two programs around the host's routing (``plan``):
    the norm with lo panels, the router, the shared expert (gate / up, down); then the gathered norm (XP), the experts'
    gate / up and down (work lists), the combine and the residual norm. The routing from the NPU's logits equals the
    oracle's top 8 where the gaps are clear; the delta (routed + shared, ``moe_ffn`` + ``shared_expert`` on the routing
    the NPU's logits give) within 2.5e-2 of fp64 (the experts' bfp16 inputs and weights, bf16 H and y); the residual
    x + delta exactly the bf16 sum."""
    from artlite.operators.fused import Library
    from artlite.operators.moe import plan as MP
    from artlite.operators.moe.expert_gemm import N_PARAM, ExpertGemmOperator, M
    from artlite.operators.moe.gather import CHAIN, CombineOperator, GatherNormOperator, ds_row
    from artlite.txn import device_address

    g, img, W = _image()
    cfg = img.cfg
    T, H, F, Fs, E, K = 256, cfg.hidden_size, cfg.expert_ffn, cfg.shared_ffn, cfg.n_experts, cfg.n_experts_used
    eps, cap = cfg.rms_norm_eps, 34
    mi, sd = img.region(L, "moe_in"), img.region(L, "sdown")
    pitch = MG.shared_pitch(img, [L])
    norm_lo = PrefillNormOperator(T=T, E=H, eps=eps, lo=True)
    router, sgu = MG.router(img, mi, T), MG.shared_gate_up(img, mi, T, width=Fs)
    sdown = MG.row_group_gemm(sd, T, pitch=pitch, row=ds_row(H, pitch))
    gnorm = GatherNormOperator(E=H, eps=eps, cap=cap)
    gu = ExpertGemmOperator.of("gu", img.region(L, "exp_gu"), cap=cap, out=F)
    dn = ExpertGemmOperator.of("down", img.region(L, "exp_down"), cap=cap, out=H)
    comb = CombineOperator(E=H, T=T, ds_pitch=pitch, y_layout=dn.y_layout)
    norm = PrefillNormOperator(T=T, E=H, eps=eps)
    ops = [norm_lo, router.op, sgu.op, sdown.op, gnorm, gu, dn, comb, norm]
    lib = Library(f"t_moe_block{L}", ops).compile().load()
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
