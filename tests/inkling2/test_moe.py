"""Inkling 2's MoE in the prefill (``artlite.moe`` on ``models.inkling2.image``'s expert units: 512, 512 and 128
channels of h; NVFP4 layers 0 .. 19, Q8_0 20 .. 23): the experts' work-list GEMMs against fp64 models on the operators'
own inputs, a visit's block through the prefill."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import artlite
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, inkling_gguf
from artlite.operators import bfp16
from artlite.operators.design import COLS

from ..common import to_numpy
from ..model_files import require
from ..numerics import rel, rel_max

_state: dict[str, object] = {}


def _image(L: int, quant: str = "UD-Q4_K_XL") -> tuple[GGUFFile, InklingConfig, object, np.ndarray]:
    """Layer L's image of a file, packed (the last one cached per module)."""
    if _state.get("L") != (L, quant):
        from models.inkling2.image import InklingImage

        path = inkling_gguf(quant)
        require(path)
        g = GGUFFile(path)
        img = InklingImage(g, layers=[L], vocab=COLS * 64)
        W = np.zeros(img.nbytes, np.uint8)
        img.pack(W)
        _state.update(L=(L, quant), g=g, cfg=img.cfg, img=img, W=W)
    return _state["g"], _state["cfg"], _state["img"], _state["W"]


def _situ(g: np.ndarray, u: np.ndarray, lb: float) -> np.ndarray:
    return 4 * np.tanh(g / 4) / (1 + np.exp(-g)) * lb * np.tanh(u / lb)


@pytest.mark.parametrize(("quant", "L"), [("UD-Q4_K_XL", 0), ("UD-Q4_K_XL", 20), ("UD-Q4_K_M", 20)], ids=str)
def test_expert_gemms(quant: str, L: int) -> None:
    """Inkling's expert GEMMs (``models.inkling2.moe.InkMoE``'s operators and items) in one program (per column a work
    list of 192-token chunks of random experts, rows 64 .. 191 active or not): the units' gate / up (``Region.qt``
    inputs: passes of a tail group; units 0 and 1 one design into columns 0 and 512 of H01, UD-Q4_K_XL's layer 20 (Q8_0)
    in parts of 256 columns, UD-Q4_K_M's (Q5_K: 3 windows a group, the tail tile's header records moved) into an H01 of
    1040 columns), H = situ(gate, up) on the chunk's A panels (scaled by the expert's gate / up scale, as the gathered
    norm does); their downs (units 0 and 1 as two halves: row groups of 2, two super-blocks a slot; unit 2: of 8, half a
    super-block a slot) against the exact product of the NPU's H."""
    from artlite.moe.expert_gemm import CHUNK, N_PARAM, M
    from artlite.operators.fused import Library
    from artlite.txn import device_address
    from models.inkling2.moe import InkMoE

    g, cfg, img, W = _image(L, quant)
    H, F, cap, lb = cfg.hidden_size, cfg.expert_ffn, 3, cfg.situ_linear_beta
    moe = InkMoE(img, layers=[L], sizes=(64,), sub=64, eps=cfg.rms_norm_eps)
    cap = min(cap, moe.cap)
    ops = moe.units[L][1]
    uo = img.unit_offsets(L)  # (unit 1 one unit's bytes after unit 0; an expert's run)
    lib = Library(f"t_ink_eg{L}", list(ops.values())).compile().load()
    idx = {n: i for i, n in enumerate(ops)}
    dev = lib.device
    rng = np.random.default_rng(5)
    ns = rng.integers(1, cap + 1, size=COLS)
    ns[0] = cap
    ex = [list(rng.integers(0, cfg.n_experts, size=int(n))) for n in ns]
    act = [rng.random(int(n)) < 0.5 for n in ns]
    act[1][:] = False
    gsc = moe.tab[L][2].astype(np.float64)  # (the experts' gate / up scales: ones without NVFP4)
    Wd = artlite.empty((img.nbytes // 2,), dtype=bfloat16, device=dev)
    Wd._raw.map()[:] = W
    Wd._raw.sync_to_device()
    A = torch.randn(COLS, cap, CHUNK, H, generator=torch.Generator().manual_seed(L)) * 0.5
    for c in range(COLS):
        for k in range(int(ns[c])):
            A[c, k] *= float(gsc[ex[c][k]])
    A = A.bfloat16()
    pan = np.stack([np.stack([bfp16.to_panels(to_numpy(A[c, k])) for k in range(cap)] +
                             [np.zeros_like(bfp16.to_panels(to_numpy(A[c, 0])))] * (moe.cap - cap)) for c in range(COLS)])
    XP = artlite.from_numpy(pan.reshape(-1), dev)
    UNSET = 0x7FC1
    buf = {"XP": XP}
    for name, op in (("H01", ops["gu01"]), ("H2", ops["gu2"]), ("Y01", ops["dn01"]), ("Y2", ops["dn2"])):
        buf[name] = artlite.from_numpy(np.full(COLS * moe.cap * op.c_slot, UNSET, np.int16).view(bfloat16), dev)
    items, cbs = [], []
    for _, name, slab, col0, a_in, out in moe.gemm_items[L]:
        op = ops[name]
        cb = artlite.from_numpy(np.zeros(COLS * op.cb_words * 2, bfloat16), dev)
        cbv = cb._raw.map().view(np.uint32)
        op.streams([np.asarray(e) for e in ex], act, cb=[device_address(cb, 4 * c * op.cb_words, lib._ctx) for c in range(COLS)],
                   slab=device_address(Wd, slab, lib._ctx), slab_stride=uo[1], expert_stride=uo[3], col0=col0,
                   c_address=[device_address(buf[out], 2 * c * moe.cap * op.c_slot, lib._ctx) for c in range(COLS)],
                   a_address=[device_address(buf[a_in], 2 * c * moe.cap * op.a_slot, lib._ctx) for c in range(COLS)],
                   out=[cbv[c * op.cb_words:(c + 1) * op.cb_words] for c in range(COLS)])
        cb._raw.sync_to_device()
        cbs.append(cb)
        items.append((idx[name], [device_address(b, 0, lib._ctx) for b in (cb, buf[a_in], buf[out])]))
    lib.set_params({f"{N_PARAM}{c}": int(ns[c]) for c in range(COLS)})
    lib.run(items, bufs=[b._raw for b in (Wd, *cbs, *buf.values())])
    for b in buf.values():
        b.sync_from_device()
    hv = {n: buf[n].numpy().reshape(COLS, moe.cap, -1) for n in ("H01", "H2")}
    yv = {n: ops[o].y_rows(buf[n].numpy().view(np.int16).reshape(COLS, moe.cap, -1))[..., :H].view(bfloat16)
          for n, o in (("Y01", "dn01"), ("Y2", "dn2"))}
    b = f"blk.{L}."
    errs: dict[str, list[float]] = {"H01": [], "H2": [], "Y01": [], "Y2": []}
    for c in range(COLS):
        for k in range(int(ns[c])):
            e = int(ex[c][k])
            rows = CHUNK if act[c][k] or k == ns[c] - 1 else M
            gu = np.asarray(g.rows(b + "ffn_gate_up_exps.weight", np.arange(2 * F * e, 2 * F * (e + 1))), np.float64)
            dn = np.asarray(g.rows(b + "ffn_down_exps.weight", np.arange(H * e, H * (e + 1))), np.float64)
            aq = bfp16.from_panels(pan[c, k], CHUNK, H).astype(np.float64)[:rows]
            hg, hu = aq @ gu[:F].T, aq @ gu[F:].T
            for name, lo, hi in (("H01", 0, 1024), ("H2", 1024, F)):
                width = ops["gu01" if name == "H01" else "gu2"].N  # (H's columns: whole passes)
                got = bfp16.from_panels(hv[name][c, k], CHUNK, width).astype(np.float64)[:rows, :hi - lo]
                errs[name].append(rel_max(got, _situ(hg[:, lo:hi], hu[:, lo:hi], lb)))
                y = yv["Y" + name[1:]][c, k, :rows].astype(np.float64)
                errs["Y" + name[1:]].append(rel_max(y, got @ dn[:, lo:hi].T))
    print(f"chunks {ns.tolist()}; " + ", ".join(f"{n} err max {max(v):.2e}" for n, v in errs.items()))
    # (Q8_0's weights lose more in bfp16 blocks than NVFP4's: H up to 3.8e-2 with the epilogue's bf16 situ, whose own
    # rms error is 0.8%; the downs up to 1.8e-2)
    assert max(errs["H01"] + errs["H2"]) < 4e-2 and max(errs["Y01"] + errs["Y2"]) < 2e-2


def test_moe_block() -> None:
    """Visit 0 with its FFN through the prefill (``models.inkling2.prefill``, a chunk of 200 tokens): the FFN's input
    (the depth mix after the attention) against the CPU model's, the router's logits, the top 8 the host takes from them
    against the model's sets (near ties flip: the MoE's chaos), and the combine's output (DL) against an fp64 MoE on the
    NPU's routing (the gathered norm with the experts' gate / up scales, the units' GEMMs, the combine of both downs)."""
    from artlite.moe.route import top_k_bias
    from models.inkling2.overlay import InklingOverlay
    from models.inkling2.prefill import InklingPrefill

    from .ref import Model, VisitCache, rms

    g, cfg, _, _ = _image(0)
    H, F, n, b = cfg.hidden_size, cfg.expert_ffn, 200, "blk.0."
    toks = np.random.default_rng(3).integers(0, cfg.vocab_size, n)
    x = rms(torch.from_numpy(g.rows("token_embd.weight", toks)), cfg.rms_norm_eps) * \
        torch.from_numpy(np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1))
    m = Model(g, cfg)
    m.record = {"ffn_in", "ffn_moe_logits", "ffn_moe_topk"}
    with torch.no_grad():
        m.visit(cfg.visits[0], x, [], VisitCache(), 0)
    ov = InklingOverlay(g, visits=[0], max_positions=1024, ffn=True)
    ov.load()
    pf = InklingPrefill(ov, sizes=(256,), ffn=True).compile().bind()
    ov.reset()
    dev, rows, route = ov.dev, {}, pf.moe.host_route

    def host_route(L: int, **kw: object) -> object:  # (the FFN's input as the gathered norm reads it: the experts'
        name, o = pf._buf(kw["xf"])  # outputs share its buffer after it, ``prefill.POOLS``)
        dev.fetch(name, o, 2 * n * H)
        rows["XM0"] = dev.view(name)[o:o + 2 * n * H].view(bfloat16).astype(np.float64).reshape(n, H)
        return route(L, **kw)

    pf.moe.host_route = host_route
    pf.run(x.numpy(), 0, 256)
    dev.fetch("DL", 0, 2 * n * H)
    rows["DL"] = dev.view("DL")[:2 * n * H].view(bfloat16).astype(np.float64).reshape(n, H)
    e_in = rel(rows["XM0"], m.out[("ffn_in", 0)].numpy())
    lg = pf.moe.logits(0, n=n, t=256, dev=dev)
    e_lg = rel(lg, m.out[("ffn_moe_logits", 0)].numpy())
    bias, dsc, _ = pf.moe.tab[0]
    ids, wts = top_k_bias(lg, cfg.n_experts_used, bias, dsc)
    same = (np.sort(ids, 1) == np.sort(m.out[("ffn_moe_topk", 0)].numpy(), 1)).all(1).mean()
    xn = rms(torch.from_numpy(rows["XM0"].astype(np.float32)), cfg.rms_norm_eps).double().numpy() * \
        m.vec(b + "ffn_norm.weight").double().numpy()
    sgu = m.vec(b + "ffn_gate_up_exps.scale").double().numpy()
    want = np.zeros((n, H))
    for e in np.unique(ids):
        t, j = np.nonzero(ids == e)
        gu = xn[t] @ np.asarray(g.rows(b + "ffn_gate_up_exps.weight", np.arange(2 * F * e, 2 * F * (e + 1))), np.float64).T
        gu *= sgu[e]
        h = _situ(gu[:, :F], gu[:, F:], cfg.situ_linear_beta)
        want[t] += wts[t, j][:, None] * (h @ np.asarray(g.rows(b + "ffn_down_exps.weight", np.arange(H * e, H * (e + 1))),
                                                         np.float64).T)
    e_dl = rel(rows["DL"], want)
    print(f"FFN input {e_in:.2e}, logits {e_lg:.2e}, top-8 sets as the model's for {same:.2f} of the tokens, "
          f"DL on the NPU's routing {e_dl:.2e}")
    assert e_in < 2e-2 and e_lg < 1e-2 and same > 0.8 and e_dl < 2.5e-2
