"""SD 1.5's overlay operators (artlite.operators.sd15.overlay) on the UNet's overlay designs (``AttnOverlay``: the
regular variant ``conv=True``, the transformer variant) against torch on random data: convs (the panel and the
conv-layout inputs, stride 2 into the lower level's half tiles, upsampling from the lower level), GEMMs (bf16 and
bfp16 outputs, GEGLU), GroupNorm (a concatenation's records) and LayerNorm; and a library program across the regular
and transformer variants (program-memory patches instead of reconfigurations)."""

import os
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import artlite
from artlite.operators.design import COLS, ROWS
from artlite.operators.sd15 import overlay as ov
from artlite.operators.sd15.attention import AttnOverlay
from artlite.operators.sd15.overlay import C_OBJ, SOBJ, ApplyOp, ConvL, GemmL, SumOp

SLACK = (SOBJ - C_OBJ) // 2


def bfp16(x: torch.Tensor, dim: int) -> torch.Tensor:
    """x rounded to bfp16ebs8 blocks of 8 along dim (round to nearest even, as the hardware)."""
    x = x.movedim(dim, -1)
    shape = x.shape
    xb = x.reshape(*shape[:-1], -1, 8).float()
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp_min(2.0 ** -120)))
    m = torch.round(xb / torch.exp2(e - 6))
    e = torch.where(m.amax(-1, keepdim=True) > 127, e + 1, e)
    m = torch.round(xb / torch.exp2(e - 6)).clamp(-128, 127)
    return (m * torch.exp2(e - 6)).reshape(shape).movedim(-1, dim)


def _dev(t: torch.Tensor | np.ndarray, dev: artlite.Device) -> artlite.Tensor:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t)).view(torch.bfloat16)
    return artlite.from_torch(t, dev)


def _zeros(n: int, dev: artlite.Device) -> artlite.Tensor:
    return artlite.from_torch(torch.zeros(n, dtype=torch.bfloat16), dev)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y - ref).norm() / ref.norm())


@pytest.mark.parametrize("W, ci, co, stride", [(64, 320, 320, 1), (32, 640, 640, 1), (32, 640, 640, 2)])
def test_conv(W: int, ci: int, co: int, stride: int) -> None:
    """conv3x3 of panel objects (64x64: the top and bottom halves as two ops; stride 2: the output at W / 2 as half
    tiles); ci chunks of at most 640. Within 1e-2 of the bfp16-rounded operands' conv (measured 6.5-9.2e-3)."""
    torch.manual_seed(0)
    ovl = AttnOverlay(W=W, conv=True)
    HT, n = W // stride, 2 if W == 64 and stride == 1 else 1
    cic = ConvL.max_chunk(ovl, HT // n, ci, co, stride=stride)
    ops = [[ConvL(ovl, Hp=HT // n, ci=ci, co=co, ci0=j * cic, cic=cic, stride=stride, Hp_total=HT, y0=k * HT // n)
            for k in range(n)] for j in range(ci // cic)]
    for o in (o for row in ops for o in row):
        o.compile()
    x = torch.randn(2, ci, W, W).bfloat16().float()
    w = torch.randn(co, ci, 3, 3) / (3 * ci ** 0.5)
    b = torch.randn(co) * 0.1
    dev = ops[0][0].device
    X, Z = _dev(ov.to_panels(x), dev), _zeros(ops[0][0].get_arg_spec()[1].shape[0], dev)
    y = 0
    for j, row in enumerate(ops):
        Wt = _dev(row[0].pack(w[:, j * cic:(j + 1) * cic], b if j == 0 else None), dev)
        Y = _zeros(row[0].y_elems, dev)
        for o in row:
            o.dispatch(X, Z, Wt, Y)
        y = y + ov.from_ctile(ov.ctile_values(Y.to_torch(), cbytes=row[0].cbytes), co, HT, W // stride)
    ref = F.conv2d(bfp16(x, 1), bfp16(w, 1), padding=1, stride=stride) + b.bfloat16().float()[:, None, None]
    assert _rel(y, ref) < 1e-2


def test_conv_layout_input() -> None:
    """conv_in's form: 32 channels (4 real) in the conv layout (rows padded), 64x64 as two ops."""
    torch.manual_seed(0)
    ovl = AttnOverlay(W=64, conv=True)
    ops = [ConvL(ovl, Hp=32, ci=32, co=320, panels=False, Hp_total=64, y0=32 * k) for k in range(2)]
    for o in ops:
        o.compile()
    x = torch.zeros(2, 32, 64, 64)
    x[:, :4] = torch.randn(2, 4, 64, 64)
    w = torch.zeros(320, 32, 3, 3)
    w[:, :4] = torch.randn(320, 4, 3, 3) / 6
    b = torch.randn(320) * 0.1
    dev = ops[0].device
    X, Wt, Y = _dev(ov.conv_layout(ov.pad_pairs(x)), dev), _dev(ops[0].pack(w, b), dev), _zeros(ops[0].y_elems, dev)
    for o in ops:
        o.dispatch(X, Wt, Y)
    y = ov.from_ctile(Y.to_torch().float(), 320, 64, 64)
    ref = F.conv2d(bfp16(x, 1), bfp16(w, 1), padding=1) + b.bfloat16().float()[:, None, None]
    assert _rel(y, ref) < 1e-2


def test_upsample_conv() -> None:
    """nearest 2x + conv3x3: the conv at level W reads level W / 2's panel objects (measured 9.1e-3)."""
    torch.manual_seed(0)
    W, ci = 32, 640
    ovl = AttnOverlay(W=W, conv=True)
    cic = ConvL.max_chunk(ovl, W, ci, ci)
    ops = [ConvL(ovl, Hp=W, ci=ci, co=ci, ci0=j * cic, cic=cic, up=True) for j in range(ci // cic)]
    for o in ops:
        o.compile()
    x = torch.randn(2, ci, W // 2, W // 2).bfloat16().float()
    w = torch.randn(ci, ci, 3, 3) / (3 * ci ** 0.5)
    b = torch.randn(ci) * 0.1
    dev = ops[0].device
    X, Z = _dev(ov.to_panels(x), dev), _zeros(ops[0].get_arg_spec()[1].shape[0], dev)
    y = 0
    for j, o in enumerate(ops):
        Y = _zeros(o.y_elems, dev)
        o.dispatch(X, Z, _dev(o.pack(w[:, j * cic:(j + 1) * cic], b if j == 0 else None), dev), Y)
        y = y + ov.from_ctile(Y.to_torch().float(), ci, W, W)
    xu = F.interpolate(x, scale_factor=2.0, mode="nearest")
    ref = F.conv2d(bfp16(xu, 1), bfp16(w, 1), padding=1) + b.bfloat16().float()[:, None, None]
    assert _rel(y, ref) < 1.2e-2


@pytest.mark.parametrize("W, K, N, out_bfp16", [(32, 640, 640, False), (32, 640, 640, True)])
def test_gemm(W: int, K: int, N: int, out_bfp16: bool) -> None:
    """A x W^T + b on panel objects (measured 5.4e-3; bfp16 output 8.6e-3)."""
    torch.manual_seed(0)
    ovl = AttnOverlay(W=W, conv=True)
    T = 2 * W * W // 128
    op = GemmL(ovl, T=T, K=K, N=N, out_bfp16=out_bfp16).compile()
    x = torch.randn(2, K, W, W)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    dev = op.device
    Y = _zeros(op.y_elems, dev)
    op.dispatch(_dev(ov.to_panels(x), dev), _dev(op.pack(w, b), dev), Y)
    y = ov.from_ctile(ov.ctile_values(Y.to_torch(), bfp=out_bfp16), N, W, W)
    ref = torch.einsum("bkhw,nk->bnhw", bfp16(x, 1), bfp16(w, 1)) + b.bfloat16().float()[:, None, None]
    assert _rel(y, ref) < (1e-2 if out_bfp16 else 8.5e-3)


def test_geglu() -> None:
    """ff1 with the GEGLU epilogue (C -> 8 C: 4 C outputs hid gelu(gate) in 40-channel objects), then ff2 over them
    in 640-K chunks (measured 1.3e-2: two bfp16 roundings and GELU's table). The transformer variant's code."""
    torch.manual_seed(0)
    W, C = 32, 320
    ovl = AttnOverlay(W=W)
    T = 2 * W * W // 128
    ff1 = GemmL(ovl, T=T, K=C, N=8 * C, geglu=True).compile()
    ff2 = [GemmL(ovl, T=T, K=4 * C, N=C, k0=k0, kc=640, cin=40).compile() for k0 in range(0, 4 * C, 640)]
    x = torch.randn(2, C, W, W).bfloat16().float()
    w1, b1 = torch.randn(8 * C, C) / C ** 0.5, torch.randn(8 * C) * 0.1
    w2, b2 = torch.randn(C, 4 * C) / (4 * C) ** 0.5, torch.randn(C) * 0.1
    dev = ff1.device
    H = _zeros(ff1.y_elems, dev)
    ff1.dispatch(_dev(ov.to_panels(x), dev), _dev(ff1.pack(w1, b1), dev), H)
    y = 0
    for o in ff2:
        Y = _zeros(o.y_elems, dev)
        o.dispatch(H, _dev(o.pack(w2[:, o.k0:o.k0 + 640], b2 if o.k0 == 0 else None), dev), Y)
        y = y + ov.from_ctile(Y.to_torch().float(), C, W, W)
    g = bfp16(x, 1).permute(0, 2, 3, 1) @ bfp16(w1, 1).T + b1.bfloat16().float()
    hid, gate = g.chunk(2, -1)
    hh = (hid.bfloat16().float() * F.gelu(gate.bfloat16().float(), approximate="tanh")).bfloat16().float()
    ref = (bfp16(hh, -1) @ bfp16(w2, 1).T + b2.bfloat16().float()).permute(0, 3, 1, 2)
    assert _rel(y, ref) < 2e-2


@pytest.mark.parametrize("W, Ch, Cs", [(64, 640, 320), (32, 640, 320)])
def test_group_norm_concat(W: int, Ch: int, Cs: int) -> None:
    """GroupNorm + SiLU of a skip s alone and of the concatenation [h | s]: s's sum emits a second record set (the
    concatenation's groups at h's pass offset), h's sum the concatenation's groups; one ST buffer (64x64: top and
    bottom ops). Within 9e-3 (bfp16 rounding of the result alone: 7.7e-3)."""
    torch.manual_seed(0)
    ovl = AttnOverlay(W=W, conv=True)
    TT, Ph, Ps, C = 2 * W * W // 128, Ch // 80, Cs // 80, Ch + Cs
    n = 2 if W == 64 else 1
    T = TT // n

    def tk(k: int) -> dict:
        return {"T": T, "TT": TT, "t0": k * T}
    ssum = [SumOp(ovl, P=Ps, S=1, stats="gn", C=Cs, stats2=(C, Ph), st_off=32 * k, n_st=2 * n, **tk(k)) for k in range(n)]
    hsum = [SumOp(ovl, P=Ph, S=1, stats="gn", C=C, st_off=32 * (n + k), n_st=2 * n, **tk(k)) for k in range(n)]
    apps = [ApplyOp(ovl, P=Ps, sums=ssum, **tk(k)) for k in range(n)]
    appc = [ApplyOp(ovl, P=Ph + Ps, sums=[(o, 0) for o in hsum] + [(o, 1) for o in ssum], **tk(k)) for k in range(n)]
    for o in ssum + hsum + apps + appc:
        o.compile()
    xs = (torch.randn(2, Cs, W, W) * (1 + torch.rand(1, Cs, 1, 1)) + torch.randn(1, Cs, 1, 1)).bfloat16().float()
    xh = (torch.randn(2, Ch, W, W) * (1 + torch.rand(1, Ch, 1, 1)) - torch.randn(1, Ch, 1, 1)).bfloat16().float()
    g1, b1, g2, b2 = 1 + 0.3 * torch.randn(Cs), 0.3 * torch.randn(Cs), 1 + 0.3 * torch.randn(C), 0.3 * torch.randn(C)
    dev = ssum[0].device
    sl = torch.zeros(SLACK, dtype=torch.bfloat16)
    yh, ys = hsum[0].y_elems, ssum[0].y_elems
    ST = _zeros(ssum[0].get_arg_spec()[3].shape[0], dev)
    Ys, Yh = _zeros(ys + SLACK, dev), _zeros(yh, dev)
    Xs, Xh = _dev(torch.cat([ov.to_ctile(xs), sl]), dev), _dev(torch.cat([ov.to_ctile(xh), sl]), dev)
    for o in ssum:
        o.dispatch(Xs, _dev(o.pack(None), dev), Ys, ST)
    for o in hsum:
        o.dispatch(Xh, _dev(o.pack(None), dev), Yh, ST)
    Zs = _zeros(apps[0].get_arg_spec()[-1].shape[0], dev)
    for o in apps:
        o.dispatch(Ys, ST, _dev(o.pack(g1, b1), dev), Zs)
    Xc = _dev(torch.cat([Yh.to_torch(), Ys.to_torch()[:ys], sl]), dev)
    Zc = _zeros(appc[0].get_arg_spec()[-1].shape[0], dev)
    for o in appc:
        o.dispatch(Xc, ST, _dev(o.pack(g2, b2), dev), Zc)
    for Z, x, Cx, g, b in ((Zs, xs, Cs, g1, b1), (Zc, torch.cat([xh, xs], 1), C, g2, b2)):
        ref = F.silu(F.group_norm(x, 32, g, b, 1e-5).bfloat16().float())
        assert _rel(ov.from_panels(Z.to_torch(), Cx, W, W), ref) < 9e-3


@pytest.mark.parametrize("W, C", [(64, 320), (32, 640)])
def test_layer_norm(W: int, C: int) -> None:
    """A two-summand sum with LayerNorm records, then the LayerNorm apply into panel objects (transformer variant).
    Within 9e-3 (measured 8.2-8.3e-3 with these per-pixel offsets)."""
    torch.manual_seed(0)
    ovl = AttnOverlay(W=W)
    TT, P = 2 * W * W // 128, C // 80
    n = 2 if W == 64 else 1
    T = TT // n
    sums = [SumOp(ovl, T=T, P=P, S=2, stats="ln", C=C, TT=TT, t0=k * T, st_off=32 * k, n_st=n) for k in range(n)]
    apps = [ApplyOp(ovl, T=T, P=P, sums=sums, mode="ln", TT=TT, t0=k * T) for k in range(n)]
    for o in sums + apps:
        o.compile()
    xs = [torch.randn(2, C, W, W).bfloat16().float() + torch.randn(1, 1, W, W) for _ in range(2)]
    gamma, beta = 1 + 0.3 * torch.randn(C), 0.3 * torch.randn(C)
    dev = sums[0].device
    X = _dev(torch.cat([ov.to_ctile(x) for x in xs] + [torch.zeros(SLACK, dtype=torch.bfloat16)]), dev)
    Y, ST = _zeros(sums[0].y_elems + SLACK, dev), _zeros(sums[0].get_arg_spec()[3].shape[0], dev)
    for o in sums:
        o.dispatch(X, _dev(o.pack(None), dev), Y, ST)
    Z = _zeros(apps[0].get_arg_spec()[-1].shape[0], dev)
    for o in apps:
        o.dispatch(Y, ST, _dev(o.pack(gamma, beta), dev), Z)
    h = (xs[0] + xs[1]).bfloat16().float()
    ref = F.layer_norm(h.permute(0, 2, 3, 1), (C,), gamma, beta).permute(0, 3, 1, 2)
    assert _rel(ov.from_panels(Z.to_torch(), C, W, W), ref) < 9e-3


def _core_main(elf: str, core: str) -> list[str]:
    """A core ELF's own function (the core main: the kernels it calls, by address), disassembled."""
    from artlite.operators.checks import OBJDUMP
    lines = subprocess.run([OBJDUMP, "-d", f"--disassemble-symbols={core}", elf], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    return lines[next(i for i, ln in enumerate(lines) if ln.endswith(f"<{core}>:")):]


def test_variant_patches() -> None:
    """A library program of a conv (regular variant), a GEMM (transformer variant) and the conv again: the designs are
    program-memory patches of each other (both directions; every core's main byte for byte the same: the patch
    rewrites the kernels while the cores wait in it, ``Overlay.source``), and both operators compute what they compute
    alone."""
    from artlite.operators.sd15.unet import Arena, Program
    torch.manual_seed(0)
    W, C = 32, 640
    r, t = AttnOverlay(W=W, conv=True), AttnOverlay(W=W)
    c = ConvL(r, Hp=W, ci=C, co=C)
    g = GemmL(t, T=16, K=C, N=C)
    prog = Program("sd15_variants")
    prog.add(c, "x", "z", "wc", "y")
    prog.add(g, "a", "wg", "y2")
    prog.add(c, "x", "z", "wc", "y3")
    prog.compile()
    prj = os.path.join(os.path.dirname(prog.lib.elf_path), "design.mlir.prj")  # (the fused design's devices: op<i>_<class>)
    for core in (f"core_{col}_{row}" for col in range(COLS) for row in range(2, 2 + ROWS)):
        conv, gemm = (_core_main(os.path.join(prj, f"elfs_{d}_{core}", f"elfs_{d}_{core}.elf"), core)
                      for d in ("op0_ConvL", "op1_GemmL"))
        assert conv == gemm, f"{core}: the variants' mains differ"
    patches = prog.lib.load()._asm.patches
    assert {(0, 1), (1, 0)} <= set(patches), patches.keys()
    x, a = torch.randn(2, C, W, W).bfloat16().float(), torch.randn(2, C, W, W)
    wc, bc = torch.randn(C, C, 3, 3) / (3 * C ** 0.5), torch.randn(C) * 0.1
    wg, bg = torch.randn(C, C) / C ** 0.5, torch.randn(C) * 0.1
    data = [("x", ov.to_panels(x)), ("a", ov.to_panels(a)), ("wc", c.pack(wc, bc)), ("wg", g.pack(wg, bg))]
    sizes = {"x": c.get_arg_spec()[0].shape[0], "z": c.get_arg_spec()[1].shape[0], "wc": c.w_bytes // 2,
             "y": c.y_elems, "a": g.get_arg_spec()[0].shape[0], "wg": g.w_bytes // 2, "y2": g.y_elems, "y3": c.y_elems}
    names, top = {}, 0
    for k, v in sizes.items():
        names[k], top = (top, v), top + -(-v // 2048) * 2048
    ar = Arena(prog.lib.device, names, top, data)
    prog.run(ar)
    ref_c = F.conv2d(bfp16(x, 1), bfp16(wc, 1), padding=1) + bc.bfloat16().float()[:, None, None]
    ref_g = torch.einsum("bkhw,nk->bnhw", bfp16(a, 1), bfp16(wg, 1)) + bg.bfloat16().float()[:, None, None]
    for name, ref, co in (("y", ref_c, C), ("y2", ref_g, C), ("y3", ref_c, C)):
        assert _rel(ov.from_ctile(ar.get(name).float(), co, W, W), ref) < 1e-2, name
