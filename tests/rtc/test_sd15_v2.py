"""sd15_v2 (artlite.operators.sd15_v2): the static design run by host-built transactions, its ops against torch on
random data."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artlite.operators.sd15 import overlay as old
from artlite.operators.sd15_v2 import design as D
from artlite.operators.sd15_v2 import layout as L
from artlite.operators.sd15_v2.build import Builder, Level
from artlite.operators.sd15_v2.ops import Apply, Attention, Gemm
from artlite.operators.sd15_v2.runtime import V2Device
from artlite.operators.sd15_v2.seq import ColOps, MemTask, RowOps
from artlite.operators.sd15_v2.unet import Tensor

COLS, ROWS = 8, 4


@pytest.fixture(scope="module")
def v2() -> V2Device:
    return V2Device()


def q(x: torch.Tensor) -> torch.Tensor:
    """x rounded to bfp16ebs8 blocks of 8 along the last dimension."""
    return torch.from_numpy(L.unblocks(L.blocks(np.ascontiguousarray(x.float().reshape(*x.shape[:-1], -1, 8).numpy()))
                                       )).reshape(x.shape)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y - ref).norm() / ref.norm())


def fills(v2: V2Device, name: str, images: list[np.ndarray], addr: int) -> dict[int, ColOps]:
    """Per column one fill of its image (buffer ``name``: the columns' images in a row) at memtile byte ``addr``."""
    size = images[0].size
    v2.alloc(name, COLS * size)
    v2.write(name, np.concatenate(images))
    return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": size // 4}], ddr=(v2.address(name, c * size), size, None))])
            for c in range(COLS)}


def drains(v2: V2Device, name: str, nbytes: int, addr: int) -> dict[int, ColOps]:
    v2.alloc(name, COLS * nbytes)
    return {c: ColOps(drains=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(v2.address(name, c * nbytes), nbytes,
                                                                                    None))]) for c in range(COLS)}


def merge(*ops: dict[int, ColOps]) -> dict[int, ColOps]:
    out = {}
    for o in ops:
        for c, co in o.items():
            m = out.setdefault(c, ColOps())
            m.fills += co.fills
            m.drains += co.drains
    return out


def gemm_rows(v2: V2Device, name: str, g: Gemm, w: torch.Tensor, b: torch.Tensor | None) -> dict:
    """Row images of a Gemm (per row its command, then its weights) in buffer ``name``; returns its RowOps by row."""
    imgs = [np.concatenate([g.command(b, r), g.weights(w, r)]) for r in range(ROWS)]
    per = imgs[0].size
    v2.alloc(name, ROWS * per)
    v2.write(name, np.concatenate(imgs))
    return {r: g.row_ops(v2.address(name, r * per)) for r in range(ROWS)}


@pytest.mark.parametrize("m, ntile, K, N, res, out", [
    (64, 2, 640, 320, True, "bfp16"), (64, 1, 1280, 1280, True, "bf16"), (64, 1, 320, 640, False, "bfp16"),
    (16, 1, 1280, 1280, True, "bfp16"), (16, 1, 2560, 320, False, "bf16"), (16, 2, 640, 640, True, "bf16"),
    (64, 1, 1280, 1280, False, "vt")])
def test_gemm(v2: V2Device, m: int, ntile: int, K: int, N: int, res: bool, out: str) -> None:
    """y = x W^T + b (+ r) on every column's m ntile tokens: within 8e-3 of the bfp16-rounded operands' GEMM (fp32 C,
    the bf16 / bfp16 / V^T panel output)."""
    torch.manual_seed(0)
    T = m * ntile
    x = torch.randn(COLS * T, K)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    r = torch.randn(COLS * T, N)
    X, RES = 0, K // 8 * T // 8 * L.PANEL
    Y = RES + (N // 80 * ntile * L.ct_bytes(m) if res else 0)
    g = Gemm(m=m, ntile=ntile, K=K, N=N, x=X, y=Y, res=RES if res else None, out=out)
    bfp = out != "bf16"
    ybytes = (N // 8 * T // 8 * L.PANEL) if bfp else N // 80 * ntile * L.ct_bytes(m)
    s = v2.seq()
    s.op(merge(fills(v2, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X),
               *([fills(v2, "r", [L.ctiles(r[c * T:(c + 1) * T], m) for c in range(COLS)], RES)] if res else [])), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(v2, "w", g, w, b))
    s.op(drains(v2, "y", ybytes, Y), {})
    v2.run(v2.transaction(s))
    yb = v2.read("y", COLS * ybytes).reshape(COLS, ybytes)
    unpack = {"bfp16": lambda b: L.from_panels(b, T, N), "vt": lambda b: L.from_vt_panels(b, T, N),
              "bf16": lambda b: L.from_ctiles(b, T, N, m)}[out]
    y = torch.cat([unpack(yb[c]) for c in range(COLS)])
    ref = q(x) @ q(w).T + b.bfloat16().float() + (r.bfloat16().float() if res else 0)
    assert _rel(y, ref) < 8e-3


@pytest.mark.parametrize("m, K, N", [(64, 320, 2560), (64, 640, 1280), (16, 320, 1280)])
def test_geglu(v2: V2Device, m: int, K: int, N: int) -> None:
    """ff1 with the GEGLU epilogue (N / 2 outputs hid gelu(gate), the weight's rows [hid | gate]): within 1.2e-2 of
    torch on the bfp16-rounded operands (bf16 hid and gate, GELU's table, the bfp16 output)."""
    torch.manual_seed(2)
    T = m
    x = torch.randn(COLS * T, K)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    X, Y = 0, K // 8 * T // 8 * L.PANEL
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, out="geglu")
    s = v2.seq()
    s.op(fills(v2, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), {})
    s.op({}, table_rows(v2, "tab", old._gelu_table()[0]))
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(v2, "w", g, w, b))
    ybytes = N // 16 * g.y_np * L.PANEL
    s.op(drains(v2, "y", ybytes, Y), {})
    v2.run(v2.transaction(s))
    out = v2.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([L.from_panels(out[c], g.y_np * 8, N // 2)[:T] for c in range(COLS)])
    u = q(x) @ q(w).T + b.bfloat16().float()
    hid, gate = u.chunk(2, -1)
    ref = hid.bfloat16().float() * F.gelu(gate.bfloat16().float(), approximate="tanh")
    assert _rel(y, ref) < 1.2e-2


def table_rows(v2: V2Device, name: str, tab: np.ndarray) -> dict:
    """A K_TABLE command and its table slot on every row."""
    from artlite.operators.sd15_v2 import core as K
    cmd = np.zeros(D.Q // 4, np.int32)
    cmd[K.C_KIND] = K.K_TABLE
    slot = np.zeros(D.Q, np.uint8)
    slot[:tab.size] = tab
    v2.alloc(name, 2 * D.Q)
    v2.write(name, np.concatenate([cmd.view(np.uint8), slot]))
    return {r: RowOps(runs=[(v2.address(name), 2 * D.Q)], b_objects=2, parity=False) for r in range(ROWS)}


def images(m: int) -> torch.Tensor:
    """The image of each token of the 8 columns' tiles of m (a token panel tp is image tp % 2)."""
    return (torch.arange(m // 8).repeat_interleave(8) % 2).repeat(COLS)


@pytest.mark.parametrize("mode, m, C, ntile", [("gnsilu", 64, 1280, 1), ("gn", 64, 640, 1), ("gnsilu", 16, 1280, 1),
                                              ("ln", 64, 1280, 1), ("ln", 16, 320, 1), ("gnsilu", 64, 640, 2), ("gnsilu", 64, 320, 4),
                                              ("gn", 64, 640, 2), ("ln", 64, 640, 2), ("ln", 64, 320, 4)])
def test_norm(v2: V2Device, mode: str, m: int, C: int, ntile: int) -> None:
    """A GEMM's output with its statistics (GroupNorm group sums through DDR and the B streams, LayerNorm per-token sums
    in the memtile) normalized by Apply into panels (ntile tiles of m per op): within 9e-3 of torch on the GEMM's bf16
    output (bfp16 rounding of the result alone ~7.7e-3)."""
    torch.manual_seed(1)
    T, K_ = m * ntile, 320
    x = torch.randn(COLS * T, K_)
    w = torch.randn(C, K_) / K_ ** 0.5
    b = torch.randn(C) * 2
    gamma, beta = 1 + 0.3 * torch.randn(C), 0.3 * torch.randn(C)
    gn = mode in ("gn", "gnsilu")
    X, Y = 0, K_ // 8 * T // 8 * L.PANEL
    REC = Y + C // 80 * ntile * L.ct_bytes(m)
    Z = REC + 16 * D.OCH
    g = Gemm(m=m, ntile=ntile, K=K_, N=C, x=X, y=Y, out="bf16", stats="gn" if gn else "ln", rec=REC, gsize=C // 32)
    a = Apply(m=m, nblk=C // 160, mode=mode, x=Y, y=Z, ntile=ntile, gsize=C // 32, rec=REC, tokens=COLS * T // 2, C=C)
    s = v2.seq()
    s.op(fills(v2, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(v2, "w", g, w, b))
    if gn:  # (the records: memtile -> DDR [col][core][256 B] -> each row's slot [col][256 B] + zeros)
        v2.alloc("rec", COLS * 4 * 256)
        s.op({c: ColOps(drains=[MemTask([{"addr": REC, "nwords": 256, "dims": [(64, 1), (4, D.OCH // 4)]}],
                                        ddr=(v2.address("rec", c * 1024), 1024, None))]) for c in range(COLS)}, {})
        s.ddr_sync()
        v2.alloc("zeros", D.Q)
    tabs = table_rows(v2, "tab", old.silu_table()) if mode == "gnsilu" else {}
    if tabs:
        s.op({}, tabs)
    npr = 4 // len(a.rows)  # (the producer cores whose records the i-th core sums: [npr i, ..))
    cmds = [a.command(r, npr * COLS) for r in a.rows]
    gbs = [a.gamma_beta(gamma, beta, r) for r in a.rows]
    v2.alloc("ap", ROWS * 2 * D.Q)
    v2.write("ap", np.concatenate([x for i in range(len(a.rows)) for x in (cmds[i], gbs[i])]))
    rows = {}
    for i, r in enumerate(a.rows):
        rec_runs = []
        if gn:  # (the i-th core's slot: its producers' records in every column (a shim gather), then zeros)
            rec_runs = [(v2.address("rec", npr * i * 256), npr * COLS * 256, (64 * npr, COLS, 256, 2048)),
                        (v2.address("zeros"), D.Q - npr * COLS * 256)]
        rows[r] = a.row_ops(v2.address("ap", 2 * i * D.Q), rec_runs, 1 if gn else 0)
    s.op({c: a.col_ops() for c in range(COLS)}, rows)
    nz = C // 8 * T // 8 * L.PANEL
    ny = C // 80 * ntile * L.ct_bytes(m)
    s.op(merge(drains(v2, "y", ny, Y), drains(v2, "z", nz, Z)), {})
    v2.run(v2.transaction(s))
    yb = v2.read("y", COLS * ny).reshape(COLS, -1)
    y = torch.cat([L.from_ctiles(yb[c], T, C, m) for c in range(COLS)])
    zb = v2.read("z", COLS * nz).reshape(COLS, nz)
    z = torch.cat([L.from_panels(zb[c], T, C) for c in range(COLS)])
    if gn:
        img = images(T)
        ref = torch.zeros_like(y)
        for i in range(2):
            yi = y[img == i].T[None]  # (1, C, tokens)
            ref[img == i] = F.group_norm(yi, 32, gamma, beta, 1e-5)[0].T
        if mode == "gnsilu":
            ref = F.silu(ref.bfloat16().float())
    else:
        ref = F.layer_norm(y, (C,), gamma, beta, 1e-5)
    assert _rel(z, ref) < 9e-3


@pytest.mark.parametrize("np_, d, nb, rows", [(8, 160, 1, 2), (2, 160, 1, 2), (16, 80, 2, 2), (16, 80, 2, 4),
                                              (8, 160, 1, 4)])
def test_attention(v2: V2Device, np_: int, d: int, nb: int, rows: int) -> None:
    """Self-attention of 8 heads of d 160 (16x16: 8 token panels per column; 8x8: 2) or 80 (a 32x32 band: 16 token
    panels per column, the keys of 2 bands): each column's queries (of band 0) against all keys of their image (K / V^T
    panels drained to DDR slots, the attention rows' B streams; rows 4: every core row, rows 2, 3 switched to role a):
    within 1.5e-2 of softmax attention on the bfp16-rounded operands (measured ~1.16e-2 at d 160: P in bfp16 per 8 keys
    ~8e-3, the bfp16 output ~6e-3, vexp2)."""
    torch.manual_seed(3)
    T, C = 8 * np_, 8 * d
    x = torch.randn(3, nb, COLS * T, C)
    qs, k, v = x[0, 0] * (1.4426950408889634 / d ** 0.5) * 2, x[1], x[2]
    QA = 0
    KA = QA + C // 8 * np_ * L.PANEL + D.A_OBJ  # (the last head's walk past its panels)
    VA = KA + C // 8 * np_ * L.PANEL
    OA = VA + C // 8 * np_ * L.PANEL
    cols = [slice(c * T, (c + 1) * T) for c in range(COLS)]
    ar = D.ATTN_ROWS if rows == 2 else tuple(range(ROWS))
    at = Attention(np=np_, q=QA, o=OA, kv=0, d=d, nbands=nb, rows=ar)
    v2.alloc("kv", at.kv_bytes())
    at = Attention(np=np_, q=QA, o=OA, kv=v2.address("kv"), d=d, nbands=nb, rows=ar)
    s = v2.seq()
    s.role(ar, "a")
    for b in range(nb):
        s.op(merge(fills(v2, f"k{b}", [L.panels(k[b, sl]) for sl in cols], KA),
                   fills(v2, f"v{b}", [L.vt_panels(v[b, sl]) for sl in cols], VA)), {})
        s.op({c: at.kv_drains(KA, VA, c, band=b) for c in range(COLS)}, {})
    s.ddr_sync()
    v2.alloc("acmd", D.Q)
    v2.write("acmd", at.command())
    s.op(fills(v2, "q", [L.panels(qs[sl]) for sl in cols], QA), {})
    for t in range(max(np_ // 8, 1)):  # (an op per tile)
        at = Attention(np=np_, q=QA, o=OA, kv=v2.address("kv"), d=d, nbands=nb, tile=t, rows=ar)
        s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(v2.address("acmd")))
    nbytes = C // 8 * np_ * L.PANEL
    s.op(drains(v2, "o", nbytes, OA), {})
    v2.run(v2.transaction(s))
    ob = v2.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.cat([L.from_panels(ob[c], T, C) for c in range(COLS)])
    img = images(T)
    ref = torch.zeros(COLS * T, C)
    for i in range(2):
        sel = img == i
        kk = torch.cat([q(k[b][sel]) for b in range(nb)])
        vq = torch.cat([torch.cat([q(v[b, sl].T).T for sl in cols])[sel] for b in range(nb)])
        for h in range(8):
            ch = slice(d * h, d * (h + 1))
            sc = q(qs[sel][:, ch]) @ kk[:, ch].T
            p = torch.exp2(sc - sc.max(-1, keepdim=True).values)
            ref[sel, ch] = (p @ vq[:, ch]) / p.sum(-1, keepdim=True)
    assert _rel(o, ref) < 1.5e-2


def level_tokens(x: torch.Tensor, rows: int, L_: int) -> torch.Tensor:
    """(2 images, C, H, W) -> per column its tokens (COLS, rows 2 L_ 8, C): column c's image rows c rows .., token panel
    (row, g, image), lane j = pixel x = g + L_ j."""
    _, C, H, W = x.shape
    assert H == COLS * rows and W == 8 * L_
    v = x.reshape(2, C, COLS, rows, 8, L_)  # i C c row j g
    return v.permute(2, 3, 5, 0, 4, 1).reshape(COLS, rows * L_ * 16, C)


def from_level_tokens(y: torch.Tensor, rows: int, L_: int) -> torch.Tensor:
    C = y.shape[-1]
    return y.reshape(COLS, rows, L_, 2, 8, C).permute(3, 5, 0, 1, 4, 2).reshape(2, C, COLS * rows, 8 * L_)


def qconv_w(w: torch.Tensor) -> torch.Tensor:
    """Conv weights (N, C, 3, 3) rounded as the B stream holds them (blocks of 8 input channels per tap)."""
    return q(w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@pytest.mark.parametrize("H, stride, Cin, N, res", [(16, 1, 640, 640, False), (16, 1, 640, 640, True),
                                                   (8, 1, 1280, 640, True), (16, 2, 640, 640, False)])
def test_conv(v2: V2Device, H: int, stride: int, Cin: int, N: int, res: bool) -> None:
    """A 3x3 conv (padding 1) over both images (16x16: 2 rows of 2 groups per column; 8x8: 1 of 1; stride 2: 16x16 ->
    8x8) with the halo rows from the neighbour columns, bias, a residual (one op of every pass: per pass the chunks'
    task, then the residual's): within 8e-3 of torch on the bfp16-rounded operands."""
    torch.manual_seed(4)
    Li = H // 8
    rows_in = H // COLS
    Ho = H // stride
    R, Lg = Ho // COLS, Ho // 8
    x = torch.randn(2, Cin, H, H)
    w, b = torch.randn(N, Cin, 3, 3) / (9 * Cin) ** 0.5, torch.randn(N) * 0.1
    m = 16 * R * Lg
    xt = level_tokens(x, rows_in, Li)
    T_in = xt.shape[1]
    X = 0
    Z = X + Cin // 8 * (T_in // 8) * L.PANEL
    RES = Z + 4096
    Y = RES + (N // 80 * L.ct_bytes(m) if res else 0)
    r_img = torch.randn(2, N, Ho, Ho)
    rt = level_tokens(r_img, R, Lg)
    g = Gemm(m=m, ntile=1, K=Cin, N=N, x=X, y=Y, res=RES if res else None, out="bf16", conv=stride, R=R, Lg=Lg, zero=Z,
             x_np=T_in // 8)
    assert g.conv_zero_bytes() <= 4096
    s = v2.seq()
    s.op(merge(fills(v2, "x", [L.panels(xt[c]) for c in range(COLS)], X), fills(v2, "z", [np.zeros(4096, np.uint8)] * COLS, Z),
               *([fills(v2, "r", [L.ctiles(rt[c], m) for c in range(COLS)], RES)] if res else [])), {})
    s.op({c: g.col_ops(c) for c in range(COLS)}, gemm_rows(v2, "w", g, w, b), barrier=True)
    ybytes = N // 80 * L.ct_bytes(m)
    s.op(drains(v2, "y", ybytes, Y), {})
    v2.run(v2.transaction(s))
    yb = v2.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], m, N, m) for c in range(COLS)]), R, Lg)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), stride=stride, padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    assert _rel(y, ref) < 8e-3


@pytest.mark.parametrize("Cin, N, res", [(320, 320, False), (640, 320, True)])
def test_conv_rows(v2: V2Device, Cin: int, N: int, res: bool) -> None:
    """A 3x3 conv (padding 1) of a 32x32 level in row tiles (the input's 4 rows of 4 groups per column whole in the
    memtile, an op per band of 2 output rows; halo rows from the neighbour columns), bias, a residual: within 8e-3 of
    torch on the bfp16-rounded operands."""
    torch.manual_seed(12)
    H, rc = 32, 4
    x = torch.randn(2, Cin, H, H)
    w, b = torch.randn(N, Cin, 3, 3) / (9 * Cin) ** 0.5, torch.randn(N) * 0.1
    r_img = torch.randn(2, N, H, H)
    xt, rt = level_tokens(x, rc, 4), level_tokens(r_img, rc, 4)  # (COLS, 256, C): band b = tokens [128 b, ..)
    Z, X = 0, 4096
    yb = N // 80 * 2 * L.ct_bytes(64)  # (a band's C tiles)
    Y = [X + Cin // 8 * 32 * L.PANEL + k * yb for k in range(2)]
    RES = Y if res else [None, None]  # (the output in place of the residual)
    s = v2.seq()
    s.op(merge(fills(v2, "x", [L.panels(xt[c]) for c in range(COLS)], X), fills(v2, "z", [np.zeros(4096, np.uint8)] * COLS, Z),
               *([fills(v2, f"r{k}", [L.ctiles(rt[c, 128 * k:128 * (k + 1)], 64) for c in range(COLS)], RES[k]) for k in range(2)]
                 if res else [])), {})
    for k in range(2):
        g = Gemm(m=64, ntile=2, K=Cin, N=N, x=X, y=Y[k], res=RES[k], out="bf16", conv=1, R=1, Lg=4, zero=Z, x_np=32,
                 rc=rc, r0=2 * k)
        s.op({c: g.col_ops(c) for c in range(COLS)}, gemm_rows(v2, f"w{k}", g, w, b), barrier=True)
    s.op(merge(*(drains(v2, f"y{k}", yb, Y[k]) for k in range(2))), {})
    v2.run(v2.transaction(s))
    ys = [v2.read(f"y{k}", COLS * yb).reshape(COLS, yb) for k in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(ys[k][c], 128, N, 64) for k in range(2)]) for c in range(COLS)]),
                          rc, 4)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    err = _rel(y, ref)
    assert err < 8e-3, err


def _rand_resnet(p: str, cin: int, cout: int) -> dict:
    g = torch.Generator().manual_seed(5)
    w = {p + "norm1.weight": 1 + 0.2 * torch.randn(cin, generator=g), p + "norm1.bias": 0.2 * torch.randn(cin, generator=g),
         p + "conv1.weight": torch.randn(cout, cin, 3, 3, generator=g) / (9 * cin) ** 0.5,
         p + "conv1.bias": 0.1 * torch.randn(cout, generator=g),
         p + "norm2.weight": 1 + 0.2 * torch.randn(cout, generator=g), p + "norm2.bias": 0.2 * torch.randn(cout, generator=g),
         p + "conv2.weight": torch.randn(cout, cout, 3, 3, generator=g) / (9 * cout) ** 0.5,
         p + "conv2.bias": 0.1 * torch.randn(cout, generator=g)}
    if cin != cout:
        w[p + "conv_shortcut.weight"] = torch.randn(cout, cin, 1, 1, generator=g) / cin ** 0.5
        w[p + "conv_shortcut.bias"] = 0.1 * torch.randn(cout, generator=g)
    return w


@pytest.mark.parametrize("H, cin, cout", [(8, 1280, 1280), (16, 640, 1280), (16, 1280, 1280)])
def test_resnet(v2: V2Device, H: int, cin: int, cout: int) -> None:
    """diffusers' ResnetBlock2D on the level (GN1 + SiLU, conv1 + time bias, GN2 + SiLU, conv2 + x or the 1x1
    shortcut; the input's GroupNorm records by a copy op): within 1.5e-2 of the fp32 reference."""
    from artlite.operators.sd15_v2.unet import copy_stats, resnet
    from models.sd15 import reference
    torch.manual_seed(6)
    p = "r."
    w = _rand_resnet(p, cin, cout)
    tb = 0.3 * torch.randn(cout)
    x = torch.randn(2, cin, H, H)
    lv = Level(H)
    bd = Builder(v2)
    xt = Tensor(lv, cin, bd.mem.alloc(lv.ct(cin)))
    xt_cols = level_tokens(x, lv.R, lv.Lg)
    img = bd.image("x_in", np.concatenate([L.ctiles(xt_cols[c], lv.m) for c in range(COLS)]))
    assert img
    bd.op(bd.load("x_in", xt.addr, lv.ct(cin)), {})
    xt.rec = copy_stats(bd, xt, name="x.st", st=(cin // 32, 0))
    out = resnet(bd, w, p, xt, cout=cout, time_bias=tb, eps=1e-5, stats=None)
    bd.op(bd.spill("y_out", out.addr, lv.ct(cout)), {})
    v2.run(bd.transaction())
    yb = v2.read("y_out", COLS * lv.ct(cout)).reshape(COLS, -1)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, cout, lv.m) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, x, eps=1e-5, time_bias=tb)
    assert _rel(y, ref) < 1.5e-2


@pytest.mark.slow
@pytest.mark.parametrize("cin, cout", [(640, 640), (320, 640)])
def test_resnet_bands(v2: V2Device, cin: int, cout: int) -> None:
    """ResnetBlock2D on the banded 32x32 level (x and the output in DDR per band, h parked; the shortcut where cin !=
    cout): within 1.5e-2 of the fp32 reference."""
    from artlite.operators.sd15_v2.unet import DTensor, copy_stats, resnet_bands
    from models.sd15 import reference
    torch.manual_seed(13)
    p, H = "r.", 32
    w = _rand_resnet(p, cin, cout)
    tb = 0.3 * torch.randn(cout)
    x = torch.randn(2, cin, H, H)
    lv = Level(H, 2)
    bd = Builder(v2)
    xc = level_tokens(x, lv.R, lv.Lg)
    xd = DTensor(lv, cin, "x_in", rec=[])
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    for b in range(2):
        t = xd.load(bd, b)
        xd.rec += copy_stats(bd, t, name=f"x.st{b}", st=(cin // 32, 0))
        bd.mem.release(t.addr, t.nbytes)
    out = resnet_bands(bd, w, p, xd, cout=cout, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    v2.run(bd.transaction())
    yb = [v2.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, cout, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, x, eps=1e-5, time_bias=tb)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.slow
@pytest.mark.parametrize("ch, cs", [(1280, 640), (640, 320)])
def test_resnet_cat_bands(v2: V2Device, ch: int, cs: int) -> None:
    """The up path's resnet on [h | skip] on the banded 32x32 level (K parts of at most 640 channels, partial sums
    parked in DDR): within 1.5e-2 of the fp32 reference."""
    from artlite.operators.sd15_v2.unet import DTensor, resnet_cat_bands, skip_bands
    from models.sd15 import reference
    torch.manual_seed(14)
    p, H, cout = "r.", 32, 640
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    hx, sk = torch.randn(2, ch, H, H), torch.randn(2, cs, H, H)
    lv = Level(H, 2)
    bd = Builder(v2)

    def ddr(x: torch.Tensor, name: str) -> DTensor:
        xc = level_tokens(x, lv.R, lv.Lg)
        t = DTensor(lv, x.shape[1], name)
        for b in range(2):
            bd.image(t.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
        return t
    C = ch + cs
    hd = skip_bands(bd, ddr(hx, "h_in"), name="h", st=(C // 32, 0))
    sd = skip_bands(bd, ddr(sk, "s_in"), name="s", st=(C // 32, ch))
    out = resnet_cat_bands(bd, w, p, hd, sd, cout=cout, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    v2.run(bd.transaction())
    yb = [v2.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, cout, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, torch.cat([hx, sk], 1), eps=1e-5, time_bias=tb)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


def test_downsample_bands(v2: V2Device) -> None:
    """The stride-2 downsampler of the banded 32x32 level (x in DDR per band) into 16x16 C tiles: within 8e-3 of torch
    on the bfp16-rounded input."""
    from artlite.operators.sd15_v2.unet import DTensor, downsample_bands
    torch.manual_seed(15)
    C, p, H = 640, "s.", 32
    x = torch.randn(2, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    lv = Level(H, 2)
    bd = Builder(v2)
    xc = level_tokens(x, lv.R, lv.Lg)
    xd = DTensor(lv, C, "x_in")
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    out = downsample_bands(bd, w, p, xd)
    y = _level_output(v2, bd, out, "y_out")
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), stride=2, padding=1)
    assert _rel(y, ref) < 8e-3, _rel(y, ref)


@pytest.mark.slow
def test_upsample_bands(v2: V2Device) -> None:
    """The nearest 2x upsampler with its conv from 16x16 C tiles (1280 channels) into the banded 32x32 level (K parts,
    output halves, partial sums in DDR): within 8e-3 of torch on the bfp16-rounded input."""
    from artlite.operators.sd15_v2.unet import upsample_bands
    torch.manual_seed(16)
    C, p = 1280, "s."
    x = torch.randn(2, C, 16, 16)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(v2)
    xt = _level_input(bd, Level(16), x, "x_in")
    out = upsample_bands(bd, w, p, xt, stats=None, name="u")
    bd.op({}, {})
    v2.run(bd.transaction())
    lv = out.lv
    yb = [v2.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, C, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    xq = F.interpolate(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), padding=1)
    assert _rel(y, ref) < 8e-3, _rel(y, ref)


def _rand_transformer(p: str, C: int = 1280, cd: int = 768) -> dict:
    g = torch.Generator().manual_seed(7)
    q = p + "transformer_blocks.0."

    def lin(n: int, k: int, bias: bool = True, conv: bool = False) -> tuple:
        wt = torch.randn(n, k, generator=g) / k ** 0.5
        return (wt[:, :, None, None] if conv else wt), (0.1 * torch.randn(n, generator=g) if bias else None)

    w = {}
    for name, n, k, bias, conv in (("proj_in", C, C, True, True), ("proj_out", C, C, True, True)):
        w[p + name + ".weight"], w[p + name + ".bias"] = lin(n, k, bias, conv)
    for nm in (p + "norm.", q + "norm1.", q + "norm2.", q + "norm3."):
        w[nm + "weight"], w[nm + "bias"] = 1 + 0.2 * torch.randn(C, generator=g), 0.2 * torch.randn(C, generator=g)
    for a in ("attn1", "attn2"):
        kin = C if a == "attn1" else cd
        w[q + a + ".to_q.weight"] = lin(C, C)[0]
        w[q + a + ".to_k.weight"] = lin(C, kin)[0]
        w[q + a + ".to_v.weight"] = lin(C, kin)[0]
        w[q + a + ".to_out.0.weight"], w[q + a + ".to_out.0.bias"] = lin(C, C)
    w[q + "ff.net.0.proj.weight"], w[q + "ff.net.0.proj.bias"] = lin(8 * C, C)
    w[q + "ff.net.2.weight"], w[q + "ff.net.2.bias"] = lin(C, 4 * C)
    return w


@pytest.mark.parametrize("H", [8, 16])
def test_transformer(v2: V2Device, H: int) -> None:
    """SD 1.x's Transformer2DModel on the level (GN, proj_in, self-attention, cross-attention on 77 context keys (the
    host's K / V slots), GEGLU feed-forward in two halves, proj_out + x; 16x16: x spilled to DDR): within 2e-2 of the
    fp32 reference."""
    import types

    from artlite.operators.sd15_v2.unet import copy_stats, transformer
    from models.sd15 import reference
    torch.manual_seed(8)
    p, C = "t.", 1280
    w = _rand_transformer(p)
    x = torch.randn(2, C, H, H)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    lv = Level(H)
    bd = Builder(v2)
    xt = Tensor(lv, C, bd.mem.alloc(lv.ct(C)))
    xc = level_tokens(x, lv.R, lv.Lg)
    bd.image("x_in", np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load("x_in", xt.addr, lv.ct(C)), {})
    xt.rec = copy_stats(bd, xt, name="x.st", st=(C // 32, 0))
    kv = bd.ddr("kv_self", 16 * lv.np * 2 * D.Q)
    cks = L.kv_slots(ctx @ w[q + "to_k.weight"].T, ctx @ w[q + "to_v.weight"].T)
    ckv = bd.image("kv_ctx", cks)
    out = transformer(bd, w, p, xt, kv=kv, ctx_kv=ckv, nctx=77, stats=None, spill=H == 16)
    bd.op(bd.spill("y_out", out.addr, lv.ct(C)), {})
    v2.run(bd.transaction())
    yb = v2.read("y_out", COLS * lv.ct(C)).reshape(COLS, -1)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, C, lv.m) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    assert _rel(y, ref) < 2e-2


@pytest.mark.slow
def test_transformer_bands(v2: V2Device) -> None:
    """The transformer on a banded level (32x32, 640 channels, d 80: 2 bands, x and the output in DDR, h and Q of band 0
    parked): within 2e-2 of the fp32 reference."""
    import types

    from artlite.operators.sd15_v2.unet import DTensor, copy_stats, transformer_bands
    from models.sd15 import reference
    torch.manual_seed(11)
    p, C, H = "t.", 640, 32
    w = _rand_transformer(p, C)
    x = torch.randn(2, C, H, H)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    lv = Level(H, 2)
    bd = Builder(v2)
    xc = level_tokens(x, lv.R, lv.Lg)  # (COLS, 2 T, C): band b = tokens [T b, T (b + 1))
    xd = DTensor(lv, C, "x_in")
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    xd.rec = []
    for b in range(2):  # (the input's records: a copy op per band)
        t = xd.load(bd, b)
        xd.rec += copy_stats(bd, t, name=f"x.st{b}", st=(C // 32, 0))
        bd.mem.release(t.addr, t.nbytes)
    from artlite.operators.sd15_v2.ops import Attention
    kv = bd.ddr("kv_self", Attention(np=lv.np, q=0, o=0, kv=0, d=C // 8, nbands=2).kv_bytes())
    ckv = bd.image("kv_ctx", L.kv_slots(ctx @ w[q + "to_k.weight"].T, ctx @ w[q + "to_v.weight"].T, d=C // 8))
    out = transformer_bands(bd, w, p, xd, kv=kv, ctx_kv=ckv, nctx=77, stats=None, name="y")
    bd.op({}, {})  # (the last band's deferred drain)
    v2.run(bd.transaction())
    yb = [v2.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    yc = torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, C, lv.m) for b in range(2)]) for c in range(COLS)])
    y = from_level_tokens(yc, lv.R, lv.Lg)
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 2e-2


def _level_input(bd: Builder, lv: Level, x: torch.Tensor, name: str) -> Tensor:
    """x (2, C, H, H) as the level's C tiles: a DDR image, loaded into the memtile; the Tensor (no records)."""
    C = x.shape[1]
    t = Tensor(lv, C, bd.mem.alloc(lv.ct(C)))
    xc = level_tokens(x, lv.R, lv.Lg)
    bd.image(name, np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load(name, t.addr, lv.ct(C)), {})
    return t


def _level_output(v2: V2Device, bd: Builder, t: Tensor, name: str) -> torch.Tensor:
    bd.op(bd.spill(name, t.addr, t.nbytes), {})
    v2.run(bd.transaction())
    lv = t.lv
    yb = v2.read(name, COLS * t.nbytes).reshape(COLS, -1)
    return from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, t.C, lv.m) for c in range(COLS)]), lv.R, lv.Lg)


@pytest.mark.parametrize("H, ch, cs", [(8, 1280, 1280), (16, 1280, 1280), (16, 1280, 640)])
def test_resnet_cat(v2: V2Device, H: int, ch: int, cs: int) -> None:
    """The up path's resnet on the concatenation [h | skip] (the skip from DDR; GN over both, conv1 and the shortcut in
    two K-chunks): within 1.5e-2 of the fp32 reference."""
    from artlite.operators.sd15_v2.unet import copy_stats, free, make_skip, resnet_cat
    from models.sd15 import reference
    torch.manual_seed(9)
    p, cout = "r.", 1280
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    h, sk = torch.randn(2, ch, H, H), torch.randn(2, cs, H, H)
    lv = Level(H)
    bd = Builder(v2)
    st = _level_input(bd, lv, sk, "s_in")
    skip = make_skip(bd, st, name="skip", st=((ch + cs) // 32, ch))
    free(bd, st)
    ht = _level_input(bd, lv, h, "h_in")
    ht.rec = copy_stats(bd, ht, name="h.st", st=((ch + cs) // 32, 0))
    out = resnet_cat(bd, w, p, ht, skip, cout=cout, time_bias=tb, eps=1e-5, stats=None)
    y = _level_output(v2, bd, out, "y_out")
    ref = reference.resnet(w, p, torch.cat([h, sk], 1), eps=1e-5, time_bias=tb)
    assert _rel(y, ref) < 1.5e-2


@pytest.mark.parametrize("kind", ["down", "up"])
def test_resample(v2: V2Device, kind: str) -> None:
    """The stride-2 downsampler (16x16 -> 8x8) and the nearest 2x upsampler with its conv (8x8 -> 16x16): within 8e-3
    of torch on the bfp16-rounded input."""
    from artlite.operators.sd15_v2.unet import downsample, upsample
    torch.manual_seed(10)
    C, p = 1280, "s."
    H = 16 if kind == "down" else 8
    x = torch.randn(2, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(v2)
    xt = _level_input(bd, Level(H), x, "x_in")
    out = (downsample if kind == "down" else upsample)(bd, w, p, xt, stats=None)
    y = _level_output(v2, bd, out, "y_out")
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    if kind == "up":
        xq = F.interpolate(xq, scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), stride=2 if kind == "down" else 1,
                   padding=1)
    assert _rel(y, ref) < 8e-3


@pytest.mark.slow
def test_section(v2: V2Device) -> None:
    """The 32x32 .. 8x8 levels with SD 1.5's weights (down_blocks.1 .. up_blocks.2: ``unet.Section``, the sd15 overlay's
    tensor layouts at both ends) on the reference's own down_blocks.0 downsampler output of random latents (t 500, a
    random context), spliced into the fp32 reference UNet: the step's output within 1.5e-2 of the reference's."""
    import time

    from artlite.model_config import sd15_dir
    from models import is_diffusers
    from models.sd15 import reference
    from models.sd15.host import resnet_time_biases
    path = sd15_dir()
    if not is_diffusers(path):
        pytest.skip(f"{path}: not present")
    u = reference.UNet(f"{path}/unet")
    w, eps = u.w, u.cfg["norm_eps"]
    torch.manual_seed(0)
    lat, ctx = torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768)
    biases = resnet_time_biases(w, 500.0)

    def res(p: str, h: torch.Tensor) -> torch.Tensor:
        return reference.resnet(w, p, h, eps=eps, time_bias=biases[p])

    def tr(p: str, h: torch.Tensor) -> torch.Tensor:
        return u.transformer(p, h, ctx)

    h = reference.conv(w, "conv_in.", lat)
    skips = [h]
    for j in range(2):
        h = tr(f"down_blocks.0.attentions.{j}.", res(f"down_blocks.0.resnets.{j}.", h))
        skips.append(h)
    x5 = h = reference.conv(w, "down_blocks.0.downsamplers.0.conv.", h, stride=2)
    for i in (1, 2):
        skips.append(h)
        for j in range(2):
            h = tr(f"down_blocks.{i}.attentions.{j}.", res(f"down_blocks.{i}.resnets.{j}.", h))
            skips.append(h)
        h = reference.conv(w, f"down_blocks.{i}.downsamplers.0.conv.", h, stride=2)
    skips.append(h)
    for j in range(2):
        h = res(f"down_blocks.3.resnets.{j}.", h)
        skips.append(h)
    h = res("mid_block.resnets.1.", tr("mid_block.attentions.0.", res("mid_block.resnets.0.", h)))
    for i in range(3):
        for j in range(3):
            h = res(f"up_blocks.{i}.resnets.{j}.", torch.cat([h, skips.pop()], 1))
            if i:
                h = tr(f"up_blocks.{i}.attentions.{j}.", h)
        if i < 2:
            h = reference.conv(w, f"up_blocks.{i}.upsamplers.0.conv.", F.interpolate(h, scale_factor=2.0, mode="nearest"))
    ref = h

    def finish(h: torch.Tensor) -> torch.Tensor:  # (up_blocks.2's upsampler .. conv_out)
        sk = list(skips)
        h = reference.conv(w, "up_blocks.2.upsamplers.0.conv.", F.interpolate(h, scale_factor=2.0, mode="nearest"))
        for j in range(3):
            h = tr(f"up_blocks.3.attentions.{j}.", res(f"up_blocks.3.resnets.{j}.", torch.cat([h, sk.pop()], 1)))
        return reference.conv(w, "conv_out.", reference.group_norm(w, "conv_norm_out.", h, eps, silu=True))

    t0 = time.perf_counter()
    from artlite.operators.sd15 import overlay as old
    from artlite.operators.sd15_v2.unet import Section
    halves = old.to_ctile(x5).view(torch.uint8).numpy().reshape(4, 16, 2, 10240)  # (pass, tile, half: rows 2 t, 2 t + 1)
    x_old = np.zeros((4, 32, old.C_OBJ), np.uint8)
    x_old[:, :, :10240] = halves.reshape(4, 32, 10240)
    v2.alloc("x_old", x_old.size)
    v2.write("x_old", x_old)
    v2.alloc("out_old", 8 * 16 * old.C_OBJ)
    sec = Section(v2, w, x_in=v2.address("x_old"), out=v2.address("out_old"), eps=eps)
    sec.set_time(biases)
    sec.set_context(ctx)
    print(f"section: {sec.bd.s.n} ops, built in {time.perf_counter() - t0:.1f} s")
    sec.run()
    y = old.from_panels(torch.from_numpy(v2.read("out_old", 8 * 16 * old.C_OBJ)), 640, 32, 32)
    ts = []
    for _ in range(5):
        t1 = time.perf_counter()
        sec.run()
        ts.append(time.perf_counter() - t1)
    err = _rel(finish(y), finish(ref))
    print(f"section: up_blocks.2 rel err {_rel(y, ref):.3e}, the step's {err:.3e}; {np.median(ts) * 1e3:.1f} ms per dispatch")
    assert err < 1.5e-2
