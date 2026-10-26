"""The vision overlay's ops (``artlite.vision.ops``: GEMMs and their epilogues, the norms' applies, attention, convs)
on the NPU, run by host-built transactions, against torch on random data."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artlite.operators import kmath
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.device import VisionDevice
from artlite.vision.layers import gelu_table
from artlite.vision.ops import Apply, Attention, Gemm
from artlite.vision.seq import ColOps, MemTask, RowOps

COLS, ROWS = 8, 4


@pytest.fixture(scope="module")
def npu() -> VisionDevice:
    return VisionDevice()


def q(x: torch.Tensor) -> torch.Tensor:
    """x rounded to bfp16ebs8 blocks of 8 along the last dimension."""
    return torch.from_numpy(L.unblocks(L.blocks(np.ascontiguousarray(x.float().reshape(*x.shape[:-1], -1, 8).numpy()))
                                       )).reshape(x.shape)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y - ref).norm() / ref.norm())


def fills(npu: VisionDevice, name: str, images: list[np.ndarray], addr: int) -> dict[int, ColOps]:
    """Per column one fill of its image (buffer ``name``: the columns' images in a row) at memtile byte ``addr``."""
    size = images[0].size
    npu.alloc(name, COLS * size)
    npu.write(name, np.concatenate(images))
    return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": size // 4}], ddr=(npu.address(name, c * size), size, None))])
            for c in range(COLS)}


def drains(npu: VisionDevice, name: str, nbytes: int, addr: int) -> dict[int, ColOps]:
    npu.alloc(name, COLS * nbytes)
    return {c: ColOps(drains=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(npu.address(name, c * nbytes), nbytes,
                                                                                    None))]) for c in range(COLS)}


def merge(*ops: dict[int, ColOps]) -> dict[int, ColOps]:
    out = {}
    for o in ops:
        for c, co in o.items():
            m = out.setdefault(c, ColOps())
            m.fills += co.fills
            m.drains += co.drains
    return out


def gemm_rows(npu: VisionDevice, name: str, g: Gemm, w: torch.Tensor, b: torch.Tensor | None) -> dict:
    """Row images of a Gemm (per row its command, then its weights) in buffer ``name``; returns its RowOps by row."""
    imgs = [np.concatenate([g.command(b, r), g.weights(w, r)]) for r in range(ROWS)]
    per = imgs[0].size
    npu.alloc(name, ROWS * per)
    npu.write(name, np.concatenate(imgs))
    return {r: g.row_ops(npu.address(name, r * per)) for r in range(ROWS)}


@pytest.mark.parametrize("m, ntile, K, N, res, out", [
    (64, 2, 640, 320, True, "bfp16"), (64, 1, 1280, 1280, True, "bf16"), (64, 1, 320, 640, False, "bfp16"),
    (16, 1, 1280, 1280, True, "bfp16"), (16, 1, 2560, 320, False, "bf16"), (16, 2, 640, 640, True, "bf16"),
    (64, 1, 1280, 1280, False, "vt")])
def test_gemm(npu: VisionDevice, m: int, ntile: int, K: int, N: int, res: bool, out: str) -> None:
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
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X),
               *([fills(npu, "r", [L.ctiles(r[c * T:(c + 1) * T], m) for c in range(COLS)], RES)] if res else [])), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    unpack = {"bfp16": lambda b: L.from_panels(b, T, N), "vt": lambda b: L.from_vt_panels(b, T, N),
              "bf16": lambda b: L.from_ctiles(b, T, N, m)}[out]
    y = torch.cat([unpack(yb[c]) for c in range(COLS)])
    ref = q(x) @ q(w).T + b.bfloat16().float() + (r.bfloat16().float() if res else 0)
    assert _rel(y, ref) < 8e-3


@pytest.mark.parametrize("m, K, N", [(64, 320, 2560), (64, 640, 1280), (16, 320, 1280)])
def test_geglu(npu: VisionDevice, m: int, K: int, N: int) -> None:
    """ff1 with the GEGLU epilogue (N / 2 outputs hid gelu(gate), the weight's rows [hid | gate]): within 1.2e-2 of
    torch on the bfp16-rounded operands (bf16 hid and gate, GELU's table, the bfp16 output)."""
    torch.manual_seed(2)
    T = m
    x = torch.randn(COLS * T, K)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    X, Y = 0, K // 8 * T // 8 * L.PANEL
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, out="geglu")
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), {})
    s.op({}, table_rows(npu, "tab", gelu_table()))
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    ybytes = N // 16 * g.y_np * L.PANEL
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    out = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([L.from_panels(out[c], g.y_np * 8, N // 2)[:T] for c in range(COLS)])
    u = q(x) @ q(w).T + b.bfloat16().float()
    hid, gate = u.chunk(2, -1)
    ref = hid.bfloat16().float() * F.gelu(gate.bfloat16().float(), approximate="tanh")
    assert _rel(y, ref) < 1.2e-2


@pytest.mark.parametrize("m, ntile, K, N", [(64, 1, 1280, 4480), (16, 2, 1280, 1280)])
def test_gelu(npu: VisionDevice, m: int, ntile: int, K: int, N: int) -> None:
    """A GEMM with the GELU epilogue (MiniCPM-V's ViT feed-forward: gelu of the N outputs as panels): within 1.2e-2 of
    torch on the bfp16-rounded operands (bf16 outputs, GELU's table, the bfp16 output)."""
    torch.manual_seed(4)
    T = m * ntile
    x = torch.randn(COLS * T, K)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    X, Y = 0, K // 8 * T // 8 * L.PANEL
    g = Gemm(m=m, ntile=ntile, K=K, N=N, x=X, y=Y, out="gelu")
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), {})
    s.op({}, table_rows(npu, "tab", gelu_table()))
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    ybytes = N // 8 * g.y_np * L.PANEL
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    out = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([L.from_panels(out[c], g.y_np * 8, N)[:T] for c in range(COLS)])
    u = q(x) @ q(w).T + b.bfloat16().float()
    ref = F.gelu(u.bfloat16().float(), approximate="tanh")
    assert _rel(y, ref) < 1.2e-2


def table_rows(npu: VisionDevice, name: str, tab: np.ndarray) -> dict:
    """A K_TABLE command and its table slot on every row."""
    from artlite.vision import core as K
    cmd = np.zeros(D.Q // 4, np.int32)
    cmd[K.C_KIND] = K.K_TABLE
    slot = np.zeros(D.Q, np.uint8)
    slot[:tab.size] = tab
    npu.alloc(name, 2 * D.Q)
    npu.write(name, np.concatenate([cmd.view(np.uint8), slot]))
    return {r: RowOps(runs=[(npu.address(name), 2 * D.Q)], b_objects=2, parity=False) for r in range(ROWS)}


def images(m: int) -> torch.Tensor:
    """The image of each token of the 8 columns' tiles of m (a token panel tp is image tp % 2)."""
    return (torch.arange(m // 8).repeat_interleave(8) % 2).repeat(COLS)


@pytest.mark.parametrize("mode, m, C, ntile, nrows", [
    ("gnsilu", 64, 1280, 1, 4), ("gn", 64, 640, 1, 4), ("gnsilu", 16, 1280, 1, 4), ("ln", 64, 1280, 1, 4),
    ("ln", 16, 320, 1, 4), ("gnsilu", 64, 640, 2, 4), ("gnsilu", 64, 320, 4, 4), ("gn", 64, 640, 2, 4),
    ("ln", 64, 640, 2, 4), ("ln", 64, 320, 4, 4), ("gnsilu", 64, 1280, 1, 2), ("ln", 64, 640, 2, 2)])
def test_norm(npu: VisionDevice, mode: str, m: int, C: int, ntile: int, nrows: int) -> None:
    """A GEMM's output with its statistics (GroupNorm group sums through DDR and the B streams, LayerNorm per-token sums
    in the memtile) normalized by Apply on nrows core rows into panels (ntile tiles of m per op): within 9e-3 of torch
    on the GEMM's bf16 output (bfp16 rounding of the result alone ~7.7e-3)."""
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
    a = Apply(m=m, nblk=C // (80 * nrows), mode=mode, x=Y, y=Z, ntile=ntile, gsize=C // 32, rec=REC,
              tokens=COLS * T // 2, C=C, rows=tuple(range(4 - nrows, 4)))
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    if gn:  # (the records: memtile -> DDR [col][core][256 B] -> each row's slot [col][256 B] + zeros)
        npu.alloc("rec", COLS * 4 * 256)
        s.op({c: ColOps(drains=[MemTask([{"addr": REC, "nwords": 256, "dims": [(64, 1), (4, D.OCH // 4)]}],
                                        ddr=(npu.address("rec", c * 1024), 1024, None))]) for c in range(COLS)}, {})
        s.ddr_sync()
        npu.alloc("zeros", D.Q)
    tabs = table_rows(npu, "tab", np.tile(kmath.SILU.table_words(), 2).view(np.uint8)) if mode == "gnsilu" else {}
    if tabs:
        s.op({}, tabs)
    npr = 4 // len(a.rows)  # (the producer cores whose records the i-th core sums: [npr i, ..))
    cmds = [a.command(r, npr * COLS) for r in a.rows]
    gbs = [a.gamma_beta(gamma, beta, r) for r in a.rows]
    npu.alloc("ap", ROWS * 2 * D.Q)
    npu.write("ap", np.concatenate([x for i in range(len(a.rows)) for x in (cmds[i], gbs[i])]))
    rows = {}
    for i, r in enumerate(a.rows):
        rec_runs = []
        if gn:  # (the i-th core's slot: its producers' records in every column (a shim gather), then zeros)
            rec_runs = [(npu.address("rec", npr * i * 256), npr * COLS * 256, (64 * npr, COLS, 256, 2048)),
                        (npu.address("zeros"), D.Q - npr * COLS * 256)]
        rows[r] = a.row_ops(npu.address("ap", 2 * i * D.Q), rec_runs, 1 if gn else 0)
    s.op({c: a.col_ops() for c in range(COLS)}, rows)
    nz = C // 8 * T // 8 * L.PANEL
    ny = C // 80 * ntile * L.ct_bytes(m)
    s.op(merge(drains(npu, "y", ny, Y), drains(npu, "z", nz, Z)), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ny).reshape(COLS, -1)
    y = torch.cat([L.from_ctiles(yb[c], T, C, m) for c in range(COLS)])
    zb = npu.read("z", COLS * nz).reshape(COLS, nz)
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


@pytest.mark.parametrize("np_, d, nb, rows, sink, heads, masked", [
    (8, 160, 1, 2, False, 8, False), (2, 160, 1, 2, False, 8, False), (16, 80, 2, 2, False, 8, False),
    (16, 80, 2, 4, False, 8, False), (8, 160, 1, 4, False, 8, False), (32, 40, 4, 4, False, 8, False),
    (16, 80, 2, 4, True, 8, False), (8, 160, 1, 4, True, 8, False), (8, 80, 1, 4, False, 16, False),
    (8, 80, 1, 4, True, 16, False), (16, 80, 2, 4, True, 16, False), (16, 80, 1, 4, False, 16, False),
    (16, 80, 2, 4, False, 16, False), (8, 80, 2, 4, False, 16, True), (8, 160, 1, 4, False, 8, True)])
def test_attention(npu: VisionDevice, np_: int, d: int, nb: int, rows: int, sink: bool, heads: int, masked: bool) -> None:
    """Self-attention of 8 heads of d 160 (16x16: 8 token panels per column; 8x8: 2) or 80 (a 32x32 band: 16 token
    panels per column, the keys of 2 bands): each column's queries (of band 0) against all keys of their image (K / V^T
    panels drained to DDR slots, the attention rows' B streams; rows 4: every core row; d 40: a 64x64 band, the keys of
    4 bands, role c on every row; sink: a key of the last band x 30, its scores ~90 log2 units past the first group's
    for half the queries: the reference max raised, ``raise_max``; heads 16 of d 80: MiniCPM-V's ViT, at 16x16 a key
    group of two columns' keys; masked: a random fifth of each image's keys masked, ``valid``):
    within 1.5e-2 of softmax attention on the bfp16-rounded operands (measured ~1.16e-2 at d 160: P in bfp16 per 8 keys
    ~8e-3, the bfp16 output ~6e-3, vexp2)."""
    torch.manual_seed(3)
    T, C = 8 * np_, heads * d
    x = torch.randn(3, nb, COLS * T, C)
    if sink:
        x[1, nb - 1, COLS * T - 1] *= 30
    nkeys = 4 * COLS * T * nb // 8  # (an image's: half of each column's token panels, every band)
    valid = (torch.rand(2, nkeys) > 0.2).numpy() if masked else None
    qs, k, v = x[0, 0] * (1.4426950408889634 / d ** 0.5) * 2, x[1], x[2]
    QA = 0
    KA = QA + C // 8 * np_ * L.PANEL + D.A_OBJ  # (the last head's walk past its panels)
    VA = KA + C // 8 * np_ * L.PANEL
    OA = VA + C // 8 * np_ * L.PANEL
    if OA + C // 8 * np_ * L.PANEL > D.HEAP:  # (K's and V's panels drained before Q arrives: V in Q's place, O in K's)
        VA, OA = QA, KA
    cols = [slice(c * T, (c + 1) * T) for c in range(COLS)]
    ar = (0, 1) if rows == 2 else tuple(range(ROWS))
    at = Attention(np=np_, q=QA, o=OA, kv=0, d=d, nbands=nb, rows=ar, heads=heads)
    npu.alloc("kv", at.kv_bytes())
    at = Attention(np=np_, q=QA, o=OA, kv=npu.address("kv"), d=d, nbands=nb, rows=ar, raise_max=sink, heads=heads,
                   valid=valid)
    s = npu.seq()
    for b in range(nb):
        s.op(merge(fills(npu, f"k{b}", [L.panels(k[b, sl]) for sl in cols], KA),
                   fills(npu, f"v{b}", [L.vt_panels(v[b, sl]) for sl in cols], VA)), {})
        for part in ("k", "v"):  # (a lock counts an op's tasks to 63: 16 heads' drains of K and V^T in two ops)
            s.op({c: at.kv_drains(KA, VA, c, band=b, part=part) for c in range(COLS)}, {})
    s.ddr_sync()
    npu.alloc("acmd", D.Q)
    npu.write("acmd", at.command())
    s.op(fills(npu, "q", [L.panels(qs[sl]) for sl in cols], QA), {})
    for t in range(1 if d == 40 else max(np_ // 8, 1)):  # (an op per tile; d 40: one)
        at = Attention(np=np_, q=QA, o=OA, kv=npu.address("kv"), d=d, nbands=nb, tile=t, rows=ar, raise_max=sink,
                       heads=heads, valid=valid)
        s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address("acmd")))
    nbytes = C // 8 * np_ * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.cat([L.from_panels(ob[c], T, C) for c in range(COLS)])
    img = images(T)
    ref = torch.zeros(COLS * T, C)
    for i in range(2):
        sel = img == i
        kk = torch.cat([q(k[b][sel]) for b in range(nb)])
        vq = torch.cat([torch.cat([q(v[b, sl].T).T for sl in cols])[sel] for b in range(nb)])
        for h in range(heads):
            ch = slice(d * h, d * (h + 1))
            sc = q(qs[sel][:, ch]) @ kk[:, ch].T
            if valid is not None:
                sc[:, ~torch.from_numpy(valid[i])] = -torch.inf
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
def test_conv(npu: VisionDevice, H: int, stride: int, Cin: int, N: int, res: bool) -> None:
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
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(xt[c]) for c in range(COLS)], X), fills(npu, "z", [np.zeros(4096, np.uint8)] * COLS, Z),
               *([fills(npu, "r", [L.ctiles(rt[c], m) for c in range(COLS)], RES)] if res else [])), {})
    s.op({c: g.col_ops(c) for c in range(COLS)}, gemm_rows(npu, "w", g, w, b), barrier=True)
    ybytes = N // 80 * L.ct_bytes(m)
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], m, N, m) for c in range(COLS)]), R, Lg)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), stride=stride, padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    assert _rel(y, ref) < 8e-3


@pytest.mark.parametrize("Cin, N, res", [(320, 320, False), (640, 320, True)])
def test_conv_rows(npu: VisionDevice, Cin: int, N: int, res: bool) -> None:
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
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(xt[c]) for c in range(COLS)], X), fills(npu, "z", [np.zeros(4096, np.uint8)] * COLS, Z),
               *([fills(npu, f"r{k}", [L.ctiles(rt[c, 128 * k:128 * (k + 1)], 64) for c in range(COLS)], RES[k]) for k in range(2)]
                 if res else [])), {})
    for k in range(2):
        g = Gemm(m=64, ntile=2, K=Cin, N=N, x=X, y=Y[k], res=RES[k], out="bf16", conv=1, R=1, Lg=4, zero=Z, x_np=32,
                 rc=rc, r0=2 * k)
        s.op({c: g.col_ops(c) for c in range(COLS)}, gemm_rows(npu, f"w{k}", g, w, b), barrier=True)
    s.op(merge(*(drains(npu, f"y{k}", yb, Y[k]) for k in range(2))), {})
    npu.run(npu.transaction(s))
    ys = [npu.read(f"y{k}", COLS * yb).reshape(COLS, yb) for k in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(ys[k][c], 128, N, 64) for k in range(2)]) for c in range(COLS)]),
                          rc, 4)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    err = _rel(y, ref)
    assert err < 8e-3, err


@pytest.mark.parametrize("np_, d, nb", [(8, 160, 1), (16, 80, 2), (32, 40, 4)])
def test_cross_attention(npu: VisionDevice, np_: int, d: int, nb: int) -> None:
    """Cross-attention on 77 context keys (``layout.kv_slots``, the last group's keys from 77 masked) of the queries of a
    16x16 level (d 160), a 32x32 band (d 80) or a 64x64 band (d 40, role c on every row): within 1.5e-2 of softmax
    attention on the bfp16-rounded operands."""
    torch.manual_seed(4)
    T, C = 8 * np_, 8 * d
    qs = torch.randn(COLS * T, C) * (1.4426950408889634 / d ** 0.5) * 2
    k, v = torch.randn(2, 77, C), torch.randn(2, 77, C)
    QA, OA = 0, C // 8 * np_ * L.PANEL + D.A_OBJ
    cols = [slice(c * T, (c + 1) * T) for c in range(COLS)]
    npu.alloc("ckv", L.kv_slots(k, v, d=d).size)
    npu.write("ckv", L.kv_slots(k, v, d=d))
    ar = tuple(range(ROWS))
    s = npu.seq()
    s.op(fills(npu, "q", [L.panels(qs[sl]) for sl in cols], QA), {})
    for t in range(1 if d == 40 else max(np_ // 8, 1)):
        at = Attention(np=np_, q=QA, o=OA, kv=npu.address("ckv"), nkeys=77, d=d, nbands=nb, tile=t, rows=ar)
        npu.alloc(f"ccmd{t}", D.Q)
        npu.write(f"ccmd{t}", at.command())
        s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address(f"ccmd{t}")))
    nbytes = C // 8 * np_ * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.cat([L.from_panels(ob[c], T, C) for c in range(COLS)])
    img = images(T)
    ref = torch.zeros(COLS * T, C)
    for i in range(2):
        sel = img == i
        kk, vq = q(k[i]), q(F.pad(v[i].T, (0, 3))).T[:77]  # (V^T blocks of 8 keys: the slots' zero keys)
        for h in range(8):
            ch = slice(d * h, d * (h + 1))
            sc = q(qs[sel][:, ch]) @ kk[:, ch].T
            p = torch.exp2(sc - sc.max(-1, keepdim=True).values)
            ref[sel, ch] = (p @ vq[:, ch]) / p.sum(-1, keepdim=True)
    err = _rel(o, ref)
    assert err < 1.5e-2, err


IL = (0, 2, 4, 6, 1, 3, 5, 7)  # (64x64: a row's groups in their storage order, even then odd)


def il_tokens(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``level_tokens`` of a 64x64 level's rows (8 groups): token panel (row, group IL[k], image)."""
    t = level_tokens(x, rows, 8)
    return t.reshape(COLS, rows, 8, 16, -1)[:, :, list(IL)].reshape(t.shape)


def from_il_tokens(y: torch.Tensor, rows: int) -> torch.Tensor:
    inv = [IL.index(g) for g in range(8)]
    return from_level_tokens(y.reshape(COLS, rows, 8, 16, -1)[:, :, inv].reshape(y.shape), rows, 8)


@pytest.mark.parametrize("Cin, N, res", [(320, 320, False), (320, 320, True)])
def test_conv_il(npu: VisionDevice, Cin: int, N: int, res: bool) -> None:
    """A 3x3 conv (padding 1) of a 64x64 level in its row tiles (``ops.Gemm`` il: a row's even groups, its odd ones),
    an op per band of 2 output rows per column, its input's 4 rows (the halos included, zeros past the image) filled
    per band; bias, a residual: within 8e-3 of torch on the bfp16-rounded operands."""
    torch.manual_seed(13)
    H = 64
    x = torch.randn(2, Cin, H, H)
    w, b = torch.randn(N, Cin, 3, 3) / (9 * Cin) ** 0.5, torch.randn(N) * 0.1
    r_img = torch.randn(2, N, H, H)
    xp = F.pad(x, (0, 0, 1, 1))  # (a zero row above and below)
    rt = il_tokens(r_img, 8)  # (COLS, 1024, N): band b = tokens [256 b, ..)
    Z, X = 0, 4096
    yb = N // 80 * 4 * L.ct_bytes(64)  # (a band's C tiles)
    Y = X + Cin // 8 * 64 * L.PANEL
    s = npu.seq()
    for k in range(4):
        xs = [L.panels(il_tokens(torch.cat([xp[:, :, 8 * c + 2 * k:8 * c + 2 * k + 4] for c in range(COLS)], 2), 4)[c])
              for c in range(COLS)]
        s.op(merge(fills(npu, f"x{k}", xs, X), *([fills(npu, f"r{k}", [L.ctiles(rt[c, 256 * k:256 * (k + 1)], 64)
                                                                    for c in range(COLS)], Y)] if res else [])), {})
        g = Gemm(m=64, ntile=4, K=Cin, N=N, x=X, y=Y, res=Y if res else None, out="bf16", conv=1, R=1, Lg=8, zero=Z,
                 x_np=64, rc=4, r0=1, il=True)
        s.op({c: g.col_ops(c) for c in range(COLS)}, gemm_rows(npu, f"w{k}", g, w, b), barrier=True)
        s.op(drains(npu, f"y{k}", yb, Y), {})
    npu.run(npu.transaction(s))
    ys = [npu.read(f"y{k}", COLS * yb).reshape(COLS, yb) for k in range(4)]
    y = from_il_tokens(torch.stack([torch.cat([L.from_ctiles(ys[k][c], 256, N, 64) for k in range(4)]) for c in range(COLS)]),
                       8)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    err = _rel(y, ref)
    assert err < 8e-3, err
