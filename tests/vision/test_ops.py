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
from artlite.vision.weights import Slots, as_wt

from ..numerics import rel_native
from .ops_ref import from_il_tokens, from_level_tokens, il_tokens, level_tokens, q, qconv_w

COLS, ROWS = 8, 4


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
    """Row images of a Gemm (per row its command, then its weights: ``weights.Slots``) in buffer ``name``; returns its
    RowOps by row."""
    per = D.Q + g.npo * g.nkt * D.Q
    img = np.zeros(ROWS * per, np.uint8)
    for r in range(ROWS):
        img[r * per:r * per + D.Q] = g.command(b, r)
    s = Slots()
    s.add(g, as_wt(w), {r: ("w", r * per + D.Q) for r in range(ROWS)})
    s.pack_to({"w": img})
    npu.alloc(name, ROWS * per)
    npu.write(name, img)
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
    unpack = {"bfp16": lambda b: torch.from_numpy(L.from_panels(b, T, N)), "vt": lambda b: torch.from_numpy(L.from_vt_panels(b, T, N)),
              "bf16": lambda b: torch.from_numpy(L.from_ctiles(b, T, N, m))}[out]
    y = torch.cat([unpack(yb[c]) for c in range(COLS)])
    ref = q(x) @ q(w).T + b.bfloat16().float() + (r.bfloat16().float() if res else 0)
    assert rel_native(y, ref) < 8e-3


def test_gemm_rope(npu: VisionDevice) -> None:
    """Q / K of 16 heads of 80 (``Gemm(out="rope")``): each head's pairs of channels (k, 40 + k) turned by its token's
    cos | sin (the residual tile, the same for every pass), as panels: within 8e-3 of the turned bf16 outputs of the
    bfp16-rounded operands' GEMM."""
    torch.manual_seed(1)
    m, K, N = 64, 1280, 1280
    x = torch.randn(COLS * m, K)
    w, b = torch.randn(N, K) / K ** 0.5, torch.randn(N) * 0.1
    ang = torch.rand(COLS * m, 40) * 20
    cs = torch.cat([torch.cos(ang), torch.sin(ang)], 1)  # (per token [cos 40 | sin 40])
    X, RES = 0, K // 8 * m // 8 * L.PANEL
    Y = RES + L.ct_bytes(m)
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, res=RES, out="rope")
    ybytes = N // 8 * m // 8 * L.PANEL
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X),
               fills(npu, "r", [L.ctiles(cs[c * m:(c + 1) * m], m) for c in range(COLS)], RES)), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([torch.from_numpy(L.from_panels(yb[c], m, N)) for c in range(COLS)])
    c0 = (q(x) @ q(w).T + b.bfloat16().float()).bfloat16().float().reshape(-1, 16, 2, 40)
    c, sn = (t.bfloat16().float()[:, None] for t in (cs[:, :40], cs[:, 40:]))
    ref = torch.stack([c0[:, :, 0] * c - c0[:, :, 1] * sn, c0[:, :, 1] * c + c0[:, :, 0] * sn], 2).reshape(-1, N)
    assert rel_native(y, ref) < 8e-3


@pytest.mark.parametrize("nb, frac, heads, kvh", [(5, 0.83, 16, 0), (2, 1.0, 16, 0), (9, 0.5, 16, 0), (2, 1.0, 16, 8),
                                                 (3, 0.7, 12, 0)])
def test_attention_dense(npu: VisionDevice, nb: int, frac: float, heads: int, kvh: int) -> None:
    """Joint self-attention of 16 (or 12) heads of 80 over more than 32 key groups (``Attention(dense=True)``): both
    images' keys of nb bands one sequence, about its first ``frac`` valid (the last live group's keys in a pattern; the
    rest streamed, skipped or masked), each column's band-0 queries over them; kvh: grouped-query attention (query head h
    on key / value head h mod kvh: LFM3-VL's): within 1.5e-2 of softmax attention on the bfp16-rounded operands."""
    torch.manual_seed(5)
    np_, d = 8, 80
    kvh = kvh or heads
    T, C, Ck = 8 * np_, heads * d, kvh * d
    qs = torch.randn(COLS * T, C) * (1.4426950408889634 / d ** 0.5) * 2
    k, v = torch.randn(2, nb, COLS * T, Ck)
    QA = 0
    KA = QA + C // 8 * np_ * L.PANEL + D.A_OBJ
    VA = KA + Ck // 8 * np_ * L.PANEL
    OA = VA + Ck // 8 * np_ * L.PANEL
    cols = [slice(c * T, (c + 1) * T) for c in range(COLS)]
    kw = {"np": np_, "q": QA, "o": OA, "d": d, "nbands": nb, "heads": heads, "joint": True, "dense": True,
          "kv_heads": kvh if kvh < heads else 0}
    nkeys = 2 * 4 * COLS * T * nb // 8  # (the sequence: image 0's keys of every band, then image 1's)
    nvalid = int(nkeys * frac)
    ok = np.arange(nkeys) < nvalid
    ok[nvalid // 64 * 64:nvalid] &= np.arange(nvalid % 64) % 3 != 1  # (the last live group: a pattern)
    npu.alloc("kv", Attention(kv=0, **kw).kv_bytes())
    at = Attention(kv=npu.address("kv"), valid=(ok, ok), **kw)
    s = npu.seq()
    for b in range(nb):
        s.op(merge(fills(npu, f"k{b}", [L.panels(k[b, sl]) for sl in cols], KA),
                   fills(npu, f"v{b}", [L.vt_panels(v[b, sl]) for sl in cols], VA)), {})
        for part in ("k", "v"):
            s.op({c: at.kv_drains(KA, VA, c, band=b, part=part) for c in range(COLS)}, {})
    s.ddr_sync()
    npu.alloc("acmd", D.Q)
    npu.write("acmd", at.command())
    s.op(fills(npu, "q", [L.panels(qs[sl]) for sl in cols], QA), {})
    s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address("acmd")))
    nbytes = C // 8 * np_ * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.cat([torch.from_numpy(L.from_panels(ob[c], T, C)) for c in range(COLS)])
    img = images(T)
    sel = torch.from_numpy(ok)
    kk = torch.cat([q(k[b][img == i]) for i in range(2) for b in range(nb)])[sel]
    vq = torch.cat([torch.cat([q(v[b, sl].T).T for sl in cols])[img == i] for i in range(2) for b in range(nb)])[sel]
    ref = torch.zeros(COLS * T, C)
    for h in range(heads):
        ch, ck = slice(d * h, d * (h + 1)), slice(d * (h % kvh), d * (h % kvh + 1))
        sc = q(qs[:, ch]) @ kk[:, ck].T
        p = torch.exp2(sc - sc.max(-1, keepdim=True).values)
        ref[:, ch] = (p @ vq[:, ck]) / p.sum(-1, keepdim=True)
    assert rel_native(o, ref) < 1.5e-2


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
    y = torch.cat([torch.from_numpy(L.from_panels(out[c], g.y_np * 8, N // 2))[:T] for c in range(COLS)])
    u = q(x) @ q(w).T + b.bfloat16().float()
    hid, gate = u.chunk(2, -1)
    ref = hid.bfloat16().float() * F.gelu(gate.bfloat16().float(), approximate="tanh")
    assert rel_native(y, ref) < 1.2e-2


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
    y = torch.cat([torch.from_numpy(L.from_panels(out[c], g.y_np * 8, N))[:T] for c in range(COLS)])
    u = q(x) @ q(w).T + b.bfloat16().float()
    ref = F.gelu(u.bfloat16().float(), approximate="tanh")
    assert rel_native(y, ref) < 1.2e-2


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
    y = torch.cat([torch.from_numpy(L.from_ctiles(yb[c], T, C, m)) for c in range(COLS)])
    zb = npu.read("z", COLS * nz).reshape(COLS, nz)
    z = torch.cat([torch.from_numpy(L.from_panels(zb[c], T, C)) for c in range(COLS)])
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
    assert rel_native(z, ref) < 9e-3


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
    o = torch.cat([torch.from_numpy(L.from_panels(ob[c], T, C)) for c in range(COLS)])
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
    assert rel_native(o, ref) < 1.5e-2


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
    y = from_level_tokens(torch.stack([torch.from_numpy(L.from_ctiles(yb[c], m, N, m)) for c in range(COLS)]), R, Lg)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), stride=stride, padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    assert rel_native(y, ref) < 8e-3


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
    y = from_level_tokens(torch.stack([torch.cat([torch.from_numpy(L.from_ctiles(ys[k][c], 128, N, 64)) for k in range(2)]) for c in range(COLS)]),
                          rc, 4)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    err = rel_native(y, ref)
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
    o = torch.cat([torch.from_numpy(L.from_panels(ob[c], T, C)) for c in range(COLS)])
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
    err = rel_native(o, ref)
    assert err < 1.5e-2, err


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
    y = from_il_tokens(torch.stack([torch.cat([torch.from_numpy(L.from_ctiles(ys[k][c], 256, N, 64)) for k in range(4)]) for c in range(COLS)]),
                       8)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w), b.bfloat16().float(), padding=1)
    if res:
        ref = ref + r_img.bfloat16().float()
    err = rel_native(y, ref)
    assert err < 8e-3, err


def test_norm_rms(npu: VisionDevice) -> None:
    """RMSNorm (``Apply(mode="rms")``: LayerNorm's records, the mean taken as 0, gamma only) of a GEMM's output over its
    first 1024 of 1280 channels (the rest zero: LFM3-VL's padded residual): within 9e-3 of torch."""
    torch.manual_seed(21)
    m, C, H = 64, 1280, 1024
    x = torch.randn(COLS * m, 320)
    w = torch.randn(C, 320) / 320 ** 0.5
    w[H:] = 0
    gamma = 1 + 0.3 * torch.randn(C)
    X, Y = 0, 320 // 8 * m // 8 * L.PANEL
    REC = Y + C // 80 * L.ct_bytes(m)
    Z = REC + 16 * D.OCH
    g = Gemm(m=m, ntile=1, K=320, N=C, x=X, y=Y, out="bf16", stats="ln", rec=REC)
    a = Apply(m=m, nblk=C // 320, mode="rms", x=Y, y=Z, rec=REC, C=H)
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, None))
    npu.alloc("ap", ROWS * 2 * D.Q)
    npu.write("ap", np.concatenate([x for r in range(ROWS) for x in (a.command(r), a.gamma_beta(gamma, None, r))]))
    s.op({c: a.col_ops() for c in range(COLS)}, {r: a.row_ops(npu.address("ap", 2 * r * D.Q)) for r in range(ROWS)})
    nz, ny = C // 8 * m // 8 * L.PANEL, C // 80 * L.ct_bytes(m)
    s.op(merge(drains(npu, "y", ny, Y), drains(npu, "z", nz, Z)), {})
    npu.run(npu.transaction(s))
    y = torch.cat([torch.from_numpy(L.from_ctiles(b, m, C, m)) for b in npu.read("y", COLS * ny).reshape(COLS, -1)])
    z = torch.cat([torch.from_numpy(L.from_panels(b, m, C)) for b in npu.read("z", COLS * nz).reshape(COLS, -1)])
    ref = y / torch.sqrt((y[:, :H] ** 2).mean(-1, keepdim=True) + 1e-5) * gamma
    assert rel_native(z[:, :H], ref[:, :H]) < 9e-3 and float(z[:, H:].abs().max()) == 0


@pytest.mark.parametrize("out", ["swiglu", "prod"])
def test_gated(npu: VisionDevice, out: str) -> None:
    """The gated pair epilogues of role d (the weight's rows [hid | gate], per pass [hid 40 | gate 40]): SwiGLU (``Gemm(
    out="geglu", act="silu")``: hid silu(gate) as panels, SiLU's own table) and the product (``out="prod"``: hid gate as
    bf16, 5760 B a pass): within 1.2e-2 (panels) / 8e-3 (bf16) of torch on the bfp16-rounded operands."""
    torch.manual_seed(22)
    m, K, N = 64, 1024, 2560
    x = torch.randn(COLS * m, K)
    w = torch.randn(N, K) / K ** 0.5
    X, Y = 0, K // 8 * m // 8 * L.PANEL
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, out="geglu" if out == "swiglu" else "prod", act="silu" if out == "swiglu"
             else "gelu")
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X), {})
    if out == "swiglu":
        s.op({}, table_rows(npu, "tab", np.tile(kmath.SILU.table_words(), 2).view(np.uint8)))
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, None))
    ybytes = N // 16 * g.y_np * L.PANEL if out == "swiglu" else N // 80 * 5760
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    ob = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    u = (q(x) @ q(w).T).bfloat16().float()
    hid, gate = u.chunk(2, -1)
    if out == "swiglu":
        y = torch.cat([torch.from_numpy(L.from_panels(ob[c], m, N // 2)) for c in range(COLS)])
        assert rel_native(y, hid * F.silu(gate)) < 1.2e-2
    else:  # ([pass 32][rb 8][cb 5][8 tokens][8 ch] bf16 per column, 640 B of padding a pass)
        v = torch.from_numpy(ob.reshape(COLS, N // 80, 5760)[:, :, :5120].copy()).view(torch.bfloat16).float()
        y = v.reshape(COLS, N // 80, 8, 5, 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(COLS * m, N // 2)
        assert rel_native(y, (hid * gate).bfloat16().float()) < 8e-3


def test_gemm_rope_qkn(npu: VisionDevice) -> None:
    """The RoPE epilogue with LFM3-VL's q / k norm (``Gemm(qkn=)``): each head of 80 RMS-normed over 64 channels (its
    pads zero), times the 80 weights, then turned: within 8e-3 of torch on the bfp16-rounded operands' bf16 outputs."""
    torch.manual_seed(23)
    m, K, N = 64, 1024, 1280
    x = torch.randn(COLS * m, K)
    w = torch.randn(N, K) / K ** 0.5
    pads = torch.tensor([j % 40 >= 32 for j in range(80)])
    w.reshape(16, 80, K)[:, pads] = 0
    nw = (1 + 0.3 * torch.randn(80)) * ~pads
    ang = torch.rand(COLS * m, 40) * 20
    cs = torch.cat([torch.cos(ang), torch.sin(ang)], 1)
    X, RES = 0, K // 8 * m // 8 * L.PANEL
    Y = RES + L.ct_bytes(m)
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, res=RES, out="rope", qkn=nw.numpy(), qkn_n=64, qkn_eps=1e-5)
    ybytes = N // 8 * m // 8 * L.PANEL
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X),
               fills(npu, "r", [L.ctiles(cs[c * m:(c + 1) * m], m) for c in range(COLS)], RES)), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, None))
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([torch.from_numpy(L.from_panels(yb[c], m, N)) for c in range(COLS)])
    c0 = (q(x) @ q(w).T).bfloat16().float().reshape(-1, 16, 80)
    c0 = c0 / torch.sqrt((c0 * c0).sum(-1, keepdim=True) / 64 + 1e-5) * nw.bfloat16().float()
    c0 = c0.reshape(-1, 16, 2, 40)
    c, sn = (t.bfloat16().float()[:, None] for t in (cs[:, :40], cs[:, 40:]))
    ref = torch.stack([c0[:, :, 0] * c - c0[:, :, 1] * sn, c0[:, :, 1] * c + c0[:, :, 0] * sn], 2).reshape(-1, N)
    assert rel_native(y, ref) < 8e-3


@pytest.mark.parametrize("gh, gw, nb, b", [(6, 20, 1, 0), (13, 17, 2, 1), (5, 3, 1, 0), (2, 70, 2, 0)])
def test_dwconv(npu: VisionDevice, gh: int, gw: int, nb: int, b: int) -> None:
    """LFM3-VL's depthwise 3 x 3 conv (``ops.DwConv``, 4 steps of an 80-channel block per row) of band b of a gh x gw
    window grid: the gated input from DDR ([window][class][1280] bf16: strips of 10 windows per class, the rows above /
    below gw windows away), the gate C from a band's C tiles, the image's edge masks: y = C conv(g) within 8e-3 of torch
    on the bf16 inputs (the bfp16 output's rounding), the tokens of real windows."""
    from ml_dtypes import bfloat16

    from artlite.vision import windows as Wn
    from artlite.vision.ops import DwConv
    torch.manual_seed(24)
    C, W = 1280, nb * 128
    g = torch.randn(W, 4, C).bfloat16().float()
    cg = torch.randn(nb, COLS, 64, C).bfloat16().float()  # (the band's tiles' C: token 8 (2 q + p) + j)
    taps = torch.randn(C, 3, 3) / 3
    gd = np.zeros((W + 2 * Wn.MARGIN, 4, C), bfloat16)
    gd[Wn.MARGIN:Wn.MARGIN + W] = g.numpy().astype(bfloat16)
    npu.alloc("g", Wn.conv_ddr_bytes(nb))
    npu.write("g", gd.view(np.uint8))
    ct = L.ct_bytes(64)
    npu.alloc("c", COLS * 16 * ct)
    npu.write("c", np.concatenate([L.ctiles(cg[b, c], 64) for c in range(COLS)]))
    masks = Wn.dw_masks(gh, gw, nb, b)
    npu.alloc("m", COLS * DwConv.MASK_BYTES)
    npu.write("m", masks.view(np.uint8))
    STAGE, Y = 0, DwConv.STAGE_BYTES
    s = npu.seq()
    for k in range(4):
        op = DwConv(stage=STAGE, y=Y, k=k)
        cols = {}
        for c in range(COLS):
            fl = []
            for p in range(2):
                for part in ("mid", "up", "down"):
                    off = Wn.conv_strip(nb, b, c, p, part, k) + Wn.conv_shift(part, gw)
                    fl.append(op.strip_fill(p, part, npu.address("g", off)))
            fl.append(MemTask([op.c_fill()], ddr=op.c_ddr(npu.address("c", c * 16 * ct), k)))
            fl.append(MemTask([op.mask_fill()], ddr=(npu.address("m", c * DwConv.MASK_BYTES), DwConv.MASK_BYTES, None)))
            cols[c] = ColOps(fills=fl)
        s.op(cols, {})
        img = np.concatenate([np.concatenate([op.command(), DwConv.taps(taps, 4 * r + k)]) for r in range(ROWS)])
        npu.alloc(f"t{k}", ROWS * 2 * D.Q)
        npu.write(f"t{k}", img)
        s.op({c: op.col_ops() for c in range(COLS)}, op.row_ops(npu.address(f"t{k}")))
    nbytes = C // 8 * 8 * L.PANEL
    s.op(drains(npu, "y", nbytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * nbytes).reshape(COLS, nbytes)
    got = torch.stack([torch.from_numpy(L.from_panels(yb[c], 64, C)) for c in range(COLS)])  # (column, token 8 (2 q + p) + j, C)
    wq = taps.bfloat16().float()
    ref, sel = torch.zeros_like(got), torch.zeros(COLS, 64, dtype=torch.bool)
    for c in range(COLS):
        for p in range(2):
            for j, w in enumerate(Wn.windows_of(nb, b, c, p)):
                if w >= gh * gw:
                    continue
                wy, wx = divmod(int(w), gw)
                for qq in range(4):
                    py, px = divmod(qq, 2)
                    acc = torch.zeros(C)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            y, x = 2 * wy + py + dy, 2 * wx + px + dx
                            if 0 <= y < 2 * gh and 0 <= x < 2 * gw:
                                acc += wq[:, dy + 1, dx + 1] * g[(y // 2) * gw + x // 2, (y % 2) * 2 + x % 2]
                    t = 8 * (2 * qq + p) + j
                    ref[c, t] = (acc.bfloat16().float() * cg[b, c, t]).bfloat16().float()
                    sel[c, t] = True
    assert rel_native(got[sel], ref[sel]) < 8e-3


def rope2_layout() -> np.ndarray:
    """LFM3-Audio's head of 96 in two passes of 80 (``ops.Gemm(out="rope2")``): per channel of the 160 its dim (-1: a
    pad): pass A pairs (i, 48 + i) at (i, 40 + i) for i < 40, pass B pairs 40 + i at (i, 40 + i) for i < 8."""
    out = -np.ones(160, np.int64)
    for i in range(40):
        out[i], out[40 + i] = i, 48 + i
    for i in range(8):
        out[80 + i], out[120 + i] = 40 + i, 88 + i
    return out


@pytest.mark.parametrize("N", [1280, 640])
def test_gemm_rope2(npu: VisionDevice, N: int) -> None:
    """The two-pass RoPE epilogue (``Gemm(out="rope2")``, m 16): each head of 96 dims in 160 channels RMS-normed over
    both passes, times its 160 weights, its pairs turned by the tiles' tables A and B: within 8e-3 of torch on the
    bfp16-rounded operands' bf16 outputs."""
    from artlite.vision.ops import ROPE2_TAB
    torch.manual_seed(31)
    m, nt, K = 16, 4, 768
    T = m * nt
    lay = rope2_layout()
    pads = torch.from_numpy(lay < 0)
    x = torch.randn(COLS * T, K)
    w = torch.randn(N, K) / K ** 0.5
    w.reshape(N // 160, 160, K)[:, pads] = 0
    nw = (1 + 0.3 * torch.randn(160)) * ~pads
    ang = torch.rand(COLS * T, 48) * 20
    ca, cb_ = torch.ones(COLS * T, 80), torch.ones(COLS * T, 80)
    ca[:, :40], ca[:, 40:] = torch.cos(ang[:, :40]), torch.sin(ang[:, :40])
    cb_[:, 40:] = 0
    cb_[:, :8], cb_[:, 40:48] = torch.cos(ang[:, 40:]), torch.sin(ang[:, 40:])
    X = 0
    RES = K // 8 * T // 8 * L.PANEL
    Y = RES + nt * ROPE2_TAB + D.A_OBJ
    tabs = []
    for c in range(COLS):
        t = np.zeros((nt, ROPE2_TAB), np.uint8)
        for i in range(nt):
            r = slice(c * T + m * i, c * T + m * (i + 1))
            t[i, :2560], t[i, 2560:] = L.ctiles(ca[r], m)[:2560], L.ctiles(cb_[r], m)[:2560]
        tabs.append(t.reshape(-1))
    ybytes = N // 8 * T // 8 * L.PANEL
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(x[c * T:(c + 1) * T]) for c in range(COLS)], X), fills(npu, "r", tabs, RES)), {})
    for t in range(nt):  # (an op per tile and head: a head's two passes in one op, a task of 2 BDs a send)
        for p0 in range(0, N // 320, 2):
            g = Gemm(m=m, ntile=1, K=K, N=N, x=X + 2 * t * L.PANEL, y=Y + 2 * t * L.PANEL, res=RES + t * ROPE2_TAB,
                     out="rope2", qkn=nw.numpy(), qkn_n=96, qkn_eps=1e-5, p0=p0, npo=2, x_np=T // 8, y_np=T // 8)
            s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, f"w{p0}", g, w, None))
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([torch.from_numpy(L.from_panels(yb[c], T, N)) for c in range(COLS)])
    c0 = (q(x) @ q(w).T).bfloat16().float().reshape(-1, N // 160, 160)
    c0 = c0 / torch.sqrt((c0 * c0).sum(-1, keepdim=True) / 96 + 1e-5) * nw.bfloat16().float()
    ref = torch.empty_like(c0)
    for p, tab in ((0, ca), (1, cb_)):
        h = c0[..., 80 * p:80 * p + 80].reshape(-1, N // 160, 2, 40)
        c, sn = (t.bfloat16().float()[:, None] for t in (tab[:, :40], tab[:, 40:]))
        ref[..., 80 * p:80 * p + 80] = torch.stack([h[:, :, 0] * c - h[:, :, 1] * sn, h[:, :, 1] * c + h[:, :, 0] * sn],
                                                   2).reshape(-1, N // 160, 80)
    assert rel_native(y, ref.reshape(-1, N)) < 8e-3 and float(y.reshape(-1, N // 160, 160)[..., pads].abs().max()) == 0


def test_gemm_silu(npu: VisionDevice) -> None:
    """The SiLU epilogue of role r (``Gemm(out="gelu", act="silu")``): silu of the outputs (+ bias) as panels, SiLU's own
    table: within 1.2e-2 of torch on the bfp16-rounded operands."""
    torch.manual_seed(32)
    m, K, N = 64, 256, 320
    x = torch.randn(COLS * m, K)
    w = torch.randn(N, K) / K ** 0.5
    b = torch.randn(N) * 0.3
    X, Y = 0, K // 8 * m // 8 * L.PANEL
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, out="gelu", act="silu")
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X), {})
    s.op({}, table_rows(npu, "tab", np.tile(kmath.SILU.table_words(), 2).view(np.uint8)))
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    ybytes = N // 8 * m // 8 * L.PANEL
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([torch.from_numpy(L.from_panels(yb[c], m, N)) for c in range(COLS)])
    u = (q(x) @ q(w).T + b.bfloat16().float()).bfloat16().float()
    assert rel_native(y, F.silu(u)) < 1.2e-2


@pytest.mark.parametrize("k, ntap, m", [(0, 3, 64), (2, 3, 64), (0, 7, 16), (1, 7, 16)])
def test_strip_conv1(npu: VisionDevice, k: int, ntap: int, m: int) -> None:
    """The causal 1-D conv of the strip conv op (``ops.StripConv`` mode 1: LFM3-Audio's short convs, the encoder's of 3
    taps and 960 channels on tiles of 64, its codec's of 7 taps and 640 on tiles of 16), step k (row r's block nstep r +
    k) of a band in the interleaved layout (column c's token panel j: rows 64 j + 8 c ..): the strips of B x from a DDR
    image [ntap - 1 + rows][C] bf16 (the rows before: the state), C from the memtile's C tiles: y = C conv as torch's on
    the bf16 inputs (fp32 sums of the taps, bf16 products), rounded to the output's bfp16 blocks (within 1e-3: the sums'
    order)."""
    from ml_dtypes import bfloat16

    from artlite.vision.ops import StripConv
    torch.manual_seed(33)
    C, T = (960, 512) if m == 64 else (640, 128)
    nstep, h = C // 80 // ROWS, ntap - 1
    g = torch.randn(h + T, C).bfloat16().float()
    cg = torch.randn(T, C).bfloat16().float()
    taps = torch.randn(C, ntap) / 2
    order = torch.tensor([64 * j + 8 * c + i for c in range(COLS) for j in range(m // 8) for i in range(8)]).reshape(COLS, m)
    npu.alloc("g", g.numel() * 2)
    npu.write("g", g.numpy().astype(bfloat16).view(np.uint8))
    ct = L.ct_bytes(m)
    STAGE, CT = 0, StripConv.stage1(ntap, m // 8)
    Y = CT + C // 80 * ct
    op = StripConv(mode=1, y=Y, off=tuple(80 * i for i in range(ntap)), blocks=tuple(nstep * r + k for r in range(ROWS)),
                   nobj=m // 16, y_np=m // 8)
    s = npu.seq()
    cols = fills(npu, "c", [L.ctiles(cg[order[c]], m) for c in range(COLS)], CT)
    for c in range(COLS):
        cols[c].fills.append(op.fill1(STAGE, npu.address("g", 8 * c * C * 2), C, 64))
    s.op(cols, {})
    npu.alloc("t", ROWS * 2 * D.Q)
    npu.write("t", op.image(taps.numpy(), None))
    s.op({c: ColOps(sends=op.sends1(STAGE, CT, ct), recvs=op.recvs()) for c in range(COLS)}, op.row_ops(npu.address("t")))
    nbytes = C // 8 * m // 8 * L.PANEL
    s.op(drains(npu, "y", nbytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * nbytes).reshape(COLS, nbytes)
    got = torch.stack([torch.from_numpy(L.from_panels(yb[c], m, C)) for c in range(COLS)])
    wq = taps.bfloat16().float()
    conv = sum(wq[:, j] * g[j:j + T] for j in range(ntap)).bfloat16().float()
    ref = (conv * cg).bfloat16().float()[order]
    blk = torch.zeros(C, dtype=torch.bool)
    for r in range(ROWS):
        blk[80 * (nstep * r + k):80 * (nstep * r + k) + 80] = True
    assert rel_native(got[..., blk], q(ref)[..., blk]) < 1e-3


@pytest.mark.parametrize("start", [True, False])
def test_win_attention(npu: VisionDevice, start: bool) -> None:
    """Sliding-window attention of window pairs (``ops.WinAttention``: LFM3-Audio's encoder, 8 query heads on 4 KV heads
    of 96 in 160, a window of 32) on a band in the interleaved layout: K / V^T drained into the layer's image (each group
    into its two slots), the previous dispatch's last group in unit 0 (start: the stream's first rows, masked); each
    query over the keys t - 31 .. t: within 1.5e-2 of softmax attention on the bfp16-rounded operands."""
    from artlite.vision.ops import WinAttention
    torch.manual_seed(34)
    d, T, H, KVH = 160, 512, 8, 4
    pads = torch.arange(d) >= 96
    qs = torch.randn(T, H, d) * (1.4426950408889634 / 96 ** 0.5) * 2
    k, v = torch.randn(2, 32 + T, KVH, d)  # (rows -32 .. -1: the previous group)
    qs[..., pads], k[..., pads], v[..., pads] = 0, 0, 0
    order = torch.tensor([64 * j + 8 * c + i for c in range(COLS) for j in range(8) for i in range(8)]).reshape(COLS, 64)
    QA = 0
    KA = QA + H * d // 8 * 8 * L.PANEL + D.A_OBJ
    VA = KA + KVH * d // 8 * 8 * L.PANEL
    MA = VA + KVH * d // 8 * 8 * L.PANEL
    OA = MA + 1024 + D.A_OBJ // 2
    img = WinAttention.kv_image(8)
    npu.alloc("kv", img)
    npu.write("kv", np.zeros(img, np.uint8))
    at = WinAttention(q=QA, o=OA, mask=MA, kv=npu.address("kv"), kv_head_bytes=img // ROWS)
    st = L.kv_slots(torch.cat([k[None, :32], torch.zeros(1, 32, 8 - KVH, d)], 2).reshape(1, 32, -1).repeat(2, 1, 1),
                    torch.cat([v[None, :32], torch.zeros(1, 32, 8 - KVH, d)], 2).reshape(1, 32, -1).repeat(2, 1, 1))
    st = st.reshape(2, 8, 2 * D.Q)[0]
    for h in range(KVH):  # (the previous dispatch's last group: unit 0, slot 1)
        npu.write("kv", st[h], h * img // ROWS + 2 * D.Q)
    s = npu.seq()
    rows = [order[c] for c in range(COLS)]
    s.op(merge(fills(npu, "k", [L.panels(k[32:][r].reshape(64, -1)) for r in rows], KA),
               fills(npu, "v", [L.vt_panels(v[32:][r].reshape(64, -1)) for r in rows], VA),
               fills(npu, "q", [L.panels(qs[r].reshape(64, -1)) for r in rows], QA),
               fills(npu, "m", [WinAttention.masks(c, 8, start).view(np.uint8).reshape(-1) for c in range(COLS)], MA)), {})
    s.op({c: at.kv_drains(KA, VA, c, 0, npu.address("kv")) for c in range(COLS)}, {})
    s.ddr_sync()
    npu.alloc("acmd", D.Q)
    npu.write("acmd", at.command())
    s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address("acmd"), 0))
    nbytes = H * d // 8 * 8 * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.zeros(T, H * d)
    for c in range(COLS):
        o[order[c]] = torch.from_numpy(L.from_panels(ob[c], 64, H * d))
    kq = q(k.reshape(32 + T, -1)).reshape(32 + T, KVH, d)
    vq = torch.cat([q(v.reshape(32 + T, -1)[i:i + 8].T).T for i in range(0, 32 + T, 8)]).reshape(32 + T, KVH, d)
    ref = torch.zeros(T, H, d)
    qq = q(qs.reshape(T, -1)).reshape(T, H, d)
    for t in range(T):
        lo = max(t - 31, 0 if start else -32) + 32
        for h in range(H):
            sc = kq[lo:t + 33, h // 2] @ qq[t, h]
            p = torch.exp2(sc - sc.max())
            ref[t, h] = (p @ vq[lo:t + 33, h // 2]) / p.sum()
    assert rel_native(o, ref.reshape(T, -1)) < 1.5e-2


@pytest.mark.parametrize("lead", ["t", "f"])
def test_strip_conv2(npu: VisionDevice, lead: str) -> None:
    """The stride-2 3 x 3 depthwise conv of the strip conv op (``ops.StripConv`` mode 2: LFM3-Audio's downsampler), SiLU
    of its input first, + bias, on a grid [1 + rows][1 + freqs][256] bf16 (row / column 0: the zero padding): per column
    8 output panels (lead "t": 8 times at one frequency each, "f": 8 frequencies at one time): as torch's on the bf16
    inputs (SiLU's table within 1 bf16 ULP), rounded to the output's bfp16 blocks (within 4e-3)."""
    from ml_dtypes import bfloat16

    from artlite.vision.ops import StripConv
    torch.manual_seed(35)
    C, R, NF = 256, 136, 64  # (input rows / freqs: outputs 68 x 32)
    x = torch.zeros(1 + R, 1 + NF, C)
    x[1:, 1:] = (torch.randn(R, NF, C) * 2).bfloat16().float()
    taps, bias = torch.randn(C, 3, 3) / 3, torch.randn(C) * 0.3
    npu.alloc("s", (x.numel() + 4096) * 2)
    npu.write("s", np.concatenate([x.numpy().reshape(-1), np.zeros(4096, np.float32)]).astype(bfloat16).view(np.uint8))
    pos, line = C // 2, (1 + NF) * C // 2  # (words: a position, a time row)
    if lead == "t":  # (column c: frequency 2 c + 1, panels p = 0..7: times 8 p .. at row 16 p - 1)
        outs = [[(8 * p + j, 2 * c + 1) for p in range(8) for j in range(8)] for c in range(COLS)]
        src = [npu.address("s", 4 * (2 * (2 * c + 1) * pos)) for c in range(COLS)]  # (row 0 = time -1, freq 2 f - 1)
        kw = {"pos": line, "line": pos, "step": 16 * line}
    else:  # (column c: time c, panels p = 0..3 x 2: frequencies 8 p .., two rows: times c and c + 8)
        outs = [[(c + 8 * (p // 4), 8 * (p % 4) + j) for p in range(8) for j in range(8)] for c in range(COLS)]
        src = [npu.address("s", 4 * (2 * c * line)) for c in range(COLS)]
        kw = None
    STAGE, Y = 0, StripConv.STAGE2
    op = StripConv(mode=2, y=Y, off=StripConv.taps2(1 if lead == "t" else 0), nobj=8, silu=True)  # (taps [dy][dx]: dy the time)
    s = npu.seq()
    s.op({}, table_rows(npu, "tab", np.tile(kmath.SILU.table_words(), 2).view(np.uint8)))
    cols = {}
    for c in range(COLS):
        if lead == "t":
            fl = [op.fill2(STAGE, src[c], r, **kw) for r in range(ROWS)]
        else:  # (the objects: 4 frequency panels 16 positions apart, then the next time 8 rows on: two fills of 4)
            fl = []
            for r in range(ROWS):
                for h in range(2):
                    t = op.fill2(STAGE + h * 4 * D.A_OBJ, src[c] + 4 * h * 16 * line, r, pos=pos, line=line, step=16 * pos)
                    t.repeat = 4
                    t.chain[0]["iteration"] = (D.A_OBJ // 4, 4)
                    t.ddr = (*t.ddr[:3], (16 * pos, 4))
                    fl.append(t)
        cols[c] = ColOps(fills=fl)
    s.op(cols, {})
    npu.alloc("t", ROWS * 2 * D.Q)
    npu.write("t", op.image(taps.reshape(C, 9).numpy(), bias.numpy()))
    s.op({c: ColOps(sends=op.sends2(STAGE), recvs=op.recvs()) for c in range(COLS)}, op.row_ops(npu.address("t")))
    nbytes = 320 // 8 * 8 * L.PANEL
    s.op(drains(npu, "y", nbytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * nbytes).reshape(COLS, nbytes)
    got = torch.stack([torch.from_numpy(L.from_panels(yb[c], 64, 320))[:, :C] for c in range(COLS)])
    xs = F.silu(x).bfloat16().float()
    wq, bq = taps.bfloat16().float(), bias.bfloat16().float()
    ref = torch.zeros_like(got)
    for c in range(COLS):
        for i, (to, fo) in enumerate(outs[c]):
            ref[c, i] = (sum(wq[:, dy, dx] * xs[2 * to + dy, 2 * fo + dx] for dy in range(3) for dx in range(3)) + bq)
    ref = ref.bfloat16().float()
    assert rel_native(got, q(ref)) < 4e-3


def test_gemm_rope40(npu: VisionDevice) -> None:
    """The two-head RoPE epilogue of role e (``Gemm(out="rope40")``, m 16: LFM3-Audio's depthformer): per pass two heads
    of 32 dims in 40 channels, each RMS-normed over its 32 times its weights (the pads zero), its NEOX pairs (k, 16 + k)
    turned by the tile's table: within 8e-3 of torch on the bfp16-rounded operands' bf16 outputs."""
    from artlite.vision.ops import ROPE40_TAB
    torch.manual_seed(36)
    m, K, N = 16, 1024, 1280
    pads = torch.arange(40) >= 32
    x = torch.randn(COLS * m, K)
    w = torch.randn(N, K) / K ** 0.5
    w.reshape(N // 40, 40, K)[:, pads] = 0
    nw = (1 + 0.3 * torch.randn(40)) * ~pads
    ang = torch.rand(COLS * m, 16) * 20
    tab = torch.zeros(COLS * m, 80)
    tab[:, :16], tab[:, 16:32] = torch.cos(ang), torch.sin(ang)
    X, RES = 0, K // 8 * m // 8 * L.PANEL
    Y = RES + ROPE40_TAB + D.A_OBJ
    tabs = [L.ctiles(tab[c * m:(c + 1) * m], m)[:ROPE40_TAB] for c in range(COLS)]
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, res=RES, out="rope40", qkn=torch.cat([nw, nw]).numpy(), qkn_n=32,
             qkn_eps=1e-5)
    s = npu.seq()
    s.op(merge(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X), fills(npu, "r", tabs, RES)), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, None))
    ybytes = N // 8 * m // 8 * L.PANEL
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    y = torch.cat([torch.from_numpy(L.from_panels(yb[c], m, N)) for c in range(COLS)])
    c0 = (q(x) @ q(w).T).bfloat16().float().reshape(-1, N // 40, 40)
    c0 = c0 / torch.sqrt((c0 * c0).sum(-1, keepdim=True) / 32 + 1e-5) * nw.bfloat16().float()
    c, sn = (t.bfloat16().float()[:, None] for t in (tab[:, :16], tab[:, 16:32]))
    a, b = c0[..., :16], c0[..., 16:32]
    ref = torch.cat([a * c - b * sn, b * c + a * sn, c0[..., 32:]], -1)
    assert rel_native(y, ref.reshape(-1, N)) < 8e-3 and float(y.reshape(-1, N // 40, 40)[..., pads].abs().max()) == 0


@pytest.mark.parametrize("out", ["f32", "isig"])
def test_gemm_epilogue_e(npu: VisionDevice, out: str) -> None:
    """Role e's other epilogues (m 16): "f32" the fp32 C tile (+ bias) out as it is (within 1e-6 of torch on the
    bfp16-rounded operands: the sums' order); "isig" 1 + 2^y as bf16 C tiles (LFM3's gate factors; y up to +-90:
    saturating past +-63.5, within 1e-2 of torch below)."""
    torch.manual_seed(37)
    m, K, N = 16, 512, 640
    x = torch.randn(COLS * m, K)
    w = torch.randn(N, K) / K ** 0.5 * (1 if out == "f32" else 20)
    b = torch.randn(N) * 0.3
    X, Y = 0, K // 8 * m // 8 * L.PANEL
    g = Gemm(m=m, ntile=1, K=K, N=N, x=X, y=Y, out=out)
    s = npu.seq()
    s.op(fills(npu, "x", [L.panels(x[c * m:(c + 1) * m]) for c in range(COLS)], X), {})
    s.op({c: g.col_ops() for c in range(COLS)}, gemm_rows(npu, "w", g, w, b))
    ybytes = N // 80 * (5760 if out == "f32" else L.ct_bytes(m))
    s.op(drains(npu, "y", ybytes, Y), {})
    npu.run(npu.transaction(s))
    yb = npu.read("y", COLS * ybytes).reshape(COLS, ybytes)
    u = q(x) @ q(w).T + b.bfloat16().float()
    if out == "f32":
        t = yb.reshape(COLS, N // 80, 5760)[..., :5120].copy().view(np.float32).reshape(COLS, N // 80, 2, 10, 8, 8)
        y = torch.from_numpy(np.ascontiguousarray(t.transpose(0, 2, 4, 1, 3, 5)).reshape(COLS * m, N))
        assert float((y - u).abs().max()) <= 1e-6 * float(u.abs().max())
    else:
        y = torch.cat([torch.from_numpy(L.from_ctiles(yb[c], m, N, m)) for c in range(COLS)])
        ref = 1 + torch.exp2(u.clamp(max=63.5))
        ok = u.abs() < 60
        assert float(u.abs().max()) > 64 and rel_native(y[ok], ref[ok]) < 1e-2 and bool(torch.isfinite(y).all())


def test_depth_attention(npu: VisionDevice) -> None:
    """The depthformer's attention on role c (``ops.DepthAttention``): 32 query heads on 8 KV heads of 32 dims in 40, the
    keys column 0's first 12 tokens (drained into every unit's slot, unit u: KV head u % 8), every column's 16 queries
    against them, each output over its head's gate factor 1 + e^-g (a bf16 C tile: times sigmoid(g)): within 1.5e-2 of
    softmax attention on the bfp16-rounded operands."""
    from artlite.vision.ops import DepthAttention
    torch.manual_seed(38)
    m, H, KVH, d, nk = 16, 32, 8, 40, 12
    pads = torch.arange(d) >= 32
    qs = torch.randn(COLS * m, H, d) * (1.4426950408889634 / 32 ** 0.5) * 2
    k, v = torch.randn(2, m, KVH, d)
    qs[..., pads], k[..., pads], v[..., pads] = 0, 0, 0
    k[nk:], v[nk:] = 0, 0
    gl = torch.randn(COLS * m, H) * 2
    fac = torch.zeros(COLS * m, 80)
    fac[:, :H] = 1 + torch.exp(-gl)
    QA, KA = 0, H * d // 8 * 2 * L.PANEL + D.A_OBJ
    VA = KA + KVH * d // 8 * 2 * L.PANEL
    GA = VA + KVH * d // 8 * 2 * L.PANEL
    OA = GA + L.ct_bytes(m)
    JA = OA + H * d // 8 * 2 * L.PANEL
    npu.alloc("kv", DepthAttention.kv_bytes())
    npu.write("kv", np.zeros(DepthAttention.kv_bytes(), np.uint8))
    at = DepthAttention(q=QA, o=OA, kv=npu.address("kv"), gate=GA, junk=JA, nk=nk)
    s = npu.seq()
    s.op(merge(fills(npu, "k", [L.panels(k.reshape(m, -1))] * COLS, KA),
               fills(npu, "v", [L.vt_panels(v.reshape(m, -1))] * COLS, VA),
               fills(npu, "q", [L.panels(qs[c * m:(c + 1) * m].reshape(m, -1)) for c in range(COLS)], QA),
               fills(npu, "g", [L.ctiles(fac[c * m:(c + 1) * m], m) for c in range(COLS)], GA)), {})
    for dr in at.kv_drains(KA, VA):
        s.op(dr, {})
    s.ddr_sync()
    npu.alloc("acmd", D.Q)
    npu.write("acmd", at.command())
    s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address("acmd")))
    nbytes = H * d // 8 * 2 * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.cat([torch.from_numpy(L.from_panels(ob[c], m, H * d)) for c in range(COLS)]).reshape(-1, H, d)
    kq = q(k.reshape(m, -1)).reshape(m, KVH, d)[:nk]
    vq = q(v.reshape(m, -1)[:8].T).T.reshape(8, KVH, d)
    vq = torch.cat([vq, q(v.reshape(m, -1)[8:].T).T.reshape(8, KVH, d)])[:nk]
    qq = q(qs.reshape(COLS * m, -1)).reshape(-1, H, d)
    sc = torch.einsum("thd,khd->thk", qq, kq[:, torch.arange(H) % KVH])
    p = torch.exp2(sc - sc.amax(-1, keepdim=True))
    ref = torch.einsum("thk,khd->thd", p, vq[:, torch.arange(H) % KVH]) / p.sum(-1, keepdim=True) / \
        fac[:, :H].bfloat16().float()[..., None]
    assert rel_native(o, ref) < 1.5e-2 and float(o[..., pads].abs().max()) == 0


@pytest.mark.parametrize("start", [True, False])
def test_win_attention_gated(npu: VisionDevice, start: bool) -> None:
    """The gated window attention (``ops.WinAttention`` with ``gate``, role g: LFM3-Audio's codec decoder, 8 query heads on
    2 KV heads of 64 in 160 (each KV head twice: the rows' 4), a window of 30) on a band of 128 rows in tiles of 16 (2
    units): each query over the keys t - 29 .. t, over its head's gate factor 1 + e^-g (a bf16 C tile: times sigmoid(g)):
    within 1.5e-2 of softmax attention on the bfp16-rounded operands."""
    from artlite.vision.ops import WinAttention
    torch.manual_seed(39)
    d, T, H, KVH, W, m = 160, 128, 8, 4, 30, 16
    pads = torch.arange(d) >= 64
    qs = torch.randn(T, H, d) * (1.4426950408889634 / 64 ** 0.5) * 2
    k2, v2 = torch.randn(2, 32 + T, 2, d)
    k, v = k2.repeat_interleave(2, 1), v2.repeat_interleave(2, 1)  # (KV head r // 2 for row r)
    qs[..., pads], k[..., pads], v[..., pads] = 0, 0, 0
    gl = torch.randn(T, H) * 2
    fac = torch.zeros(T, 80)
    fac[:, :H] = 1 + torch.exp(-gl)
    order = torch.tensor([64 * j + 8 * c + i for c in range(COLS) for j in range(m // 8) for i in range(8)]).reshape(COLS, m)
    QA = 0
    KA = QA + H * d // 8 * 2 * L.PANEL + D.A_OBJ
    VA = KA + KVH * d // 8 * 2 * L.PANEL
    MA = VA + KVH * d // 8 * 2 * L.PANEL
    GA = MA + 256 + D.A_OBJ // 2
    OA = GA + L.ct_bytes(m)
    img = WinAttention.kv_image(2)
    npu.alloc("kv", img)
    npu.write("kv", np.zeros(img, np.uint8))
    at = WinAttention(q=QA, o=OA, mask=MA, kv=npu.address("kv"), units=2, np=2, gate=GA, kv_head_bytes=img // ROWS)
    st = L.kv_slots(torch.cat([k[None, :32], torch.zeros(1, 32, 8 - KVH, d)], 2).reshape(1, 32, -1).repeat(2, 1, 1),
                    torch.cat([v[None, :32], torch.zeros(1, 32, 8 - KVH, d)], 2).reshape(1, 32, -1).repeat(2, 1, 1))
    st = st.reshape(2, 8, 2 * D.Q)[0]
    for h in range(KVH):  # (the previous dispatch's last group: unit 0, slot 1)
        npu.write("kv", st[h], h * img // ROWS + 2 * D.Q)
    s = npu.seq()
    rows = [order[c] for c in range(COLS)]
    s.op(merge(fills(npu, "k", [L.panels(k[32:][r].reshape(m, -1)) for r in rows], KA),
               fills(npu, "v", [L.vt_panels(v[32:][r].reshape(m, -1)) for r in rows], VA),
               fills(npu, "q", [L.panels(qs[r].reshape(m, -1)) for r in rows], QA),
               fills(npu, "m", [WinAttention.masks(c, 2, start, W).view(np.uint8).reshape(-1) for c in range(COLS)], MA),
               fills(npu, "g", [L.ctiles(fac[r], m) for r in rows], GA)), {})
    s.op({c: at.kv_drains(KA, VA, c, 0, npu.address("kv")) for c in range(COLS)}, {})
    s.ddr_sync()
    npu.alloc("acmd", D.Q)
    npu.write("acmd", at.command())
    s.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(npu.address("acmd"), 0))
    nbytes = H * d // 8 * 2 * L.PANEL
    s.op(drains(npu, "o", nbytes, OA), {})
    npu.run(npu.transaction(s))
    ob = npu.read("o", COLS * nbytes).reshape(COLS, nbytes)
    o = torch.zeros(T, H * d)
    for c in range(COLS):
        o[order[c]] = torch.from_numpy(L.from_panels(ob[c], m, H * d))
    kq = q(k.reshape(32 + T, -1)).reshape(32 + T, KVH, d)
    vq = torch.cat([q(v.reshape(32 + T, -1)[i:i + 8].T).T for i in range(0, 32 + T, 8)]).reshape(32 + T, KVH, d)
    ref = torch.zeros(T, H, d)
    qq = q(qs.reshape(T, -1)).reshape(T, H, d)
    for t in range(T):
        lo = max(t - W + 1, 0 if start else -32) + 32
        for h in range(H):
            sc = kq[lo:t + 33, h // 2] @ qq[t, h]
            p = torch.exp2(sc - sc.max())
            ref[t, h] = (p @ vq[lo:t + 33, h // 2]) / p.sum() / fac[t, h].bfloat16().float()
    assert rel_native(o, ref.reshape(T, -1)) < 1.5e-2
