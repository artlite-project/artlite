"""LFM3-Audio's codec decoder (the mmproj's ``a.gen.wav.*``, projector ``lfm3a_gen``) on the vision overlay
(``artlite.vision``; F3's speech encoder's hardware context); the ISTFT of its outputs on the host: ``speech.Istft``.

The network (llama.cpp's ``clip_graph_lfm3a_gen::build_gen_wav``, transformers' ``LiquidCodecModel.decode``): per frame
the FSQ latents of its 12 codes (4 a code: ((idx // basis) % level - level // 2) / (level // 2), levels 8, 7, 6, 6) ->
``quant.first.out_proj`` 48 -> 256 + b -> ``tfm.in_proj`` 256 -> 512 + b, each frame on 6 positions (an STFT hop of 320
samples at 24 kHz each); 8 LFM3 layers of 512 with sandwich norms [RMSNorm -> operator -> RMSNorm, + x | RMSNorm ->
SwiGLU 512 -> 2304 -> 512 -> RMSNorm, + x], the operators short convs (``conv_pw1`` -> [B; C; x], y = C conv7(B x), causal
over 7 positions, ``conv_pw2``) or attention (8 query heads, 2 KV heads of 64, q / k norms, NEOX RoPE of theta 1e6 at
the stream's position, causal over the last 30 positions, each head's output times sigmoid(``attn_gate`` n)); the output
norm and ``tfm.out_proj`` 512 -> 1282 + b: per position [log magnitude 641 | phase 641] (``speech.Istft``).

Widths: the residual's 512 channels in C = 640, the feed-forward's 2304 in 2560 (two GEGLU GEMMs of 1280), a head's 64
dims in 160 (rope2's first pass: pairs (i, 32 + i) at (i, 40 + i), its second pass zero; V's 64 then 96 zeros; O's
columns likewise), each KV head twice (the window attention's 4 rows: row r runs query heads 2 r, 2 r + 1 on its KV
head r // 2), the output's 1282 in 1600. The latents' GEMM takes hi + lo panels (K 96 of 128: the FSQ values exactly).

Layout: a band of 128 positions in tiles of 16 (``vit_level(1, 16)``), column c's token panel j holding positions
64 j + 8 c .. + 8 (``ops.WinAttention``: the columns share the windows: window pair j is panel j of each column). The
codec runs a stream band by band (``Codec``): one band program for every band; what a band leaves the next one (the
state: each conv layer's last 6 rows of B x, each attention layer's last key group) is carried by a dispatch of its own
after a complete band (``carry``). As frames arrive the band in progress runs again with more of its rows (every layer
is causal: the earlier rows are the band's; a dispatch of whole 4-frame blocks ends at a multiple of 8 positions, the
V^T blocks never cut). A band's position is its first row's (the RoPE tables, written per band; the first band's masks
keep no key before the stream).

A band (``Net.band``): the latents' panels from the host -> the two GEMMs -> x0 and its records; per layer: RMSNorm
into panels; a conv layer's C (C tiles), B x (``Gemm(out="prod")``) drained into the layer's DDR rows [6 + 128][640]
(rows 0 .. 5 the state), the strip conv (``ops.StripConv`` mode 1, 7 taps) -> y panels, out_proj -> v; an attention
layer's K / V (each KV head twice) into the window slots, Q (``rope2``), the gate's factors (``Gemm(out="isig")``), the
window attention (``ops.WinAttention``, window 30, gated), O -> v; the post-norm: v's RMSNorm into panels, an identity
GEMM + x (the residual and its records); the feed-forward likewise. The output projection writes fp32 tiles.

Precision (numpy at these rounding points against float64, 3 clips of llama.cpp-sampled codes): PCM SNR 26.5 dB, a
bfloat16 codec's 28.9 (PCM SNR is chaotic per clip: the phases; the log magnitudes' rms error 0.025 vs bf16's 0.017).
"""

from collections.abc import Callable

import numpy as np

from artlite.operators.design import COLS, ROWS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gemm, silu_table
from artlite.vision.ops import ROPE2_TAB, Gemm, StripConv, WinAttention
from artlite.vision.program import Program
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import ln, pad, plain_index, vec, vit_level
from artlite.vision.weights import Wt, cat, eye, take, values

H, C, FFN_REAL, FFN, HEADS, KVH, HD, HP = 512, 640, 2304, 2560, 8, 2, 64, 160
LAYERS = 8  # (the codec's transformer layers)
NBINS, NOUT, OUT_W = 641, 1282, 1600  # (STFT bins; the outputs, padded)
EPS, THETA = 1e-5, 1e6
WINDOW, NTAP = 30, 7
QSCALE = 1.4426950408889634 / 8.0  # (log2 e / sqrt(64): the softmax in log2 units)
LEVELS = (8, 7, 6, 6)
FRAME_STACK = 6
TM, ROWS_BAND = 16, 128
NLAT, KLAT = 96, 128  # (the latents' panels: hi + lo of 48, padded to two k-tiles)
G_ROW = C * 2
G_BYTES = (NTAP - 1 + ROWS_BAND) * G_ROW  # (a conv layer's B x: rows -6 .. -1 (the state), the band's)
OUT_BYTES = OUT_W // 80 * 5760  # (a column's fp32 output tiles)
LAT_BYTES = KLAT // 8 * 2 * L.PANEL  # (a column's latents' panels)
MASK_BYTES = 2 * 128  # (a column's masks: 2 units)


def tokens(c: int) -> np.ndarray:
    """Column c's positions of a band (its 16 tokens: panel j's 8 rows 64 j + 8 c ..)."""
    return (64 * np.arange(2)[:, None] + 8 * c + np.arange(8)[None]).reshape(-1)


def fsq(codes: np.ndarray) -> np.ndarray:
    """The FSQ latents (..., 48) of codes (..., 12): per code its 4."""
    lv = np.array(LEVELS)
    basis = np.cumprod([1, *LEVELS[:-1]])
    c = np.asarray(codes)[..., None]
    return (((c // basis) % lv - lv // 2) / (lv // 2)).reshape(*np.shape(codes)[:-1], -1).astype(np.float32)


# -- weights


def rope2_rows(heads: np.ndarray) -> np.ndarray:
    """Rows of heads of 160 in rope2's layout (pairs (i, 32 + i) of 64 dims at (i, 40 + i), the rest zero) from HF's
    heads ``heads`` of 64 (-1: a pad)."""
    j = np.full(HP, -1)
    j[:32], j[40:72] = np.arange(32), 32 + np.arange(32)
    return np.concatenate([np.where(j >= 0, h * HD + j, -1) for h in heads])


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int, kind: str) -> dict:
    """Layer il's padded weights: "ln1", "ln2", "post1", "post2" (the norms), "gu" (two GEGLU GEMMs' [up | gate]
    rows of 1280), "down"; a conv layer's "bx" ([B | x] rows, each padded to 640), "c", "taps" (640, 7), "out"; an
    attention layer's "q" (rope2 rows), "k", "v" (each KV head twice: KV head r // 2 for row r), "g" (the gate's 8 rows
    padded to 320, times -log2 e), "o" (columns by head: K 1280), "qn" / "kn" (rope2's 160 norm weights, q's times
    log2 e / 8)."""
    p = f"a.gen.wav.tfm.blk.{il}."
    w: dict = {"kind": kind, "ln1": vec(get, p + "ln1.weight", C), "ln2": vec(get, p + "ln2.weight", C),
               "post1": vec(get, p + "attn_post_norm.weight", C), "post2": vec(get, p + "ffn_post_norm.weight", C)}
    up, gt = pad(get(p + "ffn_up.weight"), FFN), pad(get(p + "ffn_gate.weight"), FFN)
    w["gu"] = [cat([up[1280 * k:1280 * (k + 1)], gt[1280 * k:1280 * (k + 1)]]) for k in range(2)]
    w["down"] = pad(get(p + "ffn_down.weight"), C)
    if kind == "conv":
        ip = get(p + "conv_pw1.weight")
        rows = lambda k: take(ip, np.arange(k * H, (k + 1) * H))
        w["bx"] = cat([pad(rows(0), C), pad(rows(2), C)])
        w["c"] = pad(rows(1), C)
        taps = values(get(p + "conv_dw.weight")).astype(np.float32).reshape(H, NTAP)
        w["taps"] = np.concatenate([taps, np.zeros((C - H, NTAP), np.float32)])
        w["out"] = pad(get(p + "conv_pw2.weight"), C)
        return w
    kv = np.repeat(np.arange(KVH), ROWS // KVH)
    w["q"] = take(get(p + "attn_q.weight"), rope2_rows(np.arange(HEADS)))
    w["k"] = take(get(p + "attn_k.weight"), rope2_rows(kv))
    w["v"] = take(get(p + "attn_v.weight"), plain_index(kv, HD, HP))
    w["g"] = pad(get(p + "attn_gate.weight"), 320) * -1.4426950408889634
    w["o"] = pad(take(get(p + "attn_out.weight"), plain_index(np.arange(HEADS), HD, HP), 1), C)
    idx = rope2_rows(np.zeros(1, np.int64))
    for n, s in (("qn", QSCALE), ("kn", 1.0)):
        nw = values(get(p + f"attn_{n[0]}_norm.weight")).astype(np.float32).reshape(-1)
        w[n] = np.where(idx >= 0, nw[np.maximum(idx, 0)], 0).astype(np.float32) * np.float32(s)
    return w


CONV_PROBE = "a.gen.wav.tfm.blk.{il}.conv_dw.weight"  # (a conv layer's tensor: ``vit.kinds_of``)


def weights(get: Callable[[str], Wt | np.ndarray], kinds: tuple[str, ...]) -> dict:
    """Every padded weight of the codec (module docstring) as ``get`` gives them: "q1" (the latents' GEMM: [W | W] for
    the hi + lo panels, rows padded to 320) / "q1_b", "in" / "in_b", per layer its ``layer_weights``, "on" (the output
    norm), "out" / "out_b" (rows padded to 1600), "eye" (C rows of K 512)."""
    w1 = get("a.gen.wav.quant.first.out_proj.weight")
    w: dict = {"kinds": kinds, "q1": pad(cat([w1, w1], 1), 320), "q1_b": vec(get, "a.gen.wav.quant.first.out_proj.bias", 320),
               "in": pad(get("a.gen.wav.tfm.in_proj.weight"), C), "in_b": vec(get, "a.gen.wav.tfm.in_proj.bias", C),
               **{il: layer_weights(get, il, k) for il, k in enumerate(kinds)}}
    w["on"] = vec(get, "a.gen.wav.tfm.output_norm.weight", C)
    w["out"] = pad(get("a.gen.wav.tfm.out_proj.weight"), OUT_W)
    w["out_b"] = vec(get, "a.gen.wav.tfm.out_proj.bias", OUT_W)
    w["eye"] = pad(eye(H), C)
    return w


def rope2_tables(pos: np.ndarray) -> np.ndarray:
    """The rope2 epilogue's tables of 16 tokens at positions pos: [A | B] C-tile layouts of 16 x 80 bf16 (``ROPE2_TAB``
    bytes: A's pairs 0 .. 31 (cos at 0 .., sin at 40 ..), the rest cos 1, sin 0; B the identity)."""
    ang = np.asarray(pos, np.float64)[:, None] * THETA ** (-np.arange(HD // 2, dtype=np.float64) * 2 / HD)
    a, b = np.ones((16, 80), np.float32), np.ones((16, 80), np.float32)
    a[:, 40:], b[:, 40:] = 0, 0
    a[:, :32], a[:, 40:72] = np.cos(ang), np.sin(ang)
    out = np.zeros(ROPE2_TAB, np.uint8)
    out[:2560], out[2560:] = L.ctiles(a, 16)[:2560], L.ctiles(b, 16)[:2560]
    return out


# -- the band program


class Net:
    """The codec's ops on a builder: ``band`` and ``carry`` over the DDR images named by ``p``."""

    def __init__(self, w: dict, p: str = "c"):
        self.w, self.p = w, p
        self.lv = vit_level(1, TM)

    def img(self, bd: Builder, name: str, nbytes: int) -> int:
        return bd.ddr(f"{self.p}.{name}", nbytes)

    def fill(self, bd: Builder, name: str, per_col: int, addr: int, nbytes: int) -> dict[int, ColOps]:
        base = self.img(bd, name, COLS * per_col)
        bd.reading(f"{self.p}.{name}")
        return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(base + c * per_col, nbytes, None))])
                for c in range(COLS)}

    def rms(self, bd: Builder, x: Tensor, n: Tensor, g: np.ndarray, name: str, rec: int) -> None:
        ln(bd, x, n, g, None, name=name, rec=rec, C=C, H=H, eps=EPS, mode="rms")

    def post(self, bd: Builder, v: Tensor, rec: int, g: np.ndarray, x: Tensor, name: str) -> int:
        """x + post_norm(v) (v's records at rec): v's RMSNorm into panels, an identity GEMM + x: x's records."""
        n = alloc(bd, self.lv, C, "pn")
        self.rms(bd, v, n, g, name, rec)
        rec = gemm(bd, Tensor(self.lv, H, n.addr, "pn", pnp=n.np), self.w["eye"], None, name=f"{self.p}.eye", out=x,
                   res=x, ln=True)
        free(bd, n)
        return rec

    def embed(self, bd: Builder) -> tuple[Tensor, int]:
        lv, w = self.lv, self.w
        a = alloc(bd, lv, KLAT, "pn")
        bd.op(self.fill(bd, "lat", LAT_BYTES, a.addr, LAT_BYTES), {})
        e = alloc(bd, lv, 320, "pn")
        gemm(bd, Tensor(lv, NLAT, a.addr, "pn", pnp=a.np), w["q1"], w["q1_b"], name=f"{self.p}.q1", out=e, kind="bfp16")
        free(bd, a)
        sp = bd.mem.alloc(12288, top=True)  # (a residual's last objects read on past it)
        x = alloc(bd, lv, C, top=True)
        bd.mem.release(sp, 12288)
        rec = gemm(bd, Tensor(lv, 256, e.addr, "pn", pnp=e.np), w["in"], w["in_b"], name=f"{self.p}.in", out=x, ln=True)
        free(bd, e)
        return x, rec

    def ffn(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln2"], name + ".ln2", rec)
        silu_table(bd)
        h = alloc(bd, lv, FFN, "pn", pnp=lv.np + 2)  # (m 16 GEGLU: a padded chunk a pass)
        for k in range(2):
            bd.gemm(Gemm(m=TM, ntile=1, K=H, N=FFN, x=n.addr, y=h.addr + 160 * k * h.np * L.PANEL, out="geglu",
                         act="silu", x_np=n.np, y_np=h.np), w["gu"][k], None, name=f"{name}.gu{k}")
        free(bd, n)
        v = alloc(bd, lv, C)
        rec = gemm(bd, Tensor(lv, FFN_REAL, h.addr, "pn", pnp=h.np), w["down"], None, name=name + ".down", out=v, ln=True)
        free(bd, h)
        rec = self.post(bd, v, rec, w["post2"], x, name + ".post2")
        free(bd, v)
        return rec

    def conv(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        gname = f"{self.p}.g{il}"
        ga = bd.ddr(gname, G_BYTES)
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln1"], name + ".ln1", rec)
        nk = Tensor(lv, H, n.addr, "pn", pnp=n.np)
        c = alloc(bd, lv, C)
        gemm(bd, nk, w["c"], None, name=name + ".c", out=c)
        pb = TM * 90  # (the product's passes: [rb 2][5 cb][64] bf16 each)
        npass = 2 * C // 80
        g = bd.mem.alloc(npass * pb)
        bd.gemm(Gemm(m=TM, ntile=1, K=H, N=2 * C, x=n.addr, y=g, out="prod", x_np=n.np), w["bx"], None, name=name + ".bx")
        free(bd, n)
        bd.written(gname)
        # (row block j = panel j's 8 rows 64 j + 8 c ..: per token its 16 passes' 5 x 8 channels -> G row 6 + the row)
        bd.defer({col: ColOps(drains=[MemTask([{"addr": g, "nwords": 8 * C // 2, "dims": [(4, 1), (5, 32), (npass, pb // 4), (0, 4)],
                                                "iteration": (160, 2)}], repeat=2,
                                              ddr=(ga + (NTAP - 1 + 8 * col) * G_ROW, 8 * G_ROW, None, (64 * G_ROW // 4, 2)))])
                  for col in range(COLS)}, release=[(g, npass * pb)])
        bd.flush()
        y = alloc(bd, lv, C, "pn")
        nstep = C // 80 // ROWS
        st = bd.mem.alloc(2 * StripConv.stage1(NTAP, 2))
        bd.reading(gname)
        ops = [StripConv(mode=1, y=y.addr, off=tuple(80 * i for i in range(NTAP)), blocks=tuple(nstep * r + k for r in range(ROWS)),
                         nobj=1, y_np=lv.np) for k in range(nstep)]
        stage = StripConv.stage1(NTAP, 2)

        def fills(k: int) -> dict[int, ColOps]:  # (panel j's strip: G rows 64 j + 8 c .. (6 before its first token))
            return {col: ColOps(fills=[ops[k].fill1(st + (k % 2) * stage, ga + 8 * col * G_ROW, C, 64)]) for col in range(COLS)}
        bd.op(fills(0), {})
        ct = L.ct_bytes(TM)
        for k, op in enumerate(ops):
            img = bd.image(f"{name}.dw{k}", op.image(w["taps"], None))
            cols = {col: ColOps(sends=op.sends1(st + (k % 2) * stage, c.addr, ct), recvs=op.recvs()) for col in range(COLS)}
            bd.op(bd.merge(cols, fills(k + 1)) if k < nstep - 1 else cols, op.row_ops(img))
        bd.mem.release(st, 2 * stage)
        free(bd, c)
        v = alloc(bd, lv, C)
        rec = gemm(bd, Tensor(lv, H, y.addr, "pn", pnp=y.np), w["out"], None, name=name + ".out", out=v, ln=True)
        free(bd, y)
        rec = self.post(bd, v, rec, w["post1"], x, name + ".post1")
        free(bd, v)
        return self.ffn(bd, il, x, rec)

    def attn(self, bd: Builder, il: int, x: Tensor, rec: int, rt: int, mk: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        kvname = f"{self.p}.kv{il}"
        img = WinAttention.kv_image(2)
        kv = bd.ddr(kvname, img)
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln1"], name + ".ln1", rec)
        nk = Tensor(lv, H, n.addr, "pn", pnp=n.np)
        kt, vt = alloc(bd, lv, ROWS * HP, "pn"), alloc(bd, lv, ROWS * HP, "pn")
        bd.gemm(Gemm(m=TM, ntile=1, K=H, N=ROWS * HP, x=n.addr, y=kt.addr, res=rt, out="rope2", qkn=w["kn"], qkn_n=HD,
                     qkn_eps=EPS, x_np=n.np, y_np=kt.np), w["k"], None, name=name + ".k")
        gemm(bd, nk, w["v"], None, name=name + ".v", out=vt, kind="vt")
        at = WinAttention(q=0, o=0, mask=0, kv=kv, units=2, np=lv.np, kv_head_bytes=img // ROWS)
        bd.written(kvname)
        bd.defer({col: at.kv_drains(kt.addr, vt.addr, col, 0, kv) for col in range(COLS)},
                 release=[(kt.addr, kt.size), (vt.addr, vt.size)])
        qt = alloc(bd, lv, HEADS * HP, "pn")
        for p0 in range(0, HEADS * HP // 320, 2):
            bd.gemm(Gemm(m=TM, ntile=1, K=H, N=HEADS * HP, x=n.addr, y=qt.addr, res=rt, out="rope2", qkn=w["qn"], qkn_n=HD,
                         qkn_eps=EPS, x_np=n.np, y_np=qt.np, p0=p0, npo=2), w["q"], None, name=f"{name}.q{p0}")
        gt = alloc(bd, lv, 320)
        gemm(bd, nk, w["g"], None, name=name + ".g", out=gt, kind="isig")
        free(bd, n)
        o = alloc(bd, lv, HEADS * HP, "pn")
        at = WinAttention(q=qt.addr, o=o.addr, mask=mk, kv=kv, units=2, np=lv.np, gate=gt.addr, kv_head_bytes=img // ROWS)
        bd.flush()
        bd.reading(kvname)
        cmd = bd.image(f"{self.p}.win", at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd, 0))
        free(bd, qt)
        free(bd, gt)
        v = alloc(bd, lv, C)
        rec = gemm(bd, o, w["o"], None, name=name + ".o", out=v, ln=True)
        free(bd, o)
        rec = self.post(bd, v, rec, w["post1"], x, name + ".post1")
        free(bd, v)
        return self.ffn(bd, il, x, rec)

    def band(self, bd: Builder) -> None:
        w, lv = self.w, self.lv
        x, rec = self.embed(bd)
        rt = bd.mem.alloc(ROPE2_TAB + D.A_OBJ)  # (each residual object reads A_OBJ bytes)
        mk = bd.mem.alloc(MASK_BYTES + D.A_OBJ // 2)  # (each unit's mask object reads A_OBJ / 2 bytes on)
        bd.op(bd.merge(self.fill(bd, "rope", ROPE2_TAB, rt, ROPE2_TAB), self.fill(bd, "mask", MASK_BYTES, mk, MASK_BYTES)), {})
        for il, kd in enumerate(w["kinds"]):
            rec = self.conv(bd, il, x, rec) if kd == "conv" else self.attn(bd, il, x, rec, rt, mk)
        bd.mem.release(rt, ROPE2_TAB + D.A_OBJ)
        bd.mem.release(mk, MASK_BYTES + D.A_OBJ // 2)
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["on"], f"{self.p}.on", rec)
        free(bd, x)
        y = bd.mem.alloc(OUT_BYTES)
        bd.gemm(Gemm(m=TM, ntile=1, K=H, N=OUT_W, x=n.addr, y=y, out="f32", x_np=n.np), w["out"], w["out_b"],
                name=f"{self.p}.out")
        free(bd, n)
        out = self.img(bd, "y", COLS * OUT_BYTES)
        bd.written(f"{self.p}.y")
        bd.op({c: ColOps(drains=[MemTask([{"addr": y, "nwords": OUT_BYTES // 4}], ddr=(out + c * OUT_BYTES, OUT_BYTES, None))])
               for c in range(COLS)}, {})
        bd.mem.release(y, OUT_BYTES)

    def carried(self) -> list[tuple[str, int, int, int, int]]:
        """What ``carry`` copies: (image, its bytes, from, to, bytes) - each conv layer's last 6 rows of B x into rows
        0 .. 5, each attention layer's unit 2 slot 1 (the band's last group) into unit 0's (every KV head row)."""
        jobs = []
        ub, img = WinAttention.unit_bytes(), WinAttention.kv_image(2)
        for il, kd in enumerate(self.w["kinds"]):
            if kd == "conv":
                jobs.append((f"g{il}", G_BYTES, ROWS_BAND * G_ROW, 0, (NTAP - 1) * G_ROW))
            else:
                jobs += [(f"kv{il}", img, h * img // ROWS + 2 * ub + 2 * D.Q, h * img // ROWS + 2 * D.Q, 2 * D.Q)
                         for h in range(ROWS)]
        return jobs

    def carry(self, bd: Builder) -> None:
        """The state into the next band's place (``carried``), through column 0's memtile."""
        at, fills, drains = 0, [], []
        for name, size, src, dst, nb in self.carried():
            base = self.img(bd, name, size)
            bd.reading(f"{self.p}.{name}")
            fills.append(MemTask([{"addr": at, "nwords": nb // 4}], ddr=(base + src, nb, None)))
            drains.append(MemTask([{"addr": at, "nwords": nb // 4}], ddr=(base + dst, nb, None)))
            bd.written(f"{self.p}.{name}")
            at += -(-nb // 64) * 64
        assert at <= D.HEAP
        for k in range(0, len(fills), 3):
            bd.op({0: ColOps(fills=fills[k:k + 3])}, {})
        for k in range(0, len(drains), 3):
            bd.op({0: ColOps(drains=drains[k:k + 3])}, {})


# -- the host side


def latent_panels(lat: np.ndarray) -> np.ndarray:
    """The latents' panels of a band's positions (DDR image <p>.lat: [column][kb 16][np 2][8][9]): lat (128, 48) per
    position its frame's latents (zero past the frames) -> per column its tokens' [hi | lo | 0] (K 128)."""
    hi = L.unblocks(L.blocks(lat.reshape(ROWS_BAND, -1, 8))).reshape(ROWS_BAND, -1)
    lo = lat - hi
    x = np.zeros((ROWS_BAND, KLAT), np.float32)
    x[:, :48], x[:, 48:96] = hi, lo
    return np.concatenate([L.panels(x[tokens(c)]) for c in range(COLS)])


def rope_image(r0: int) -> np.ndarray:
    """The band's rope2 tables (<p>.rope: [column][ROPE2_TAB]) of rows at positions r0 + their row."""
    return np.concatenate([rope2_tables(r0 + tokens(c)) for c in range(COLS)])


def mask_image(start: bool) -> np.ndarray:
    """The columns' window masks (<p>.mask: [column][unit 2][128 B]); start: a stream's first band."""
    return np.concatenate([WinAttention.masks(c, 2, start, WINDOW).view(np.uint8).reshape(-1) for c in range(COLS)])


class Codec:
    """The codec (module docstring) as dispatches on a VisionDevice: ``band`` and ``carry`` programs. A stream's frames
    (``frames``: their codes as they come) -> the STFT frames they complete; ``reset``: a new stream."""

    def __init__(self, npu: VisionDevice, w: dict, *, name: str = "c", state: dict | None = None):
        self.npu, self.w, self.p = npu, w, name
        self.net = Net(w, name)
        self.bd = Builder(npu, reuse=state is not None)
        self.programs = dict(state["programs"]) if state else {"band": Program.build(npu, self.bd, self.net.band),
                                                               "carry": Program.build(npu, self.bd, self.net.carry)}
        self.band_tr, self.carry_tr = (self.programs[k].transaction(npu) for k in ("band", "carry"))
        self.reset()

    def state(self) -> dict:
        """What a restored device's codec takes besides its images: the programs."""
        return {"programs": dict(self.programs)}

    def reset(self) -> None:
        """A new stream: its state zero (the conv layers' rows -6 .. -1; the first band's masks keep no key before it)."""
        for il, kd in enumerate(self.w["kinds"]):
            if kd == "conv":
                self.npu.alloc(f"{self.p}.g{il}", G_BYTES)
                self.npu.write(f"{self.p}.g{il}", np.zeros((NTAP - 1) * G_ROW, np.uint8))
        self.band, self.rows = 0, 0  # (the band in progress, the stream's positions so far)
        self.lat = np.zeros((ROWS_BAND, 48), np.float32)  # (the band in progress's latents)
        self._carry, self._fresh = False, True

    def _run(self, upto: int) -> np.ndarray:
        """The band in progress with its rows up to ``upto`` (of the band): their STFT frames (rows, 1282)."""
        if self._carry:
            self.npu.run(self.carry_tr)
            self._carry = False
        if self._fresh:  # (the band's first run: its tables)
            self.npu.write(f"{self.p}.rope", rope_image(ROWS_BAND * self.band))
            self.npu.write(f"{self.p}.mask", mask_image(self.band == 0))
            self._fresh = False
        self.npu.write(f"{self.p}.lat", latent_panels(self.lat))
        self.npu.run(self.band_tr)
        raw = self.npu.read(f"{self.p}.y", COLS * OUT_BYTES).reshape(COLS, OUT_W // 80, 5760)[..., :5120].copy()
        t = raw.view(np.float32).reshape(COLS, OUT_W // 80, 2, 10, 8, 8)  # ([col][pass][rb][cb][tok][ch])
        rows = np.ascontiguousarray(t.transpose(0, 2, 4, 1, 3, 5)).reshape(COLS, TM, OUT_W)
        out = np.empty((ROWS_BAND, OUT_W), np.float32)
        for c in range(COLS):
            out[tokens(c)] = rows[c]
        return out[:upto, :NOUT]

    def frames(self, codes: np.ndarray) -> np.ndarray:
        """The STFT frames (6 n, 1282) [log magnitude | phase] float32 of the stream's next n frames (codes (n, 12))."""
        lat = np.repeat(fsq(np.asarray(codes).reshape(-1, 12)), FRAME_STACK, 0)
        out, i = [], 0
        while i < len(lat):
            r0 = self.rows - ROWS_BAND * self.band  # (the band's rows so far)
            n = min(len(lat) - i, ROWS_BAND - r0)
            self.lat[r0:r0 + n] = lat[i:i + n]
            out.append(self._run(r0 + n)[r0:])
            self.rows += n
            i += n
            if self.rows == ROWS_BAND * (self.band + 1):  # (a complete band: its state carried before the next)
                self.band += 1
                self.lat[:] = 0
                self._carry, self._fresh = True, True
        return np.concatenate(out) if out else np.zeros((0, NOUT), np.float32)
