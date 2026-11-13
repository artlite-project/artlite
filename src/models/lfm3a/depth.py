"""LFM3-Audio's depthformer (the mmproj's ``a.gen.code.*``, projector ``lfm3a_gen``) on the vision overlay
(``artlite.vision``: one design for every network, its ops built by host-built transactions; F3's speech encoder's
hardware context).

The network (llama.cpp's ``clip_graph_lfm3a_gen::build_gen_code``, transformers' ``Lfm3AudioDepthDecoder``): a frame's
12 codebooks scored from the language model's hidden state h (2048, after its output norm) and the codes revealed so far:
slot cb = ``proj_in``(h)[1024 cb ..] + b plus its code's embedding (``embd`` codebook cb's row) where revealed, else
``mask_embd``; 3 bidirectional layers [RMSNorm -> attention (32 query heads, 8 KV heads of 32, q / k RMSNorm per head,
NEOX RoPE of theta 1e6 at the slots' positions 0 .. 11, each head's output times sigmoid(``attn_gate`` n)) | RMSNorm ->
SwiGLU 1024 -> 2816 -> 1024]; the output norm, the codebook's ``head_norm`` and its logits embd[cb] x (the head tied to
the embeddings). A frame runs ``speech.SCHEDULE``'s 3 passes on one hidden state, each scoring the codebooks it samples.

Widths: the residual's 1024 channels in C = 1280 (passes of 80 on 4 core rows; the pads zero), a head's 32 dims in 40
(the rope40 epilogue's two heads a pass: ``ops.Gemm(out="rope40")``; V's and O's likewise), the feed-forward's 2816 in
2880 (two GEGLU GEMMs of 1440: a command's biases hold 14 passes), a codebook's 2017 logits in 2240. The query heads in
``ops.DepthAttention.order`` (Q's rows, O's columns, the gates'), so that query head u reads KV head u mod 8.

Layout: a tile of 16 tokens per column (``vit_level(1, 16)``); the 12 slots are column 0's tokens 0 .. 11 (the other
columns' tokens zero: every column runs the same ops, their outputs unused). Two programs (dispatches):
- ``proj``: the hidden state's panels (column 0's token 0, from the host) through ``proj_in`` (4 GEMMs of 3840: each
  slot's 1024 rows padded to 1280) + b, column 0's token 0 drained to DDR as P [15360] bf16 (a gather of its C tiles);
- ``pass``: x0 = P + E (the host's: E the revealed codebooks' embedding rows and ``mask_embd``) as C tiles, the layers,
  the output norm (RMSNorm into panels, an identity GEMM back to C tiles with their records), then for each of the
  pass's 4 codebooks its head_norm (an RMS apply of those records) and its head (a GEMM of fp32 outputs: ``Gemm(
  out="f32")``), column 0's tiles drained. The program is built for codebooks 0 .. 3 and every codebook's head and norm
  have images of one layout: a pass rewrites the program's shim addresses of them for its codebooks
  (``VisionDevice.resolve``).
The attention (``ops.DepthAttention``): K and V^T of column 0's tokens drained into every unit's slot of the layer's DDR
image, 12 keys valid; the gate a GEMM of the factors 1 / sigmoid (``Gemm(out="isig")``, weights times -log2 e).

Precision (numpy at these rounding points against float64, 24 passes of llama.cpp-sampled frames): the per-codebook KL
~2.25x llama.cpp's own (its Q8_0 activations), from the bfp16 weights and GEMM inputs about equally; bf16 logits would
add ~40% (fp32 out instead); hi + lo weights everywhere 1.9x at twice the weights' stream.
"""

import math
from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from artlite.operators.design import COLS
from artlite.overlay import Transaction
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gemm, ln_rec, silu_table
from artlite.vision.ops import ROPE40_TAB, Apply, DepthAttention, Gemm
from artlite.vision.program import Program
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import ln, ln_records, pad, plain_index, vec, vit_level
from artlite.vision.weights import Wt, cat, eye, take, values

H, C, HD, HP, HEADS, KVH, FFN_REAL, FFN = 1024, 1280, 32, 40, 32, 8, 2816, 2880
LAYERS = 3  # (the depthformer's layers)
NCB, NCODES, NLOG = 12, 2017, 2240  # (codebooks, codes a codebook, its logits padded)
PROJ, PROJ_OP = NCB * C, 3840  # (proj_in's outputs: a slot's 1280; per GEMM)
NPICK = 4  # (codebooks a pass scores)
EPS, THETA = 1e-5, 1e6
LOG2E = 1.4426950408889634
QSCALE = LOG2E / math.sqrt(HD)  # (the queries' scale: the softmax in log2 units)
TM = 16  # (a column's tile)
LOG_BYTES = NLOG // 80 * 5760  # (a head's fp32 tiles of a column: [pass][rb][cb][64] in 5760 B each)


# -- weights


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int) -> dict:
    """Layer il's padded weights: "ln1", "ln2"; "q" (the NPU's head order), "k", "v" (heads of 40), "g" (the gate's 32
    rows in the head order, padded to 320, times -log2 e), "o" (columns by head: K 1280), "qn" / "kn" (rope40's 80 norm
    weights: a head's 40 twice, q's times log2 e / sqrt(32)); "gu" (the two GEGLU GEMMs' [up | gate] rows of 1440 each),
    "down" (K 2816)."""
    p = f"a.gen.code.blk.{il}."
    order = DepthAttention.order()
    w: dict = {"ln1": vec(get, p + "ln1.weight", C), "ln2": vec(get, p + "ln2.weight", C)}
    w["q"] = take(get(p + "attn_q.weight"), plain_index(order, HD, HP))
    w["k"] = take(get(p + "attn_k.weight"), plain_index(np.arange(KVH), HD, HP))
    w["v"] = take(get(p + "attn_v.weight"), plain_index(np.arange(KVH), HD, HP))
    w["g"] = pad(take(get(p + "attn_gate.weight"), order), 320) * -LOG2E
    w["o"] = pad(take(get(p + "attn_out.weight"), plain_index(order, HD, HP), 1), C)
    for n, s in (("qn", QSCALE), ("kn", 1.0)):
        nw = np.zeros(HP, np.float32)
        nw[:HD] = values(get(p + f"attn_{n[0]}_norm.weight")).reshape(-1) * np.float32(s)
        w[n] = np.concatenate([nw, nw])
    up, gt = pad(get(p + "ffn_up.weight"), FFN), pad(get(p + "ffn_gate.weight"), FFN)
    w["gu"] = [cat([up[1440 * k:1440 * (k + 1)], gt[1440 * k:1440 * (k + 1)]]) for k in range(2)]
    w["down"] = pad(get(p + "ffn_down.weight"), C)
    return w


def weights(get: Callable[[str], Wt | np.ndarray], layers: int) -> dict:
    """Every padded weight of the depthformer (module docstring) as ``get`` gives them (views of the file): "proj" /
    "proj_b" (each slot's rows padded to 1280), per layer its ``layer_weights``, "on" (the output norm), "hn" (the
    head norms (12, 1280)), "heads" (per codebook its embedding rows padded to 2240), "eye" (C rows of K 1024)."""
    rows = np.array([s * H + i if i < H else -1 for s in range(NCB) for i in range(C)])
    w: dict = {"layers": layers, "proj": take(get("a.gen.code.proj_in.weight"), rows),
               "proj_b": take(values(get("a.gen.code.proj_in.bias")).reshape(-1), rows),
               **{il: layer_weights(get, il) for il in range(layers)}}
    w["on"] = vec(get, "a.gen.code.output_norm.weight", C)
    w["hn"] = np.stack([pad(r, C) for r in values(get("a.gen.code.head_norm.weight")).reshape(NCB, H)])
    emb = get("a.gen.code.embd.weight")
    w["heads"] = [pad(emb[NCODES * cb:NCODES * (cb + 1)], NLOG) for cb in range(NCB)]
    w["eye"] = pad(eye(H), C)
    return w


def rope_table() -> np.ndarray:
    """The rope40 epilogue's table of a column's tile (``ROPE40_TAB`` bytes: [rb][cb][64] bf16, cos in cb 0, 1 and sin
    in cb 2, 3 of pairs 0 .. 15) at the tokens' positions 0 .. 15 (the slots')."""
    ang = np.arange(TM, dtype=np.float64)[:, None] * THETA ** (-np.arange(HD // 2, dtype=np.float64) * 2 / HD)
    t = np.zeros((TM, 80), np.float32)
    t[:, :16], t[:, 16:32] = np.cos(ang), np.sin(ang)
    return L.ctiles(t, TM)[:ROPE40_TAB]


# -- the programs


class Net:
    """The depthformer's ops on a builder: ``proj`` and ``pass_`` (module docstring) over the DDR images named by
    ``p``."""

    def __init__(self, w: dict, p: str = "d"):
        self.w, self.p = w, p
        self.lv = vit_level(1, TM)

    def img(self, bd: Builder, name: str, nbytes: int) -> int:
        return bd.ddr(f"{self.p}.{name}", nbytes)

    def fill(self, bd: Builder, name: str, addr: int, nbytes: int) -> dict[int, ColOps]:
        """Fills of DDR image ``name`` ([col][nbytes]) into the memtile at addr."""
        base = self.img(bd, name, COLS * nbytes)
        bd.reading(f"{self.p}.{name}")
        return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(base + c * nbytes, nbytes, None))])
                for c in range(COLS)}

    def proj(self, bd: Builder) -> None:
        lv, w = self.lv, self.w
        hp = alloc(bd, lv, 2048, "pn")
        bd.op(self.fill(bd, "h", hp.addr, hp.nbytes), {})
        pt = alloc(bd, lv, PROJ_OP)
        out = self.img(bd, "p", 2 * PROJ)
        nblk = PROJ_OP // 80
        for k in range(PROJ // PROJ_OP):
            sl = slice(PROJ_OP * k, PROJ_OP * (k + 1))
            gemm(bd, hp, w["proj"][sl], w["proj_b"][sl], name=f"{self.p}.proj{k}", out=pt)
            bd.written(f"{self.p}.p")  # (column 0's token 0: per block of 80 its 10 runs of 8 channels -> P in order)
            bd.op({0: ColOps(drains=[MemTask([{"addr": pt.addr, "nwords": 40 * nblk,
                                               "dims": [(4, 1), (10, 32), (nblk, L.ct_bytes(TM) // 4)]}],
                                             ddr=(out + 2 * PROJ_OP * k, 2 * PROJ_OP, None))])}, {})
        free(bd, pt)
        free(bd, hp)

    def layer(self, bd: Builder, il: int, x: Tensor, rec: int, rt: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        n = alloc(bd, lv, C, "pn")
        ln(bd, x, n, w["ln1"], None, name=name + ".ln1", rec=rec, C=C, H=H, eps=EPS, mode="rms")
        nk = Tensor(lv, H, n.addr, "pn", pnp=n.np)
        q = alloc(bd, lv, HEADS * HP, "pn")
        kt, vt = alloc(bd, lv, KVH * HP, "pn"), alloc(bd, lv, KVH * HP, "pn")
        gt = alloc(bd, lv, 320)
        for wn, t, qkn in (("q", q, w["qn"]), ("k", kt, w["kn"])):
            bd.gemm(Gemm(m=TM, ntile=1, K=H, N=t.C, x=n.addr, y=t.addr, res=rt, out="rope40", qkn=qkn, qkn_n=HD,
                         qkn_eps=EPS, x_np=n.np, y_np=t.np), w[wn], None, name=f"{name}.{wn}")
        gemm(bd, nk, w["v"], None, name=name + ".v", out=vt, kind="vt")
        gemm(bd, nk, w["g"], None, name=name + ".g", out=gt, kind="isig")
        free(bd, n)
        kvname = f"{self.p}.kv{il}"
        kv = bd.ddr(kvname, DepthAttention.kv_bytes())
        o = alloc(bd, lv, HEADS * HP, "pn")
        junk = bd.mem.alloc(D.OCH)
        at = DepthAttention(q=q.addr, o=o.addr, kv=kv, gate=gt.addr, junk=junk)
        bd.written(kvname)
        for dr in at.kv_drains(kt.addr, vt.addr):
            bd.op(dr, {})
        bd.reading(kvname)
        cmd = bd.image(f"{self.p}.attn", at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
        bd.mem.release(junk, D.OCH)
        for t in (q, kt, vt, gt):
            free(bd, t)
        rec = gemm(bd, o, w["o"], None, name=name + ".o", out=x, res=x, ln=True)
        free(bd, o)
        n = alloc(bd, lv, C, "pn")
        ln(bd, x, n, w["ln2"], None, name=name + ".ln2", rec=rec, C=C, H=H, eps=EPS, mode="rms")
        silu_table(bd)
        h = alloc(bd, lv, FFN, "pn", pnp=lv.np + 2)  # (m 16 GEGLU: a padded chunk a pass)
        for k in range(2):
            bd.gemm(Gemm(m=TM, ntile=1, K=H, N=FFN, x=n.addr, y=h.addr + 180 * k * h.np * L.PANEL, out="geglu",
                         act="silu", x_np=n.np, y_np=h.np), w["gu"][k], None, name=f"{name}.gu{k}")
        free(bd, n)
        rec = gemm(bd, Tensor(lv, FFN_REAL, h.addr, "pn", pnp=h.np), w["down"], None, name=name + ".down", out=x,
                   res=x, ln=True)
        free(bd, h)
        return rec

    def pass_(self, bd: Builder, cbs: tuple[int, ...] = (0, 1, 2, 3)) -> None:
        """A pass scoring codebooks ``cbs`` (the program's: ``Depth`` rewrites them per pass)."""
        lv, w = self.lv, self.w
        sp = bd.mem.alloc(12288, top=True)  # (a residual's last objects read on past it)
        x = alloc(bd, lv, C, top=True)
        bd.mem.release(sp, 12288)
        rt = bd.mem.alloc(ROPE40_TAB + D.A_OBJ)  # (each residual object reads A_OBJ bytes)
        bd.op(bd.merge(self.fill(bd, "x0", x.addr, x.nbytes), self.fill(bd, "rope", rt, ROPE40_TAB)), {})
        rec = ln_records(bd, x, name=f"{self.p}.rec0", C=C)
        for il in range(w["layers"]):
            rec = self.layer(bd, il, x, rec, rt)
        bd.mem.release(rt, ROPE40_TAB + D.A_OBJ)
        n = alloc(bd, lv, C, "pn")
        ln(bd, x, n, w["on"], None, name=f"{self.p}.on", rec=rec, C=C, H=H, eps=EPS, mode="rms")
        free(bd, x)
        t = alloc(bd, lv, C)
        rec = gemm(bd, Tensor(lv, H, n.addr, "pn", pnp=n.np), w["eye"], None, name=f"{self.p}.eye", out=t, ln=True)
        free(bd, n)
        lg = bd.mem.alloc(LOG_BYTES)
        for j, cb in enumerate(cbs):
            hn = alloc(bd, lv, C, "pn")
            self.head(bd, cb, t, hn, lg, rec)
            out = self.img(bd, f"lg{j}", LOG_BYTES)
            bd.written(f"{self.p}.lg{j}")
            bd.defer({0: ColOps(drains=[MemTask([{"addr": lg, "nwords": LOG_BYTES // 4}], ddr=(out, LOG_BYTES, None))])})
            free(bd, hn)
        bd.flush()
        bd.mem.release(lg, LOG_BYTES)
        bd.mem.release(rec, ln_rec(lv))
        free(bd, t)

    def head(self, bd: Builder, cb: int, t: Tensor, hn: Tensor, lg: int, rec: int) -> None:
        """Codebook cb's head norm (an RMS apply of t's records ``rec``, kept) into hn and its logits (fp32) at lg."""
        a = Apply(m=TM, nblk=C // 320, mode="rms", x=t.addr, y=hn.addr, rec=rec, C=H, eps=EPS, y_np=hn.np)
        bd.apply(a, self.w["hn"][cb], None, name=self.hn_image(cb))
        bd.gemm(Gemm(m=TM, ntile=1, K=H, N=NLOG, x=hn.addr, y=lg, out="f32", x_np=hn.np), self.w["heads"][cb], None,
                name=self.head_image(cb))

    def hn_image(self, cb: int) -> str:
        return f"{self.p}.hn{cb}"

    def head_image(self, cb: int) -> str:
        return f"{self.p}.head{cb}"


# -- the host side


class Depth:
    """The depthformer (module docstring) as dispatches on a VisionDevice: ``begin`` (a frame's hidden state: the proj
    program, P read back) and ``logits`` (a pass: x0 written, the pass program for its codebooks)."""

    def __init__(self, npu: VisionDevice, w: dict, emb: Callable[[np.ndarray], np.ndarray], mask: np.ndarray, *,
                 name: str = "d", state: dict | None = None):
        """emb: rows (k,) of the merged codebook tables (2017 cb + code) -> their embeddings (k, 1024); mask: the
        ``mask_embd`` (1024); state: a ``state()`` on a restored device."""
        self.npu, self.w, self.p = npu, w, name
        self.emb, self.mask = emb, np.asarray(mask, np.float32).reshape(-1)
        self.net = Net(w, name)
        self.bd = Builder(npu, reuse=state is not None)
        if state is None:
            self.programs = {"proj": Program.build(npu, self.bd, self.net.proj),
                             "pass": Program.build(npu, self.bd, self.net.pass_)}
            others = self.bd.fork()  # (the other codebooks' heads and norms: images of the program's layout)
            lv = self.net.lv
            t, hn = alloc(others, lv, C), alloc(others, lv, C, "pn")
            lg, rec = others.mem.alloc(LOG_BYTES), others.mem.alloc(ln_rec(lv))
            for cb in range(NPICK, NCB):
                self.net.head(others, cb, t, hn, lg, rec)
            others.slots.pack(npu)
            self.bd.put(f"{name}.rope", np.tile(rope_table(), COLS))
        else:
            self.programs = dict(state["programs"])
        self.proj_tr = self.programs["proj"].transaction(npu)
        self._trs: dict[tuple[int, ...], Transaction] = {}
        self.P = np.zeros((NCB, C), np.float32)

    def state(self) -> dict:
        """What a restored device's depthformer takes besides its images: the programs."""
        return {"programs": dict(self.programs)}

    def begin(self, h: np.ndarray) -> None:
        """A frame's hidden state h (2048): P = proj_in(h) + b (the proj program)."""
        x = np.zeros((TM, 2048), np.float32)
        x[0] = h
        self.npu.write(f"{self.p}.h", L.panels(x))  # (column 0's)
        self.npu.run(self.proj_tr)
        self.P = self.npu.read(f"{self.p}.p", 2 * PROJ).view(bfloat16).astype(np.float32).reshape(NCB, C)

    def inputs(self, codes: np.ndarray) -> np.ndarray:
        """The slots' inputs x0 (12, 1280) of P and the codes (12: -1 masked)."""
        codes = np.asarray(codes)
        e = np.zeros((NCB, C), np.float32)
        e[:, :H] = self.mask
        known = np.flatnonzero(codes >= 0)
        if len(known):
            e[known, :H] = self.emb(NCODES * known + codes[known])
        return self.P + e

    def logits(self, codes: np.ndarray, cbs: list[int] | tuple[int, ...]) -> np.ndarray:
        """A pass's logits (len(cbs), 2017) float32 of codebooks cbs (<= 4) with the codes revealed (12: -1 masked)."""
        x = np.zeros((TM, C), np.float32)
        x[:NCB] = self.inputs(codes)
        self.npu.write(f"{self.p}.x0", L.ctiles(x, TM))
        key = tuple(cbs) + tuple(range(NPICK))[len(cbs):]
        tr = self._trs.get(key)
        if tr is None:
            ren = {self.net.hn_image(j): self.net.hn_image(cb) for j, cb in enumerate(key)} | \
                {self.net.head_image(j): self.net.head_image(cb) for j, cb in enumerate(key)}
            tr = self._trs[key] = self.programs["pass"].transaction(self.npu, lambda n: ren.get(n, n))
            if len(self._trs) > 64:
                self._trs.pop(next(iter(self._trs)))
        self.npu.run(tr)
        out = np.empty((len(cbs), NCODES), np.float32)
        for j, cb in enumerate(cbs):
            raw = self.npu.read(f"{self.p}.lg{j}", LOG_BYTES).reshape(NLOG // 80, 5760)[:, :5120].copy()
            t = raw.view(np.float32).reshape(NLOG // 80, 2, 10, 8, 8)  # ([pass][rb][cb][tok][ch])
            out[j] = t[:, cb // 8, :, cb % 8, :].reshape(-1)[:NCODES]
        return out
