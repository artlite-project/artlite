"""MOSS-Transcribe-Diarize's audio tower (projector ``moss``: a Whisper encoder and the VQ adaptor) on the vision
overlay (``artlite.vision``: one design for every network, its ops built by host-built transactions; the layers
``artlite.vision.vit``'s, channels and heads padded there).

The network (transformers' ``WhisperEncoder``, the checkpoint's ``VQAdaptor``): a 30 s chunk's log-mel spectrogram (80
bins, 3000 frames) through conv1 (80 -> 1024 channels, 3 taps) + GELU and conv2 (3 taps, stride 2) + GELU, a row per 2
frames (1500), + the position table; 24 layers of attention over every row and a feed-forward of 4096 (GELU erf;
LayerNorm eps 1e-5, 16 heads of 64); the final LayerNorm; every 4 rows merged (4 x 1024) -> Linear -> SiLU -> Linear
-> LayerNorm (eps 1e-6): 375 rows of the language model's width, of which a chunk's own tokens are the first.

Tower (``tower``): ``vit.Vit`` widths 1024 in C 1280, heads of 64 in 80, the feed-forward's 4096 in 4160.

Layout: ``artlite.vision.windows``'s with a merged token as a window and its 4 rows as the window's patches (3 bands
of 512 rows; row 4 w + s in window w's patch s: ``slot_rows``), so that the merge is the windows' (``windows.
merged_panels``: the final norm's panels go to DDR as the adaptor's tile of merged tokens) and the keys one dense joint
sequence whose last 36 are masked (the grid's rows past the chunk's 1500).

The stem: a row j of conv2 reads conv1 at frames 2 j - 1, 2 j, 2 j + 1, so conv1 runs at those frames, a GEMM per tap
of the host's panels (``stem_panels``: a frame's 3 x 80 inputs and a constant 1 that carries the bias, all zero for
the frame before the chunk: conv2's padding), each writing its GELU's panels where conv2's operand has that tap's 1024
channels (a tap's 256 pad channels land on the next tap's place before it is written). conv2 is then a GEMM of K 3072
with the GELU epilogue, and an identity GEMM turns its panels into the residual's bf16 C tiles with the position rows
as its residual and the first layer's LayerNorm records. Both convs take hi + lo weights (``ops.Gemm``).

The adaptor: its first linear on the merged tokens' tile (K 4 x 1280, the pads' weights zero) with the SiLU epilogue,
the second with LayerNorm records, the LayerNorm into bf16 C tiles (``ops.Apply(ctiles=True)``); hi + lo weights.

A chunk is one dispatch (``Encoder``): the host writes the stem's panels, runs the program and reads the adaptor's
rows; a recording's chunks run one after another, the next one's panels made on a thread meanwhile (``stream``).
"""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from artlite.operators import kmath
from artlite.operators.design import COLS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision import windows as W
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gelu_table, gemm, ln_rec, silu_words
from artlite.vision.ops import Apply
from artlite.vision.program import Program
from artlite.vision.vit import Bands, Keys, Vit, blocks4, k_view, kv_bytes, layer_bands, pad, pad_heads
from artlite.vision.vit import vit_level as level
from artlite.vision.weights import Wt, cat, computed, eye, f32, values
from artlite.vision.windows import Layout, Names

from .host import FRAMES, N_MEL, ROWS

TAPS = 3  # (the convs' taps)
ONE, K1 = TAPS * N_MEL, 256  # (conv1's inputs of a frame: its taps' bins, then the constant 1; whole k-tiles)
MERGE = 4  # (encoder rows a token)
TOKENS = ROWS // MERGE  # (a chunk's tokens: 375)
LAYOUT = Layout(1, TOKENS)  # (a token a window, its rows the window's patches: 3 bands)
NB = LAYOUT.nb


def tower(hidden: int, heads: int, ffn: int, eps: float) -> Vit:
    """The widths of a Whisper encoder (NotImplementedError: one the overlay does not run)."""
    return Vit.of(hidden, heads, ffn, eps, gelu="gelu_erf")


def slot_rows() -> np.ndarray:
    """Per slot ([band][column][token 64]) its encoder row: row 4 w + s of window w's patch s (a slot past the chunk
    takes a row of a real window: ``windows.Layout``)."""
    w, s, _ = LAYOUT.slot_windows()
    return MERGE * w + s


# -- weights


def layer_weights(get: Callable[..., Wt | np.ndarray], il: int, vt: Vit) -> dict[str, Wt | np.ndarray]:
    """Layer il's padded weights by role (``vit.layer_bands``'): q (x log2 e / sqrt HD), k, v (rows by head), o (columns
    by head), up, down and their biases (``*_b``; k has none), ln1 / ln2 gamma and beta (``*_g``, ``*_b``); q, k, v and
    up of K = the hidden."""
    p = f"a.blk.{il}."
    C, A, F = vt.C, vt.HEADS * vt.HP, vt.FFN
    vec = lambda n: values(get(p + n)).reshape(-1)
    qs = np.float32(vt.qscale)
    w = {"q": pad(pad_heads(f32(get(p + "attn_q.weight")) * float(qs), vt=vt), A),
         "q_b": pad_heads(vec("attn_q.bias") * qs, vt=vt),
         "k": pad(pad_heads(f32(get(p + "attn_k.weight")), vt=vt), A), "k_b": np.zeros(A, np.float32),
         "v": pad(pad_heads(f32(get(p + "attn_v.weight")), vt=vt), A), "v_b": pad_heads(vec("attn_v.bias"), vt=vt),
         "o": pad(pad_heads(f32(get(p + "attn_out.weight")), 1, vt), C, A), "o_b": pad(vec("attn_out.bias"), C),
         "up": pad(f32(get(p + "ffn_up.weight")), F), "up_b": pad(vec("ffn_up.bias"), F),
         "down": pad(f32(get(p + "ffn_down.weight")), C, F), "down_b": pad(vec("ffn_down.bias"), C)}
    for n in ("ln1", "ln2"):
        w[n + "_g"], w[n + "_b"] = pad(vec(n + ".weight"), C), pad(vec(n + ".bias"), C)
    return w


def conv1_weights(get: Callable[..., Wt | np.ndarray], vt: Vit) -> Wt:
    """conv1 as a GEMM of ``stem_panels``' inputs (C, K1): a tap's bins at 80 tap + bin, the bias at input ``ONE``
    (computed on the host at every load)."""
    w = values(get("a.conv1d.1.weight")).reshape(vt.H, N_MEL, TAPS)
    m = np.zeros((vt.C, K1), np.float32)
    m[:vt.H, :ONE] = w.transpose(0, 2, 1).reshape(vt.H, ONE)
    m[:vt.H, ONE] = values(get("a.conv1d.1.bias")).reshape(-1)
    return computed("moss.conv1", m)


def weights(get: Callable[..., Wt | np.ndarray], layers: int, proj: int, vt: Vit) -> dict:
    """Every padded weight of the tower: "c1" (``conv1_weights``), "c2" (conv2 on its taps' channels, K 3 x H: the
    file's filters by tap) and its bias, "eye" (the identity that makes the stem's panels C tiles), per layer il its
    ``layer_weights``, "post_g" / "post_b" (the final LayerNorm), the adaptor's "up" (on the merged 4 x C channels) /
    "down" and biases, its LayerNorm "out_g" / "out_b"; "pos" (the position table, float32), "layers", "proj", "vit"
    (the widths). Matrices as ``get`` gives them (``weights.Wt``: views of the file), never materialized."""
    C, H = vt.C, vt.H
    vec = lambda n: values(get(n)).reshape(-1)
    c2 = Wt.of(get("a.conv1d.2.weight", rows=H).parts[0].source, (H, H, TAPS))
    w: dict = {"c1": conv1_weights(get, vt), "c2": pad(cat([c2[:, :, t] for t in range(TAPS)], 1), C),
               "c2_b": pad(vec("a.conv1d.2.bias"), C), "eye": pad(eye(H), C),
               "pos": values(get("a.position_embd.weight")).reshape(-1, H),
               "layers": layers, "proj": proj, "vit": vt, **{il: layer_weights(get, il, vt) for il in range(layers)}}
    w["post_g"], w["post_b"] = pad(vec("a.post_ln.weight"), C), pad(vec("a.post_ln.bias"), C)
    w["up"] = pad(blocks4(get("mm.a.mlp.0.weight"), 1, H, C), C, MERGE * C)
    w["up_b"] = pad(vec("mm.a.mlp.0.bias"), C)
    w["down"], w["down_b"] = pad(f32(get("mm.a.mlp.2.weight")), C), pad(vec("mm.a.mlp.2.bias"), C)
    w["out_g"], w["out_b"] = pad(vec("mm.a.mlp.3.weight"), C), pad(vec("mm.a.mlp.3.bias"), C)
    return w


# -- the program


def keys_of(name: str, vt: Vit) -> Keys:
    """The attention's keys: one dense joint sequence, the grid's rows past the chunk masked."""
    ok = LAYOUT.keys()
    return Keys(name, joint=True, dense=True, heads=vt.HEADS, valid=(ok, ok))


def stem(bd: Builder, w: dict, nm: Names, vt: Vit) -> None:
    """The convs and the position rows (module docstring): per band the host's panels (DDR image <p>.mel.b: [column]
    [tap][kb][token panel][72 B]) -> conv1 per tap -> conv2 -> + the position rows (<p>.pos.b) -> the residual's
    bands before layer 0 and their records."""
    lv = level(NB)
    C, H = vt.C, vt.H
    out = Bands(lv, nm.x(0), C)
    tap, mb = lv.pn(H), TAPS * lv.pn(K1)
    for b in range(NB):
        a = bd.mem.alloc(mb)
        bd.op(bd.fill(f"{nm.p}.mel.{b}", mb, mem=a, nbytes=mb), {})
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        a2 = alloc(bd, lv, TAPS * H + C - H, "pn")
        for t in range(TAPS):
            gemm(bd, Tensor(lv, K1, a + t * lv.pn(K1), "pn"), w["c1"], None, name=f"{nm.p}.conv1",
                 out=Tensor(lv, C, a2.addr + t * tap, "pn"), kind="gelu", hilo=True)
        bd.mem.release(a, mb)
        h = alloc(bd, lv, C, "pn")
        gemm(bd, Tensor(lv, TAPS * H, a2.addr, "pn"), w["c2"], w["c2_b"], name=f"{nm.p}.conv2", out=h, kind="gelu",
             hilo=True)
        free(bd, a2)
        x = alloc(bd, lv, C)
        bd.op(bd.load(f"{nm.p}.pos.{b}", x.addr, x.nbytes), {})
        rec = gemm(bd, k_view(h, w["eye"]), w["eye"], None, name=f"{nm.p}.x0", out=x, res=x, ln=True)
        free(bd, h)
        out.store(bd, b, x, rec)


def layers(bd: Builder, w: dict, nm: Names, il: range, kv: int, vt: Vit) -> None:
    """Layers ``il`` (the residual from image x(il[0]) to x(il[-1] + 1))."""
    lv = level(NB)
    keys = keys_of(nm.attn(NB), vt)
    for i in il:
        layer_bands(bd, w[i], nm.layer(i), Bands(lv, nm.x(i), vt.C), Bands(lv, nm.x(i + 1), vt.C), kv_ddr=kv, keys=keys,
                    park=lambda b: f"{nm.p}.park", vt=vt)


def final(bd: Builder, w: dict, nm: Names, nl: int, vt: Vit, eps: float) -> None:
    """The final LayerNorm of the residual after nl layers, the merge and the adaptor (module docstring) -> the
    tokens' rows as C tiles (<p>.out: [column][C / 80][ct_bytes(64)], a column's 64 tokens)."""
    C, H = vt.C, vt.H
    md = W.Merger(H=H, C=C, eps=vt.EPS)
    l1 = level(1)
    W.merged_panels(bd, w, nm, NB, nl, md)
    mgb = l1.pn(md.merged)
    mg = alloc(bd, l1, md.merged, "pn")
    spare = bd.mem.alloc(D.A_OBJ)  # (the GEMM's A walk reads past its panels' end)
    bd.op(bd.fill(f"{nm.p}.mg.0", mgb, mem=mg.addr, nbytes=mgb), {})
    bd.table("silu", silu_words())
    h = alloc(bd, l1, C, "pn")
    gemm(bd, mg, w["up"], w["up_b"], name=f"{nm.p}.up", out=h, kind="gelu", act="silu", hilo=True)
    bd.mem.release(spare, D.A_OBJ)
    free(bd, mg)
    y = alloc(bd, l1, C)
    rec = gemm(bd, k_view(h, w["down"]), w["down"], w["down_b"], name=f"{nm.p}.down", out=y, ln=True, hilo=True)
    free(bd, h)
    e = alloc(bd, l1, C)
    bd.apply(Apply(m=l1.m, nblk=C // 320, mode="ln", x=y.addr, y=e.addr, ntile=l1.ntile, rec=rec, C=H, eps=eps,
                   ctiles=True), w["out_g"], w["out_b"], name=f"{nm.p}.norm")
    bd.mem.release(rec, ln_rec(l1))
    free(bd, y)
    bd.op(bd.spill(f"{nm.p}.out", e.addr, e.nbytes), {})
    free(bd, e)


# -- the host side of a chunk


def stem_panels(mel: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    """Per band the stem's panels (DDR image <p>.mel.b) of a chunk's log-mel spectrogram (N_MEL, FRAMES): slot row j's
    tap t the inputs of conv1 at frame 2 j + t - 1 (the bins of frames f - 1, f, f + 1, zero outside the chunk, then
    the constant 1), all zero for a frame outside the chunk; a frame's 8 bins a bfp16 block. rows: ``slot_rows``."""
    nb = N_MEL // 8
    bins = np.zeros((FRAMES + 2, nb, 9), np.uint8)  # (per frame its bins' blocks; a zero frame before and after)
    bins[1:-1] = L.blocks(np.ascontiguousarray(mel.T).reshape(FRAMES, nb, 8))
    x = np.zeros((FRAMES + 1, K1 // 8, 9), np.uint8)  # (per frame its inputs' blocks; the last: a frame outside the chunk)
    x[:FRAMES, :TAPS * nb] = np.lib.stride_tricks.sliding_window_view(bins, TAPS, axis=0).transpose(0, 3, 1, 2).reshape(
        FRAMES, TAPS * nb, 9)
    x[:FRAMES, ONE // 8] = L.blocks(np.eye(1, 8, dtype=np.float32))[0]
    f = 2 * rows[..., None] + np.arange(-1, TAPS - 1)  # ([band][column][token][tap])
    g = x[np.where((f >= 0) & (f < FRAMES), f, FRAMES)]  # (band, column, token panel 8 lane, tap, kb, 9)
    g = g.reshape(NB, COLS, 8, 8, TAPS, K1 // 8, 9).transpose(0, 1, 4, 5, 2, 3, 6)
    return [np.ascontiguousarray(g[b]).reshape(-1) for b in range(NB)]


def band_rows(x: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    """Per band the C tiles ([column][C tiles of 64]) of per-row values x (rows, C) at their slots."""
    return [np.concatenate([L.ctiles(x[rows[b, c]], 64) for c in range(COLS)]) for b in range(NB)]


class Encoder:
    """The tower (module docstring) as a dispatch on a VisionDevice: its program (built at construction, or a cached
    state's), a chunk's inputs and rows."""

    def __init__(self, npu: VisionDevice, w: dict, *, eps: float, name: str = "m", state: dict | None = None):
        """w: ``weights``; eps: the adaptor's LayerNorm's; state: a ``state()`` on a device whose images were restored
        (``VisionDevice.restore``: its program taken)."""
        self.npu, self.nm, self.vt, self.proj = npu, Names(name), w["vit"], w["proj"]
        vt, nm, nl = self.vt, self.nm, w["layers"]
        self.bd = Builder(npu, reuse=state is not None)
        kv = self.bd.ddr(f"{name}.kv", kv_bytes(level(NB), keys_of("", vt)))

        def net(bd: Builder) -> None:
            stem(bd, w, nm, vt)
            layers(bd, w, nm, range(nl), kv, vt)
            final(bd, w, nm, nl, vt, eps)
        self.program = state["program"] if state else Program.build(npu, self.bd, net)
        self.tr = self.program.transaction(npu)
        self.slots = slot_rows()
        pos = np.zeros((ROWS, vt.C), np.float32)
        pos[:, :vt.H] = w["pos"]
        for b, data in enumerate(band_rows(pos, self.slots)):  # (the file's rows, written at every load)
            npu.write(f"{name}.pos.{b}", data)

    def state(self) -> dict:
        """What a restored device's encoder takes besides its images (``__init__``'s state): the program."""
        return {"program": self.program}

    def inputs(self, mel: np.ndarray) -> list[np.ndarray]:
        """A chunk's DDR images (``stem_panels``) of its log-mel spectrogram."""
        return stem_panels(mel, self.slots)

    def run(self, inputs: list[np.ndarray]) -> np.ndarray:
        """A chunk's dispatch on its ``inputs``: the adaptor's C tiles as read."""
        for b, data in enumerate(inputs):
            self.npu.write(f"{self.nm.p}.mel.{b}", data)
        self.npu.run(self.tr)
        return self.npu.read(f"{self.nm.p}.out", COLS * level(1).ct(self.vt.C))

    def rows(self, raw: np.ndarray, tokens: int = TOKENS) -> np.ndarray:
        """``run``'s C tiles -> the chunk's first ``tokens`` tokens' rows (tokens, proj) float32."""
        raw = raw.reshape(COLS, -1)
        tiles = np.stack([L.from_ctiles(raw[c], 64, self.vt.C, 64) for c in range(COLS)])
        return np.ascontiguousarray(LAYOUT.merged(tiles[None])[:tokens, :self.proj])

    def __call__(self, mel: np.ndarray, tokens: int = TOKENS) -> np.ndarray:
        """A chunk's log-mel spectrogram (N_MEL, FRAMES) -> its first ``tokens`` tokens' rows (tokens, proj)."""
        return self.rows(self.run(self.inputs(mel)), tokens)

    def stream(self, chunks: Iterable[tuple[np.ndarray, int]], mel: Callable[[np.ndarray], np.ndarray]) -> Iterator[np.ndarray]:
        """``__call__`` of each chunk (its samples, its tokens) in turn, the host's part of the next one (``mel`` of
        its samples, its panels) and of the last one (its rows) on a thread while the NPU runs one."""
        it = iter(chunks)

        def prep() -> tuple[list[np.ndarray], int] | None:
            c = next(it, None)
            return None if c is None else (self.inputs(mel(c[0])), c[1])
        done = None
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(prep)
            while (got := job.result()) is not None:
                job = pool.submit(prep)
                raw = self.run(got[0])
                if done is not None:
                    yield done.result()
                done = pool.submit(self.rows, raw, got[1])
            if done is not None:
                yield done.result()
