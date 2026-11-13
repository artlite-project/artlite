"""Vision encoders over an image's grid of 2 x 2 windows of patches on the vision overlay (Qwen3-VL's, LFM3-VL's): the
layout of a band's tiles, the encoder's programs and dispatches, the final merger, the depthwise 3 x 3 conv's inputs.

Layout (``Layout``): an image of gh x gw windows (2 x 2 patches each: the merged tokens) on a level of ``nb`` bands of
``WIN`` = 128 windows; window w (raster order) is at (p, band, column, lane) = w's digits in (2, nb, 8, 8) (p the
slowest: a tile's half), its patch q = (py, px) in token panel 2 q + p of that band and column, at that lane: a tile
half is 8 consecutive windows of the raster. The attention's keys are the joint sequence [p][band][column][q][lane]
(``ops.Attention``: the overlay's two images are p), whose valid keys are whole 64-key groups up to the one of the last
window (a lane pattern), so that ``ops.Attention(dense=True)`` takes any image size as data: the transactions depend on
nb only. Slots past the image take the patches of window w mod windows (typical values in the bfp16 blocks over tokens
that mix real and padding tokens; their keys masked). The merge: a band's 4 token-panel pairs are its windows' 4
patches, so the final norm's panels go to DDR in the merged layout (``merger``): 4 bands' merged tokens are one m-64
tile (token panel 2 (band % 4) + p) for the merger's MLP, its weights streamed once per 4 bands.

Programs (``WindowEncoder``): per nb a program of pieces, one dispatch each (built on first use, ``program``): nb 1 one
piece (the patch embedding, every layer, the merger); larger images the patch embedding, pieces of layers, the merger.
A video's pair of frames (one temporal patch: the patch embedding's two kernels on its two frames) runs a program of
its own patch embedding, its other pieces the image program's of that nb. Two pairs of one size run as one where that
takes fewer bands (``Layout.two``: a pair a half, nb bands of 64 windows each; the attention's per-image key masks over
the joint sequence instead of its dense form, so at most 4 bands; the conv's edge masks each pair's own): the same
program, other inputs, each pair's embeddings bit for bit its own run's (a long video's pairs of up to 64 tokens: a
band for two). ``stream`` runs a sequence of images or pairs with the host's part of each (its inputs, the last one's
embeddings) on a thread while the NPU runs the one before.
A piece of layers is built once from a layer s and run at layer at (its layers' kinds those from s: ``kinds``) by
rewriting its shim addresses for layer at's images (``VisionDevice.refs`` / ``resolve``, ``Names.rename``: the
residual's ping-pong images swapped at an odd distance). Shim addresses a piece's fills take per image size (tagged:
``seq.MemTask``) are set before each dispatch (``WindowEncoder.shifts``).

The depthwise conv (``ops.DwConv``): a patch's neighbour (dy, dx) is patch (py + dy, px + dx) of its window's
neighbour (cy, cx) = ((py + dy) >> 1, (px + dx) >> 1), class ((py + dy) & 1, (px + dx) & 1): a half's taps read per
class a strip of 10 windows from the half's first - 1 (cy 0), and classes 2, 3 the strip one window row above, 0, 1
the one below (gw windows away: those fills' shim addresses are set per image size, ``conv_shift``). A gated input g
lives in DDR as [window][class 4][channels] bf16 (``conv_ddr_bytes``: ``MARGIN`` windows before the first and after
the last, which the strips of an image up to ``GW_MAX`` windows wide read past its rows: masked); its edge masks per
(band, column, half, class, tap) are the image's (``dw_masks``).
"""

import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..operators import kmath
from ..operators.design import COLS
from ..overlay import Transaction
from . import layout as L
from .build import Builder
from .device import VisionDevice
from .layers import Tensor, alloc, free, gelu_table, gemm
from .ops import _dims
from .program import Program, ProgramStore
from .vit import ln, vit_level, xfer

WIN = 128  # (windows of a band: 8 columns x 2 halves x 8 lanes)
GW_MAX = 1024  # (an image's windows per row at most: 4096 merged tokens at transformers' aspect bound of 200)
MARGIN = GW_MAX + 2  # (windows of the conv input's margins)
TOKEN_PX = 32  # (a merged token's pixels per side: patches of 16, merged 2 x 2)
PIX, ONE, KP = 768, 768, 832  # (a patch's inputs (c, ky, kx) of 3 x 16 x 16; the constant input 1, a bfp16 block of
# its own; the panels' K: whole k-tiles)
PIECE_BAND_LAYERS = 16  # (a piece's band-layers at most: its build time)
PIECE_MS = 500.0  # (a dispatch's estimated time at most)


def piece_layers(nb: int, layers: int, cost: tuple[float, float]) -> int:
    """Layers of a dispatch at nb bands: all at nb 1, else at most PIECE_BAND_LAYERS band-layers and about PIECE_MS
    (a layer ~ a nb + b nb^2 ms for ``cost`` (a, b): its GEMMs and its attention over every key)."""
    if nb == 1:
        return layers
    return max(1, min(layers, PIECE_BAND_LAYERS // nb, int(PIECE_MS / (cost[0] * nb + cost[1] * nb * nb))))


@dataclass(frozen=True)
class Layout:
    """An image of gh x gw merged tokens (windows of 2 x 2 patches) on a level of ``nb`` bands (module docstring);
    ``two``: two images of that size, one a half (p: window w of image p at (p, band, column, lane) = w's digits in
    (nb, 8, 8)), each the other's padding: masked keys, masked conv neighbours (two videos' pairs a dispatch)."""

    gh: int
    gw: int
    two: bool = False

    @property
    def windows(self) -> int:
        return self.gh * self.gw

    @property
    def nb(self) -> int:
        return -(-self.windows // (WIN // 2 if self.two else WIN))

    def slot_windows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per slot ([band][column][token 64], token = 8 token panel + lane) its window (the real one: w mod windows;
        ``two``: image p's at p windows + its own) and patch s; plus the real windows' mask."""
        nb = self.nb
        b, c, t = np.meshgrid(np.arange(nb), np.arange(COLS), np.arange(64), indexing="ij")
        tp, lane = np.divmod(t, 8)
        s, p = np.divmod(tp, 2)
        if self.two:
            w = (b * COLS + c) * 8 + lane
            return p * self.windows + w % self.windows, s, w < self.windows
        w = ((p * nb + b) * COLS + c) * 8 + lane
        return w % self.windows, s, w < self.windows

    def slot_patches(self) -> np.ndarray:
        """Per slot ([band][column][64]) its patch's index in the image's raster order (2 gh x 2 gw patches; ``two``:
        image p's after image 0's)."""
        w, s, _ = self.slot_windows()
        img, w = np.divmod(w, self.windows)
        r, c = np.divmod(w, self.gw)
        return img * 4 * self.windows + (2 * r + s // 2) * (2 * self.gw) + 2 * c + s % 2

    def keys(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """The valid keys of the joint sequence [p][band][column][s][lane] (``two``: each image's, a pair)."""
        nb = self.nb
        p, b, c, _, lane = np.meshgrid(*(np.arange(n) for n in (2, nb, COLS, 4, 8)), indexing="ij")
        if self.two:
            ok = (b * COLS + c) * 8 + lane < self.windows
            return (ok & (p == 0)).reshape(-1), (ok & (p == 1)).reshape(-1)
        return (((p * nb + b) * COLS + c) * 8 + lane < self.windows).reshape(-1)

    def merged(self, out: np.ndarray) -> np.ndarray:
        """The MLP's tiles (quads of 4 bands, [quad][column][token 64]: token panel 2 (band % 4) + p, lane; channels)
        -> the windows' embeddings in raster order (``two``: image 0's, then image 1's)."""
        nb = self.nb
        q, c, t = np.meshgrid(np.arange(-(-nb // 4)), np.arange(COLS), np.arange(64), indexing="ij")
        j, lane = np.divmod(t, 8)
        b, p = 4 * q + j // 2, j % 2
        if self.two:
            w = (b * COLS + c) * 8 + lane
            ok = (b < nb) & (w < self.windows)
            w = p * self.windows + w
        else:
            w = ((p * nb + b) * COLS + c) * 8 + lane
            ok = (b < nb) & (w < self.windows)
        emb = np.empty(((2 if self.two else 1) * self.windows, out.shape[-1]), np.float32)
        emb[w[ok]] = out.reshape(-1, out.shape[-1])[ok.reshape(-1)]
        return emb


def rope_table(ph: int, pw: int, head_dim: int, theta: float, hp: int = 80) -> np.ndarray:
    """Per patch (raster order) its ``hp`` channels of the RoPE epilogue (``ops.Gemm(out="rope")``: pairs (k, hp / 2 +
    k)): cos of pairs 0 .. head_dim / 2 - 1, then 1; sin of the pairs, then 0; pair i's angle y s^i (i < head_dim / 4),
    x s^(i - head_dim / 4) above, s = theta^(-4 / head_dim) (the axial 2-D RoPE of the vision encoders)."""
    q = head_dim // 4
    f = theta ** (-np.arange(q, dtype=np.float64) / q)
    y, x = np.divmod(np.arange(ph * pw), pw)
    ang = np.concatenate([y[:, None] * f, x[:, None] * f], 1)
    out = np.zeros((ph * pw, hp), np.float32)
    out[:, :hp // 2] = 1
    out[:, :head_dim // 2], out[:, hp // 2:hp // 2 + head_dim // 2] = np.cos(ang), np.sin(ang)
    return out


class Names:
    """The DDR images of an encoder (``name`` their prefix)."""

    def __init__(self, name: str):
        self.p = name

    def x(self, i: int) -> str:  # (the residual before layer i: two images, alternating)
        return f"{self.p}.x{'ab'[i % 2]}"

    def layer(self, il: int) -> str:
        return f"{self.p}.l{il}"

    def attn(self, nb: int) -> str:
        return f"{self.p}.attn.{nb}"

    def rename(self, off: int) -> Callable[[str], str]:
        """The images of a piece built from layer s when run from layer s + ``off``: layer il's -> il + off's, the
        residual's two swapped at odd off."""
        lay = re.compile(rf"^{re.escape(self.p)}\.l(\d+)\.")
        xs = {f"{self.p}.xa.": f"{self.p}.xb.", f"{self.p}.xb.": f"{self.p}.xa."}

        def f(n: str) -> str:
            n = lay.sub(lambda m: f"{self.p}.l{int(m[1]) + off}.", n)
            k = n[:len(self.p) + 4]
            return xs[k] + n[len(k):] if off % 2 and k in xs else n
        return f


@dataclass(frozen=True)
class Merger:
    """The final merger's widths: the norm's ``H`` real channels of ``C`` per patch, its eps, mm.0's outputs in chunks
    ``up`` (padded: ``FUP``, passes of 320 each), mm.2's outputs (the language model's width in whole passes,
    ``out_width``) in chunks of ``dn``."""

    H: int
    C: int = 1280
    eps: float = 1e-6
    up: tuple[int, ...] = (960,) * 5
    dn: int = 640

    @property
    def FUP(self) -> int:
        return sum(self.up)

    @property
    def merged(self) -> int:
        return 4 * self.C


def out_width(proj: int) -> int:
    """The embeddings' channels on the NPU: the language model's width in whole passes (320 a row of 4)."""
    return -(-proj // 320) * 320


def merger(bd: Builder, w: dict, nm: Names, nb: int, nl: int, md: Merger) -> None:
    """The final LayerNorm of the residual after nl layers and the merger's MLP: per band LayerNorm -> panels drained
    to DDR in the merged layout of its quad (<p>.mg.q: [column][kb 4 C / 8][token panel 8][72 B]; band b's patch s,
    image p at kb C / 8 s + kb, token panel 2 (b % 4) + p); per quad its tile of 64 merged tokens: mm.0 + GELU (erf) in
    chunks to DDR (<p>.hid.q), mm.2 in chunks -> the embeddings' C tiles (<p>.out.q: [column][out_width / 80]
    [ct_bytes(64)]). w: "post_g", "post_b", "up", "up_b", "down", "down_b"."""
    from .vit import Bands  # (the ViT's banded residual)
    C, MERGED, FUP = md.C, md.merged, md.FUP
    lv, l1 = vit_level(nb), vit_level(1)
    x = Bands(lv, nm.x(nl), C)
    pn, mgb, hid = lv.pn(C), l1.pn(MERGED), l1.pn(FUP)
    outw = w["down"].shape[0]
    walk = _dims((18, 1), (2, 18), (C // 8, 144), (4, 36))  # (stream order: s, kb, p, a panel's 18 words)
    for b in range(nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, lv, C, "pn")
        ln(bd, xb, n, w["post_g"], w["post_b"], name=f"{nm.p}.post", rec=rec, C=C, H=md.H, eps=md.eps)
        free(bd, xb)
        bd.op(xfer(bd, f"{nm.p}.mg.{b // 4}", mgb, fill=False, mem=n.addr, nbytes=pn, off=2 * (b % 4) * L.PANEL,
                   mdims=walk, sdims=(36, MERGED // 8, 144, 144 * MERGED // 8)), {})
        free(bd, n)
    for qd in range(-(-nb // 4)):
        mg = alloc(bd, l1, MERGED, "pn")
        spare = bd.mem.alloc(9216)  # (the GEMM's A walk reads past its panels' end)
        bd.op(xfer(bd, f"{nm.p}.mg.{qd}", mgb, fill=True, mem=mg.addr, nbytes=mgb), {})
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        g = alloc(bd, l1, max(md.up), "pn")
        u0 = 0
        for k, u in enumerate(md.up):
            gk = Tensor(l1, u, g.addr, "pn")
            gemm(bd, mg, w["up"][u0:u0 + u], w["up_b"][u0:u0 + u], name=f"{nm.p}.up{k}", out=gk, kind="gelu", hilo=True)
            bd.op(xfer(bd, f"{nm.p}.hid.{qd}", hid, fill=False, mem=g.addr, nbytes=gk.nbytes, off=l1.pn(u0)), {})
            u0 += u
        free(bd, g)
        bd.mem.release(spare, 9216)
        free(bd, mg)
        h = alloc(bd, l1, FUP, "pn")
        spare = bd.mem.alloc(9216)
        bd.op(xfer(bd, f"{nm.p}.hid.{qd}", hid, fill=True, mem=h.addr, nbytes=hid), {})
        y = alloc(bd, l1, md.dn)
        for k in range(outw // md.dn + (outw % md.dn > 0)):
            n = min(md.dn, outw - md.dn * k)
            yk = Tensor(l1, n, y.addr)
            gemm(bd, h, w["down"][md.dn * k:md.dn * k + n], w["down_b"][md.dn * k:md.dn * k + n], name=f"{nm.p}.down{k}",
                 out=yk, hilo=True)
            bd.op(xfer(bd, f"{nm.p}.out.{qd}", l1.ct(outw), fill=False, mem=y.addr, nbytes=yk.nbytes,
                       off=k * l1.ct(md.dn)), {})
        free(bd, y)
        bd.mem.release(spare, 9216)
        free(bd, h)


@dataclass
class Piece(Program):
    """A dispatch of a window encoder's program: the layers it runs (a piece of layers: ``layers`` built from layer
    ``start``, run at another layer by ``Names.rename``; else None)."""

    layers: int | None = None
    start: int = 0


class WindowEncoder:
    """An encoder on the window layout (module docstring) as dispatches on a VisionDevice: per nb a program (``program``:
    its pieces built on first use, over the same weight images: ``Builder.fork``). Per image the host writes its inputs
    (``inputs``: patch panels, the size's rows: ``size_inputs``), runs the pieces and reads the embeddings. A subclass
    gives the network: ``patch_embed`` (``pair=True``: a video's pair of frames, where the network takes videos),
    ``layers``, ``final``, ``size_inputs``, ``layers_per_piece``, ``kinds`` (the layers' kinds: a piece of layers runs
    where the kinds are its own), ``pixel_panels`` (an image's pixels or a pair's), ``shift`` (a tagged address's bytes
    for an image)."""

    PATCH = 16

    def __init__(self, npu: VisionDevice, w: dict, *, max_bands: int, name: str, state: dict | None = None,
                 store: ProgramStore | None = None, kv_bytes: int = 0):
        """w: the network's weights (``w["layers"]``: its layer count, ``w["proj"]``: the embeddings' width, ``w["down"]``:
        mm.2's padded rows); max_bands: the largest level (the K / V^T slots' DDR, ``kv_bytes``, allocated first);
        state: a ``state()`` on a device whose images were restored (``VisionDevice.restore``: its programs taken, new
        ones built over the images as they are); store: where the programs built later are kept (``program``: those
        of the cached state's images only)."""
        self.npu, self.w, self.nm = npu, w, Names(name)
        self.nl, self.proj, self.outw = w["layers"], w["proj"], w["down"].shape[0]
        self.bd = Builder(npu, reuse=state is not None)
        self.kv = self.bd.ddr(f"{name}.kv", kv_bytes)
        self.max_bands = max_bands
        for nb in range(1, max_bands + 1):  # (every level's attention command, before any program)
            self.bd.ddr(self.nm.attn(nb), 5760)
        # (programs by nb; a video's pairs' by ("pair", nb))
        self.programs: dict[int | tuple[str, int], list[Piece]] = {} if state is None else dict(state["programs"])
        self.store = store
        self._kept: set[str] | None = set(npu._images) if state is not None else None  # (the cached state's images)
        self.tr: dict[tuple, Transaction] = {}  # (by (program key, piece): a pair's shared pieces the image's)
        self._inputs: OrderedDict[Layout, dict[str, np.ndarray]] = OrderedDict()
        self._at: dict[tuple, tuple] = {}  # (a transaction's last (layer, size) its words are set for)
        self.program(1)  # (every layer's images: nb 1 runs them all in one dispatch)
        self.program(max_bands)  # (every band's DDR images: the layout that later programs take)

    kinds: tuple[str, ...] = ()

    def patch_embed(self, bd: Builder, nb: int) -> None:
        raise NotImplementedError

    def layers(self, bd: Builder, nb: int, il: range) -> None:
        raise NotImplementedError

    def final(self, bd: Builder, nb: int) -> None:
        raise NotImplementedError

    def layers_per_piece(self, nb: int) -> int:
        raise NotImplementedError

    def size_inputs(self, lay: Layout) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def pixel_panels(self, lay: Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def shift(self, tag: object, lay: Layout) -> int:
        raise NotImplementedError

    def state(self) -> dict:
        """What a restored device's encoder takes besides its images (``__init__``'s state): the programs built. The
        device's images as they are now are the cached state's: a program built later is cached (``built``) only where
        it takes none but these (a restored device has them where they were; an image allocated after the state, a
        pair's pixels say, is not there)."""
        self._kept = set(self.npu._images)
        return {"programs": dict(self.programs)}

    def _runs(self, nb: int) -> list[tuple[int, int]]:
        """The pieces of layers of nb bands (``layers_per_piece``): (first layer, count), each built from the first
        run of its kinds."""
        k = self.layers_per_piece(nb)
        out, i = [], 0
        while i < self.nl:
            n = min(k, self.nl - i)
            out.append((i, n))
            i += n
        return out

    def program(self, nb: int, pair: bool = False) -> list[Piece]:
        """The pieces of nb bands (built on first use): nb 1 one; else the patch embedding, the layers' pieces (one per
        distinct run of kinds: ``schedule`` runs each wherever its kinds are), the final merger. ``pair``: a video's
        pair of frames (the pair's patch embedding, then the image program's pieces)."""
        pkey = ("pair", nb) if pair else nb
        if pkey in self.programs:
            return self.programs[pkey]
        assert nb <= self.max_bands, (nb, self.max_bands)
        embed = (lambda b, n: self.patch_embed(b, n, pair=True)) if pair else self.patch_embed
        if nb == 1:
            def full(b: Builder) -> None:
                embed(b, 1)
                self.layers(b, 1, range(self.nl))
                self.final(b, 1)
            out = [Piece.build(self.npu, self.bd, full)]
        elif pair:
            out = [Piece.build(self.npu, self.bd, lambda b: embed(b, nb)), *self.program(nb)[1:]]
        else:
            out = [Piece.build(self.npu, self.bd, lambda b: self.patch_embed(b, nb))]
            built: dict[tuple, int] = {}
            for s, n in self._runs(nb):
                key = tuple(self.kinds[s:s + n]) if self.kinds else (n,)
                if key not in built:
                    built[key] = len(out)
                    out.append(Piece.build(self.npu, self.bd, lambda b, s=s, n=n: self.layers(b, nb, range(s, s + n)),
                                           layers=n, start=s))
            out.append(Piece.build(self.npu, self.bd, lambda b: self.final(b, nb)))
        self.programs[pkey] = out
        if self.store is not None:
            self.store.save(pkey, out, self._kept is not None and set().union(*(pc.images for pc in out)) <= self._kept)
        return out

    def schedule(self, nb: int, pair: bool = False) -> list[tuple[int, int]]:
        """The dispatches of an image (``pair``: a video's pair) of nb bands: (piece, its first layer): the layers
        covered by the program's pieces of layers, at each layer the longest whose kinds are the next layers'."""
        prog = self.program(nb, pair)
        if nb == 1:
            return [(0, 0)]
        kinds = self.kinds or ("",) * self.nl
        pieces = sorted(((p.layers, i) for i, p in enumerate(prog) if p.layers is not None), reverse=True)
        out, at = [(0, 0)], 0
        while at < self.nl:
            i = next(i for n, i in pieces if kinds[at:at + n] == kinds[prog[i].start:prog[i].start + n]
                     and at + n <= self.nl)
            out.append((i, at))
            at += prog[i].layers
        return out + [(len(prog) - 1, 0)]

    def _transaction(self, nb: int, i: int, at: int, lay: Layout, pair: bool = False) -> Transaction:
        """Piece i of nb's program (``pair``: a video pair's) for a run from layer ``at`` on an image of layout lay: its
        shim addresses resolved on this device's images (another process's program: its images restored at other
        addresses), a layer piece's per run for layer at's images, its tagged ones for lay's size, rewritten in
        place."""
        pc = self.program(nb, pair)[i]
        key = (("pair", nb) if pair and (nb == 1 or i == 0) else nb, i)
        if key not in self.tr:
            self.tr[key] = pc.transaction(self.npu)
        tr = self.tr[key]
        want = (at, (lay.gh, lay.gw) if pc.tags else None)
        if self._at.get(key) != want:
            if pc.layers is not None:
                self.npu.resolve(tr.words, pc.refs, self.nm.rename(at - pc.start))
            pc.retarget(self.npu, tr, lambda tag, base: base + self.shift(tag, lay))
            tr.flush()
            self._at[key] = want
        return tr

    def inputs(self, lay: Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        """An image's DDR images: its patch panels (``pixel_panels``) and its size's rows (``size_inputs``, kept for the
        last 4 sizes)."""
        out = self.pixel_panels(lay, px)
        key = lay
        if key not in self._inputs:
            self._inputs[key] = self.size_inputs(lay)
            if len(self._inputs) > 4:
                self._inputs.popitem(last=False)
        self._inputs.move_to_end(key)
        return out | self._inputs[key]

    def embeddings(self, lay: Layout) -> np.ndarray:
        """The MLP's output C tiles -> the image's embeddings (windows, proj) in raster order."""
        return self._merge(lay, self._outputs(lay))

    def _outputs(self, lay: Layout) -> list[np.ndarray]:
        """The MLP's output C tiles as read (per quad of bands)."""
        l1 = vit_level(1)
        return [self.npu.read(f"{self.nm.p}.out.{qd}", COLS * l1.ct(self.outw)) for qd in range(-(-lay.nb // 4))]

    def _merge(self, lay: Layout, raw: list[np.ndarray]) -> np.ndarray:
        tiles = np.empty((len(raw), COLS, 64, self.outw), np.float32)
        for qd, r in enumerate(raw):
            r = r.reshape(COLS, -1)
            for c in range(COLS):
                tiles[qd, c] = L.from_ctiles(r[c], 64, self.outw, 64)
        emb = lay.merged(tiles)[:, :self.proj]
        return emb.reshape(2, lay.windows, -1) if lay.two else emb

    def packs(self, gh: int, gw: int) -> bool:
        """Whether two images (a video's pairs) of gh x gw tokens run as one (``Layout.two``): a subclass's network that
        takes it, where it is fewer bands than two dispatches."""
        return False

    def __call__(self, px: np.ndarray, two: bool = False) -> np.ndarray:
        """An image's pixels (uint8 (H, W, 3), multiples of ``TOKEN_PX``) or a video's pair of frames ((2, H, W, 3)) ->
        its
        embeddings (H / 32 W / 32, proj) in raster order; ``two``: two of them of one size (px stacked; ``packs``) in
        one run -> (2, H / 32 W / 32, proj)."""
        pair = px.ndim == 4 + two
        lay = Layout(px.shape[-3] // TOKEN_PX, px.shape[-2] // TOKEN_PX, two)
        assert lay.gh * TOKEN_PX == px.shape[-3] and lay.gw * TOKEN_PX == px.shape[-2], px.shape
        with ThreadPoolExecutor(1) as pool:  # (the inputs while the program builds, on first use)
            host = pool.submit(self.inputs, lay, px)
            plan = self.schedule(lay.nb, pair)
            for n, data in host.result().items():
                self.npu.write(n, data)
        for i, at in plan:
            self.npu.run(self._transaction(lay.nb, i, at, lay, pair))
        return self.embeddings(lay)

    def stream(self, items: Iterable[tuple[np.ndarray, bool]]) -> Iterator[np.ndarray]:
        """``__call__`` of each (px, two) of ``items`` in turn (a generator's too: a video's pairs as its frames are
        decoded), on a thread while the NPU runs one: the next one's inputs taken and computed, the last one's outputs
        read into embeddings."""
        def prep(it: Iterator[tuple[np.ndarray, bool]]) -> tuple | None:
            x = next(it, None)
            if x is None:
                return None
            px, two = x
            lay = Layout(px.shape[-3] // TOKEN_PX, px.shape[-2] // TOKEN_PX, two)
            assert lay.gh * TOKEN_PX == px.shape[-3] and lay.gw * TOKEN_PX == px.shape[-2], px.shape
            return px.ndim == 4 + two, lay, self.inputs(lay, px)
        it = iter(items)
        done = None
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(prep, it)
            while (got := job.result()) is not None:
                pair, lay, inputs = got
                job = pool.submit(prep, it)
                plan = self.schedule(lay.nb, pair)
                for n, data in inputs.items():
                    self.npu.write(n, data)
                for i, at in plan:
                    self.npu.run(self._transaction(lay.nb, i, at, lay, pair))
                if done is not None:
                    yield done.result()
                done = pool.submit(self._merge, lay, self._outputs(lay))
            if done is not None:
                yield done.result()


def patch_panels(lay: Layout, px: np.ndarray, name: str, *, kp: int, one: int, patch: int = 16,
                 tag: str = "pix") -> dict[str, np.ndarray]:
    """Per band its patches' integer panels (DDR image <name>.<tag>.b: q = p - 128 of the patch's (c, ky, kx) inputs,
    frame by frame, the constant 1 at input ``one``, kp inputs) of an image's pixels px (uint8 (H, W, 3)) or a video
    pair's ((2, H, W, 3)); ``lay.two``: two of them, stacked."""
    nb = lay.nb
    ph, pw = 2 * lay.gh, 2 * lay.gw

    def patches(x: np.ndarray) -> np.ndarray:
        return np.concatenate([f.reshape(ph, patch, pw, patch, 3).transpose(0, 2, 4, 1, 3).reshape(ph * pw, -1)
                               for f in (x if x.ndim == 4 else x[None])], 1)
    q = (np.concatenate([patches(x) for x in px]) if lay.two else patches(px)).astype(np.int16) - 128
    sp = lay.slot_patches().reshape(nb, COLS, 64)
    out = {}
    pix = np.zeros((COLS, 64, kp), np.int8)
    pix[..., one] = 1
    for b in range(nb):
        pix[..., :q.shape[1]] = q[sp[b]]
        out[f"{name}.{tag}.{b}"] = np.concatenate([L.int_panels(pix[c]) for c in range(COLS)])
    return out


def band_rows(lay: Layout, rows: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Per band the C tiles of per-patch rows (raster order, any channels in passes of 80) at its slots: DDR images
    <name>.b ([column][C tiles of 64])."""
    sp = lay.slot_patches().reshape(lay.nb, COLS, 64)
    return {f"{name}.{b}": np.concatenate([L.ctiles(rows[sp[b, c]], 64) for c in range(COLS)])
            for b in range(lay.nb)}


def windows_of(nb: int, b: int, c: int, p: int) -> np.ndarray:
    """The windows of band b, column c, half p (8, raster order)."""
    return ((p * nb + b) * COLS + c) * 8 + np.arange(8)


def dw_masks(gh: int, gw: int, nb: int, b: int, two: bool = False) -> np.ndarray:
    """Band b's depthwise conv masks: per column [row 4][half 2][class 4][tap 9][2] uint32 (the 4 core rows' copies;
    per tap of class q = (py, px) the lanes 8 j + c of its half's windows j whose neighbour patch is inside the image:
    bits 0..31 windows 0..3, the second word 4..7). two: an image a half (``Layout.two``): its own windows."""
    out = np.zeros((COLS, 4, 2, 4, 9, 2), np.uint32)
    ph, pw = 2 * gh, 2 * gw
    for c in range(COLS):
        for p in range(2):
            w = (b * COLS + c) * 8 + np.arange(8) if two else windows_of(nb, b, c, p)
            wy, wx = np.divmod(w, gw)
            real = w < gh * gw
            for q in range(4):
                py, px = divmod(q, 2)
                for t in range(9):
                    dy, dx = t // 3 - 1, t % 3 - 1
                    y, x = 2 * wy + py + dy, 2 * wx + px + dx
                    ok = real & (y >= 0) & (y < ph) & (x >= 0) & (x < pw)
                    bits = [sum(0xFF << (8 * (j % 4)) for j in range(4 * h, 4 * h + 4) if ok[j]) for h in range(2)]
                    out[c, :, p, q, t] = bits
    return out


def conv_ddr_bytes(nb: int, channels: int = 1280) -> int:
    """Bytes of a level's gated conv input in DDR: [window][class 4][channels] bf16 of nb bands' windows, ``MARGIN``
    windows before and after (window w at ``conv_window(w)``)."""
    return (nb * WIN + 2 * MARGIN) * 4 * channels * 2


def conv_window(w: int, channels: int = 1280) -> int:
    """The byte offset of window w in the conv input's DDR image."""
    return (MARGIN + w) * 4 * channels * 2


def conv_strip(nb: int, b: int, c: int, p: int, part: str, k: int, channels: int = 1280) -> int:
    """The byte offset in the conv input's DDR image of a strip (``ops.DwConv.strip_fill``'s first window, class and
    block) of band b, column c, half p, at gw = 0: "up" / "down" move by -gw / +gw windows per image (``conv_shift``)."""
    w0 = int(windows_of(nb, b, c, p)[0]) - 1
    q0 = 2 if part == "up" else 0
    return conv_window(w0, channels) + q0 * channels * 2 + k * 160


def conv_shift(part: str, gw: int, channels: int = 1280) -> int:
    """The bytes a strip fill of ``part`` moves for an image gw windows wide."""
    assert gw <= GW_MAX, gw
    return {"mid": 0, "up": -gw, "down": gw}[part] * 4 * channels * 2
