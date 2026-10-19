"""References of MiniCPM-V 4.7's vision path (the mmproj of projector ``minicpmv4_7``): llama.cpp goldens and the
encoder in torch.

Goldens: ``golden`` runs an image prompt through llama.cpp's libmtmd on the CPU (``minicpmv_dump.cpp``, built against
the llama.cpp tree of ``model_config.LLAMA_CPP_DIR``): the prompt's chunks (text tokens; image chunks with llama.cpp's
decoder positions, which for MiniCPM-V 4.7 differ from the official ``get_rope_index`` at the markers and after the
image: llama.cpp PR #30510), each image's embeddings, the encoder's tensors per entry (the overview, then the slices
row by row; inp_raw: the entry's pixels) and the logits after every text token. Cached under
``tests/rtc/refs/<model>/vgolden_<hash>/`` (``--refresh-refs`` recomputes). ``seq_golden``: the language model alone on
a sequence of embedding rows at given (the official) positions.

The encoder (``Encoder``): llama.cpp's ``clip_graph_minicpmv4_6`` (``tools/mtmd/models/minicpmv.cpp``) on the mmproj's
tensors for one entry's pixels (3, H, W): the patch embedding, the learned position embedding over a 70 x 70 grid, the
ViT layers (LayerNorm with bias, 16 heads of 72, GELU tanh), the ViT merger after layer ``insert`` (attention inside
2 x 2 windows, then a 2 x 2 merge: 4 x 1152 -> LayerNorm -> GELU tanh MLP (17216) -> 1152, plus the 4 tokens' mean;
16x files only), the post LayerNorm and the final 2 x 2 merge (4608 -> LayerNorm -> 4608 GELU erf -> 1024), then the
entry's separator rows of ``v.tok_embd_sep``. In float64, or with ``llama=True`` llama.cpp's CPU rounding: the patch
embedding in fp16 (pixels and kernel: ggml_conv_2d's im2col), a matmul against BF16 weights takes its activations as
bf16, GELU tanh goes through ggml's fp16 table. ``NpuEncoder``: the encoder at the NPU's rounding points
(``artlite.operators.vision.minicpmv`` on sd15's design).
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import LLAMA_CPP_DIR, ClipConfig
from artlite.operators.base import CACHE_DIR
from artlite.operators.vision import minicpmv as vision

from . import common
from .qwen35_ref import _lib_identity, _llama_lib

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_SRC = os.path.join(HERE, "minicpmv_dump.cpp")
# the separator rows of v.tok_embd_sep after an entry by its place (clip-impl.h clip_suffix_rows): 0 </image>,
# 1 <slice>, 2 </slice>, 3 "\n"
SUFFIX_ROWS = {"ov": [0], "ov_slice": [0, 1], "slice": [2, 1], "row_end": [2, 3, 1], "last": [2]}


# ----------------------------------------------------------------------
# llama.cpp goldens
# ----------------------------------------------------------------------


def _mtmd_identity() -> str:
    lib = os.path.realpath(os.path.join(os.path.dirname(_llama_lib()), "libmtmd.so"))
    return f"{lib}:{os.path.getsize(lib)}:{int(os.path.getmtime(lib))}"


def dump_tool() -> str:
    """The path of the built ``minicpmv_dump`` (compiled on first use into the artlite cache)."""
    with open(DUMP_SRC, "rb") as f:
        key = hashlib.sha256(f.read() + (_lib_identity() + _mtmd_identity()).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"minicpmv_dump_{key}")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        bindir = os.path.dirname(_llama_lib())
        tmp = f"{path}.{os.getpid()}.tmp"
        subprocess.run(["g++", "-O2", "-std=c++17", "-Wno-deprecated-declarations", f"-I{LLAMA_CPP_DIR}/include",
                        f"-I{LLAMA_CPP_DIR}/ggml/include", f"-I{LLAMA_CPP_DIR}/tools/mtmd", DUMP_SRC, "-o", tmp,
                        f"-L{bindir}", "-lmtmd", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{bindir}"], check=True)
        os.replace(tmp, path)
    return path


class VisionGolden:
    """The outputs of one ``minicpmv_dump`` run: ``chunks`` (the chunks.json list), ``entries``, ``embd(i)``,
    ``tensor(name, entry, layer)``, ``logits`` and ``logit_tokens`` (the sequence index of each row)."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "chunks.json")) as f:
            j = json.load(f)
        self.chunks, self.logit_tokens, self.entries = j["chunks"], j["logit_tokens"], j["entries"]

    def embd(self, i: int = 0) -> np.ndarray:
        return np.load(os.path.join(self.path, f"embd-{i}.npy"))

    def tensor(self, name: str, entry: int, layer: int | None = None) -> np.ndarray:
        """``name`` ("layer_out", ...; "Kcur.1": the second of that name) of the encoder's entry, of a layer."""
        base, dot, dup = name.rpartition(".")
        if not (dot and dup.isdigit()):
            base, dup = name, ""
        stem = base if layer is None else f"{base}-{layer}"
        path = os.path.join(self.path, stem + (f".{dup}" if dup else "") + f".e{entry}.npy")
        if not os.path.exists(path):
            have = sorted(f for f in os.listdir(self.path) if f.endswith(f".e{entry}.npy"))
            raise KeyError(f"{name} {layer} e{entry}: not dumped (have {have[:40]})")
        return np.load(path)

    @property
    def logits(self) -> np.ndarray:
        return np.load(os.path.join(self.path, "logits.npy"), mmap_mode="r")


def golden(model: str, mmproj: str, image: str, prompt: str, *, answer: list[int] | tuple[int, ...] = (),
           names: str = "", embd: np.ndarray | None = None, threads: int = 16) -> VisionGolden:
    """llama.cpp's vision path for an image and a prompt (the text with ``<__media__>`` where the image goes) and
    teacher-forced answer tokens; ``names``: a regex of the encoder tensors to dump (without the "-<il>" suffix);
    ``embd``: the image's embeddings (tokens, n_embd) the language model takes instead of the encoder's."""
    ans = np.ascontiguousarray(answer, np.int32)
    emb = None if embd is None else np.ascontiguousarray(embd, np.float32)
    with open(image, "rb") as f:
        img_hash = hashlib.sha256(f.read()).hexdigest()
    key = (common.model_key(model), common.model_key(mmproj), img_hash, prompt, hashlib.sha256(ans.tobytes()).hexdigest(),
           names, None if emb is None else hashlib.sha256(emb.tobytes()).hexdigest(), _lib_identity(), _mtmd_identity())
    h = hashlib.sha256(repr(key).encode())
    with open(DUMP_SRC, "rb") as f:
        h.update(f.read())
    folder = common.model_refs(model)
    path = os.path.join(folder, f"vgolden_{h.hexdigest()[:16]}")
    if os.path.exists(os.path.join(path, "meta.json")) and not common.REFRESH_REFS:
        return VisionGolden(path)
    os.makedirs(folder, exist_ok=True)
    tool = dump_tool()
    tmp = tempfile.mkdtemp(dir=folder, prefix=".tmp_")
    try:
        with open(os.path.join(tmp, "prompt.txt"), "w") as f:
            f.write(prompt)
        cmd = [tool, "-m", model, "--mmproj", mmproj, "--image", image, "-p", os.path.join(tmp, "prompt.txt"), "-o",
               tmp, "--threads", str(threads)]
        if ans.size:
            ans.tofile(os.path.join(tmp, "answer.i32"))
            cmd += ["-a", os.path.join(tmp, "answer.i32")]
        if names:
            cmd += ["-n", names]
        if emb is not None:
            emb.tofile(os.path.join(tmp, "embd.f32"))
            cmd += ["-e", os.path.join(tmp, "embd.f32")]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"{' '.join(cmd)}: exit {r.returncode}\n{r.stderr[-4000:]}")
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return VisionGolden(path)


def seq_golden(model: str, rows: np.ndarray, pos: np.ndarray, first: int = 0, *, threads: int = 16) -> VisionGolden:
    """llama.cpp's language model on a whole sequence of embedding rows (n, n_embd) (the text tokens' too) at the
    official M-RoPE positions ``pos`` (n, 3) (T, H, W), causal in token order (``minicpmv_dump --seq``): ``logits``
    of tokens ``first`` .. n - 1."""
    rows = np.ascontiguousarray(rows, np.float32)
    pos = np.ascontiguousarray(pos, np.int32)
    assert pos.shape == (rows.shape[0], 3), (rows.shape, pos.shape)
    key = (common.model_key(model), hashlib.sha256(rows.tobytes() + pos.tobytes()).hexdigest(), first, _lib_identity(),
           _mtmd_identity())
    h = hashlib.sha256(repr(key).encode())
    with open(DUMP_SRC, "rb") as f:
        h.update(f.read())
    folder = common.model_refs(model)
    path = os.path.join(folder, f"vseq_{h.hexdigest()[:16]}")
    if os.path.exists(os.path.join(path, "meta.json")) and not common.REFRESH_REFS:
        return VisionGolden(path)
    os.makedirs(folder, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=folder, prefix=".tmp_")
    try:
        rows.tofile(os.path.join(tmp, "seq.f32"))
        pos.tofile(os.path.join(tmp, "pos.i32"))
        cmd = [dump_tool(), "-m", model, "--seq", os.path.join(tmp, "seq.f32"), "--pos", os.path.join(tmp, "pos.i32"),
               "--from", str(first), "-o", tmp, "--threads", str(threads)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"{' '.join(cmd)}: exit {r.returncode}\n{r.stderr[-4000:]}")
        for f in ("seq.f32", "pos.i32"):
            os.remove(os.path.join(tmp, f))
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return VisionGolden(path)


# ----------------------------------------------------------------------
# The encoder
# ----------------------------------------------------------------------


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_table(x: torch.Tensor) -> torch.Tensor:
    """ggml's CPU GELU (tanh): an fp16 table of the fp16-rounded input (|x| < 10), the identity / 0 outside."""
    xh = x.to(torch.float16).to(torch.float32)
    y = _gelu_tanh(xh).to(torch.float16).to(x.dtype)
    return torch.where(x <= -10, torch.zeros_like(x), torch.where(x >= 10, x, y))


class Encoder:
    """MiniCPM-V 4.7's vision encoder from its mmproj file (module docstring); ``llama``: llama.cpp's CPU rounding."""

    def __init__(self, mmproj: str, *, llama: bool = False):
        self.g = g = GGUFFile(mmproj)
        self.cfg = cfg = ClipConfig.from_gguf(g)
        assert cfg.projector == "minicpmv4_7", cfg.projector
        self.hidden, self.heads, self.hd = cfg.hidden, cfg.heads, cfg.head_dim
        self.layers, self.patch, self.eps = cfg.layers, cfg.patch_size, cfg.eps
        self.merge, self.insert = cfg.merge, cfg.merger_layer  # (the ViT merger and the final merge, or the final only)
        self.llama = llama
        self.dtype = torch.float32 if llama else torch.float64
        self._w: dict[str, torch.Tensor] = {}

    def w(self, name: str) -> torch.Tensor:
        """A tensor as (rows, cols) of self.dtype (cached)."""
        if name not in self._w:
            t = self.g.tensor(name)
            self._w[name] = t.dequantize().reshape(t.rows, t.cols).to(self.dtype)
        return self._w[name]

    def has(self, name: str) -> bool:
        return name in self.g

    def _mm(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        w = self.w(name)
        if self.llama and self.g.tensor_info[name][1] == "BF16":  # (ggml's BF16 dot products: bf16 activations)
            x = _bf16(x)
        y = x @ w.T
        return y + self.w(bias).reshape(-1) if bias else y

    def _ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.w(prefix + ".weight").reshape(-1) \
            + self.w(prefix + ".bias").reshape(-1)

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_table(x) if self.llama else _gelu_tanh(x)

    def _attn(self, x: torch.Tensor, p: str, mask: torch.Tensor | None = None) -> torch.Tensor:
        n = x.shape[0]
        q = self._mm(x, p + "attn_q.weight", p + "attn_q.bias").reshape(n, self.heads, self.hd).transpose(0, 1)
        k = self._mm(x, p + "attn_k.weight", p + "attn_k.bias").reshape(n, self.heads, self.hd).transpose(0, 1)
        v = self._mm(x, p + "attn_v.weight", p + "attn_v.bias").reshape(n, self.heads, self.hd).transpose(0, 1)
        s = q @ k.transpose(1, 2) / math.sqrt(self.hd)
        if mask is not None:
            s = s + mask
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(n, self.hidden)
        return self._mm(o, p + "attn_out.weight", p + "attn_out.bias")

    def _layer(self, x: torch.Tensor, il: int) -> torch.Tensor:
        p = f"v.blk.{il}."
        x = x + self._attn(self._ln(x, p + "ln1"), p)
        h = self._ln(x, p + "ln2")
        h = self._mm(self._gelu(self._mm(h, p + "ffn_up.weight", p + "ffn_up.bias")), p + "ffn_down.weight",
                     p + "ffn_down.bias")
        return x + h

    @staticmethod
    def positions(ph: int, pw: int) -> np.ndarray:
        """The learned position embedding's rows of a ph x pw patch grid (clip.cpp: buckets of a 70 x 70 grid)."""
        bh = np.floor(70.0 * np.arange(ph) / ph).astype(np.int64)
        bw = np.floor(70.0 * np.arange(pw) / pw).astype(np.int64)
        return (bh[:, None] * 70 + bw[None, :]).reshape(-1)

    @staticmethod
    def ds_index(h: int, w: int) -> list[np.ndarray]:
        """The 2 x 2 merge's gather indices of an h x w grid (row-major): the 4 tokens (0, 0), (0, 1), (1, 0), (1, 1)
        of each 2 x 2 block, the blocks row-major."""
        i, j = np.meshgrid(np.arange(h // 2), np.arange(w // 2), indexing="ij")
        return [((2 * i + r) * w + (2 * j + c)).reshape(-1) for r in (0, 1) for c in (0, 1)]

    def _merge(self, x: torch.Tensor, h: int, w: int, ln: str, up: str, down: str, gelu: object) -> torch.Tensor:
        parts = [x[torch.from_numpy(ix)] for ix in self.ds_index(h, w)]
        y = self._ln(torch.cat(parts, -1), ln)
        y = self._mm(gelu(self._mm(y, up + ".weight", up + ".bias")), down + ".weight", down + ".bias")
        return y, parts

    def vit_merger(self, x: torch.Tensor, h: int, w: int, tr: dict | None = None) -> torch.Tensor:
        """The ViT merger of the residual x (h w tokens, raster order): attention inside 2 x 2 windows, then the 2 x 2
        merge plus the 4 tokens' mean -> (h / 2 w / 2, hidden)."""
        tr = tr if tr is not None else {}
        n_pos = h * w
        win = np.concatenate([ix[:, None] for ix in self.ds_index(h, w)], 1).reshape(-1)  # (window-major)
        inv = np.empty_like(win)
        inv[win] = np.arange(n_pos)
        mask = torch.full((n_pos, n_pos), -torch.inf, dtype=self.dtype)
        for b in range(n_pos // 4):
            mask[4 * b:4 * b + 4, 4 * b:4 * b + 4] = 0
        a = self._attn(self._ln(x, "v.vit_merger.ln1")[torch.from_numpy(win)], "v.vit_merger.", mask)
        x = x + a[torch.from_numpy(inv)]
        tr["vit_merger_attn_residual"] = x
        y, parts = self._merge(x, h, w, "v.vit_merger.ds_ln", "v.vit_merger.ds_ffn_up", "v.vit_merger.ds_ffn_down",
                               self._gelu)
        x = y + (parts[0] + parts[1] + parts[2] + parts[3]) * 0.25
        tr["vit_merger_ds_out"] = x
        return x

    def forward(self, pixels: np.ndarray | torch.Tensor, suffix: str | None = None,
                trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """One entry: pixels (3, H, W) (normalized, as llama.cpp's inp_raw) -> its embeddings (tokens, 1024), then the
        separator rows of its place (``SUFFIX_ROWS``: "ov", "ov_slice", "slice", "row_end", "last"; None: none).
        ``trace``: filled with the intermediates by llama.cpp's names ("pos_embed", "layer_out-<il>",
        "vit_merger_ds_out", "post_ln", "merger_out")."""
        P, H = self.patch, self.hidden
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        ph, pw = x.shape[1] // P, x.shape[2] // P
        kern = self.w("v.patch_embd.weight").reshape(H, 3, P, P)
        if self.llama:  # (ggml_conv_2d: im2col in fp16 for an F32 kernel, the kernel converted for the F16 dot products)
            x, kern = (t.to(torch.float16).to(self.dtype) for t in (x, kern))
        e = torch.nn.functional.conv2d(x[None], kern, stride=P)[0]  # (H, ph, pw)
        x = e.reshape(H, -1).T + self.w("v.patch_embd.bias").reshape(-1)
        x = x + self.w("v.position_embd.weight")[torch.from_numpy(self.positions(ph, pw))]
        tr = trace if trace is not None else {}
        tr["pos_embed"] = x
        h, w = ph, pw
        for il in range(self.layers):
            x = self._layer(x, il)
            tr[f"layer_out-{il}"] = x
            if il != self.insert:
                continue
            x = self.vit_merger(x, h, w, tr)
            h, w = h // 2, w // 2
        x = self._ln(x, "v.post_ln")
        tr["post_ln"] = x
        erf = lambda t: 0.5 * t * (1 + torch.erf(t / math.sqrt(2)))
        x, _ = self._merge(x, h, w, "mm.input_norm", "mm.up", "mm.down", erf)
        tr["merger_out"] = x
        if suffix is not None:
            x = torch.cat([x, self.w("v.tok_embd_sep")[SUFFIX_ROWS[suffix]]])
        return x


def _bfp(x: torch.Tensor) -> torch.Tensor:
    """bfp16ebs8 blocks of 8 along the last axis (RNE)."""
    from artlite.operators import bfp16
    n = x.shape[-1]
    xp = torch.nn.functional.pad(x, (0, -n % 8)) if n % 8 else x
    return torch.from_numpy(bfp16.values(xp.to(torch.float32).numpy())).to(x.dtype)[..., :n]


class NpuEncoder(Encoder):
    """The encoder at the rounding points of sd15's design (``artlite.operators.vision.minicpmv``; float64 between
    them): a GEMM's A as bfp16 panels (the patch embedding's: the pixels' integers, exact) and its weights bfp16 (hi +
    lo those of ``minicpmv.HILO``), C over the whole K from the bf16 bias plus a bf16 residual, rounded once (bf16 C
    tiles: the residual stream, a LayerNorm's input; bfp16 panels: a GEMM's or the attention's input, V^T blocked over
    8 tokens); LayerNorm statistics S1, S2 of the bf16 values in fp32, var = max(S2 / n - mean^2, 0); GELU of bf16
    values to bf16 (act_lut); attention with q (x log2 e / sqrt d) and k single bfp16, P = 2^(s - max) in bf16 as bfp16
    panels, l their sums, o / l as panels."""

    def __init__(self, mmproj: str):
        super().__init__(mmproj)
        self._wq: dict[str, torch.Tensor] = {}

    @staticmethod
    def _round(w: torch.Tensor, name: str) -> torch.Tensor:
        """The weights w (N, K) of the mmproj's tensor ``name`` as the NPU takes them: bfp16 blocks along K (hi + lo:
        ``minicpmv.HILO``)."""
        hi = _bfp(w)
        return hi + _bfp(w - hi) if name in vision.HILO else hi

    def wq(self, name: str) -> torch.Tensor:
        if name not in self._wq:
            self._wq[name] = self._round(self.w(name), name)
        return self._wq[name]

    def gemm(self, a: torch.Tensor, name: str, bias: str | None, res: torch.Tensor | None = None) -> torch.Tensor:
        """a: the A panels' values -> C + bias + residual, unrounded."""
        c = a @ self.wq(name).T
        if bias:
            c = c + _bf16(self.w(bias).reshape(-1))
        return c if res is None else c + res

    def ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        """x: bf16 values -> the LayerNorm's bfp16 panels."""
        x32 = x.to(torch.float32)
        n = x.shape[-1]
        mean = x32.sum(-1, keepdim=True) / n
        var = torch.clamp((x32 * x32).sum(-1, keepdim=True) / n - mean * mean, min=0)
        rstd = 1 / torch.sqrt(var + self.eps)
        y = (x - mean.to(x.dtype)) * rstd.to(x.dtype) * self.w(prefix + ".weight").reshape(-1) \
            + self.w(prefix + ".bias").reshape(-1)
        return _bfp(y)

    def gelu_panels(self, c: torch.Tensor) -> torch.Tensor:
        return _bfp(_bf16(_gelu_tanh(_bf16(c))))

    def attention(self, a: torch.Tensor, p: str) -> torch.Tensor:
        """a: the LayerNorm's panels (n, hidden) -> o / l panels (n, hidden)."""
        n, H, d = a.shape[0], self.heads, self.hd
        q = self.gemm(a, p + "attn_q.weight", p + "attn_q.bias") * (math.log2(math.e) / math.sqrt(d))
        k = self.gemm(a, p + "attn_k.weight", p + "attn_k.bias")
        v = self.gemm(a, p + "attn_v.weight", p + "attn_v.bias")
        q = _bfp(q.reshape(n, H, d).transpose(0, 1))
        k = _bfp(k.reshape(n, H, d).transpose(0, 1))
        v = _bfp(v.reshape(n, H, d).permute(1, 2, 0)).transpose(1, 2)  # (V^T panels: blocks over 8 tokens)
        s = q @ k.transpose(1, 2)  # (log2 units)
        pr = _bfp(_bf16(torch.exp2(s - s.amax(-1, keepdim=True))))
        o = (pr @ v) / pr.sum(-1, keepdim=True)
        return _bfp(o.transpose(0, 1).reshape(n, self.hidden))

    def layer(self, x: torch.Tensor, il: int) -> torch.Tensor:
        """Layer il on the residual x (bf16 values)."""
        p = f"v.blk.{il}."
        o = self.attention(self.ln(x, p + "ln1"), p)
        x = _bf16(self.gemm(o, p + "attn_out.weight", p + "attn_out.bias", x))
        h = self.gelu_panels(self.gemm(self.ln(x, p + "ln2"), p + "ffn_up.weight", p + "ffn_up.bias"))
        return _bf16(self.gemm(h, p + "ffn_down.weight", p + "ffn_down.bias", x))

    def window_attention(self, a: torch.Tensor, p: str) -> torch.Tensor:
        """The ViT merger's attention inside windows of 4 consecutive tokens: a the LayerNorm's panels -> o panels (Q
        (x log2 e / sqrt d), K bfp16 along the dims, V bfp16 over 8 windows, scores fp32, P = 2^(s - max) / l in
        bf16)."""
        n, H, d = a.shape[0], self.heads, self.hd
        q = self.gemm(a, p + "attn_q.weight", p + "attn_q.bias") * (math.log2(math.e) / math.sqrt(d))
        q = _bfp(q.reshape(n // 4, 4, H, d))
        k = _bfp(self.gemm(a, p + "attn_k.weight", p + "attn_k.bias").reshape(n // 4, 4, H, d))
        v = self.gemm(a, p + "attn_v.weight", p + "attn_v.bias").reshape(n // 4, 4, H, d)
        nw = n // 4
        vp = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 0, 0, -nw % 8)).permute(1, 2, 3, 0)  # (s, h, d, window)
        v = _bfp(vp)[..., :nw].permute(3, 0, 1, 2)
        s = torch.einsum("wihd,wjhd->whij", q, k)
        pr = torch.exp2(s - s.amax(-1, keepdim=True))
        pr = _bf16(pr / pr.sum(-1, keepdim=True))
        o = torch.einsum("whij,wjhd->wihd", pr, v)
        return _bfp(o.reshape(n, self.hidden))

    def vit_merger(self, x: torch.Tensor, h: int, w: int, tr: dict | None = None, *,
                   mean_panels: bool = True) -> torch.Tensor:
        """``Encoder.vit_merger`` at the NPU's rounding points (mean_panels: the mean of 4 from the tokens' bfp16
        panels, summed with the down GEMM; else their bf16 mean)."""
        tr = tr if tr is not None else {}
        H = self.hidden
        win = np.concatenate([ix[:, None] for ix in self.ds_index(h, w)], 1).reshape(-1)  # (window order)
        xw = x[torch.from_numpy(win)]
        o = self.window_attention(self.ln(xw, "v.vit_merger.ln1"), "v.vit_merger.")
        xw = _bf16(self.gemm(o, "v.vit_merger.attn_out.weight", "v.vit_merger.attn_out.bias", xw))
        cat = xw.reshape(-1, 4 * H)
        g = self.gelu_panels(self.gemm(self.ln(cat, "v.vit_merger.ds_ln"), "v.vit_merger.ds_ffn_up.weight",
                                       "v.vit_merger.ds_ffn_up.bias"))
        mean4 = _bfp(xw).reshape(-1, 4, H).sum(1) / 4 if mean_panels else _bf16(xw.reshape(-1, 4, H).mean(1))
        x = _bf16(self.gemm(g, "v.vit_merger.ds_ffn_down.weight", "v.vit_merger.ds_ffn_down.bias", mean4))
        tr["vit_merger_ds_out"] = x
        return x

    def forward(self, pixels: np.ndarray | torch.Tensor, suffix: str | None = None,
                trace: dict[str, torch.Tensor] | None = None, *, mean_panels: bool = True) -> torch.Tensor:
        """``Encoder.forward`` at the NPU's rounding points (tokens in window order through the ViT merger: the merge
        of 4 consecutive tokens). mean_panels: the merger's mean of 4 from the tokens' bfp16 panels, summed with the
        down GEMM (else their bf16 mean)."""
        P, H = self.patch, self.hidden
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        ph, pw = x.shape[1] // P, x.shape[2] // P
        # (the patches' inputs (c, ky, kx) as the integers q = p - 128 of the 8-bit pixels p (exact panels); the
        # normalization x = a q + o folded into the weights, the offset sum_k w_k o_k on a constant input of its own)
        mean, std = (torch.tensor(v, dtype=self.dtype) for v in (self.cfg.mean, self.cfg.std))
        q = torch.round((x * std[:, None, None] + mean[:, None, None]) * 255) - 128
        cols = q.reshape(3, ph, P, pw, P).permute(1, 3, 0, 2, 4).reshape(ph * pw, 3 * P * P)
        a = (1 / (255 * std)).repeat_interleave(P * P)
        o = 128 * a - (mean / std).repeat_interleave(P * P)
        w, name = self.w("v.patch_embd.weight").reshape(H, -1), "v.patch_embd.weight"
        c = cols @ self._round(w * a, name).T + self._round((w @ o)[:, None], name)[:, 0] \
            + _bf16(self.w("v.patch_embd.bias").reshape(-1))
        x = _bf16(c + _bf16(self.w("v.position_embd.weight")[torch.from_numpy(self.positions(ph, pw))]))
        tr = trace if trace is not None else {}
        tr["pos_embed"] = x
        h, w = ph, pw
        for il in range(self.layers):
            x = self.layer(x, il)
            tr[f"layer_out-{il}"] = x
            if il != self.insert:
                continue
            x = self.vit_merger(x, h, w, tr, mean_panels=mean_panels)
            h, w = h // 2, w // 2
        y = _bf16(self.ln(x, "v.post_ln"))  # (panels, then their values as C tiles)
        tr["post_ln"] = y
        idx = np.concatenate([ix[:, None] for ix in self.ds_index(h, w)], 1).reshape(-1)
        cat = y[torch.from_numpy(idx)].reshape(-1, 4 * H)
        up = self.gemm(self.ln(cat, "mm.input_norm"), "mm.up.weight", "mm.up.bias")
        z = _bf16(up)
        g = _bfp(_bf16(0.5 * z * (1 + torch.erf(z / math.sqrt(2)))))
        out = _bf16(self.gemm(g, "mm.down.weight", "mm.down.bias"))
        tr["merger_out"] = out
        if suffix is not None:
            out = torch.cat([out, self.w("v.tok_embd_sep")[SUFFIX_ROWS[suffix]]])
        return out
