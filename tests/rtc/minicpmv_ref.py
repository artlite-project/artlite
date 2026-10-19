"""References of MiniCPM-V 4.7's vision path (the mmproj of projector ``minicpmv4_7``): llama.cpp goldens and the
encoder in torch.

Goldens: ``golden`` runs an image prompt through llama.cpp's libmtmd on the CPU (``minicpmv_dump.cpp``, built against
the llama.cpp tree of ``model_config.LLAMA_CPP_DIR``): the prompt's chunks (text tokens; image chunks with llama.cpp's
decoder positions, which for MiniCPM-V 4.7 differ from the official ``get_rope_index`` at the markers and after the
image: llama.cpp PR #30510), each image's embeddings, the encoder's tensors per entry (the overview, then the slices
row by row; inp_raw: the entry's pixels) and the logits after every text token. Cached under
``tests/rtc/refs/<model>/vgolden_<hash>/`` (``--refresh-refs`` recomputes).

The encoder (``Encoder``): llama.cpp's ``clip_graph_minicpmv4_6`` (``tools/mtmd/models/minicpmv.cpp``) on the mmproj's
tensors for one entry's pixels (3, H, W): the patch embedding, the learned position embedding over a 70 x 70 grid, the
ViT layers (LayerNorm with bias, 16 heads of 72, GELU tanh), the ViT merger after layer ``insert`` (attention inside
2 x 2 windows, then a 2 x 2 merge: 4 x 1152 -> LayerNorm -> GELU tanh MLP (17216) -> 1152, plus the 4 tokens' mean;
16x files only), the post LayerNorm and the final 2 x 2 merge (4608 -> LayerNorm -> 4608 GELU erf -> 1024), then the
entry's separator rows of ``v.tok_embd_sep``. In float64, or with ``llama=True`` llama.cpp's CPU rounding: the patch
embedding in fp16 (pixels and kernel: ggml_conv_2d's im2col), a matmul against BF16 weights takes its activations as
bf16, GELU tanh goes through ggml's fp16 table.
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
        n_pos, h, w = ph * pw, ph, pw
        for il in range(self.layers):
            x = self._layer(x, il)
            tr[f"layer_out-{il}"] = x
            if il != self.insert:
                continue
            # the ViT merger: attention inside 2 x 2 windows, then the 2 x 2 merge plus the 4 tokens' mean
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
            h, w = h // 2, w // 2
            n_pos = h * w
        x = self._ln(x, "v.post_ln")
        tr["post_ln"] = x
        erf = lambda t: 0.5 * t * (1 + torch.erf(t / math.sqrt(2)))
        x, _ = self._merge(x, h, w, "mm.input_norm", "mm.up", "mm.down", erf)
        tr["merger_out"] = x
        if suffix is not None:
            x = torch.cat([x, self.w("v.tok_embd_sep")[SUFFIX_ROWS[suffix]]])
        return x
