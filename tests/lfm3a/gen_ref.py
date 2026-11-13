"""References of LFM3-Audio's speech output (the mmproj's generator, projector ``lfm3a_gen``): the depthformer, the codec
decoder and the ISTFT in float64 torch, and llama.cpp goldens through libmtmd (``gen_dump.cpp``).

The depthformer (``Depth``: llama.cpp's ``clip_graph_lfm3a_gen::build_gen_code``, transformers'
``Lfm3AudioDepthDecoder``) scores a frame's 12 codebooks from the language model's hidden state h (2048, after its
output norm) and the codes revealed so far: slot cb = ``proj_in``(h)[1024 cb ..] + b, plus its code's embedding
(``embd`` codebook cb's row) where revealed, else ``mask_embd``; 3 bidirectional layers [RMSNorm -> attention (32
query heads, 8 KV heads of 32, q / k RMSNorm per head, NEOX RoPE of theta 1e6 at the slots' positions 0 .. 11, each
head's output times sigmoid(``attn_gate`` n)) | RMSNorm -> SwiGLU 1024 -> 2816 -> 1024] (pre-norm only); then the output
norm, each codebook's ``head_norm`` and its logits embd[cb] x (the head tied to the embeddings; code 2016 ends the
audio). A frame's feedback row (the trunk's next input) is the sum over its codebooks of ``out_embd``[2017 cb + code].

The codec decoder (``Codec``: ``build_gen_wav``, transformers' ``LiquidCodecModel.decode``): per frame the FSQ latents
of its 12 codes (``quant.first.codebook``: 4 a code) -> ``quant.first.out_proj`` 48 -> 256 + b -> ``tfm.in_proj`` 256
-> 512 + b, each frame repeated on 6 positions (an STFT hop each); 8 LFM3 layers of 512 with sandwich norms [RMSNorm ->
operator -> RMSNorm | RMSNorm -> SwiGLU 512 -> 2304 -> 512 -> RMSNorm]: short convs (``conv_pw1`` -> [B; C; x], y = C
conv7(B x), causal over 7 positions, ``conv_pw2``) or attention (8 query heads, 2 KV heads of 64, q / k norms, NEOX RoPE
of theta 1e6 at the stream's position, causal over the last 30 positions, the sigmoid gate 512 -> 8); the output norm
and ``tfm.out_proj`` 512 -> 1282 + b: per position [log magnitude 641 | phase 641]. Streaming state: the position, per
conv layer the last 6 rows of B x, per attention layer the last 29 rows of K / V.

The ISTFT (``Istft``: llama.cpp's ``mtmd_audio_streaming_istft``, transformers' ``LiquidCodecISTFT`` and its streaming
cache): exp(m) e^(i phi) -> irfft (1280, backward) times a periodic Hann window, overlap-add at hop 320 divided by the
window-square envelope; the first 480 samples dropped, a stream's last 960 held until the next frames or the end (its
last 480 dropped there).
"""

import hashlib
import math
import os

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import GenConfig

from .. import common
from ..goldens.runner import build_tool, cached_golden, run_tool
from ..llama_cpp import lib_identity

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_SRC = os.path.join(HERE, "gen_dump.cpp")

THETA = 1e6
N_CB, N_CODES = 12, 2017  # (codebooks, codes a codebook: 2016 FSQ codes and the end of audio)
EOA = N_CODES - 1
LEVELS = (8, 7, 6, 6)
FRAME_STACK, N_FFT, HOP = 6, 1280, 320
WINDOW = 30  # (the codec's attention: positions t - 29 .. t)
N_BINS = N_FFT // 2 + 1
TRIM = (N_FFT - HOP) // 2


class Weights:
    """The mmproj's tensors in numpy's shape order (GGUF's dims reversed), float64 (cached)."""

    def __init__(self, mmproj: str, dtype: torch.dtype = torch.float64):
        self.g = GGUFFile(mmproj)
        self.dtype = dtype
        self._w: dict[str, torch.Tensor] = {}

    def __call__(self, name: str) -> torch.Tensor:
        if name not in self._w:
            t = self.g.tensor(name)
            dims = tuple(reversed(self.g.tensor_info[name][0]))
            self._w[name] = torch.from_numpy(t.dequantize().reshape(dims)).to(self.dtype)
        return self._w[name]

    def has(self, name: str) -> bool:
        return name in self.g.tensor_info


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x (n, heads, hd) turned at positions pos (n,): NEOX pairs (i, i + hd / 2), angle pos theta^(-2 i / hd)."""
    hd = x.shape[-1]
    h = hd // 2
    ang = pos.to(torch.float64)[:, None] * THETA ** (-torch.arange(h, dtype=torch.float64) * 2 / hd)
    c, s = torch.cos(ang)[:, None].to(x.dtype), torch.sin(ang)[:, None].to(x.dtype)
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


def attention(w: Weights, p: str, x: torch.Tensor, heads: int, kv_heads: int, eps: float, pos: torch.Tensor,
              mask: torch.Tensor | None, kv: tuple[torch.Tensor, torch.Tensor] | None = None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LFM3's gated attention of the normed rows x (n, width) at positions pos, after the cached K / V rows kv; mask
    (n, keys) True where a key is kept (None: all). Returns the output (o's input projected) and the whole K, V."""
    n = x.shape[0]
    hd = w(p + "attn_q_norm.weight").numel()
    q = rms((x @ w(p + "attn_q.weight").T).reshape(n, heads, hd), w(p + "attn_q_norm.weight"), eps)
    k = rms((x @ w(p + "attn_k.weight").T).reshape(n, kv_heads, hd), w(p + "attn_k_norm.weight"), eps)
    v = (x @ w(p + "attn_v.weight").T).reshape(n, kv_heads, hd)
    q, k = rope(q, pos), rope(k, pos)
    if kv is not None:
        k, v = torch.cat([kv[0], k]), torch.cat([kv[1], v])
    rep = heads // kv_heads  # (query head h: KV head h // rep)
    s = torch.einsum("nhd,mhd->hnm", q, k.repeat_interleave(rep, 1)) / math.sqrt(hd)
    if mask is not None:
        s = s.masked_fill(~mask[None], float("-inf"))
    o = torch.einsum("hnm,mhd->nhd", torch.softmax(s, -1), v.repeat_interleave(rep, 1))
    g = torch.sigmoid(x @ w(p + "attn_gate.weight").T)  # (n, heads)
    return (o * g[..., None]).reshape(n, heads * hd) @ w(p + "attn_out.weight").T, k, v


def ffn(w: Weights, p: str, x: torch.Tensor) -> torch.Tensor:
    g = x @ w(p + "ffn_gate.weight").T
    return (torch.nn.functional.silu(g) * (x @ w(p + "ffn_up.weight").T)) @ w(p + "ffn_down.weight").T


class Depth:
    """The depthformer from an mmproj file (module docstring), float64."""

    def __init__(self, mmproj: str | Weights):
        self.w = w = mmproj if isinstance(mmproj, Weights) else Weights(mmproj)
        self.cfg = c = GenConfig.from_gguf(w.g)
        assert c.projector == "lfm3a_gen", c.projector
        self.H, self.eps = c.hidden, c.eps

    def inputs(self, h: np.ndarray | torch.Tensor, codes: np.ndarray) -> torch.Tensor:
        """The slots' inputs x0 (12, 1024) of hidden state h (2048) and codes (12: -1 masked)."""
        w = self.w
        hh = torch.as_tensor(np.asarray(h, np.float64), dtype=w.dtype)
        P = (w("a.gen.code.proj_in.weight") @ hh + w("a.gen.code.proj_in.bias").reshape(-1)).reshape(N_CB, self.H)
        emb = w("a.gen.code.embd.weight")  # (12, 2017, 1024)
        rows = torch.stack([emb[cb, int(c)] if c >= 0 else w("a.gen.code.mask_embd").reshape(-1)
                            for cb, c in enumerate(codes)])
        return P + rows

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """The 3 layers over the slots x (12, 1024)."""
        w, c = self.w, self.cfg
        pos = torch.arange(N_CB)
        for il in range(c.layers):
            p = f"a.gen.code.blk.{il}."
            x = x + attention(w, p, rms(x, w(p + "ln1.weight"), self.eps), c.heads, c.kv_heads, self.eps, pos, None)[0]
            x = x + ffn(w, p, rms(x, w(p + "ln2.weight"), self.eps))
        return x

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """The logits (12, 2017) of the trunk's output x."""
        w = self.w
        x = rms(x, w("a.gen.code.output_norm.weight"), self.eps)
        x = rms(x, w("a.gen.code.head_norm.weight"), self.eps)  # (12, 1024): per codebook
        return torch.einsum("cvd,cd->cv", w("a.gen.code.embd.weight"), x)

    def logits(self, h: np.ndarray | torch.Tensor, codes: np.ndarray) -> np.ndarray:
        """A pass's logits (12, 2017) float64 of hidden state h and the codes revealed (12: -1 masked)."""
        return self.head(self.trunk(self.inputs(h, codes))).double().numpy()

    def feedback(self, codes: np.ndarray) -> np.ndarray:
        """A frame's feedback row (2048) float64: the sum of its codes' ``out_embd`` rows."""
        ob = self.w("a.gen.code.out_embd.weight")  # (24204, 2048)
        return ob[torch.as_tensor(np.arange(N_CB) * N_CODES + np.asarray(codes))].sum(0).double().numpy()


def fsq_latents(codes: np.ndarray) -> np.ndarray:
    """A code's 4 FSQ latents ((idx // basis) % level - level // 2) / (level // 2) of codes (...) -> (..., 4)."""
    lv = np.array(LEVELS)
    basis = np.cumprod([1, *LEVELS[:-1]])
    c = np.asarray(codes)[..., None]
    return ((c // basis) % lv - lv // 2) / (lv // 2)


class Codec:
    """The codec decoder from an mmproj file (module docstring), float64 (or ``dtype``: bfloat16 = HF's codec's)."""

    def __init__(self, mmproj: str | Weights, dtype: torch.dtype = torch.float64):
        self.w = w = mmproj if isinstance(mmproj, Weights) else Weights(mmproj, dtype)
        self.dtype = w.dtype
        self.eps = 1e-5
        self.layers = sum(1 for il in range(64) if w.has(f"a.gen.wav.tfm.blk.{il}.ln1.weight"))
        p = "a.gen.wav.tfm.blk.{}."
        att = next(il for il in range(self.layers) if not self.is_conv(il))
        hd = w(p.format(att) + "attn_q_norm.weight").numel()
        self.heads = w(p.format(att) + "attn_q.weight").shape[0] // hd
        self.kv_heads = w(p.format(att) + "attn_k.weight").shape[0] // hd
        self.hd = hd
        self.width = w("a.gen.wav.tfm.in_proj.weight").shape[0]

    def is_conv(self, il: int) -> bool:
        return self.w.has(f"a.gen.wav.tfm.blk.{il}.conv_dw.weight")

    def zero_state(self) -> dict:
        z = lambda *s: torch.zeros(*s, dtype=self.dtype)
        st: dict = {"pos": 0}
        for il in range(self.layers):
            st[il] = z(6, self.width) if self.is_conv(il) else (z(0, self.kv_heads, self.hd), z(0, self.kv_heads, self.hd))
        return st

    def embed(self, codes: np.ndarray) -> torch.Tensor:
        """The frames' decoder inputs (frames, 512) of codes (frames, 12)."""
        w = self.w
        q = torch.as_tensor(fsq_latents(codes).reshape(len(codes), -1), dtype=self.dtype)
        q = q.to(torch.bfloat16).to(self.dtype) if self.dtype == torch.bfloat16 else q
        e = q @ w("a.gen.wav.quant.first.out_proj.weight").T + w("a.gen.wav.quant.first.out_proj.bias").reshape(-1)
        return e @ w("a.gen.wav.tfm.in_proj.weight").T + w("a.gen.wav.tfm.in_proj.bias").reshape(-1)

    def frames(self, codes: np.ndarray, state: dict | None = None) -> np.ndarray:
        """The STFT frames (6 frames, 1282) float64 [log magnitude | phase] of codes (frames, 12) after the stream's
        state (updated in place; None: a stream's start)."""
        w, eps = self.w, self.eps
        st = self.zero_state() if state is None else state
        x = self.embed(np.asarray(codes)).repeat_interleave(FRAME_STACK, 0)
        n = x.shape[0]
        pos = torch.arange(st["pos"], st["pos"] + n)
        for il in range(self.layers):
            p = f"a.gen.wav.tfm.blk.{il}."
            h = rms(x, w(p + "ln1.weight"), eps)
            if self.is_conv(il):
                bcx = h @ w(p + "conv_pw1.weight").T
                b, c, xx = bcx[:, :self.width], bcx[:, self.width:2 * self.width], bcx[:, 2 * self.width:]
                bx = torch.cat([st[il], b * xx])
                k = w(p + "conv_dw.weight")  # (512, 7): tap j reads row t + j - 6
                conv = sum(k[:, j] * bx[j:j + n] for j in range(7))
                st[il] = bx[-6:]
                y = (c * conv) @ w(p + "conv_pw2.weight").T
            else:
                K0, V0 = st[il]
                m = K0.shape[0]
                kpos = torch.arange(-m, n)
                qpos = torch.arange(n)
                mask = (kpos[None] <= qpos[:, None]) & (kpos[None] > qpos[:, None] - WINDOW)
                y, K, V = attention(w, p, h, self.heads, self.kv_heads, eps, pos, mask, (K0, V0))
                st[il] = (K[-(WINDOW - 1):], V[-(WINDOW - 1):])
            x = x + rms(y, w(p + "attn_post_norm.weight"), eps)
            x = x + rms(ffn(w, p, rms(x, w(p + "ln2.weight"), eps)), w(p + "ffn_post_norm.weight"), eps)
        st["pos"] += n
        x = rms(x, w("a.gen.wav.tfm.output_norm.weight"), eps)
        return (x @ w("a.gen.wav.tfm.out_proj.weight").T + w("a.gen.wav.tfm.out_proj.bias").reshape(-1)).double().numpy()


class Istft:
    """The streaming ISTFT (module docstring): ``push(frames)`` (STFT frames (n, 1282) [log magnitude | phase]) -> the
    samples they complete; ``flush()`` the rest."""

    def __init__(self):
        self.win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # (periodic Hann)
        self.buf = np.zeros(N_FFT)
        self.env = np.zeros(N_FFT)
        self.drop = TRIM

    def push(self, frames: np.ndarray) -> np.ndarray:
        f = np.asarray(frames, np.float64)
        spec = np.exp(f[:, :N_BINS]) * np.exp(1j * f[:, N_BINS:])
        seg = np.fft.irfft(spec, N_FFT, axis=1) * self.win
        out = []
        for s in seg:
            self.buf += s
            self.env += self.win ** 2
            e = self.env[:HOP]
            out.append(np.where(e > 1e-8, self.buf[:HOP] / np.where(e > 1e-8, e, 1), self.buf[:HOP]))
            self.buf = np.concatenate([self.buf[HOP:], np.zeros(HOP)])
            self.env = np.concatenate([self.env[HOP:], np.zeros(HOP)])
        y = np.concatenate(out) if out else np.zeros(0)
        k = min(self.drop, y.size)
        self.drop -= k
        return y[k:]

    def flush(self) -> np.ndarray:
        n = N_FFT - HOP
        e = self.env[:n]
        y = np.where(e > 1e-8, self.buf[:n] / np.where(e > 1e-8, e, 1), self.buf[:n])
        return y[:max(0, n - TRIM)]


def pcm(codec: Codec, codes: np.ndarray, chunk: int = 0) -> np.ndarray:
    """A stream's samples (float64) of codes (frames, 12): the codec in calls of ``chunk`` frames (0: one call), the
    ISTFT, the flush."""
    st, ist = codec.zero_state(), Istft()
    n = len(codes)
    step = chunk or max(n, 1)
    out = [ist.push(codec.frames(codes[i:i + step], st)) for i in range(0, n, step)]
    return np.concatenate(out + [ist.flush()])


# ----------------------------------------------------------------------
# llama.cpp goldens
# ----------------------------------------------------------------------


class GenGolden:
    """The outputs of one ``lfm3a_gen_dump`` run (``gen_dump.cpp``'s files): ``h``, ``prompt_ids``, ``logits(tag)``,
    ``fb(tag)``, ``hf(i)``, ``codes`` and ``hg(i)`` (sampled frames), ``pcm``, ``pcm_calls``, ``tensor(name, graph)``."""

    def __init__(self, path: str):
        self.path = path

    def _load(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self.path, f"{name}.npy"))

    @property
    def h(self) -> np.ndarray:
        return self._load("h")

    @property
    def prompt_ids(self) -> np.ndarray:
        return self._load("prompt_ids")

    def logits(self, tag: int | str) -> np.ndarray:
        return self._load(f"logits-{tag}")

    def fb(self, tag: int | str) -> np.ndarray:
        return self._load(f"fb-{tag}")

    def hf(self, i: int) -> np.ndarray:
        return self._load(f"hf-{i}")

    def hg(self, i: int) -> np.ndarray:
        return self._load(f"hg-{i}")

    @property
    def codes(self) -> np.ndarray:
        return self._load("codes")

    @property
    def pcm(self) -> np.ndarray:
        return self._load("pcm")

    @property
    def pcm_calls(self) -> np.ndarray:
        return self._load("pcm_calls")

    def tensor(self, name: str, graph: int) -> np.ndarray:
        return self._load(f"{name}.g{graph}")


def golden(model: str, mmproj: str, *, prompt: str | None = None, h: np.ndarray | None = None,
           passes: np.ndarray | None = None, feed: np.ndarray | None = None, frames: np.ndarray | None = None,
           chunk: int = 4, gen: int = 0, seed: int = 0, names: str = "", threads: int = 16) -> GenGolden:
    """llama.cpp's generator on the CPU (``gen_dump.cpp``'s options: a prompt's hidden state, passes of codes on it or on
    ``h``, ``gen`` frames sampled after the prompt (``seed``), teacher-forced frames, frames through the codec in calls
    of ``chunk``), cached under
    ``tests/refs/<model>/ggolden_<hash>/`` (``--refresh-refs`` recomputes)."""
    arrs = {"h": None if h is None else np.ascontiguousarray(h, np.float32),
            "passes": None if passes is None else np.ascontiguousarray(passes, np.int32),
            "feed": None if feed is None else np.ascontiguousarray(feed, np.int32),
            "frames": None if frames is None else np.ascontiguousarray(frames, np.int32)}
    key = (common.model_key(model), common.model_key(mmproj), prompt, chunk, gen, seed, names, lib_identity(),
           lib_identity("libmtmd.so"),
           {k: None if v is None else hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrs.items()})

    def run(tmp: str) -> None:
        cmd = [build_tool("lfm3a_gen_dump", DUMP_SRC, mtmd=True), "-m", model, "--mmproj", mmproj, "-o", tmp,
               "--threads", str(threads), "--chunk", str(chunk)]
        if gen:
            cmd += ["--gen", str(gen), "--seed", str(seed)]
        if prompt is not None:
            with open(os.path.join(tmp, "prompt.txt"), "w") as f:
                f.write(prompt)
            cmd += ["-p", os.path.join(tmp, "prompt.txt")]
        for k, v in arrs.items():
            if v is not None:
                v.tofile(os.path.join(tmp, f"{k}.bin"))
                cmd += [f"--{k}", os.path.join(tmp, f"{k}.bin")]
        if names:
            cmd += ["-n", names]
        run_tool(cmd)
        for k in arrs:
            if os.path.exists(os.path.join(tmp, f"{k}.bin")):
                os.remove(os.path.join(tmp, f"{k}.bin"))
        open(os.path.join(tmp, "done"), "w").close()

    return GenGolden(cached_golden(model, "ggolden", key, DUMP_SRC, run, marker="done"))
