"""References of LFM3-Audio's speech encoder (the mmproj of projector ``lfm3a``): the encoder in float64 torch, and
llama.cpp goldens through libmtmd (``tests.minicpmv.ref.golden``: the dump tool runs any mmproj, an audio file as
its media; ``embd(0)`` the clip's rows, ``tensor("inp_raw", 0)`` its log-mel spectrogram).

The encoder (``Encoder``: llama.cpp's ``clip_graph_lfm3a``, ``tools/mtmd/models/lfm3a.cpp``; transformers'
``Lfm3AudioEncoder`` + ``Lfm3AudioProjector``) on the mmproj's tensors for a clip's log-mel spectrogram (128, frames):
- the downsampler, 8x in time and frequency, every conv 3 x 3 of stride 2 and padding 1 (zeros around each stage's
  input): conv_in (``a.conv1d.0``, 1 -> 256 channels) + b, SiLU; conv_mid (``a.conv1d.2``, depthwise) + b, lin_mid
  (``a.conv1d.3``, 1 x 1) + b, SiLU; conv_end (``.5``) + b, lin_end (``.6``) + b, SiLU; flattened channel-major (c 16 +
  f) and lin_out (``a.pre_encode.out``, 4096 -> 768) + b: a row per 8 frames;
- 17 LFM2 layers [RMSNorm -> operator | RMSNorm -> SwiGLU 768 -> 1792 -> 768] with residuals (eps 1e-5, no biases):
  short-conv layers (in_proj ``conv_pw1`` 768 -> [B; C; x], y = C conv(B x) with the causal 3-tap conv ``conv_dw``,
  out_proj ``conv_pw2``) and attention layers (8 query heads, 4 KV heads of 96; q / k RMSNorm per head; NEOX RoPE of
  theta 1e6 on all 96 dims at the row's position; causal over the last ``window`` (32) rows, scale 1 / sqrt(96));
- the final RMSNorm (``a.post_ln``), the projector: LayerNorm (``mm.a.mlp.0``, eps 1e-5) -> ``mm.a.mlp.1`` (768 ->
  2048) + b -> GELU (erf) -> ``mm.a.mlp.3`` + b.

Streaming (``Stream``): the state a clip carries from chunk to chunk (the downsampler's left context as the last 8 mel
frames, each conv layer's last 2 rows of B x, each attention layer's last window - 1 K / V rows, the position); chunks
of whole rows (multiples of 8 frames) give the rows of the clip at once (``test_ref``).
"""

import math

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig
from models.lfm3a import host as A

from ..minicpmv.ref import VisionGolden, golden

__all__ = ["Encoder", "Stream", "VisionGolden", "golden"]

THETA = 1e6  # (the encoder's RoPE base: config encoder.lfm.rope_theta, llama.cpp's hparams.rope_theta)
LN_EPS = 1e-5  # (the projector's LayerNorm: nn.LayerNorm's default)


class Encoder:
    """LFM3-Audio's speech encoder and projector from its mmproj file (module docstring), in float64."""

    def __init__(self, mmproj: str):
        self.g = g = GGUFFile(mmproj)
        self.cfg = cfg = AudioConfig.from_gguf(g)
        assert cfg.projector == "lfm3a", cfg.projector
        self.hidden, self.heads, self.kv_heads, self.hd = cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim
        self.layers, self.eps, self.window = cfg.layers, cfg.eps, cfg.window
        self._w: dict[str, torch.Tensor] = {}

    def w(self, name: str) -> torch.Tensor:
        """A tensor in numpy's shape order (GGUF's dims reversed), float64 (cached)."""
        if name not in self._w:
            t = self.g.tensor(name)
            dims = tuple(reversed(self.g.tensor_info[name][0]))
            self._w[name] = torch.from_numpy(t.dequantize().reshape(dims)).to(torch.float64)
        return self._w[name]

    def vec(self, name: str) -> torch.Tensor:
        return self.w(name).reshape(-1)

    def is_conv(self, il: int) -> bool:
        return f"a.blk.{il}.conv_dw.weight" in self.g

    def _mm(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        y = x @ self.w(name).reshape(-1, x.shape[-1]).T
        return y + self.vec(bias) if bias else y

    def _rms(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.vec(name)

    # -- the downsampler

    def _conv(self, x: torch.Tensor, k: int, depthwise: bool) -> torch.Tensor:
        """Stage conv ``a.conv1d.<k>`` (3 x 3, stride 2, padding 1) + b of x (C, F, T)."""
        w = self.w(f"a.conv1d.{k}.weight")  # (out, in, kh = frequency, kw = time)
        y = torch.nn.functional.conv2d(x[None], w, stride=2, padding=1, groups=x.shape[0] if depthwise else 1)[0]
        return y + self.vec(f"a.conv1d.{k}.bias")[:, None, None]

    def _pw(self, x: torch.Tensor, k: int) -> torch.Tensor:
        w = self.w(f"a.conv1d.{k}.weight").reshape(x.shape[0], x.shape[0])
        return torch.einsum("oc,cft->oft", w, x) + self.vec(f"a.conv1d.{k}.bias")[:, None, None]

    def stages(self, mel: torch.Tensor) -> list[torch.Tensor]:
        """The downsampler's stage outputs (256, F / 2^s, T / 2^s) of mel (128, T): after conv_in, mid, end."""
        silu = torch.nn.functional.silu
        s1 = silu(self._conv(mel[None], 0, False))
        s2 = silu(self._pw(self._conv(s1, 2, True), 3))
        s3 = silu(self._pw(self._conv(s2, 5, True), 6))
        return [s1, s2, s3]

    def downsample(self, mel: np.ndarray | torch.Tensor) -> torch.Tensor:
        """The rows (rows, hidden) of a log-mel spectrogram (128, frames)."""
        s3 = self.stages(torch.as_tensor(np.asarray(mel), dtype=torch.float64))[2]
        c, f, t = s3.shape
        return self._mm(s3.reshape(c * f, t).T, "a.pre_encode.out.weight", "a.pre_encode.out.bias")

    # -- the layers

    def _shortconv(self, x: torch.Tensor, p: str, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """y and the new history (the last 2 rows of B x) of x (n, hidden) after the history hist (2, hidden)."""
        H = self.hidden
        bcx = self._mm(x, p + "conv_pw1.weight")
        b, c, xx = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        bx = torch.cat([hist, b * xx])
        w = self.w(p + "conv_dw.weight")  # (hidden, 3): tap k reads row t + k - 2
        conv = w[:, 0] * bx[:-2] + w[:, 1] * bx[1:-1] + w[:, 2] * bx[2:]
        return self._mm(c * conv, p + "conv_pw2.weight"), bx[-2:]

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x (n, heads, hd) turned at positions pos (n,): NEOX pairs (i, i + hd / 2), angle pos theta^(-2 i / hd)."""
        h = self.hd // 2
        ang = pos.to(torch.float64)[:, None] * THETA ** (-torch.arange(h, dtype=torch.float64) * 2 / self.hd)
        c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
        a, b = x[..., :h], x[..., h:]
        return torch.cat([a * c - b * s, b * c + a * s], -1)

    def _attn(self, x: torch.Tensor, p: str, pos: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor]
              ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """The attention of x (n, hidden) at positions pos after the cached K / V rows kv ((m, kv heads, hd) each, the
        positions before pos[0]); returns its output and the last window - 1 rows of K / V."""
        n, hd = x.shape[0], self.hd
        q = self._rms(self._mm(x, p + "attn_q.weight").reshape(n, self.heads, hd), p + "attn_q_norm.weight")
        k = self._rms(self._mm(x, p + "attn_k.weight").reshape(n, self.kv_heads, hd), p + "attn_k_norm.weight")
        v = self._mm(x, p + "attn_v.weight").reshape(n, self.kv_heads, hd)
        q, k = self.rope(q, pos), self.rope(k, pos)
        K, V = torch.cat([kv[0], k]), torch.cat([kv[1], v])
        m = K.shape[0] - n
        kpos = torch.arange(K.shape[0]) - m  # (the keys' positions relative to pos[0])
        qpos = torch.arange(n)
        mask = (kpos[None] <= qpos[:, None]) & (kpos[None] > qpos[:, None] - self.window)
        rep = self.heads // self.kv_heads
        Kh, Vh = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)  # (query head h: KV head h // rep)
        s = torch.einsum("nhd,mhd->hnm", q, Kh) / math.sqrt(hd)
        s = s.masked_fill(~mask[None], float("-inf"))
        o = torch.einsum("hnm,mhd->nhd", torch.softmax(s, -1), Vh).reshape(n, self.heads * hd)
        keep = self.window - 1
        return self._mm(o, p + "attn_out.weight"), (K[-keep:], V[-keep:])

    def layer(self, x: torch.Tensor, il: int, pos: torch.Tensor, state: dict) -> torch.Tensor:
        """Layer il on the rows x at positions pos, its streaming state (``Stream.state``'s entry) updated."""
        p = f"a.blk.{il}."
        h = self._rms(x, p + "ln1.weight")
        if self.is_conv(il):
            y, state[il] = self._shortconv(h, p, state[il])
        else:
            y, state[il] = self._attn(h, p, pos, state[il])
        x = x + y
        h = self._rms(x, p + "ln2.weight")
        g = self._mm(h, p + "ffn_gate.weight")
        return x + self._mm(torch.nn.functional.silu(g) * self._mm(h, p + "ffn_up.weight"), p + "ffn_down.weight")

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """The final RMSNorm and the projector of the encoder's rows x -> (rows, projection_dim)."""
        x = self._rms(x, "a.post_ln.weight")
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + LN_EPS) * self.vec("mm.a.mlp.0.weight") + self.vec("mm.a.mlp.0.bias")
        h = self._mm(x, "mm.a.mlp.1.weight", "mm.a.mlp.1.bias")
        h = 0.5 * h * (1 + torch.erf(h / math.sqrt(2)))
        return self._mm(h, "mm.a.mlp.3.weight", "mm.a.mlp.3.bias")

    def zero_state(self) -> dict:
        H, KVH, hd = self.hidden, self.kv_heads, self.hd
        z = torch.zeros(0, KVH, hd, dtype=torch.float64)
        return {il: torch.zeros(2, H, dtype=torch.float64) if self.is_conv(il) else (z, z) for il in range(self.layers)}

    def forward(self, mel: np.ndarray | torch.Tensor, trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """A whole clip's log-mel spectrogram (128, frames) -> its embeddings (rows, projection_dim). ``trace``: the
        intermediates ("downsample.out", "layer_out-<il>", "embedding_norm" before the projector's LayerNorm)."""
        tr = trace if trace is not None else {}
        x = self.downsample(mel)
        tr["downsample.out"] = x
        pos = torch.arange(x.shape[0])
        state = self.zero_state()
        for il in range(self.layers):
            x = self.layer(x, il, pos, state)
            tr[f"layer_out-{il}"] = x
        tr["embedding_norm"] = self._rms(x, "a.post_ln.weight")
        return self.project(x)


class Stream:
    """A clip encoded in chunks of mel frames (module docstring): ``push(frames)`` -> the rows those frames complete
    (``final``: the clip's last chunk, its partial row too)."""

    def __init__(self, enc: Encoder):
        self.enc = enc
        self.state = enc.zero_state()
        self.tail = torch.zeros(A.N_MEL, 0, dtype=torch.float64)  # (frames of rows not emitted yet, after 8 of context)
        self.rows = 0  # (rows emitted)

    def push(self, mel: np.ndarray, *, final: bool = False) -> torch.Tensor:
        enc = self.enc
        f = torch.cat([self.tail, torch.as_tensor(np.asarray(mel), dtype=torch.float64)], 1)
        ctx = A.STRIDE if self.rows else 0  # (the frames of the last emitted row, before the new rows' frames)
        n = (f.shape[1] - ctx + (A.STRIDE - 1 if final else 0)) // A.STRIDE
        if not n:
            self.tail = f
            return torch.zeros(0, enc.cfg.projection_dim, dtype=torch.float64)
        x = enc.downsample(f)[ctx // A.STRIDE:ctx // A.STRIDE + n]
        pos = torch.arange(self.rows, self.rows + n)
        for il in range(enc.layers):
            x = enc.layer(x, il, pos, self.state)
        self.rows += n
        self.tail = f[:, ctx + A.STRIDE * (n - 1):]
        return enc.project(x)
