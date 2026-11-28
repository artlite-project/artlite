"""The reference of MOSS-Transcribe-Diarize's audio tower: the encoder and the adaptor in float64 torch, on the
mmproj's tensors (a checkpoint directory's: ``artlite.hf_model``'s virtual mmproj).

The network (transformers' ``WhisperEncoder``, the checkpoint's ``VQAdaptor``) on a chunk's log-mel spectrogram (80,
3000):
- conv1 (``a.conv1d.1``: 80 -> 1024 channels, 3 taps, padding 1) + b, GELU (erf); conv2 (``a.conv1d.2``: 3 taps, stride
  2, padding 1) + b, GELU: 1500 rows; + the position table (``a.position_embd``);
- 24 layers [LayerNorm -> attention | LayerNorm -> 1024 -> 4096 -> GELU -> 1024] with residuals (eps 1e-5; 16 heads of
  64, every row a key of every row; q, v and the output with biases, k without);
- the final LayerNorm (``a.post_ln``); a chunk's first 4 n rows as n rows of 4 x 1024 (4 consecutive rows each), the
  adaptor: ``mm.a.mlp.0`` (4096 -> 1024) + b, SiLU, ``mm.a.mlp.2`` + b, LayerNorm (``mm.a.mlp.3``, eps 1e-6).
"""

import torch

from artlite.hf_model import open_mmproj
from artlite.model_config import AudioConfig

F = torch.nn.functional


class Encoder:
    """The audio tower of an mmproj (projector ``moss``) in float64 (module docstring)."""

    def __init__(self, mmproj: str):
        self.g = g = open_mmproj(mmproj)
        self.cfg = cfg = AudioConfig.from_gguf(g)
        assert cfg.projector == "moss", cfg.projector
        self.merge = int(g.meta["clip.audio.projector.stack_factor"])
        self.eps_out = float(g.meta["clip.audio.projector.layer_norm_epsilon"])
        self._w: dict[str, torch.Tensor] = {}

    def w(self, name: str) -> torch.Tensor:
        """A tensor in numpy's shape order (GGUF's dims reversed), float64 (cached)."""
        if name not in self._w:
            dims = tuple(reversed(self.g.tensor_info[name][0]))
            self._w[name] = torch.from_numpy(self.g.tensor(name).dequantize().reshape(dims)).to(torch.float64)
        return self._w[name]

    def _lin(self, x: torch.Tensor, name: str, bias: bool = True) -> torch.Tensor:
        y = x @ self.w(name + ".weight").T
        return y + self.w(name + ".bias").reshape(-1) if bias else y

    def _ln(self, x: torch.Tensor, name: str, eps: float) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.w(name + ".weight").reshape(-1), self.w(name + ".bias").reshape(-1), eps)

    def stem(self, mel: torch.Tensor) -> torch.Tensor:
        """A chunk's log-mel spectrogram (80, 3000) -> the rows before layer 0 (1500, 1024)."""
        x = F.gelu(F.conv1d(mel[None], self.w("a.conv1d.1.weight"), self.w("a.conv1d.1.bias").reshape(-1), padding=1))
        x = F.gelu(F.conv1d(x, self.w("a.conv1d.2.weight"), self.w("a.conv1d.2.bias").reshape(-1), stride=2, padding=1))
        return x[0].T + self.w("a.position_embd.weight")

    def layer(self, x: torch.Tensor, il: int) -> torch.Tensor:
        cfg, p = self.cfg, f"a.blk.{il}."
        n = self._ln(x, p + "ln1", cfg.eps)
        heads = lambda t: t.reshape(len(x), cfg.heads, cfg.head_dim).transpose(0, 1)
        q, k = heads(self._lin(n, p + "attn_q")), heads(self._lin(n, p + "attn_k", bias=False))
        a = torch.softmax(q @ k.transpose(1, 2) / cfg.head_dim ** 0.5, -1) @ heads(self._lin(n, p + "attn_v"))
        x = x + self._lin(a.transpose(0, 1).reshape(len(x), -1), p + "attn_out")
        return x + self._lin(F.gelu(self._lin(self._ln(x, p + "ln2", cfg.eps), p + "ffn_up")), p + "ffn_down")

    def rows(self, mel: torch.Tensor, layers: int | None = None) -> torch.Tensor:
        """The encoder's rows (1500, 1024) of a chunk's spectrogram after ``layers`` layers (default all, then the final
        LayerNorm)."""
        x = self.stem(mel.to(torch.float64))
        for il in range(self.cfg.layers if layers is None else layers):
            x = self.layer(x, il)
        return self._ln(x, "a.post_ln", self.cfg.eps) if layers is None else x

    def adapt(self, rows: torch.Tensor) -> torch.Tensor:
        """The adaptor's rows (n, hidden) of a chunk's first 4 n encoder rows."""
        m = self.merge
        x = rows[:len(rows) // m * m].reshape(len(rows) // m, -1)
        x = self._lin(F.silu(self._lin(x, "mm.a.mlp.0")), "mm.a.mlp.2")
        return self._ln(x, "mm.a.mlp.3", self.eps_out)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor, tokens: int) -> torch.Tensor:
        """A chunk's spectrogram (80, 3000) -> its ``tokens`` tokens' rows (the language model's width), float64."""
        return self.adapt(self.rows(mel)[:self.merge * tokens])
