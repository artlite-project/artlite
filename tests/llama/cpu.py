"""Llama on the CPU in fp32: the reference of the NPU path (tests/llama/test_npu.py), on a GGUF file's dequantized
tensors (GGUF names; Q / K rows in rotate-half order, as ``LlamaImage`` lays them out: the same model). ``logits(seq,
rows)`` runs the tokens from position 0 in chunks through a KV cache and returns the logits of the positions ``rows``."""

import torch

from artlite.gguf_model import GGUFFile, unpermute_rows
from artlite.model_config import LlamaConfig


class LlamaCpu:
    """A Llama GGUF file's model in fp32 (its weights dequantized at construction)."""

    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf
        self.cfg = cfg = LlamaConfig.from_gguf(gguf)
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        def w(name: str) -> torch.Tensor:
            return gguf.tensor(name).dequantize().float()

        self.emb = w("token_embd.weight")
        self.out = w("output.weight") if "output.weight" in gguf else self.emb
        self.norm = w("output_norm.weight").reshape(-1)
        qi, ki = torch.from_numpy(unpermute_rows(nh, nh * hd)), torch.from_numpy(unpermute_rows(nkv, nkv * hd))
        self.layers = []
        for L in range(cfg.num_hidden_layers):
            b = f"blk.{L}."
            q, k, v = w(b + "attn_q.weight")[qi], w(b + "attn_k.weight")[ki], w(b + "attn_v.weight")
            self.layers.append({"an": w(b + "attn_norm.weight").reshape(-1), "qkv": torch.cat([q, k, v]),
                                "o": w(b + "attn_output.weight"), "fn": w(b + "ffn_norm.weight").reshape(-1),
                                "gu": torch.cat([w(b + "ffn_gate.weight"), w(b + "ffn_up.weight")]),
                                "down": w(b + "ffn_down.weight")})
        self.inv = torch.from_numpy(cfg.rope_inv_freq())

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * w

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x (heads, T, hd): the pairs (i, i + hd / 2) rotated."""
        a = pos.double()[:, None] * self.inv[None]
        c, s = torch.cos(a).float(), torch.sin(a).float()
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    @torch.no_grad()
    def logits(self, seq: list[int], rows: slice, *, chunk: int = 1024) -> torch.Tensor:
        """The fp32 logits (n, vocab) of the positions ``rows`` of the tokens seq (from position 0)."""
        cfg = self.cfg
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        P = len(seq)
        want = torch.zeros(P, dtype=torch.bool)
        want[rows] = True
        kc = [torch.empty(nkv, P, hd) for _ in self.layers]
        vc = [torch.empty(nkv, P, hd) for _ in self.layers]
        out = []
        for p0 in range(0, P, chunk):
            T = min(chunk, P - p0)
            pos = torch.arange(p0, p0 + T)
            x = self.emb[torch.tensor(seq[p0:p0 + T])]
            mask = torch.arange(p0 + T)[None, :] <= pos[:, None]
            for L, lw in enumerate(self.layers):
                y = self._rms(x, lw["an"]) @ lw["qkv"].T
                q = y[:, :nh * hd].view(T, nh, hd).transpose(0, 1)
                k = y[:, nh * hd:(nh + nkv) * hd].view(T, nkv, hd).transpose(0, 1)
                kc[L][:, p0:p0 + T] = self._rope(k, pos)
                vc[L][:, p0:p0 + T] = y[:, (nh + nkv) * hd:].view(T, nkv, hd).transpose(0, 1)
                o = torch.nn.functional.scaled_dot_product_attention(
                    self._rope(q, pos)[None], kc[L][None, :, :p0 + T], vc[L][None, :, :p0 + T],
                    attn_mask=mask[None, None], enable_gqa=True)[0]
                x = x + o.transpose(0, 1).reshape(T, nh * hd) @ lw["o"].T
                g, u = (self._rms(x, lw["fn"]) @ lw["gu"].T).chunk(2, -1)
                x = x + (torch.nn.functional.silu(g) * u) @ lw["down"].T
            sel = want[p0:p0 + T]
            if sel.any():
                out.append(self._rms(x[sel], self.norm) @ self.out.T)
        return torch.cat(out)
