"""DSpark (XHToken's Spark-X2.6 drafters, GGUF ``dflash`` with ``markov_w1``) on the CPU in fp32: the reference of the
NPU drafter (``models.spark.dspark``).

The fork's graph (XHToken/llama.cpp 22c0ae18 ``src/models/dflash.cpp``, ``common/speculative.cpp``):

- injection: a committed position's taps (the residual entering the target's layers ``target_layers``, 5 x hidden)
  -> per tap RMSNorm x ``fc_norm[tap]`` -> ``fc`` (5 hidden -> hidden) -> RMSNorm x ``enc.output_norm`` = g; per layer
  K = rope(wk g), V = wv g at that position (no k norm);
- block at P (the anchor = the last committed token, not yet run by the target): rows [anchor, mask x 6] at P .. P + 6
  through 5 sliding-window layers (the target's sliding-window layer: RoPE on all 128 pairs, theta 1e4; per-head
  sigmoid output gate; GeGLU), every row seeing the keys P - 512 .. P + 6 (non-causal; the window before the anchor:
  the fork's n_swa + 1), ``output_norm`` -> the target's LM head -> logits[7];
- the Markov head, greedy: prev = anchor; row i: col = logits[i] + w2 (w1[prev]); draft_i = argmax(col);
  conf_i = sigmoid(conf . [h_i; w1[prev]] + b) (h_i: the output-normed row); prev = draft_i.

The arithmetic's steps are methods (``mm``, ``kv_rows``, ``attend``, ``gate``, ``markov``) for models of other
rounding (N0's simulations)."""

import numpy as np
import torch

from artlite.gguf_model import GGUFFile

from .cpu import gelu


class DSparkCpu:
    """A DSpark file's drafter in fp32 (weights dequantized at construction); ``emb``, ``head_w``: the target's token
    embeddings and LM head rows."""

    def __init__(self, path: str, *, emb: torch.Tensor, head_w: torch.Tensor, gelu_form: str = "erf"):
        self.gguf = g = GGUFFile(path)
        assert g.arch == "dflash" and "markov_w1.weight" in g
        m = g.arch_meta
        self.n_layers = int(m("block_count"))
        self.hidden = int(m("embedding_length"))
        self.nh, self.nkv, self.hd = int(m("attention.head_count")), int(m("attention.head_count_kv")), int(m("attention.key_length"))
        self.eps = float(m("attention.layer_norm_rms_epsilon"))
        self.block = int(m("block_size"))
        self.window = int(m("attention.sliding_window"))  # keys P - window .. P - 1 before the anchor
        self.taps = [int(x) for x in m("target_layers")]
        rot = int(m("rope.dimension_count_swa", self.hd))
        base = float(m("rope.freq_base_swa", 10000.0))
        self.inv = torch.from_numpy(base ** (-np.arange(0, rot, 2, dtype=np.float64) / rot))
        self.mask_id = int(g.meta["tokenizer.ggml.mask_token_id"])
        self.gelu_form = gelu_form
        self.emb, self.head_w = emb, head_w

        def w(name: str) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(g.tensor(name).dequantize(), np.float32))

        self.fc = w("fc.weight")  # (hidden, taps * hidden)
        self.fc_norm = w("fc_norm.weight").reshape(len(self.taps), self.hidden)
        self.enc_norm = w("enc.output_norm.weight").reshape(-1)
        self.out_norm = w("output_norm.weight").reshape(-1)
        self.w1 = w("markov_w1.weight")  # (vocab, rank)
        self.w2 = w("markov_w2.weight")  # (vocab, rank)
        self.conf_w = w("conf_proj.weight").reshape(-1)  # (hidden + rank)
        self.conf_b = float(w("conf_proj.bias").reshape(-1)[0])
        self.layers = []
        for L in range(self.n_layers):
            b = f"blk.{L}."
            self.layers.append({"an": w(b + "attn_norm.weight").reshape(-1), "q": w(b + "attn_q.weight"),
                                "k": w(b + "attn_k.weight"), "v": w(b + "attn_v.weight"), "g": w(b + "attn_gate.weight"),
                                "o": w(b + "attn_output.weight"), "fn": w(b + "ffn_norm.weight").reshape(-1),
                                "gate": w(b + "ffn_gate.weight"), "up": w(b + "ffn_up.weight"),
                                "down": w(b + "ffn_down.weight")})
        self.reset()

    def reset(self) -> None:
        """The cache: per layer K, V (kv heads, positions, hd), positions 0 .. n - 1 (injected)."""
        self.K = [torch.zeros(self.nkv, 0, self.hd) for _ in range(self.n_layers)]
        self.V = [torch.zeros(self.nkv, 0, self.hd) for _ in range(self.n_layers)]

    # -- operations ------------------------------------------------------------------------------------------------

    def rms(self, x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y if w is None else y * w

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x @ w.T

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        r = self.inv.numel()
        a = pos.double()[:, None] * self.inv[None]
        c, s = torch.cos(a).float(), torch.sin(a).float()
        x1, x2 = x[..., :r], x[..., r:2 * r]
        out = x.clone()
        out[..., :r], out[..., r:2 * r] = x1 * c - x2 * s, x2 * c + x1 * s
        return out

    def kv_rows(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K, V (kv, T, hd) as the cache keeps them."""
        return k, v

    def attend(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q[None], K[None, None], V[None, None])[0]

    def gate(self, g: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(g)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.mm(h, self.head_w)

    # -- the drafter ---------------------------------------------------------------------------------------------

    def encode(self, taps: torch.Tensor) -> torch.Tensor:
        """taps (T, n_taps, hidden) -> g (T, hidden)."""
        x = self.rms(taps, self.fc_norm[None])
        return self.rms(self.mm(x.reshape(len(taps), -1), self.fc), self.enc_norm)

    @torch.no_grad()
    def inject(self, taps: torch.Tensor, p0: int) -> None:
        """The committed positions p0 .. p0 + T - 1 (taps (T, n_taps, hidden)) into every layer's cache."""
        g = self.encode(taps)
        T = len(g)
        pos = torch.arange(p0, p0 + T)
        for L, lw in enumerate(self.layers):
            k = self.rope(self.mm(g, lw["k"]).reshape(T, self.nkv, self.hd).transpose(0, 1), pos)
            v = self.mm(g, lw["v"]).reshape(T, self.nkv, self.hd).transpose(0, 1)
            k, v = self.kv_rows(k, v)
            self.K[L] = torch.cat([self.K[L][:, :p0], k], 1)
            self.V[L] = torch.cat([self.V[L][:, :p0], v], 1)

    @torch.no_grad()
    def block_rows(self, anchor: int, P: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The block at P: (logits (block, vocab) before the Markov bias, h (block, hidden) output-normed)."""
        B, nh, nkv, hd = self.block, self.nh, self.nkv, self.hd
        x = self.emb[torch.tensor([anchor] + [self.mask_id] * (B - 1))]
        pos = torch.arange(P, P + B)
        lo = max(0, P - self.window)
        for L, lw in enumerate(self.layers):
            xn = self.rms(x, lw["an"])
            q = self.rope(self.mm(xn, lw["q"]).reshape(B, nh, hd).transpose(0, 1), pos)
            k = self.rope(self.mm(xn, lw["k"]).reshape(B, nkv, hd).transpose(0, 1), pos)
            v = self.mm(xn, lw["v"]).reshape(B, nkv, hd).transpose(0, 1)
            k, v = self.kv_rows(k, v)
            K = torch.cat([self.K[L][:, lo:P], k], 1)
            V = torch.cat([self.V[L][:, lo:P], v], 1)
            per = nh // nkv
            o = torch.cat([self.attend(q[h * per:(h + 1) * per], K[h], V[h]) for h in range(nkv)])
            o = o * self.gate(self.mm(xn, lw["g"])).T[:, :, None]
            x = x + self.mm(o.transpose(0, 1).reshape(B, nh * hd), lw["o"])
            xn = self.rms(x, lw["fn"])
            x = x + self.mm(gelu(self.mm(xn, lw["gate"]), self.gelu_form) * self.mm(xn, lw["up"]), lw["down"])
        h = self.rms(x, self.out_norm)
        return self.head(h), h

    def markov(self, prev: int) -> torch.Tensor:
        """The Markov bias (vocab) after token prev."""
        return self.w2 @ self.w1[prev]

    @torch.no_grad()
    def draft(self, anchor: int, P: int) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """The greedy chain: drafts (block), the biased logits (block, vocab), the confidences (block)."""
        logits, h = self.block_rows(anchor, P)
        prev, drafts, cols, conf = anchor, [], [], []
        for i in range(self.block):
            w1p = self.w1[prev]
            col = logits[i] + self.markov(prev)
            conf.append(torch.sigmoid(self.conf_w @ torch.cat([h[i], w1p]) + self.conf_b))
            prev = int(col.argmax())
            drafts.append(prev)
            cols.append(col)
        return drafts, torch.stack(cols), torch.stack(conf)
