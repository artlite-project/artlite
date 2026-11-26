"""A torch fp32 model of Inkling 2's DSpark drafter (``artlite.model_config.InklingDSparkConfig``; vLLM's
``models/inkling2/nvidia/dspark.py``, llama.cpp's ``src/models/dspark.cpp``) from its GGUF file's dequantized weights and
the model's (its embedding and ``token_embd_norm``, its LM head).

``Drafter`` holds per layer the K / V of the positions it was given: ``inject`` takes the model's taps of positions
(``taps``: (n, 4, hidden), the residual entering the visits ``cfg.taps`` and the final normed state; ``ref.Model``'s:
``model_taps``), mixes them per layer (``context``: the softmax of ``dspark_fuse``'s row, RMS-normed with the layer's
``dflare_norm``) and stores K (``attn_k_ctx``, per head RMS-normed with ``attn_k_ctx_norm``, rotated) and V
(``attn_v_ctx``). ``block`` runs a block at P: rows [the anchor's normed embedding, ``mask_embd`` x 7] at P .. P + 7
through the layers (keys: the local layers' ``window`` positions before P, the full layer's all of them, and the block
whole), normed by ``output_norm``: the rows' states and their base logits (the model's head). ``drafts`` runs the
Markov head over rows 1 .. 7 (row i guesses position P + i): a gated recurrent state from the anchor on, each step's
logits the base ones plus the head of ``dspark_hidden_out``'s projection of its latent, softcapped; greedy over the
whole vocabulary, or (``top``) over the base logits' top candidates of each row.
"""

import numpy as np
import torch

from artlite.gguf_model import GGUFFile, tensor_f32
from artlite.model_config import InklingConfig, InklingDSparkConfig

from .ref import Model, State, VisitCache, rms

Tensor = torch.Tensor


def rope(x: Tensor, th: Tensor) -> Tensor:
    """NeoX rotation of x (n, heads, d) by angles th (n, d / 2): pairs (i, i + d / 2)."""
    h = x.shape[-1] // 2
    c, s = torch.cos(th)[:, None, :], torch.sin(th)[:, None, :]
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


def model_taps(model: Model, tokens: list[int], state: State, taps: tuple[int, ...]) -> Tensor:
    """The taps (T, len(taps), H) of tokens after the positions ``state`` holds (advanced): the residual entering
    visits ``taps`` (the last one, ``len(visits)``: the final normed state, ``Model.final``)."""
    p0, out = state.position, {}

    def run(v, cur, blocks, messages):
        out[v.index] = cur.clone()
        return model.visit(v, cur, blocks, state.visits.setdefault(v.index, VisitCache()), p0, messages)

    with torch.no_grad():
        out[len(model.cfg.visits)] = model.final(*model.walk(model.embed(tokens), run))
    state.position += len(tokens)
    return torch.stack([out[t] for t in taps], 1)


class Drafter:
    """The drafter of file ``dg`` for the model of file ``tg`` (module docstring)."""

    def __init__(self, dg: GGUFFile, tg: GGUFFile):
        self.dg, self.tg = dg, tg
        self.cfg = cfg = InklingDSparkConfig.from_gguf(dg)
        self.tcfg = InklingConfig.from_gguf(tg)
        self._w: dict[str, Tensor] = {}
        f = torch.from_numpy(tensor_f32(dg, "dspark_fuse.weight")).reshape(cfg.layers + 1, len(cfg.taps))
        self.alpha = torch.softmax(f.double(), -1).float()
        self.mask = self.vec("mask_embd.weight")
        self.head: Tensor | None = None  # (the model's LM head (vocab_size, H), read at the first block)
        self.fold: tuple[Tensor, Tensor] | None = None  # (head @ W_ho, head @ b_ho: ``drafts``)
        self.reset()

    def reset(self) -> None:
        """No positions."""
        self.k: list[dict[int, Tensor]] = [{} for _ in range(self.cfg.layers)]
        self.v: list[dict[int, Tensor]] = [{} for _ in range(self.cfg.layers)]

    def w(self, name: str) -> Tensor:
        """A drafter weight (rows, cols) fp32, kept once read."""
        if name not in self._w:
            t = self.dg.tensor(name)
            w = self.dg.rows(name, np.arange(t.rows)) if t.rows > 1 else t.dequantize()
            self._w[name] = torch.from_numpy(np.ascontiguousarray(w, np.float32)).reshape(t.rows, t.cols)
        return self._w[name]

    def vec(self, name: str) -> Tensor:
        return self.w(name).reshape(-1)

    def angles(self, positions: range | list[int]) -> Tensor:
        return torch.from_numpy(self.cfg.rope_theta(positions))

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def context(self, taps: Tensor, row: int) -> Tensor:
        """Mix ``row`` of the taps (n, taps, H) -> (n, H): RMS-normed sum alpha_j tap_j times the row's norm weight (a
        layer's ``dflare_norm``; the Markov head's, row ``layers``: ``dspark_norm``)."""
        x = torch.einsum("ntd,t->nd", taps, self.alpha[row])
        name = f"blk.{row}.dflare_norm.weight" if row < self.cfg.layers else "dspark_norm.weight"
        return rms(x, self.cfg.rms_norm_eps) * self.vec(name)

    def ctx_kv(self, L: int, taps: Tensor, p0: int) -> tuple[Tensor, Tensor]:
        """Layer L's K and V (n, kv heads, head_dim) of positions p0 .. from their taps (n, taps, H)."""
        c, b = self.cfg, f"blk.{L}."
        x = self.context(taps, L)
        n = x.shape[0]
        k = (x @ self.w(b + "attn_k_ctx.weight").T).reshape(n, c.kv_heads, c.head_dim)
        k = rope(rms(k, c.rms_norm_eps) * self.vec(b + "attn_k_ctx_norm.weight"), self.angles(range(p0, p0 + n)))
        v = (x @ self.w(b + "attn_v_ctx.weight").T).reshape(n, c.kv_heads, c.head_dim)
        return k, v

    def inject(self, taps: Tensor, p0: int) -> None:
        """The positions p0 .. p0 + n - 1 from their taps (n, taps, H): each layer's K and V."""
        with torch.no_grad():
            for L in range(self.cfg.layers):
                k, v = self.ctx_kv(L, taps, p0)
                for i in range(k.shape[0]):
                    self.k[L][p0 + i], self.v[L][p0 + i] = k[i], v[i]

    def f_rnn(self, taps: Tensor) -> Tensor:
        """The Markov head's context (H,) of one position's taps (taps, H)."""
        return self.context(taps[None], self.cfg.layers)[0]

    # ------------------------------------------------------------------
    # The block
    # ------------------------------------------------------------------

    def embed(self, token: int) -> Tensor:
        """The anchor's row: the model's embedding RMS-normed with its ``token_embd_norm``."""
        x = torch.from_numpy(self.tg.rows("token_embd.weight", np.asarray([token]))[0])
        return rms(x, self.tcfg.rms_norm_eps) * torch.from_numpy(tensor_f32(self.tg, "token_embd_norm.weight"))

    def keys(self, L: int, P: int) -> list[int]:
        """The positions a block at P attends to in layer L (before the block's own)."""
        lo = max(P - self.cfg.window, 0) if self.cfg.local[L] else 0
        return [p for p in range(lo, P) if p in self.k[L]]

    def layer(self, L: int, x: Tensor, P: int) -> Tensor:
        """Layer L of the block's rows x (B, H) at P .."""
        c, b, eps = self.cfg, f"blk.{L}.", self.cfg.rms_norm_eps
        B, hd = x.shape[0], c.head_dim
        a = rms(x, eps) * self.vec(b + "attn_norm.weight")
        th = self.angles(range(P, P + B))
        q = rope(rms((a @ self.w(b + "attn_q.weight").T).reshape(B, c.heads, hd), eps) * self.vec(b + "attn_q_norm.weight"), th)
        k = rope(rms((a @ self.w(b + "attn_k.weight").T).reshape(B, c.kv_heads, hd), eps) * self.vec(b + "attn_k_norm.weight"), th)
        v = (a @ self.w(b + "attn_v.weight").T).reshape(B, c.kv_heads, hd)
        ps = self.keys(L, P)
        if ps:
            k = torch.cat([torch.stack([self.k[L][p] for p in ps]), k])
            v = torch.cat([torch.stack([self.v[L][p] for p in ps]), v])
        g = c.heads // c.kv_heads
        s = torch.einsum("bhd,phd->bhp", q, k.repeat_interleave(g, 1)) * (1.0 / np.sqrt(hd))
        o = torch.einsum("bhp,phd->bhd", torch.softmax(s, -1), v.repeat_interleave(g, 1)).reshape(B, -1)
        x = x + o @ self.w(b + "attn_output.weight").T
        f = rms(x, eps) * self.vec(b + "ffn_norm.weight")
        gate, up = f @ self.w(b + "ffn_gate.weight").T, f @ self.w(b + "ffn_up.weight").T
        return x + (torch.nn.functional.silu(gate) * up) @ self.w(b + "ffn_down.weight").T

    def _head(self) -> Tensor:
        if self.head is None:
            n = self.cfg.vocab_size
            self.head = torch.from_numpy(np.ascontiguousarray(self.tg.rows("output.weight", np.arange(n)), np.float32))
        return self.head

    def block(self, anchor: int, P: int) -> tuple[Tensor, Tensor]:
        """A block at P: its rows' states (B, H) (normed by ``output_norm``) and base logits (B, vocab_size) fp32."""
        with torch.no_grad():
            x = torch.stack([self.embed(anchor), *[self.mask] * (self.cfg.block - 1)])
            for L in range(self.cfg.layers):
                x = self.layer(L, x, P)
            h = rms(x, self.cfg.rms_norm_eps) * self.vec("output_norm.weight")
            return h, h @ self._head().T

    # ------------------------------------------------------------------
    # The Markov head
    # ------------------------------------------------------------------

    def _fold(self) -> tuple[Tensor, Tensor]:
        """head @ W_ho (vocab_size, rank) and head @ b_ho (vocab_size,), in fp64 (vLLM's fold)."""
        if self.fold is None:
            W, bias = self.w("dspark_hidden_out.weight").double(), self.vec("dspark_hidden_out.bias").double()
            head = self._head()
            fw, fb = torch.empty(head.shape[0], W.shape[1]), torch.empty(head.shape[0])
            for lo in range(0, head.shape[0], 8192):
                hd = head[lo:lo + 8192].double()
                fw[lo:lo + 8192], fb[lo:lo + 8192] = (hd @ W).float(), (hd @ bias).float()
            self.fold = fw, fb
        return self.fold

    def drafts(self, h: Tensor, base: Tensor, f_rnn: Tensor, anchor: int, n: int | None = None, *, top: int = 0
               ) -> tuple[list[int], Tensor]:
        """The drafts of rows 1 .. n (default all: block - 1) of a block (``block``'s h, base) after the anchor, f_rnn the
        round's last committed position's (``f_rnn``); ``top``: each step's argmax over the base logits' ``top``
        largest only (0: the whole vocabulary). Returns them and their rows' logits (n, vocab_size) fp32 (softcapped)."""
        c = self.cfg
        n = c.block - 1 if n is None else n
        R = c.rank
        with torch.no_grad():
            Wj, bj = self.w("joint_proj.weight"), self.vec("joint_proj.bias")
            ctx = bj + h[1:n + 1] @ Wj[:, 2 * R:2 * R + c.hidden].T + f_rnn @ Wj[:, 2 * R + c.hidden:].T  # (n, 3R)
            fw, fb = self._fold()
            state, prev, out, rows = torch.zeros(R), anchor, [], []
            for i in range(n):
                w1 = torch.from_numpy(self.dg.rows("markov_w1.weight", np.asarray([prev]))[0])
                raw = ctx[i] + Wj[:, :R] @ state + Wj[:, R:2 * R] @ w1
                gate, cand = torch.sigmoid(raw[:R]), torch.tanh(raw[R:2 * R])
                state = gate * state + (1 - gate) * cand
                col = base[i + 1] + fw @ torch.tanh(raw[2 * R:]) + fb
                if c.softcap:
                    col = torch.tanh(col / c.softcap) * c.softcap
                if top:
                    cand_ids = torch.topk(base[i + 1], top).indices
                    prev = int(cand_ids[torch.argmax(col[cand_ids])])
                else:
                    prev = int(torch.argmax(col))
                out.append(prev)
                rows.append(col)
            return out, torch.stack(rows)
