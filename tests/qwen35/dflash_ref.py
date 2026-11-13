"""A DFlash2 block drafter's reference (llama.cpp's ``src/models/dflash.cpp``) in float32 numpy."""

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import DFlashConfig

from ..moe.ref import swiglu
from ..np_ops import f32, rms_norm
from .ref import rope_partial


def dflash_conv(x: np.ndarray, dyn: np.ndarray, base: np.ndarray, side: int, group: int) -> np.ndarray:
    """llama.cpp's ``build_dflash2_conv`` along the block: x (T, H) rows of consecutive positions, dyn (T, 2 tap G) the
    conv projection's rows ([side][tap][group]), base (2, taps, H) fp32; out_i = sum over the taps of (base + dyn_i) x_(i -
    tap), per channel, the row before the first zeros."""
    T, H = x.shape
    taps = base.shape[1]
    w = base[side] + np.repeat(f32(dyn).reshape(T, 2, taps, H // group)[:, side], group, -1)
    out = w[:, 0] * x
    for t in range(1, min(taps, T)):
        out[t:] += w[t:, t] * x[:-t]
    return out


class DFlash:
    """A DFlash2 drafter (llama.cpp's ``src/models/dflash.cpp``, float32) of a model file: ``inject`` the model's taps of
    rows into its caches (the rows' K and V per layer), ``block`` at a position over them. ``head``: the LM rows it reads
    (token ids; None: the whole vocabulary). Its tensors load at their first use and stay."""

    def __init__(self, path: str, model: str, capacity: int, *, head: np.ndarray | None = None):
        self.gguf, self.model = GGUFFile(path), GGUFFile(model)
        self.cfg = cfg = DFlashConfig.from_gguf(self.gguf)
        self._w: dict[str, np.ndarray] = {}
        self.head = None if head is None else np.asarray(head, np.int64)
        shape = (cfg.layers, capacity, cfg.kv_heads, cfg.head_dim)
        self.K, self.V = np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def mat(self, name: str) -> np.ndarray:
        if name not in self._w:
            t = self.gguf.tensor(name)
            self._w[name] = f32(t.dequantize()).reshape(t.rows, t.cols)
        return self._w[name]

    def vec(self, name: str) -> np.ndarray:
        if name not in self._w:
            self._w[name] = f32(self.gguf.tensor(name).dequantize()).reshape(-1)
        return self._w[name]

    def base(self, name: str) -> np.ndarray:
        """A conv's base weights (2, taps, H)."""
        if name not in self._w:
            shape = self.gguf.tensor_info[name][0]
            self._w[name] = f32(self.gguf.tensor(name).raw).reshape(shape[::-1])
        return self._w[name]

    def theta(self, pos: np.ndarray) -> np.ndarray:
        return self.cfg.rope_theta(pos)

    def inject(self, taps: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """The rows at positions pos: taps (T, len(cfg.taps), H) -> their K, V in the caches; returns the fused rows
        (T, H) (llama.cpp's ``inp_g_embeddings``)."""
        cfg = self.cfg
        T = taps.shape[0]
        g = rms_norm(f32(taps).reshape(T, -1) @ self.mat("fc.weight").T, self.vec("enc.output_norm.weight"), cfg.rms_norm_eps)
        th = self.theta(pos)
        for L in range(cfg.layers):
            b = f"blk.{L}."
            k = (g @ self.mat(b + "attn_k.weight").T).reshape(T, cfg.kv_heads, cfg.head_dim)
            self.K[L, pos] = rope_partial(rms_norm(k, self.vec(b + "attn_k_norm.weight"), cfg.rms_norm_eps), th)
            self.V[L, pos] = (g @ self.mat(b + "attn_v.weight").T).reshape(T, cfg.kv_heads, cfg.head_dim)
        return g

    def attention(self, L: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, p: int) -> np.ndarray:
        """Layer L's attention of a block at p: rows q (T, heads, d), its own K and V (T, KV heads, d) over the cached
        positions before p: non-causal in the block, a row at r past the positions of r - window and before."""
        cfg = self.cfg
        T = q.shape[0]
        ws = max(0, p - cfg.window + 1)
        kc, vc = np.concatenate([self.K[L, ws:p], k]), np.concatenate([self.V[L, ws:p], v])
        P = np.arange(ws, p + T)
        mask = P[None, :] <= (p + np.arange(T))[:, None] - cfg.window
        kv = np.arange(cfg.heads) // (cfg.heads // cfg.kv_heads)
        s = np.einsum("thd,phd->thp", q, kc[:, kv]) * np.float32(1 / np.sqrt(cfg.head_dim))
        s = np.where(mask[:, None, :], -np.inf, s)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        return np.einsum("thp,phd->thd", a, vc[:, kv]).astype(np.float32)

    def block(self, emb: np.ndarray, p: int) -> dict[str, np.ndarray]:
        """The block of rows emb (T, H) (the anchor's and the mask token's embeddings) at p: its tensors by llama.cpp's
        names, the layers' as name-L, ``result_norm`` (T, H), ``logits`` (T, the head's rows), ``gate`` (T, rank)."""
        cfg = self.cfg
        eps, G, T = cfg.rms_norm_eps, cfg.conv_group, emb.shape[0]
        x, out = f32(emb), {}
        th = self.theta(np.arange(p, p + T))
        for L in range(cfg.layers):
            b = f"blk.{L}."
            xn = out[f"noise_norm-{L}"] = rms_norm(x, self.vec(b + "attn_norm.weight"), eps)
            dyn = xn @ self.mat(b + "attn_conv_proj.weight").T
            xc = out[f"attn_conv_in-{L}"] = dflash_conv(xn, dyn, self.base(b + "attn_conv_base"), 0, G)
            q = (xc @ self.mat(b + "attn_q.weight").T).reshape(T, cfg.heads, cfg.head_dim)
            k = (xc @ self.mat(b + "attn_k.weight").T).reshape(T, cfg.kv_heads, cfg.head_dim)
            v = (xc @ self.mat(b + "attn_v.weight").T).reshape(T, cfg.kv_heads, cfg.head_dim)
            q = out[f"Qcur-{L}"] = rope_partial(rms_norm(q, self.vec(b + "attn_q_norm.weight"), eps), th)
            k = out[f"Kcur-{L}"] = rope_partial(rms_norm(k, self.vec(b + "attn_k_norm.weight"), eps), th)
            out[f"Vcur-{L}"] = v
            o = self.attention(L, q, k, v, p).reshape(T, -1) @ self.mat(b + "attn_output.weight").T
            o = out[f"attn_conv_out-{L}"] = dflash_conv(o, dyn, self.base(b + "attn_conv_base"), 1, G)
            x1 = out[f"ffn_inp-{L}"] = x + o
            yn = out[f"ffn_norm-{L}"] = rms_norm(x1, self.vec(b + "ffn_norm.weight"), eps)
            dyn2 = yn @ self.mat(b + "ffn_conv_proj.weight").T
            yc = out[f"ffn_conv_in-{L}"] = dflash_conv(yn, dyn2, self.base(b + "ffn_conv_base"), 0, G)
            y = out[f"ffn_out-{L}"] = swiglu(yc, self.mat(b + "ffn_gate.weight"), self.mat(b + "ffn_up.weight"),
                                             self.mat(b + "ffn_down.weight"))
            y = out[f"ffn_conv_out-{L}"] = dflash_conv(y, dyn2, self.base(b + "ffn_conv_base"), 1, G)
            x = out[f"l_out-{L}"] = x1 + y
        xh = out["result_norm"] = rms_norm(x, self.vec("output_norm.weight"), eps)
        out["logits"] = xh @ self.lm().T
        out["gate"] = xh @ self.mat("selector_hidden.weight").T
        return out

    def lm(self) -> np.ndarray:
        """The LM rows of ``head`` (fp32)."""
        if "lm" not in self._w:
            t = self.model.tensor("output.weight")
            ids = np.arange(t.rows) if self.head is None else self.head
            self._w["lm"] = f32(self.model.rows("output.weight", ids))
        return self._w["lm"]

    def lattice(self, logits: np.ndarray, gate: np.ndarray, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        """The selector: per block position its top_k candidates (T, top_k) (token ids) and their scores (T, top_k,
        top_k) [predecessor][candidate]: at position 1 the same row for every predecessor (the anchor is the only one);
        position 0 reads nothing."""
        cfg = self.cfg
        T, k = logits.shape[0], cfg.top_k
        idx = np.argpartition(-logits, k - 1, -1)[:, :k]
        cand = idx if self.head is None else self.head[idx]
        un = np.take_along_axis(logits, idx, -1)
        succ = self._rows("selector_successor.weight", cand.reshape(-1)).reshape(T, k, cfg.rank)
        score = np.zeros((T, k, k), np.float32)
        for i in range(1, T):
            ids = cand[i - 1] if i > 1 else np.full(k, anchor)
            pred = self._rows("selector_predecessor.weight", ids)
            score[i] = (pred * gate[i]) @ succ[i].T + un[i][None, :]
        return cand, score

    def _rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        return f32(self.gguf.rows(name, ids))

    @staticmethod
    def walk(cand: np.ndarray, score: np.ndarray) -> list[int]:
        """llama.cpp's greedy walk of the lattice: the block's drafts (positions 1 ..)."""
        out, c = [], 0
        for i in range(1, cand.shape[0]):
            c = int(np.argmax(score[i, c]))
            out.append(int(cand[i, c]))
        return out
