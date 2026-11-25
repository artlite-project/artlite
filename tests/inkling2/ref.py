"""A torch fp32 model of Inkling 2 from a GGUF file's dequantized weights (llama.cpp's ``src/models/inkling2.cpp`` and
``inkling2-loop.cpp``; its goldens: ``goldens.text``), tokens as the leading axis.

``Model`` runs a sequence of tokens after the positions its ``State`` holds: ``forward`` the visits in order
(``InklingConfig.visits``: the prelude, the loop core's passes with their banks and messages, the coda) and the head.
Its pieces: ``mix`` (a depth mix of the block states and the residual), ``control`` (a loop layer's controller),
``conv`` (a causal conv of ``conv_taps`` taps over a history, the full layers' per-token deltas), ``clip``,
``attention`` (the relative-position bias, the sink, GQA, the window), ``moe`` (sigmoid + bias routing, situ experts),
``visit`` (one visit: llama.cpp's names of its tensors, keyed by its cache slot as llama.cpp's goldens are), ``walk``
(the visits' order, the passes' banks and messages: a slice test drives it with llama.cpp's own visit outputs),
``final`` and ``head``.

``q8``: each matmul's activations rounded as llama.cpp's CPU rounds them for the weight's type (``act``: Q8_0 blocks for
Q8_0 and NVFP4 weights, bf16 for BF16, none for F32), so that a block fed llama.cpp's input agrees with its output to
the summation order; without it the matmuls take the fp32 activations (the model's own math). The rest is fp32 as
llama.cpp computes it (the depth mixes, convs, router and controllers included). The experts' weights are dequantized
per visit and dropped; the other weights are kept once read (~2.7 GB for UD-Q4_K_XL, the head 1.9 GB of it).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, Visit

Tensor = torch.Tensor


def q8_0(x: Tensor) -> Tensor:
    """llama.cpp's Q8_0 rounding of x (..., K) on the CPU (quantize_row_q8_0): per block of 32 d = max|x| / 127 (fp16),
    codes rint(x 127 / max|x|); returns d q."""
    xb = x.reshape(*x.shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    q = torch.round(xb * torch.where(amax != 0, 127.0 / amax, torch.zeros_like(amax)))
    return (q * (amax / 127.0).half().float()).reshape(x.shape)


def rms(x: Tensor, eps: float) -> Tensor:
    """ggml_rms_norm: x / sqrt(mean(x^2) + eps) over the last axis (the squares summed in double)."""
    ms = ((x * x).double().sum(-1, keepdim=True) / x.shape[-1]).float()
    return x * (1.0 / torch.sqrt(ms + eps))


@dataclass
class VisitCache:
    """A visit's state: K / V as attended (``k``: clipped, ``v``: normed; (P, kv heads, head_dim)) and the last
    ``conv_taps`` - 1 rows of each conv's raw input (``hist``: "k", "v", "attn", "ffn")."""

    k: Tensor | None = None
    v: Tensor | None = None
    hist: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class State:
    """A sequence's caches: per visit index a ``VisitCache``, and the positions they hold."""

    position: int = 0
    visits: dict[int, VisitCache] = field(default_factory=dict)


class Model:
    """Inkling 2 on the CPU (module docstring). ``record``: names of llama.cpp's tensors ``forward`` keeps per visit
    (its graph's: "Qcur", "Kcur", "Vcur", "Rcur", "attn_out", "ffn_inp", "ffn_moe_logits", "ffn_moe_topk",
    "ffn_moe_weights_scaled", "ffn_out", "l_out"; ours: "attn_in" (the attention's mixed inputs, (T, 3, H)), "ffn_in"
    (the FFN's mixed input before its norm), "messages" (T, 4, H)) and "result_norm"; ``out``: (name, slot) -> tensor."""

    def __init__(self, gguf: GGUFFile, cfg: InklingConfig | None = None, *, q8: bool = True):
        self.g = gguf
        self.cfg = cfg or InklingConfig.from_gguf(gguf)
        self.q8 = q8
        self._w: dict[str, Tensor] = {}
        self._mq: dict[int, Tensor] = {}
        self.record: set[str] = set()
        self.out: dict[tuple[str, int], Tensor] = {}

    # ------------------------------------------------------------------
    # Weights and matmuls
    # ------------------------------------------------------------------

    def w(self, name: str) -> Tensor:
        """A weight (rows, cols) fp32, kept once read."""
        if name not in self._w:
            t = self.g.tensor(name)
            w = self.g.rows(name, np.arange(t.rows)) if t.rows > 1 else t.dequantize()
            self._w[name] = torch.from_numpy(np.ascontiguousarray(w, np.float32)).reshape(t.rows, t.cols)
        return self._w[name]

    def vec(self, name: str) -> Tensor:
        return self.w(name).reshape(-1)

    def act(self, x: Tensor, qtype: str) -> Tensor:
        """x as llama.cpp's CPU matmul of a ``qtype`` weight reads it (``q8``; module docstring)."""
        if not self.q8 or qtype == "F32":
            return x
        if qtype in ("Q8_0", "NVFP4", "Q4_0", "IQ4_NL"):
            return q8_0(x)
        if qtype == "BF16":
            return x.bfloat16().float()
        raise NotImplementedError(f"llama.cpp's activations of a {qtype} weight")

    def mm(self, x: Tensor, name: str) -> Tensor:
        """x (..., K) @ W.T of a 2-D weight."""
        return self.act(x, self.g.tensor_info[name][1]) @ self.w(name).T

    def msg_queries(self, layer: int) -> Tensor:
        if layer not in self._mq:
            self._mq[layer] = torch.from_numpy(self.cfg.msg_queries(self.g, layer))
        return self._mq[layer]

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    def mix(self, blocks: list[Tensor], cur: Tensor, queries: Tensor) -> Tensor:
        """The depth mix (llama.cpp's ``mix``): the states [blocks..., cur] (each (T, H)) RMS-normed without weight,
        per query q (nq, H) softmax over the states of q . x / sqrt(H); (T, nq, H) the weighted sums of the normed
        states."""
        s = rms(torch.stack([*reversed(blocks), cur], 1), InklingConfig.MIX_EPS)  # (T, n, H): llama.cpp's order
        scores = torch.einsum("tnh,qh->tqn", s, queries) * (1.0 / np.sqrt(self.cfg.hidden_size))
        return torch.einsum("tqn,tnh->tqh", torch.softmax(scores, -1), s)

    def control(self, layer: int, stream: str, cur: Tensor, msg: Tensor) -> Tensor:
        """A loop layer's controller of a stream (q, k, v, ffn): g = D [rms(cur); rms(msg)], (1 - tanh(F g)) cur +
        tanh(G g) msg (llama.cpp's ``control``)."""
        eps, c = InklingConfig.MIX_EPS, self.cfg
        g = self.mm(torch.cat([rms(cur, eps), rms(msg, eps)], -1), c.ctrl_tensor(layer, stream, "down"))
        gain = torch.tanh(self.mm(g, c.ctrl_tensor(layer, stream, "gain")))
        forget = torch.tanh(self.mm(g, c.ctrl_tensor(layer, stream, "forget")))
        return (cur - forget * cur) + gain * msg

    def conv(self, x: Tensor, w: Tensor, hist: Tensor | None, dyn: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """The causal short conv with its residual (llama.cpp's ``convolve``): x (T, C), w (C, taps) (tap taps - 1:
        the current row), ``hist`` the last taps - 1 inputs before x (None: zeros); ``dyn`` (T, heads x taps): the full
        layers' per-token taps of each head's channels (lag 0: the current row). Returns (x + conv) / sqrt(taps + 1)
        (with ``dyn``: sqrt(2 taps + 1)) and the new history."""
        T, C = x.shape
        n = w.shape[1]
        h = torch.zeros(n - 1, C) if hist is None else hist
        xp = torch.cat([h, x], 0)  # (n - 1 + T, C)
        out = sum(xp[k:k + T] * w[:, k] for k in range(n))
        if dyn is not None:
            heads = dyn.shape[1] // n
            taps = dyn.reshape(T, heads, n)
            for lag in range(n):  # (lag 0: the current row)
                window = xp[n - 1 - lag:n - 1 - lag + T].reshape(T, heads, -1)
                out = out + (window * taps[:, :, lag:lag + 1]).reshape(T, C)
        scale = 1.0 / np.sqrt((2 if dyn is not None else 1) * n + 1)
        return (x + out) * scale, xp[-(n - 1):].clone()

    @staticmethod
    def clip(x: Tensor, scale: Tensor | None, threshold: float) -> Tensor:
        """Each row (last axis) times ``scale``, then scaled to an L2 norm of at most ``threshold`` (llama.cpp's
        ``clip``: the norm + 1e-6)."""
        if scale is not None:
            x = x * scale
        norm = torch.sqrt((x * x).sum(-1, keepdim=True)) + 1e-6
        return x * torch.clamp(threshold / norm, 0.0, 1.0)

    def attention(self, layer: int, q: Tensor, k: Tensor, v: Tensor, r: Tensor, sink: Tensor, p0: int, *,
                  rows: int = 512) -> Tensor:
        """Scores q k / sqrt(d) + R . E[:, distance] (distances below the layer's extent: the window on a "swa"
        layer, ``rel_extent`` on a full one; 0 beyond), a sink logit per query and head (``sink`` (T, Hq): an extra
        term with no value), causal (a "swa" layer: the last ``sliding_window`` positions). q (T, Hq, d) at positions
        p0 .., k, v (P, Hkv, d) at 0 .. P - 1, r (T, Hq, rel); the queries ``rows`` at a time. Returns (T, Hq d)."""
        if q.shape[0] > rows:
            return torch.cat([self.attention(layer, q[i:i + rows], k, v, r[i:i + rows], sink[i:i + rows], p0 + i,
                                             rows=rows) for i in range(0, q.shape[0], rows)])
        c = self.cfg
        T, Hq, d = q.shape
        P = k.shape[0]
        swa = c.kinds[layer] == "swa"
        extent = c.sliding_window if swa else c.rel_extent
        table = torch.einsum("thr,er->the", r, self.w(f"blk.{layer}.attn_rel_bias.weight"))  # (T, Hq, extent)
        dist = (p0 + torch.arange(T))[:, None] - torch.arange(P)[None, :]  # (T, P)
        idx = torch.clamp(dist, 0, extent - 1)
        bias = torch.gather(table, 2, idx[:, None, :].expand(T, Hq, P))
        bias = torch.where(((dist >= 0) & (dist < extent))[:, None, :], bias, torch.zeros(()))
        bias = bias - sink[:, :, None]
        masked = (dist < 0) | ((dist >= c.sliding_window) if swa else torch.zeros_like(dist, dtype=torch.bool))
        g = Hq // k.shape[1]
        s = torch.einsum("thd,phd->thp", q, k.repeat_interleave(g, 1)) * (1.0 / np.sqrt(d))
        s = torch.where(masked[:, None, :], torch.tensor(-torch.inf), s + bias)
        s = torch.cat([s, torch.zeros(T, Hq, 1)], -1)  # (the sink: 0 after the bias's - t)
        p = torch.softmax(s, -1)[..., :P]
        return torch.einsum("thp,phd->thd", p, v.repeat_interleave(g, 1)).reshape(T, Hq * d)

    def moe(self, layer: int, x: Tensor, slot: int) -> Tensor:
        """The routed experts of a visit on its normed input x (T, H): logits of the fp32 router, the top
        ``n_experts_used`` of sigmoid + bias (ties: the lower id), weights the sigmoids over their sum (+ 1e-20) times
        ``expert_weights_scale``; expert e: [g | u] = W_gu x (times the expert's scale), situ, down (times its scale);
        the weighted sum."""
        c, b = self.cfg, f"blk.{layer}."
        logits = self.mm(x, b + "ffn_gate_inp.weight")
        probs = torch.sigmoid(logits)
        keys = probs + self.vec(b + "exp_probs_b.bias") if b + "exp_probs_b.bias" in self.g else probs
        ids = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :c.n_experts_used]
        w = torch.gather(probs, 1, ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * c.expert_weights_scale
        self._keep("ffn_moe_logits", slot, logits)
        self._keep("ffn_moe_topk", slot, ids)
        self._keep("ffn_moe_weights_scaled", slot, w)
        gu_name, dn_name = b + "ffn_gate_up_exps.weight", b + "ffn_down_exps.weight"
        F, H = c.expert_ffn, c.hidden_size
        s_gu = self.vec(b + "ffn_gate_up_exps.scale") if b + "ffn_gate_up_exps.scale" in self.g else None
        s_dn = self.vec(b + "ffn_down_exps.scale") if b + "ffn_down_exps.scale" in self.g else None
        out = torch.zeros(x.shape[0], c.n_experts_used, H)
        xa = self.act(x, self.g.tensor_info[gu_name][1])
        for e in torch.unique(ids).tolist():
            t, j = torch.nonzero(ids == e, as_tuple=True)
            gu = xa[t] @ self._rows(gu_name, e * 2 * F, (e + 1) * 2 * F).T
            if s_gu is not None:
                gu = gu * s_gu[e]
            gate, up = gu[:, :F], gu[:, F:]
            a = c.situ_beta * torch.tanh(gate / c.situ_beta) * torch.sigmoid(gate)
            a = a * (c.situ_linear_beta * torch.tanh(up / c.situ_linear_beta))
            y = self.act(a, self.g.tensor_info[dn_name][1]) @ self._rows(dn_name, e * H, (e + 1) * H).T
            if s_dn is not None:
                y = y * s_dn[e]
            out[t, j] = y
        y = out[:, 0] * w[:, :1]
        for j in range(1, c.n_experts_used):
            y = y + out[:, j] * w[:, j:j + 1]
        return y

    def _rows(self, name: str, lo: int, hi: int) -> Tensor:
        """Rows lo .. hi of a weight, dequantized now (an expert's)."""
        return torch.from_numpy(self.g.rows(name, np.arange(lo, hi)))

    def _keep(self, name: str, slot: int, x: Tensor) -> None:
        if name in self.record:
            self.out[name, slot] = x

    # ------------------------------------------------------------------
    # A visit, the loop, the head
    # ------------------------------------------------------------------

    def embed(self, tokens: Iterable[int]) -> Tensor:
        """The tokens' embedding rows RMS-normed with ``token_embd_norm`` (T, H)."""
        x = torch.from_numpy(self.g.rows("token_embd.weight", np.asarray(list(tokens))))
        return rms(x, self.cfg.rms_norm_eps) * self.vec("token_embd_norm.weight")

    def visit(self, v: Visit, cur: Tensor, blocks: list[Tensor], cache: VisitCache, p0: int,
              messages: Tensor | None = None, *, ffn: bool = True) -> Tensor:
        """One visit of the tokens' rows ``cur`` (T, H) at positions p0 .. with the completed block states ``blocks``
        (appended to where the layer begins a block) and the visit's cache; ``messages`` (T, 4, H): the loop's passes
        after the first. Returns the visit's output (the next residual; ``ffn`` False: its attention part's)."""
        c, L, s = self.cfg, v.layer, v.slot
        b, T, eps = f"blk.{L}.", cur.shape[0], c.rms_norm_eps
        hd, kvh, nh = c.head_dim, c.num_key_value_heads, c.num_attention_heads
        swa = v.kind == "swa"
        inputs = [cur] * 3
        if v.queries:
            mixed = self.mix(blocks, cur, self.w(b + "attn_res_query.weight"))
            inputs = [mixed[:, 0 if swa else j] for j in range(3)]
        if messages is not None:
            inputs = [self.control(L, st, inputs[j], messages[:, j]) for j, st in enumerate(c.CTRL_STREAMS[:3])]
        self._keep("attn_in", s, torch.stack(inputs, 1))
        if swa and messages is not None:
            xq, xk, xv = (rms(x, eps) * self.vec(b + "attn_norm.weight") for x in inputs)
        elif swa:
            xq = xk = xv = rms(inputs[0], eps) * self.vec(b + "attn_norm.weight")
        else:
            xq, xk, xv = (rms(x, eps) * self.vec(b + f"attn_{n}_inp_norm.weight") for x, n in zip(inputs, "qkv"))
        q = self.mm(xq, b + "attn_q.weight")
        kraw, vraw = self.mm(xk, b + "attn_k.weight"), self.mm(xv, b + "attn_v.weight")
        kdyn = None if swa else self.mm(xk, b + "attn_k_conv_proj.weight")
        vdyn = None if swa else self.mm(xv, b + "attn_v_conv_proj.weight")
        k, cache.hist["k"] = self.conv(kraw, self.w(b + "attn_k_conv.weight"), cache.hist.get("k"), kdyn)
        vv, cache.hist["v"] = self.conv(vraw, self.w(b + "attn_v_conv.weight"), cache.hist.get("v"), vdyn)
        r = self.mm(xq, b + "attn_rel.weight")
        q = self.clip(q.reshape(T, nh, hd), self.w(b + "attn_q_clip.weight"), c.qk_clip)
        k = self.clip(k.reshape(T, kvh, hd), None, c.qk_clip)
        vv = rms(vv.reshape(T, kvh, hd), eps)
        r = self.clip(r.reshape(T, nh, c.rel_length), self.w(b + "attn_rel_clip.weight"), c.rel_clip)
        if not swa:
            tau = torch.from_numpy(c.tau(range(p0, p0 + T)))[:, None, None]
            q, r = q * tau, r * tau
        for name, x in (("Qcur", q), ("Kcur", k), ("Vcur", vv), ("Rcur", r)):
            self._keep(name, s, x)
        cache.k = k if cache.k is None else torch.cat([cache.k, k], 0)
        cache.v = vv if cache.v is None else torch.cat([cache.v, vv], 0)
        o = self.attention(L, q, cache.k, cache.v, r, self.mm(xq, b + "attn_sink_proj.weight"), p0)
        o = o * torch.sigmoid(self.mm(xq, b + "attn_gate.weight"))
        o, cache.hist["attn"] = self.conv(self.mm(o, b + "attn_output.weight"), self.w(b + "attn_post_conv.weight"),
                                          cache.hist.get("attn"))
        self._keep("attn_out", s, o)
        if v.block:
            blocks.append(cur)
            cur = o
        else:
            cur = cur + o
        self._keep("ffn_inp", s, cur)
        if not ffn:
            return cur
        x = self.mix(blocks, cur, self.w(b + "ffn_res_query.weight"))[:, 0]
        if messages is not None:
            x = self.control(L, "ffn", x, messages[:, 3])
        self._keep("ffn_in", s, x)
        f = self.moe(L, rms(x, eps) * self.vec(b + "ffn_norm.weight"), s)
        f, cache.hist["ffn"] = self.conv(f, self.w(b + "ffn_post_conv.weight"), cache.hist.get("ffn"))
        self._keep("ffn_out", s, f)
        cur = cur + f
        self._keep("l_out", s, cur)
        return cur

    def walk(self, x: Tensor, run: Callable[[Visit, Tensor, list[Tensor], Tensor | None], Tensor]
             ) -> tuple[Tensor, list[Tensor]]:
        """The visits in order from the normed embeddings x (T, H): ``run(visit, cur, blocks, messages)`` gives each
        visit's output from its input (appending ``cur`` to the block states where its layer begins a block); the
        passes restart from the prelude's output and block states, and the passes after the first take their messages
        (T, 4, H) from the previous pass's bank: its block states after the core's first, its output (llama.cpp's
        ``inkling2_loop`` graph). Returns the last visit's output and the block states."""
        c = self.cfg
        cur, blocks, bank = x, [], []
        prelude: tuple[Tensor, list[Tensor]] | None = None
        for v in c.visits:
            if v.loop >= 0 and v.layer == c.core_start:  # (a pass begins: from the prelude's output)
                if v.loop == 0:
                    prelude = cur, list(blocks)
                else:
                    bank = [*blocks[len(prelude[1]) + 1:], cur]
                    cur, blocks = prelude[0], list(prelude[1])
            messages = None
            if v.messages:
                messages = self.mix(bank[:-1], bank[-1], self.msg_queries(v.layer))
                self._keep("messages", v.slot, messages)
            cur = run(v, cur, blocks, messages)
        return cur, blocks

    def final(self, cur: Tensor, blocks: list[Tensor]) -> Tensor:
        """The final states (llama.cpp's ``result_norm``): the output's depth mix of the last visit's output and the
        block states, ``output_norm``."""
        x = self.mix(blocks, cur, self.w("output_res_query.weight"))[:, 0]
        x = rms(x, self.cfg.rms_norm_eps) * self.vec("output_norm.weight")
        self._keep("result_norm", -1, x)
        return x

    def hidden(self, tokens: list[int], state: State | None = None) -> Tensor:
        """The tokens' final states (T, H) after the positions ``state`` holds (advanced)."""
        state = state if state is not None else State()
        p0 = state.position

        def run(v: Visit, cur: Tensor, blocks: list[Tensor], messages: Tensor | None) -> Tensor:
            return self.visit(v, cur, blocks, state.visits.setdefault(v.index, VisitCache()), p0, messages)

        out = self.final(*self.walk(self.embed(tokens), run))
        state.position += len(tokens)
        return out

    def head(self, x: Tensor, *, step: int = 1 << 15) -> Tensor:
        """The LM head's logits (T, vocab_size) of final states x (T, H): softcapped, the padding rows dropped."""
        out = torch.empty(x.shape[0], self.cfg.vocab_size)
        xa = self.act(x, self.g.tensor_info["output.weight"][1])
        for lo in range(0, self.cfg.vocab_size, step):
            hi = min(lo + step, self.cfg.vocab_size)
            out[:, lo:hi] = xa @ self.w("output.weight")[lo:hi].T
        cap = self.cfg.softcap
        return torch.tanh(out * (1.0 / cap)) * cap if cap else out

    def forward(self, tokens: list[int], state: State | None = None) -> Tensor:
        """The logits (T, vocab_size) of the tokens after the positions ``state`` holds."""
        with torch.no_grad():
            return self.head(self.hidden(tokens, state))
