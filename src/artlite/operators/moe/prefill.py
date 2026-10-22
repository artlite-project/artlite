"""An MoE model's MoE blocks in its chunk prefill (``prefill.chunks.ChunkPrefill``; qwen35's ``QwenPrefill``): per
layer, after the post-attention norm with lo panels (PXM: per token [x hi | x lo] panels, ``PrefillNormOperator(lo=True)``)

1. per sub-chunk the router (LG: fp32 logits, ``gemm.router``) — a piece of the program ends here;
2. per sub-chunk the shared expert's gate / up (HS) and down (DS, rows of ``gather.ds_row``: COLS ``pitch``, then the
   token's weights; ``gemm``) — while this piece runs, the host routes the chunk (``host_step``: the logits fetched,
   the top k, ``plan.Plan``, the columns' control streams, the weights into DS, the chunk counts as scratchpad
   parameters) before the next piece starts;
3. over the chunk the gathered norm (XP, ``gather.GatherNormOperator``), the experts' gate / up (H) and down (Y,
   ``expert_gemm``), the combine (DL: the delta per token, ``gather.CombineOperator``);

then the model's residual norm adds DL. The buffers are the chunk's, shared by the layers (one layer's block in
flight). ``cap``: chunks per column at most (the worst case of a chunk of ``max_t`` tokens: ceil((256 + max_t k /
CHUNK) / COLS))."""

from types import SimpleNamespace

import numpy as np

from ... import txn
from ..design import COLS
from ..prefill.norm import PrefillNormOperator
from . import gemm as MG
from . import plan as MP
from .expert_gemm import CHUNK, N_PARAM, R_PARAM, ExpertGemmOperator
from .gather import CHAIN, CombineOperator, GatherNormOperator, ds_row


class MoEPrefill:
    """The MoE blocks of ``layers`` of an image (``qwen35.image.QwenImage``'s MoE regions) for chunks of ``sizes``
    tokens (sub-chunks of at most ``sub``)."""

    def __init__(self, image: object, *, layers: list[int], sizes: tuple[int, ...], sub: int, eps: float):
        cfg = image.cfg
        self.img, self.layers, self.sub, self.eps = image, layers, sub, eps
        self.H, self.E, self.K = cfg.hidden_size, cfg.n_experts, cfg.n_experts_used
        self.F, self.Fs = cfg.expert_ffn, cfg.shared_ffn
        max_t = max(sizes)
        self.cap = -(-(self.E + -(-max_t * self.K // CHUNK)) // COLS)
        self.pitch = MG.shared_pitch(image, layers)
        self._gemms: dict = {}
        self._ops: dict = {}
        self.gnorm = GatherNormOperator(E=self.H, eps=eps, cap=self.cap)
        self.gu = {L: self._op(ExpertGemmOperator.of("gu", image.region(L, "exp_gu"), cap=self.cap, out=self.F))
                   for L in layers}
        self.dn = {L: self._op(ExpertGemmOperator.of("down", image.region(L, "exp_down"), cap=self.cap, out=self.H))
                   for L in layers}
        self.comb = {(T, op.y_layout): CombineOperator(E=self.H, T=T, ds_pitch=self.pitch, y_layout=op.y_layout)
                     for T in sizes for op in self.dn.values()}
        self.y_pitch = max(op.N for op in self.dn.values())
        self.ln = {t: PrefillNormOperator(T=t, E=self.H, eps=eps, lo=True) for t in {min(T, sub) for T in sizes}}

    def _op(self, op: ExpertGemmOperator) -> ExpertGemmOperator:
        """One object per distinct design (the layers share them)."""
        key = (op.kind, op.fmt, op.R, op.passes, op.G, op.cap, op.N)
        return self._ops.setdefault(key, op)

    # ------------------------------------------------------------------
    # Operators and buffers
    # ------------------------------------------------------------------

    def gemms(self, L: int, t: int) -> SimpleNamespace:
        """Layer L's GEMMs of sub-chunks of t tokens: the router, the shared gate / up and down."""
        mi = self.img.region(L, "moe_in")
        return SimpleNamespace(router=MG.router(self.img, mi, t, ops=self._gemms),
                               sgu=MG.shared_gate_up(self.img, mi, t, width=self.Fs, ops=self._gemms),
                               sdown=MG.row_group_gemm(self.img.region(L, "sdown"), t, pitch=self.pitch,
                                                       row=ds_row(self.H, self.pitch), ops=self._gemms))

    def rows(self) -> dict[str, int]:
        """Bytes per token of the per-token buffers."""
        g = self.gemms(self.layers[0], self.sub)
        return {"PXM": 2 * self.H * 9 // 8, "LG": 2 * g.router.op.N, "HS": self.Fs * 9 // 8, "DS": 2 * ds_row(self.H, self.pitch),
                "DL": 2 * self.H}

    def fixed(self) -> dict[str, int]:
        """Bytes of the chunk-independent buffers (the control streams, XP, H, Y)."""
        gu, dn = next(iter(self.gu.values())), max(self.dn.values(), key=lambda op: op.cb_words)
        return {"CBG": 4 * COLS * self.gnorm.cb_words, "CBU": 4 * COLS * gu.cb_words, "CBD": 4 * COLS * dn.cb_words,
                "CBC": 4 * COLS * max(op.cb_words for op in self.comb.values()),
                "XP": 2 * COLS * self.cap * self.gnorm.slot, "H": 2 * COLS * self.cap * gu.c_slot,
                "Y": 2 * COLS * self.cap * CHUNK * self.y_pitch}

    def items(self, L: int, T: int, *, tok: object, sl: object, w: object, norm: str) -> tuple[list[tuple], int, int]:
        """Layer L's block for a chunk of T tokens (after its norm with lo panels, before the residual norm that adds
        DL): the runlist items and the two that start a piece (the shared expert's, the experts'). ``tok(buf, j, op,
        i, row)`` / ``sl(buf, off, op, i)``: the prefill's slices; ``w(gemm)``: a GEMM's weight slice; ``norm``: the
        post-attention norm weight's slice."""
        t = min(T, self.sub)
        S = T // t
        g = self.gemms(L, t)
        rb = self.rows()
        rl: list[tuple] = []
        r = g.router
        rl += [(r.op, tok("PXM", j, r.op, 0, rb["PXM"]), w(r), tok("LG", j, r.op, 2, rb["LG"])) for j in range(S)]
        i_shared = len(rl)
        rl += [(g.sgu.op, tok("PXM", j, g.sgu.op, 0, rb["PXM"]), w(g.sgu), tok("HS", j, g.sgu.op, 2, rb["HS"]))
               for j in range(S)]
        rl += [(g.sdown.op, tok("HS", j, g.sdown.op, 0, rb["HS"]), w(g.sdown), tok("DS", j, g.sdown.op, 2, rb["DS"]))
               for j in range(S)]
        i_experts = len(rl)
        gu, dn = self.gu[L], self.dn[L]
        cb = self.comb[T, dn.y_layout]
        rl.append((self.gnorm, sl("CBG", 0, self.gnorm, 0), norm, sl("XP", 0, self.gnorm, 2)))
        rl.append((gu, sl("CBU", 0, gu, 0), sl("XP", 0, gu, 1), sl("H", 0, gu, 2)))
        rl.append((dn, sl("CBD", 0, dn, 0), sl("H", 0, dn, 1), sl("Y", 0, dn, 2)))
        rl.append((cb, sl("CBC", 0, cb, 0), sl("DS", 0, cb, 1), sl("DL", 0, cb, 2)))
        return rl, i_shared, i_experts

    @property
    def operators(self) -> list[object]:
        return [self.gnorm, *self._ops.values(), *self.comb.values()]

    # ------------------------------------------------------------------
    # The host step
    # ------------------------------------------------------------------

    def host_step(self, L: int, *, n: int, T: int, dev: object, lib: object) -> MP.Plan:
        """Route the n valid tokens of a chunk of T from layer L's logits (LG) and write the experts' work: the control
        streams (CBG, CBU, CBD, CBC), the weights (DS rows' last ``WT_ROW`` fp32: the shared down writes the rest of
        them meanwhile), the chunk counts (``lib``'s scratchpad parameters)."""
        rb = self.rows()
        dev.fetch("LG", 0, T * rb["LG"])
        lg = dev.view("LG")[:T * rb["LG"]].view(np.float32).reshape(T, -1)
        pitch = lg.shape[1] // COLS
        ids, wts = MP.route(lg[:n], self.K, E=self.E, per_col=self.E // COLS, pitch=pitch)
        sg = lg[:n, MG.PASS // 2].astype(np.float64)
        plan = MP.Plan.of(ids, cap=self.cap, E=self.E)
        ctx = lib._ctx

        def addr(name: str, off: int = 0) -> int:
            return txn.device_address(dev.buffers[name], off, ctx)

        gu, dn = self.gu[L], self.dn[L]
        cb = self.comb[T, dn.y_layout]
        for name, op, region, out in (("CBU", gu, self.img.region(L, "exp_gu"), "H"),
                                      ("CBD", dn, self.img.region(L, "exp_down"), "Y")):
            v = dev.view(name).view(np.uint32)
            slab = addr("W", region.offset)
            for c in range(COLS):
                op.stream(plan.experts[c], cb=addr(name, 4 * c * op.cb_words), slab=slab, slab_stride=region.col_stride,
                          c_address=addr(out, 2 * c * self.cap * op.c_slot), out=v[c * op.cb_words:(c + 1) * op.cb_words])
        px = addr("PX") + 2 * self.H * np.arange(T, dtype=np.int64)
        v = dev.view("CBG").view(np.uint32)
        w = self.gnorm.cb_words
        for c in range(COLS):
            self.gnorm.stream(px[plan.rows[c]], plan.sizes[c], cb=addr("CBG", 4 * c * w),
                              xp=addr("XP", 2 * c * self.cap * self.gnorm.slot), out=v[c * w:(c + 1) * w])
        pos = np.zeros((T, self.K), np.int64)
        pos[:n] = plan.pos
        y = addr("Y") + 2 * dn.y_offset(pos)
        v = dev.view("CBC").view(np.uint32)
        tpc, w = T // COLS, cb.cb_words
        for c in range(COLS):
            cb.stream(y[c * tpc:(c + 1) * tpc], cb=addr("CBC", 4 * c * w), out=v[c * w:(c + 1) * w])
        wt = dev.view("DS")[:T * rb["DS"]].view(np.float32).reshape(T, -1)[:, COLS * self.pitch // 2:]
        wt[:] = 0
        wt[:n, :self.K] = wts
        wt[:n, self.K] = 1 / (1 + np.exp(-sg))
        for name in ("CBG", "CBU", "CBD", "CBC"):
            dev.flush(name)
        dev.flush("DS", 0, T * rb["DS"])  # (the weights' lines: the host writes no other line of DS)
        counts = plan.counts
        groups = [int((-(-plan.sizes[c] // CHAIN)).sum()) for c in range(COLS)]  # (the gather norm's)
        lib.set_params({**{f"{N_PARAM}{c}": counts[c] | groups[c] << 8 for c in range(COLS)},
                        **{f"{R_PARAM}{c}": counts[c] - 1 for c in range(COLS)}})
        return plan

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def layer_seconds(self, L: int, T: int, *, weight_s: float, row_s: float) -> float:
        """Estimated time of layer L's block for a chunk of T tokens: the experts' weights (an expert per chunk of
        CHUNK pairs: about min(E, T k) experts' slabs at weight_s per byte), the gathered rows and the outputs at
        row_s per byte."""
        slab = sum(self.img.region(L, r).col_bytes for r in ("exp_gu", "exp_down")) * 2  # (an expert: two halves)
        pairs = T * self.K
        chunks = min(self.E, pairs) + pairs / CHUNK
        rows = pairs * (2 * self.H + self.H * 9 / 8 + 2 * self.H + 2 * self.F * 9 / 8 + 2 * self.y_pitch * 2)
        return chunks * slab * weight_s + rows * row_s
