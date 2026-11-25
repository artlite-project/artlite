"""Inkling's MoE in the prefill (``prefill.InklingPrefill``'s FFN; ``artlite.moe``'s work-list operators on
``image``'s expert units of 512, 512 and 128 channels of h, NVFP4). Per visit, after its FFN norm (XN: single bfp16
panels of rms(x) ffn_norm; XF: the FFN's input x as bf16 rows):

1. the router (``prefill.gemm.plan_phase``'s "router": its F32 rows as BFP16 planes, each expert's pair summed into
   fp32 logits, LG) — a piece of the program ends there;
2. the host's routing (``host_route``): the top 8 of sigmoid(l) + exp_probs_b (ties: the lower id), the weights
   sigmoid(l) / (their sum + 1e-20) times ``expert_weights_scale`` and the expert's down scale (``moe.route.top_k_bias``,
   the decode's); the chunks (``moe.plan.Plan``); the control streams of the gathered norm (CBG; its row groups' gate /
   up scales, GS: the experts' NVFP4 tensor scales, which the decode applies in its GEMVs), of the units' GEMMs (CBU0,
   CBU1, CBU2, CBD01, CBD2) and of the combine (CBC), the weights (DS), the columns' chunk counts;
3. the gathered norm (XP: rms(x) ffn_norm times the expert's gate / up scale), the units' gate / up (H01: units 0 and 1,
   one design, into columns 0 and 512; H2: unit 2), C = situ(gate, up); their downs (Y01: units 0 and 1 as the two
   halves, row groups of 2; Y2: unit 2's, row groups of 8); the combine (DL: per token sum_i w_i (Y01_i + Y2_i), bf16
   rows), which the FFN's post conv takes.

The buffers are the chunk's, shared by the visits (one visit's block in flight). ``cap``: chunks per column at most
(a chunk of ``max(sizes)`` tokens: ceil((E + max_t k / CHUNK) / COLS))."""

import numpy as np

from artlite import txn
from artlite.moe import plan as MP
from artlite.moe.expert_gemm import CHUNK, N_GROUPS, N_PARAM, ExpertGemmOperator, M
from artlite.moe.gather import CHAIN, CombineOperator, GatherNormOperator, ds_row
from artlite.moe.route import top_k_bias
from artlite.operators.design import COLS
from artlite.prefill import gemm as PG

UNITS01 = 1024  # (H01's columns: units 0 and 1)


class InkMoE:
    """The MoE blocks of an Inkling image's physical ``layers`` for chunks of ``sizes`` tokens in sub-chunks of at most
    ``sub`` (module docstring)."""

    def __init__(self, image: object, *, layers: list[int], sizes: tuple[int, ...], sub: int, eps: float):
        cfg = image.cfg
        self.img, self.cfg, self.layers, self.sub = image, cfg, list(layers), sub
        self.H, self.E, self.K = cfg.hidden_size, cfg.n_experts, cfg.n_experts_used
        self.cap = -(-(self.E + -(-max(sizes) * self.K // CHUNK)) // COLS)
        self._gemms: dict = {}
        self._ops: dict = {}
        self.gnorm = GatherNormOperator(E=self.H, eps=eps, cap=self.cap, scales=True)
        lb = cfg.situ_linear_beta
        self.units = {}
        for L in self.layers:
            r = {n: image.region(L, n) for n in ("exp_gu0", "exp_gu1", "exp_gu2", "exp_dn0", "exp_dn2")}
            self.units[L] = (r, {
                "gu01": self._op(ExpertGemmOperator.of("gu", r["exp_gu0"], cap=self.cap, out=UNITS01, expert_halves=1,
                                                       act="situ", situ_lb=lb)),
                "gu2": self._op(ExpertGemmOperator.of("gu", r["exp_gu2"], cap=self.cap, out=cfg.expert_ffn - UNITS01,
                                                      expert_halves=1, act="situ", situ_lb=lb)),
                "dn01": self._op(ExpertGemmOperator.of("down", r["exp_dn0"], cap=self.cap, out=self.H)),
                "dn2": self._op(ExpertGemmOperator.of("down", r["exp_dn2"], cap=self.cap, out=self.H, expert_halves=1))})
        dn01, dn2 = (self.units[self.layers[0]][1][n] for n in ("dn01", "dn2"))
        assert all(u["dn01"].y_layout == dn01.y_layout and u["dn2"].y_layout == dn2.y_layout
                   for _, u in self.units.values()), "one combine per size"
        self.ds_pitch = self.H // COLS
        self.comb = {T: CombineOperator(E=self.H, T=T, ds_pitch=self.ds_pitch, y_layout=dn01.y_layout, K=self.K,
                                        y2=dn2.y_layout) for T in sizes}
        # the host's tables per layer: exp_probs_b, the downs' scales times expert_weights_scale, the gate / up scales
        g = image.gguf

        def vec(name: str) -> np.ndarray:
            return np.asarray(g.tensor(name).dequantize(), np.float64).reshape(-1)

        self.tab = {L: (vec(f"blk.{L}.exp_probs_b.bias").astype(np.float32),
                        vec(f"blk.{L}.ffn_down_exps.scale") * cfg.expert_weights_scale,
                        vec(f"blk.{L}.ffn_gate_up_exps.scale").astype(np.float32)) for L in self.layers}
        self.unit_offsets = {L: image.unit_offsets(L) for L in self.layers}

    def _op(self, op: ExpertGemmOperator) -> ExpertGemmOperator:
        """One object per distinct design (the layers share them)."""
        return self._ops.setdefault((op.kind, op.fmt, op.R, op.N, op.G, op.passes, op.cap), op)

    @property
    def operators(self) -> list[object]:
        return [self.gnorm, *self._ops.values(), *self.comb.values()]

    # ------------------------------------------------------------------
    # Operators and buffers
    # ------------------------------------------------------------------

    def router(self, L: int, t: int) -> PG.PhasePlan:
        """Layer L's router over sub-chunks of t tokens."""
        return PG.plan_phase(self.img, self.img.region(L, "router"), t, ops=self._gemms)

    def rows(self, t: int) -> dict[str, int]:
        """Bytes per token of the per-token buffers (LG: the router's rows at sub-chunks of t; DS: the combine's
        zero ds rows and the weights; DL: the delta)."""
        return {"LG": 2 * self.router(self.layers[0], t).layout.N, "DS": 2 * ds_row(self.H, self.ds_pitch),
                "DL": 2 * self.H}

    def fixed(self) -> dict[str, int]:
        """Bytes of the chunk-independent buffers (the control streams, GS, XP, H01, H2, Y01, Y2)."""
        u = self.units[self.layers[0]][1]
        cb = {name: 4 * COLS * max(us[1][op].cb_words for us in self.units.values())
              for name, op in (("CBU0", "gu01"), ("CBU1", "gu01"), ("CBU2", "gu2"), ("CBD01", "dn01"), ("CBD2", "dn2"))}
        return {"CBG": 4 * COLS * self.gnorm.cb_words, **cb, "CBC": 4 * COLS * max(c.cb_words for c in self.comb.values()),
                "GS": 4 * COLS * self.gnorm.n_groups, "XP": 2 * COLS * self.cap * self.gnorm.slot,
                "H01": 2 * COLS * self.cap * u["gu01"].c_slot, "H2": 2 * COLS * self.cap * u["gu2"].c_slot,
                "Y01": 2 * COLS * self.cap * u["dn01"].c_slot, "Y2": 2 * COLS * self.cap * u["dn2"].c_slot}

    def items(self, L: int, T: int, *, sl: object, norm: str) -> list[tuple]:
        """Layer L's block for a chunk of T tokens after its router (the host's routing before the first): the
        gathered norm, the units' GEMMs, the combine. ``sl(buf, off, op, i)``: the prefill's slices; ``norm``: the
        FFN norm weight's slice."""
        _, u = self.units[L]
        g = self.gnorm
        cb = self.comb[T]
        rl = [(g, sl("CBG", 0, g, 0), norm, sl("GS", 0, g, 2), sl("XP", 0, g, 3))]
        for name, op, a, c in (("CBU0", "gu01", "XP", "H01"), ("CBU1", "gu01", "XP", "H01"), ("CBU2", "gu2", "XP", "H2"),
                               ("CBD01", "dn01", "H01", "Y01"), ("CBD2", "dn2", "H2", "Y2")):
            o = u[op]
            rl.append((o, sl(name, 0, o, 0), sl(a, 0, o, 1), sl(c, 0, o, 2)))
        rl.append((cb, sl("CBC", 0, cb, 0), sl("DS", 0, cb, 1), sl("DL", 0, cb, 2)))
        return rl

    # ------------------------------------------------------------------
    # The host step
    # ------------------------------------------------------------------

    def logits(self, L: int, *, n: int, t: int, dev: object) -> np.ndarray:
        """The n valid tokens' router logits (n, E) fp32 (LG: rows of the router's layout at sub-chunks of t)."""
        lay = self.router(L, t).layout
        rb = 2 * lay.N
        dev.fetch("LG", 0, n * rb)
        lg = dev.view("LG")[:n * rb].view(np.float32).reshape(n, COLS, lay.pitch // 2)
        o, per = lay.off["router"] // 2, self.E // COLS
        return np.ascontiguousarray(lg[:, :, o:o + per].reshape(n, self.E))

    def host_route(self, L: int, *, n: int, T: int, t: int, xf: str, dev: object, lib: object) -> MP.Plan:
        """Route the n valid tokens of a chunk of T from layer L's logits (LG) and write the block's work (module
        docstring's step 2): the gathered norm reads XF's rows (``xf``: its buffer)."""
        bias, dsc, gsc = self.tab[L]
        ids, wts = top_k_bias(self.logits(L, n=n, t=t, dev=dev), self.K, bias, dsc)
        plan = MP.Plan.of(ids.astype(np.int32), cap=self.cap, E=self.E)

        def addr(name: str, off: int = 0) -> int:
            return txn.device_address(dev.buffers[name], off, lib._ctx)

        g = self.gnorm
        px = addr(xf) + 2 * self.H * np.arange(T, dtype=np.int64)
        w = g.cb_words
        v = dev.view("CBG").view(np.uint32)
        g.streams([px[r] for r in plan.rows], plan.sizes, cb=[addr("CBG", 4 * c * w) for c in range(COLS)],
                  xp=[addr("XP", 2 * c * self.cap * g.slot) for c in range(COLS)],
                  out=[v[c * w:(c + 1) * w] for c in range(COLS)])
        gs = g.group_scales(plan.sizes, [gsc[e] for e in plan.experts])
        dev.view("GS")[:gs.nbytes] = gs.view(np.uint8).reshape(-1)
        regs, u = self.units[L]
        uo = self.unit_offsets[L]
        active = [sz > M for sz in plan.sizes]  # (chunks of more than one MAC row's tokens)
        for name, op, region, col0, a_buf, out in (("CBU0", "gu01", "exp_gu0", 0, "XP", "H01"),
                                                   ("CBU1", "gu01", "exp_gu1", 512, "XP", "H01"),
                                                   ("CBU2", "gu2", "exp_gu2", 0, "XP", "H2"),
                                                   ("CBD01", "dn01", "exp_dn0", 0, "H01", "Y01"),
                                                   ("CBD2", "dn2", "exp_dn2", 0, "H2", "Y2")):
            o = u[op]
            v = dev.view(name).view(np.uint32)
            w = o.cb_words
            o.streams(plan.experts, active, cb=[addr(name, 4 * c * w) for c in range(COLS)],
                      slab=addr("W", regs[region].offset), slab_stride=uo[1], expert_stride=uo[3], col0=col0,
                      c_address=[addr(out, 2 * c * self.cap * o.c_slot) for c in range(COLS)],
                      a_address=[addr(a_buf, 2 * c * self.cap * o.a_slot) for c in range(COLS)],
                      out=[v[c * w:(c + 1) * w] for c in range(COLS)])
        pos = np.zeros((T, self.K), np.int64)
        pos[:n] = plan.pos
        rows = np.concatenate([addr("Y01") + 2 * u["dn01"].y_offset(pos), addr("Y2") + 2 * u["dn2"].y_offset(pos)], 1)
        cb = self.comb[T]
        v = dev.view("CBC").view(np.uint32)
        tpc, w = T // COLS, cb.cb_words
        for c in range(COLS):
            cb.stream(rows[c * tpc:(c + 1) * tpc], cb=addr("CBC", 4 * c * w), out=v[c * w:(c + 1) * w])
        rb = 2 * ds_row(self.H, self.ds_pitch)
        wt = dev.view("DS")[:T * rb].view(np.float32).reshape(T, -1)[:, COLS * self.ds_pitch // 2:]
        wt[:] = 0
        wt[:n, :self.K] = wts
        for name in ("CBG", "GS", "CBU0", "CBU1", "CBU2", "CBD01", "CBD2", "CBC"):
            dev.flush(name)
        dev.flush("DS", 0, T * rb)
        groups = [int((-(-plan.sizes[c] // CHAIN)).sum()) for c in range(COLS)]  # (the gathered norm's)
        lib.set_params({f"{N_PARAM}{c}": plan.counts[c] | groups[c] << N_GROUPS for c in range(COLS)})
        return plan


__all__ = ["InkMoE"]
