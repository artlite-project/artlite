"""Inkling's MoE in the prefill (``prefill.InklingPrefill``'s FFN; ``artlite.moe``'s work-list operators on
``image``'s expert units of 512, 512 and 128 channels of h, NVFP4). Per visit, after its FFN norm (XN: single bfp16
panels of rms(x) ffn_norm; XF: the FFN's input x as bf16 rows):

1. the router (``prefill.gemm.plan_phase``'s "router": its F32 rows as BFP16 planes, each expert's pair summed into
   fp32 logits) against x hi and x lo (XR: the FFN norm's [hi | lo] panels; LG, LG2, which the host sums: the decode's
   router takes x hi + lo too; single panels changed 2% of the top-8 sets a visit and 2.6x the prefill's KL to decode
   steps) — a piece of the program ends there;
2. the host's routing (``host_route``): the top 8 of sigmoid(l) + exp_probs_b (ties: the lower id), the weights
   sigmoid(l) / (their sum + 1e-20) times ``expert_weights_scale`` and the expert's down scale (``moe.route.top_k_bias``,
   the decode's); the chunks (``moe.plan.Plan``); the control streams of the gathered norm (CBG; its row groups' gate /
   up scales, GS: the experts' NVFP4 tensor scales, which the decode applies in its GEMVs), of the units' GEMMs (CBU0,
   CBU1, CBU2, CBD01, CBD2) and of the combine (CBC), the weights (DS), the columns' chunk counts;
3. the gathered norm (XP: rms(x) ffn_norm times the expert's gate / up scale), the units' gate / up (H01: units 0 and 1,
   one design, into columns 0 and 512, a unit of Q8_0's 8-row tiles in two parts of 256 columns: a chunk's drains within
   its lock's range; H2: unit 2; UD-Q4_K_M's Q5_K units end in zero windows: H01 of 1040 columns, H2 of 144, which the
   downs read at that pitch), C = situ(gate, up); their downs (Y01: units 0 and 1 as the two halves, row groups of
   2; Y2: unit 2's, row groups of 8); the combine (DL: per token sum_i w_i (Y01_i + Y2_i), bf16 rows; one per size and
   the downs' layouts: NVFP4's, Q8_0's), which the FFN's post conv takes.

The buffers are the chunk's, shared by the visits (one visit's block in flight). ``cap``: chunks per column at most
(a chunk of ``max(sizes)`` tokens: ceil((E + max_t k / CHUNK) / COLS))."""

import numpy as np

from artlite import txn
from artlite.moe import plan as MP
from artlite.moe.expert_gemm import CHUNK, MARKER, MAX_REPEAT, N_GROUPS, N_PARAM, ExpertGemmOperator, M
from artlite.moe.gather import CHAIN, CombineOperator, GatherNormOperator, ds_row
from artlite.moe.route import top_k_bias
from artlite.operators.design import COLS
from artlite.prefill import gemm as PG

UNITS01 = 1024  # (H01's columns: units 0 and 1)


def gu_span(region: object) -> int:
    """The H columns a unit's gate / up writes (a ``Region.qt`` slab region's): its groups' whole windows, 8 outputs
    each."""
    (b,) = region.blocks
    return region.groups(b) * 8 * (region.qg * b.R // 16)


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
        self.units, self.gemm_items = {}, {}
        for L in self.layers:
            r = {n: image.region(L, n) for n in ("exp_gu0", "exp_gu1", "exp_gu2", "exp_dn0", "exp_dn2")}
            g0 = r["exp_gu0"]
            tiles, qg = g0.blocks[0].tiles, g0.qg
            parts = 1 if 2 * tiles // qg <= MAX_REPEAT else 2  # (a part's drains: 2 per pass of a group)
            assert tiles % (parts * qg) == 0 and 2 * tiles // qg // parts <= MAX_REPEAT, (L, tiles)
            # (H's columns: a unit's whole passes from its first column, past the expert's outputs where its last group
            # ends in zero windows (Q5_K's: ``image.unit_gu_rows``); unit 0's spare ones are unit 1's, which comes later)
            w01, w2 = max(UNITS01, UNITS01 // 2 + gu_span(r["exp_gu1"])), max(cfg.expert_ffn - UNITS01, gu_span(r["exp_gu2"]))
            ops = {"gu01": self._op(ExpertGemmOperator.of("gu", g0, cap=self.cap, out=w01, expert_halves=1, act="situ",
                                                          situ_lb=lb, row_tiles=tiles // parts)),
                   "gu2": self._op(ExpertGemmOperator.of("gu", r["exp_gu2"], cap=self.cap, out=w2,
                                                         expert_halves=1, act="situ", situ_lb=lb)),
                   "dn01": self._op(ExpertGemmOperator.of("down", r["exp_dn0"], cap=self.cap, out=self.H,
                                                          a_pitch=w01 if w01 != UNITS01 else 0)),
                   "dn2": self._op(ExpertGemmOperator.of("down", r["exp_dn2"], cap=self.cap, out=self.H, expert_halves=1,
                                                         a_pitch=w2 if w2 != cfg.expert_ffn - UNITS01 else 0))}
            self.units[L] = (r, ops)
            # (per item: its control stream, its operator, its slab's offset in the image, its H column, A, C)
            gs = (qg * g0.n_kc - qg + 1) * g0.tile_bytes  # (a tail group: its row tiles' full chunks, its tail tile)
            part = UNITS01 // 2 // parts
            self.gemm_items[L] = [(f"CBU{u}{q}", "gu01", r[f"exp_gu{u}"].offset + q * tiles // parts // qg * gs,
                                   UNITS01 // 2 * u + part * q, "XP", "H01") for u in (0, 1) for q in range(parts)] + \
                [("CBU2", "gu2", r["exp_gu2"].offset, 0, "XP", "H2"), ("CBD01", "dn01", r["exp_dn0"].offset, 0, "H01", "Y01"),
                 ("CBD2", "dn2", r["exp_dn2"].offset, 0, "H2", "Y2")]
        self.ds_pitch = self.H // COLS
        self.comb = {}
        for _, u in self.units.values():
            for T in sizes:
                key = (T, u["dn01"].y_layout, u["dn2"].y_layout)
                if key not in self.comb:
                    self.comb[key] = CombineOperator(E=self.H, T=T, ds_pitch=self.ds_pitch, y_layout=key[1], K=self.K,
                                                     y2=key[2])
        # the host's tables per layer: exp_probs_b, the downs' scales times expert_weights_scale, the gate / up scales
        g = image.gguf

        def vec(name: str, default: float = 1.0) -> np.ndarray:  # (a layer without NVFP4 experts has no scales)
            if name not in g:
                return np.full(self.E, default)
            return np.asarray(g.tensor(name).dequantize(), np.float64).reshape(-1)

        self.tab = {L: (vec(f"blk.{L}.exp_probs_b.bias", 0.0).astype(np.float32),
                        vec(f"blk.{L}.ffn_down_exps.scale") * cfg.expert_weights_scale,
                        vec(f"blk.{L}.ffn_gate_up_exps.scale").astype(np.float32)) for L in self.layers}
        self.unit_offsets = {L: image.unit_offsets(L) for L in self.layers}

    def _op(self, op: ExpertGemmOperator) -> ExpertGemmOperator:
        """One object per distinct design (the layers share them)."""
        return self._ops.setdefault((op.kind, op.fmt, op.R, op.N, op.G, op.passes, op.cap, op.a_pitch), op)

    @property
    def operators(self) -> list[object]:
        return [self.gnorm, *self._ops.values(), *self.comb.values()]

    # ------------------------------------------------------------------
    # Operators and buffers
    # ------------------------------------------------------------------

    def router(self, L: int, t: int) -> PG.PhasePlan:
        """Layer L's router over sub-chunks of t tokens, its A panels [x hi | x lo] (``PrefillNormOperator(lo=True)``):
        an item per half (``lo_off``), the host sums their logits (``logits``)."""
        return PG.plan_phase(self.img, self.img.region(L, "router"), t, ops=self._gemms, a_width=2 * self.H)

    @property
    def lo_off(self) -> int:
        """Bytes from a token group's x hi panels to its x lo panels (the router's second item's A)."""
        return self.H // 8 * 72

    def rows(self, t: int) -> dict[str, int]:
        """Bytes per token of the per-token buffers (LG: the router's rows at sub-chunks of t; DS: the combine's
        zero ds rows and the weights; DL: the delta)."""
        lg = 2 * self.router(self.layers[0], t).layout.N
        return {"XR": 2 * self.H * 9 // 8, "LG": lg, "LG2": lg, "DS": 2 * ds_row(self.H, self.ds_pitch),
                "DL": 2 * self.H}

    def fixed(self) -> dict[str, int]:
        """Bytes of the chunk-independent buffers (the control streams, GS, XP, H01, H2, Y01, Y2: the layers' largest)."""
        out = {"CBG": 4 * COLS * self.gnorm.cb_words, "CBC": 4 * COLS * max(c.cb_words for c in self.comb.values()),
               "GS": 4 * COLS * self.gnorm.n_groups, "XP": 2 * COLS * self.cap * self.gnorm.slot}
        for L, items in self.gemm_items.items():
            ops = self.units[L][1]
            for name, op, _, _, _, c in items:
                o = ops[op]
                out[name] = max(out.get(name, 0), 4 * COLS * o.cb_words)
                out[c] = max(out.get(c, 0), 2 * COLS * self.cap * o.c_slot)
        return out

    def items(self, L: int, T: int, *, sl: object, norm: str) -> list[tuple]:
        """Layer L's block for a chunk of T tokens after its router (the host's routing before the first): the
        gathered norm, the units' GEMMs, the combine. ``sl(buf, off, op, i)``: the prefill's slices; ``norm``: the
        FFN norm weight's slice."""
        _, u = self.units[L]
        g = self.gnorm
        cb = self.comb[(T, u["dn01"].y_layout, u["dn2"].y_layout)]
        rl = [(g, sl("CBG", 0, g, 0), norm, sl("GS", 0, g, 2), sl("XP", 0, g, 3))]
        for name, op, _, _, a, c in self.gemm_items[L]:
            o = u[op]
            rl.append((o, sl(name, 0, o, 0), sl(a, 0, o, 1), sl(c, 0, o, 2)))
        rl.append((cb, sl("CBC", 0, cb, 0), sl("DS", 0, cb, 1), sl("DL", 0, cb, 2)))
        return rl

    # ------------------------------------------------------------------
    # The host step
    # ------------------------------------------------------------------

    def logits(self, L: int, *, n: int, t: int, dev: object) -> np.ndarray:
        """The n valid tokens' router logits (n, E) fp32: the sums of x hi's (LG) and x lo's (LG2), rows of the router's
        layout at sub-chunks of t."""
        lay = self.router(L, t).layout
        rb = 2 * lay.N
        o, per = lay.off["router"] // 2, self.E // COLS
        out = []
        for name in ("LG", "LG2"):
            dev.fetch(name, 0, n * rb)
            out.append(dev.view(name)[:n * rb].view(np.float32).reshape(n, COLS, lay.pitch // 2)[:, :, o:o + per])
        return (out[0] + out[1]).reshape(n, self.E)

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
        # (per buffer the ranges the host wrote, flushed alone: the buffers hold a 2048-token chunk's worst case)
        dirty = [("CBG", 4 * c * w, 4 * len(x)) for c, x in enumerate(
            g.streams([px[r] for r in plan.rows], plan.sizes, cb=[addr("CBG", 4 * c * w) for c in range(COLS)],
                      xp=[addr("XP", 2 * c * self.cap * g.slot) for c in range(COLS)],
                      out=[v[c * w:(c + 1) * w] for c in range(COLS)]))]
        gs = g.group_scales(plan.sizes, [gsc[e] for e in plan.experts])
        dev.view("GS")[:gs.nbytes] = gs.view(np.uint8).reshape(-1)
        _, u = self.units[L]
        uo = self.unit_offsets[L]
        active = [sz > M for sz in plan.sizes]  # (chunks of more than one MAC row's tokens)
        for name, op, slab, col0, a_buf, out in self.gemm_items[L]:
            o = u[op]
            v = dev.view(name).view(np.uint32)
            w = o.cb_words
            res = o.streams(plan.experts, active, cb=[addr(name, 4 * c * w) for c in range(COLS)],
                            slab=addr("W", slab), slab_stride=uo[1], expert_stride=uo[3], col0=col0,
                            c_address=[addr(out, 2 * c * self.cap * o.c_slot) for c in range(COLS)],
                            a_address=[addr(a_buf, 2 * c * self.cap * o.a_slot) for c in range(COLS)],
                            out=[v[c * w:(c + 1) * w] for c in range(COLS)])
            for c, x in enumerate(res):  # (its words, and the marker panel at the column's end)
                dirty += [(name, 4 * c * w, 4 * len(x)), (name, 4 * ((c + 1) * w - MARKER), 4 * MARKER)]
        pos = np.zeros((T, self.K), np.int64)
        pos[:n] = plan.pos
        rows = np.concatenate([addr("Y01") + 2 * u["dn01"].y_offset(pos), addr("Y2") + 2 * u["dn2"].y_offset(pos)], 1)
        cb = self.comb[(T, u["dn01"].y_layout, u["dn2"].y_layout)]
        v = dev.view("CBC").view(np.uint32)
        tpc, w = T // COLS, cb.cb_words
        for c in range(COLS):
            x = cb.stream(rows[c * tpc:(c + 1) * tpc], cb=addr("CBC", 4 * c * w), out=v[c * w:(c + 1) * w])
            dirty.append(("CBC", 4 * c * w, 4 * len(x)))
        rb = 2 * ds_row(self.H, self.ds_pitch)
        wt = dev.view("DS")[:T * rb].view(np.float32).reshape(T, -1)[:, COLS * self.ds_pitch // 2:]
        wt[:] = 0
        wt[:n, :self.K] = wts
        for name, a, b in dirty:
            dev.flush(name, a, b)
        dev.flush("GS", 0, gs.nbytes)
        dev.flush_at("DS", rb * np.arange(T, dtype=np.int64) + 2 * COLS * self.ds_pitch)  # (the weights' lines alone)
        groups = [int((-(-plan.sizes[c] // CHAIN)).sum()) for c in range(COLS)]  # (the gathered norm's)
        lib.set_params({f"{N_PARAM}{c}": plan.counts[c] | groups[c] << N_GROUPS for c in range(COLS)})
        return plan


__all__ = ["InkMoE"]
