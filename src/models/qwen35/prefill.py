"""The prefill of a qwen35 model: a prompt chunk of up to ``MAX_T`` tokens through every layer on the decode overlay's
buffers (``overlay.QwenOverlay``: the weight image, the norm weights, the GDN states and conv states, the KV caches), so
the decode continues from the state the chunk leaves.

A chunk of T tokens (``SIZES``: ``SMALL``, multiples of 256) runs layer-major in S = T / ``SUB`` sub-chunks (a ``SMALL``
size: one sub-chunk of operators of its own size, for short prompts): each operator of a layer runs over the
sub-chunks back to back on buffer slices a sub-chunk apart (consecutive runs of one operator share its configuration:
the ConfigureOps, ~75 ms of the 27B's chunk, are paid per chunk instead of per sub-chunk), the flash attention once over
the whole chunk. Per layer (the decode's order): the input norm (x += the last
projection's output,
``PrefillNormOperator``: bf16 residual, bfp16 panels of rms(x) w), the input projection (``prefill.gemm``), then

- GDN layer: ``GdnPrepOperator`` (q / k once per k head and the gates, d = beta = 0 for the padding tokens: the mask
  PM; records in PR; per sub-chunk, its conv state from the 3 rows of G before it: the previous sub-chunk's, the
  first's from the layer's rows of CVL, which the host fills from the decode's conv states in GHDR before each chunk),
  ``GdnScanOperator`` once over the whole chunk (the layer's states in GSTATE read and written once, its v conv state
  from CVL), ``GdnOutOperator`` (ssm_out's panels; the new conv state of the chunk's last valid tokens into GHDR);
- attention layer: ``AttnPrepOperator`` per sub-chunk (its K / V into the layer's KV cache: slices of the cache and the
  cos / sin table shifted by the sub-chunk's positions, the parameters the chunk's: the table's rows of its text
  positions, or of an image's tokens written past them), ``PrefillAttnOperator`` once;

the output projection, the post-attention norm, gate/up (silu(gate) up panels) and down. An output projection whose A
strip would be refetched per pass runs in K parts whose strips stay in the memtile where they cost less
(``prefill.gemm.resident_k_part``: the 27B's down in 4 or 6 parts, its attention and GDN outputs in 2), the next norm
adds the parts. The next layer's input norm ends the layer (the last layer's: x only; the LM head reads the last valid
row of x through the overlay, ``QwenOverlay.logits_of``).

The control code is assembled on the host from one library ELF and runs in pieces of whole layers
(``prefill.chunks.ChunkPrefill``).

A GEMM writes whole MAC tiles (GemmQd: 288 rows for 256 tokens): its padding rows land on the next sub-chunk's first
rows, which the next sub-chunk's GEMM writes afterwards (the last's in the buffers' padding); a projection in P parts
writes [part 0 | part 1 | ..] per sub-chunk, in slots of the largest part's C (the norm's d_stride). Every buffer of
the chunk (the prefill's own and the overlay's) lives on the overlay's device.

Chunks start at multiples of 32 positions (whole cache groups) with at least 3 valid tokens (the conv states: the
last three); the padding tokens after the valid ones leave the GDN states alone and write cache records that later
tokens overwrite. A non-causal model's prompt (``QwenConfig.causal``: vllm-d3) is one chunk at position 0, its
attention over all of its tokens (``attn_prefill.PrefillAttnOperator``'s non-causal design); a chunk of up to SUB tokens
runs the packed GDN operators with one segment from its first token (``fresh_sizes``: the scan starts from zero states
itself, so the model's reset zeroes none, ``QwenOverlay.reset``; the states it leaves are no sequence's), a larger one
zeroes the states first.

A readout model's prefill also runs packed chunks (``pack``, ``run_packed``): several inputs (segments) back to back in
a chunk at position 0, each from a state of its own. The operators that keep them apart: GdnPrep / GdnScan's packed
variants (the mask PM per token: 1 + its distance from its segment's first token; the state reset there exactly, a
segment's first three tokens' convs from its first token on), the packed attention (each row its segment's keys: up to
itself, or all of them non-causal; ``SEG``); every other operator works row by row. Each token's rotary position counts
from its segment's start (the overlay's rope rows: the chunk's cos / sin rows past the text positions, ``_cs_row``).
The chunk leaves no sequence in the caches and states.

A causal readout model's prefill also runs tree chunks (``tree_sizes``, ``run_packed`` at p0 > 0): segments that each
continue the cached prefix (a decision request's questions after its shared state), from multiples of ``GP.C`` tokens.
GdnPrep / GdnScan's tree variants start a segment's first scan chunk from the prefix's state (held in the scan cores' L1
beside the running one) and its conv windows from the prefix's rows (CVL); the packed attention sees the history; GdnOut
writes the prefix's conv state back from CVL. The chunk leaves the prefix's states as they were (its cache records past
p0 are the next prompt's to overwrite).
"""

import functools
import math
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from artlite.moe.prefill import MoEPrefill
from artlite.operators.design import COLS
from artlite.operators.fused.layout import parse_name
from artlite.operators.util import arg_bytes
from artlite.prefill import gemm as PG
from artlite.prefill.chunks import PIECE_BUDGET_S, ChunkPrefill
from artlite.prefill.norm import PrefillNormOperator

from . import attn_prefill as AP
from . import gdn as GD
from . import gdn_prefill as GP
from .dflash_prefill import DFlashPrefill
from .draft_prefill import DraftPrefill
from .overlay import CS_OFF, KV_PAD, QwenOverlay

SUB = 256  # tokens per sub-chunk (the operators' T)
SMALL = (64, 128)  # chunk sizes of operators of their own (short prompts: fewer passes)
SIZES = SMALL + tuple(range(SUB, 2049, SUB))  # chunk sizes (a flash attention design each)
MOE_SIZES = SIZES + (4096,)  # an MoE model's (its experts' weights stream once per chunk)
MAX_T = MOE_SIZES[-1]
assert MAX_T - SUB <= KV_PAD
PAD_ROWS = 32  # a GEMM's padding rows past a sub-chunk (GemmQd: 288 for 256)
# the time model's constants (``layer_seconds``; fitted to the chunk times of the 1B, 9B and 27B Q4_K_M, T = 256 ..
# 2048 at p0 = 0 .. 30K: within ~2%): per sub-chunk a byte of a layer's weights, a GEMM item, a norm per hidden unit,
# the GDN ops per v head, the attention prep per q head; a ConfigureOp (without the array reset: ``fused.program``)
WEIGHT_S, GEMM_S, NORM_S, GDN_S, PREP_S, CFG_S = 28.4e-12, 0.056e-3, 43e-9, 21.7e-6, 17e-6, 0.085e-3
# a SMALL size's operators: S + SMALL_S (1 - S) of a sub-chunk's terms (the weights stream whatever the tokens; T = 64
# / 128 within ~9%)
SMALL_S = 0.42
# an MoE layer's experts (``moe.prefill.MoEPrefill.layer_seconds``): per byte of their weights, of their rows
MOE_WEIGHT_S, MOE_ROW_S = 1 / 38e9, 1 / 38e9
# the flash attention per cached position and sub-chunk: the larger of its cache bytes at ATTN_BW (each pass streams
# the history, K and V of every KV head) and the cores' cycles per 32-key group and row panel (two streams / one)
ATTN_BW, ATTN_CYC_TWO, ATTN_CYC_ONE, CORE_HZ = 58e9, 1020, 1500, 1.81e9


def _key(lay: PG.Layout) -> tuple:
    return lay.pitch, tuple(sorted(lay.off.items()))


class QwenPrefill(ChunkPrefill):
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (default ``SIZES``, an MoE model's
    ``MOE_SIZES``; those within the overlay's caches: a larger chunk's transfers leave them, and aiecc rejects the
    design); ``run`` runs one."""

    SUB, SMALL, MAX_T, MIN_TOKENS, NAME = SUB, SMALL, MAX_T, 3, "qwen_prefill"

    def __init__(self, overlay: QwenOverlay, *, sizes: tuple[int, ...] | None = None, draft: object = None,
                 dflash: object = None, precise: bool = False, pack: bool = False):
        """draft, dflash: an MTP drafter, a DFlash2 drafter whose records a chunk writes; precise: the attention's P as
        hi + lo panels (``attn_prefill.PrefillAttnOperator``'s ``pp``; d3-lite's attention error 9e-3 -> 4e-3, the
        output's own rounding: the readout models); pack: packed chunks of every size (``run_packed``)."""
        default = MOE_SIZES if overlay.cfg.moe else SIZES
        super().__init__(overlay, sizes or default)
        self.max_t = default[-1]  # (the model's largest chunk: GdnOut's conv-state rows of G)
        cfg, img = self.cfg, self.img
        self.layers = overlay.layers
        self.gdn = [L for L in self.layers if not cfg.attention[L]]
        self.att = [L for L in self.layers if cfg.attention[L]]
        self.kv_layers = self.att
        self.pitch = {name: PG.phase_pitch(img, name, SUB) for name, ls in (("gdn_in", self.gdn), ("attn_in", self.att))
                      if ls}
        self.d_pitch = PG.d_pitch(img, SUB)
        # the norms' panel rows: the hidden size padded to whole K chunks (the padding columns never written: zero)
        self.kh = -(-cfg.hidden_size // img.kc) * img.kc
        self._out_pitch = self.kh if self.kh != cfg.hidden_size else 0
        # (an MoE model's FFN: ``moe.prefill``; its marks per size: (layer, the items that start its pieces: the shared
        # expert's, the gathered norm's, the expert GEMMs'))
        self.moe = MoEPrefill(img, layers=self.layers, sizes=self.sizes, sub=SUB, eps=cfg.rms_norm_eps) \
            if cfg.moe else None
        self._moe_marks: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
        self._gemms: dict[str, object] = {}  # (one object per distinct GEMM)
        self._k_parts: dict[tuple, int] = {}  # (per region shape and size: output projections in resident K parts)
        self.sz = {t: self._size_ops(t) for t in sorted({self._t(T) for T in self.sizes})}
        self.plans = self.sz[SUB].plans if SUB in self.sz else self.sz[max(self.sz)].plans
        self._ops: dict[tuple, object] = {}
        # (the prefill's own static buffers: per GDN layer its scan parameters, per attention layer its q / k norms)
        self.n_p = GP.params_bytes(overlay.gsh.hc) if self.gdn else 0
        self.pp_off = {L: i * self.n_p for i, L in enumerate(self.gdn)}
        self.qn_off = {L: i * 4 * 2 * AP.HD for i, L in enumerate(self.att + ([draft.L] if draft is not None else []))}
        # (an MTP drafter's cache records of the prompt: ``draft_prefill``)
        self.dp = DraftPrefill(draft, self) if draft is not None else None
        self.fp = DFlashPrefill(dflash, self) if dflash is not None else None
        self.attn = {T: AP.PrefillAttnOperator(T=T, cfg=cfg, head_bytes=overlay.ash.head_bytes, causal=cfg.causal,
                                               pp=2 if precise else 1)
                     for T in self.sizes} if self.att else {}
        pack = pack and not cfg.moe and draft is None and dflash is None  # (a dense model's prefill alone)
        self.pack_sizes = self.sizes if pack else ()
        self.span = self.sizes[-1] if pack else 0  # (the attention streams a packed chunk whole: any input that fits)
        self.pattn = {T: AP.PrefillAttnOperator(T=T, cfg=cfg, head_bytes=overlay.ash.head_bytes, causal=cfg.causal,
                                                pp=2 if precise else 1, pack=True)
                      for T in self.pack_sizes} if self.att else {}
        # (tree chunks: a causal model's segments after a prefix; GdnOut's conv state the prefix's, from CVL)
        self.tree_sizes = self.pack_sizes if cfg.causal and self.gdn else ()
        # (a non-causal model's chunks of up to SUB tokens: the packed GDN operators, one segment from the first token, so
        # that no reset zeroes its states: ``run``)
        self.fresh_sizes = tuple(T for T in self.pack_sizes if T <= SUB) if not cfg.causal and self.gdn else ()
        self._starts: dict[int, list[int]] = {}
        self._rows()
        self._library()
        self.runlists |= {("fresh", T): self._runlist_of(T, fresh=True) for T in self.fresh_sizes}

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    @staticmethod
    def _t(T: int) -> int:
        """The sub-chunk of a chunk of size T (a ``SMALL`` size: itself)."""
        return min(T, SUB)

    def _size_ops(self, t: int) -> SimpleNamespace:
        """The operators of sub-chunks of t tokens: the GEMM plans (the input projections at SUB's pitch, so G's layout
        and the conv states' rows in it are every size's: gdn_in's planar passes SUB's where t's would widen the pitch
        or move the qkv block; the output projections at t's own), the norms."""
        cfg, img, H = self.cfg, self.img, self.H
        dpt = PG.d_pitch(img, t)

        def plan(L: int, name: str) -> PG.PhasePlan:
            region = img.region(L, name)
            pitch = self.pitch.get(name, dpt) if name != "gate_up" else 0
            key = (name, region.K, region.kc_group, tuple(b.fmt for b in region.blocks), t)
            if key not in self._k_parts:
                self._k_parts[key] = PG.resident_k_part(img, region, t, pitch=pitch)
            plan = PG.plan_phase(img, region, t, pitch=pitch, ops=self._gemms, k_part=self._k_parts[key])
            if name == "gdn_in" and t != SUB:  # (the conv states' rows in G: SUB's pitch and qkv block)
                sub = PG.plan_phase(img, region, SUB, pitch=pitch).layout
                if (plan.layout.pitch, plan.layout.off["qkv"]) != (sub.pitch, sub.off["qkv"]):
                    plan = PG.plan_phase(img, region, t, pitch=pitch, ops=self._gemms, k_part=self._k_parts[key],
                                         layout_T=SUB)
            return plan

        plans = {(L, name): plan(L, name) for L in self.layers for name in self._phases(L)}
        eps, out = cfg.rms_norm_eps, self._out_pitch
        dp = dpt if dpt != H // COLS else 0
        cols = 4 if t < SUB else COLS  # (a SMALL size's norms on half the columns: d3-lite T 64 / 128 -0.7 / -0.3 ms in
        # one process, the 9B no slower; at 256 no faster)
        # (an output projection in P K parts: per sub-chunk [part 0 C | part 1 C | ..], slots of the largest part's C;
        # one norm adds them, per P)
        parted = [g.op for p in plans.values() if p.parts > 1 for g in p.gemms]
        f_part = max(arg_bytes(op, 2) for op in parted) if parted else 0
        return SimpleNamespace(
            plans=plans, d_pitch=dpt, f_part=f_part, rb_y=2 * COLS * dpt,
            pn0=PrefillNormOperator(T=t, add=False, E=H, eps=eps, out_pitch=out, cols=cols),
            pn=PrefillNormOperator(T=t, E=H, eps=eps, d_pitch=dp, out_pitch=out, cols=cols),
            # (an MoE layer's delta: rows of H)
            pnm=PrefillNormOperator(T=t, E=H, eps=eps, out_pitch=out, cols=cols) if dp else None,
            pnd={P: PrefillNormOperator(T=t, E=H, eps=eps, d_pitch=dp, deltas=P, d_stride=f_part // 2, out_pitch=out,
                                        cols=cols)
                 for P in sorted({p.parts for p in plans.values() if p.parts > 1})})

    def _phases(self, L: int) -> tuple[str, ...]:
        """(an MoE layer's FFN: ``moe``)"""
        mixer = ("attn_in", "attn_out") if self.cfg.attention[L] else ("gdn_in", "gdn_out")
        return mixer if self.cfg.moe else mixer + ("gate_up", "down")

    def _op(self, cls: type, lay: PG.Layout | None = None, **kw: object) -> object:
        """One operator object per class, layout and arguments (the layers share them: one design each in an ELF)."""
        key = (cls, None if lay is None else _key(lay), tuple(sorted(kw.items())))
        if key not in self._ops:
            args = {"T": SUB, "cfg": self.cfg} | ({"layout": lay} if lay is not None else {}) | kw
            self._ops[key] = cls(**args)
        return self._ops[key]

    def _gdn_ops(self, lay: PG.Layout, T: int, packed: bool = False, tree: bool = False
                 ) -> tuple[GP.GdnPrepOperator, GP.GdnScanOperator, GP.GdnOutOperator]:
        """(the scan over the chunk of T tokens in one call: the states read and written once; ``packed``, ``tree``:
        those variants; a tree chunk's GdnOut reads the conv state from the layer's 3 rows of CVL)"""
        t = self._t(T)
        pk = {"pack": True} if packed else {"tree": True} if tree else {}
        return (self._op(GP.GdnPrepOperator, lay, T=t, **pk), self._op(GP.GdnScanOperator, lay, T=T, sub=t, **pk),
                self._op(GP.GdnOutOperator, lay, T=t, hslot=self.ov.hslot, cs_off=CS_OFF,
                         gc_rows=3 if tree else self.max_t))

    def _prep(self, lay: PG.Layout, T: int) -> AP.AttnPrepOperator:
        """(its cos / sin table: the text positions, one sub-chunk past them for a chunk's own rows: ``_cs_row``)"""
        return self._op(AP.AttnPrepOperator, lay, T=self._t(T), head_bytes=self.ov.ash.head_bytes,
                        cs_positions=self.ov.max_positions + SUB)

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers (G and A: per phase kind)."""
        cfg, H = self.cfg, self.H
        self.rb = {"PX": 2 * H, "PXN": 9 * self.kh // 8, "Y": 2 * COLS * self.d_pitch, "F": 2 * COLS * self.d_pitch, "PM": 4}
        if self.moe:
            self.rb |= self.moe.rows()
        else:  # (down's K: the ffn padded to whole K chunks)
            self.rb["HP"] = 9 * self.img.region(self.layers[0], "down").K // 8
        if self.dp is not None:
            self.rb |= self.dp.rows()
        if self.fp is not None:
            self.rb |= self.fp.rows()
        self.rb_g = {k: 2 * COLS * p for k, p in self.pitch.items()}  # (gdn_in / attn_in rows)
        self.rb_a = {"gdn_in": 9 * cfg.ssm_v_heads * GD.D // 8, "attn_in": 9 * cfg.num_attention_heads * AP.HD // 8}
        self.rb_o = 2 * cfg.ssm_v_heads * GD.D

    def _norm(self, L: int, name: str) -> str:
        o = self.ov.norm_off[(L, name)]
        return f"NORM[{o}:{o + 2 * self.H}]"

    def _next_norm(self, L: int) -> str:
        """The norm weight after layer L's down: the next layer's input norm (the last layer: the LM head's, unused)."""
        i = self.layers.index(L)
        return self._norm(self.layers[i + 1], "attn_norm") if i + 1 < len(self.layers) else self._norm(-1, "output_norm")

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False, fresh: bool = False) -> list[tuple]:
        """(sub-chunks of ``_t`` tokens: a ``SMALL`` size one of its own operators; the layers' first items in
        ``_starts``; ``packed``, ``tree``: the packed or tree GDN operators and the packed attention, the segments in
        SEG; ``fresh``: the packed GDN operators and the unpacked attention)"""
        ov, cfg, sl = self.ov, self.cfg, self._sl
        t = self._t(T)
        S, o = T // t, self.sz[t]

        def tok(buf: str, j: int, op: object, *, i: int, row: int = 0, extra: int = 0) -> str:
            return sl(buf, j * t * (row or self.rb[buf]) + extra, op, i=i)

        ry = o.rb_y  # (bytes per token of Y and F at these operators' pitch)
        rl: list[tuple] = [(o.pn0, tok("PX", j, o.pn0, i=0), self._norm(self.layers[0], "attn_norm"),
                            tok("PXN", j, o.pn0, i=2)) for j in range(S)]
        if self.dp is not None:
            rl += self.dp.start_items(T, tok)
        starts = []
        marks = []
        for L in self.layers:
            starts.append(len(rl))
            if self.fp is not None:
                rl += self.fp.tap_items(L, T, tok)
            kind = self._phases(L)[0]
            p_in = o.plans[(L, kind)]
            g_rb, a_rb = self.rb_g[kind], self.rb_a[kind]
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("G", j, g.op, i=2, row=g_rb))
                   for g in p_in.gemms for j in range(S)]
            if cfg.attention[L]:
                prep, attn = self._prep(p_in.layout, T), (self.pattn if packed or tree else self.attn)[T]
                kv0, hb = ov.kv_off[L], ov.ash.kvh * ov.ash.head_bytes
                qn = f"PQN[{self.qn_off[L]}:{self.qn_off[L] + 4 * 2 * AP.HD}]"
                qs_g = COLS * 2 * prep.qc * AP.SLOT  # (bytes of a token group's Q slots: QS pass-major)
                grp = t // 32 * AP.GROUP  # (a sub-chunk's cache groups)
                rl += [(prep, tok("G", j, prep, i=0, row=g_rb), sl("PCS", j * t * 256, prep, i=1), qn,
                        sl("QS", j * t // 8 * qs_g, prep, i=3), f"KV[{kv0 + j * grp}:{kv0 + j * grp + hb}]")
                       for j in range(S)]
                rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + hb}]", sl("A", 0, attn, i=2),
                           *([sl("SEG", 0, attn, i=3)] if packed or tree else [])))
            else:
                gprep, scan, gout = self._gdn_ops(p_in.layout, T, packed or fresh, tree)
                i = self.gdn.index(L)
                pp = f"PP[{self.pp_off[L]}:{self.pp_off[L] + self.n_p}]"
                gh = f"GHDR[{ov.ghdr[L]}:{ov.ghdr[L] + gout.ghdr_bytes}]"
                gs = f"GSTATE[{ov.gstate[L]}:{ov.gstate[L] + COLS * ov.gsh.n_panels * GD.PANEL}]"
                cv = [sl("CVL", 3 * i * g_rb, gprep, i=1)] + [sl("G", (j * t - 3) * g_rb, gprep, i=1)
                                                              for j in range(1, S)]
                pr = arg_bytes(gprep, 4)  # (a sub-chunk's records)
                pm = arg_bytes(gprep, 3) if packed or tree or fresh else 4 * t  # (an object per sub-chunk, GdnPrep.mask)
                rl += [(gprep, tok("G", j, gprep, i=0, row=g_rb), cv[j], pp, sl("PM", j * pm, gprep, i=3),
                        sl("PR", j * pr, gprep, i=4)) for j in range(S)]
                rl.append((scan, sl("PR", 0, scan, i=0), tok("G", 0, scan, i=1, row=g_rb),
                           sl("CVL", 3 * i * g_rb, scan, i=2), pp, gs, tok("O", 0, scan, i=5, row=self.rb_o)))
                gc = sl("CVL", 3 * i * g_rb, gout, i=2) if tree else sl("G", 0, gout, i=2)
                rl += [(gout, tok("O", j, gout, i=0, row=self.rb_o), tok("G", j, gout, i=1, row=g_rb), gc, pp, gh,
                        tok("A", j, gout, i=5, row=a_rb)) for j in range(S)]
            out, fp = o.plans[(L, self._phases(L)[1])], o.f_part
            pan = self._norm(L, "post_attention_norm")
            if self.moe:  # (the out projection, the norm with lo panels, the MoE block, the residual norm)
                assert out.parts == 1, "an MoE layer's out projection in one part"
                (g,) = out.gemms
                ln = self.moe.ln[t]
                rl += [(g.op, tok("A", j, g.op, i=0, row=a_rb), self._w(g), tok("Y", j, g.op, i=2, row=ry))
                       for j in range(S)]
                rl += [(ln, tok("PX", j, ln, i=0), tok("Y", j, ln, i=1, row=ry), pan, tok("PX", j, ln, i=3),
                        tok("PXM", j, ln, i=4)) for j in range(S)]
                items, at = self.moe.items(L, T, tok=lambda b, j, op, i, row: tok(b, j, op, i=i, row=row),
                                           sl=lambda b, off, op, i: sl(b, off, op, i=i), w=self._w, norm=pan)
                marks.append((L, tuple(len(rl) + i for i in at)))
                rl += items
                pm = o.pnm or o.pn
                rl += [(pm, tok("PX", j, pm, i=0), tok("DL", j, pm, i=1), self._next_norm(L), tok("PX", j, pm, i=3),
                        tok("PXN", j, pm, i=4)) for j in range(S)]
                continue
            if out.parts == 1:
                (g,) = out.gemms
                rl += [(g.op, tok("A", j, g.op, i=0, row=a_rb), self._w(g), tok("Y", j, g.op, i=2, row=ry))
                       for j in range(S)]
                rl += [(o.pn, tok("PX", j, o.pn, i=0), tok("Y", j, o.pn, i=1, row=ry), pan, tok("PX", j, o.pn, i=3),
                        tok("PXN", j, o.pn, i=4)) for j in range(S)]
            else:  # (sub-chunk j's part k at (P j + k) f_part)
                P, pn = out.parts, o.pnd[out.parts]
                rl += [(g.op, tok("A", j, g.op, i=0, row=a_rb, extra=2 * g.a_off), self._w(g),
                        sl("Y", (P * j + g.part) * fp, g.op, i=2)) for g in out.gemms for j in range(S)]
                rl += [(pn, tok("PX", j, pn, i=0), sl("Y", P * j * fp, pn, i=1), pan, tok("PX", j, pn, i=3),
                        tok("PXN", j, pn, i=4)) for j in range(S)]
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("HP", j, g.op, i=2))
                   for g in o.plans[(L, "gate_up")].gemms for j in range(S)]
            down = o.plans[(L, "down")]
            nxt = self._next_norm(L)
            if down.parts == 1:
                (g,) = down.gemms
                rl += [(g.op, tok("HP", j, g.op, i=0), self._w(g), tok("F", j, g.op, i=2, row=ry)) for j in range(S)]
                rl += [(o.pn, tok("PX", j, o.pn, i=0), tok("F", j, o.pn, i=1, row=ry), nxt, tok("PX", j, o.pn, i=3),
                        tok("PXN", j, o.pn, i=4)) for j in range(S)]
            else:
                P, pn = down.parts, o.pnd[down.parts]
                rl += [(g.op, tok("HP", j, g.op, i=0, extra=2 * g.a_off), self._w(g),
                        sl("F", (P * j + g.part) * fp, g.op, i=2)) for g in down.gemms for j in range(S)]
                rl += [(pn, tok("PX", j, pn, i=0), sl("F", P * j * fp, pn, i=1), nxt, tok("PX", j, pn, i=3),
                        tok("PXN", j, pn, i=4)) for j in range(S)]
        if self.dp is not None:
            rl += self.dp.end_items(T, tok, lambda b, off, op, i: sl(b, off, op, i=i))
        if self.fp is not None:
            rl += self.fp.end_items(T, tok, lambda b, off, op, i: sl(b, off, op, i=i))
        self._starts[T] = [*starts, len(rl)]
        self._moe_marks[T] = marks
        return rl

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        self.cs_rows = own.get("PCS", 0) // 256  # (the cos / sin table: every position the sub-chunks' slices reach)

    def _bind(self) -> None:
        """The GDN layers' scan parameters (PP), the attention layers' q / k norms (PQN), the cos / sin table (PCS)."""
        ov, dev, cfg, own = self.ov, self.ov.dev, self.cfg, self.own
        if "PP" in own:
            pp = dev.view("PP")
            for L, o in self.pp_off.items():
                pp[o:o + self.n_p] = GP.scan_params(ov.gguf, cfg, L, self.img.region(L, "gdn_in").block("qkv").rows)
            dev.flush("PP")
        if "PQN" in own:
            qn = dev.view("PQN")
            for L in self.att:
                o = self.qn_off[L]
                w = np.concatenate([ov._f32(f"blk.{L}.attn_q_norm.weight"), ov._f32(f"blk.{L}.attn_k_norm.weight")])
                qn[o:o + w.nbytes] = w.view(np.uint8)
            dev.flush("PQN")
        if "PCS" in own:
            cs = AP.cs_table(cfg, self.cs_rows).view(np.uint8).reshape(-1)
            dev.view("PCS")[:cs.size] = cs
            dev.flush("PCS")
        if self.dp is not None:
            self.dp.static(dev, ov)
        if self.fp is not None:
            self.fp.static(dev)
        if self.gdn:
            self._conv_index()

    def _external(self) -> set[str]:
        return (self.dp.external() if self.dp is not None else set()) | (self.fp.external() if self.fp is not None else set())

    def _program(self, key: int | tuple[str, int]) -> list[tuple[int, list[int]]]:
        """(a chunkwise GDN core of 4: the GDN states in the half that the overlay's next position's block starts from,
        ``QwenOverlay._half``)"""
        ov = self.ov
        if not (ov.blocks and ov._gcur):
            return super()._program(key)
        k1 = ("half", key)
        if k1 not in self._items:
            def other(name: str) -> str:
                base, a, b = parse_name(name)
                return f"GSTATE[{a + ov.ghalf}:{b + ov.ghalf}]" if base == "GSTATE" else name

            rl = [(op, *map(other, names)) for op, *names in self.runlists[key]]
            self._items[k1] = self.lib.items(rl, self._buffers())
        return self._items[k1]

    def _buffers(self) -> dict[str, object]:
        return self.ov.dev.buffers | (self.dp.buffers() if self.dp is not None else {}) | \
            (self.fp.buffers() if self.fp is not None else {})

    def _conv_index(self) -> None:
        """The GHDR bytes of the GDN layers' conv states (3 rows x the column's channels, bf16) and their places in CVL
        (the layers' rows in G's layout: layer i's 3 rows from row 3 i)."""
        ov = self.ov
        ch, rb = ov.gsh.ch, self.rb_g["gdn_in"]
        src, dst = [], []
        for i, L in enumerate(self.gdn):
            lay = self.plans[(L, "gdn_in")].layout
            for c in range(COLS):
                s = ov.ghdr[L] + c * ov.hslot + CS_OFF + 2 * np.arange(3 * ch)
                r, k = np.divmod(np.arange(3 * ch), ch)
                src.append(s)
                dst.append((3 * i + r) * rb + 2 * (c * lay.pitch + lay.off["qkv"] + k))
        self._cv_src, self._cv_dst = np.concatenate(src) // 2, np.concatenate(dst) // 2  # (bf16 indices)
        lo, hi = ov.ghdr[self.gdn[0]], ov.ghdr[self.gdn[-1]] + COLS * ov.hslot
        self._cv_range = (lo, hi - lo)
        self._cvl_zero = False  # (CVL's conv rows zero: only the host writes CVL)

    def _conv_states(self) -> None:
        """CVL from GHDR: the GDN layers' conv states as the decode left them (or the last chunk)."""
        dev = self.ov.dev
        dev.fetch("GHDR", *self._cv_range)
        cvl = dev.view("CVL").view(np.uint16)
        cvl[self._cv_dst] = dev.view("GHDR").view(np.uint16)[self._cv_src]
        dev.flush("CVL", 0, 2 * (int(self._cv_dst.max()) + 1))
        self._cvl_zero = False

    def _conv_zero(self) -> None:
        """CVL's conv-state rows zero (a chunk at position 0; nothing to do when they are since the last
        ``_conv_states``: ~0.35 ms of a short readout's host work)."""
        if self._cvl_zero:
            return
        dev = self.ov.dev
        dev.view("CVL").view(np.uint16)[self._cv_dst] = 0
        dev.flush("CVL", 0, 2 * (int(self._cv_dst.max()) + 1))
        self._cvl_zero = True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int, n: int, cs_row: int | None = None, tree: bool = False) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0 with n valid tokens (their cos / sin from
        table row ``cs_row``, default p0; ``tree``: a tree chunk's)."""
        out = self.moe.params(T) if self.moe else {}
        if self.gdn and not tree:  # (a tree chunk's GdnOut: the conv state at CVL's rows, no parameter)
            out[GP.CONV_OFF] = self._gdn_ops(self.sz[self._t(T)].plans[(self.gdn[0], "gdn_in")].layout, T)[2] \
                .conv_offset(n)
        if self.att:
            prep = self._prep(self.sz[self._t(T)].plans[(self.att[0], "attn_in")].layout, T)
            out |= prep.params(p0, cs_row) | self.attn[T].params(p0, n)
        if self.fp is not None:
            out |= self.fp.params(T, p0)
        return out

    def _cs_row(self, p0: int, n: int) -> int:
        """The cos / sin table row of a chunk's first token: its text position where the n valid tokens' positions
        are consecutive text ones (``QwenOverlay.rope``), else the rows past the text positions, written with the
        tokens'."""
        rope = self.ov.rope[p0:p0 + n]
        r0 = int(rope[0, 0])
        if (rope == r0 + np.arange(n)[:, None]).all():
            return r0
        R = self.ov.max_positions
        dev = self.ov.dev
        dev.view("PCS")[256 * R:256 * (R + n)] = AP.cs_rows(self.cfg, rope).view(np.uint8).reshape(-1)
        dev.flush("PCS", 256 * R, 256 * n)
        return R

    def _fits_budget(self, T: int, p0: int) -> bool:
        """(an attention layer's flash attention within the budget)"""
        return not self.att or self.attention_seconds(p0, self.subs(T)) <= PIECE_BUDGET_S

    def run(self, x: np.ndarray, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the first layer's input rows of
        positions p0 .. p0 + n - 1 (the embeddings; the caches and states hold the positions before). Leaves the last
        layer's output in PX (``hidden``)."""
        n, dev = x.shape[0], self.ov.dev
        T = self._size(n, p0, T)
        assert 3 <= n or not self.gdn, n
        px = dev.view("PX")[:2 * T * self.H].view(bfloat16).reshape(T, self.H)
        px[:n] = np.asarray(x, np.float32)
        px[n:] = 0
        dev.flush("PX", 0, 2 * T * self.H)
        fresh = p0 == 0 and T in self.fresh_sizes  # (a non-causal model's: the packed scan zeroes the states itself)
        if self.gdn:
            m = GP.GdnPrepOperator.mask(n, T, np.zeros(n, np.int64), self._t(T)) if fresh else np.arange(T) < n
            dev.view("PM").view(np.float32)[:m.size] = m
            dev.flush("PM", 0, 4 * m.size)
            if p0 == 0:  # (a sequence's first chunk: no conv history)
                self.ov.conv_fresh()
                self._conv_zero()
                if not self.cfg.causal and not fresh:  # (its reset leaves the states as they are)
                    self.ov.zero_state()
            else:
                self.ov.conv_for_prefill()  # (the decode's conv states into GHDR)
                self._conv_states()
        if self.dp is not None:
            self.dp.before(dev, p0)
        if self.fp is not None:
            self.fp.before(dev, p0, T)
        hooks = {}
        for L, (_, i_e, i_g) in self._moe_marks.get(T, ()):
            hooks[i_e] = functools.partial(self.moe.host_route, L, n=n, T=T, dev=dev, lib=self.lib)
            hooks[i_g] = functools.partial(self.moe.host_streams, L, n=n, T=T, dev=dev, lib=self.lib)
        self._launch(T, p0, self.params(T, p0, n, self._cs_row(p0, n) if self.att else None), hooks or None,
                     key=("fresh", T) if fresh else None)
        if self.dp is not None:
            self.dp.after(dev, n)
        if self.fp is not None:
            self.fp.after(p0, n, T)

    def run_packed(self, x: np.ndarray, p0: int, pos: np.ndarray, lo: np.ndarray, T: int = 0) -> None:
        """A packed chunk of size T at position p0 (default: the smallest of ``packed_sizes(p0)`` that holds x): x (n,
        hidden) the segments' first layer input rows back to back, ``pos`` (n,) their text positions or (n, 3) their
        M-RoPE ones (images), ``lo`` (n,) per token its segment's first (an index into x). Leaves the last layer's output rows in PX (``hidden``). At position
        0 the caches and states hold no sequence after it; after a prefix (p0 > 0: a tree chunk, segments from multiples
        of ``seg_unit``, the tokens between them -1 in ``lo``: padding of the segment before them, zero rows that see its
        keys, so that its last V records round as alone) they hold the prefix as before it."""
        n, dev, tree = x.shape[0], self.ov.dev, p0 > 0
        sizes = self.packed_sizes(p0)
        T = T or next(t for t in sizes if t >= n)
        gap = lo < 0
        assert T in sizes and (lo <= np.arange(n)).all() and (tree or not gap.any()), (T, p0)
        self._size(n, p0, T)
        px = dev.view("PX")[:2 * T * self.H].view(bfloat16).reshape(T, self.H)
        px[:n] = np.asarray(x, np.float32)
        px[n:] = 0
        px[np.flatnonzero(gap)] = 0
        dev.flush("PX", 0, 2 * T * self.H)
        if self.gdn:
            m = GP.GdnPrepOperator.mask(n, T, lo, self._t(T), tree=tree)
            dev.view("PM").view(np.float32)[:m.size] = m
            dev.flush("PM", 0, 4 * m.size)
            if tree:  # (the prefix's conv states: the prep's and scan's window at a segment's start, GdnOut's)
                self.ov.conv_for_prefill()
                self._conv_states()
            else:
                self.ov.conv_fresh()
                self._conv_zero()
        cs_row, rope = None, None
        if self.att:  # (the padding after a segment, between segments or the chunk's last, as alone's: its positions
            # continue the segment's, its rows see the segment's keys)
            i = np.arange(T)
            seg = np.maximum.accumulate(np.r_[lo, np.full(T - n, -1)])  # (each token's segment's first)
            real = np.r_[~gap, np.zeros(T - n, bool)]
            last = np.zeros(T, np.int64)
            np.maximum.at(last, seg[real], i[real])
            pos = np.asarray(pos, np.int64)
            ext = np.r_[pos.reshape(n, -1), np.zeros((T - n, pos.size // n), np.int64)]  # ((T, 1) or (T, 3))
            lr = last[seg]
            pos_t = np.where(real[:, None], ext, ext[lr] + (i - lr)[:, None])  # (padding: on from its segment's last)
            rope = self.ov.rope[p0:p0 + T].copy()
            self.ov.rope[p0:p0 + T] = pos_t
            cs_row = self._cs_row(p0, T)
            sg = self.pattn[T].seg(seg, i if self.cfg.causal else last[seg], p0)
            dev.view("SEG")[:sg.size] = sg
            dev.flush("SEG", 0, sg.size)
        try:
            self._launch(T, p0, self.params(T, p0, n, cs_row, tree=tree), key=("tree" if tree else "pack", T))
        finally:
            if rope is not None:  # (the cached sequence's text positions)
                self.ov.rope[p0:p0 + T] = rope

    def packed_sizes(self, p0: int) -> tuple[int, ...]:
        """(after a prefix: the tree chunks')"""
        return self.tree_sizes if p0 else self.pack_sizes

    def seg_unit(self, p0: int) -> int:
        """(a tree chunk's segments start at scan chunks)"""
        return GP.C if p0 else 1

    def packed_seconds(self, T: int, p0: int) -> float:
        """(a chunk's: its attention streams every pass's groups as an unpacked one's)"""
        return self.chunk_seconds(T, p0)

    def _splits(self, T: int, p0: int, budget: float = PIECE_BUDGET_S) -> list[int]:
        """(an MoE layer: pieces from its shared expert and from its experts on: ``moe``)"""
        out = super()._splits(T, p0, budget)
        return sorted({*out, *(i for _, at in self._moe_marks.get(T, ()) for i in at)})

    def hidden(self, i: int, n: int = 1) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) of the last chunk's output residual (bf16 values, fp32)."""
        dev, H = self.ov.dev, self.H
        dev.fetch("PX", 2 * H * i, 2 * H * n)
        rows = dev.view("PX")[2 * H * i:2 * H * (i + n)].copy()
        return rows.view(bfloat16).astype(np.float32).reshape(n, H)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, p0: int, S: float) -> float:
        """Estimated time of an attention layer's flash attention over a chunk of S sub-chunks (a ``SMALL`` size: the
        fraction T / SUB of one) at p0 (the 1B / 9B / 27B: 0.144 / 0.488 / 0.636 ms per 1K cached positions and
        sub-chunk; non-causal: every pass streams the whole chunk, as a causal chunk's half way through it)."""
        at = self.attn[round(S * SUB)]
        if not at.causal:
            p0 += round(S * SUB) // 2
        passes = SUB / (32 * at.qg)  # (per sub-chunk)
        dma = passes * at.kvh * (AP.AT.REC_K + AP.AT.REC_V) / 8 / ATTN_BW
        cores = passes * at.R * (ATTN_CYC_TWO if at.two else ATTN_CYC_ONE) / 32 / CORE_HZ
        return max(dma, cores) * sum(min(1.0, S - j) * (p0 + SUB * j) for j in range(math.ceil(S)))

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        """(per sub-chunk the GEMMs by the bytes of their weights and their items, the norms by the hidden size, the GDN
        ops by the v heads, the attention prep by the q heads; the flash attention (``attention_seconds``); CFG_S per
        ConfigureOp)"""
        cfg = self.cfg
        gemms = [g for name in self._phases(L) for g in self.plans[(L, name)].gemms]
        wbytes = sum(self.img.region(L, name).nbytes for name in self._phases(L))
        if self.moe:  # (the router's, the shared expert's weights per sub-chunk; their items: one per chunk)
            wbytes += sum(self.img.region(L, name).nbytes for name in ("moe_in", "sdown"))
        Ss = S if S >= 1 else S + SMALL_S * (1 - S)
        s = Ss * (WEIGHT_S * wbytes + GEMM_S * len(gemms) + 2 * NORM_S * self.H) + (3 * GEMM_S if self.moe else 0)
        if cfg.attention[L]:
            s += Ss * PREP_S * cfg.num_attention_heads + self.attention_seconds(p0, S)
        else:
            s += Ss * GDN_S * cfg.ssm_v_heads
        if self.moe:
            s += self.moe.layer_seconds(L, round(S * SUB), weight_s=MOE_WEIGHT_S, row_s=MOE_ROW_S)
        if self.dp is not None and L == self.layers[-1]:  # (the drafter's records after the last layer)
            s += self.dp.seconds(Ss, weight_s=WEIGHT_S, gemm_s=GEMM_S, norm_s=NORM_S, prep_s=PREP_S, cfg_s=CFG_S)
        if self.fp is not None and L == self.layers[-1]:
            s += self.fp.seconds(Ss, weight_s=WEIGHT_S, gemm_s=GEMM_S, norm_s=NORM_S, prep_s=PREP_S, cfg_s=CFG_S)
        return s + CFG_S * self.layer_configs(L)

    def layer_configs(self, L: int) -> int:
        """(its GEMM objects, the GDN or attention ops, the norms; an MoE layer's block: 7)"""
        gemms = {id(g.op) for name in self._phases(L) for g in self.plans[(L, name)].gemms}
        return len(gemms) + (2 if self.cfg.attention[L] else 3) + 2 + (7 if self.moe else 0)
