"""A prompt chunk's records of an MTP drafter's cache (``qwen35.draft.QwenDraft``) in the model's chunk prefill
(``qwen35.prefill.QwenPrefill(draft=)``): the drafter's row at position q takes token q's embedding and the model's
output-normed hidden state of q - 1, and only its attention layer's K / V records outlive the prompt (the decode's
drafts read them), so a chunk computes eh_proj, attn_norm and the attention's input projection over its rows and
writes the records (``AttnPrepOperator``) into the drafter's KV cache at the chunk's positions:

- at the chunk's start (PX holds the embeddings): enorm(emb) as bfp16 panels into columns 0 .. H of PEH's rows of
  2 H (``PrefillNormOperator(out_pitch=, out_col=)``);
- after the last layer (PX holds its output x): hnorm(output_norm(x)) = (x w_out) w_h / rms(x w_out) into columns H ..
  (``PrefillNormOperator(pre=)``);
- eh_proj in two K parts (``prefill.gemm.plan_phase(k_part=)``: the enorm half, the hnorm half) into DY's two regions
  of R0 rows, the hnorm half's one row later: row q of the second region holds the projection of h_{q - 1} (its row
  0 the previous chunk's last valid row: ``carry``, zeros at position 0, after decode steps the host's: ``carry_of``);
  the residual norm adds both parts to a zero x (DZ) in fp32 and writes attn_norm(x0)'s panels (DPN);
- the drafter's attn_in GEMMs (DG), then the attention prep into the drafter's cache (its q / k norms in the prefill's
  PQN; its Q slots into QS, unused).

Buffers of the drafter's device by "D" + their name (``buffers``): its weight image DW, its KV cache DKV.
"""

from types import SimpleNamespace

import numpy as np
import torch

from ..design import COLS
from ..prefill import gemm as PG
from ..prefill.norm import PrefillNormOperator
from . import attn_prefill as AP
from .draft import QwenDraft


class DraftPrefill:
    """The records of ``draft``'s cache for chunks of ``sizes`` (sub-chunks of at most ``sub``) of the prefill of the
    model ``prefill`` (its GEMM objects, its prep operators)."""

    def __init__(self, draft: QwenDraft, prefill: object):
        self.d, self.pf = draft, prefill
        cfg, img, H = draft.cfg, draft.image, draft.H
        self.L, self.H, self.eps = draft.L, H, cfg.rms_norm_eps
        self.R0 = max(prefill.sizes) + 64  # (rows of each of DY's regions: a chunk, its GEMMs' padding, the carry)
        self.pitch = PG.phase_pitch(img, "attn_in", prefill.SUB)
        self.d_pitch = PG.d_pitch(img, prefill.SUB)
        assert self.d_pitch == H // 8, "eh's rows dense (the residual norm's deltas)"
        self.carry = np.zeros(2 * H, np.uint8)  # (DY's second region's row 0: the last chunk's last valid row)
        self._w_h: np.ndarray | None = None  # (eh_proj's hnorm half, fp32: ``carry_of``)
        self.ops = {t: self._ops(t) for t in sorted({prefill._t(T) for T in prefill.sizes})}

    def _ops(self, t: int) -> SimpleNamespace:
        img, H, eps = self.d.image, self.H, self.eps
        eh = PG.plan_phase(img, img.region(self.L, "eh"), t, pitch=self.d_pitch, ops=self.pf._gemms, k_part=2)
        assert eh.parts == 2, "eh_proj as [enorm half | hnorm half]"
        return SimpleNamespace(
            eh=eh, attn=PG.plan_phase(img, img.region(self.L, "attn_in"), t, pitch=self.pitch, ops=self.pf._gemms),
            ne=PrefillNormOperator(T=t, add=False, E=H, eps=eps, out_pitch=2 * H, out_col=0),
            nh=PrefillNormOperator(T=t, add=False, E=H, eps=eps, pre=True, out_pitch=2 * H, out_col=H),
            nx=PrefillNormOperator(T=t, E=H, eps=eps, deltas=2, d_stride=self.R0 * H))

    def rows(self) -> dict[str, int]:
        """Bytes per token of the token buffers."""
        H = self.H
        return {"PEH": 9 * 2 * H // 8, "DZ": 2 * H, "DX0": 2 * H, "DPN": 9 * H // 8, "DG": 2 * COLS * self.pitch}

    def buffers(self) -> dict[str, object]:
        """The drafter's device buffers by the names the items use."""
        return {"D" + k: v for k, v in self.d.dev.buffers.items()}

    def external(self) -> set[str]:
        """(their names before the drafter is loaded)"""
        return {"D" + k for k in ("W", *self.d.sizes)}

    @staticmethod
    def _w(g: PG.Gemm) -> str:
        return f"DW[{g.w_off}:{g.w_off + g.op.w_bytes}]"

    def start_items(self, T: int, tok: object) -> list[tuple]:
        """A chunk's first items (PX: the embeddings): enorm(emb) into PEH. ``tok(buf, j, op, i=, row=, extra=)``:
        the prefill's sub-chunk slices."""
        t = self.pf._t(T)
        ne = self.ops[t].ne
        return [(ne, tok("PX", j, ne, i=0), self._norm(0), tok("PEH", j, ne, i=2)) for j in range(T // t)]

    def end_items(self, T: int, tok: object, sl: object) -> list[tuple]:
        """A chunk's last items (PX: the last layer's output): the records (module docstring)."""
        t = self.pf._t(T)
        S, o, H = T // t, self.ops[t], self.H
        row = 2 * H
        rl: list[tuple] = [(o.nh, tok("PX", j, o.nh, i=0), self._norm(1, 2), tok("PEH", j, o.nh, i=2))
                           for j in range(S)]
        for g in o.eh.gemms:  # (part 1's rows one later: row q the projection of h_{q - 1})
            base = (self.R0 + 1) * row if g.part else 0
            rl += [(g.op, tok("PEH", j, g.op, i=0, extra=2 * g.a_off), self._w(g), sl("DY", base + j * t * row, g.op, i=2))
                   for j in range(S)]
        rl += [(o.nx, tok("DZ", j, o.nx, i=0), sl("DY", j * t * row, o.nx, i=1), self._norm(3), tok("DX0", j, o.nx, i=3),
                tok("DPN", j, o.nx, i=4)) for j in range(S)]
        (g,) = o.attn.gemms
        g_rb = self.rows()["DG"]
        rl += [(g.op, tok("DPN", j, g.op, i=0), self._w(g), tok("DG", j, g.op, i=2, row=g_rb)) for j in range(S)]
        prep = self.pf._prep(o.attn.layout, T)
        d, L = self.d, self.L
        kv0, hb = d.kv_off[L], d.ash.kvh * d.ash.head_bytes
        qn = f"PQN[{self.pf.qn_off[L]}:{self.pf.qn_off[L] + 4 * 2 * AP.HD}]"
        qs_g = COLS * 2 * prep.qc * AP.SLOT  # (bytes of a token group's Q slots)
        grp = t // 32 * AP.GROUP  # (a sub-chunk's cache groups)
        rl += [(prep, tok("DG", j, prep, i=0, row=g_rb), sl("PCS", j * t * 256, prep, i=1), qn,
                sl("QS", j * t // 8 * qs_g, prep, i=3), f"DKV[{kv0 + j * grp}:{kv0 + j * grp + hb}]") for j in range(S)]
        return rl

    def seconds(self, Ss: float, *, weight_s: float, gemm_s: float, norm_s: float, prep_s: float, cfg_s: float) -> float:
        """Estimated time of a chunk's records (``QwenPrefill.layer_seconds``' constants; Ss sub-chunks)."""
        img = self.d.image
        wbytes = sum(img.region(self.L, name).nbytes for name in ("eh", "attn_in"))
        heads = self.d.cfg.num_attention_heads
        return Ss * (weight_s * wbytes + 3 * gemm_s + 4 * norm_s * self.H + prep_s * heads) + 6 * cfg_s

    def _norm(self, k: int, n: int = 1) -> str:
        """Slots k .. k + n - 1 of PDN: [enorm | the model's output_norm | hnorm | attn_norm] (H bf16 each)."""
        return f"PDN[{2 * self.H * k}:{2 * self.H * (k + n)}]"

    def static(self, dev: object, ov: object) -> None:
        """PDN's weights and the drafter's q / k norms in PQN (``QwenPrefill._bind``)."""
        d, b = self.d, f"blk.{self.L}."
        w = [d._f32(b + "nextn.enorm.weight"), ov._f32("output_norm.weight"), d._f32(b + "nextn.hnorm.weight"),
             d._f32(b + "attn_norm.weight")]
        pdn = torch.from_numpy(np.concatenate(w)).bfloat16().view(torch.uint8).numpy()
        dev.view("PDN")[:pdn.size] = pdn
        dev.flush("PDN")
        o = self.pf.qn_off[self.L]
        qn = np.concatenate([d._f32(b + "attn_q_norm.weight"), d._f32(b + "attn_k_norm.weight")])
        dev.view("PQN")[o:o + qn.nbytes] = qn.view(np.uint8)
        dev.flush("PQN")

    def before(self, dev: object, p0: int) -> None:
        """The carry (row 0 of DY's second region) of a chunk at p0: the last chunk's, zeros at 0."""
        if p0 == 0:
            self.carry[:] = 0
        o = 2 * self.H * self.R0
        dev.view("DY")[o:o + self.carry.size] = self.carry
        dev.flush("DY", o, self.carry.size)

    def carry_of(self, h: torch.Tensor) -> None:
        """The carry of a chunk after decode steps (not after a chunk): eh_proj's hnorm half of hnorm(h), h (hidden,) the
        model's output-normed hidden state of the position before the chunk."""
        if self._w_h is None:
            g = self.d.gguf
            w = g.tensor(f"blk.{self.L}.nextn.eh_proj.weight").dequantize().numpy().reshape(self.H, 2 * self.H)
            self._w_h = np.ascontiguousarray(w[:, self.H:])
        x = h.float().numpy().astype(np.float32)
        x = x / np.sqrt((x * x).mean(dtype=np.float32) + np.float32(self.eps)) * self.d._f32(
            f"blk.{self.L}.nextn.hnorm.weight")
        y = torch.from_numpy(self._w_h @ x).bfloat16()
        self.carry[:] = y.view(torch.uint8).numpy()

    def after(self, dev: object, n: int) -> None:
        """The next chunk's carry: this one's projection of its last valid row's h; the drafter's cache records of the
        chunk's last record are its (record mode)."""
        o = 2 * self.H * (self.R0 + n)
        dev.fetch("DY", o, self.carry.size)
        self.carry[:] = dev.view("DY")[o:o + self.carry.size]
        self.d.vring.from_prefill.add(self.L)
