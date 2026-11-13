"""A prompt chunk's rows of the DFlash2 drafter's cache (``dflash``) in the model's chunk prefill (``qwen35.prefill.QwenPrefill(dflash=)``):
the chunk leaves the records of its positions, in the ring (``dflash_core``), and the open rows of a record it does not fill. The rows
are the injection's (``dflash``), in GEMMs:

- at each tap layer's start (PX holds the model's residual there): x as bfp16 panels into columns i H .. of FT's rows of len(taps) H
  (``PrefillNormOperator(raw=True)``);
- after the last layer: fc in K parts of hidden (``prefill.gemm.plan_phase(k_part=)``, one per tap) into FY; the residual norm adds them to a
  zero x (FZ) and writes g normed by enc.output_norm as panels (FPN);
- per layer the K and V rows of g (``DFlashImage.ikv``, the same tiles as the block's) into FG, then ``DFlashPrepOperator``: K (RMS-normed,
  rotated) and V as the ring's records of the chunk's positions.

The ring's slots follow the positions; a chunk's writes run on past its end (``DFlashShape.over``, slices of a chunk are consecutive), and
the host moves what they wrote there to the ring's first slots (``after``: at most once per ring). The cos / sin of a chunk's positions (FCS)
are the host's, per chunk. Records of the padding tokens past a chunk's valid ones are masked until injections overwrite them. The open
rows, the exact K and V of a record the chunk's last token leaves incomplete, come from that record's panels (bf16 holds them exactly).

Buffers of the drafter's device by "F" + their name (``buffers``): its weight image FW, its cache FKV.
"""

from types import SimpleNamespace

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.operators import templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.bfp16 import block_values
from artlite.operators.chain import chained_fill
from artlite.operators.design import COLS, MAX_BD_ITER
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.util import Attr, arg_bytes
from artlite.prefill import gemm as PG
from artlite.prefill.norm import PrefillNormOperator

from .dflash import DFlashOverlay
from .dflash_core import HD, ND, PAIR, POS_PER_GROUP, REC_K, ROWS_B

QD_KV = "qd_kv"  # bf16 offset of the chunk's first object in a layer's ring

_PREP = templating.load("models/qwen35/dflash_prefill.cc.jinja")


class DFlashPrepOperator(ArtcOperator):
    """A sub-chunk's records (module docstring). Runtime sequence: G (the K and V rows, ``layout``), CS (per token [cos | sin] fp32), NRM
    (k_norm fp32, in one object of 8 tokens' CS), KV (a layer's rings, ``hb`` bytes per KV head). Per column a K core and a V core.
    Parameter: ``QD_KV`` (T's): the first object's bf16 offset in a ring."""

    STACK = 1536
    pfx = kernel_prefix("dfp")

    def __init__(self, *, T: int, layout: PG.Layout, hb: int, obj: int, eps: float):
        assert T % 32 == 0 and T // 8 <= MAX_BD_ITER and {"k", "v"} <= set(layout.off)
        self.T, self.layout, self.hb, self.obj, self.eps = T, layout, hb, obj, eps
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return {"nrm": 0, "row": 4 * HD, "stage": 8 * HD, "kf": 8 * HD + 4 * 64 * ND, "size": 8 * HD + 4 * 64 * ND + REC_K}

    def get_kernel_src(self) -> str:
        return _PREP.render(p=self.pfx, HD=HD, ND=ND, w=Attr(self.workspace()), eps=repr(self.eps))

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * self.layout.N,), bfloat16), ArgSpec("in", (T * HD * 2,), bfloat16),
                ArgSpec("in", (8 * HD * 2,), bfloat16), ArgSpec("out", (COLS * self.hb // 2,), bfloat16)]

    def params(self, slot: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk whose first object lies at ``slot`` of the rings."""
        return {f"{QD_KV}{self.T}": self.obj // 2 * slot}

    def get_mlir_artifact(self) -> Module:
        p, T, lay = self.pfx, self.T, self.layout
        N, pitch = lay.N, lay.pitch
        bf = np.dtype[bfloat16]
        kv_ty = np.ndarray[(8 * HD,), bf]
        in_ty = np.ndarray[(2 * 8 * HD,), bf]
        cs_ty = np.ndarray[(8 * 2 * HD,), bf]
        rec_ty = np.ndarray[(PAIR // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [kv_ty, cs_ty, ws_ty, rec_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [kv_ty, ws_ty, rec_ty, np.int32])
        p_kv = ScratchpadParameter(f"{QD_KV}{T}", np.int32)
        n_tg = T // 8
        a3 = [ObjectFifo(in_ty, name=f"dp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([0, 8 * HD], tile=Tile(c, 1), depths=[2, 2], obj_types=[kv_ty, kv_ty],
                                 names=[f"dp_ak{c}", f"dp_av{c}"]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"dp_b{c}", depth=2) for c in range(COLS)]
        ok = [ObjectFifo(rec_ty, name=f"dp_ok{c}", depth=2) for c in range(COLS)]
        ov = [ObjectFifo(rec_ty, name=f"dp_ov{c}", depth=2) for c in range(COLS)]

        def core_k(a_h, b_h, o_h, prm, k, ws):
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg // 2):
                    o = o_h.acquire(1)
                    for half in range(2):
                        x = a_h.acquire(1)
                        cs = b_h.acquire(1)
                        k(x, cs, ws, o, half)
                        b_h.release(1)
                        a_h.release(1)
                    o_h.release(1)

        def core_v(a_h, o_h, v, ws):
            for _ in forever_():
                for _ in range_(n_tg // 2):
                    o = o_h.acquire(1)
                    for half in range(2):
                        x = a_h.acquire(1)
                        v(x, ws, o, half)
                        a_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            wk, wv = Buffer(ws_ty, name=f"dp_wk{c}"), Buffer(ws_ty, name=f"dp_wv{c}")
            workers.append(Worker(core_k, [a1[c][0].cons(), b[c].cons(), ok[c].prod(), k_prm, k_k, wk], tile=Tile(c, 2),
                                  stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
            workers.append(Worker(core_v, [a1[c][1].cons(), ov[c].prod(), k_v, wv], tile=Tile(c, 3), stack_size=self.STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))
        n_kv, hb, o2 = COLS * self.hb // 2, self.hb // 2, self.obj // 2

        def seq(G, CS, NRM, KV, ah, bh, kh, vh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                taps = [TensorAccessPattern((T * N,), c * pitch + lay.off[kind], [n_tg, 1, 8, HD], [8 * N, 0, N, 1]) for kind in "kv"]
                chained_fill(ah[c], G, taps, tg)
                bh[c].fill(NRM, TensorAccessPattern((8 * 2 * HD,), 0, [1, 1, 8, 2 * HD], [0, 0, 2 * HD, 1]), group=tg)
                bh[c].fill(CS, TensorAccessPattern((T * 2 * HD,), 0, [1, 1, T, 2 * HD], [0, 0, 2 * HD, 1]), group=tg)
                for h, at in ((kh, 0), (vh, PAIR)):  # (the object's K or V part: rows of 576 elements)
                    h[c].drain(KV, TensorAccessPattern((n_kv,), c * hb + at, [1, T // 32, 4, 576], [0, o2, 576, 1]),
                               offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(T * 2 * HD,), bf], np.ndarray[(8 * 2 * HD,), bf],
                           np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(ok)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(ov)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


class DFlashPrefill:
    """The records of ``draft``'s cache for chunks of the model's ``prefill`` (its GEMM objects)."""

    def __init__(self, draft: DFlashOverlay, prefill: object):
        self.d, self.pf = draft, prefill
        cfg, img = draft.cfg, draft.image
        self.H, self.eps = cfg.hidden, cfg.rms_norm_eps
        self.n = len(cfg.taps)
        self.taps = {L: i for i, L in enumerate(cfg.taps)}
        ikv = SimpleNamespace(regions=list(img.ikv.values()))
        self.pitch = PG.phase_pitch(ikv, "ikv", prefill.SUB)
        self.fc_pitch = PG.d_pitch(SimpleNamespace(regions=[img.region(-2, "fc")]), prefill.SUB)
        assert self.fc_pitch == self.H // COLS, "fc's rows dense (the residual norm's deltas)"
        self.ops = {t: self._ops(t) for t in sorted({prefill._t(T) for T in prefill.sizes})}

    def _ops(self, t: int) -> SimpleNamespace:
        d, H = self.d, self.H
        img, eps, pf = d.image, self.eps, self.pf
        r = img.region(-2, "fc")
        fc = PG.plan_phase(img, r, t, pitch=self.fc_pitch, ops=pf._gemms, k_part=r.n_kc // self.n)
        assert fc.parts == self.n, "fc in a K part per tap"
        kv = {L: PG.plan_phase(img, img.ikv[L], t, pitch=self.pitch, ops=pf._gemms) for L in range(d.cfg.layers)}
        fp = max(arg_bytes(g.op, 2) for g in fc.gemms)
        prep = {p.layout.key: DFlashPrepOperator(T=t, layout=p.layout, hb=d.ring_bytes, obj=d.T, eps=eps) for p in kv.values()}
        return SimpleNamespace(fc=fc, kv=kv, f_part=fp, prep=prep,
                               tap=[PrefillNormOperator(T=t, add=False, E=H, eps=eps, raw=True, out_pitch=self.n * H, out_col=i * H)
                                    for i in range(self.n)],
                               nx=PrefillNormOperator(T=t, E=H, eps=eps, deltas=self.n, d_stride=fp // 2))

    def rows(self) -> dict[str, int]:
        """Bytes per token of the token buffers."""
        H = self.H
        return {"FT": 9 * self.n * H // 8, "FZ": 2 * H, "FX0": 2 * H, "FPN": 9 * H // 8, "FG": 2 * COLS * self.pitch, "FCS": 2 * 2 * HD}

    def buffers(self) -> dict[str, object]:
        """The drafter's device buffers by the names the items use."""
        return {"F" + k: v for k, v in self.d.dev.buffers.items() if k in ("W", "KV")}

    def external(self) -> set[str]:
        """(their names before the drafter is loaded)"""
        return {"FW", "FKV"}

    def _w(self, g: PG.Gemm) -> str:
        return f"FW[{g.w_off}:{g.w_off + g.op.w_bytes}]"

    def _ones(self) -> str:
        return f"FNR[0:{2 * self.H}]"

    def tap_items(self, L: int, T: int, tok: object) -> list[tuple]:
        """A chunk's items at layer L's start (PX: its input), when it is a tap's."""
        if L not in self.taps:
            return []
        t = self.pf._t(T)
        op = self.ops[t].tap[self.taps[L]]
        return [(op, tok("PX", j, op, i=0), self._ones(), tok("FT", j, op, i=2)) for j in range(T // t)]

    def end_items(self, T: int, tok: object, sl: object) -> list[tuple]:
        """A chunk's last items (module docstring)."""
        t = self.pf._t(T)
        S, o, d = T // t, self.ops[t], self.d
        fp, P = o.f_part, self.n
        rl: list[tuple] = [(g.op, tok("FT", j, g.op, i=0, extra=2 * g.a_off), self._w(g), sl("FY", (P * j + g.part) * fp, g.op, i=2))
                           for g in o.fc.gemms for j in range(S)]
        rl += [(o.nx, tok("FZ", j, o.nx, i=0), sl("FY", P * j * fp, o.nx, i=1), self._norm(0), tok("FX0", j, o.nx, i=3),
                tok("FPN", j, o.nx, i=4)) for j in range(S)]
        g_rb = self.rows()["FG"]
        kb = COLS * d.ring_bytes
        for L in range(d.cfg.layers):
            plan = o.kv[L]
            rl += [(g.op, tok("FPN", j, g.op, i=0), self._w(g), tok("FG", j, g.op, i=2, row=g_rb)) for g in plan.gemms for j in range(S)]
            prep = o.prep[plan.layout.key]
            rg, kv0 = t // POS_PER_GROUP * d.T, d.ring_off + L * kb
            rl += [(prep, tok("FG", j, prep, i=0, row=g_rb), sl("FCS", j * t * 2 * 2 * HD, prep, i=1), self._nrm(L),
                    f"FKV[{kv0 + j * rg}:{kv0 + j * rg + kb}]") for j in range(S)]
        return rl

    def _nrm(self, L: int) -> str:
        """Layer L's k_norm object in FNR: [ones H bf16 | per layer its object | enc.output_norm H bf16]."""
        o = 2 * self.H + L * 8 * 4 * HD
        return f"FNR[{o}:{o + 8 * 4 * HD}]"

    def _norm(self, k: int) -> str:
        o = 2 * self.H + self.d.cfg.layers * 8 * 4 * HD
        return f"FNR[{o}:{o + 2 * self.H}]"

    def static(self, dev: object) -> None:
        """FNR's weights."""
        d, n = self.d, 2 * self.H
        fnr = dev.view("FNR")
        fnr[:n] = np.ones(self.H, bfloat16).view(np.uint8)
        for L in range(d.cfg.layers):
            w = d._f32(f"blk.{L}.attn_k_norm.weight")
            fnr[n + L * 8 * 4 * HD:][:w.nbytes] = w.view(np.uint8)
        w = d._f32("enc.output_norm.weight").astype(bfloat16).view(np.uint8)
        fnr[n + d.cfg.layers * 8 * 4 * HD:][:w.size] = w
        dev.flush("FNR")

    def params(self, T: int, p0: int) -> dict[str, int]:
        """A chunk's scratchpad parameters: the rings' slot of its first position."""
        o = self.ops[self.pf._t(T)]
        return next(iter(o.prep.values())).params(p0 // POS_PER_GROUP % self.d.shape.ring)

    def before(self, dev: object, p0: int, T: int) -> None:
        """The cos / sin of a chunk's positions (FCS: [position][cos | sin] fp32)."""
        th = self.d.cfg.rope_theta(np.arange(p0, p0 + T)).astype(np.float64)
        cs = np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)
        dev.view("FCS")[:cs.nbytes] = cs.view(np.uint8).reshape(-1)
        dev.flush("FCS", 0, cs.nbytes)

    def after(self, p0: int, n: int, T: int) -> None:
        """After a chunk of n valid tokens of T at p0: what it wrote past the rings' end into their first slots, the open rows of a record
        its last token leaves incomplete."""
        d, dev, G, Tb = self.d, self.d.dev, self.d.shape.ring, self.d.T
        kv = dev.view("KV")
        slot, cnt = p0 // POS_PER_GROUP % G, T // POS_PER_GROUP
        heads = [(L, c) for L in range(d.cfg.layers) for c in range(COLS)]
        if slot + cnt > G:
            m = slot + cnt - G
            for L, c in heads:
                a = d._ring(L, c)
                dev.fetch("KV", a + G * Tb, m * Tb)
                kv[a:a + m * Tb] = kv[a + G * Tb:a + (G + m) * Tb]
                dev.flush("KV", a, m * Tb)
        end = p0 + n
        d.next = end
        r = end >> 3
        if not end & 7:
            return
        at = (r >> 2) % G * Tb + ((r & 3) >> 1) * PAIR + (r & 1) * 72
        for L, c in heads:
            a = d._ring(L, c)
            o = d._open(L, c)
            dev.fetch("KV", a + at, 2 * PAIR + 144 * ND)
            rows = np.zeros((2, ND, 8, 8), np.float32)
            for part in range(2):
                pan = np.stack([kv[a + at + part * 2 * PAIR + 144 * db:][:72] for db in range(ND)]).reshape(ND, 8, 9)
                v = block_values(pan)
                rows[part] = v if part == 0 else v.transpose(0, 2, 1)
            bits = rows.astype(bfloat16).view(np.uint8)
            kv[o + (r & 1) * ROWS_B:][:ROWS_B] = bits[0].reshape(-1)
            kv[o + (2 + (r & 1)) * ROWS_B:][:ROWS_B] = bits[1].reshape(-1)
            dev.flush("KV", o, 4 * ROWS_B)

    def seconds(self, Ss: float, *, weight_s: float, gemm_s: float, norm_s: float, prep_s: float, cfg_s: float) -> float:
        """Estimated time of a chunk's records (``QwenPrefill.layer_seconds``' constants; Ss sub-chunks)."""
        d = self.d
        wbytes = d.image.region(-2, "fc").nbytes + sum(d.image.ikv[L].nbytes for L in range(d.cfg.layers))
        return Ss * (weight_s * wbytes + (self.n + 2 * d.cfg.layers) * gemm_s + 2 * (self.n + 1) * norm_s * self.H + prep_s * COLS) + 12 * cfg_s


__all__ = ["DFlashPrefill", "DFlashPrepOperator"]
