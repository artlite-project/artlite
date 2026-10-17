"""SD 1.5's UNet step as one library program (``UNetProgram``) over one device buffer (``Arena``): the 64x64 / 32x32
levels by overlay operators, the 16x16 / 8x8 levels by ``sd15_v2``'s section in one raw item.

Each operator shape is one library entry (``UNetProgram.get``); a program item is an entry with arena addresses. The
overlay design and the section's design cost two reconfigurations per step; the transformer blocks run in the
overlay's transformer variant, entered and left by program-memory patches. The section takes down_blocks.1's
downsampler conv's half tiles (``v2_in``) and writes the panel objects up_blocks.1's upsampler conv reads (``v2_out``).

Data flow (overlay.py's layouts):
- Sums carry the residuals: every tensor buffer has room for summands after it, and a residual sum's stack is
  [x | conv / GEMM chunks] in x's buffer. Chunks: convs over ci > 640 and GEMMs over K > 640 run in K chunks, summed
  by the consumer's sum (bf16 C tiles hold partial sums of at most 640 products' worth of rounding).
- Skips in place: concatenation buffers [h | s | room]. The skip's producing sum writes s there with a second GroupNorm
  record set (the concatenation's groups, h's pass offset); h's producer writes before it; the up block's resnet reads
  [h | s].
- Stride-2 convs write half tiles of the lower level, which its sum reads (``SumOp(half_in)``); upsampling convs read
  the lower level's panel objects (``ConvL(up)``).
- conv_in reads the latents in the conv layout (4 channels padded to 32); conv_out writes 80 channels (4 real).
- Per step the host writes the latents and the 10 overlay resnets' time biases (their first sums' weight images); per
  prompt the cross attentions' K / V (``kv_packs``). The caller writes the section's (``sd15_v2.unet.Section``: its
  resnets' time biases, its context K / V^T).

Arena: the weights (~0.23 GB of bfp16 images), then the activations, whose space is reused after their last item (first
fit by item order, ~0.17 GB).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import artlite
from artlite import txn
from artlite.operators.fused import Library
from artlite.operators.fused.program import RawTxn
from artlite.operators.sd15 import overlay as ov
from artlite.operators.sd15.attention import AttnOverlay, attn_block
from artlite.operators.sd15.overlay import C_OBJ, SOBJ, ApplyOp, ConvL, GemmL, SumOp
from artlite.operators.sd15_v2.design import V2Design

C2 = C_OBJ // 2  # (bf16 of an object)
SLACK = (SOBJ - C_OBJ) // 2  # (bf16 after a staged tensor: its last objects' A objects read past it)
LEVELS = (64, 32)  # (the overlay's; 16x16 / 8x8: the section)
ALIGN = 2048  # (bf16: arena offsets 4 KB aligned)


def tt(W: int) -> int:
    """Tiles of a level (both images)."""
    return 2 * W * W // 128


def nops(W: int) -> int:
    """Operators per logical operator (64x64: the top and bottom halves)."""
    return 2 if W == 64 else 1


def ysz(C: int, W: int) -> int:
    """bf16 elements of a C-tile tensor."""
    return C // 80 * tt(W) * C2


def chunk80(K: int, cap: int = 640) -> int:
    """The largest multiple of 80 dividing K, at most cap."""
    return max(c for c in range(80, cap + 1, 80) if K % c == 0)


@dataclass
class Act:
    """A tensor in the arena: C-tile objects at buf + off (bf16 elements), C channels at level W; its GroupNorm /
    LayerNorm records: ``sums`` [(sum op, record set)] in the ST buffer ``st``."""

    buf: str
    off: int
    C: int
    W: int
    st: str | None = None
    sums: list = field(default_factory=list)


@dataclass
class Dest:
    """Where a block's last sum writes (buf + off) and its records: groups of C (set 0), ``stats2`` (C2, pass offset)
    or None, into ST buffer ``st`` from object ``st_off`` (``n_st`` operators' records in it)."""

    buf: str
    off: int
    C: int
    st: str
    st_off: int
    n_st: int
    stats2: tuple | None = None


class Arena:
    """One device buffer holding a program's tensors at fixed bf16 offsets (``names``: name -> (offset, size))."""

    def __init__(self, dev: artlite.Device, names: dict[str, tuple[int, int]], n: int,
                 data: list[tuple[str, np.ndarray]]):
        host = torch.zeros(n, dtype=torch.bfloat16)
        for name, img in data:
            o = names[name][0]
            host[o:o + img.size // 2] = torch.from_numpy(np.ascontiguousarray(img)).view(torch.bfloat16)
        self.names = names
        self.t = artlite.from_torch(host, dev)
        self.base = txn.device_address(self.t)
        self.view = torch.from_numpy(self.t._raw.map()).view(torch.bfloat16)  # (host mapping: sync the ranges used)

    def addr(self, name: str, at: int = 0) -> int:
        return self.base + 2 * (self.names[name][0] + at)

    def put(self, name: str, data: torch.Tensor, at: int = 0) -> None:
        """Write bf16 ``data`` at element ``at`` of tensor ``name`` (and sync that range to the device)."""
        off = self.names[name][0] + at
        self.view[off:off + data.numel()] = data.reshape(-1)
        self.t._raw.sync_to_device(2 * data.numel(), 2 * off)

    def get(self, name: str, n: int | None = None, at: int = 0) -> torch.Tensor:
        """n bf16 elements (all: the tensor) from element ``at`` of tensor ``name``."""
        off, size = self.names[name]
        n = size - at if n is None else n
        self.t._raw.sync_from_device(2 * n, 2 * (off + at))
        return self.view[off + at:off + at + n].clone()


class Program:
    """A library program: its entries (operators, one per shape) and items (an entry with arena tensors and element
    offsets as its arguments)."""

    def __init__(self, name: str):
        self.name = name
        self.ops: list = []
        self.items: list[tuple[int, list[tuple[str, int]]]] = []
        self._entry: dict[int, int] = {}
        self.raw: dict[int, RawTxn] = {}  # (item index -> its host-built transaction: ``add_raw``)

    def add(self, op: object, *args: str | tuple[str, int]) -> None:
        e = self._entry.setdefault(id(op), len(self.ops))
        if e == len(self.ops):
            self.ops.append(op)
        assert len(args) == len(op.get_arg_spec()), (type(op).__name__, len(args))
        self.items.append((e, [a if isinstance(a, tuple) else (a, 0) for a in args]))

    def add_raw(self, op: object, raw: RawTxn, *touches: str) -> None:
        """A host-built transaction on op's design (``fused.program.RawTxn``); touches: the arena tensors it reads or
        writes (their liveness: the arena's layout)."""
        e = self._entry.setdefault(id(op), len(self.ops))
        if e == len(self.ops):
            self.ops.append(op)
        self.raw[len(self.items)] = raw
        self.items.append((e, [(t, 0) for t in touches]))

    def compile(self) -> None:
        self.lib = Library(self.name, self.ops).compile()  # (entries: self.ops in order)
        self.lib.load()

    def check_sizes(self, sizes: dict[str, int]) -> None:
        """Every argument range lies inside its tensor."""
        for i, (e, args) in enumerate(self.items):
            if i in self.raw:
                continue
            for (n, at), spec in zip(args, self.ops[e].get_arg_spec(), strict=True):
                assert at + spec.shape[0] <= sizes[n], (type(self.ops[e]).__name__, n, at, spec.shape[0], sizes[n])

    def run(self, arena: Arena, bufs: list = ()) -> None:
        """bufs: other BOs the items address (a raw transaction's)."""
        items = [(e, self.raw[i] if i in self.raw else [arena.addr(n, at) for n, at in args])
                 for i, (e, args) in enumerate(self.items)]
        self.lib.run(items, bufs=[arena.t._raw, *bufs])


class UNetProgram:
    """The UNet step (``build``) as one library program on the diffusers weights ``w`` (fp32 tensors by name): overlay
    operators, and between down_blocks.1's downsampler conv and up_blocks.1's upsampler conv the 16x16 / 8x8 levels by
    ``sd15_v2.unet.Section``: a raw transaction on its design ``v2_design`` (``v2_raw``, its words the caller's once the
    arena is laid out; its input ``v2_in``, output ``v2_out``)."""

    # (level, channels) of skip i, the channels of h in its concatenation (skips 6 .. 11: the section's)
    SKIP = ((64, 320), (64, 320), (64, 320), (32, 320), (32, 640), (32, 640))
    HCH = (320, 320, 640, 640, 640, 1280)

    def __init__(self, w: dict[str, torch.Tensor]):
        self.w = w
        self.v2_design = self.v2_raw = self.v2_in = self.v2_out = None
        self.ov = {W: AttnOverlay(W=W, conv=True) for W in LEVELS}
        self.ovt = {W: AttnOverlay(W=W, conv=False) for W in LEVELS}
        self.t = False  # (operators built now: in the transformer variant)
        self.prog = Program("sd15_unet")
        self.specs: dict[str, int] = {}
        self.data: list[tuple[str, np.ndarray]] = []
        self.ops: dict = {}
        self.temb_imgs: list[tuple[str, str, SumOp]] = []  # (image, resnet prefix, sum op): the per-step time biases
        self.cat_sums: dict[int, dict[str, list]] = {}
        self.outs: list[tuple[str, Act]] = []  # (block prefix, output)
        self.kv_packs: list = []  # (arena name, fn(context) -> bf16 tensor): per-prompt buffers (cross attention K / V)
        self.alloc("zeros", 65536)
        self.alloc("st_none", C2)

    # -- arena and operators
    def alloc(self, name: str, n: int) -> str:
        assert name not in self.specs, name
        self.specs[name] = n
        return name

    def tensor(self, name: str, C: int, W: int, room: int = 2) -> str:
        """A C-tile tensor with room for ``room`` summands after it."""
        return self.alloc(name, (1 + room) * ysz(C, W) + SLACK)

    def wimg(self, name: str, img: np.ndarray) -> str:
        self.alloc(name, img.size // 2)
        self.data.append((name, img))
        return name

    def get(self, cls: type, level: int, /, **kw: object) -> object:
        """The operator cls(the level's overlay, **kw), one object per distinct shape (one library entry)."""
        def h(v: object) -> object:
            return tuple((id(o), s) for o, s in v) if isinstance(v, list) else v
        key = (cls.__name__, level, self.t, tuple(sorted((k, h(v)) for k, v in kw.items())))
        if key not in self.ops:
            self.ops[key] = cls((self.ovt if self.t else self.ov)[level], **kw)
        return self.ops[key]

    def add(self, op: object, *args: str | tuple[str, int]) -> None:
        self.prog.add(op, *args)

    @staticmethod
    def tk(W: int, k: int) -> dict:
        """Tile range keywords of op k of a level's logical operator."""
        T = tt(W) // nops(W)
        return {"T": T, "TT": tt(W), "t0": k * T}

    @staticmethod
    def ck(W: int, k: int) -> dict:
        """Row range keywords of conv k of a level's logical conv."""
        Hp = W // nops(W)
        return {"Hp": Hp, "Hp_total": W, "y0": k * Hp}

    # -- destinations
    def cat(self, i: int) -> tuple[str, str]:
        """Skip i's concatenation buffer [h | s | room] and its ST buffer (s's sums at objects 0.., h's after)."""
        W, Cs = self.SKIP[i]
        name = f"cat{i}"
        if name not in self.specs:
            self.alloc(name, ysz(self.HCH[i], W) + 3 * ysz(Cs, W) + SLACK)
            self.alloc(name + ".st", 2 * nops(W) * 32 * C2)
        return name, name + ".st"

    def skip_dest(self, i: int) -> Dest:
        W, Cs = self.SKIP[i]
        buf, st = self.cat(i)
        Ch = self.HCH[i]
        return Dest(buf, ysz(Ch, W), Cs, st, 0, 2 * nops(W), stats2=(Ch + Cs, Ch // 80))

    def h_dest(self, i: int) -> Dest:
        W, Cs = self.SKIP[i]
        buf, st = self.cat(i)
        return Dest(buf, 0, self.HCH[i] + Cs, st, nops(W) * 32, 2 * nops(W))

    def new_dest(self, name: str, C: int, W: int, room: int = 2) -> Dest:
        buf = self.tensor(name, C, W, room)
        return Dest(buf, 0, C, self.alloc(name + ".st", nops(W) * 32 * C2), 0, nops(W))

    def cat_act(self, i: int) -> Act:
        W, Cs = self.SKIP[i]
        buf, st = self.cat(i)
        cs = self.cat_sums[i]
        return Act(buf, 0, self.HCH[i] + Cs, W, st, [(o, 0) for o in cs["h"]] + [(o, 1) for o in cs["s"]])

    # -- operators
    def final_sum(self, W: int, P: int, S: int, stack: tuple[str, int], dest: Dest, name: str, *,
                  half_in: bool = False, stats: str | None = "gn") -> Act:
        """The S summands at stack -> dest (with its GroupNorm records; stats "ln": LayerNorm records, None: none)."""
        ops = []
        for k in range(nops(W)):
            s = self.get(SumOp, W, P=P, S=S, stats=stats, C=dest.C, stats2=dest.stats2,
                         st_off=dest.st_off + 32 * k, n_st=dest.n_st, half_in=half_in, **self.tk(W, k))
            self.add(s, stack, self.wimg(f"{name}.{k}", s.pack(None)), (dest.buf, dest.off), *([dest.st] if stats else []))
            ops.append(s)
        return Act(dest.buf, dest.off, 80 * P, W, dest.st, [(o, 0) for o in ops])

    def plain(self, name: str, x: Act) -> str:
        """x -> bfp16 panel objects (no normalization)."""
        W = x.W
        pl = self.alloc(name, ysz(x.C, W) + SLACK)
        for k in range(nops(W)):
            a = self.get(ApplyOp, W, P=x.C // 80, sums=[], mode="plain", **self.tk(W, k))
            self.add(a, (x.buf, x.off), "st_none", self.wimg(f"{name}.w{k}", a.pack()), pl)
        return pl

    def norm(self, name: str, x: Act, gamma: torch.Tensor, beta: torch.Tensor, *, silu: bool = True,
             eps: float = 1e-5, mode: str = "gn") -> str:
        """GroupNorm (+ SiLU; mode "ln": LayerNorm) of x -> bfp16 panel objects."""
        W = x.W
        out = self.alloc(name, ysz(x.C, W) + SLACK)
        for k in range(nops(W)):
            a = self.get(ApplyOp, W, P=x.C // 80, sums=x.sums, silu=silu, eps=eps, mode=mode, **self.tk(W, k))
            self.add(a, (x.buf, x.off), x.st, self.wimg(f"{name}.w{k}", a.pack(gamma, beta)), out)
        return out

    def conv(self, name: str, W: int, xp: str, *, ci: int, co: int, wt: torch.Tensor, b: torch.Tensor,
             out: tuple[str, int], up: bool = False) -> int:
        """conv3x3 of panels xp (ci channels) in ci chunks -> summands at out (consecutive): their count."""
        cic = ConvL.max_chunk(self.ov[W], W // nops(W), ci, co)
        for j in range(ci // cic):
            for k in range(nops(W)):
                c = self.get(ConvL, W, ci=ci, co=co, ci0=j * cic, cic=cic, up=up, **self.ck(W, k))
                self.add(c, xp, "zeros", self.wimg(f"{name}.{j}.{k}", c.pack(wt[:, j * cic:(j + 1) * cic],
                                                                           b if j == 0 else None)),
                         (out[0], out[1] + j * ysz(co, W)))
        return ci // cic

    def gemm(self, name: str, W: int, xp: str, *, K: int, N: int, wt: torch.Tensor, b: torch.Tensor | None,
             out: tuple[str, int], cin: int = 80, geglu: bool = False, cap: int = 640) -> int:
        """xp (panel objects of cin channels) W^T + b in K chunks (at most cap) -> summands at out: their count."""
        kc = max(c for c in range(cin, cap + 1, cin) if K % c == 0 and c % 32 == 0)
        for j in range(K // kc):
            for k in range(nops(W)):
                g = self.get(GemmL, W, K=K, N=N, k0=j * kc, kc=kc, cin=cin, geglu=geglu, **self.tk(W, k))
                self.add(g, xp, self.wimg(f"{name}.{j}.{k}", g.pack(wt[:, j * kc:(j + 1) * kc], b if j == 0 else None)),
                         (out[0], out[1] + j * ysz(N, W)))
        return K // kc

    # -- blocks
    def transformer(self, p: str, x: Act, dest: Dest) -> Act:
        """GroupNorm and proj_in (the regular variant, then the transformer variant), (LayerNorm, self attention),
        (LayerNorm, cross attention), (LayerNorm, GEGLU feed-forward), proj_out, the residual sum (the regular
        variant: GroupNorm records for the next resnet). Residuals: summand 0 of the sums, each sum's other summands in
        the stack after it."""
        w, W, C = self.w, x.W, x.C
        y = ysz(C, W)
        gp = self.norm(p + "gn", x, w[p + "norm.weight"], w[p + "norm.bias"], silu=False, eps=1e-6)
        self.t = True
        st = self.alloc(p + "pi.stack", 2 * y + SLACK)
        n = self.gemm(p + "pi", W, gp, K=C, N=C, wt=w[p + "proj_in.weight"].reshape(C, C), b=w[p + "proj_in.bias"],
                      out=(st, 0))
        h = self.final_sum(W, C // 80, n, (st, 0), self.new_dest(p + "h0", C, W), p + "pi.s", stats="ln")
        q = p + "transformer_blocks.0."
        nf = 4 * C // max(c for c in range(40, 641, 40) if 4 * C % c == 0 and c % 32 == 0)  # (ff2's K chunks)
        for i, a in ((1, "attn1."), (2, "attn2.")):
            nl = self.norm(f"{p}n{i}", h, w[f"{q}norm{i}.weight"], w[f"{q}norm{i}.bias"], mode="ln")
            n = attn_block(self, q + a, W, C, nl, (h.buf, h.off + y), cross=i == 2)
            h = self.final_sum(W, C // 80, 1 + n, (h.buf, h.off), self.new_dest(f"{p}h{i}", C, W, nf if i == 2 else 2),
                               f"{p}s{i}", stats="ln")
        nl = self.norm(p + "n3", h, w[q + "norm3.weight"], w[q + "norm3.bias"], mode="ln")
        gb = self.alloc(p + "ff1", ysz(8 * C, W) + SLACK)
        self.gemm(p + "ff1", W, nl, K=C, N=8 * C, wt=w[q + "ff.net.0.proj.weight"], b=w[q + "ff.net.0.proj.bias"],
                  out=(gb, 0), geglu=True, cap=1280)  # (GELU of the whole sum: one chunk)
        self.gemm(p + "ff2", W, gb, K=4 * C, N=C, wt=w[q + "ff.net.2.weight"], b=w[q + "ff.net.2.bias"],
                  out=(h.buf, h.off + y), cin=40)  # (into h's buffer, which has room for ff2's chunks)
        h = self.final_sum(W, C // 80, 1 + nf, (h.buf, h.off), self.new_dest(p + "h3", C, W), p + "s3", stats=None)
        pl = self.plain(p + "pl", h)
        n = self.gemm(p + "po", W, pl, K=C, N=C, wt=w[p + "proj_out.weight"].reshape(C, C), b=w[p + "proj_out.bias"],
                      out=(x.buf, x.off + y))
        self.t = False
        return self.final_sum(W, C // 80, 1 + n, (x.buf, x.off), dest, p + "s")

    def resnet(self, p: str, x: Act, co: int, dest: Dest) -> Act:
        """GroupNorm + SiLU, conv1, + the time bias (its sum's weight image: per step), GroupNorm + SiLU, conv2, + the
        residual (or the 1x1 shortcut of x)."""
        w, W, ci = self.w, x.W, x.C
        p1 = self.norm(p + "p1", x, w[p + "norm1.weight"], w[p + "norm1.bias"])
        h1 = self.alloc(p + "h1", ci // ConvL.max_chunk(self.ov[W], W // nops(W), ci, co) * ysz(co, W) + SLACK)
        n1 = self.conv(p + "c1", W, p1, ci=ci, co=co, wt=w[p + "conv1.weight"], b=w[p + "conv1.bias"], out=(h1, 0))
        h = self.final_sum(W, co // 80, n1, (h1, 0), self.new_dest(p + "h", co, W), p + "s1")
        for k in range(nops(W)):
            self.temb_imgs.append((f"{p}s1.{k}", p, h.sums[k][0]))
        p2 = self.norm(p + "p2", h, w[p + "norm2.weight"], w[p + "norm2.bias"])
        if ci == co:  # (the residual: summand 0, the conv chunks after it in x's buffer)
            n2 = self.conv(p + "c2", W, p2, ci=co, co=co, wt=w[p + "conv2.weight"], b=w[p + "conv2.bias"],
                           out=(x.buf, x.off + ysz(co, W)))
            assert n2 <= 2 and x.off + 3 * ysz(co, W) <= self.specs[x.buf]
            return self.final_sum(W, co // 80, 1 + n2, (x.buf, x.off), dest, p + "s2")
        kc = chunk80(ci)
        nsc = ci // kc
        s2 = self.alloc(p + "s2buf", (co // ConvL.max_chunk(self.ov[W], W // nops(W), co, co) + nsc) * ysz(co, W) + SLACK)
        n2 = self.conv(p + "c2", W, p2, ci=co, co=co, wt=w[p + "conv2.weight"], b=w[p + "conv2.bias"], out=(s2, 0))
        pl = self.plain(p + "pl", x)
        wsc = w[p + "conv_shortcut.weight"].reshape(co, ci)
        for j in range(nsc):
            for k in range(nops(W)):
                g = self.get(GemmL, W, K=ci, N=co, k0=j * kc, kc=kc, **self.tk(W, k))
                self.add(g, pl, self.wimg(f"{p}sc.{j}.{k}", g.pack(wsc[:, j * kc:(j + 1) * kc],
                                                                   w[p + "conv_shortcut.bias"] if j == 0 else None)),
                         (s2, (n2 + j) * ysz(co, W)))
        return self.final_sum(W, co // 80, n2 + nsc, (s2, 0), dest, p + "s2")

    def down_stack(self, p: str, x: Act) -> str:
        """down()'s conv without its sum: the stride-2 conv's half tiles in one summand (the section's input)."""
        w, W, C = self.w, x.W, x.C
        pl = self.plain(p + "pl", x)
        assert ConvL.max_chunk(self.ov[W], W // 2, C, C, stride=2) == C, "one summand"
        st = self.alloc(p + "stack", 2 * ysz(C, W // 2) + SLACK)
        c = self.get(ConvL, W, Hp=W // 2, ci=C, co=C, ci0=0, cic=C, stride=2)
        self.add(c, pl, "zeros", self.wimg(f"{p}c.0", c.pack(w[p + "weight"], w[p + "bias"])), (st, 0))
        return st

    def v2_section(self) -> None:
        """The section's raw item, then up_blocks.1's upsampler conv from its panels (``v2_out``)."""
        p = "up_blocks.1.upsamplers.0.conv."
        self.v2_out = self.alloc(p + "pl", ysz(1280, 16) + SLACK)
        self.v2_design = V2Design()
        self.v2_raw = RawTxn(np.zeros(4, np.uint32))
        self.prog.add_raw(self.v2_design, self.v2_raw, self.v2_in, self.v2_out)
        st = self.alloc(p + "stack", 2 * ysz(1280, 32) + SLACK)
        n = self.conv(p + "c", 32, self.v2_out, ci=1280, co=1280, wt=self.w[p + "weight"], b=self.w[p + "bias"],
                      out=(st, 0), up=True)
        h = self.final_sum(32, 16, n, (st, 0), self.h_dest(5), p + "s")
        self.outs.append((p, h))
        self.cat_sums[5]["h"] = [o for o, _ in h.sums]

    def down(self, p: str, x: Act, dest: Dest) -> Act:
        """conv3x3 stride 2 at x's level -> level W / 2 (half tiles, summed there)."""
        w, W, C = self.w, x.W, x.C
        pl = self.plain(p + "pl", x)
        cic = ConvL.max_chunk(self.ov[W], W // 2, C, C, stride=2)
        n = C // cic
        st = self.alloc(p + "stack", n * 2 * ysz(C, W // 2) + SLACK)
        for j in range(n):
            c = self.get(ConvL, W, Hp=W // 2, ci=C, co=C, ci0=j * cic, cic=cic, stride=2)
            assert c.y_elems == 2 * ysz(C, W // 2)
            self.add(c, pl, "zeros", self.wimg(f"{p}c.{j}", c.pack(w[p + "weight"][:, j * cic:(j + 1) * cic],
                                                                  w[p + "bias"] if j == 0 else None)), (st, j * c.y_elems))
        return self.final_sum(W // 2, C // 80, n, (st, 0), dest, p + "s", half_in=True)

    def up(self, p: str, x: Act, dest: Dest) -> Act:
        """nearest 2x + conv3x3: the conv at level 2 W reads x's panels."""
        w, W, C = self.w, 2 * x.W, x.C
        pl = self.plain(p + "pl", x)
        st = self.alloc(p + "stack", 2 * ysz(C, W) + SLACK)
        n = self.conv(p + "c", W, pl, ci=C, co=C, wt=w[p + "weight"], b=w[p + "bias"], out=(st, 0), up=True)
        return self.final_sum(W, C // 80, n, (st, 0), dest, p + "s")

    def build(self) -> None:
        """The step's items: conv_in, down_blocks.0 / 1, the section (``v2_section``), up_blocks.2 / 3, the output
        conv; the time biases are zero until written (``time_bias_images``)."""
        w = self.w
        self.alloc("lat", 66 * 8 * 4 * 144 // 2)  # (the latents in the conv layout: 66 rows x 8 groups x 4 cb x 144 B)
        win = torch.zeros(320, 32, 3, 3)
        win[:, :4] = w["conv_in.weight"]
        st = self.alloc("cin.stack", ysz(320, 64) + SLACK)
        for k in range(2):
            c = self.get(ConvL, 64, ci=32, co=320, panels=False, **self.ck(64, k))
            self.add(c, "lat", self.wimg(f"cin.{k}", c.pack(win, w["conv_in.bias"])), (st, 0))
        h = self.final_sum(64, 4, 1, (st, 0), self.skip_dest(0), "cin.s")
        self.cat_sums[0] = {"s": [o for o, _ in h.sums]}
        for i, C in enumerate((320, 640)):
            for j in range(2):
                p = f"down_blocks.{i}.resnets.{j}."
                d = self.skip_dest(len(self.cat_sums))
                h = self.resnet(p, h, C, self.new_dest(p + "out", C, h.W))
                self.outs.append((p, h))
                p = f"down_blocks.{i}.attentions.{j}."
                h = self.transformer(p, h, d)
                self.outs.append((p, h))
                self.cat_sums[len(self.cat_sums)] = {"s": [o for o, _ in h.sums]}
            if i == 0:
                p = "down_blocks.0.downsamplers.0.conv."
                h = self.down(p, h, self.skip_dest(3))
                self.outs.append((p, h))
                self.cat_sums[3] = {"s": [o for o, _ in h.sums]}
        self.v2_in = self.down_stack("down_blocks.1.downsamplers.0.conv.", h)  # (the section's input: not summed here)
        self.v2_section()
        for i, C in ((2, 640), (3, 320)):
            for j in range(3):
                si = 11 - 3 * i - j
                x = self.cat_act(si)
                p = f"up_blocks.{i}.resnets.{j}."
                d = self.h_dest(si - 1) if j < 2 else self.new_dest(f"up{i}", C, x.W)
                h = self.resnet(p, x, C, self.new_dest(p + "out", C, x.W))
                self.outs.append((p, h))
                p = f"up_blocks.{i}.attentions.{j}."
                h = self.transformer(p, h, d)
                self.outs.append((p, h))
                if j < 2:
                    self.cat_sums[si - 1]["h"] = [o for o, _ in h.sums]
            if i == 2:
                p = "up_blocks.2.upsamplers.0.conv."
                h = self.up(p, h, self.h_dest(2))
                self.outs.append((p, h))
                self.cat_sums[2]["h"] = [o for o, _ in h.sums]
        po = self.norm("pout", h, w["conv_norm_out.weight"], w["conv_norm_out.bias"])
        wout = torch.zeros(80, 320, 3, 3)
        wout[:4] = w["conv_out.weight"]
        bout = torch.zeros(80)
        bout[:4] = w["conv_out.bias"]
        self.alloc("out", ysz(80, 64) + SLACK)
        self.conv("cout", 64, po, ci=320, co=80, wt=wout, b=bout, out=("out", 0))

    def time_bias_images(self, biases: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
        """(arena name, bf16 weight image) of each overlay resnet's time-bias sum, from the biases (out channels,) by
        resnet prefix (both images: one timestep)."""
        return [(name, torch.from_numpy(op.pack(biases[p][None].expand(2, -1).contiguous())).view(torch.bfloat16))
                for name, p, op in self.temb_imgs]

    def layout(self) -> tuple[dict[str, tuple[int, int]], int]:
        """Arena placement (name -> (bf16 offset, size), the total): the weights, zero rows, latents, output and
        per-prompt buffers first; the activations by their item ranges, first fit, reusing the space of dead ones."""
        def al(v: int) -> int:
            return -(-v // ALIGN) * ALIGN
        persistent = ({n for n, _ in self.data} | {"zeros", "st_none", "lat", "out"} | {n for n, _ in self.kv_packs})
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        for i, (_, args) in enumerate(self.prog.items):
            for n, _ in args:
                first.setdefault(n, i)
                last[n] = i
        names, top = {}, 0
        for n, size in self.specs.items():
            if n in persistent:
                names[n], top = (top, size), top + al(size)
        live: list[tuple[int, int, int]] = []  # (offset, end, last use)
        end = top
        for n in sorted((n for n in self.specs if n not in persistent and n in first), key=lambda n: first[n]):
            live = [x for x in live if x[2] >= first[n]]
            size = al(self.specs[n])
            o = top
            for a, b, _ in sorted(live):
                if o + size <= a:
                    break
                o = max(o, b)
            names[n] = (o, self.specs[n])
            live.append((o, o + size, last[n]))
            end = max(end, o + size)
        return names, end

    def arena(self) -> Arena:
        """The arena of the compiled program (``prog.compile``), the weight images written."""
        names, n = self.layout()
        self.prog.check_sizes({k: v[1] for k, v in names.items()})
        return Arena(self.prog.lib.device, names, n, self.data)


def latents_layout(x: torch.Tensor) -> torch.Tensor:
    """Latents (2, 4, 64, 64) -> conv_in's input (``UNetProgram`` "lat"): the conv layout of 32 channels, bf16 (only
    the first of its 4 channel blocks converted: the others' zero blocks are zero bytes)."""
    x8 = torch.zeros(2, 8, 64, 64)
    x8[:, :4] = x
    out = np.zeros((66, 8, 4, 2 * 8 * 9), np.uint8)  # [row][group][cb][pair, lane, 9 B]
    out[:, :, :1] = ov.conv_layout(ov.pad_pairs(x8)).reshape(66, 8, 1, -1)
    return torch.from_numpy(out.reshape(-1)).view(torch.bfloat16)


def output_values(y: torch.Tensor) -> torch.Tensor:
    """conv_out's output (``UNetProgram`` "out": 80-channel C tiles at 64x64) -> the noise prediction (2, 4, 64, 64)."""
    return ov.from_ctile(y.float(), 80, 64, 64)[:, :4]
