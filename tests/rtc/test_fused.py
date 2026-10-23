"""Tests for FusedOperator: compile and dispatch multiple ops as one ELF."""

import numpy as np
import pytest
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter, TaskGroup, Worker, sync_parameters
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

import artlite
from artlite import backend
from artlite.operators import FusedOperator
from artlite.operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from artlite.operators.fused import Library

from .common import verify_close
from .elementwise import SiluOperator, StreamOperator

_ADD_SRC = JINJA_ENV.from_string("""
extern "C"
void {{ name }}(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict c) {
    event0();
    aie::set_rounding(aie::rounding_mode::conv_even);
    auto ones = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    for (int i = 0; i < {{ ts }} / 32; i++) {
        aie::accum<accfloat, 32> acc;
        acc.from_vector(aie::load_v<32>(a), 0);
        aie::store_v(c, aie::mac(acc, ones, aie::load_v<32>(b)).template to_vector<bfloat16>());
        a += 32; b += 32; c += 32;
    }
    event1();
}
""")


class AddOperator(StreamOperator):
    """c = a + b on 1-D bf16 vectors (a two-input op for the runlist tests)."""

    def __init__(self, *, size: int):
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(torch.bfloat16, torch.bfloat16),
                         dtypes_out=(torch.bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"test_add_bf16_{self.tile}"

    def get_kernel_src(self) -> str:
        return _ADD_SRC.render(name=self.kernel_name, ts=self.tile)


class TestFusedOperator:
    """Basic fused operator tests using simple element-wise ops."""

    def test_silu_add(self):
        """Fuse silu + add:  out = silu(in) + in."""
        SIZE = 1024
        silu = SiluOperator(size=SIZE)
        add = AddOperator(size=SIZE)

        fused = FusedOperator(
            name="test_silu_add",
            runlist=[
                (silu, "in", "scratch"),  # scratch = silu(in)
                (add, "in", "scratch", "out"),  # out = in + scratch
            ],
            input_args=["in"],
            output_args=["out"],
        )
        fused.compile()

        torch.manual_seed(42)
        inp = torch.randn(SIZE, dtype=torch.bfloat16)

        ref_silu = torch.nn.functional.silu(inp)
        ref = inp + ref_silu

        result = fused.run(inputs={"in": inp})
        verify_close(result["out"], ref, rel_tol=0.04, abs_tol=1e-6, label="silu_add")

    def test_double_silu(self):
        """Fuse silu twice:  out = silu(silu(in))."""
        SIZE = 1024
        silu = SiluOperator(size=SIZE)

        fused = FusedOperator(
            name="test_double_silu",
            runlist=[
                (silu, "in", "scratch"),  # scratch = silu(in)
                (silu, "scratch", "out"),  # out = silu(scratch)
            ],
            input_args=["in"],
            output_args=["out"],
        )
        fused.compile()

        torch.manual_seed(123)
        inp = torch.randn(SIZE, dtype=torch.bfloat16)

        ref = torch.nn.functional.silu(torch.nn.functional.silu(inp))

        result = fused.run(inputs={"in": inp})
        verify_close(result["out"], ref, rel_tol=0.04, abs_tol=1e-6, label="double_silu")

    def test_get_buffer_write(self):
        """Test manual buffer access: write inputs, call, read outputs."""
        SIZE = 1024
        silu = SiluOperator(size=SIZE)

        fused = FusedOperator(
            name="test_buf_write",
            runlist=[
                (silu, "in", "out"),
            ],
            input_args=["in"],
            output_args=["out"],
        )
        fused.compile()

        torch.manual_seed(7)
        inp = torch.randn(SIZE, dtype=torch.bfloat16)
        ref = torch.nn.functional.silu(inp)

        fused.get_buffer("in").copy_(inp)
        fused.sync_inputs()
        fused()
        fused.sync_outputs()
        output = fused.get_buffer("out").clone()

        verify_close(output, ref, rel_tol=0.04, abs_tol=1e-6, label="get_buffer")


_SHIFT_SRC = JINJA_ENV.from_string("""
extern "C"
void {{ name }}(bfloat16 *__restrict a, bfloat16 *__restrict c) {
    for (int i = 0; i < {{ n }} / 32; i++)
        aie::store_v(c + 32 * i, aie::load_v<32>(a + 32 * i));
}
""")


class ShiftOperator(ArtcOperator):
    """out = x[s : s + n] with s the scratchpad parameter ``t_shift`` (bf16 elements; x holds 2 n): an operator that
    reads a runtime parameter."""

    PARAM = "t_shift"

    def __init__(self, *, n: int):
        self.n = n
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SHIFT_SRC.render(name=f"test_shift_{self.n}", n=self.n)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (2 * self.n,), torch.bfloat16), ArgSpec("out", (self.n,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        n = self.n
        ty = np.ndarray[(n,), np.dtype[bfloat16]]
        k = Kernel(f"test_shift_{n}", "kernel.o", [ty, ty])
        i_f, o_f = ObjectFifo(ty, name="sh_i"), ObjectFifo(ty, name="sh_o")
        shift = ScratchpadParameter(self.PARAM, np.int32)

        def core(i_h, o_h, kern):
            a, c = i_h.acquire(1), o_h.acquire(1)
            kern(a, c)
            o_h.release(1)
            i_h.release(1)

        def seq(x, out, ih, oh):
            sync_parameters()
            tg = TaskGroup()
            ih.fill(x, TensorAccessPattern((2 * n,), 0, [1, 1, 1, n], [0, 0, 0, 1]), offset_parameter=shift, group=tg)
            oh.drain(out, TensorAccessPattern((n,), 0, [1, 1, 1, n], [0, 0, 0, 1]), group=tg, wait=True)
            tg.finish()

        w = Worker(core, [i_f.cons(), o_f.prod(), k], tile=Tile(0, 2))
        rt = Runtime(seq, [np.ndarray[(2 * n,), np.dtype[bfloat16]], ty, i_f.prod(tile=Tile(0, 0)),
                           o_f.cons(tile=Tile(0, 0))])
        return Program(NPU2(), rt, workers=[w]).resolve_program()


def test_library_program():
    """A program assembled from a library ELF's blocks (``Library``): 4 items of 2 entries in 3 dispatches of one
    context, the arguments bound per item; a parameter written once is read in the first and the second piece."""
    from artlite.txn import device_address

    n = 1024
    shift, add = ShiftOperator(n=n), AddOperator(size=n)
    lib = Library("t_library", [shift, add]).compile().load()  # (on XRT, buffer addresses need a context first)
    dev = artlite.Device()
    x = torch.randn(2 * n).bfloat16()
    bx = artlite.from_torch(x.clone(), dev)
    t, u, v, out = (artlite.empty((n,), dtype=torch.bfloat16, device=dev) for _ in range(4))
    a = {k: device_address(b, 0, lib._ctx) for k, b in {"x": bx, "t": t, "u": u, "v": v, "out": out}.items()}
    items = [(0, [a["x"], a["t"]]), (1, [a["t"], a["t"], a["u"]]), (0, [a["x"], a["v"]]), (1, [a["u"], a["v"], a["out"]])]
    lib.set_params({ShiftOperator.PARAM: 128})
    assert lib.run(items, splits=[1, 3], bufs=[b._raw for b in (bx, t, u, v, out)]) == 3
    out.sync_from_device()
    want = (3 * x[128:128 + n].float()).bfloat16()  # (2 x[s : s + n] + x[s : s + n])
    torch.testing.assert_close(out.to_torch().float(), want.float(), rtol=1e-2, atol=1e-2)


def test_library_program_packets(monkeypatch):
    """The program of ``test_library_program`` with every configuration's program memory by control packets (the
    loader of ``ctrlpkt``) gives the outputs of the one with blockwrites."""
    from artlite.operators.fused import ctrlpkt
    from artlite.txn import device_address

    n = 1024
    shift, add = ShiftOperator(n=n), AddOperator(size=n)
    dev = artlite.Device()
    x = torch.randn(2 * n).bfloat16()
    outs = []
    for packets in (False, True):
        monkeypatch.setattr(ctrlpkt, "ENABLED", packets)
        monkeypatch.setattr(ctrlpkt, "worth", lambda loader: True)
        lib = Library("t_library", [shift, add]).compile()
        lib.load()
        assert all(bool(blk.packets) == packets for blk in lib._asm.blocks)
        bx = artlite.from_torch(x.clone(), dev)
        t, u, v, out = (artlite.empty((n,), dtype=torch.bfloat16, device=dev) for _ in range(4))
        a = {k: device_address(b, 0, lib._ctx) for k, b in {"x": bx, "t": t, "u": u, "v": v, "out": out}.items()}
        items = [(0, [a["x"], a["t"]]), (1, [a["t"], a["t"], a["u"]]), (0, [a["x"], a["v"]]),
                 (1, [a["u"], a["v"], a["out"]])]
        lib.set_params({ShiftOperator.PARAM: 64})
        lib.run(items, bufs=[b._raw for b in (bx, t, u, v, out)])
        out.sync_from_device()
        outs.append(out.to_torch().clone())
    assert torch.equal(outs[0], outs[1])
    torch.testing.assert_close(outs[1].float(), (3 * x[64:64 + n].float()).bfloat16().float(), rtol=1e-2, atol=1e-2)


class AxpyOperator(AddOperator):
    """c = a + s b, s a compile-time constant: designs of different s differ only in program memory."""

    def __init__(self, *, size: int, s: float):
        self.s = s
        super().__init__(size=size)

    @property
    def kernel_name(self) -> str:
        return f"test_axpy_bf16_{self.tile}_{int(self.s)}"

    def get_kernel_src(self) -> str:
        return _ADD_SRC.render(name=self.kernel_name, ts=self.tile).replace("(bfloat16)1.0f", f"(bfloat16){self.s}f")


def test_library_program_memory_variants(monkeypatch):
    """Two entries whose configurations differ only in program memory, with it by control packets (the rest of their
    configurations then identical): each item runs its own entry's program."""
    from artlite.operators.fused import ctrlpkt
    from artlite.txn import device_address

    n = 1024
    monkeypatch.setattr(ctrlpkt, "worth", lambda loader: True)
    lib = Library("t_library_pm", [AxpyOperator(size=n, s=2.0), AxpyOperator(size=n, s=3.0)]).compile()
    lib.load()
    b0, b1 = lib._asm.blocks

    def bare(blk: object) -> np.ndarray:  # (its configuration without its packets' buffer addresses)
        c = blk.config.copy()
        for w, _ in blk.packets:
            c[w:w + 2] = 0
        return c[4:]

    assert b0.packets and np.array_equal(bare(b0), bare(b1)) and b0.design != b1.design
    dev = artlite.Device()
    x, y = torch.randn(n).bfloat16(), torch.randn(n).bfloat16()
    bx, by = artlite.from_torch(x.clone(), dev), artlite.from_torch(y.clone(), dev)
    t, u = (artlite.empty((n,), dtype=torch.bfloat16, device=dev) for _ in range(2))
    a = {k: device_address(b, 0, lib._ctx) for k, b in {"x": bx, "y": by, "t": t, "u": u}.items()}
    lib.run([(0, [a["x"], a["y"], a["t"]]), (1, [a["x"], a["y"], a["u"]])], bufs=[b._raw for b in (bx, by, t, u)])
    for s, out in ((2, t), (3, u)):
        out.sync_from_device()
        torch.testing.assert_close(out.to_torch().float(), (x.float() + s * y.float()).bfloat16().float(), rtol=1e-2,
                                   atol=1e-2)


def test_control_packets_encoding(tmp_path):
    """``ctrlpkt.packets`` encodes a tile's writes as mlir-aie's aie-translate does (stream and control headers with
    their parity, up to 4 words a packet)."""
    import os
    import subprocess

    from artlite.operators.base import AIECC
    from artlite.operators.fused import ctrlpkt

    rng = np.random.default_rng(1)
    writes = [(0x20000, rng.integers(0, 1 << 32, 9, dtype=np.uint64).astype(np.uint32)),
              (0x20400, rng.integers(0, 1 << 32, 4, dtype=np.uint64).astype(np.uint32))]
    ops = "\n".join(f"      aiex.control_packet {{address = {(3 << 25) | (4 << 20) | (a + 4 * i)} : ui32, data = "
                     f"array<i32: {', '.join(str(int(np.int32(v))) for v in d[i:i + 4])}>, opcode = 0 : i32, "
                     f"stream_id = 0 : i32}}" for a, d in writes for i in range(0, len(d), 4))
    mlir = tmp_path / "pk.mlir"
    mlir.write_text(f"module {{\n  aie.device(npu2) {{\n    %t = aie.tile(3, 4) {{controller_id = "
                    f"#aie.packet_info<pkt_type = 0, pkt_id = {ctrlpkt.ROW_ID[4]}>}}\n    aie.runtime_sequence() {{\n"
                    f"{ops}\n    }}\n  }}\n}}\n")
    out = tmp_path / "pk.bin"
    subprocess.run([os.path.join(os.path.dirname(AIECC), "aie-translate"), "--aie-ctrlpkt-to-bin", str(mlir), "-o", str(out)], check=True)
    want = np.fromfile(out, np.uint32)
    assert np.array_equal(ctrlpkt.packets(ctrlpkt.ROW_ID[4], writes), want)


def _add_chain(name: str, **kw: object) -> FusedOperator:
    """y = (x + w) + w with w a scratch weight buffer (external if given)."""
    add = AddOperator(size=1024)
    return FusedOperator(name=name, runlist=[(add, "x", "w", "t"), (add, "t", "w", "y")], input_args=["x"],
                         output_args=["y"], **kw)


class TestFusedLifecycle:
    """Runlist validation, control-program patching and buffers shared between ELFs."""

    def test_reads_before_writes(self):
        assert _add_chain("t_rbw")._reads_before_writes() == ["w"]
        assert _add_chain("t_rbw_u", uninitialized=["w"])._reads_before_writes() == []
        with pytest.raises(AssertionError, match="nope"):
            _add_chain("t_rbw_x", uninitialized=["nope"])

    def test_patch_external(self):
        """Shim BDs addressing an external scratch buffer point into its own BO."""
        from artlite.txn import device_address

        dev = artlite.Device()
        ext = dev.alloc_host_buffer(1 << 16)
        f = _add_chain("t_patch_ext", buffer_sizes={"w": 2048}, external_buffers={"w": (ext, 4096)}).compile().load()
        w, t = f.span("w"), f.span("t")
        assert w.offset > t.offset, "external buffers are laid out last in scratch"
        words = f._insts_bo.map().view(np.uint32)
        in_ext = 0
        for bd in f.shim_bds():
            addr = int(words[bd.offset // 4 + 1]) | (int(words[bd.offset // 4 + 2]) & 0xFFFF) << 32
            if bd.space == "scratch" and w.offset <= bd.start < w.offset + w.nbytes:
                in_ext += 1
                assert addr == device_address(ext, 4096 + bd.start - w.offset, f._ctx)
            else:
                assert addr == device_address(*f._bo_at(bd.space, bd.start), f._ctx)
        assert in_ext == 2, f"expected the two reads of w, got {in_ext}"

    def test_async_dispatches(self):
        """Several dispatches in flight; a wait whose timeout expires leaves them pending. Each dispatch is compute
        bound and long (SiLU 48 times over each tile on one core: ~22 ms) against what bounds the work still queued at
        the wait: a 1 ms wait returns 4-8 ms late (the driver's tick) and a submit blocks once a few dispatches are
        queued (~1 ms each)."""
        SIZE = 1 << 20
        fused = FusedOperator(name="test_async_silu", runlist=[(SiluOperator(size=SIZE, repeat=48), "in", "out")],
                              input_args=["in"], output_args=["out"]).compile()
        torch.manual_seed(0)
        inp = torch.randn(SIZE, dtype=torch.bfloat16)
        fused.get_buffer("in").copy_(inp)
        fused.sync_inputs()
        kern = fused._kernel()
        for _ in range(3):
            kern.dispatch_elf_async(fused._insts_bo, fused._dispatch_bufs)
        assert not kern.dispatch_poll()
        with pytest.raises(artlite.DispatchError, match="did not finish within 1 ms") as e:
            kern.dispatch_wait(timeout_ms=1)
        assert e.value.state_name in ("NEW", "QUEUED", "RUNNING") and e.value.health is None
        kern.dispatch_wait()
        assert kern.dispatch_poll()
        fused.sync_outputs()
        verify_close(fused.get_buffer("out"), torch.nn.functional.silu(inp), rel_tol=0.04, abs_tol=1e-6,
                     label="async")

    def test_missing_host_write(self):
        f = _add_chain("t_missing").compile()
        f.get_buffer("x").zero_()
        try:
            f()
        except RuntimeError as e:
            assert "'w'" in str(e)
        else:
            raise AssertionError("dispatch without writing w must fail")

    def test_shared_buffer(self):
        """ELF 2 reads ELF 1's scratch buffer in place (external_buffers)."""
        add = AddOperator(size=1024)
        a = FusedOperator(name="t_share_a", runlist=[(add, "x", "x", "s")], input_args=["x"], output_args=["y"],
                          buffer_sizes={"y": 2})
        a.compile()
        # (the external BO exists once ELF 1 is loaded)
        b = FusedOperator(name="t_share_b", runlist=[(add, "s", "x2", "y")], input_args=["x2"], output_args=["y"],
                          buffer_sizes={"s": 2048}, external_buffers={"s": a.load().buffer_bo("s")})
        b.compile()
        torch.manual_seed(3)
        x, x2 = torch.randn(1024).bfloat16(), torch.randn(1024).bfloat16()
        a.get_buffer("x").copy_(x)
        a.sync_input_only()
        a()
        b.get_buffer("x2").copy_(x2)
        b.sync_input_only()
        b()
        b.sync_outputs()
        ref = (x.float() + x.float()).bfloat16().float() + x2.float()
        verify_close(b.get_buffer("y").clone(), ref.bfloat16(), rel_tol=0.01, abs_tol=1e-2, label="shared")

    def test_user_buffer(self):
        """External buffers in a user-memory BO (Device.alloc_user_buffer): the NPU reads w from it and writes t
        into it."""
        dev = artlite.Device()
        ub = dev.alloc_user_buffer(4 << 20)
        f = _add_chain("t_user", buffer_sizes={"w": 2048, "t": 2048},
                       external_buffers={"w": (ub, 1 << 20), "t": (ub, 3 << 20)}).compile().load()
        assert (backend.XRT or ub.paddr % (2 << 20) == 0) and not ub.map().any()  # (amdxdna: 2 MB pages)
        torch.manual_seed(7)
        x, w = torch.randn(1024).bfloat16(), torch.randn(1024).bfloat16()
        ub.map()[1 << 20:(1 << 20) + 2048] = w.view(torch.uint8).numpy()
        ub.sync_to_device(2048, 1 << 20)
        f.get_buffer("x").copy_(x)
        f.sync_input_only()
        f()
        f.sync_outputs()
        ub.sync_from_device(2048, 3 << 20)
        t = torch.from_numpy(ub.map()[3 << 20:(3 << 20) + 2048].copy()).view(torch.bfloat16)
        verify_close(t, (x.float() + w.float()).bfloat16(), rel_tol=0.01, abs_tol=1e-2, label="user t")
        ref = ((x.float() + w.float()).bfloat16().float() + w.float()).bfloat16()
        verify_close(f.get_buffer("y").clone(), ref, rel_tol=0.01, abs_tol=1e-2, label="user y")

    def test_scratch_segments(self, monkeypatch):
        """Scratch larger than one sequence argument's 32-bit relocation addends is passed as
        several arguments ("segments"; forced here with a 2 KB limit): the patched BDs still
        address the right buffers, also inside an external one."""
        from artlite.operators.fused import layout

        monkeypatch.setattr(layout, "SEGMENT_BYTES", 2048)
        a = FusedOperator(name="t_seg_a", runlist=[(AddOperator(size=1024), "x", "x", "s")],
                          input_args=["x"], output_args=["y"], buffer_sizes={"y": 2})
        a.compile()
        f = _add_chain("t_seg", buffer_sizes={"w": 2048}, external_buffers={"w": a.load().buffer_bo("s")}).compile()
        assert len(f.segments) == 2, f.segments  # t | w (each 2 KB)
        torch.manual_seed(5)
        x, w = torch.randn(1024).bfloat16(), torch.randn(1024).bfloat16()
        a.get_buffer("x").copy_(w * 0.5)
        a.sync_input_only()
        a()  # s = w
        f.get_buffer("x").copy_(x)
        f.sync_input_only()
        f()
        f.sync_outputs()
        wv = (w * 0.5).float() + (w * 0.5).float()
        ref = ((x.float() + wv).bfloat16().float() + wv).bfloat16()
        verify_close(f.get_buffer("y").clone(), ref, rel_tol=0.01, abs_tol=1e-2, label="segments")
        assert {bd.space for bd in f.shim_bds()} == {"input", "output", "scratch"}


def test_scratch_segments_overlap():
    """A scratch slice that crosses a segment's limit (a prefill GEMM's weight slice in an image above
    3.75 GB) starts a new segment that overlaps the previous one (host only)."""
    from artlite.operators.fused.layout import Span, scratch_segments

    spans = ((0, 30), (30, 70), (60, 100))
    segs = scratch_segments([Span("scratch", a, b - a) for a, b in spans], 64)
    assert segs == [(0, 30), (30, 70), (60, 100)], segs
    assert all(any(s <= a and b <= e for s, e in segs) for a, b in spans)


def test_patch_rejects_unknown_relocations():
    """Control code with a relocation the patcher does not handle is an error, not silently unpatched (host only)."""
    from artlite.elf_parser import (
        ElfGroup,
        ElfRelocation,
        ElfSection,
        ElfSymbol,
        ParsedElf,
        patch_control_code,
    )

    text = ElfSection(name=".ctrltext.0", index=1, offset=0, size=16, addr=0, data=bytes(16))
    elf = ParsedElf(groups={0: ElfGroup(0, text)}, section_to_group={1: 0},
                    sections=[ElfSection("", 0, 0, 0, 0, b""), text],
                    symbols=[ElfSymbol("", 0, 0, 0), ElfSymbol("0", 0, 16, 1)],
                    relocations=[ElfRelocation(offset=0, sym_index=1, rela_type=5, addend=4)])
    data, bds = patch_control_code(elf, address=lambda bd: 0x8000_1000 + bd.start, pdis={}, scratchpad=None)
    assert bds == [(0, "0", 4, 0)] and int.from_bytes(data[4:8], "little") == 0x8000_1004
    for rela_type, sym in ((3, "0"), (8, "scratch-pad-ctrl")):  # scalar_32bit; a scratchpad without its address
        elf.relocations = [ElfRelocation(offset=0, sym_index=1, rela_type=rela_type, addend=0)]
        elf.symbols[1].name = sym
        with pytest.raises(ValueError, match="relocation"):
            patch_control_code(elf, address=lambda bd: 0, pdis={}, scratchpad=None)


def test_without_sequences():
    """The design key of a library's device sharing: the module text without its runtime sequences."""
    from artlite.operators.fused.design import without_sequences
    text = 'aie.device(npu2) {\n  %t = aie.tile(0, 0)\n  aie.runtime_sequence @sequence(%a: memref<8xi32>) {\n' \
           '    aiex.npu.write32 {address = 0 : ui32, value = 1 : ui32}\n  }\n  aie.end\n}'
    assert without_sequences(text) == 'aie.device(npu2) {\n  %t = aie.tile(0, 0)\n  aie.end\n}'


def test_skip_known_bds():
    """A memory tile's BD written again with the values it holds is left out (its words, operation count and the BD
    address words after it moved); a BD with an iteration (word 6: the DMA advances its current step) never is."""
    from artlite.operators.fused.program import Run
    reg = (3 << 25) | (1 << 20) | 0xA0040  # (column 3's memory tile, BD 2)

    def run(word6: int) -> Run:  # (a blockwrite of the BD's 8 words, a masked write of its address field, a push)
        bd = [1, 0, reg, 48, 7, 0, 0, 0, 0, 0, word6, 0x8000, 3, 0, reg + 4, 0, 0x100, 0x7FFFF, 24]
        push = [0, 0, (1 << 20) | 0x1D214, 0, 2, 24]  # (a write32 after it: a BD address word at +2)
        words = np.array(bd + push, np.uint32)
        return Run(words, 3, np.array([len(words) - 4]))

    state = np.full((64, 384), -1, np.int64)
    r1 = run(0)
    words, _, n = r1.emit(state)
    assert len(words) == len(r1.words) and n == 3
    words, bd_words, n = r1.emit(state)
    assert np.array_equal(words, r1.words[-6:]) and n == 1 and bd_words[0] == 2
    r3 = run(5 << 17)  # (Iteration_Wrap 5)
    for _ in range(2):
        words, _, n = r3.emit(state)
        assert len(words) == len(r3.words) and n == 3
