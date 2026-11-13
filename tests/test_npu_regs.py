"""The tile registers of ``artlite.npu_regs`` against aie-rt's ``xaie2pgbl_params.h`` (the NPU's register map: shipped in
the mlir_aie wheel), and the BD lock fields' decoder."""

import glob
import os
import re
from importlib.util import find_spec

import pytest

from artlite import npu_regs, txn
from artlite.npu_regs import CORE_TILE, MEM_TILE, SHIM_TILE, BdLocks

_SUFFIXES = ("_MASK", "_DEFVAL", "_LSB", "_WIDTH")


class Header:
    """The register addresses and field positions of the header: ``reg("MEMORY_MODULE_LOCK0_VALUE")``,
    ``span("CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_")``: [first, last + 4) of the registers with that prefix."""

    def __init__(self, path: str):
        self.defs: dict[str, int] = {}
        with open(path) as f:
            for line in f:
                m = re.match(r"#define XAIE2PGBL_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)\b", line)
                if m:
                    self.defs[m[1]] = int(m[2], 0)

    def reg(self, name: str) -> int:
        return self.defs[name]

    def has(self, name: str) -> bool:
        return name in self.defs

    def span(self, prefix: str) -> tuple[int, int]:
        regs = [v for k, v in self.defs.items() if k.startswith(prefix) and not k.endswith(_SUFFIXES)]
        assert regs, prefix
        return min(regs), max(regs) + 4

    def lsb(self, name: str) -> int:
        return self.defs[name + "_LSB"]

    def width(self, name: str) -> int:
        return self.defs[name + "_WIDTH"]


@pytest.fixture(scope="module")
def h() -> Header:
    wheel = find_spec("mlir_aie").submodule_search_locations[0]
    found = glob.glob(os.path.join(wheel, "runtime_lib", "*", "xaiengine", "include", "xaiengine", "xaie2pgbl_params.h"))
    if not found:
        pytest.skip("the mlir_aie wheel ships no xaie2pgbl_params.h")
    return Header(found[0])


def _dma(h: Header, module: str, tile: npu_regs.TileMap, channels: tuple[int, int]) -> None:
    """The DMA channel registers of a tile: S2MM then MM2S, 8 B a channel (control, start queue)."""
    s2mm, mm2s = channels
    assert h.reg(f"{module}_DMA_S2MM_0_CTRL") == tile.dma[0] and tile.s2mm_channels == s2mm
    assert h.reg(f"{module}_DMA_S2MM_0_START_QUEUE") == tile.dma[0] + 4 == tile.start_queues[0]
    assert h.reg(f"{module}_DMA_S2MM_{s2mm - 1}_CTRL") == tile.dma[0] + 8 * (s2mm - 1)
    assert h.reg(f"{module}_DMA_MM2S_0_CTRL") == tile.mm2s_ctrl
    assert h.reg(f"{module}_DMA_MM2S_{mm2s - 1}_CTRL") == tile.dma[1] - 8
    assert not h.has(f"{module}_DMA_S2MM_{s2mm}_CTRL") and not h.has(f"{module}_DMA_MM2S_{mm2s}_CTRL")
    assert len(tile.channel_ctrl) == s2mm + mm2s
    for bits in ("S2MM", "MM2S"):
        assert (1 << h.width(f"{module}_DMA_{bits}_0_START_QUEUE_START_BD_ID")) - 1 == tile.start_bd_mask


def _locks_and_bds(h: Header, module: str, tile: npu_regs.TileMap, n_locks: int, n_bds: int, word: int) -> None:
    assert h.reg(f"{module}_LOCK0_VALUE") == tile.locks[0]
    assert h.reg(f"{module}_LOCK1_VALUE") - tile.locks[0] == npu_regs.LOCK_STRIDE
    assert h.reg(f"{module}_LOCK{n_locks - 1}_VALUE") + npu_regs.LOCK_STRIDE == tile.locks[1]
    assert not h.has(f"{module}_LOCK{n_locks}_VALUE")
    assert h.reg(f"{module}_DMA_BD0_0") == tile.bds[0] and h.reg(f"{module}_DMA_BD1_0") - tile.bds[0] == npu_regs.BD_STRIDE
    assert h.reg(f"{module}_DMA_BD{n_bds - 1}_0") + npu_regs.BD_STRIDE == tile.bds[1] and tile.n_bds == n_bds
    assert not h.has(f"{module}_DMA_BD{n_bds}_0")
    assert h.reg(f"{module}_DMA_BD0_{word}") == tile.bd_register(0, word)
    f, bd = tile.lock_word, f"{module}_DMA_BD0_{word}_"
    assert f.word == word
    assert f.valid == h.lsb(bd + "VALID_BD") and f.acq_enable == h.lsb(bd + "LOCK_ACQ_ENABLE")
    assert f.acq_value == h.lsb(bd + "LOCK_ACQ_VALUE") and h.width(bd + "LOCK_ACQ_VALUE") == 7
    assert f.rel_id == h.lsb(bd + "LOCK_REL_ID") and f.rel_value == h.lsb(bd + "LOCK_REL_VALUE")
    assert f.acq_id_width == h.width(bd + "LOCK_ACQ_ID") == h.width(bd + "LOCK_REL_ID") and h.lsb(bd + "LOCK_ACQ_ID") == 0
    assert h.width(bd + "LOCK_REL_VALUE") == 7 and h.lsb(bd + "VALID_BD") == h.lsb(bd + "VALID_BD")


def _switch(h: Header, module: str, tile: npu_regs.TileMap) -> None:
    prefix = f"{module}_STREAM_SWITCH_"
    assert h.span(prefix + "MASTER_CONFIG_") == tile.masters
    assert h.span(prefix + "SLAVE_CONFIG_") == tile.slaves
    assert min(v for k, v in h.defs.items() if k.startswith(prefix + "SLAVE_") and k.endswith("_SLOT0")) == tile.slot0
    assert h.reg(prefix + "MASTER_CONFIG_NORTH0") == tile.north0_master
    if tile.south0_slave is not None:
        assert h.reg(prefix + "SLAVE_CONFIG_SOUTH_0") == tile.south0_slave
        assert h.reg(prefix + "SLAVE_SOUTH_0_SLOT0") == tile.south0_slots
        assert h.reg(prefix + "MASTER_CONFIG_TILE_CTRL") == tile.tile_ctrl_master
    master = prefix + "MASTER_CONFIG_NORTH0"
    assert h.reg(master + "_MASTER_ENABLE_MASK") == npu_regs.SWITCH_ENABLE
    assert h.reg(master + "_PACKET_ENABLE_MASK") == npu_regs.SWITCH_PACKET


def test_core_tile(h: Header) -> None:
    _locks_and_bds(h, "MEMORY_MODULE", CORE_TILE, 16, 16, 5)
    _dma(h, "MEMORY_MODULE", CORE_TILE, (2, 2))
    _switch(h, "CORE_MODULE", CORE_TILE)
    assert h.reg("CORE_MODULE_PROGRAM_MEMORY") == CORE_TILE.program[0]
    assert h.reg("CORE_MODULE_PROGRAM_MEMORY_ERROR_INJECTION") == CORE_TILE.program[1]
    assert h.reg("CORE_MODULE_CORE_CONTROL") == CORE_TILE.control[0]
    assert h.lsb("CORE_MODULE_CORE_CONTROL_RESET") == 1 and npu_regs.CORE_CONTROL_RESET == 1 << 1
    assert h.reg("MEMORY_MODULE_DATAMEMORY") == 0
    assert CORE_TILE.data_bytes <= min(v for k, v in h.defs.items() if k.startswith("MEMORY_MODULE_") and v
                                       and not k.startswith("MEMORY_MODULE_DATAMEMORY") and not k.endswith(_SUFFIXES))
    assert CORE_TILE.own_locks == range(16) and CORE_TILE.iteration_word is None


def test_memory_tile(h: Header) -> None:
    _locks_and_bds(h, "MEM_TILE_MODULE", MEM_TILE, 64, 48, 7)
    _dma(h, "MEM_TILE_MODULE", MEM_TILE, (6, 6))
    _switch(h, "MEM_TILE_MODULE", MEM_TILE)
    assert h.reg("MEM_TILE_MODULE_DATAMEMORY") == 0
    assert MEM_TILE.data_bytes <= min(v for k, v in h.defs.items() if k.startswith("MEM_TILE_MODULE_") and v
                                      and not k.startswith("MEM_TILE_MODULE_DATAMEMORY") and not k.endswith(_SUFFIXES))
    word = MEM_TILE.iteration_word
    bd = f"MEM_TILE_MODULE_DMA_BD0_{word}_ITERATION_"
    assert h.lsb(bd + "STEPSIZE") == 0 and h.lsb(bd + "WRAP") == h.width(bd + "STEPSIZE")
    assert MEM_TILE.iteration_mask == (1 << (h.lsb(bd + "WRAP") + h.width(bd + "WRAP"))) - 1  # (not Iteration_Current)
    assert h.lsb(bd + "CURRENT") == h.lsb(bd + "WRAP") + h.width(bd + "WRAP")


def test_shim_tile(h: Header) -> None:
    _switch(h, "PL_MODULE", SHIM_TILE)
    assert h.reg("PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_SOUTH_3") == npu_regs.SHIM_SLAVE_SOUTH3
    assert h.reg("NOC_MODULE_MUX_CONFIG") == npu_regs.SHIM_MUX
    assert 1 << h.lsb("NOC_MODULE_MUX_CONFIG_SOUTH3") == npu_regs.SHIM_MUX_SOUTH3_DMA  # (1: the DMA)


def test_txn_registers(h: Header) -> None:
    """The shim DMA words ``txn`` owns, and the memory and core tile registers it takes from ``npu_regs``."""
    assert h.reg("NOC_MODULE_DMA_BD0_0") == txn.SHIM_BD0 and h.reg("NOC_MODULE_DMA_BD1_0") - txn.SHIM_BD0 == txn.BD_BYTES
    assert h.reg("NOC_MODULE_DMA_S2MM_0_CTRL") == txn.CTRL[txn.S2MM][0]
    assert h.reg("NOC_MODULE_DMA_MM2S_1_TASK_QUEUE") == txn.QUEUE[txn.MM2S][1]
    assert txn.MEM_BD0 == MEM_TILE.bds[0] and txn.MEM_LOCK0 == MEM_TILE.locks[0] and txn.CORE_LOCK0 == CORE_TILE.locks[0]
    assert h.reg("MEM_TILE_MODULE_DMA_S2MM_0_CTRL") == txn.MEM_S2MM_CTRL
    assert h.reg("MEM_TILE_MODULE_DMA_MM2S_0_CTRL") == txn.MEM_MM2S_CTRL


def test_bd_locks() -> None:
    """A core tile's BD word 5 and a memory tile's BD word 7."""
    w = 1 << 25 | 7 << 18 | 9 << 13 | 1 << 12 | (-2 & 0x7F) << 5 | 3  # (valid, release 7 of 9, acquire 3 >= 2)
    assert CORE_TILE.bd_locks(w) == BdLocks(True, True, 3, -2, 9, 7)
    assert CORE_TILE.bd_locks(0) == BdLocks(False, False, 0, 0, 0, 0)
    w = 1 << 31 | 5 << 24 | 130 << 16 | 1 << 15 | 1 << 8 | 70
    assert MEM_TILE.bd_locks(w) == BdLocks(True, True, 70, 1, 130, 5)
    assert CORE_TILE.own_lock_register(3) == 0x1F000 + 3 * 16 and MEM_TILE.own_lock_register(66) == 0xC0000 + 2 * 16
