"""Measure AIE loop schedules from LLVM optimization records and assembly.

Loop model (verified against llvm-aie sources and the emitted assembly):

- The loops are the final code's loops. The asm printer's loop comments
  (``=>This Loop Header: Depth=N``, ``Parent Loop``, ``in Loop: Header=``)
  give every loop header, its nesting and its blocks, including multi-block
  branch loops that no remark names by block. The assembly read is the one of
  the remarks compile: ``-fsave-optimization-record`` implies line tables, so
  its ``.loc`` directives give each bundle's source line (its code is the
  clean compile's).
- The per-loop *final* result is a single ``!Passed`` remark
  ``Pass: pipeliner, Name: schedule`` with a ``Pipeliner`` argument, emitted by
  ``AIEInterBlockScheduling::emitLoopRemarks`` after both pipeliners and
  register allocation have run, for single-block loops only.
  ``Pipeliner: postpipeliner`` = post-RA pipelined (II/NS from the schedule),
  ``Pipeliner: prepipeliner`` = pre-RA pipelined (II = actual loop-region
  length after code generation, NS from metadata), ``Pipeliner: loop-aware``
  = the loop stayed unpipelined (II = body length). Prologue/Epilogue block
  names and bundle counts are included for all three cases. The remark names
  the loop block ``bb.N.<IR name>``; block numbers can change before the asm
  printer runs, and a pre-RA pipelined loop's prologue keeps its IR name, so
  the remark goes to the loop header the asm printer reports with that IR
  name, II and NS, else to the loop of that IR name, in layout order.
- The pre-RA MachinePipeliner remarks (``canPipelineLoop``, ``schedule``)
  know a loop only by its statement's source location
  (``MachineLoop::getStartLoc``); every loop gets at least one.
  ``!Analysis ... "Schedule found with Initiation Interval"`` is the pre-RA
  *search* result: a plan made before code generation and register
  allocation, often shorter than the generated loop; reported as a note,
  never as the schedule. ``!Passed ... "Pipelined succesfully!"`` only says
  the plan was applied. Rejections (``Not a single basic block: N``, ``The
  loop structure is not supported``, ``Unable to find schedule``) matter only
  for loops that stayed unpipelined: the post-RA pipeliner may still succeed.
- These source loops are matched with the final loops: a pre-RA schedule
  names its loop block; otherwise the source lines of a loop's own bundles
  (the loop statement, its extent parsed from the source, that contains most
  of them, consistently with the nesting), then the lines of its preheader
  (the guard of the loop statement), then elimination. A source loop that
  matches no final loop is reported as unattributed.
- A zero-overhead loop (ZOL) starts at the label written to ``ls``
  (``aie-hardware-loops !Analysis`` reports the same for loops with a single
  loop-control block). ``aie-asm-printer !Analysis`` reports each block's
  BundleCount/ByteCount per region, in layout order, and on a scheduled loop
  block's first region the II and NS of its final remark.
- Hardware-loop trip counts are recovered from the assembly: a ZOL is set up
  by ``add.nc lc, rX, #imm`` and executes exactly ``lc`` times. The register
  value is read with start-of-bundle semantics (a ``mov`` to rX in the same
  bundle is not visible to the ``add.nc``). For pipelined loops the pipeliner
  peels ``NS-1`` iterations into prologue+epilogue, so the source trip count
  is ``lc + (NS-1)``.
- ``regalloc`` ``(Loop)SpillReloadCopies`` remarks count spilled and reloaded
  registers per loop statement (inner loops included) and per function; each
  run of the allocator emits them again, the last run counts all register
  classes. The assembly's ``N-byte [Folded] Spill/Reload`` comments count
  instructions: a register wider than 64 bytes moves in 64-byte parts (a
  2048-bit accumulator spill is 4 stores).
- Costs count a bundle as a cycle. A loop's iteration adds its inner loops'
  iterations (their prologue and epilogue blocks are the enclosing loop's own
  blocks) and its calls, a call costing the callee's code outside loops, its
  loops and its calls. Branches count every path (an upper bound). The
  remark's prologue and epilogue are whole blocks (the preheader, the exit):
  between two consecutive loops one block is the first one's epilogue and the
  second one's prologue; it counts in the first loop's cost only.
- The asm printer's comment on a ``.loc`` lists the inlining chain of the
  bundle (``vector.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]``):
  the source matching uses the innermost frame in a loop statement, so a
  helper's loop inlined from a header (whose own bundles come from deeper
  headers) still finds its statement, and the frame in the measured file
  after it is the call site.

Modules, in the order a kernel goes through them: ``compilation`` (the two compiles), ``remarks`` (the YAML record),
``asm`` (functions, blocks, loops and ZOL trip counts of the assembly), ``statements`` and ``trips`` (the loop
statements and constant trip counts of the sources, on ``artlite.cxx_scan``), ``match`` (source loops to final loops),
``costs``, ``issues``, ``slots`` (issue slots and shared ports), ``render`` (the printed report), ``driver``
(``measure``); ``report`` holds the data classes and ``demangle`` the symbol names.
"""

from .costs import call_cycles, loop_cycles
from .driver import measure
from .report import KernelReport, LoopReport, SourceLoop
from .slots import PORTS, SLOTS, SlotCounts, count_slots, ports_of, slot_of

__all__ = [
    "PORTS",
    "SLOTS",
    "KernelReport",
    "LoopReport",
    "SlotCounts",
    "SourceLoop",
    "call_cycles",
    "count_slots",
    "loop_cycles",
    "measure",
    "ports_of",
    "slot_of",
]
