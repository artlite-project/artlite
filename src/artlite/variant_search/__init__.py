"""Kernel variant search: compile named variants of an operator's kernel and rank each loop of the target
functions.

A variant is the operator's rendered kernel with any of: a loop pragma on the target functions' loops and those of
the always_inline helpers they call (``post`` = AIE_PREPARE_FOR_POSTPIPELINING, ``none``, ``iiN`` =
AIE_TRY_INITIATION_INTERVAL(N), any text),
source substitutions (each must match exactly once), replaced function definitions, other operator (Jinja)
parameters. All variants compile in parallel with measure_ii; per loop of each target function one table ranks
them by cycles per entry: II, stages, iterations, cycles (prologue and epilogue included), body bundles, spill
code, stack, the ops per issue slot of the loop body in the assembly and the binding slot. For an unpipelined
loop (or every loop with ``why``) a debug build of llvm-aie gives the post-pipeliner's ResMII / RecMII and why
each II failed (the report says when that build's commit is not the wheel's). With ``npu``, each variant also
runs on the NPU, one process per build, one at a time: the time of one dispatch and the event-window counters of
named regions next to their static cycles.

    python -m artlite.variant_search 'tests.decode.mmv.MmvOperator(fmt="Q5_K", n_tiles=4)' \\
        --fn _q5k --pragmas post,none,ii44 [--sub NAME OLD NEW] [--define NAME FN @def.cc] \\
        [--param NAME key=value] [--variants FILE.py] [--why] [--npu --region codes=_q5k]

OLD / NEW / definitions starting with ``@`` are read from that file. Python: ``search(...)``, ``print_report``,
``run_npu``.
"""

from .npu import run_npu
from .report import print_report
from .search import search
from .variants import Variant

__all__ = ["Variant", "print_report", "run_npu", "search"]
