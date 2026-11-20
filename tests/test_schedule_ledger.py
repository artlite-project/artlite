"""The schedule ledger's own logic (``schedule_ledger.py``; the apps' ledgers against the baseline are
``test_app_designs.py``'s): the core bodies' calls with scratchpad parameters, the diff's classes, a prefill chunk's
weights (no NPU)."""

import pytest

from . import schedule_ledger as sl
from .apps import APPS
from .model_files import require

_DESIGN = """
module {
  aie.device(npu2) @op0_Test {
    %tile_0_2 = aie.tile(0, 2)
    func.func private @f(i32)
    func.func private @g()
    func.func private @h()
    %core_0_2 = aie.core(%tile_0_2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %n = aiex.read_scratchpad_parameter @n : i32
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %m = arith.muli %n, %c2 : i32
        scf.for %i = %c0 to %m step %c1 : i32 {
          func.call @f(%i) : (i32) -> ()
          %first = arith.cmpi eq, %i, %c0 : i32
          scf.if %first {
            func.call @g() : () -> ()
          }
        }
        %more = arith.cmpi sgt, %n, %c1 : i32
        scf.if %more {
          func.call @h() : () -> ()
        } else {
          func.call @g() : () -> ()
        }
        scf.yield
      }
      aie.end
    }
  }
  aiex.scratchpad_parameter @n : i32
}
"""


@pytest.mark.parametrize(("n", "calls"), [(4, {"f": 8, "g": 1, "h": 1}), (1, {"f": 2, "g": 2})])
def test_core_calls(n: int, calls: dict[str, int]) -> None:
    """One pass of the main loop; a loop bound and branches on a parameter, a branch on the loop's index."""
    core = sl.core_calls(_DESIGN, {"n": n})["op0_Test"][(0, 2)]
    assert dict(core.calls) == calls and core.unknown == 0


def test_core_calls_unknown() -> None:
    core = sl.core_calls(_DESIGN, {})["op0_Test"][(0, 2)]
    assert core.unknown == 2 and not core.calls  # (the loop's bound, the branch)


def _ledger(sched: str, ii: int, core_bytes: int) -> sl.Ledger:
    loop = {"at": "f+3", "parent": -1, "sched": sched, "ii": ii, "ns": 2, "bound": 8, "iters": 16, "spills": 0,
            "bytes": 96}
    fn = {"cycles": 100 + 16 * ii, "upper": False, "bytes": 400, "stack": 32, "calls": [], "loops": [loop]}
    return sl.Ledger({"s": ("Op (m.py)", "a=1")},
                     {"app": {("fn", "s", "f"): fn, ("core", "", "s", 2): {"bytes": core_bytes}}})


def test_diff() -> None:
    """The diff's classes, worst first; an unchanged ledger has none."""
    base = _ledger("post-RA", 8, 16000)
    assert sl.diff(base, base) == []
    classes = [c for c, *_ in sl.diff(base, _ledger("none", 12, 16200))]
    assert classes == ["pipelining lost", "cycles up", f"cores below {sl.LOW} B free", "core bytes up"]
    assert [c for c, *_ in sl.diff(base, _ledger("post-RA", 9, 16000))] == ["II up", "cycles up"]


def test_ledger_rows(tmp_path: object) -> None:
    """A ledger's rows read back as the same ledger."""
    led = _ledger("post-RA", 8, 16000)
    path = f"{tmp_path}/ledger.jsonl"
    led.save(path)
    assert sl.Ledger.load(path) == led


@pytest.mark.parametrize("app", ["llama-1B-Q4_K_M"])
def test_chunk_weights(app: str) -> None:
    """Every item of a chunk weighted, nothing unknown; the GEMMs' MAC loop the heaviest."""
    path, build = APPS[app]
    require(path)
    w = sl.chunk_weights(build(path), 1024)
    assert w.unknown == 0 and sum(k[0] for k in w.kinds.values()) == w.items > 0
    (fn, _), _ = max(w.loops.items(), key=lambda x: x[1][0])
    assert "gq_mm" in w.functions[fn]["short"]
