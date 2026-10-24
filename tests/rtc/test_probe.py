"""artlite-probe (models.probe) on the NPU: its table, the array clock and the DDR rates in their ranges."""

import pytest

from models import probe


def test_probe(capsys: pytest.CaptureFixture[str]) -> None:
    probe.main(["--seconds", "0.2"])
    out = capsys.readouterr().out
    rows = {cells[1].strip(): cells[2].strip() for line in out.splitlines()
            if line.startswith("| ") and len(cells := line.split("|")) == 4 + len(probe.REFERENCES)}
    assert all(set(ref) <= set(rows) for _, ref in probe.REFERENCES), out
    assert 400 < float(rows[probe.CLOCK]) < 3000, out
    assert 400 < float(rows[probe.CLOCK_LOAD]) < 3000, out
    assert 0 < float(rows[probe.ROUND_TRIP]) < 10000, out
    one, all16 = float(rows[probe.CASES[0][0]]), float(rows[probe.CASES[2][0]])
    assert 2 < one < all16, out
    assert 4 < float(rows[probe.PER_CYCLE]) < 128, out
