"""artlite-probe (models.probe) on the NPU: its table, the array clock and the DDR rates in their ranges; the rows of
``--pages`` on the host."""

import collections

import numpy as np
import pytest

from artlite import txn
from models import probe

BUFFER = 16 * probe.CHANNEL_BYTES


def _channel_rows(words: np.ndarray) -> dict[tuple[int, int], tuple[np.ndarray, int, int]]:
    """Per (column, BD) of a transaction: its rows' byte addresses (rows joined where they touch), a row's bytes and
    the task's repeats (from the queue pushes)."""
    bds, repeats = {}, collections.Counter()
    for name, f in txn.decode(words):
        reg = f.get("addr", 0) & 0xFFFFF
        if name == "blockwrite" and txn.SHIM_BD0 <= reg < txn.SHIM_BD0 + txn.SHIM_BDS * txn.BD_BYTES:
            d = f["data"]
            n0, n1 = (d[3] >> 20) & 0x3FF, (d[4] >> 20) & 0x3FF
            s1, s2 = (d[4] & 0xFFFFF) + 1, (d[5] & 0xFFFFF) + 1
            count, step = (d[6] >> 20 & 0x3F) + 1, (d[6] & 0xFFFFF) + 1 if d[6] else 0  # (the iteration)
            r = np.arange(d[0] // n0)
            starts = d[1] + (d[2] << 32) + 4 * (r % n1 * s1 + r // n1 * s2)
            run = 4 * n0
            if n1 > 1 and s1 == n0:  # (a group's rows touch: one row)
                starts, run = starts[::n1], run * n1
            starts = np.concatenate([starts + 4 * step * j for j in range(count)])  # (the repeats of one iteration)
            bds[f["addr"] >> 25, (reg - txn.SHIM_BD0) // txn.BD_BYTES] = starts, run
        elif name == "write32" and reg in {q for qs in txn.QUEUE.values() for q in qs}:
            repeats[f["addr"] >> 25, f["value"] & 0xF] += ((f["value"] >> 16) & 0xFF) + 1
    return {key: (*rows, repeats[key]) for key, rows in bds.items()}


def test_page_cases() -> None:
    """``--pages``' rows do what their names say (the decoded BDs, no NPU): the pages a channel's task visits, rows per
    page, rows within their pages, channels on pages of their own and inside the probe's buffers."""
    src, dst = 1 << 32, 2 << 32  # (page-aligned stand-ins of the buffers' device addresses)
    names = [case.name for case in probe.PAGE_CASES]
    assert len(set(names)) == len(names)
    for case in probe.PAGE_CASES[1:]:
        rows = case.rows
        for reads, writes in case.tasks():
            words, total = probe.page_transaction(case, reads, writes, src, dst)
            channels = _channel_rows(words)
            assert len(channels) == len(reads) + len(writes), case.name
            seen: dict[int, set[int]] = {src: set(), dst: set()}
            for starts, run, repeat in channels.values():
                base = src if starts[0] < dst else dst
                assert repeat == case.repeat and run == rows.run, case.name
                assert base <= starts.min() and starts.max() + run <= base + BUFFER, case.name
                first, last = (starts - base) // probe.PAGE, (starts - base + run - 1) // probe.PAGE
                if run > probe.PAGE:
                    assert (starts % probe.PAGE == 0).all() and (last - first + 1 == run // probe.PAGE).all()
                elif "slices" not in case.name:
                    assert (first == last).all(), f"{case.name}: a row crosses a page"
                pages = set(first.tolist())
                assert not pages & seen[base], f"{case.name}: channels share pages"
                seen[base] |= pages
                visits = 1 + np.count_nonzero(first[1:] != first[:-1])  # (runs of rows on one page)
                if "slices" in case.name:  # (512 rows a task, a page or two each)
                    assert len(first) == visits == 512 and ((last - first) <= 1).all(), case.name
                elif " rows" in case.name:  # (in_pages: k rows a page, each of its pages once)
                    k = int(case.name.split(", ")[1].split(" a page")[0]) if run < probe.PAGE else 1
                    assert len(first) == visits * k and len(pages) == visits, case.name
                    assert str(len(pages)) in case.name.split(":")[1] and visits * k * run == case.nbytes
                else:  # (cycle: 16 turns through n pages, a page each row; the skewed rows move a few on)
                    n = case.nbytes * (case.iteration or (0, 1))[1] // (16 * 256)
                    assert visits == len(first) == 16 * n and n <= len(pages) < 2 * n, case.name
                    assert all(len(set(first[t * n:(t + 1) * n].tolist())) == n for t in range(16)), case.name
                    assert str(n) in case.name.split(" pages")[0]
            assert total == case.repeat * case.nbytes * len(channels)


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


def test_probe_pages(capsys: pytest.CaptureFixture[str]) -> None:
    probe.main(["--pages", "--seconds", "0.05"])
    out = capsys.readouterr().out
    rows = {cells[1].strip(): cells[2].strip() for line in out.splitlines()
            if line.startswith("| ") and len(cells := line.split("|")) == 4 + len(probe.PAGE_REFERENCES)}
    rows.pop("")  # (the header)
    assert list(rows) == probe.PAGE_ROWS, out
    rates = {name: [float(x) for x in value.split(" / ")] for name, value in rows.items() if name != probe.CLOCK}
    assert all(len(v) == (1 if case.mixed else 2) for case, v in zip(probe.PAGE_CASES, rates.values(), strict=True))
    assert all(10 < x < 1000 for x in rates[probe.PAGE_CASES[0].name]), out  # (GB/s)
    assert all(0.5 < x < 2000 for v in rates.values() for x in v), out
