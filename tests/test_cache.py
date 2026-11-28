"""The cache directory (artlite/cache.py) and ``artlite-cache`` (models/cache.py): the owned directories, what a cleaning
removes, and what it never touches (files that are not Artlite's, links and what they point at, entries a build holds).
No NPU, no toolchain run: every cache is a temporary root."""

import os
import threading
import time

import pytest

from artlite import cache, portable
from models import cache as cli

HEX16 = "0123456789abcdef"
OLD = 40 * cache.DAY


@pytest.fixture
def root(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    path = str(tmp_path / "cache")
    os.makedirs(path)
    monkeypatch.setattr(cache, "ROOT", path)
    return path


def _age(path: str, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t), follow_symlinks=False)


def _file(path: str, text: str = "x", age: float = 0.0) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    _age(path, age)
    return path


def _build(name: str, age: float = 0.0, label: str = "L") -> str:
    """A build directory of this toolchain, last used ``age`` seconds ago."""
    path = cache.toolchain_dir(name)
    _file(os.path.join(path, "main.pdi"))
    cache.write_entry(path, name.split("_")[0], label)
    _age(os.path.join(path, cache.ENTRY_JSON), age)
    return path


def _sentinels(tmp_path: object) -> str:
    """A directory outside the cache whose files must survive every cleaning."""
    out = str(tmp_path / "outside")
    _file(os.path.join(out, "precious.txt"), age=OLD)
    _file(os.path.join(out, f"rtc_{HEX16}", "main.pdi"), age=OLD)
    _age(os.path.join(out, f"rtc_{HEX16}"), OLD)
    return out


def _tree(path: str) -> set[str]:
    out = set()
    for top, dirs, files in os.walk(path):
        out |= {os.path.relpath(os.path.join(top, n), path) for n in dirs + files}
    return out


def test_owned_directories(root: str) -> None:
    """The root holds only tagged directories of Artlite's; the root itself gets no tag (tar --exclude-caches, borg and
    restic would skip a user's directory that holds the cache)."""
    tc = cache.toolchain_dir()
    assert os.path.basename(tc.rstrip(os.sep)) == cache.TC_PREFIX + cache.toolchain_id()
    assert cache.is_tag(os.path.join(tc, cache.MARKER)) and cache.is_tag(cache.common(cache.MARKER))
    assert os.path.exists(os.path.join(tc, cache.TOOLCHAIN_JSON))
    assert not os.path.exists(os.path.join(root, cache.MARKER))
    assert sorted(os.listdir(root)) == sorted([cache.COMMON, os.path.basename(tc.rstrip(os.sep))])
    _file(os.path.join(root, "other", cache.MARKER), "Signature: 8a477f597d28d172789f06886806bc55\n")
    assert not cache.is_tag(os.path.join(root, "other", cache.MARKER))  # (another program's tag)


def test_touch(root: str) -> None:
    path = _file(os.path.join(root, "f"), age=2 * cache.TOUCH_S)
    cache.touch(path)
    assert os.path.getmtime(path) > time.time() - 60
    _age(path, cache.TOUCH_S / 2)
    before = os.path.getmtime(path)
    cache.touch(path)  # (recent: not written again)
    assert os.path.getmtime(path) == before
    cache.touch(os.path.join(root, "missing"))


def test_clean_spares_what_is_not_artlite(root: str, tmp_path: object) -> None:
    """Every cleaning, ``all`` included, leaves the user's files: names no kind matches, directories without Artlite's
    tag, links (and what they point at), even where they are named like Artlite's entries."""
    outside = _sentinels(tmp_path)
    tc = cache.toolchain_dir()
    keep = [
        _file(os.path.join(root, "notes.txt"), age=OLD),
        _file(os.path.join(root, "kmath", "notes.txt"), age=OLD),  # (a legacy directory's name, not its entries)
        _file(os.path.join(root, f"fused_{HEX16}"), age=OLD),  # (a legacy build's name, but a file)
        _file(os.path.join(root, "tc-0123456789ab", f"rtc_{HEX16}", "main.pdi"), age=OLD),  # (no tag)
        _file(os.path.join(root, "common_x", "pace.json"), age=OLD),
        _file(os.path.join(tc, "notes.txt"), age=OLD),
        _file(os.path.join(tc, "pch", "notes.txt"), age=OLD),
    ]
    links = [os.path.join(root, f"rtc_{HEX16}"), os.path.join(tc, f"fused_{HEX16}"), os.path.join(root, "server"),
             os.path.join(root, "tc-fedcba987654")]
    for link in links:
        os.symlink(outside, link)
    _file(os.path.join(outside, cache.MARKER), cache.TAG)  # (a tagged directory reached through a link: not entered)
    victim = _build("rtc_" + "1" * 16, age=OLD)
    os.symlink(outside, os.path.join(victim, "link"))  # (removing an entry unlinks its links, never follows them)
    _file(os.path.join(outside, "models.json"))  # (a legacy state's path through a link: not adopted)
    for policy in (cache.Policy(unused=0.0, legacy=0.0, other_toolchains=True, size=0), cache.Policy(all=True)):
        report = cache.clean(policy)
        assert all(os.path.exists(p) for p in keep), [p for p in keep if not os.path.exists(p)]
        assert all(os.path.islink(p) for p in links)
        assert _tree(outside) >= {"precious.txt", "models.json", f"rtc_{HEX16}", os.path.join(f"rtc_{HEX16}", "main.pdi")}
        assert not os.path.exists(victim) and not report.errors
    found = cache.scan()
    assert {os.path.basename(p) for p, _ in found.skipped} >= {f"rtc_{HEX16}", "server", "tc-0123456789ab"}


def test_rules(root: str) -> None:
    """What each rule removes: unused entries, leftovers and failed builds after an hour, the trash at once; state only
    with ``all``; the size rule least recently used first, never what was used in the last hour."""
    old = _build("rtc_" + "1" * 16, age=OLD)
    recent = _build("rtc_" + "2" * 16, age=cache.DAY)
    failed = _build("fused_" + "3" * 16, age=2 * cache.TEMP_S)
    _file(os.path.join(failed, cache.IN_PROGRESS))
    _age(os.path.join(failed, cache.ENTRY_JSON), 2 * cache.TEMP_S)
    building = _build("fused_" + "4" * 16)
    _file(os.path.join(building, cache.IN_PROGRESS))
    leftover = _file(cache.toolchain_dir("pch", f"{HEX16}.tmp.pch"), age=2 * cache.TEMP_S)
    young_leftover = _file(cache.toolchain_dir("render", f"{HEX16 * 2}.json.1.2.tmp"))
    trash = _file(cache.toolchain_dir("trash", "x.1.0", "f"))
    pace = _file(cache.common("pace.json"), "{}", age=OLD)
    fit = _file(cache.common("kmath", f"silu_{HEX16}.npy"), age=OLD)
    report = cache.clean(cache.Policy())
    assert not any(os.path.exists(p) for p in (old, failed, leftover, trash, fit))
    assert all(os.path.exists(p) for p in (recent, building, young_leftover, pace))
    assert report.removed.keys() == {"unused", "failed build", "leftover", "trash"}

    lru = [_build(f"rtc_{i}" + "5" * 15, age=(3 - i) * cache.DAY) for i in range(3)]
    now = _build("rtc_" + "6" * 16)
    size = sum(e.size for e in cache.scan().entries)
    cache.clean(cache.Policy(size=size - 1))  # (the least recently used goes first)
    assert not os.path.exists(lru[0]) and all(os.path.exists(p) for p in lru[1:])
    cache.clean(cache.Policy(size=0))
    assert os.path.exists(now) and not any(os.path.exists(p) for p in lru)  # (the last hour's stays)
    cache.clean(cache.Policy(all=True))
    assert not os.path.exists(pace) and not os.listdir(root)


def test_other_toolchains(root: str) -> None:
    """A directory of another toolchain empties by the rules and goes once empty; ``other_toolchains`` empties it at
    once; this toolchain's directory stays even when empty."""
    mine = _build("rtc_" + "1" * 16)
    other = os.path.join(root, "tc-fedcba987654")
    for name in (cache.MARKER, cache.TOOLCHAIN_JSON):
        _file(os.path.join(other, name), cache.TAG if name == cache.MARKER else "{}", age=2 * cache.TEMP_S)
    entry = os.path.join(other, "rtc_" + "2" * 16)
    _file(os.path.join(entry, cache.ENTRY_JSON), "{}", age=cache.DAY)
    cache.clean(cache.Policy())
    assert os.path.exists(entry)
    cache.clean(cache.Policy(other_toolchains=True))
    assert not os.path.exists(other) and os.path.exists(mine)
    cache.clean(cache.Policy(all=True))
    assert not os.listdir(root)


def test_entry_a_build_holds_is_left(root: str) -> None:
    """A cleaning takes a build's lock without waiting: an entry whose build runs stays."""
    path = _build("rtc_" + "1" * 16, age=OLD)
    with cache.lock(os.path.basename(path)):
        report = cache.clean(cache.Policy())
    assert os.path.exists(path) and report.left == [(path, "in use")]
    cache.clean(cache.Policy())
    assert not os.path.exists(path) and not os.path.exists(cache.toolchain_dir("locks", "rtc_" + "1" * 16 + ".lock"))


def test_file_lock_survives_its_deletion(tmp_path: object) -> None:
    """A holder may delete its lock file: a waiter then locks the file the path names, not the deleted one (two
    builders of one key never run at once)."""
    path = str(tmp_path / "x.lock")
    got = threading.Event()
    seen = []

    def wait() -> None:
        with portable.file_lock(path):
            seen.append(os.path.exists(path))
            got.set()

    with portable.file_lock(path):
        t = threading.Thread(target=wait)
        t.start()
        time.sleep(0.1)
        os.remove(path)
    t.join(5)
    assert got.is_set() and seen == [True]
    with portable.file_lock(path), portable.file_lock(path, wait=False) as held:
        assert held is False


def test_legacy_layout(root: str) -> None:
    """The layout before the owned directories: its state is adopted into ``common/``, its entries go once a day
    unchanged (``legacy``) or at once (``legacy=0``)."""
    _file(os.path.join(root, "pace.json"), '{"m": 1}', age=OLD)
    _file(os.path.join(root, "server", "models.json"), "{}", age=OLD)
    _file(os.path.join(root, "ui", "b1", "index.html"))  # (a UI build: used through its index.html)
    entries = [_file(os.path.join(root, f"rtc_{HEX16}", "main.pdi")),
               _file(os.path.join(root, f"rtc_{HEX16}.lock")),
               _file(os.path.join(root, "operators", f"{HEX16}.pch")),
               _file(os.path.join(root, "render", f"{HEX16 * 2}.json")),
               _file(os.path.join(root, "minicpmv", "0123456789ab", HEX16, "state.pkl")),
               _file(os.path.join(root, f"qwen35_dump_{HEX16}"))]
    cache.clean(cache.Policy())
    with open(cache.common("pace.json")) as f:
        assert f.read() == '{"m": 1}'
    assert os.path.exists(cache.common("server", "models.json")) and os.path.exists(cache.common("ui", "b1"))
    assert all(os.path.exists(p) for p in entries)  # (younger than a day)
    cache.clean(cache.Policy(legacy=0.0))
    assert sorted(os.listdir(root)) == [cache.COMMON]


def test_dry_run_changes_nothing(root: str) -> None:
    _build("rtc_" + "1" * 16, age=OLD)
    _file(os.path.join(root, f"rtc_{HEX16}", "main.pdi"), age=OLD)
    _age(os.path.join(root, f"rtc_{HEX16}"), OLD)  # (a legacy build's last use: its directory's time)
    _file(cache.toolchain_dir("trash", "x"))
    before = _tree(root)
    report = cache.clean(cache.Policy(), dry_run=True)
    assert _tree(root) == before and report.entries == 3


def test_interrupted_removal(root: str) -> None:
    """An entry renamed into the trash and left there (a killed cleaning) goes with the next cleaning."""
    _file(cache.toolchain_dir("trash", f"rtc_{HEX16}.1.0", "design.mlir.prj", "f"))
    cache.clean(cache.Policy())
    assert not os.path.exists(cache.toolchain_dir("trash"))


def test_discard(root: str, tmp_path: object) -> None:
    outside = _sentinels(tmp_path)
    link = os.path.join(root, "link")
    os.symlink(outside, link)
    cache.discard(link)
    cache.discard(outside)
    assert os.path.islink(link) and os.path.exists(os.path.join(outside, "precious.txt"))
    d = _file(os.path.join(root, "common", "ui", "x", "f"))
    cache.discard(os.path.dirname(d))
    assert not os.path.exists(os.path.dirname(d))


def test_parse() -> None:
    assert [cache.parse_duration(t) for t in ("30m", "12h", "1d", "2w", "90", "off")] == \
        [1800, 43200, cache.DAY, 14 * cache.DAY, 90, None]
    assert [cache.parse_size(t) for t in ("500M", "30G", "1.5g", "100", "off")] == \
        [500 << 20, 30 << 30, 3 << 29, 100, None]
    for bad in ("1y", "-1d", ""):
        with pytest.raises(ValueError):
            cache.parse_duration(bad)


def test_command(root: str, capsys: pytest.CaptureFixture) -> None:
    old = _build("rtc_" + "1" * 16, age=OLD, label="MyOp")
    assert cli.main([]) == 0 and "would remove 1 entries" in capsys.readouterr().out
    assert cli.main(["list"]) == 0 and "MyOp" in capsys.readouterr().out
    assert cli.main(["clean", "-n"]) == 0 and os.path.exists(old)
    assert cli.main(["clean", "--unused-for", "1y"]) == 2
    capsys.readouterr()
    assert cli.main(["clean"]) == 0 and "removed 1 entries" in capsys.readouterr().out and not os.path.exists(old)
    assert cli.main(["path"]) == 0 and capsys.readouterr().out.strip() == cache.toolchain_dir()


def _manifest(base: str, key: str, identity: str, work: str, age: float) -> str:
    return _file(os.path.join(base, "render", f"{key * 32}.json"),
                 f'{{"work_dir": "{work}", "identity": "{identity}", "label": "L"}}', age=age)


def test_superseded(root: str) -> None:
    """Per identity of a render the most recently used manifest wins: a build only losers name goes a day after its last
    use, with the losing manifests (across toolchains too); a build a winner names, a loser used today, a manifest whose
    build is gone (``dangling``) and an old precompiled header behave as their rules say."""
    tc = cache.toolchain_dir()
    old, new = _build("rtc_" + "1" * 16, age=2 * cache.DAY), _build("rtc_" + "2" * 16)
    alone = _build("rtc_" + "3" * 16, age=2 * cache.DAY)
    shared = _build("fused_" + "4" * 16, age=2 * cache.DAY)
    today = _build("rtc_" + "5" * 16, age=cache.TEMP_S)
    m_old = _manifest(tc, "a", "A", os.path.basename(old), 2 * cache.DAY)
    m_new = _manifest(tc, "b", "A", os.path.basename(new), 0)
    m_alone = _manifest(tc, "c", "B", os.path.basename(alone), 2 * cache.DAY)
    _manifest(tc, "d", "C", os.path.basename(shared), 2 * cache.DAY)  # (C's winner and D's loser name it)
    _manifest(tc, "e", "D", os.path.basename(shared), 3 * cache.DAY)
    _manifest(tc, "f", "D", os.path.basename(today), cache.TEMP_S)
    m_today = _manifest(tc, "0", "E", os.path.basename(today), cache.TEMP_S)  # (a loser used today: kept)
    _manifest(tc, "9", "E", os.path.basename(new), 0)
    dangling = _manifest(tc, "7", "F", "rtc_" + "8" * 16, 2 * cache.TEMP_S)
    pch = _file(os.path.join(tc, "pch", f"{HEX16}.pch"), age=2 * cache.DAY)
    other = os.path.join(root, "tc-fedcba987654")
    _file(os.path.join(other, cache.MARKER), cache.TAG)
    _file(os.path.join(other, cache.TOOLCHAIN_JSON), "{}")
    other_build = _file(os.path.join(other, "fused_" + "6" * 16, cache.ENTRY_JSON), "{}", age=2 * cache.DAY)
    _manifest(other, "6", "A", "fused_" + "6" * 16, 2 * cache.DAY)
    report = cache.clean(cache.Policy())
    gone = (old, m_old, dangling, pch, os.path.dirname(other_build))
    assert not any(os.path.exists(p) for p in gone), [p for p in gone if os.path.exists(p)]
    assert all(os.path.exists(p) for p in (new, m_new, alone, m_alone, shared, today, m_today))
    assert report.removed.keys() == {"superseded", "dangling"}
    assert cache.clean(cache.Policy(superseded=None)).entries == 0


def test_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cache.parse_policy("auto") == cache.Policy() and cache.parse_policy("off") is None
    p = cache.parse_policy("superseded=4h, size=30G")
    assert (p.superseded, p.legacy, p.size, p.unused) == (4 * 3600, 4 * 3600, 30 << 30, 30 * cache.DAY)
    assert cache.parse_policy("superseded=4h,legacy=off").legacy is None
    for bad in ("sise=1G", "unused", "unused=soon"):
        with pytest.raises(ValueError):
            cache.parse_policy(bad)
    monkeypatch.setenv("ARTLITE_CACHE_GC", "off")
    assert cache.policy() == cache.Policy(interval=None)
    monkeypatch.setenv("ARTLITE_CACHE_GC", "unused=1w")
    assert cache.policy().unused == 7 * cache.DAY


def test_automatic_cleaning(root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that built something starts the cleaning at its exit, at most once per interval; the cleaning records
    its run in ``common/gc.json``; ``off`` never starts it."""
    started = []
    monkeypatch.setattr(cache.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw["env"])))
    old = _build("rtc_" + "1" * 16, age=OLD)
    cache._start_auto()
    assert started[0][0][1:] == ["-c", "from artlite import cache; cache.run_auto()"]
    assert started[0][1]["ARTLITE_CACHE_DIR"] == root
    report = cache.auto()
    assert report.entries == 1 and not os.path.exists(old)
    assert cache.last_auto()["removed"] == {"unused": [1, report.bytes]}
    assert cache.auto() is None  # (not due for a day)
    cache._start_auto()
    assert len(started) == 1
    monkeypatch.setenv("ARTLITE_CACHE_GC", "interval=0")
    assert cache.auto() is not None
    monkeypatch.setenv("ARTLITE_CACHE_GC", "off")
    os.remove(cache.common("gc.json"))
    cache._start_auto()
    assert cache.auto() is None and len(started) == 1
