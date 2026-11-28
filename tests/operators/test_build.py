"""The builds (operators/build.py): one content-only cache key per spec, artifacts published whole (no NPU, no
toolchain run: the tools are replaced by fakes)."""

import dataclasses
import json
import os
import time

import pytest

from artlite import cache
from artlite.operators import build, toolchain
from artlite.operators.build import IN_PROGRESS, BuildKind, BuildSpec

RTC = BuildSpec(kind=BuildKind.RTC, label="T", source="void f() {}\n", mlir="module {}\n")
FUSED = BuildSpec(kind=BuildKind.FUSED, label="F", source="void f() {}\n", mlir="module {}\n", expand_load_pdis=True)


def _toolchain_at(monkeypatch: pytest.MonkeyPatch, venv: str, checkout: str, *, flag: str = "-O2") -> None:
    """A toolchain installed in ``venv`` and Artlite checked out at ``checkout`` (no toolchain needed)."""
    monkeypatch.setattr(toolchain, "peano_dir", lambda: f"{venv}/llvm-aie")
    monkeypatch.setattr(toolchain, "mlir_aie_dir", lambda: f"{venv}/mlir_aie")
    monkeypatch.setattr(toolchain, "INCLUDE_DIR", f"{checkout}/include")
    monkeypatch.setattr(toolchain, "clang_cmd_common", lambda: [
        f"{venv}/llvm-aie/bin/clang++", flag, f"-I{checkout}/include", f"-I{venv}/mlir_aie/include"])


@pytest.fixture
def compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compile command whose paths are the only thing a test changes."""
    monkeypatch.setattr(build, "headers_key", lambda: "headers")
    _toolchain_at(monkeypatch, "/venv", "/checkout")


def test_key_fields(compiler: None) -> None:
    """Every field of a spec that reaches the build is in its key (a field that changed what is built without changing
    the key would serve a stale build)."""
    changes = {
        "source": "void g() {}\n", "mlir": "module { }\n", "clang_flags": ("-DX=1",), "aiecc_flags": ("--x",),
        "profile_key": ("counters",),
    }
    for spec in (RTC, FUSED):
        base = spec.cache_key()
        assert spec.cache_key() == base
        for name, value in {**changes, "expand_load_pdis": not spec.expand_load_pdis}.items():
            assert dataclasses.replace(spec, **{name: value}).cache_key() != base, (spec.kind, name)
    assert RTC.cache_key() != dataclasses.replace(FUSED, expand_load_pdis=False).cache_key()  # (the kinds' outputs)
    assert dataclasses.replace(RTC, label="U").cache_key() == RTC.cache_key()  # (the label names, not builds)
    assert dataclasses.replace(RTC, indirect_calls=(("a", "b"),)).cache_key() == RTC.cache_key()  # (a check's)


def test_key_is_content_only(compiler: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The key holds no paths the cache covers otherwise and no modification times: another checkout or venv with the
    same headers, a moved cache or a reinstalled toolchain finds the same builds; the headers' contents and the flags
    count, and so does another compiler or include directory (by path: a prototype llvm-aie's builds stay apart)."""
    base = RTC.cache_key()
    _toolchain_at(monkeypatch, "/other-venv", "/elsewhere")
    monkeypatch.setattr(cache, "ROOT", "/moved")
    assert RTC.cache_key() == base
    _toolchain_at(monkeypatch, "/other-venv", "/elsewhere", flag="-O3")
    assert RTC.cache_key() != base
    _toolchain_at(monkeypatch, "/venv", "/checkout")
    assert RTC.cache_key() == base
    monkeypatch.setattr(toolchain, "clang_cmd_common", lambda: ["/proto/bin/clang++", "-O2"])
    other = RTC.cache_key()
    monkeypatch.setattr(toolchain, "clang_cmd_common", lambda: ["/venv/llvm-aie/bin/clang++", "-O2", "-I/exp/inc"])
    assert len({base, other, RTC.cache_key()}) == 3
    _toolchain_at(monkeypatch, "/venv", "/checkout")
    monkeypatch.setattr(build, "headers_key", lambda: "other headers")
    assert RTC.cache_key() != base


class _Dist:
    def __init__(self, version: str, record: str) -> None:
        self.version, self._record = version, record

    def read_text(self, name: str) -> str | None:
        return self._record if name == "RECORD" else None


def test_toolchain_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The toolchain's directory is named by the wheels' versions and files: not by the venv (entry scripts, the
    installer's files), so a reinstall or another venv finds it; a wheel with other files gets another."""
    record = ("llvm-aie/bin/clang++,sha256=AAA,221704\n../../../bin/aiecc,sha256=VENV1,314\n"
              "x.dist-info/INSTALLER,sha256=uv,2\nx.dist-info/RECORD,,\nmod/__pycache__/a.pyc,,\n")
    dists = {"llvm_aie": _Dist("22.0", record), "mlir_aie": _Dist("1.4", record)}
    monkeypatch.setattr(cache.metadata, "distribution", lambda name: dists[name])

    def tc_id() -> str:
        cache._toolchain.cache_clear()
        return cache.toolchain_id()

    try:
        base = tc_id()
        dists["mlir_aie"] = _Dist("1.4", record.replace("VENV1", "VENV2").replace("uv", "pip"))
        assert tc_id() == base
        dists["mlir_aie"] = _Dist("1.4", record.replace("AAA", "BBB"))
        assert tc_id() != base
        dists["mlir_aie"] = _Dist("1.5", record)
        assert tc_id() != base
    finally:
        cache._toolchain.cache_clear()
    assert cache.record_files(record) == ["llvm-aie/bin/clang++,sha256=AAA,221704"]


class FakeTools:
    """The build tools: clang writes its ``-o`` file (and its ``.su`` next to it), aiecc the pdi and the insts, after
    ``link_files``; ``kill`` names a step at which the tool dies, leaving what it had written."""

    def __init__(self) -> None:
        self.kill: str | None = None
        self.aiecc_runs = 0
        self.clang_runs = 0

    def run(self, cmd: list[str], cwd: str | None = None, capture: bool = True) -> None:
        if "-o" not in cmd:
            return
        out = cmd[cmd.index("-o") + 1]
        self.clang_runs += 1
        with open(out, "w") as f:
            f.write("partial" if self.kill == "clang" else "object")
        if self.kill == "clang":
            raise KeyboardInterrupt
        if "-fstack-usage" in cmd:
            with open(out[: -len(".o")] + ".su", "w") as f:
                f.write("")

    def run_aiecc(self, cmd: list[str], *, cwd: str, link_files: object) -> None:
        self.aiecc_runs += 1
        link_files()
        with open(os.path.join(cwd, "main.pdi"), "w") as f:
            f.write("partial" if self.kill == "aiecc" else "pdi")
        if self.kill == "aiecc":
            raise KeyboardInterrupt
        with open(os.path.join(cwd, "insts.bin"), "w") as f:
            f.write("insts")


@pytest.fixture
def tools(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> FakeTools:
    fake = FakeTools()
    tool = tmp_path / "tool"
    tool.write_text("")
    for name, value in {"aiecc": lambda: str(tool), "clangpp": lambda: str(tool), "peano_dir": lambda: "/peano",
                        "clang_cmd_common": lambda: ["clang++"], "run": fake.run, "run_aiecc": fake.run_aiecc}.items():
        monkeypatch.setattr(toolchain, name, value)
    monkeypatch.setattr(cache, "ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ARTLITE_SKIP_CHECKS", "1")
    monkeypatch.setenv("ARTLITE_IGNORE_CACHE", "0")
    return fake


def test_build_caches(tools: FakeTools) -> None:
    for spec, outputs in ((RTC, ("main.pdi", "insts.bin")), (FUSED, ("F.elf",))):
        tools.aiecc_runs = 0
        work = build.build(spec)
        if spec is FUSED:  # (the fake aiecc writes the rtc outputs: the fused one's ELF)
            open(os.path.join(work, "F.elf"), "w").close()
        assert os.path.exists(os.path.join(work, "kernel.o")) and os.path.exists(os.path.join(work, "kernel.su"))
        assert not os.path.exists(os.path.join(work, IN_PROGRESS)), spec.kind
        assert build.build(spec) == work and tools.aiecc_runs == 1
        assert all(os.path.exists(os.path.join(work, f)) for f in outputs)
        assert os.path.dirname(work) == cache.toolchain_dir().rstrip(os.sep)


def test_build_entry(tools: FakeTools, monkeypatch: pytest.MonkeyPatch) -> None:
    """A build directory says what it holds (``entry.json``), a hit records its use, and aiecc's unstamped copy of a
    full ELF is not kept."""
    def run_aiecc(cmd: list[str], *, cwd: str, link_files: object) -> None:
        FakeTools.run_aiecc(tools, cmd, cwd=cwd, link_files=link_files)
        os.makedirs(os.path.join(cwd, "design.mlir.prj"), exist_ok=True)
        for name in ("F.elf", "design.mlir.prj/F.elf.unstamped"):
            open(os.path.join(cwd, name), "w").close()

    monkeypatch.setattr(toolchain, "run_aiecc", run_aiecc)
    work = build.build(FUSED)
    entry = os.path.join(work, cache.ENTRY_JSON)
    with open(entry) as f:
        assert {k: v for k, v in json.load(f).items() if k != "created"} == {"kind": "fused", "label": "F"}
    assert not os.path.exists(os.path.join(work, "design.mlir.prj", "F.elf.unstamped"))
    old = time.time() - 2 * cache.TOUCH_S
    os.utime(entry, (old, old))
    build.build(FUSED)
    assert os.path.getmtime(entry) > old + cache.TOUCH_S


def test_rebuild_drops_packed_temps(tools: FakeTools, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory built again loses the archive of its earlier build's intermediates (``cache.pack``)."""
    work = build.build(RTC)
    open(os.path.join(work, cache.TEMPS), "w").close()
    monkeypatch.setenv("ARTLITE_IGNORE_CACHE", "1")
    build.build(RTC)
    assert not os.path.exists(os.path.join(work, cache.TEMPS))


def test_killed_clang_leaves_no_object(tools: FakeTools, tmp_path: object) -> None:
    """A clang that dies mid-write leaves nothing the next build takes for the object."""
    build.compile_pch(clang_cmd=RTC.clang_cmd)
    tools.kill = "clang"
    with pytest.raises(KeyboardInterrupt):
        build.build(RTC)
    (work,) = [d for d in os.listdir(cache.toolchain_dir()) if d.startswith("rtc_")]
    files = os.listdir(cache.toolchain_dir(work))
    assert "kernel.o" not in files and not [f for f in files if f.endswith((".tmp.o", ".tmp.su"))], files
    tools.kill = None
    assert os.path.exists(os.path.join(build.build(RTC), "kernel.o"))


def test_killed_aiecc_is_built_again(tools: FakeTools) -> None:
    """The outputs of an aiecc that died are not cached: the directory has the in-progress mark until a build of it
    succeeded (the partial main.pdi is no cache hit)."""
    tools.kill = "aiecc"
    with pytest.raises(KeyboardInterrupt):
        build.build(RTC)
    tools.kill = None
    work = build.build(RTC)
    with open(os.path.join(work, "main.pdi")) as f:
        assert tools.aiecc_runs == 2 and f.read() == "pdi"
    assert not os.path.exists(os.path.join(work, IN_PROGRESS))
    build.build(RTC)
    assert tools.aiecc_runs == 2


def test_killed_pch_is_built_again(tools: FakeTools, tmp_path: object) -> None:
    tools.kill = "clang"
    with pytest.raises(KeyboardInterrupt):
        build.compile_pch(clang_cmd=["clang++"])
    pchs = os.listdir(cache.toolchain_dir("pch"))
    assert not [f for f in pchs if f.endswith(".pch") and not f.endswith(".tmp.pch")], pchs
    tools.kill = None
    assert os.path.exists(build.compile_pch(clang_cmd=["clang++"]))
