"""The builds (operators/build.py): one cache key per spec, artifacts published whole (no NPU, no toolchain run: the
tools are replaced by fakes)."""

import dataclasses
import os
from importlib.metadata import version

import pytest

from artlite.operators import build, toolchain
from artlite.operators.build import IN_PROGRESS, BuildKind, BuildSpec

RTC = BuildSpec(kind=BuildKind.RTC, label="T", source="void f() {}\n", mlir="module {}\n")
FUSED = BuildSpec(kind=BuildKind.FUSED, label="F", source="void f() {}\n", mlir="module {}\n", expand_load_pdis=True)
PCH = "/cache/operators/abc.pch"


def test_key_fields() -> None:
    """Every field of a spec that reaches the build is in its key (a field that changed what is built without changing
    the key would serve a stale build)."""
    changes = {
        "source": "void g() {}\n", "mlir": "module { }\n", "clang_flags": ("-DX=1",), "aiecc_flags": ("--x",),
        "profile_key": ("counters",), "expand_load_pdis": not FUSED.expand_load_pdis,
    }
    for spec in (RTC, FUSED):
        base = spec.cache_key(PCH)
        assert spec.cache_key(PCH) == base and spec.cache_key("/cache/operators/other.pch") != base
        for name, value in changes.items():
            if (spec.kind, name) in ((BuildKind.FUSED, "profile_key"), (BuildKind.RTC, "expand_load_pdis")):
                continue  # (a fused design is not profiled; the rtc build has no PDI expansion)
            assert dataclasses.replace(spec, **{name: value}).cache_key(PCH) != base, (spec.kind, name)
    assert dataclasses.replace(RTC, label="U").cache_key(PCH) == RTC.cache_key(PCH)  # (the label names, not builds)
    assert dataclasses.replace(RTC, indirect_calls=(("a", "b"),)).cache_key(PCH) == RTC.cache_key(PCH)  # (a check's)


def test_keys_unchanged() -> None:
    """The key lists of the spec kinds: the builds a cache holds stay found (a change here rebuilds every design)."""
    flags = toolchain.CORE_LLC_FLAGS
    assert RTC.cache_key(PCH) == toolchain.cache_key(RTC.source, RTC.mlir, PCH, None, flags)
    assert FUSED.cache_key(PCH) == toolchain.cache_key(FUSED.source, FUSED.mlir, PCH, version("mlir_aie"),
                                                       version("llvm_aie"), True, flags)
    rtc = dataclasses.replace(RTC, clang_flags=("-DX",), aiecc_flags=("--y",), profile_key=("p",))
    assert rtc.cache_key(PCH) == toolchain.cache_key(rtc.source, "-DX", rtc.mlir, PCH, ("p",), flags, "--y")
    assert RTC.cache_key(PCH) == "38bc0566d75549e2"  # (the key of the formula before the spec owned it)


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
    monkeypatch.setattr(build, "CACHE_DIR", str(tmp_path / "cache"))
    os.makedirs(tmp_path / "cache")
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


def test_killed_clang_leaves_no_object(tools: FakeTools, tmp_path: object) -> None:
    """A clang that dies mid-write leaves nothing the next build takes for the object."""
    build.compile_pch(clang_cmd=RTC.clang_cmd)
    tools.kill = "clang"
    with pytest.raises(KeyboardInterrupt):
        build.build(RTC)
    (work,) = [d for d in os.listdir(tmp_path / "cache") if d.startswith("rtc_") and not d.endswith(".lock")]
    files = os.listdir(tmp_path / "cache" / work)
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
    pchs = os.listdir(tmp_path / "cache" / "operators")
    assert not [f for f in pchs if f.endswith(".pch") and not f.endswith(".tmp.pch")], pchs
    tools.kill = None
    assert os.path.exists(build.compile_pch(clang_cmd=["clang++"]))
