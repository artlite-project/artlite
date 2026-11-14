"""The environment switches (``artlite.settings``): the registry is the only reader and namer of ``ARTLITE_*``
variables, the documentation table is its table, an undeclared variable is warned about, and each rule parses as its
readers always did. Host-only."""

import ast
import os
import re
import warnings
from pathlib import Path

import pytest

from artlite import settings

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TOKEN = re.compile(r"ARTLITE_[A-Z0-9_]*")


def test_no_module_names_a_variable() -> None:
    """Outside settings.py no string of src/ is an ``ARTLITE_*`` variable's name or prefix: a module takes its
    switch's object (``settings.PACE.get()``, ``.name``), so every read has one parse and one default."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "artlite" / "settings.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and TOKEN.fullmatch(node.value):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value!r}")
    assert not found, "ARTLITE_* names outside artlite/settings.py:\n" + "\n".join(found)


def test_registry_is_consistent() -> None:
    names = [s.name for s in settings.SWITCHES]
    assert len(names) == len(set(names))
    assert all(n.startswith(settings.PREFIX) and n == n.upper() for n in names)
    assert all(s.doc for s in settings.SWITCHES), "every switch is documented"
    assert all(s.why for s in settings.SWITCHES if s.read != "call"), "a read at another time says why"
    assert {s.name for s in settings.BUILD_SWITCHES} == {"ARTLITE_IGNORE_CACHE", "ARTLITE_MEASURE_II",
                                                          "ARTLITE_GENERATE_ASM"}
    members = [s for s in settings.SWITCHES if s.family is settings.PROFILE_FAMILY]
    assert members and all(settings.PROFILE_FAMILY.owns(s.name) for s in members)
    assert settings.PROFILE_FAMILY.owns(settings.PROFILE.name)


def test_documentation_table_is_the_registry() -> None:
    text = (ROOT / "docs" / "development.md").read_text()
    match = re.search(r"<!-- settings -->\n(.*?)\n<!-- /settings -->", text, re.DOTALL)
    assert match, "docs/development.md: the table between <!-- settings --> and <!-- /settings -->"
    assert match[1] == settings.table(), (
        "docs/development.md's table differs from the registry: replace it with\n" + settings.table())
    for s in settings.SWITCHES:
        assert f"`{s.name}`" in match[1]


def test_unknown_variable_is_named_with_the_closest_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"ARTLITE_PACEE": "1", "ARTLITE_PROFILE_TILE": "all", "ARTLITE_ARG_CTX_SIZE": "4096", "ARTLITE_PACE": "off",
           "ARTLITE_QQQQQQQQQ": "1", "HOME": "/"}
    assert settings.unknown(env) == {"ARTLITE_PACEE": "ARTLITE_PACE", "ARTLITE_PROFILE_TILE": "ARTLITE_PROFILE_TILES",
                                     "ARTLITE_QQQQQQQQQ": None}
    for key in [k for k in os.environ if settings.is_artlite(k)]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("ARTLITE_PACEE", "1")
    monkeypatch.setattr(settings, "_warned", False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        settings.warn_unknown()
        settings.warn_unknown()
    assert [str(w.message) for w in seen if "PACEE" in str(w.message)] == [
        "ARTLITE_PACEE is set, but Artlite has no such switch (did you mean ARTLITE_PACE?)"], "warned once"
    monkeypatch.delenv("ARTLITE_PACEE")
    monkeypatch.setattr(settings, "_warned", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settings.warn_unknown()


@pytest.mark.parametrize("value, flag_one, off_when_zero, convert", [
    (None, False, True, True), ("", False, True, False), ("0", False, False, False), ("1", True, True, True),
    ("true", False, True, False), ("other", False, True, False)])
def test_flag_rules(value: str | None, flag_one: bool, off_when_zero: bool, convert: bool) -> None:
    """``1`` turns most flags on (nothing else does); ``PM_PACKETS`` is on unless ``0``; ``PROFILE_CONVERT`` is on
    unset and when ``1``."""
    assert settings.IGNORE_CACHE.parse(value) is flag_one
    assert settings.PM_PACKETS.parse(value) is off_when_zero
    assert settings.PROFILE_CONVERT.parse(value) is convert


def test_value_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ARTLITE_COMPILE_JOBS", "ARTLITE_ROUTER_MODELS", "ARTLITE_PROFILE_TRACE_SIZE", "ARTLITE_PACE",
                 "ARTLITE_BACKEND", "ARTLITE_MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert settings.COMPILE_JOBS.get() is None
    monkeypatch.setenv("ARTLITE_COMPILE_JOBS", "3")
    assert settings.COMPILE_JOBS.get() == 3
    for bad in ("0", "", "-1", "two"):
        monkeypatch.setenv("ARTLITE_COMPILE_JOBS", bad)
        with pytest.raises(ValueError, match="ARTLITE_COMPILE_JOBS.*positive integer"):
            settings.COMPILE_JOBS.get()
    assert settings.ROUTER_MODELS.get() == 1
    monkeypatch.setenv("ARTLITE_ROUTER_MODELS", "")
    assert settings.ROUTER_MODELS.get() == 1
    monkeypatch.setenv("ARTLITE_ROUTER_MODELS", "2")
    assert settings.ROUTER_MODELS.get() == 2
    monkeypatch.setenv("ARTLITE_PROFILE_TRACE_SIZE", "x")
    with pytest.raises(ValueError, match="ARTLITE_PROFILE_TRACE_SIZE: expected an integer"):
        settings.PROFILE_TRACE_SIZE.get()
    assert settings.PACE.get() == "auto"
    monkeypatch.setenv("ARTLITE_PACE", "")
    assert settings.PACE.get() == ""
    monkeypatch.setenv("ARTLITE_BACKEND", "")
    assert settings.BACKEND.get() == "", "empty: the platform's own"
    monkeypatch.setenv("ARTLITE_MODELS_DIR", "")
    assert settings.MODELS_DIR.get() == "", "set and empty stays empty"


def test_without_and_take(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"ARTLITE_PROFILE": "1", "ARTLITE_PROFILE_DIR": "d", "ARTLITE_MEASURE_II": "1", "ARTLITE_PACE": "off", "A": "b"}
    assert settings.without(env, settings.PROFILE_FAMILY, settings.MEASURE_II) == {"ARTLITE_PACE": "off", "A": "b"}
    assert settings.arg_name("CTX_SIZE") == "ARTLITE_ARG_CTX_SIZE"
    assert settings.arg_option("ARTLITE_ARG_CTX_SIZE") == "ctx_size"
    assert [settings.arg_flag(v) for v in ("1", "TRUE", "on", "Yes", "enabled", "0", "", "no")] == [True] * 5 + [False] * 3
    monkeypatch.setenv("ARTLITE_DIAGNOSE_ROOT", "/tmp/f")
    assert settings.DIAGNOSE_ROOT.take() == "/tmp/f"
    assert settings.DIAGNOSE_ROOT.take() == "" and "ARTLITE_DIAGNOSE_ROOT" not in os.environ
