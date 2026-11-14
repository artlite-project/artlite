"""The environment switches: every ``ARTLITE_*`` variable Artlite reads, declared once.

A ``Switch`` is a variable's name, kind (flag, int, path, choice, string), default, one-line documentation and read
semantics: ``get()`` parses the variable as the switch's own rules say (each keeps the parse its readers had: which
values turn a flag on, how an int or a path is read). Reads happen at the call (a test's ``monkeypatch.setenv``
holds), except where a declaration says ``read`` otherwise: a value the process fixes at import (the models directory,
the build cache, the backend) or at the first use (the profiling configuration).

Nothing else in ``src/`` reads or names an ``ARTLITE_*`` variable (``tests/test_settings.py`` scans for it): a module
takes the switch object, ``settings.PACE.get()``, and a child process's environment is built from ``name`` and
``without``. ``Family`` declares a prefix of variables: ``ARG`` (``artlite-server``'s options, one per option, open: the
server's option table names them) and ``PROFILE`` (the profiling configuration, closed: its members are switches).
``warn_unknown`` (``import artlite``) names the closest declared switch of a variable that none declares. The table of
``docs/development.md`` is ``table()`` (``tests/test_settings.py`` compares and prints the fix).

This module imports nothing of ``artlite``.
"""

import difflib
import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

PREFIX = "ARTLITE_"
Kind = Literal["flag", "int", "path", "choice", "string"]
Read = Literal["call", "import", "first use"]


@dataclass(frozen=True)
class Family:
    """The variables that start with ``prefix``. ``open``: any name after it is valid (``warn_unknown``); else the
    members are the switches that name it as their ``family``."""

    prefix: str
    doc: str
    open: bool

    def owns(self, name: str) -> bool:
        return name.startswith(self.prefix)


@dataclass(frozen=True)
class Switch[T]:
    """One variable. ``default``: its documented text; ``parse``: the raw value (None: unset) -> its value.
    ``affects_build``: the switch changes what a build produces or prints (``render_cache`` skips its manifest
    while one is on). ``read``: when the process reads it (the module docstring); ``why``: for a read other than at the
    call, the reason. ``native``: the C++ module reads it, not Python."""

    name: str
    kind: Kind
    default: str
    doc: str
    parse: Callable[[str | None], T]
    affects_build: bool = False
    read: Read = "call"
    why: str = ""
    native: bool = False
    family: Family | None = None

    def raw(self, environ: Mapping[str, str] | None = None) -> str | None:
        """The variable's text, None when it is unset."""
        return (os.environ if environ is None else environ).get(self.name)

    def get(self, environ: Mapping[str, str] | None = None) -> T:
        """The switch's value; ``environ``: a mapping to read instead of the process's environment."""
        return self.parse(self.raw(environ))

    def take(self) -> T:
        """The value, removing the variable from the process's environment (a one-shot hand-over the process's own
        children must not inherit)."""
        value = self.get()
        os.environ.pop(self.name, None)
        return value


# ---------------------------------------------------------------------------
# Parses: one per rule a variable is read by
# ---------------------------------------------------------------------------


def _on_when_one(raw: str | None) -> bool:
    """A flag that is off unless it is ``1``."""
    return raw == "1"


def _off_when_zero(raw: str | None) -> bool:
    """A flag that is on unless it is ``0``."""
    return raw != "0"


def _on_unless(default: bool) -> Callable[[str | None], bool]:
    """A flag whose value is ``default`` when unset and whether it is ``1`` otherwise."""
    return lambda raw: default if raw is None else raw == "1"


def _text(default: str) -> Callable[[str | None], str]:
    """The text, ``default`` when unset (set and empty: empty)."""
    return lambda raw: default if raw is None else raw


def _text_or(default: str) -> Callable[[str | None], str]:
    """The text, ``default`` when unset or empty."""
    return lambda raw: raw or default


def _home_path(*parts: str) -> Callable[[str | None], str]:
    """The path, a directory under the home directory (``parts``) when unset."""
    return lambda raw: raw if raw is not None else os.path.join(os.path.expanduser("~"), *parts)


def _lower(default: str) -> Callable[[str | None], str]:
    return lambda raw: (default if raw is None else raw).lower()


def _integer(name: str, default: int) -> Callable[[str | None], int]:
    def parse(raw: str | None) -> int:
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{name}: expected an integer, got {raw!r}") from None
    return parse


def _positive(name: str) -> Callable[[str | None], int | None]:
    """A positive integer (digits only), None when unset."""
    def parse(raw: str | None) -> int | None:
        if raw is None:
            return None
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{name}={raw!r}: expected a positive integer")
        return int(raw)
    return parse


def _count(raw: str | None) -> int:
    """An integer, 1 when unset or empty."""
    return int(raw or 1)


def _present(raw: str | None) -> bool:
    return raw is not None


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

OFF = "off; `1` turns it on"
ON = "on; `0` turns it off"

SWITCHES: list[Switch[Any]] = []
FAMILIES: list[Family] = []


def _declare[T](name: str, kind: Kind, default: str, doc: str, parse: Callable[[str | None], T],
                **fields: Any) -> Switch[T]:
    switch = Switch(PREFIX + name, kind, default, doc, parse, **fields)
    SWITCHES.append(switch)
    return switch


def _family(prefix: str, doc: str, *, open: bool) -> Family:
    family = Family(PREFIX + prefix, doc, open)
    FAMILIES.append(family)
    return family


# Where files live
MODELS_DIR = _declare(
    "MODELS_DIR", "path", "~/.artlite/models", "the models directory",
    _home_path(".artlite", "models"), read="import",
    why="``model_config.MODELS_DIR`` is a constant of the models' paths")
LLAMA_CPP_DIR = _declare(
    "LLAMA_CPP_DIR", "path", "/src/llama.cpp", "the llama.cpp build the Qwen reference tests load",
    _text("/src/llama.cpp"), read="import", why="``model_config.LLAMA_CPP_DIR`` is a constant")
CACHE_DIR = _declare(
    "CACHE_DIR", "path", "~/.cache/artlite", "the build cache",
    _home_path(".cache", "artlite"), read="import",
    why="the cache's paths are module constants (``toolchain.CACHE_DIR``)")
BACKEND = _declare(
    "BACKEND", "choice", "the platform's own (xdna on Linux, xrt on Windows)",
    "the NPU backend: ``xdna`` (the amdxdna driver) or ``xrt`` (XRT)",
    _text_or(""), read="import", why="the backend module is ``artlite._native``, imported with the package")

# Builds
COMPILE_JOBS = _declare(
    "COMPILE_JOBS", "int", "the CPU count", "concurrent design builds (1: serial)", _positive("ARTLITE_COMPILE_JOBS"))
IGNORE_CACHE = _declare(
    "IGNORE_CACHE", "flag", OFF, "rebuild every design instead of loading it from the cache",
    _on_when_one, affects_build=True)
LOG_COMMANDS = _declare("LOG_COMMANDS", "flag", OFF, "print every Peano and aiecc command line", _on_when_one)
GENERATE_ASM = _declare(
    "GENERATE_ASM", "flag", OFF, "also write each built kernel's assembly (``kernel.cc.s`` in its cache directory)",
    _on_when_one, affects_build=True)
MEASURE_II = _declare(
    "MEASURE_II", "flag", OFF, "print the per-loop schedule report of every kernel compiled (measure-ii skill)",
    _on_when_one, affects_build=True)
SKIP_CHECKS = _declare(
    "SKIP_CHECKS", "flag", OFF, "skip the build-time design checks (stack depth, kernel signatures, rounding mode)",
    _on_when_one)
DEBUG_CLANG = _declare(
    "DEBUG_CLANG", "path", "llvm-aie's build directory next to the checkout",
    "an assertions build of llvm-aie's ``clang++``, for ``variant_search``'s post-pipeliner traces", _text_or(""))
PLANNING = _declare(
    "PLANNING", "flag", OFF, "a planning process: ``compile()`` hands its spec to the precompiler instead of building "
    "(set by ``python -m artlite.parallel_compile``)", _on_when_one)
PLAN_SINK = _declare(
    "PLAN_SINK", "string", "", "``<key>:<address>`` of the precompiler's listener that planning processes send to "
    "(set with ``ARTLITE_PLANNING``)", _text(""))

# Runtime
VERIFY_TXN = _declare(
    "VERIFY_TXN", "flag", OFF, "check every host-built transaction where it is built or loaded (``txn.verify``: "
    "AxQoS 0, valid BDs, token controllers, the shim channels' queues; the tests turn it on)", _on_when_one)
PM_PACKETS = _declare(
    "PM_PACKETS", "flag", ON, "a library configuration's tile writes go as control packets where that is cheaper "
    "(off: all in the transaction, and no configuration without the array reset)", _off_when_zero)
PDI_RESET = _declare(
    "PDI_RESET", "flag", OFF, "every library configuration with the array reset (none without it after another of "
    "the same piece)", _on_when_one)
PACE = _declare(
    "PACE", "string", "auto", "the prefill GEMMs' pace levels: ``auto`` (calibrated per machine and cached), ``off``, "
    "``recalibrate``, or levels (``gu=3,qkv=2``, or one level for every class)", _text("auto"))
NB_LEAK_WARNINGS = _declare(
    "NB_LEAK_WARNINGS", "flag", "off; any value turns it on",
    "nanobind's leak warnings at exit (set to anything to show them)", _present, native=True)

# The server
ROUTER_MODELS = _declare(
    "ROUTER_MODELS", "int", "1", "a router's model server: the router's ``--models-max``, which divides the default "
    "prompt cache's part of the RAM (set by the router)", _count)
ARG = _family("ARG_", "``artlite-server``'s options, one variable per option (``ARTLITE_ARG_CTX_SIZE``)", open=True)

# Profiling
PROFILE_FAMILY = _family("PROFILE", "the profiling configuration", open=False)
PROFILE = _declare(
    "PROFILE", "choice", "0", "on-NPU profiling: ``1`` / ``counters`` (event-window counters) or ``trace`` (stream "
    "traces); ``0``, ``off``, ``false``, ``no`` or empty: off (profiling skill)", _lower("0"), read="first use",
    why="the profiling configuration is built once per process (``profiling.get_config``)", family=PROFILE_FAMILY)


def _profile[T](name: str, kind: Kind, default: str, doc: str, parse: Callable[[str | None], T]) -> Switch[T]:
    return _declare(f"PROFILE_{name}", kind, default, doc, parse, read="first use", family=PROFILE_FAMILY,
                    why="the profiling configuration is built once per process (``profiling.get_config``)")


PROFILE_DIR = _profile("DIR", "path", "./artlite_profile", "profiling: the output directory", _text("artlite_profile"))
PROFILE_COUNTERS = _profile(
    "COUNTERS", "string", "core 33:34, 34:33, 33:33, 34:34",
    "profiling: counter event pairs, ``[module:]start:stop`` comma-separated (modules ``m``, ``mt``, ``sh``)",
    _text_or(""))
PROFILE_EVENTS = _profile(
    "EVENTS", "string", "", "trace mode: core event names or hex ids, comma-separated", _text_or(""))
PROFILE_TRACE_SIZE = _profile(
    "TRACE_SIZE", "int", "4194304", "trace mode: the trace buffer's size in bytes",
    _integer(PREFIX + "PROFILE_TRACE_SIZE", 4 * 1024 * 1024))
PROFILE_TILES = _profile(
    "TILES", "string", "all", "trace mode: the traced tiles: ``all``, ``first:N`` or ``;``-separated ``col,row``",
    _text("all"))
PROFILE_LATERAL = _profile(
    "LATERAL", "flag", OFF, "trace mode: pre-lower with lateral routing (a spare shim)", _on_when_one)
PROFILE_DISTRIBUTE = _profile(
    "DISTRIBUTE", "flag", OFF, "trace mode: pre-lower with distributed channels (two S2MM)", _on_when_one)
PROFILE_CONVERT = _profile(
    "CONVERT", "flag", "on; any value but `1` turns it off", "convert the ``.ttbin`` dumps to JSON at exit",
    _on_unless(True))
PROFILE_SAMPLE_EVERY = _profile(
    "SAMPLE_EVERY", "int", "1", "counters mode: sample every Nth dispatch (at least 1)",
    _integer(PREFIX + "PROFILE_SAMPLE_EVERY", 1))

# The diagnosis
DIAGNOSE_ROOT = _declare(
    "DIAGNOSE_ROOT", "path", "", "``artlite-probe --diagnose``: the file of the root readings that the sudo user's "
    "run continues with (set by the tool)", _text(""))

BUILD_SWITCHES = tuple(s for s in SWITCHES if s.affects_build)  # (``render_cache``: while one is on, no manifest)


def without(environ: Mapping[str, str], *drop: Switch[Any] | Family) -> dict[str, str]:
    """A copy of ``environ`` without the switches and families in ``drop`` (a child process's environment)."""
    names = {d.name for d in drop if isinstance(d, Switch)}
    families = [d for d in drop if isinstance(d, Family)]
    return {k: v for k, v in environ.items() if k not in names and not any(f.owns(k) for f in families)}


def is_artlite(name: str) -> bool:
    """Whether the variable is one of Artlite's (the user's settings a child process keeps)."""
    return name.startswith(PREFIX)


# ---------------------------------------------------------------------------
# The families' variables
# ---------------------------------------------------------------------------


def arg_name(suffix: str) -> str:
    """The variable of a server option: ``suffix`` is the long option's name in upper case, dashes as underscores."""
    return ARG.prefix + suffix


def arg_option(name: str) -> str:
    """The option's name (lower case, underscores) of an ``ARTLITE_ARG_`` variable."""
    return name[len(ARG.prefix):].lower()


def arg_read(name: str) -> str | None:
    """The text of an ``ARTLITE_ARG_`` variable, None when it is unset."""
    return os.environ.get(name)


def arg_flag(value: str) -> bool:
    """Whether an ``ARTLITE_ARG_`` value of a switch option turns it on."""
    return value.lower() in ("1", "true", "on", "yes", "enabled")


# ---------------------------------------------------------------------------
# Checks and the documentation
# ---------------------------------------------------------------------------

_warned = False


def unknown(environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """The ``ARTLITE_*`` variables of the environment that no switch or open family declares, each with the closest
    declared name (None: nothing close)."""
    declared = {s.name for s in SWITCHES}
    open_prefixes = tuple(f.prefix for f in FAMILIES if f.open)
    out: dict[str, str | None] = {}
    for key in sorted(os.environ if environ is None else environ):
        if is_artlite(key) and key not in declared and not key.startswith(open_prefixes):
            close = difflib.get_close_matches(key, sorted(declared), n=1, cutoff=0.6)
            out[key] = close[0] if close else None
    return out


def warn_unknown() -> None:
    """Warn once per process about each ``ARTLITE_*`` variable no switch declares (a misspelled one is otherwise
    ignored)."""
    global _warned
    if _warned:
        return
    _warned = True
    for key, close in unknown().items():
        hint = f"did you mean {close}?" if close else "docs/development.md lists the switches"
        warnings.warn(f"{key} is set, but Artlite has no such switch ({hint})", stacklevel=2)


def table() -> str:
    """The markdown table of ``docs/development.md``: every switch and family, in declaration order."""
    rows = ["| variable | effect | default |", "|---|---|---|"]
    for s in SWITCHES:
        notes = f" Read at {s.read}." if s.read != "call" else ""
        rows.append(f"| `{s.name}` | {s.doc}.{notes} | {s.default or 'none'} |")
        if s is ROUTER_MODELS:
            rows.append(f"| `{ARG.prefix}<OPTION>` | {ARG.doc}. | the option's own |")
    return "\n".join(rows).replace("``", "`")

