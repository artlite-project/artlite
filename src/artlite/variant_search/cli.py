"""The command line of ``python -m artlite.variant_search``."""

import argparse
import ast
import functools
import importlib
import importlib.util
import runpy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .npu import parse_regions, run_npu, run_worker
from .patch import pragma_text
from .report import print_report
from .search import search
from .variants import Variant


def parse_operator(expression: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    """'Name(key=literal, ...)': Name a submodule's attribute of the artlite package (gemm.gemm_q.GemmQOperator,
    decode.gemv.render) or a dotted import path (tests.decode.mmv.MmvOperator)."""
    package = "artlite"

    call = ast.parse(expression, mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError(f"expected Name(key=value, ...), got {expression!r}")
    parts = ast.unparse(call.func).split(".")

    def module(name: str) -> Any:
        try:
            return importlib.import_module(name) if importlib.util.find_spec(name) else None
        except ImportError:
            return None

    # (the longest prefix that is a submodule of artlite: decode.gemv.render)
    sub = next((i for i in range(len(parts), 0, -1) if module(".".join([package, *parts[:i]])) is not None), 0)
    if sub:
        base: Any = module(".".join([package, *parts[:sub]]))
        rest = parts[sub:]
    else:
        index = next((i for i in range(len(parts), 0, -1) if module(".".join(parts[:i])) is not None), 0)
        if not index:
            raise ValueError(f"cannot resolve {ast.unparse(call.func)}")
        base, rest = module(".".join(parts[:index])), parts[index:]
    factory = functools.reduce(getattr, rest, base)
    return factory, {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg}


def _text(argument: str) -> str:
    return Path(argument[1:]).read_text() if argument.startswith("@") else argument


def cli_variants(args: argparse.Namespace) -> list[Variant]:
    variants: dict[str, Variant] = {}

    def get(name: str) -> Variant:
        return variants.setdefault(name, Variant(name))

    if args.variants:
        loaded = runpy.run_path(args.variants)["VARIANTS"]
        for variant in loaded.values() if isinstance(loaded, Mapping) else loaded:
            variants[variant.name] = variant
    for token in filter(None, args.pragmas.split(",")):
        get(token).pragma = pragma_text(token)
    for name, text in args.pragma:
        get(name).pragma = pragma_text(text)
    for name, old, new in args.sub:
        get(name).subs.append((_text(old), _text(new)))
    for name, function, definition in args.define:
        get(name).defines[function] = _text(definition)
    for name, assignment in args.param:
        key, _, value = assignment.partition("=")
        try:
            get(name).params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            get(name).params[key] = value
    if args.loop:
        for variant in variants.values():
            variant.loops = args.loop
    if "base" not in variants:
        variants = {"base": Variant("base", loops=args.loop or None), **variants}
    return list(variants.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m artlite.variant_search",
        description="Compile kernel variants of an operator and rank the loops of target functions.",
    )
    parser.add_argument("operator", nargs="?", help="Name(key=value, ...): an operator class (gemm.gemm_q.GemmQOperator, "
                        "tests.decode.mmv.MmvOperator) or a function returning kernel source (decode.gemv.render)")
    parser.add_argument("--fn", action="append", default=[], help="target function (name substring); repeatable")
    parser.add_argument("--total", action="append", default=[], help="also report this function's cycles per call")
    parser.add_argument("--pragmas", default="", help="variants named by pragma shorthands: base,post,none,ii44")
    parser.add_argument("--pragma", nargs=2, action="append", default=[], metavar=("VARIANT", "PRAGMA"))
    parser.add_argument("--sub", nargs=3, action="append", default=[], metavar=("VARIANT", "OLD", "NEW"),
                        help="replace OLD (must match once) by NEW; @file reads the text")
    parser.add_argument("--define", nargs=3, action="append", default=[], metavar=("VARIANT", "FN", "DEFINITION"),
                        help="replace the definition of function FN; @file reads it")
    parser.add_argument("--param", nargs=2, action="append", default=[], metavar=("VARIANT", "KEY=VALUE"),
                        help="an operator keyword argument of the variant (a Python literal)")
    parser.add_argument("--variants", metavar="FILE.py", help="a file defining VARIANTS (a list of Variant)")
    parser.add_argument("--loop", type=int, action="append", help="pragmas go on these rolled loops (index in "
                        "source order over the target functions, then the always_inline helpers they call) instead "
                        "of the innermost ones")
    parser.add_argument("--why", action="store_true", help="post-pipeliner traces for every variant (debug llvm-aie)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="concurrent compiles (default: ARTLITE_COMPILE_JOBS, else the CPU count)")
    parser.add_argument("--out", help="directory for the sources and reports")
    parser.add_argument("--npu", action="store_true", help="also time each variant on the NPU")
    parser.add_argument("--reps", type=int, default=30, help="timed dispatches per variant")
    parser.add_argument("--region", action="append", default=[], metavar="NAME[=FN[*N]+...]",
                        help="counter windows of the operator's win=NAME build, static cycles from N calls of FN")
    parser.add_argument("--win-param", default="win", help="the operator parameter that selects an event window")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        run_worker(args.worker, compile_only=args.compile_only)
        return
    if not args.operator or not args.fn:
        parser.error("an operator and at least one --fn are required")
    factory, kwargs = parse_operator(args.operator)
    result = search(factory, kwargs, functions=args.fn, variants=cli_variants(args), totals=args.total,
                    jobs=args.jobs, out_dir=args.out, why=args.why)
    if args.npu:
        run_npu(result, reps=args.reps, regions=parse_regions(args.region), win_param=args.win_param,
                jobs=args.jobs)
    print_report(result)
