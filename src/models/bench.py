"""artlite-bench: prompt processing and generation speeds of GGUF models on the NPU, measured as llama.cpp's
``llama-bench`` measures them, with a subset of its options and its output formats.

    artlite-bench -m ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf -p 1024,2048,4096,8192 -n 0 -o jsonl

The tests, per model and depth (``-d``) in llama-bench's order: ``pp<n>`` runs a prompt of n tokens (``-p``),
``tg<n>`` generates n tokens (``-n``), ``pp<n>+tg<m>`` does both (``-pg``); `` @ d<n>``: after a prompt of n tokens,
which is not timed and runs again before every repetition. Each test runs once untimed (the warmup: a prompt from
position 0, one generated token), then ``-r`` times from an empty KV cache; t/s = (prompt + generated tokens) /
time, averaged over the repetitions. The tokens are random, starting with the model's BOS if it has one. A prompt
runs as ``artlite-cli`` runs it (``NpuModel.forward``: prefill chunks, then the last token's logits); a generated
token is one decode step and the top logit of the next (``NpuModel.top_k``).

Unlike llama-bench, which creates a context per test, each model is loaded once, with one KV cache sized for its
largest test (``auto_capacity``; ``n_ctx`` in the results). The results go to stdout (``-o``) and stderr (``-oe``)
with llama-bench's field names; ``backends`` is "NPU" and ``gpu_info`` is the NPU's name.
"""

import argparse
import csv
import gc
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

import gguf
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import MODELS_DIR, model_path

from . import open_model
from .common import NpuModel, auto_capacity, stale_install

# a result's fields (llama-bench's that apply, plus n_ctx); json and jsonl add samples_ns and samples_ts
FIELDS = ("build_commit", "build_number", "cpu_info", "gpu_info", "backends", "model_filename", "model_type",
          "model_size", "model_n_params", "n_ctx", "n_prompt", "n_gen", "n_depth", "test_time", "avg_ns",
          "stddev_ns", "avg_ts", "stddev_ts")


def int_list(text: str) -> list[int]:
    """llama-bench's value list: comma-separated values or ranges ``first-last``, ``first-last+step``,
    ``first-last*mult``."""
    out = []
    for part in text.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+)(?:([+*])(\d+))?)?", part)
        if not m:
            raise argparse.ArgumentTypeError(f"invalid range {part!r}")
        i, last, mult, step = int(m[1]), int(m[2] or m[1]), m[3] == "*", int(m[4] or 1)
        while i <= last:
            out.append(i)
            nxt = i * step if mult else i + step
            if nxt <= i:
                raise argparse.ArgumentTypeError(f"invalid range {part!r}")
            i = nxt
    return out


def pair(text: str) -> tuple[int, int]:
    """``-pg pp,tg``."""
    try:
        pp, tg = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected <pp>,<tg>, not {text!r}") from None
    return pp, tg


def model_list(text: str) -> list[str]:
    return [model_path(p) for p in text.split(",")]


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="artlite-bench", description="Benchmark GGUF models on the NPU (as llama.cpp's llama-bench)",
        epilog="Multiple values can be given for each test parameter by separating them with ',' or by specifying "
               "the parameter multiple times. Ranges can be given as 'first-last' or 'first-last+step' or "
               "'first-last*mult'.")
    p.add_argument("-m", "--model", action="extend", type=model_list, required=True,
                   help=f"GGUF files; a relative path is also looked up in the models directory ({MODELS_DIR}, "
                        "ARTLITE_MODELS_DIR)")
    p.add_argument("-p", "--n-prompt", action="extend", type=int_list, help="prompt tokens (default: 512)")
    p.add_argument("-n", "--n-gen", action="extend", type=int_list, help="generated tokens (default: 128)")
    p.add_argument("-pg", action="append", type=pair, metavar="PP,TG",
                   help="a prompt, then generated tokens, timed together")
    p.add_argument("-d", "--n-depth", action="extend", type=int_list,
                   help="tokens in the KV cache before each test (default: 0)")
    p.add_argument("-r", "--repetitions", type=int, default=5, help="times to repeat each test (default: 5)")
    p.add_argument("--delay", type=float, default=0, help="seconds of pause before each test (default: 0)")
    p.add_argument("-o", "--output", choices=("csv", "json", "jsonl", "md"), default="md",
                   help="output format printed to stdout (default: md)")
    p.add_argument("-oe", "--output-err", choices=("csv", "json", "jsonl", "md", "none"), default="none",
                   help="output format printed to stderr (default: none)")
    p.add_argument("--progress", action="store_true", help="print test progress indicators")
    p.add_argument("--no-warmup", action="store_true", help="skip the warmup run before each test")
    return p


@dataclass(frozen=True)
class Test:
    model: str  # GGUF file
    n_prompt: int
    n_gen: int
    n_depth: int

    @property
    def name(self) -> str:
        """llama-bench's: pp512, tg128, pp512+tg128, tg128 @ d4096."""
        name = "+".join(f"{kind}{n}" for kind, n in (("pp", self.n_prompt), ("tg", self.n_gen)) if n)
        return name + (f" @ d{self.n_depth}" if self.n_depth else "")


def plan(args: argparse.Namespace) -> list[Test]:
    """The tests in llama-bench's order: per model and depth, the prompts, the generations, then the pairs."""
    n_prompt, n_gen, depths = args.n_prompt or [512], args.n_gen or [128], args.n_depth or [0]
    return [t for m in args.model for d in depths
            for t in ([Test(m, p, 0, d) for p in n_prompt if p] + [Test(m, 0, n, d) for n in n_gen if n]
                      + [Test(m, p, n, d) for p, n in args.pg or [] if p or n])]


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), *args], capture_output=True,
                           text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _read(path: str, prefix: str = "") -> str:
    """A system file's first line starting with ``prefix`` (its value after a colon, if any), or ""."""
    try:
        with open(path) as f:
            line = next((s for s in f if s.startswith(prefix)), "")
    except OSError:
        return ""
    return (line.split(":", 1)[1] if prefix else line).strip()


def environment() -> dict[str, object]:
    """The fields that describe the build and the machine."""
    return {"build_commit": _git("rev-parse", "--short=9", "HEAD") or "unknown",
            "build_number": int(_git("rev-list", "--count", "HEAD") or 0),
            "cpu_info": _read("/proc/cpuinfo", "model name"),
            "gpu_info": _read("/sys/class/accel/accel0/device/vbnv"),  # (the device the runtime opens)
            "backends": "NPU"}


def file_type_name(ftype: object) -> str:
    """llama.cpp's name of a GGUF ``general.file_type``: Q4_0, Q4_K - Medium, BF16, ..."""
    try:
        name = gguf.LlamaFileType(int(ftype)).name.removeprefix("MOSTLY_")
    except (TypeError, ValueError):
        return "unknown"
    return re.sub(r"_K_([SML])$", lambda m: "_K - " + {"S": "Small", "M": "Medium", "L": "Large"}[m[1]], name)


def file_info(path: str, *, gguf: GGUFFile | None = None) -> tuple[dict[str, object], int]:
    """A GGUF file's fields (llama.cpp's model_type, size and parameters) and its vocabulary size; ``gguf``: the file
    already open."""
    f = gguf or GGUFFile(path)
    info = {"model_filename": path,
            "model_type": f"{f.arch} {f.meta.get('general.size_label', '?B')} "
                          f"{file_type_name(f.meta.get('general.file_type'))}",
            "model_size": sum(nbytes or 0 for _, _, _, nbytes in f.tensor_info.values()),
            "model_n_params": sum(math.prod(dims) for dims, _, _, _ in f.tensor_info.values())}
    return info, f.string_count("tokenizer.ggml.tokens")


def token_source(n_vocab: int, bos: list[int], rng: torch.Generator) -> Callable[[int, int], list[int]]:
    """Random input tokens: ``(n, position)`` -> n tokens from a position, the first ``bos`` at position 0."""

    def tokens(n: int, position: int) -> list[int]:
        ids = torch.randint(n_vocab, (n,), generator=rng).tolist()
        return bos + ids[len(bos):] if ids and position == 0 else ids

    return tokens


def measure(npu: NpuModel, test: Test, tokens: Callable[[int, int], list[int]], *, reps: int, warmup: bool,
            progress: str | None) -> list[int]:
    """The test's samples (ns): each a prompt and / or generated tokens after the depth's prompt (not timed);
    ``tokens``: ``token_source``'s; ``progress``: the prefix of progress lines to stderr (None: none)."""

    def say(what: str) -> None:
        if progress is not None:
            sys.stderr.write(f"{progress}{what}\n")

    for n in (test.n_depth, test.n_prompt):
        npu.prepare(n)  # (compiles before anything is timed)
    if warmup:
        npu.reset()
        if test.n_prompt:
            say("warmup prompt run")
            npu.forward(tokens(test.n_prompt, 0))
        if test.n_gen:
            say("warmup generation run")
            npu.top_k(tokens(1, npu.position)[0], 1)
    samples = []
    for r in range(1, reps + 1):
        npu.reset()
        if test.n_depth:
            say(f"depth run {r}/{reps}")
            npu.forward(tokens(test.n_depth, 0))
        prompt, gen = tokens(test.n_prompt, test.n_depth), tokens(test.n_gen, test.n_depth + test.n_prompt)
        t0 = time.perf_counter_ns()
        if prompt:
            say(f"prompt run {r}/{reps}")
            npu.forward(prompt)
        if gen:
            say(f"generation run {r}/{reps}")
        for token in gen:
            npu.top_k(token, 1)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def result(test: Test, samples: list[int], **fields: object) -> dict[str, object]:
    """A test's result: ``fields`` (environment, file, n_ctx, test_time) and the statistics of its samples."""
    ts = [(test.n_prompt + test.n_gen) * 1e9 / ns for ns in samples]

    def stdev(v: list[float]) -> float:
        return statistics.stdev(v) if len(v) > 1 else 0.0

    r = {**fields, "n_prompt": test.n_prompt, "n_gen": test.n_gen, "n_depth": test.n_depth,
         "avg_ns": int(statistics.fmean(samples)), "stddev_ns": int(stdev(samples)), "avg_ts": statistics.fmean(ts),
         "stddev_ts": stdev(ts)}
    return {k: r[k] for k in FIELDS} | {"samples_ns": samples, "samples_ts": ts}


class Printer:
    """An output format of llama-bench's: ``header``, ``test`` per result, ``footer``."""

    def __init__(self, out: TextIO, env: dict[str, object]):
        self.out = out
        self.env = env

    def header(self) -> None:
        pass

    def test(self, r: dict[str, object]) -> None:
        raise NotImplementedError

    def footer(self) -> None:
        pass


class Csv(Printer):
    def header(self) -> None:
        self.out.write(",".join(FIELDS) + "\n")

    def test(self, r: dict[str, object]) -> None:
        csv.writer(self.out, quoting=csv.QUOTE_ALL, lineterminator="\n").writerow(
            f"{r[k]:.6f}" if isinstance(r[k], float) else r[k] for k in FIELDS)


class Json(Printer):
    first = True

    def header(self) -> None:
        self.out.write("[")

    def test(self, r: dict[str, object]) -> None:
        self.out.write(("\n" if self.first else ",\n") + textwrap.indent(json.dumps(r, indent=2), "  "))
        self.first = False

    def footer(self) -> None:
        self.out.write("\n]\n")


class Jsonl(Printer):
    def test(self, r: dict[str, object]) -> None:
        self.out.write(json.dumps(r) + "\n")


class Markdown(Printer):
    # (name, width): < 0 left-aligned
    COLUMNS = (("model", -30), ("size", 10), ("params", 10), ("backend", -10), ("test", 15), ("t/s", 20))

    def _row(self, values: list[str]) -> None:
        self.out.write("|" + "".join(f" {v:{'<' if w < 0 else '>'}{abs(w)}} |"
                                     for v, (_, w) in zip(values, self.COLUMNS, strict=True)) + "\n")

    def header(self) -> None:
        self._row([name for name, _ in self.COLUMNS])
        self.out.write("|" + "".join(f" {'-' * (abs(w) - 1)}{':' if w > 0 else '-'} |" for _, w in self.COLUMNS) + "\n")

    def test(self, r: dict[str, object]) -> None:
        size, params = r["model_size"], r["model_n_params"]
        test = Test(str(r["model_filename"]), int(r["n_prompt"]), int(r["n_gen"]), int(r["n_depth"]))
        self._row([str(r["model_type"]),
                   f"{size / 2**20:.2f} MiB" if size < 2**30 else f"{size / 2**30:.2f} GiB",
                   f"{params / 1e6:.2f} M" if params < 1e9 else f"{params / 1e9:.2f} B",
                   str(r["backends"]), test.name, f"{r['avg_ts']:.2f} ± {r['stddev_ts']:.2f}"])

    def footer(self) -> None:
        self.out.write(f"\nbuild: {self.env['build_commit']} ({self.env['build_number']})\n")


PRINTERS: dict[str, type[Printer]] = {"csv": Csv, "json": Json, "jsonl": Jsonl, "md": Markdown}


def main(argv: list[str] | None = None) -> None:
    args = arg_parser().parse_args(argv)
    if args.repetitions < 1:
        sys.exit("artlite-bench: --repetitions must be at least 1")
    if stale := stale_install():
        sys.stderr.write(f"artlite-bench: warning: {stale}\n")
    tests = plan(args)
    env = environment()
    printers = [PRINTERS[fmt](out, env) for fmt, out in ((args.output, sys.stdout), (args.output_err, sys.stderr))
                if fmt != "none"]
    for p in printers:
        p.header()
    rng = torch.Generator().manual_seed(0)
    i = 0
    for path, group in itertools.groupby(tests, key=lambda t: t.model):  # (a model stays loaded for its tests)
        group = list(group)
        name = os.path.basename(path)
        t0 = time.perf_counter()
        capacity = auto_capacity(max(t.n_depth + t.n_prompt + t.n_gen for t in group))
        try:
            info, n_vocab = file_info(path)
            model = open_model(path)
            npu = model.load(capacity=capacity)
        except (OSError, ValueError, NotImplementedError) as e:
            sys.exit(f"artlite-bench: {name}: {e}")
        sys.stderr.write(f"artlite-bench: {name} ({model.name}): loaded in {time.perf_counter() - t0:.1f} s: "
                         f"{npu.describe()}\n")
        tokens = token_source(n_vocab, model.encode("")[:1], rng)  # (encode: the model's BOS, if it has one)
        for test in group:
            i += 1
            progress = f"artlite-bench: benchmark {i}/{len(tests)}: " if args.progress else None
            if progress:
                sys.stderr.write(f"{progress}starting\n")
            if args.delay:
                time.sleep(args.delay)
            test_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                samples = measure(npu, test, tokens, reps=args.repetitions, warmup=not args.no_warmup,
                                  progress=progress)
            except (ValueError, NotImplementedError) as e:
                sys.exit(f"artlite-bench: {name}: {test.name}: {e}")
            r = result(test, samples, **env, **info, n_ctx=capacity, test_time=test_time)
            for p in printers:
                p.test(r)
                p.out.flush()
        del model, npu, tokens
        gc.collect()  # (the model's NPU buffers and context before the next model's)
    for p in printers:
        p.footer()
        p.out.flush()


if __name__ == "__main__":
    main()
