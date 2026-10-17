"""artlite-bench (models.bench) on the host: llama-bench's value lists, test order, output formats and model fields
(no NPU)."""

import argparse
import io
import json
import os

import pytest

from artlite.model_config import gemma_gguf
from models.bench import FIELDS, PRINTERS, arg_parser, file_info, file_type_name, int_list, plan, result
from models.bench import Test as BenchTest  # (not a pytest class)


def test_int_list() -> None:
    assert int_list("1024,2048") == [1024, 2048]
    assert int_list("1-4") == [1, 2, 3, 4]
    assert int_list("0-64+32") == [0, 32, 64]
    assert int_list("128-1024*2") == [128, 256, 512, 1024]
    for bad in ("", "1,,2", "a", "-4", "0-8*1"):
        with pytest.raises(argparse.ArgumentTypeError):
            int_list(bad)


def test_plan() -> None:
    """llama-bench's order: per model and depth, the prompts, the generations, then the pairs; zeros skipped."""
    args = arg_parser().parse_args(["-m", "a.gguf,b.gguf", "-p", "64,0", "-n", "0,16", "-pg", "8,4", "-d", "0",
                                    "-d", "512"])
    names = [(os.path.basename(t.model), t.name) for t in plan(args)]
    assert names == [(m, n) for m in ("a.gguf", "b.gguf")
                     for n in ("pp64", "tg16", "pp8+tg4", "pp64 @ d512", "tg16 @ d512", "pp8+tg4 @ d512")]
    defaults = plan(arg_parser().parse_args(["-m", "a.gguf"]))
    assert [t.name for t in defaults] == ["pp512", "tg128"]


def test_printers() -> None:
    env = {"build_commit": "abc", "build_number": 7, "cpu_info": "cpu", "gpu_info": "NPU", "backends": "NPU"}
    info = {"model_filename": "m.gguf", "model_type": "gemma4 7.5B Q4_0", "model_size": 4574983336,
            "model_n_params": 7463013674}
    r = result(BenchTest("m.gguf", 512, 0, 4096), [250_000_000, 260_000_000], **env, **info, n_ctx=8192,
               test_time="2026-10-09T00:00:00Z")
    assert r["avg_ns"] == 255_000_000 and r["avg_ts"] == pytest.approx((2048 + 512 / 0.26) / 2)
    out = {}
    for fmt, cls in PRINTERS.items():
        f = io.StringIO()
        p = cls(f, env)
        p.header()
        p.test(r)
        p.test(r)
        p.footer()
        out[fmt] = f.getvalue()
    assert json.loads(out["json"]) == [r, r]
    assert [json.loads(line) for line in out["jsonl"].splitlines()] == [r, r]
    csv_lines = out["csv"].splitlines()
    assert csv_lines[0] == ",".join(FIELDS) and csv_lines[1].startswith('"abc","7","cpu"') and len(csv_lines) == 3
    md = out["md"].splitlines()
    assert md[2] == ("| gemma4 7.5B Q4_0               |   4.26 GiB |     7.46 B | NPU        |   pp512 @ d4096 "
                     "|      2008.62 ± 55.70 |")
    assert len(md[0]) == len(md[1]) == len(md[2]) and md[-1] == "build: abc (7)"


def test_file_info() -> None:
    """model_type, model_size and model_n_params as llama.cpp's (size: the GGUF's size label)."""
    assert file_type_name(2) == "Q4_0" and file_type_name(15) == "Q4_K - Medium" and file_type_name(None) == "unknown"
    path = gemma_gguf("Q4_0")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not downloaded")
    info, n_vocab = file_info(path)
    assert info["model_type"] == "gemma4 7.5B Q4_0" and n_vocab == 262144
    assert (info["model_size"], info["model_n_params"]) == (4574983336, 7463013674)  # (llama-bench's)
