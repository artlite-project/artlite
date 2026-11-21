"""llama.cpp's own conversion of a Hugging Face checkpoint (``convert_hf_to_gguf.py`` of ``model_config.LLAMA_CPP_DIR``,
``--outtype bf16``) as the reference of ``artlite.hf_model``'s virtual GGUF: ``tests/refs/hf/<case>-BF16.gguf``
(gitignored). The converter needs torch, transformers and sentencepiece, which Artlite's environment does not hold, so
the files are made once by this tool with a Python that has them; the tests skip without them:

    python -m tests.goldens.hf --python /path/to/venv/bin/python qwen3-0.6B qwen3.5-0.8B llama-1B

A checkpoint's ``tokenizer.model`` is left out of the converter's view (a Llama 3 checkpoint's is a tiktoken file the
converter's SentencePiece attempt fails on before it falls back to tokenizer.json)."""

import argparse
import os
import subprocess
import sys
import tempfile

from artlite.model_config import HF_CHECKPOINTS, LLAMA_CPP_DIR, checkpoint

from ..common import REFS_DIR


def converted(case: str) -> str:
    """The path of a case's converted GGUF (it may not exist)."""
    return os.path.join(REFS_DIR, "hf", f"{case}-BF16.gguf")


def convert(case: str, python: str) -> str:
    """Convert a case's checkpoint with llama.cpp's converter (run by ``python``) into ``converted(case)``."""
    out = converted(case)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    src = checkpoint(case)
    with tempfile.TemporaryDirectory() as tmp:
        for name in os.listdir(src):
            if name != "tokenizer.model":
                os.symlink(os.path.join(os.path.realpath(src), name), os.path.join(tmp, name))
        subprocess.run([python, os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py"), tmp, "--outtype", "bf16",
                        "--outfile", f"{out}.tmp"], check=True)
    os.replace(f"{out}.tmp", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", default=sys.executable, help="a Python with torch, transformers and sentencepiece")
    ap.add_argument("cases", nargs="+", choices=sorted(HF_CHECKPOINTS))
    a = ap.parse_args()
    for case in a.cases:
        print(convert(case, a.python))


if __name__ == "__main__":
    main()
