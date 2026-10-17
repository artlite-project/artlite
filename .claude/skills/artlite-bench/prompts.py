"""Chat prompts of exactly N - 1 tokens (the model's chat template included) from the Paul Graham essays in
tests/datasets/paul_graham_essays: whole essays in file order, then the words of the next one, then a request to
summarize them.

    uv run python .claude/skills/artlite-bench/prompts.py MODEL.gguf OUT_DIR [N ...]   (default N: 1024 32768)

writes OUT_DIR/prompt<N>.txt and prints each prompt's token count.
"""

import os
import sys
from collections.abc import Callable

from artlite.model_config import model_path
from models import open_model

ESSAYS = os.path.join(os.path.dirname(__file__), "../../../tests/datasets/paul_graham_essays")
ASK = "\n\nSummarize the essays above in a few paragraphs."


def build(count: Callable[[str], int], essays: list[str], target: int) -> str:
    """The longest text (essays, then words of the next one) whose prompt has at most target - 1 tokens."""
    text = ""
    for e in essays:
        if count(text + e + "\n\n" + ASK) >= target:
            words = e.split(" ")
            lo, hi = 0, len(words)
            while lo < hi:  # (the most words of e that fit)
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if count(text + " ".join(words[:mid]) + ASK) < target else (lo, mid - 1)
            return text + " ".join(words[:lo]) + ASK
        text += e + "\n\n"
    raise ValueError(f"the essays are shorter than {target} tokens")


def main() -> None:
    model = open_model(model_path(sys.argv[1]))
    out = sys.argv[2]
    essays = []
    for name in sorted(os.listdir(ESSAYS)):
        with open(os.path.join(ESSAYS, name)) as f:
            essays.append(f.read())

    def count(text: str) -> int:
        return len(model.prompt(text))

    for target in [int(n) for n in sys.argv[3:]] or [1024, 32768]:
        text = build(count, essays, target)
        with open(os.path.join(out, f"prompt{target}.txt"), "w") as f:
            f.write(text)
        print(f"prompt{target}.txt: {count(text)} tokens")


if __name__ == "__main__":
    main()
