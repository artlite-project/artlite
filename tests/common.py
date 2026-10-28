import hashlib
import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest
import torch

from artlite.model_config import llama_gguf  # noqa: F401 (the tests import it from here)
from models.common import Sampling

# CPU reference results of the slow tests (whole-model logits), cached across runs in a folder per model: gitignored
REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
REFRESH_REFS = False  # --refresh-refs (conftest): recompute and overwrite the cached references
# the long-prompt corpus (gitignored): tests/datasets/fetch_datasets.py fetches it
ESSAYS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "paul_graham_essays")


def essay(name: str) -> str:
    """The text of one essay of the long-prompt corpus, or skip the test if the corpus was not fetched."""
    path = os.path.join(ESSAYS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing: run tests/datasets/fetch_datasets.py")
    with open(path) as f:
        return f.read()


def essays() -> str:
    """The whole corpus: the essays concatenated sorted by file name."""
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    if not names:
        pytest.skip(f"{ESSAYS_DIR} missing: run tests/datasets/fetch_datasets.py")
    return "".join(essay(n) for n in names)


def draft_prompts(host: Any) -> dict[str, tuple[list[int], tuple[int, ...]]]:
    """The drafter gate's prompts (``draft_gate``) as a model's host side (``models.common.Model``) tokenizes them: the
    ids and where later prompts start (``OverlayModel.forced_drafts(turns=)``). Replies that copy their context (a
    summary, a quote: the drafter attends to the prompt's positions), the quote's second half as a later prompt (decode
    steps up to a cache group, then a chunk after them), a question without context (the prefill's smallest chunks) and
    an essay to continue (no chat template)."""
    quote = host.prompt(essay("founders.txt")[:3000] + "\n\nQuote the essay's first two paragraphs exactly, then "
                        "explain their main point in one sentence.")
    return {
        "summary": (host.prompt(essay("apple.txt")[:3000] + "\n\nSummarize the essay above in a few paragraphs."), ()),
        "quote": (quote, (len(quote) // 2 | 1,)),  # (an odd position: inside a cache group)
        "question": (host.prompt("Explain how photosynthesis works in plants, step by step."), ()),
        "essay": (host.prompt(essay("gba.txt")[:3500], raw=True), ()),
    }


def draft_gate(model: Any, prompts: dict[str, tuple[list[int], tuple[int, ...]]], n: int = 128
               ) -> dict[str, np.ndarray]:
    """A drafter's teacher-forced drafts (``OverlayModel.forced_drafts``) after each prompt through the prefill and as
    decode steps, over the model's greedy reply of up to n tokens after the decode steps' prompt (the reference the
    prefill is held to). Per path ("prefill", "steps"), per position of the replies: the tokens a round starting there
    gives (1 + the leading guesses the reply has), whether the first guess is the model's argmax and its probability
    under the model (temperature 1: a draft is kept where the sampled token is it) (positions, 3)."""
    out: dict[str, list[np.ndarray]] = {"prefill": [], "steps": []}
    for name, (ids, turns) in prompts.items():
        model.reset()
        pf, model.prefill = model.prefill, None
        try:
            reply = list(model.generate(model.forward(ids), n, Sampling(greedy=True), ids))
        finally:
            model.prefill = pf
        seq = ids + reply
        line = f"{name}: prompt {len(ids)}, reply {len(reply)}:"
        for path, rows in out.items():
            r = []
            for q, (d, logits) in enumerate(model.forced_drafts(seq, len(ids), turns=turns, prefill=path == "prefill")):
                after = seq[len(ids) + q + 1:]
                a = next((i for i, (x, y) in enumerate(zip(d, after)) if x != y), min(len(d), len(after)))
                r.append((1 + a, bool(d) and d[0] == int(logits.argmax()),
                          float(torch.softmax(logits, -1)[d[0]]) if d else 0.0))
            rows.append(np.array(r, np.float64))
            line += f"  {path} " + " / ".join(f"{x:.3f}" for x in rows[-1].mean(0))
        print(line)
    got = {k: np.concatenate(v) for k, v in out.items()}
    print(f"{len(got['steps'])} positions (tokens per round / first guess the argmax / its probability): " +
          ", ".join(f"{k} " + " / ".join(f"{x:.4f}" for x in v.mean(0)) for k, v in got.items()))
    return got


def _compact(obj: Any) -> Any:
    """Tensors cloned (a view, e.g. some rows of the logits, would be saved with its whole storage)."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_compact(x) for x in obj)
    return obj


def model_key(path: str) -> tuple[str, int]:
    """A model file's identity in a reference cache key: its name and size."""
    return os.path.basename(path), os.path.getsize(path)


def model_refs(model: str) -> str:
    """The folder of a model file's cached references: ``REFS_DIR/<file name without .gguf>``."""
    return os.path.join(REFS_DIR, os.path.basename(model).removesuffix(".gguf"))


def cached_ref(model: str, name: str, key: object, compute: Callable[[], Any], *, sources: Iterable[str] = ()) -> Any:
    """``compute()`` (tensors, or containers of them), cached in the ``model`` file's folder of ``REFS_DIR`` under
    a hash of ``key`` (model identity, inputs) and of the ``sources`` files (the reference implementation: editing
    it invalidates its results). ``--refresh-refs`` recomputes."""
    h = hashlib.sha256(repr(key).encode())
    for src in sources:
        with open(src, "rb") as f:
            h.update(f.read())
    folder = model_refs(model)
    path = os.path.join(folder, f"{name}_{h.hexdigest()[:16]}.pt")
    if os.path.exists(path) and not REFRESH_REFS:
        return torch.load(path)
    out = _compact(compute())
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)
    return out


def verify_close(output, reference, *, rel_tol, abs_tol, label="output"):
    """Assert elements are within tolerance.

    Uses IRON's verify_buffer formula:
      pass if abs_err < max(abs_tol, rel_tol * (|out| + |ref|))

    Handles reference in float64 (high-precision) by converting to float32.
    """
    assert output.shape == reference.shape, f"{label}: shape mismatch {output.shape} vs {reference.shape}"
    out_f = output.float().reshape(-1)
    ref_f = reference.float().reshape(-1)  # Converts float64 -> float32 if needed
    abs_err = (out_f - ref_f).abs()
    norm = (out_f.abs() + ref_f.abs()).clamp(max=3.4e38)
    threshold = torch.maximum(
        torch.tensor(abs_tol),
        torch.tensor(rel_tol) * norm,
    )
    ok = abs_err < threshold
    n_bad = (~ok).sum().item()
    total = output.numel()
    error_rate = n_bad / total if total > 0 else 0.0

    if n_bad == 0:
        return

    bad_idx = (~ok).nonzero(as_tuple=True)[0]
    worst_abs = abs_err[bad_idx].max().item()
    rel_err = abs_err / ref_f.abs().clamp(min=1e-30)
    worst_rel = rel_err[bad_idx].max().item()
    samples = bad_idx[:5]
    details = [
        f"  [{i.item()}] out={out_f[i].item():.6f} ref={ref_f[i].item():.6f} "
        f"abs={abs_err[i].item():.6f} rel={rel_err[i].item():.6f}"
        for i in samples
    ]
    pytest.fail(
        f"{label}: {n_bad}/{total} ({error_rate:.2%}) elements exceed tolerance "
        f"(rel_tol={rel_tol}, abs_tol={abs_tol})\n"
        f"  worst abs_err={worst_abs:.6f}, worst rel_err={worst_rel:.6f}\n" + "\n".join(details)
    )

