import hashlib
import os
import shutil
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from models.sampling import Sampling

# CPU reference results of the slow tests (whole-model logits), cached across runs in a folder per model: gitignored
REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
REFRESH_REFS = False  # --refresh-refs (conftest): recompute and overwrite the cached references
# the long-prompt corpus (gitignored): tests/datasets/fetch_datasets.py fetches it
ESSAYS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "paul_graham_essays")


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """A CPU tensor as a numpy array (bf16 as ``ml_dtypes.bfloat16``, through an int16 view; no copy when contiguous):
    what the runtime API (``artlite.from_numpy``, ``op.run``, ``FusedOperator.get_buffer``) takes."""
    t = t.detach().contiguous()
    return t.view(torch.int16).numpy().view(bfloat16) if t.dtype == torch.bfloat16 else t.numpy()


def to_torch(a: np.ndarray) -> torch.Tensor:
    """A numpy array as a CPU tensor sharing its memory (``ml_dtypes.bfloat16`` as torch.bfloat16, through an int16
    view): the runtime API's results (``Tensor.numpy()``, ``op.run``) in the torch references."""
    a = np.asarray(a)
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if a.dtype == bfloat16 else torch.from_numpy(a)


def as_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """A tensor of a torch reference's or a runtime result's values (``to_torch``; a tensor as it is)."""
    return to_torch(x) if isinstance(x, np.ndarray) else x


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


# synthetic videos (``video_file``): name -> (size, rate, seconds, audio, the video stream's start); "astronaut": the
# README's astronaut picture panned and zoomed, a LibriSpeech clip (6930-75918-0001, 14.2 s) as its soundtrack (none in
# "astronaut-silent")
VIDEOS = {
    "odd-25fps": ("320x240", "25", 2.6, True, 0.0),  # (5 frames: an odd last pair)
    "ntsc-start": ("640x360", "30000/1001", 6.5, True, 0.5),  # (a fractional rate; the video starting at 0.5 s)
    "short": ("200x120", "10", 1.0, True, 0.0),  # (4 frames at least)
    "big-1080p": ("1920x1080", "30", 7.0, True, 0.0),  # (14 frames over the pixel bound: scaled down)
    "silent": ("480x272", "24", 4.0, False, 0.0),  # (no audio stream)
    "astronaut": ("960x540", "25", 14.2, True, 0.0),
    "astronaut-silent": ("960x540", "25", 14.2, False, 0.0),
    "astronaut-small": ("320x180", "25", 4.0, False, 0.0),  # (8 frames of 60 tokens: a decision's prompt)
}
ASTRONAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "sd15-astronaut.jpg")
SPEECH_CLIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "librispeech", "6930-75918-0001.wav")
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "videos")


def video_file(name: str) -> str:
    """A synthetic MP4 of ``VIDEOS`` (H.264, AAC), made by ffmpeg once into ``VIDEOS_DIR`` (gitignored), named by a
    hash of its command (skip without ffmpeg)."""
    import subprocess
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("no ffmpeg / ffprobe")
    size, rate, sec, audio, start = VIDEOS[name]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if name.startswith("astronaut"):
        if audio and not os.path.exists(SPEECH_CLIP):
            pytest.skip(f"{SPEECH_CLIP} missing: run tests/datasets/fetch_datasets.py")
        w, h = size.split("x")
        pan = (f"scale=1920:-2,zoompan=z='1+0.002*on':x='iw/2-(iw/zoom/2)+2*on':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}"
               f":fps={rate}")
        cmd += ["-loop", "1", "-framerate", rate, "-i", ASTRONAUT, *(["-i", SPEECH_CLIP] if audio else []), "-t", str(sec),
                "-vf", pan]
    else:
        if start:
            cmd += ["-itsoffset", str(start)]
        cmd += ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={sec}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:beep_factor=4:duration={sec + start}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else [])
    path = os.path.join(VIDEOS_DIR, f"{name}-{hashlib.sha256(repr(cmd).encode()).hexdigest()[:12]}.mp4")
    if os.path.exists(path):
        return path
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.mp4"
    subprocess.run([*cmd, tmp], check=True)
    os.replace(tmp, path)
    for f in os.listdir(VIDEOS_DIR):  # (the video's earlier versions)
        if f.startswith(name + "-") and len(f) == len(os.path.basename(path)) and f != os.path.basename(path):
            os.remove(os.path.join(VIDEOS_DIR, f))
    return path


def draft_prompts(host: Any) -> dict[str, tuple[list[int], tuple[int, ...]]]:
    """The drafter gate's prompts (``draft_gate``) as a model's host side (``models.model.Model``) tokenizes them: the
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
                          float(torch.softmax(as_torch(logits), -1)[d[0]]) if d else 0.0))
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


def ref_entry(model: str, name: str, key: object, version: bytes, suffix: str = "") -> str:
    """The path of a cached reference (a file or a folder): ``REFS_DIR/<model>/<name>_<hash of key>_<hash of version>
    <suffix>``. ``key``: what the reference is of (model identity, inputs); ``version``: what computes it (the
    reference's source files, the llama.cpp build), so that a new version replaces the entry of the same key
    (``stored``)."""
    k = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    v = hashlib.sha256(version).hexdigest()[:16]
    return os.path.join(model_refs(model), f"{name}_{k}_{v}{suffix}")


def source_version(sources: Iterable[str], extra: object = ()) -> bytes:
    """A reference's version: its source files' bytes and ``extra`` (e.g. ``llama_cpp.lib_identity()``)."""
    out = repr(extra).encode()
    for src in sources:
        with open(src, "rb") as f:
            out += f.read()
    return out


def hit(path: str) -> bool:
    """Whether a cached reference exists (and ``--refresh-refs`` is off); a hit marks it used: an entry's mtime is
    its last use, so ``find tests/refs -mindepth 2 -maxdepth 2 -mtime +30`` lists those no test read for 30 days."""
    if os.path.exists(path) and not REFRESH_REFS:
        os.utime(path)
        return True
    return False


def computing() -> None:
    """Before computing a missing reference: in the tests' planning process (``artlite.parallel_compile``: it runs
    the selected tests to find their designs) stop the test there, as at its first device access; the test run
    computes it."""
    from artlite import parallel_compile

    if parallel_compile.planning_active():
        raise parallel_compile.PlanStop("planning: a reference is computed by the test run")


def stored(path: str) -> None:
    """After a reference was stored at ``path`` (``ref_entry``): remove its other versions (the same name and key)."""
    folder, base = os.path.split(path)
    stem, _, rest = base.rpartition("_")
    for f in os.listdir(folder):
        if f != base and len(f) == len(base) and f.startswith(stem + "_") and f.endswith(rest[16:]):
            old = os.path.join(folder, f)
            shutil.rmtree(old) if os.path.isdir(old) else os.remove(old)


def cached_ref(model: str, name: str, key: object, compute: Callable[[], Any], *, sources: Iterable[str] = (),
               version: object = ()) -> Any:
    """``compute()`` (tensors, or containers of them), cached in the ``model`` file's folder of ``REFS_DIR``
    (``ref_entry``): ``key``, what it is of (model identity, inputs); ``sources`` (the reference implementation's
    files) and ``version`` (e.g. the llama.cpp build), what computes it: a new version replaces the old result.
    ``--refresh-refs`` recomputes."""
    path = ref_entry(model, name, key, source_version(sources, version), ".pt")
    if hit(path):
        return torch.load(path)
    computing()
    out = _compact(compute())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)
    stored(path)
    return out
