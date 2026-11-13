"""llama.cpp's media path as a reference: ``golden`` runs a prompt with media (images, audio clips, videos) through
libmtmd on the CPU (``minicpmv_dump.cpp``, built against the llama.cpp tree of ``model_config.LLAMA_CPP_DIR``): the
prompt's chunks (text tokens; image chunks with llama.cpp's decoder positions, which for MiniCPM-V 4.7 differ from the
official ``get_rope_index`` at the markers and after the image: llama.cpp PR #30510), each media chunk's embeddings,
the encoder's tensors per entry (MiniCPM-V: the overview, then the slices row by row; inp_raw: the entry's pixels) and
the logits after every text token. Cached under ``tests/refs/<model>/vgolden_<hash>/`` (``--refresh-refs``
recomputes). ``seq_golden``: the language model alone on a sequence of embedding rows at given (the official)
positions; ``video_items``: what llama.cpp's video helper reads from a video.
"""

import hashlib
import json
import os

import numpy as np

from .. import common
from ..llama_cpp import lib_identity
from .runner import build_tool, cached_golden, run_tool

DUMP_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minicpmv_dump.cpp")


def dump_tool() -> str:
    """The path of the built ``minicpmv_dump`` (compiled on first use into the artlite cache)."""
    return build_tool("minicpmv_dump", DUMP_SRC, mtmd=True)


class VisionGolden:
    """The outputs of one ``minicpmv_dump`` run: ``chunks`` (the chunks.json list), ``entries``, ``embd(i)``,
    ``tensor(name, entry, layer)``, ``logits`` and ``logit_tokens`` (the sequence index of each row)."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "chunks.json")) as f:
            j = json.load(f)
        self.chunks, self.logit_tokens, self.entries = j["chunks"], j["logit_tokens"], j["entries"]

    def embd(self, i: int = 0) -> np.ndarray:
        return np.load(os.path.join(self.path, f"embd-{i}.npy"))

    def tensor(self, name: str, entry: int, layer: int | None = None) -> np.ndarray:
        """``name`` ("layer_out", ...; "Kcur.1": the second of that name) of the encoder's entry, of a layer."""
        base, dot, dup = name.rpartition(".")
        if not (dot and dup.isdigit()):
            base, dup = name, ""
        stem = base if layer is None else f"{base}-{layer}"
        path = os.path.join(self.path, stem + (f".{dup}" if dup else "") + f".e{entry}.npy")
        if not os.path.exists(path):
            have = sorted(f for f in os.listdir(self.path) if f.endswith(f".e{entry}.npy"))
            raise KeyError(f"{name} {layer} e{entry}: not dumped (have {have[:40]})")
        return np.load(path)

    @property
    def logits(self) -> np.ndarray:
        return np.load(os.path.join(self.path, "logits.npy"), mmap_mode="r")


def golden(model: str, mmproj: str, image: str | list[str], prompt: str, *, answer: list[int] | tuple[int, ...] = (),
           names: str = "", embd: np.ndarray | None = None, no_soundtrack: bool = False,
           threads: int = 16) -> VisionGolden:
    """llama.cpp's media path for media files (``image``: an image, an audio file or a video, or a list of them) and a
    prompt (the text with ``<__media__>`` where each goes, in order) and teacher-forced answer tokens; ``names``: a
    regex of the encoder tensors to dump (without the "-<il>" suffix); ``embd``: the media chunks' embeddings (their
    tokens in order, n_embd) the language model takes instead of the encoders'; ``no_soundtrack``: a video's frames
    only."""
    ans = np.ascontiguousarray(answer, np.int32)
    emb = None if embd is None else np.ascontiguousarray(embd, np.float32)
    media = [image] if isinstance(image, str) else list(image)
    hashes = []
    for m in media:
        with open(m, "rb") as f:
            hashes.append(hashlib.sha256(f.read()).hexdigest())
    key = (common.model_key(model), common.model_key(mmproj), hashes[0] if len(hashes) == 1 else tuple(hashes), prompt,
           hashlib.sha256(ans.tobytes()).hexdigest(), names, None if emb is None else hashlib.sha256(emb.tobytes()).hexdigest(),
           lib_identity(), lib_identity("libmtmd.so")) + (("no_soundtrack",) if no_soundtrack else ())

    def run(tmp: str) -> None:
        with open(os.path.join(tmp, "prompt.txt"), "w") as f:
            f.write(prompt)
        cmd = [dump_tool(), "-m", model, "--mmproj", mmproj, *(a for m in media for a in ("--image", m)), "-p",
               os.path.join(tmp, "prompt.txt"), "-o", tmp, "--threads", str(threads)] + \
            (["--no-soundtrack"] if no_soundtrack else [])
        if ans.size:
            ans.tofile(os.path.join(tmp, "answer.i32"))
            cmd += ["-a", os.path.join(tmp, "answer.i32")]
        if names:
            cmd += ["-n", names]
        if emb is not None:
            emb.tofile(os.path.join(tmp, "embd.f32"))
            cmd += ["-e", os.path.join(tmp, "embd.f32")]
        run_tool(cmd)

    return VisionGolden(cached_golden(model, "vgolden", key, DUMP_SRC, run))


def video_items(model: str, mmproj: str, video: str, *, no_soundtrack: bool = False) -> list[dict]:
    """The items llama.cpp's video helper reads from a video (``minicpmv_dump --video-items``: as llama-server reads a
    video's bytes): {"text": s}, {"frame": RGB uint8 (h, w, 3)}, {"audio": float32 samples} in order. Cached as the
    goldens are."""
    with open(video, "rb") as f:
        vh = hashlib.sha256(f.read()).hexdigest()
    key = (common.model_key(model), common.model_key(mmproj), vh, no_soundtrack, lib_identity(), lib_identity("libmtmd.so"))

    def run(tmp: str) -> None:
        run_tool([dump_tool(), "-m", model, "--mmproj", mmproj, "--video-items", video, "-o", tmp] +
                 (["--no-soundtrack"] if no_soundtrack else []))

    path = cached_golden(model, "vitems", key, DUMP_SRC, run, marker="items.json")
    with open(os.path.join(path, "items.json")) as f:
        out = json.load(f)
    for it in out:
        if "frame" in it:
            it["frame"] = np.load(os.path.join(path, f"frame-{it['frame']}.npy"))
        elif "audio" in it:
            it["audio"] = np.load(os.path.join(path, f"audio-{it['audio']}.npy"))
    return out


def seq_golden(model: str, rows: np.ndarray, pos: np.ndarray, first: int = 0, *, threads: int = 16) -> VisionGolden:
    """llama.cpp's language model on a whole sequence of embedding rows (n, n_embd) (the text tokens' too) at the
    official M-RoPE positions ``pos`` (n, 3) (T, H, W), causal in token order (``minicpmv_dump --seq``): ``logits``
    of tokens ``first`` .. n - 1."""
    rows = np.ascontiguousarray(rows, np.float32)
    pos = np.ascontiguousarray(pos, np.int32)
    assert pos.shape == (rows.shape[0], 3), (rows.shape, pos.shape)
    key = (common.model_key(model), hashlib.sha256(rows.tobytes() + pos.tobytes()).hexdigest(), first, lib_identity(),
           lib_identity("libmtmd.so"))

    def run(tmp: str) -> None:
        rows.tofile(os.path.join(tmp, "seq.f32"))
        pos.tofile(os.path.join(tmp, "pos.i32"))
        run_tool([dump_tool(), "-m", model, "--seq", os.path.join(tmp, "seq.f32"), "--pos", os.path.join(tmp, "pos.i32"),
                  "--from", str(first), "-o", tmp, "--threads", str(threads)])
        for f in ("seq.f32", "pos.i32"):
            os.remove(os.path.join(tmp, f))

    return VisionGolden(cached_golden(model, "vseq", key, DUMP_SRC, run))
