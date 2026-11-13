"""A video file's frames and soundtrack, read by ffprobe and ffmpeg (system tools on PATH) with llama.cpp's commands
(mtmd's video helper), so that a prompt's frames and samples are llama-server's for the same file.

- ``probe``: the first video stream's size, rate (``r_frame_rate`` as a float32, llama.cpp's float), frame count
  (``nb_frames``; a container that does not record it: the stream's packets counted), start time, and whether its
  frames keep to their rate (``regular``: each within a quarter of a frame of its slot from the file's start, none of
  its packets discarded).
- ``Frames``: the frames at given indices (display order; an index may repeat), as RGB, scaled to a size by ffmpeg's
  bicubic (B 0, C 0.5: the closest to Pillow's) with accurate rounding and full chroma interpolation when the size is
  not the video's, as ffmpeg decodes them (a thread drains its output, so that it decodes ahead while the frames are
  used: a video's pairs encode on the NPU meanwhile); ``frames``: all of them at once. llama.cpp's command converts
  every frame up to the last index (ffmpeg's constant-rate output: a frame per slot of the rate, a stream that strays
  from it duplicated or dropped to fit) and keeps the indices'. A regular stream's slots are its frames, so ffmpeg
  picks the indices' frames before converting them (the same pixels; a 3-minute 720p video's 5400 frames decoded, 360
  converted); any other goes the whole way.
- ``soundtrack``: the first audio stream as mono float32 samples at a rate (ffmpeg's resampler), over the video stream's
  span: from its start time for n_frames / fps seconds; None without an audio stream.

The file goes to the tools through their standard input (``cache:pipe:0``: seekable, an MP4's index may be at its end)
as llama.cpp feeds them. Without ffmpeg / ffprobe a video is a ``VideoError`` that names them.
"""

import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np

from artlite.portable import widen_pipe

INPUT = ("-read_ahead_limit", "-1", "-i", "cache:pipe:0")  # (stdin, seekable; no read-ahead limit: the moov atom)


class VideoError(ValueError):
    """A video that cannot be read: no ffmpeg / ffprobe, or a file they do not decode."""


@dataclass(frozen=True)
class Info:
    """A video stream: ``width`` x ``height`` pixels, ``fps`` frames a second (a float32's value), ``n_frames`` frames,
    ``start`` its start time in seconds; ``regular``: ffmpeg's constant-rate output is its frames, each once."""

    width: int
    height: int
    fps: float
    n_frames: int
    start: float
    regular: bool = False

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoError(f"videos need {name} (FFmpeg) on PATH")
    return path


def _run(cmd: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=data, capture_output=True, check=False)


def _rate(s: str) -> float:
    """``r_frame_rate`` ("30000/1001", "25") as llama.cpp's float."""
    num, _, den = s.partition("/")
    try:
        if den:
            n, d = int(num), int(den)
            return float(np.float32(n) / np.float32(d)) if d > 0 else 0.0
        return float(np.float32(float(num)))
    except ValueError:
        return 0.0


def probe(data: bytes) -> Info:
    """The first video stream of a file's bytes (module docstring)."""
    r = _run([_tool("ffprobe"), "-v", "quiet", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,r_frame_rate,nb_frames,start_time:format=start_time:packet=pts_time,flags", "-of",
              "compact", *INPUT[:2], INPUT[3]], data)
    stream, file, pts, packets, dropped = {}, {}, [], 0, False
    for line in r.stdout.decode(errors="replace").splitlines():
        kind, _, rest = line.partition("|")
        kv = dict(f.split("=", 1) for f in rest.split("|") if "=" in f)
        if kind == "packet":
            packets += 1
            t = kv.get("pts_time", "N/A")
            if "D" in kv.get("flags", "") or t in ("N/A", ""):  # (a packet the decoder discards, or without a time)
                dropped = True
            else:
                pts.append(float(t))
        elif kind in ("stream", "format"):
            (stream if kind == "stream" else file).update(kv)
    num = lambda kv, k, f, d: f(kv[k]) if kv.get(k, "N/A") not in ("N/A", "") else d
    width, height = num(stream, "width", int, 0), num(stream, "height", int, 0)
    fps = _rate(stream.get("r_frame_rate", ""))
    if width <= 0 or height <= 0 or fps <= 0:
        raise VideoError("not a video file (ffprobe found no video stream)")
    n = num(stream, "nb_frames", int, 0) or packets  # (a container without the count: the packets counted)
    if n <= 0:
        raise VideoError("a video without frames")
    slots = (np.sort(np.asarray(pts, np.float64)) - num(file, "start_time", float, 0.0)) * fps - np.arange(len(pts))
    regular = bool(pts) and not dropped and bool(np.abs(slots).max() < 0.25)
    return Info(width, height, fps, n, num(stream, "start_time", float, 0.0), regular)


def _select(frames: list[int]) -> str:
    """ffmpeg's select filter of the frames numbered ``frames`` (a sum of eq(n, i) as a balanced tree: the expression
    parser's depth)."""
    def tree(xs: list[int]) -> str:
        return f"eq(n,{xs[0]})" if len(xs) == 1 else f"({tree(xs[:len(xs) // 2])}+{tree(xs[len(xs) // 2:])})"
    return f"select='{tree(frames)}'"


def _feed(proc: subprocess.Popen, data: bytes) -> threading.Thread:
    """data into a child's stdin from a thread (the child may stop reading early)."""
    def run() -> None:
        try:
            proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class Frames:
    """The frames at ``indices`` (ascending) at ``size`` (width, height), each uint8 (height, width, 3), as ffmpeg
    decodes them: ffmpeg starts at once and decodes up to ``ahead`` frames before they are taken (``with`` the stream,
    iterate it); fewer where the stream ends before an index (llama.cpp keeps the frames it got)."""

    CHUNK = 1 << 20  # (bytes of a read: the frames ffmpeg wrote meanwhile, the interpreter taken once for them)

    def __init__(self, data: bytes, info: Info, indices: list[int] | np.ndarray, size: tuple[int, int], *,
                 ahead: int = 64):
        w, h = self.size = size
        self.idx = [int(i) for i in indices]
        self.keep = sorted(set(self.idx)) if info.regular else list(range(self.idx[-1] + 1))  # (ffmpeg's output frames)
        vf = [_select(self.keep) + ",setpts=N/FRAME_RATE/TB"] if info.regular else []  # (the picked frames, in slots
        # of their own: the constant-rate output passes each once)
        if (w, h) != (info.width, info.height):
            vf.append(f"format=rgb24,scale={w}:{h}:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5")
        cmd = [_tool("ffmpeg"), "-nostdin", *INPUT, *(["-vf", ",".join(vf)] if vf else []), "-frames:v",
               str(len(self.keep)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "error"]
        self.nbytes = w * h * 3
        self.q: queue.Queue = queue.Queue(ahead)
        self.stop = threading.Event()
        self.errors = b""  # (the end of ffmpeg's messages)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            widen_pipe(self.proc.stdout.fileno(), self.CHUNK)
            self.feeder = _feed(self.proc, data)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.drainer = threading.Thread(target=self._drain, daemon=True)
            self.reader.start()
            self.drainer.start()
        except BaseException:  # (no process left behind by a constructor that fails)
            self.stop.set()
            self.proc.kill()
            self.proc.wait()
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc.stderr.close()
            raise

    def _put(self, x: bytes | None) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(x, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _read(self) -> None:
        """ffmpeg's frames in order, several a read; the indices' kept, an index's frame once per occurrence."""
        idx, keep, n = self.idx, self.keep, self.nbytes
        fd, buf, at, k, src = self.proc.stdout.fileno(), bytearray(), 0, 0, 0
        try:
            while src < len(keep):
                chunk = os.read(fd, max(self.CHUNK, n))
                if not chunk:
                    break
                buf += chunk
                while len(buf) - at >= n and src < len(keep):
                    while k < len(idx) and idx[k] == keep[src]:
                        if not self._put(bytes(buf[at:at + n])):
                            return
                        k += 1
                    at, src = at + n, src + 1
                del buf[:at]
                at = 0
        except OSError:
            pass
        finally:
            self._put(None)

    def _drain(self) -> None:
        """ffmpeg's messages (its error pipe read as they come: a verbose one cannot block it), the last 2 KB kept."""
        try:
            while chunk := self.proc.stderr.read1(4096):
                self.errors = (self.errors + chunk)[-2048:]
        except (OSError, ValueError):
            pass

    def __iter__(self) -> Iterator[np.ndarray]:
        """The frames; VideoError: none decoded, or ffmpeg exited with an error status."""
        w, h = self.size
        got = 0
        while (raw := self.q.get()) is not None:
            got += 1
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        status = self.proc.wait()
        self.drainer.join()
        if got == 0 or status:
            err = self.errors.decode(errors="replace").strip()[-200:]
            what = "decoded no frame" if got == 0 else f"exited with status {status}"
            raise VideoError(f"ffmpeg {what}: {err}")

    def close(self) -> None:
        self.stop.set()
        self.proc.kill()
        self.proc.wait()
        self.reader.join()
        self.drainer.join()
        self.feeder.join()
        self.proc.stdout.close()
        self.proc.stderr.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def frames(data: bytes, info: Info, indices: list[int] | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``Frames``' frames at once: uint8 (frames, height, width, 3)."""
    with Frames(data, info, indices, size) as fr:
        return np.stack(list(fr))


def soundtrack(data: bytes, info: Info, rate: int = 16000) -> np.ndarray | None:
    """The first audio stream over the video stream's span (module docstring), mono float32 at ``rate``; None: no audio
    stream (or none where the video's frames are)."""
    r = _run([_tool("ffmpeg"), "-nostdin", *INPUT, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le",
              "pipe:1", "-loglevel", "error"], data)
    pcm = np.frombuffer(r.stdout[:len(r.stdout) // 4 * 4], "<f4")
    start = int(max(0.0, np.rint(info.start * rate)))
    n = int(np.rint(info.n_frames / np.float64(np.float32(info.fps)) * rate))
    if start >= pcm.size:
        return None
    return pcm[start:start + n].astype(np.float32)


def read(src: str | bytes) -> bytes:
    """A video file's bytes (a path or the bytes)."""
    if isinstance(src, bytes):
        return src
    with open(src, "rb") as f:
        return f.read()
