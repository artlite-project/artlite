"""A video file's frames and soundtrack, read by ffprobe and ffmpeg (system tools on PATH) with llama.cpp's commands
(mtmd's video helper), so that a prompt's frames and samples are llama-server's for the same file.

- ``probe``: the first video stream's size, rate (``r_frame_rate`` as a float32, llama.cpp's float), frame count
  (``nb_frames``; a container that does not record it: the stream's packets counted) and start time.
- ``Frames``: the frames at given indices (display order; an index may repeat), as RGB, scaled to a size by ffmpeg's
  bicubic (B 0, C 0.5: the closest to Pillow's) with accurate rounding and full chroma interpolation when the size is
  not the video's, as ffmpeg decodes them (a thread drains its output, so that it decodes ahead while the frames are
  used: a video's pairs encode on the NPU meanwhile); ``frames``: all of them at once.
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
    ``start`` its start time in seconds."""

    width: int
    height: int
    fps: float
    n_frames: int
    start: float

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
    r = _run([_tool("ffprobe"), "-v", "quiet", "-show_entries",
              "stream=width,height,r_frame_rate,nb_frames,duration,start_time", "-select_streams", "v:0", "-of",
              "default=noprint_wrappers=1", *INPUT[:2], INPUT[3]], data)
    kv = dict(line.split("=", 1) for line in r.stdout.decode(errors="replace").splitlines() if "=" in line)
    num = lambda k, f, d: f(kv[k]) if kv.get(k, "N/A") not in ("N/A", "") else d
    width, height = num("width", int, 0), num("height", int, 0)
    fps = _rate(kv.get("r_frame_rate", ""))
    if width <= 0 or height <= 0 or fps <= 0:
        raise VideoError("not a video file (ffprobe found no video stream)")
    n = num("nb_frames", int, 0)
    if n <= 0:  # (a container without the count: the packets counted)
        c = _run([_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                  "stream=nb_read_packets", "-of", "csv=p=0", *INPUT[:2], INPUT[3]], data)
        n = int(c.stdout.decode().strip() or 0) if c.stdout.decode().strip().isdigit() else 0
    if n <= 0:
        raise VideoError("a video without frames")
    return Info(width, height, fps, n, num("start_time", float, 0.0))


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
        cmd = [_tool("ffmpeg"), "-nostdin", *INPUT]
        if (w, h) != (info.width, info.height):
            cmd += ["-vf", f"format=rgb24,scale={w}:{h}:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5"]
        cmd += ["-frames:v", str(self.idx[-1] + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel",
                "error"]
        self.nbytes = w * h * 3
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        widen_pipe(self.proc.stdout.fileno(), self.CHUNK)
        self.feeder = _feed(self.proc, data)
        self.q: queue.Queue = queue.Queue(ahead)
        self.stop = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

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
        idx, n = self.idx, self.nbytes
        fd, buf, at, k, src = self.proc.stdout.fileno(), bytearray(), 0, 0, 0
        try:
            while src <= idx[-1]:
                chunk = os.read(fd, max(self.CHUNK, n))
                if not chunk:
                    break
                buf += chunk
                while len(buf) - at >= n and src <= idx[-1]:
                    while k < len(idx) and idx[k] == src:
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

    def __iter__(self) -> Iterator[np.ndarray]:
        w, h = self.size
        got = 0
        while (raw := self.q.get()) is not None:
            got += 1
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        if got == 0:
            self.proc.wait()
            err = self.proc.stderr.read().decode(errors="replace").strip()[:200]
            raise VideoError(f"ffmpeg decoded no frame: {err}")

    def close(self) -> None:
        self.stop.set()
        self.proc.kill()
        self.proc.wait()
        self.reader.join()
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
