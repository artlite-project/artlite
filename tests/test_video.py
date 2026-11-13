"""A video's frames and soundtrack (models/video.py) and LFM3-VL's video rules (models/lfm3vl/host.py ``VideoPlan``)
against llama.cpp's mtmd video helper (``tests.goldens.media.video_items``: as llama-server reads a video's bytes) on
videos made by ffmpeg: the frames' indices, size and RGB bit for bit, the pairs' timestamps, the soundtrack's pieces'
samples bit for bit; LFM3-Omni's mmproj (frames and soundtrack) and LFM3-VL-2B's (frames only). No NPU."""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from artlite.model_config import lfm3omni_gguf, lfm3omni_mmproj, lfm3vl_gguf, lfm3vl_mmproj
from models import video as VD
from models.lfm3vl import host as H

from .common import VIDEOS, video_file
from .goldens.media import video_items


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("videos"))


def _ours(path: str, heard: bool) -> list[dict]:
    """The items of a video as Artlite reads it: per pair its timestamp, its frames, its soundtrack piece."""
    data = VD.read(path)
    info = VD.probe(data)
    plan = H.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=heard)
    fr = VD.frames(data, info, plan.indices, (plan.width, plan.height))
    assert len(fr) == len(plan.indices)
    st = VD.soundtrack(data, info) if heard else None
    sl = plan.slices(len(st)) if st is not None else None
    out = []
    for g, (pair, stamp) in enumerate(zip(plan.pairs(), plan.stamps(), strict=True)):
        out.append({"text": stamp})
        out += [{"frame": fr[f]} for f in pair]
        if sl is not None and sl[g][1] > sl[g][0]:
            out.append({"audio": st[sl[g][0]:sl[g][1]]})
    return out


@pytest.mark.parametrize("name", list(VIDEOS))
@pytest.mark.parametrize("model", ["omni", "vl"])
def test_items(videos: str, name: str, model: str) -> None:
    """Artlite's frames, timestamps and soundtrack pieces are llama.cpp's, bit for bit (LFM3-Omni: with the
    soundtrack; LFM3-VL: frames only)."""
    m, mm = (lfm3omni_gguf(), lfm3omni_mmproj()) if model == "omni" else (lfm3vl_gguf(), lfm3vl_mmproj())
    for p in (m, mm):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    path = video_file(videos, name)
    want = video_items(m, mm, path)
    got = _ours(path, heard=model == "omni")
    kinds = lambda items: [next(k for k in ("text", "frame", "audio") if k in it) for it in items]
    assert kinds(got) == kinds(want)
    for a, b in zip(got, want, strict=True):
        if "text" in a:
            assert a["text"] == b["text"]
        else:
            k = "frame" if "frame" in a else "audio"
            assert a[k].shape == b[k].shape and np.array_equal(a[k], b[k]), (k, a[k].shape, b[k].shape)


def test_plans() -> None:
    """The frame count's bounds (4; 768, or 384 with the soundtrack; the video's frames), the indices (round of
    linspace), the size's rule (the pixel bound over all frames; small sides scaled up; a frame over the tower's
    largest image takes an image's size)."""
    p = H.VideoPlan.of(9000, 30.0, 1280, 720, heard=True)  # (5 min at 30 fps: 600 frames at 2 fps)
    assert len(p.indices) == 384 and p.indices[0] == 0 and p.indices[-1] == 8999
    assert len(H.VideoPlan.of(9000, 30.0, 1280, 720, heard=False).indices) == 600
    assert len(H.VideoPlan.of(3, 30.0, 64, 64, heard=False).indices) == 3  # (fewer frames than 4)
    assert H.VideoPlan.of(30, 30.0, 64, 64, heard=False).indices == (0, 10, 19, 29)  # (round(9.67), round(19.33))
    w, h = H.video_size(1280, 720, 60)  # (a 30 s 720p video at 2 fps: its 60 frames over 25.2 Mpx)
    assert w % 32 == h % 32 == 0 and 60 * w * h <= H.VIDEO_PIXELS[1] and (w, h) == (832, 480)
    assert H.video_size(16, 600, 8) == (32, 1216)  # (the narrow side scaled up to 32 first)
    big = H.VideoPlan.of(120, 60.0, 3840, 2160, heard=True)  # (2 s of 4K: 4 frames of 6 Mpx under the video bound)
    assert len(big.indices) == 4 and big.grid.tokens <= H.MAX_TOKENS  # (the tower's largest image instead)
    rows = H.VideoPlan.of(65, 25.0, 320, 240, heard=True)
    sl = rows.slices(41600)
    assert sl == [(0, 20480), (20480, 40960), (40960, 41600)] and rows.rows(sl, 33) == [(0, 16), (16, 32), (32, 33)]


def test_no_count() -> None:
    """A container that does not record the frame count (WebM): ``probe`` counts the stream's packets."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.webm")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=2",
                        "-c:v", "libvpx-vp9", "-deadline", "realtime", path], check=True)
        info = VD.probe(VD.read(path))
        assert info.n_frames == 24 and (info.width, info.height) == (160, 120)


def test_regular(videos: str) -> None:
    """A regular stream's frames are picked before ffmpeg converts them (``Info.regular``), with the pixels of the whole
    way (llama.cpp's command: every frame converted, the indices' kept): the tests' videos but the one that starts late
    (ffmpeg's constant-rate output duplicates its first frame from the file's start); a variable-rate stream (frames
    off their slots) and a stream with discarded packets (an edit list's lead-in) go the whole way."""
    import dataclasses
    with tempfile.TemporaryDirectory() as tmp:
        vfr = os.path.join(tmp, "vfr.mp4")  # (every third frame 40% of a slot late)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=192x128:rate=10:duration=3", "-vf",
                        "settb=1/1000,setpts='N*100+if(eq(mod(N\\,3)\\,2)\\,40\\,0)'", "-fps_mode", "passthrough",
                        "-enc_time_base:v", "1/1000", vfr], check=True)
        cut = os.path.join(tmp, "cut.mp4")  # (stream-copied from a mid-GOP time: discarded lead-in packets)
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.55", "-i", video_file(videos, "astronaut"), "-t", "4",
                        "-c", "copy", cut], check=True)
        paths = {n: video_file(videos, n) for n in VIDEOS} | {"vfr": vfr, "cut": cut}
        for name, path in paths.items():
            data = VD.read(path)
            info = VD.probe(data)
            assert info.regular == (name not in ("ntsc-start", "vfr", "cut")), name
            plan = H.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=True)
            size = (plan.width, plan.height)
            fast = VD.frames(data, info, plan.indices, size)
            assert np.array_equal(fast, VD.frames(data, dataclasses.replace(info, regular=False), plan.indices, size))
