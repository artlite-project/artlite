"""Stable Diffusion 1.5's host side (models.sd15): the UNet's time inputs, the text encoder against the fp32 reference,
text to image in a couple of steps on the reference's networks, artlite-cli's image path and its refusals, diffusers
directories in ``models``. The fp32 networks (ref.py) match diffusers' and transformers' to fp32 rounding."""

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artlite import parallel_compile
from artlite.model_config import sd15_dir
from models import architecture, is_diffusers, open_pipeline
from models.cli import main
from models.sd15.host import resnet_time_biases, time_embedding, timestep_embedding
from models.sd15.pipeline import StableDiffusion
from models.sd15.text import ClipText
from models.sd15.tokenizer import ClipTokenizer
from models.sd15.weights import load_safetensors

from . import ref

# host only: nothing for the background compilation's planning run to see (it would run the CPU pipeline again)
pytestmark = pytest.mark.skipif(parallel_compile.planning_active(), reason="host only")


def _model_dir() -> str:
    path = sd15_dir()
    if not is_diffusers(path):
        pytest.skip(f"{path}: not present")
    return path


def test_timestep_embedding() -> None:
    np.testing.assert_array_equal(timestep_embedding(0), np.concatenate([np.ones(160), np.zeros(160)]))
    e = timestep_embedding(517.37)
    for i in (0, 1, 80, 159):
        f = 517.37 * 10000 ** (-i / 160)
        assert float(e[i]) == pytest.approx(math.cos(f), abs=1e-4) and float(e[160 + i]) == pytest.approx(math.sin(f),
                                                                                                          abs=1e-4)


def test_diffusers_dir(tmp_path: Path) -> None:
    assert not is_diffusers(str(tmp_path))
    with open(tmp_path / "model_index.json", "w") as f:
        json.dump({"_class_name": "FluxPipeline"}, f)
    assert is_diffusers(str(tmp_path))
    with pytest.raises(NotImplementedError, match="FluxPipeline"):
        open_pipeline(str(tmp_path))
    with pytest.raises(ValueError, match="diffusers"):
        architecture(str(tmp_path))


@pytest.mark.parametrize("args,message", [
    (["-W", "640"], "640x512: only 512x512"),
    (["-H", "256"], "512x256: only 512x512"),
    (["--steps", "0"], "at least one"),
])
def test_cli_refuses(args: list[str], message: str) -> None:
    with pytest.raises(SystemExit) as e:
        main(["-m", _model_dir(), "-p", "a cat", *args])
    assert message in str(e.value.code)


@pytest.fixture(scope="module")
def sd() -> StableDiffusion:
    return StableDiffusion(_model_dir(), backend=ref.backend(_model_dir()))


def test_time_biases() -> None:
    w = load_safetensors(f"{_model_dir()}/unet/diffusion_pytorch_model.safetensors", contains=("time_emb", "conv1.weight"))
    biases = resnet_time_biases(w, 517.37)
    assert len(biases) == 22  # 2 per down block, 2 in the middle, 3 per up block
    assert list(biases)[:2] == ["down_blocks.0.resnets.0.", "down_blocks.0.resnets.1."]
    for p, b in biases.items():
        assert b.shape == (w[p + "conv1.weight"].shape[0],) and b.dtype == np.float32
    p = "mid_block.resnets.1."
    t = {k: torch.from_numpy(w[p + k]) for k in ("time_emb_proj.weight", "time_emb_proj.bias")}
    expect = F.linear(F.silu(torch.from_numpy(time_embedding(w, 517.37))), t["time_emb_proj.weight"], t["time_emb_proj.bias"])
    torch.testing.assert_close(torch.from_numpy(biases[p]), expect)
    assert not np.allclose(resnet_time_biases(w, 20.0)[p], biases[p])


def test_text_encoder() -> None:
    """The host's numpy text encoder against the fp32 torch reference, to fp32 rounding; a kept prompt's encoding
    returned again."""
    path = _model_dir()
    tok = ClipTokenizer(f"{path}/tokenizer")
    ids = np.array([tok.encode(""), tok.encode("a photograph of an astronaut riding a horse")], np.int64)
    got = ClipText(f"{path}/text_encoder")
    want = ref.ClipText(f"{path}/text_encoder")(torch.from_numpy(ids)).numpy()
    a = got(ids)
    err = float(np.abs(a - want).max() / np.abs(want).max())
    print(f"text encoder: max |d| {err:.2e} of the largest")
    assert a.shape == (2, 77, 768) and a.dtype == np.float32 and err < 1e-5
    assert np.array_equal(got(ids[1:]), a[1:])


@pytest.mark.slow
def test_txt2img(sd: StableDiffusion) -> None:
    out = sd.txt2img("a photograph of an astronaut riding a horse", steps=2, seed=42)
    assert out.pixels.shape == (512, 512, 3) and out.pixels.dtype.name == "uint8"
    assert out.latents.shape == (1, 4, 64, 64) and np.isfinite(out.latents).all()
    assert out.pixels.std() > 20  # (not a flat image)
    assert min(out.t_text, out.t_steps, out.t_vae) > 0


@pytest.mark.slow
def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "out.png"
    main(["-m", _model_dir(), "-p", "a red fox in the snow", "--steps", "1", "--sampler", "euler", "-s", "1",
          "-o", str(path)])
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (512, 512)
    err = capsys.readouterr().err
    assert "text encoder:" in err and "1 step (euler" in err and "VAE decoder:" in err
    assert os.path.basename(path) in err
