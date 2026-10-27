"""artlite-cli: generate text from a prompt with a GGUF model on the NPU, its architecture read from the file
(``models.open_model``), or an image with a diffusers pipeline (``models.open_pipeline``: Stable Diffusion 1.5). The
options follow llama.cpp's ``llama-cli`` where it has an equivalent, the image options stable-diffusion.cpp's
``sd-cli`` (but ``--sampler``).

    artlite-cli -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf -p "..."
    artlite-cli -m openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf \\
        --mmproj openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf --image photo.jpg -p "Describe the image."
    artlite-cli -m stable-diffusion-v1-5/stable-diffusion-v1-5 -p "..." -o image.png

Text: the prompt goes through the model's chat template (``--raw``: as it is), with ``--image`` files (a model with a
vision encoder: ``--mmproj``) as llama.cpp's ``llama-mtmd-cli`` places them; tokens are then sampled until an
end-of-sequence token, ``-n`` tokens or a full KV cache. The text streams to stdout, the load and speed report to
stderr. Images: the PNG goes to ``-o``, the load time and each phase's time to stderr.
"""

import argparse
import dataclasses
import os
import sys
import time

from artlite.model_config import MODELS_DIR, model_path

from . import is_diffusers, open_model, open_pipeline
from .common import MEDIA_MARKER, StreamDecoder, auto_capacity, performance_mode, seed, stale_install
from .sd15.pipeline import BACKENDS, check_request
from .sd15.png import write_png
from .sd15.sampling import SAMPLERS


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="artlite-cli", description="Generate text with a GGUF model on the NPU, or an "
                                "image with a diffusers pipeline (Stable Diffusion 1.5)")
    p.add_argument("-m", "--model", required=True, type=model_path,
                   help=f"GGUF file or diffusers directory; a relative path is also looked up in the models directory "
                        f"({MODELS_DIR}, ARTLITE_MODELS_DIR)")
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("-p", "--prompt", help="the user message (an image's description)")
    prompt.add_argument("-f", "--file", help="read the user message from this file")
    p.add_argument("-sys", "--system-prompt", help="system message")
    p.add_argument("--raw", action="store_true", help="the prompt is the raw text (no chat template)")
    p.add_argument("--think", action="store_true", help="let the model think first (default: an empty thought)")
    p.add_argument("-n", "--n-predict", type=int, default=-1,
                   help="tokens to generate (default: -1, until an end-of-sequence token or a full KV cache)")
    p.add_argument("-c", "--ctx-size", type=int, default=0,
                   help="KV cache positions (default: 0, a power of two >= prompt + n-predict tokens, from 4096)")
    p.add_argument("--temp", type=float, help="sampling temperature (default: the model's; 0: greedy)")
    p.add_argument("--top-k", type=int, help="sample among the k most likely tokens (default: the model's)")
    p.add_argument("--top-p", type=float, help="nucleus sampling threshold (default: the model's)")
    p.add_argument("--repeat-penalty", type=float,
                   help="divide the logits of the last --repeat-last-n tokens by this (default: the model's; 1: none)")
    p.add_argument("--repeat-last-n", type=int,
                   help="the tokens the penalties look back at (default: the model's, 64; 0: none)")
    p.add_argument("--presence-penalty", type=float,
                   help="subtract this from the logits of the last --repeat-last-n tokens (default: the model's, 0)")
    p.add_argument("--frequency-penalty", type=float, help="subtract this times their count from the logits of the "
                   "last --repeat-last-n tokens (default: the model's, 0)")
    p.add_argument("--greedy", action="store_true", help="always the most likely token")
    p.add_argument("-s", "--seed", type=int, default=1608560892, help="sampling seed")
    p.add_argument("--no-prefill", action="store_true", help="run the prompt as decode steps")
    p.add_argument("--no-lookup", action="store_true",
                   help="no draft tokens from the context (prompt lookup) in the multi-token steps")
    p.add_argument("--kv-k", choices=("single", "hilo"),
                   help="K precision of the KV cache (Llama, Gemma): one bfp16 block per 8 dims (Llama's default), or "
                        "hi + lo blocks (Gemma's default; +50%% cache bytes and read time)")
    vis = p.add_argument_group("images in the prompt (MiniCPM-V 4.7)")
    vis.add_argument("--mmproj", type=model_path, help="the vision encoder's GGUF file (mmproj-model-f16.gguf)")
    vis.add_argument("--image", action="append", default=[],
                     help=f"an image file (repeatable): where the prompt has {MEDIA_MARKER}, else before it")
    vis.add_argument("--max-slices", type=int,
                     help="an image's slices at most (default: the mmproj's, 9; the model card uses 36)")
    img = p.add_argument_group("images (a diffusers directory)")
    img.add_argument("--negative-prompt", default="", help="what the image should not show (default: none)")
    img.add_argument("--steps", type=int, default=20, help="sampling steps (default: 20)")
    img.add_argument("--cfg-scale", type=float, default=7.0, help="classifier-free guidance scale (default: 7.0)")
    img.add_argument("--sampler", choices=SAMPLERS, default="euler_a", help="sampling method (default: euler_a)")
    img.add_argument("-o", "--output", default="output.png", help="the PNG file to write (default: output.png)")
    img.add_argument("-W", "--width", type=int, default=512, help="image width (512 only for now)")
    img.add_argument("-H", "--height", type=int, default=512, help="image height (512 only for now)")
    img.add_argument("--backend", choices=BACKENDS, default="npu", help="what runs the networks (default: npu)")
    return p


def prompt_text(args: argparse.Namespace) -> str:
    """The prompt of ``-p``, or the contents of ``-f``."""
    if args.file:
        with open(args.file) as f:
            return f.read()
    return args.prompt


def generate_image(args: argparse.Namespace) -> None:
    """Text to image with a diffusers pipeline: the PNG to ``-o``, the load time and each phase's time to stderr."""
    name = os.path.basename(os.path.normpath(args.model))
    t0 = time.perf_counter()
    try:
        check_request(width=args.width, height=args.height, steps=args.steps)
        text = prompt_text(args)
        sd = open_pipeline(args.model, backend=args.backend)
    except (OSError, ValueError, NotImplementedError) as e:
        sys.exit(f"artlite-cli: {e}")
    sys.stderr.write(f"{name} ({sd.name}, {args.backend} backend): loaded in {time.perf_counter() - t0:.1f} s\n")
    tty = sys.stderr.isatty()

    def progress(step: int, steps: int) -> None:
        if tty:
            sys.stderr.write(f"\rstep {step}/{steps}" + ("\n" if step == steps else ""))
            sys.stderr.flush()

    out = sd.txt2img(text, negative_prompt=args.negative_prompt, steps=args.steps, cfg_scale=args.cfg_scale,
                     sampler=args.sampler, seed=args.seed, width=args.width, height=args.height, progress=progress)
    try:
        write_png(args.output, out.pixels)
    except OSError as e:
        sys.exit(f"artlite-cli: {e}")
    steps = f"{args.steps} step{'s' * (args.steps > 1)}"
    sys.stderr.write(f"text encoder: {out.t_text:.2f} s; {steps} ({args.sampler}, cfg scale {args.cfg_scale:g}): "
                     f"{out.t_steps:.2f} s ({out.t_steps / args.steps:.2f} s per step); VAE decoder: {out.t_vae:.2f} s\n"
                     f"{args.output}: {args.width}x{args.height}, seed {args.seed}\n")


def main(argv: list[str] | None = None) -> None:
    args = arg_parser().parse_args(argv)
    if stale := stale_install():
        sys.stderr.write(f"artlite-cli: warning: {stale}\n")
    performance_mode(lambda line: sys.stderr.write(line + "\n"))
    if is_diffusers(args.model):
        generate_image(args)
        return
    seed(args.seed)
    name = os.path.basename(args.model)
    t0 = time.perf_counter()
    options = {"kv_k": args.kv_k, "mmproj": args.mmproj, "max_slices": args.max_slices}
    try:
        text = prompt_text(args)
        model = open_model(args.model, **{k: v for k, v in options.items() if v})
        ids = model.prompt(text, system=args.system_prompt, raw=args.raw, think=args.think, images=args.image)
        sampling = model.sampling(think=args.think)
        given = {"temp": args.temp, "top_k": args.top_k, "top_p": args.top_p, "repeat_penalty": args.repeat_penalty,
                 "repeat_last_n": args.repeat_last_n, "presence_penalty": args.presence_penalty,
                 "frequency_penalty": args.frequency_penalty}
        sampling = dataclasses.replace(sampling, **{k: v for k, v in given.items() if v is not None},
                                       greedy=sampling.greedy or args.greedy)
    except (OSError, ValueError, NotImplementedError) as e:
        sys.exit(f"artlite-cli: {e}")
    if model.vision is not None and args.image:
        v = model.vision
        sys.stderr.write(f"{len(args.image)} image(s): {v.entries} entries through the encoder in {v.t_encode:.2f} s "
                         f"({1e3 * v.t_encode / max(v.entries, 1):.0f} ms each)\n")
    capacity = args.ctx_size or auto_capacity(len(ids) + max(args.n_predict, 0))
    if len(ids) >= capacity:
        sys.exit(f"artlite-cli: a prompt of {len(ids)} tokens does not fit a KV cache of {capacity} positions")
    try:
        npu = model.load(capacity=capacity, prefill=not args.no_prefill)
    except NotImplementedError as e:
        sys.exit(f"artlite-cli: {name}: not supported yet: {e}")
    npu.lookup = not args.no_lookup
    npu.prepare(len(ids))
    sys.stderr.write(f"{name} ({model.name}): {len(ids)} prompt tokens; loaded in {time.perf_counter() - t0:.1f} s: "
                     f"{npu.describe()}\n")
    out = StreamDecoder(model.tokenizer)
    t0 = time.perf_counter()
    logits = npu.forward(ids)
    t1 = time.perf_counter()
    npu_s, host_s, passes = npu.t_npu, npu.t_host, npu.dispatches
    n = 0
    for token in npu.generate(logits, args.n_predict, sampling, ids):
        n += 1
        if token not in model.eos_token_ids:
            print(out.feed(token), end="", flush=True)
    t2 = time.perf_counter()
    passes = max(npu.dispatches - passes, 1)
    host = f", {(npu.t_host - host_s) / passes * 1e3:.1f} ms of host inputs each" if npu.t_host > host_s else ""
    sys.stderr.write(f"\n\nprompt: {len(ids)} tokens in {t1 - t0:.2f} s ({len(ids) / (t1 - t0):.1f} tokens/s); "
                     f"generation: {n} tokens in {t2 - t1:.2f} s ({n / max(t2 - t1, 1e-9):.2f} tokens/s): {passes} "
                     f"dispatches of {(npu.t_npu - npu_s) / passes * 1e3:.1f} ms{host}"
                     f"{'; ' + npu.stats() if npu.stats() else ''}\n")


if __name__ == "__main__":
    main()
