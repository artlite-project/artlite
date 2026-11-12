"""artlite-cli: generate text from a prompt with a GGUF model on the NPU, its architecture read from the file
(``models.open_model``), or an image with a diffusers pipeline (``models.open_pipeline``: Stable Diffusion 1.5). The
options follow llama.cpp's ``llama-cli`` where it has an equivalent, the image options stable-diffusion.cpp's
``sd-cli`` (but ``--sampler``).

    artlite-cli -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf -p "..."
    artlite-cli -m openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf \\
        --mmproj openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf --image photo.jpg -p "Describe the image."
    artlite-cli -m stable-diffusion-v1-5/stable-diffusion-v1-5 -p "..." -o image.png

Text: the prompt goes through the model's chat template (``--raw``: as it is), with ``--image`` and ``--audio`` files (a
model with a vision or speech encoder: ``--mmproj``) as llama.cpp's ``llama-mtmd-cli`` places them (``--audio`` alone:
a transcription, the system message llama.cpp's ASR prompt for ``--language``); tokens are then sampled until an
end-of-sequence token, ``-n`` tokens or a full KV cache. The text streams to stdout, the load and speed report to
stderr. Images: the PNG goes to ``-o``, the load time and each phase's time to stderr.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import jinja2

from artlite.model_config import MODELS_DIR, model_path

from . import is_diffusers, open_model, open_pipeline
from .common import (
    MEDIA_MARKER,
    TEMPLATE_RESERVED,
    StreamDecoder,
    add_spec_options,
    auto_capacity,
    performance_mode,
    seed,
    spec_options,
    stale_install,
    template_option,
)
from .lfm3a.host import asr_system
from .sd15.pipeline import check_request
from .sd15.png import write_png
from .sd15.sampling import SAMPLERS


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="artlite-cli", description="Generate text with a GGUF model on the NPU, or an "
                                "image with a diffusers pipeline (Stable Diffusion 1.5)")
    p.add_argument("-m", "--model", required=True, type=model_path,
                   help=f"GGUF file or diffusers directory; a relative path is also looked up in the models directory "
                        f"({MODELS_DIR}, ARTLITE_MODELS_DIR)")
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument("-p", "--prompt", help="the user message (an image's description)")
    prompt.add_argument("-f", "--file", help="read the user message from this file")
    p.add_argument("-sys", "--system-prompt", help="system message")
    p.add_argument("--raw", action="store_true", help="the prompt is the raw text (no chat template)")
    p.add_argument("--think", action="store_true", help="let the model think first (default: an empty thought)")
    p.add_argument("--reasoning-effort", metavar="LEVEL", default="default",
                   help="the chat template's reasoning_effort, with --think (Qwen3.8's: xhigh, medium, low; default: "
                        "default, the template's own)")
    p.add_argument("--chat-template-kwargs", type=json.loads, default={},
                   help="chat template variables as a JSON object (reasoning_effort, tools_in_user_message, ...)")
    p.add_argument("--chat-template", metavar="JINJA_TEMPLATE",
                   help="a Jinja chat template in place of the model file's (tokenizer.chat_template)")
    p.add_argument("--chat-template-file", metavar="FILE", help="the same from a file")
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
    add_spec_options(p)
    p.add_argument("--kv-k", choices=("single", "hilo"),
                   help="K precision of the KV cache (Llama, Qwen3, Gemma, LFM3): one bfp16 block per 8 dims (Llama's "
                        "and LFM3's default), or hi + lo blocks (Qwen3's and Gemma's default; +50%% cache bytes and read "
                        "time)")
    vis = p.add_argument_group("images in the prompt (Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7, LFM3-VL)")
    vis.add_argument("--mmproj", type=model_path, help="the vision or speech encoder's GGUF file (mmproj-*.gguf)")
    vis.add_argument("--image", action="append", default=[],
                     help=f"an image file (repeatable): where the prompt has {MEDIA_MARKER}, else before it")
    vis.add_argument("--max-slices", type=int,
                     help="an image's slices at most (MiniCPM-V; default: the mmproj's, 9; the model card uses 36)")
    vis.add_argument("--image-min-tokens", type=int,
                     help="an image's tokens at least (Qwen3-VL's, LFM3-VL's mmproj; default 8 and 64, llama.cpp's)")
    vis.add_argument("--image-max-tokens", type=int,
                     help="an image's tokens at most (Qwen3-VL's, LFM3-VL's mmproj; default 4096, llama.cpp's)")
    aud = p.add_argument_group("audio in the prompt (LFM3-Audio)")
    aud.add_argument("--audio", action="append", default=[],
                     help=f"an audio file (repeatable; WAV, MP3, FLAC, Ogg Vorbis): where the prompt has {MEDIA_MARKER}, "
                          "else before it, after the images; without -p / -f a transcription (ASR)")
    aud.add_argument("--language", default="",
                     help="the transcription's language (a code: en, de, ...; or a name; default: the model detects it)")
    aud.add_argument("--live", nargs="?", const="fast", choices=("fast", "realtime"),
                     help="live transcription (English) of the --audio files, the text while the audio streams in: as "
                          "fast as it goes, or paced as a microphone's (realtime: each chunk's latency)")
    img = p.add_argument_group("images (a diffusers directory)")
    img.add_argument("--negative-prompt", default="", help="what the image should not show (default: none)")
    img.add_argument("--steps", type=int, default=20, help="sampling steps (default: 20)")
    img.add_argument("--cfg-scale", type=float, default=7.0, help="classifier-free guidance scale (default: 7.0)")
    img.add_argument("--sampler", choices=SAMPLERS, default="euler_a", help="sampling method (default: euler_a)")
    img.add_argument("-o", "--output", default="output.png", help="the PNG file to write (default: output.png)")
    img.add_argument("-W", "--width", type=int, default=512, help="image width (512 only for now)")
    img.add_argument("-H", "--height", type=int, default=512, help="image height (512 only for now)")
    return p


def prompt_text(args: argparse.Namespace) -> str:
    """The prompt of ``-p``, or the contents of ``-f``; "" for a transcription (``--audio`` alone)."""
    if args.file:
        with open(args.file) as f:
            return f.read()
    if args.prompt is None:
        if not getattr(args, "audio", None):
            raise ValueError("a prompt: -p or -f (or --audio files to transcribe)")
        return ""
    return args.prompt


def generate_image(args: argparse.Namespace) -> None:
    """Text to image with a diffusers pipeline: the PNG to ``-o``, the load time and each phase's time to stderr."""
    name = os.path.basename(os.path.normpath(args.model))
    t0 = time.perf_counter()
    try:
        check_request(width=args.width, height=args.height, steps=args.steps)
        text = prompt_text(args)
        sd = open_pipeline(args.model)
    except (OSError, ValueError, NotImplementedError) as e:
        sys.exit(f"artlite-cli: {e}")
    sys.stderr.write(f"{name} ({sd.name}): loaded in {time.perf_counter() - t0:.1f} s\n")
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


def live(args: argparse.Namespace, options: dict[str, object]) -> None:
    """``--live``: each --audio file as a stream (``models.lfm3a.live``), pushed 0.1 s at a time; the text to stdout,
    the chunks' times to stderr."""
    from .lfm3a.host import HOP, N_FFT, SR, STRIDE, load_audio
    from .lfm3a.live import CHUNK_ROWS, LiveAsr

    row = STRIDE * HOP  # (samples of a row)

    if not args.audio or args.prompt is not None or args.file:
        sys.exit("artlite-cli: --live transcribes --audio files (no -p / -f)")
    t0 = time.perf_counter()
    try:
        model = open_model(args.model, **{k: v for k, v in options.items() if v is not None})
        if model.audio is None:
            raise ValueError("--live needs the model's speech encoder (--mmproj: LFM3-Audio's)")
        clips = [load_audio(a) for a in args.audio]
        npu = model.load(capacity=args.ctx_size or 2048, prefill=not args.no_prefill)
        asr = LiveAsr(model, npu, model.audio.encoder)
    except (OSError, ValueError, NotImplementedError) as e:
        sys.exit(f"artlite-cli: {e}")
    npu.prepare(64)
    sys.stderr.write(f"{os.path.basename(args.model)} ({model.name}): live ASR; loaded in {time.perf_counter() - t0:.1f} "
                     f"s: {npu.describe()}\n")
    step = SR // 10
    for x in clips:
        asr.start()
        lat: list[float] = []  # (realtime: from the end of the audio a chunk's rows need to its text)
        t_npu, cpu0 = npu.t_npu, time.process_time()
        t0 = time.perf_counter()
        for i in range(0, len(x), step):
            if args.live == "realtime":
                time.sleep(max(0.0, t0 + (i + step) / SR - time.perf_counter()))
            n = asr.chunks
            print(asr.push(x[i:i + step]), end="", flush=True)
            if args.live == "realtime" and asr.chunks > n:
                lat.append(time.perf_counter() - t0 - min(len(x), row * asr.stream.taken + N_FFT - HOP) / SR)
        print(asr.end(), flush=True)
        dt, audio, cpu = time.perf_counter() - t0, len(x) / SR, time.process_time() - cpu0
        st = asr.stream
        busy = st.t_encode + npu.t_npu - t_npu
        sys.stderr.write(f"\n{audio:.1f} s of audio in {dt:.2f} s: {asr.chunks} chunks of {CHUNK_ROWS} rows (0.96 s); "
                         f"the encoder: {st.dispatches} dispatches in {st.t_encode:.2f} s; the text: {asr.t_text:.2f} s; "
                         f"the NPU busy {busy:.2f} s ({100 * busy / audio:.1f}% of the audio's time), the host's CPU "
                         f"{cpu:.2f} s ({100 * cpu / audio:.1f}%)"
                         + (f"; latency: median {1e3 * sorted(lat)[len(lat) // 2]:.0f} ms, max {1e3 * max(lat):.0f} ms"
                            if lat else "") + "\n")


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
    options = {"kv_k": args.kv_k, "mmproj": args.mmproj, "max_slices": args.max_slices,
               "image_min_tokens": args.image_min_tokens, "image_max_tokens": args.image_max_tokens, **spec_options(args)}
    variables = {k: v for k, v in args.chat_template_kwargs.items() if k not in TEMPLATE_RESERVED} | \
        ({} if args.reasoning_effort == "default" else {"reasoning_effort": args.reasoning_effort})
    if args.live:
        live(args, options)
        return
    try:
        text = prompt_text(args)
        template = template_option(args.chat_template, args.chat_template_file)
        model = open_model(args.model, chat_template=template, **{k: v for k, v in options.items() if v is not None})
        asr = bool(args.audio) and args.prompt is None and not args.file
        system = args.system_prompt or (asr_system(args.language) if asr else None)
        ids = model.prompt(text, system=system, raw=args.raw, think=args.think, images=args.image, audio=args.audio,
                           variables=variables)
        sampling = model.sampling(think=args.think)
        given = {"temp": args.temp, "top_k": args.top_k, "top_p": args.top_p, "repeat_penalty": args.repeat_penalty,
                 "repeat_last_n": args.repeat_last_n, "presence_penalty": args.presence_penalty,
                 "frequency_penalty": args.frequency_penalty}
        sampling = dataclasses.replace(sampling, **{k: v for k, v in given.items() if v is not None},
                                       greedy=sampling.greedy or args.greedy)
    except jinja2.TemplateError as e:  # (the template refuses the prompt or a variable)
        sys.exit(f"artlite-cli: the chat template failed: {e}")
    except (OSError, ValueError, NotImplementedError) as e:
        sys.exit(f"artlite-cli: {e}")
    if model.vision is not None and args.image:
        v = model.vision
        sys.stderr.write(f"{len(args.image)} image(s): {v.entries} entries through the encoder in {v.t_encode:.2f} s "
                         f"({1e3 * v.t_encode / max(v.entries, 1):.0f} ms each)\n")
    if model.audio is not None and args.audio:
        a = model.audio
        sys.stderr.write(f"{len(args.audio)} audio clip(s) through the encoder in {a.t_encode:.2f} s\n")
    capacity = args.ctx_size or auto_capacity(len(ids) + max(args.n_predict, 0))
    if len(ids) >= capacity:
        sys.exit(f"artlite-cli: a prompt of {len(ids)} tokens does not fit a KV cache of {capacity} positions")
    try:
        npu = model.load(capacity=capacity, prefill=not args.no_prefill)
    except NotImplementedError as e:
        sys.exit(f"artlite-cli: {name}: not supported yet: {e}")
    npu.lookup = not args.no_lookup
    if args.spec_draft_n_max is not None:
        npu.DRAFTS = min(npu.DRAFTS, args.spec_draft_n_max)
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
