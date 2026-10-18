# artlite-cli

Generates text from a prompt with a GGUF model, or an image with a diffusers pipeline (Stable Diffusion 1.5). The
options follow llama.cpp's `llama-cli` where it has an equivalent, the image options stable-diffusion.cpp's `sd-cli`
(except `--sampler`).

```bash
artlite-cli -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "Factorize x^2 + x - 6 = 0"
artlite-cli -m stable-diffusion-v1-5/stable-diffusion-v1-5 -p "a photograph of an astronaut riding a horse" \
    -o astronaut.png
```

`-m` (required) is a GGUF file or a diffusers directory, or a path relative to the models directory
(`ARTLITE_MODELS_DIR`, default `~/.artlite/models`). A GGUF file's architecture comes from its metadata; the files
that run: [Models](models/README.md). The first run of a model builds its NPU designs
([build cache](install.md#build-cache)).

## Text

The prompt goes through the model's chat template (`--raw`: as it is); tokens are then sampled until an
end-of-sequence token, `-n` tokens or a full KV cache. The answer streams to stdout; the load summary and the prompt
and generation speeds go to stderr.

| option | |
|---|---|
| `-p TEXT`, `--prompt`; `-f FILE`, `--file` | the user message, or a file holding it (one of them is required) |
| `-sys TEXT`, `--system-prompt` | system message |
| `--raw` | the prompt as it is, without the chat template |
| `--think` | let the model think first (Gemma 4, Qwen; default: an empty thought) |
| `-n N`, `--n-predict` | tokens to generate (default -1: until an end-of-sequence token or a full KV cache) |
| `-c N`, `--ctx-size` | KV cache positions (default 0: a power of two >= prompt + `-n` tokens, from 4096) |
| `--temp X` | sampling temperature (default: the model's; 0: greedy) |
| `--top-k N` | sample among the k most likely tokens (default: the model's) |
| `--top-p X` | nucleus sampling threshold (default: the model's) |
| `--greedy` | always the most likely token |
| `-s SEED`, `--seed` | sampling seed (default 1608560892) |
| `--no-prefill` | run the prompt as decode steps |
| `--kv-k single`, `--kv-k hilo` | the KV cache's K precision (Llama, Gemma 4), below |

The model's sampling settings: Llama temperature 0.7, top-k 50; Gemma 4 temperature 1.0, top-k 64, top-p 0.95; Qwen
the file's `general.sampling.*`, else its model card's.

`--kv-k` stores K as one bfp16 value per 8 dims (`single`) or as bfp16 hi + lo (`hilo`: 1.5x the cache and its read
time). Llama's default is `single`: `hilo` makes decode at 32K 1.2x slower with no measured quality gain. Gemma 4's
default is `hilo`: `single` takes 2/3 of the cache at KL 0.00028 instead of 0.00023 against the same file on the CPU.

## Images

A diffusers directory as `-m` generates a 512x512 PNG ([Stable Diffusion 1.5](models/sd15.md)). The PNG goes to
`-o`; the load time and each phase's time go to stderr.

| option | |
|---|---|
| `-p TEXT`, `-f FILE` | the image's description, or a file holding it |
| `--negative-prompt TEXT` | what the image should not show (default: none) |
| `--steps N` | sampling steps (default 20) |
| `--cfg-scale X` | classifier-free guidance scale (default 7.0) |
| `--sampler NAME` | `euler_a` (default), `euler`, `ddim` |
| `-s SEED`, `--seed` | the noise's seed (default 1608560892) |
| `-o FILE`, `--output` | the PNG file (default `output.png`) |
| `-W N`, `--width`; `-H N`, `--height` | image size (512 only for now) |
| `--backend NAME` | `npu` (default: the UNet and the VAE decoder on the NPU) or `cpu` (every network in fp32 torch) |
