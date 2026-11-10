# artlite-cli

Generates text from a prompt with a GGUF model, or an image with a diffusers pipeline (Stable Diffusion 1.5). The
options follow llama.cpp's `llama-cli` where it has an equivalent, the image options stable-diffusion.cpp's `sd-cli`
(except `--sampler`).

```bash
artlite-cli -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "Factorize x^2 + x - 6 = 0"
artlite-cli -m openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf --image photo.jpg -p "Describe the image."
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
| `--think` | let the model think first (Gemma 4, Qwen, LFM3; default: an empty thought) |
| `--reasoning-effort LEVEL` | the chat template's `reasoning_effort`, with `--think` (Qwen3.8: `xhigh`, `medium`, `low`; default `default`: the template's own) |
| `--chat-template-kwargs JSON` | chat template variables as a JSON object (`{"tools_in_user_message": false}`, `{"date_string": "26 Jul 2024"}`, ...) |
| `--chat-template JINJA_TEMPLATE`, `--chat-template-file FILE` | a Jinja chat template, or a file holding one, in place of the model file's (`tokenizer.chat_template`); it gets the variables the file's gets (`enable_thinking`, `bos_token`, ...) |
| `-n N`, `--n-predict` | tokens to generate (default -1: until an end-of-sequence token or a full KV cache) |
| `-c N`, `--ctx-size` | KV cache positions (default 0: a power of two >= prompt + `-n` tokens, from 4096) |
| `--temp X` | sampling temperature (default: the model's; 0: greedy) |
| `--top-k N` | sample among the k most likely tokens (default: the model's) |
| `--top-p X` | nucleus sampling threshold (default: the model's) |
| `--repeat-penalty X` | divide the logits of the last `--repeat-last-n` tokens, the prompt's too, by X (multiply the negative ones) (default: the model's; 1: none) |
| `--repeat-last-n N` | the tokens the penalties look back at (default: the model's, 64; 0: none) |
| `--presence-penalty X`, `--frequency-penalty X` | subtract X, or X times their count, from those tokens' logits (default: the model's, 0) |
| `--greedy` | always the most likely token |
| `-s SEED`, `--seed` | sampling seed (default 1608560892) |
| `--no-prefill` | run the prompt as decode steps |
| `--no-lookup` | no draft tokens from the prompt, below |
| `--spec-type auto\|none\|draft-mtp\|draft-dflash` | the model's drafter, below (llama-server's names): `auto` (default) the one found by its file name, `none` no drafter (its file is not loaded, Qwen3.8 27B's MTP layer is left out of the weights), `draft-mtp` an MTP drafter, `draft-dflash` a DFlash2 block drafter (Qwen3.8 27B: [qwen35](models/qwen35.md#multi-token-prediction-qwen38-27b-qwen36-35b-a3b-occamy-10)); one asked for that the model does not have is an error |
| `-md FILE`, `--spec-draft-model` | the drafter's GGUF file in place of the one found by name (an MTP or a DFlash2 drafter, by its architecture) |
| `--spec-draft-n-max N` | draft tokens per step at most, from the prompt or the drafter (default and at most: the step's slots - 1, 3 or Qwen3.8 27B's 7; 0: steps of one token, no drafter loaded) |
| `--kv-k single`, `--kv-k hilo` | the KV cache's K precision (Llama, Qwen3, Gemma 4, LFM3), below |
| `--mmproj FILE` | the vision encoder's file (Qwen3.5 / 3.6 / 3.8: `mmproj-<model>-BF16.gguf`; MiniCPM-V 4.7: `mmproj-model-f16.gguf`): images in the prompt |
| `--image FILE` | an image (repeatable), where the prompt has `<__media__>` (llama.cpp's marker), else before it ([Qwen](models/qwen35.md#images-qwen35--36--38), [MiniCPM-V 4.7](models/qwen35.md#images-minicpm-v-47)) |
| `--image-min-tokens N`, `--image-max-tokens N` | an image's tokens of 32 x 32 pixels at least / at most (Qwen3-VL's and LFM3-VL's mmproj; default 8 / 4096 and 64 / 4096, llama.cpp's) |
| `--max-slices N` | an image's slices at most (MiniCPM-V; default: the mmproj's, 9; the model card uses 36) |

The model's sampling settings: Llama temperature 0.7, top-k 50; Gemma 4 temperature 1.0, top-k 64, top-p 0.95; Qwen3
its model card's (thinking: temperature 0.6, top-k 20, top-p 0.95; else 0.7, 20, 0.8); Qwen3.5 and later and LFM3
the file's `general.sampling.*`, else the model card's (LFM3: temperature 0.1, top-k 50, repetition penalty 1.1).

`--kv-k` stores K as one bfp16 value per 8 dims (`single`) or as bfp16 hi + lo (`hilo`: 1.5x the cache and its read
time). Llama's default is `single`: `hilo` makes decode at 32K 1.2x slower with no measured quality gain. Gemma 4's
default is `hilo`: `single` takes 2/3 of the cache at KL 0.00028 instead of 0.00023 against the same file on the CPU.
Qwen3's default is `hilo`: its K norm's weights (up to ~100) leave dims a thousand times below their block's largest,
and `single` takes the KL against the same file on the CPU from 0.0037 to 0.0085 (Qwen3 0.6B, a chat). LFM3's default
is `single`: its q / k norm weights stay below 2.5, and at 32K `single` costs a KL of 1.4e-4 against `hilo`'s 1e-5 for
a third less of the cache.

Generation runs in steps of up to 4 tokens (Qwen3.8 27B: 8): the next token and draft tokens after it, which a step checks at a
fraction of the cost of their own steps (the weights stream once). The drafts come from the prompt where the text so
far repeats a part of it (prompt lookup: an edit's old text, a file rewritten, a list of names again), or from a
model's drafter (multi-token prediction: Gemma 4's `mtp-` file, Qwen3.8 27B's DFlash2 file, else its own MTP layer,
Qwen3.6 35B-A3B's and Occamy's `mtp` files); each step takes the ones that pay at the model's measured step times, none where
nothing does. A draft is kept only where the model's own
token at its position is that draft, so the text is the one without drafts (greedy: the same tokens, except at near
ties on Qwen3.6 35B-A3B and Occamy, whose steps of several tokens sum their experts in another order). The summary on
stderr counts each kind's draft tokens and the accepted ones.

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
