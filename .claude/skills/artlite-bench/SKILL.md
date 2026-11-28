---
name: artlite-bench
description: "Measure a model's numbers for the README's performance charts (docs/data/artlite.json) the same way every time: TTFT at 1k / 8k / 32k / 64k and prefill speed with artlite-bench, decode speed at 1k / 32k with artlite-cli on essay prompts, for a model with a drafter (MTP) the best tokens per dispatch S at each length, and for a model with a vision encoder an image prompt's TTFT through artlite-server (and other runtimes' servers). Use when asked to benchmark a model, add or refresh a model's chart numbers, or compare prefill / decode / MTP / image-prompt speeds across days, models or runtimes."
---

# Benchmarking a model for the README charts

A model's entry in `docs/data/artlite.json` and where each number comes from (keys are prompt lengths in tokens):

| field | tool | value |
|---|---|---|
| `ttft_s`: 1024 8192 32768 65536 (and 131072 where a compared runtime has it) | `artlite-bench -p 1024,8192,32768,65536 -n 0` | `avg_ns` / 1e9 of `pp<n>` (the prompt's chunks plus the last token's logits) |
| `prefill_tps`: 1024 32768 | the same run | `avg_ts` of `pp1024`, `pp32768` |
| `decode_tps`: 1024 32768 | `artlite-cli` on an essay prompt of 1023 / 32767 tokens, `-n 256 --greedy --no-lookup` | its `generation: ... tokens/s` |
| `mtp`: `decode_tps`, `tokens_per_step` | the decode runs, repeated per draft count | the best S at each length, and its speed |
| `image_ttft_s`: encoder, then image | `image_ttft.py` against `artlite-server --mmproj FILE` | the median per image (section 5) |

Decode is not taken from `artlite-bench`'s `tg` tests: their tokens are random, so a drafter's guesses are never
accepted and MTP cannot show its effect; `tg` also times `top_k` steps rather than the app's `generate`. Decode runs
greedy, so the text, and with it the drafts' acceptance, is the same on every run (sampling changes the text per seed
and per S).

## 0. A quiet machine

Run one NPU job at a time, with no other heavy CPU work (stop a VS Code cpptools indexer or a build). Each step below
loads the model in a separate process, and the first load after a code change compiles the designs (cached in
`~/.cache/artlite`). Run a throwaway `artlite-cli -n 8` first when the cache may be cold; `artlite-bench` warms up by
itself.

Embedding rows come from the mapped GGUF file (`GGUFFile.rows`), and a table that is not part of the NPU image (Llama
8B's untied `token_embd`, Qwen's, Gemma's per-layer table) is not read at load time. After a reboot, or on a machine
where other models' files evicted it from the page cache, a test's random prompt tokens read their rows from disk.
`GGUFFile.rows` requests all of a lookup's rows at once (1024 of Gemma's per-layer rows take ~15 ms cold vs. 152 ms
when they faulted in one by one, which dropped Gemma's pp1024 from ~2700 to 1100-2000 t/s). Still, read the file once
first (`cat MODEL.gguf > /dev/null`).

## 1. TTFT and prefill

```bash
artlite-bench -m MODEL.gguf -p 1024,8192,32768,65536 -n 0 -r 3 -o jsonl --progress > pp.jsonl
```

This is one load, with a KV cache sized for the largest test. A length beyond the model's context (or its prefill's
limit) is left out of the entry; drop it from `-p`, or the run stops there. Repetitions agree within ~1%, so `-r 3` is
enough.

## 2. Prompts

```bash
python .claude/skills/artlite-bench/prompts.py MODEL.gguf OUT_DIR          # prompt1024.txt, prompt32768.txt
```

Each prompt is Paul Graham essays (`tests/datasets/paul_graham_essays`, in file-name order) followed by a request to
summarize them, cut so that the chat prompt is N - 1 tokens with the model's template. The essays are the same for
every model; the cut depends on the tokenizer.

## 3. Decode

```bash
artlite-cli -m MODEL.gguf -f OUT_DIR/prompt1024.txt  -n 256 --greedy --no-lookup > /dev/null   # (the speeds go to stderr)
artlite-cli -m MODEL.gguf -f OUT_DIR/prompt32768.txt -n 256 --greedy --no-lookup > /dev/null
```

The stderr line `prompt: ... generation: N tokens in T s (R tokens/s): D dispatches of X ms ...` gives R. If the
model stops early (N well below 256 because of an end-of-sequence token), note that next to the number. The prompt
speed on that line cross-checks step 1 (its TTFT excludes tokenization, as in step 1), minus the ~5-10 ms that a
process's first prompt spends assembling its chunks' control code, which artlite-bench's warmup absorbs.

## 4. MTP: the best S

For a model with a drafter, S = drafts + 1 tokens per dispatch. The drafters are:

- Gemma 4: its `mtp-<file>` next to the model.
- Qwen3.8 27B: z-lab's DFlash2 file (`model_config.qwen_dflash_of`); `--spec-type draft-mtp` uses the MTP layer inside
  the model file instead.
- Qwen3.6 35B-A3B: ggml-org's `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` (`model_config.qwen_mtp_of`).
- `-md FILE` selects another drafter.

A block drafter's rounds depend only on the text, so a run repeats exactly. An MTP chain's length follows the run's
step times (Qwen3.8 27B ranges over 7.5-8.1 tok/s at 1K and 6.1-7.0 at 32K on one prompt), so compare drafters on
several prompts, not on one. Any change to the drafts or to a step's grouping also re-rolls greedy text at its near
ties, and with it a single prompt's rate (Qwen3.8 27B's DFlash2 branches: the 32K prompt went from 6.56 to 6.51 tok/s
on a harder text, while the same texts gained +4.7%). By default `artlite-cli` drafts up to the maximum, using per step
only the drafts that pay off (`models.drafts`); `--spec-draft-n-max D` caps it (D = 0 runs steps of one token without
loading a drafter, which gives the entry's `decode_tps`):

```bash
for D in 0 1 2 3; do
  artlite-cli -m MODEL.gguf -f OUT_DIR/prompt1024.txt -n 256 --greedy --no-lookup --spec-draft-n-max $D 2>&1 >/dev/null | grep prompt:
done   # and the same for prompt32768.txt
```

Sweep every D at both lengths, because the best S depends on the context. An S-token dispatch costs about the same as
one token at short contexts (the weights stream once), but its attention cost grows with S at long contexts, while the
drafts' acceptance falls with their depth. The `D draft tokens (X ms each), A accepted` part of the line explains a
result: the acceptance per draft and the drafter's own cost (which also grows with the context).

## 5. Image prompts (a model with a vision encoder)

```bash
artlite-server -m MODEL.gguf --mmproj MMPROJ.gguf --port 8080 --no-ui &      # (one per encoder file)
python .claude/skills/artlite-bench/image_ttft.py http://127.0.0.1:8080 IMAGE ...
```

The images are the `image_prompts` in `docs/data/artlite.json` (their sources are listed in `docs/methodology.md`).
The script sends each image with fresh pixels, so neither kept embeddings nor the prompt cache can serve a request (it
stops if a server reports cached tokens). After a warmup, it prints the median of 3 runs per image, which reproduces
within ~2%. Stop the server with SIGTERM, which waits for its NPU dispatch to finish.

Other runtimes' servers work with the same script: `llama-server -m MODEL.gguf --mmproj MMPROJ.gguf -c 32768 -np 1
-ngl 99` (Vulkan), and `flm serve <tag> --ctx-len <native> --prefill-chunk-len 4096` with the NPU in turbo mode. An
iGPU's sessions (a server start and its requests) differ by up to 8%, so take the best of 3 after the machine has idled.

## 6. The entry and the charts

```json
{
  "model": "Gemma 4 E4B it",
  "quant": "Q4_0",
  "ttft_s": {"1024": 0.38, "8192": 2.63, "32768": 12.9, "65536": 33.7},
  "prefill_tps": {"1024": 2716, "32768": 2530},
  "decode_tps": {"1024": 16.7, "32768": 13.7},
  "mtp": {"decode_tps": {"1024": 28.8, "32768": 16.9}, "tokens_per_step": {"1024": 4, "32768": 2}}
}
```

- `model` names the chart group: use the same string for every file of a model (groups keep the file's order).
- Write TTFT with 3 significant digits (0.44, 3.14, 16.7), prefill in whole tok/s, and decode with one decimal.
  Without a drafter, omit `mtp`.
- Image prompts: `"image_ttft_s": {"16x": {"airplane.jpeg": 0.570, ...}, "4x": {...}}`, with the encoders and images
  taken from `image_prompts` (a new image there needs its size and slice grid, from
  `models.minicpmv.host.slice_image`'s `Grid.grid`).
- A run that needs an explanation goes in `notes` (`"notes": {"decode_tps": {"1024": "stopped at an end-of-sequence
  token after 143 of the 256 tokens"}}`), which becomes a footnote on that bar.

Then `python docs/charts.py` redraws `docs/images` (light and dark SVGs; commit both), and the model table in
`docs/models/README.md` gets the file. Add the file's row to the "Measured files" table in `docs/methodology.md`: the
Hugging Face repository and file it came from (match its sha256 against the repository's LFS object ids, because a
local directory's name is not proof), the first 12 hex digits of the sha256, and the date of the runs (plus that
date's toolchain pins in the table below it). Update "Test system" when the machine, driver or firmware changed. Write
today's numbers only from runs of the current commit.

Another runtime's numbers go in a separate `docs/data/<runtime>.json` with the same fields. Two fields pair its rows
with Artlite's: `comparable_quants` (the GGUF quants of the same model that its row is compared with, first match
first) and `same_shapes_as` (chart it under a model with the same architecture and sizes).
