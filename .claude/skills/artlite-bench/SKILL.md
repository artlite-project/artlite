---
name: artlite-bench
description: "Measure a model's numbers for the README's performance charts (docs/data/artlite.json) the same way every time: TTFT at 1k / 8k / 32k / 64k and prefill speed with artlite-bench, decode speed at 1k / 32k with artlite-cli on essay prompts, for a model with a drafter (MTP) the best tokens per dispatch S at each length, and for a model with a vision encoder an image prompt's TTFT through artlite-server (and other runtimes' servers). Use when asked to benchmark a model, add or refresh a model's chart numbers, or compare prefill / decode / MTP / image-prompt speeds across days, models or runtimes."
---

# Benchmarking a model for the README charts

A model's entry in `docs/data/artlite.json` and where each number comes from (keys: prompt lengths in tokens):

| field | tool | value |
|---|---|---|
| `ttft_s`: 1024 8192 32768 65536 (and 131072 where a compared runtime has it) | `artlite-bench -p 1024,8192,32768,65536 -n 0` | `avg_ns` / 1e9 of `pp<n>` (the prompt's chunks and the last token's logits) |
| `prefill_tps`: 1024 32768 | the same run | `avg_ts` of `pp1024`, `pp32768` |
| `decode_tps`: 1024 32768 | `artlite-cli` on an essay prompt of 1023 / 32767 tokens, `-n 256 --greedy` | its `generation: ... tokens/s` |
| `mtp`: `decode_tps`, `tokens_per_step` | the decode runs again per draft count | the best S at each length, and its speed |
| `image_ttft_s`: encoder, then image | `image_ttft.py` against `artlite-server --mmproj FILE` | the median per image (section 5) |

Decode is not taken from `artlite-bench`'s `tg` tests: their tokens are random, so a drafter's guesses are never
accepted and MTP cannot show; and `tg` times `top_k` steps, not the app's `generate`. Decode runs greedy: the text,
and with it the drafts' acceptance, is then the same on every run (sampling changes the text per seed and per S).

## 0. A quiet machine

One NPU job at a time; no other heavy CPU work (stop a VS Code cpptools indexer, a build). Each step below loads the
model in its own process; the first load after a code change compiles the designs (cached in `~/.cache/artlite`):
run a throwaway `artlite-cli -n 8` first when the cache may be cold, or `artlite-bench` warms up by itself.

Embedding rows come from the mapped GGUF file (`GGUFFile.rows`), and a table that is not part of the NPU image (Llama
8B's untied `token_embd`, Qwen's, Gemma's per-layer table) is not read at load: after a reboot a test's random prompt
tokens read their rows from disk (Llama 8B 1k TTFT +0.1 s). Read the file once first (`cat MODEL.gguf > /dev/null`).

## 1. TTFT and prefill

```bash
artlite-bench -m MODEL.gguf -p 1024,8192,32768,65536 -n 0 -r 3 -o jsonl --progress > pp.jsonl
```

One load with a KV cache sized for the largest test. A length past the model's context (or its prefill's limit) is
left out of the entry; drop it from `-p` or the run stops at it. Repetitions agree within ~1%, so `-r 3` is enough.

## 2. Prompts

```bash
python .claude/skills/artlite-bench/prompts.py MODEL.gguf OUT_DIR          # prompt1024.txt, prompt32768.txt
```

Paul Graham essays (`tests/datasets/paul_graham_essays`, file-name order) then a request to summarize them, cut so
that the chat prompt is N - 1 tokens with the model's template. The essays are the same for every model; the cut
depends on the tokenizer.

## 3. Decode

```bash
artlite-cli -m MODEL.gguf -f OUT_DIR/prompt1024.txt  -n 256 --greedy > /dev/null   # (the speeds go to stderr)
artlite-cli -m MODEL.gguf -f OUT_DIR/prompt32768.txt -n 256 --greedy > /dev/null
```

The stderr line `prompt: ... generation: N tokens in T s (R tokens/s): D dispatches of X ms ...` gives R. If the
model stops early (N well below 256, an end-of-sequence token), say so next to the number. The prompt's speed on that
line cross-checks step 1 (TTFT there excludes tokenization, as in step 1), less the ~1 s a process's first prompt
spends loading the prefill's library ELF (`Library.load`), which artlite-bench's warmup absorbs.

## 4. MTP: the best S

For a model with a drafter (`NpuModel.DRAFTS`: Gemma 4 with its `mtp-<file>` next to the model), S = drafts + 1 tokens
per dispatch. `artlite-cli` uses the maximum; `drafts.py` sets the count (0: steps of one token, the entry's
`decode_tps`):

```bash
for D in 0 1 2 3; do
  python .claude/skills/artlite-bench/drafts.py $D -m MODEL.gguf -f OUT_DIR/prompt1024.txt -n 256 --greedy 2>&1 >/dev/null | grep prompt:
done   # and the same for prompt32768.txt
```

Sweep every D at both lengths: the best S depends on the context. An S-token dispatch costs about the same as one
token at short contexts (the weights stream once), but its attention grows with S at long ones, while the drafts'
acceptance falls with their depth. The `D draft tokens (X ms each), A accepted` part of the line explains a result:
acceptance per draft, and the drafter's own cost (it also grows with the context).

## 5. Image prompts (a model with a vision encoder)

```bash
artlite-server -m MODEL.gguf --mmproj MMPROJ.gguf --port 8080 --no-ui &      # (one per encoder file)
python .claude/skills/artlite-bench/image_ttft.py http://127.0.0.1:8080 IMAGE ...
```

The images are `docs/data/artlite.json`'s `image_prompts` (where each comes from: `docs/methodology.md`). The script
sends each image with fresh pixels, so no kept embeddings or prompt cache serve a request (it stops if a server reports
cached tokens). It prints the median of 3 per image after a warmup, which reproduces within ~2%. Stop the server with
SIGTERM, which waits for its NPU dispatch.

Other runtimes' servers take the same script: `llama-server -m MODEL.gguf --mmproj MMPROJ.gguf -c 32768 -np 1
-ngl 99` (Vulkan), and `flm serve <tag> --ctx-len <native> --prefill-chunk-len 4096` with the NPU in turbo. An iGPU's
sessions (a server start and its requests) differ by up to 8%: take the best of 3, after the machine has idled.

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

`model` names the chart group (the same string for every file of a model; groups keep the file's order). TTFT with 3
significant digits (0.44, 3.14, 16.7), prefill in whole tok/s, decode with one decimal. Without a drafter: no `mtp`.
Image prompts: `"image_ttft_s": {"16x": {"airplane.jpeg": 0.570, ...}, "4x": {...}}`, the encoders and images those of
`image_prompts` (a new image there needs its size and slice grid, `minicpmv.slice_image`'s `Grid.grid`).
A run to explain goes in `notes` (`"notes": {"decode_tps": {"1024": "stopped at an end-of-sequence token after 143 of
the 256 tokens"}}`: a footnote on that bar). Then `python docs/charts.py` redraws `docs/images` (light and dark SVGs;
commit both), and the model table in `docs/models/README.md` gets the file. `docs/methodology.md`'s "Measured files"
table gets its row: the Hugging Face repository and file it came from (match its sha256 against the repository's LFS
object ids: a local directory's name is not proof), the first 12 hex digits of the sha256, the date of the runs
(and that date's toolchain pins in the table below it); update "Test system" when the machine, driver or
firmware changed. Write today's numbers only from runs of the current commit.

Another runtime's numbers go in their own `docs/data/<runtime>.json` with the same fields; `comparable_quants` (the
GGUF quants of the same model its row is compared with, first match first) and `same_shapes_as` (charted under a
model of the same architecture and sizes) pair its rows with Artlite's.
