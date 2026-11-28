# Inkling 2 on the NPU

Thinking Machines' Inkling 2 Mini (GGUF `inkling2_loop`, or unsloth's looped `inkling2`): 26B parameters in 24 layers,
a mixture of 128 experts of which 8 run per token, the middle 12 layers run 3 times (48 layer visits a token), each layer
mixing the earlier blocks' states (depth mixes) and, past the loop's first pass, its passes' outputs through learned
controllers. Attention: 40 of the 48 visits a sliding window of 1024 keys, the others full, K and V through short
convolutions, a relative-position bias, learned sinks. The architecture, the tokenizer and the chat template come from
the file's metadata. Running it: [artlite-cli](../cli.md), [artlite-server](../server.md); default sampling: the
file's (temperature 1, all tokens).

## Model files

```
$ARTLITE_MODELS_DIR/unsloth/Inkling-2-Mini-GGUF/        unsloth/Inkling-2-Mini-GGUF
    Inkling-2-Mini-UD-Q4_K_XL.gguf                      (NVFP4 and Q8_0 experts, Q8_0 attention, BF16 router, controllers)
    Inkling-2-Mini-UD-Q4_K_M.gguf                       (the same, the last 4 layers' expert gate / up Q5_K: 1 GB less)
    dspark-Inkling-2-Mini-Q8_0.gguf                     (its DSpark drafter: optional)
    mmproj-F16.gguf                                     (its vision encoder: optional, for images)
```

Either layout of a file: mainline llama.cpp's `inkling2_loop` (the repository's root, and its `Mainline_Support/`
directory) or unsloth's looped `inkling2` (its `Old_Support/` directory; a `.gguf_file` there renamed to `.gguf`).

The drafter is the first `dspark-*.gguf` beside the model file (or in the repository's root, or a `dspark/` directory
there; Q8_0 first) whose vocabulary and hidden size are the model's; `--spec-type none` runs without it,
`-md FILE` names another (its drafts come from the screened LM head: with `ARTLITE_LM_SCREEN=0` it is left out). Its
file in either form: mainline llama.cpp's (the root's `dspark-Inkling-2-Mini-Q8_0.gguf`) or unsloth's
(`dspark-Inkling-2-Mini-Q8_0_Old.gguf`, `dspark/dspark-Inkling-2-Mini-BF16.gguf`), the same weights.
The models directory: [Installation](../install.md#models-directory).

## Thinking and tools

The template puts a system block "Thinking effort level: x" before the first message (`reasoning_effort`: `none` 0,
`minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9, `xhigh` and `max` 0.99, or a number through
`chat_template_kwargs`): with [artlite-cli](../cli.md)'s `--think` the template's default (0.9) or
`--reasoning-effort`'s level, without it `none`; through [artlite-server](../server.md) a request's `reasoning_effort`
(Responses' `reasoning.effort`, Anthropic's `output_config.effort`), the thinking switch off `none`. The level is the
model's own control of its thought: no token budget comes with it (a request's `reasoning_budget_tokens` sets one), and
the model writes a short thought even at `none`.

The template opens no turn for the answer: the model writes its turn as blocks, each `<|message_model|>`, a kind, its
text and `<|end_message|>`: a thought (`<|content_thinking|>`), the answer's text (`<|content_text|>`), a tool call
(`NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}`), then `<|content_model_end_sampling|>` ends it. These
markers are text in the generation (llama.cpp's preserved tokens: artlite-cli prints them). artlite-server reads the
blocks as llama-server does: the thoughts come back as `reasoning_content`, the text blocks as `content`, the calls as
`tool_calls` (streamed: a call's name first, then its arguments as the model writes them); with `reasoning_format`
`none` the thought stays in the content (`<|content_thinking|>..<|end_message|>`). Tools are declared to the model as
the template writes them (one JSON line in a `tool_declare` block). A grammar holds the whole turn: with tools, the
calls are of the request's functions and their arguments follow the schemas (their keys sorted, as the template writes
them), after the model's thought (and text) blocks; `tool_choice` `required` lets thoughts alone come first, `none`
allows no call block; a response format (a JSON schema, a grammar) is the answer's text block, after the thoughts. A
request whose answer is one of a few options with `logprobs` generates (the model thinks first), instead of scoring
the options. The prompt cache takes a checkpoint where the first user message begins (the system blocks, the tools and
the effort level before it, which an agent's sessions share).

## Images

With its vision encoder (`--mmproj unsloth/Inkling-2-Mini-GGUF/mmproj-F16.gguf`; artlite-server's router pairs the
mmproj of the model's directory), a prompt takes images: [artlite-cli](../cli.md)'s `--image`, `image_url` parts through
[artlite-server](../server.md). An image goes in at its original size, padded at the bottom and the right to multiples
of 40 pixels: a token per 40 x 40 pixels (2 x 2 patches of 20), 520 tokens for a 1024 x 768 image; one of more than 4096
tokens (`--image-max-tokens`) is resized under that first, as llama.cpp does. The encoder (23 layers 1024 wide over the
image's patches and a class token) runs on the NPU in a separate hardware context (the vision overlay:
`src/models/inkling2v/`), then the image's tokens go through the prefill as text does; a conversation's image runs the
encoder once.

Speed (Strix Halo; artlite-server, one image and a question, its pixels fresh for each request: the artlite-bench
skill's `image_ttft.py`, the median of 3), against llama.cpp's llama-server on the CPU (16 threads) and on the iGPU
(Vulkan) with the same files:

| image (pixels) | tokens | the encoder | time to the first token | llama.cpp CPU | llama.cpp Vulkan |
|---|---|---|---|---|---|
| sd15-astronaut.jpg (512 x 512) | 169 | 0.05 s | 1.50 s | 1.12 s | 1.13 s |
| test-1.jpeg (640 x 488) | 208 | 0.05 s | 1.58 s | 1.29 s | 1.29 s |
| hk_OCR.jpg (1300 x 858) | 726 | 0.21 s | 2.60 s | 3.01 s | 3.07 s |
| airplane.jpeg (1920 x 1440) | 1728 | 0.75 s | 5.18 s | 7.47 s | 7.26 s |

The prompt (the image's tokens and ~25 of text) takes most of the time, through the prefill, whose fixed costs dominate
a short one (196 tokens: 1.25 s): llama.cpp is ahead on the small images, Artlite 1.2x and 1.4x ahead on the larger ones.

Accuracy: per token the encoder's embeddings are ~4-5% from a float64 encoder's (median; llama.cpp's own are 1.3% off:
its pixels rounded to fp16, which the network's 23 layers amplify). Teacher-forced over a description of an image after
an essay paragraph, the model with the NPU's embeddings is at KL 0.0025 from llama.cpp's run of the same file and
mmproj (on llama.cpp's own image rows: 0.0012), the argmax the same on 80 of 81 tokens.

## How the NPU path works

Paths below are under `src/models/inkling2/` and `src/artlite/` (what the models share). One design, compiled once:
the decode overlay of 4 token slots (`artlite.decode`) with the attention cores in Inkling's mode (`decode/attn.py`)
and the MoE layers of `artlite.moe` with an expert's units of 512 and 128 channels.

- A dispatch runs up to 4 token slots through the 48 visits and the LM head, streaming the weights once for all of them
  (`overlay.py`, `InklingOverlay`): the projections on each column's main and helper cores, the residual stream's
  states (the blocks' outputs, the loop's banks) in the helpers' memory, the depth mixes and the controllers' blends
  there; the attention, its caches and the K / V convolutions on the attention cores; the experts' activation (situ)
  on the helpers; the LM head screened (the host computes its largest logits exactly).
- Prompts run in prefill chunks of 64 .. 8192 tokens (`prefill.py`, `InklingPrefill`, on the same weight image and
  caches): per visit the depth mixes, the loop's controllers, the attention (its prep, the relative bias and sinks on the
  attention cores; past 2048 tokens in items of 2048), the post convs, and the MoE (`moe.py`: the router on the NPU on x
  hi + lo, the routing on the host, the experts' work lists on `artlite.moe`); the prompt's last token runs as a decode
  step (the LM head's). The experts stream their weights once per chunk: long prompts run in chunks of 8192.

- The DSpark drafter (`dspark.py`, `dspark_image.py`) runs in a separate design and hardware context, with 4 token slots
  of the same decode overlay, the attention cores with q / k norms, RoPE and a non-causal block. Each step and prompt
  chunk leaves the model's taps of its positions (the residual entering visits 5, 23, 41 and the final normed state),
  which the drafter's injections turn into the K / V of its 3 layers (layer 1 attends to every position, the other two
  to the last 1024). A round: one dispatch of the drafter, the injections of the positions since its last round and a
  block of 4 rows (the token, 3 mask rows) through its layers and the model's screened LM head, while the host patches
  the model's step of the round; its Markov head on the host over each row's candidates (their logits exact from the
  file's rows), then the model's step of the token and the drafts that pay at its step times. A step of S tokens costs
  ~1.35, 1.6, 1.8 times a step of one for S = 2, 3, 4 (each token routes to different experts), so drafting pays off
  only when the drafter guesses well: the rounds' planner leaves the drafter out when its rounds give fewer tokens per
  second than steps of one.

Memory: the weights the NPU streams are re-laid-out into device buffers at load, in memory, at their quantization's
size: 17.15 GB, UD-Q4_K_M's 16.22 GB (a K of 2304 as two chunks of 1024 and a quarter of one, nothing padded but
UD-Q4_K_M's Q5_K gate / up, 14% over the file's bytes: a quarter chunk's 4 tiles of 13 rows hold 3 windows of 8 gate
and 8 up rows); nothing is written to disk. A decode step streams 1.09 GB of dense weights and the 8 experts of each
visit (1.86 GB of expert tiles a token, UD-Q4_K_M's 1.80 GB). The
sliding-window visits' caches are rings past 16K positions of capacity (16384 positions, two of the prefill's largest
chunks): they, the full visits' caches and the convs' state take 3.4 GB at 33K positions (each sequence of `-np` its
own), 6.2 GB at 132K. Without the prefill the rings hold 2048 positions (1.38 GB at 33K). The convs' state holds the last
positions only, so a conversation goes back only to a checkpoint (artlite-server takes them). The drafter: 0.30 GB of
weights, 36 MB of caches at 4K positions (~140 MB at 33K). With `-np` above 1 the drafter is not loaded.

## Speed and accuracy (Strix Halo)

| | 1K | 32K |
|---|---|---|
| a prompt (the prefill, artlite-bench) | 472 tokens/s (2.17 s for 1024 tokens) | 578 tokens/s (56.7 s; 8K 12.8 s, 64K 132 s) |
| generation (greedy, 256 tokens, essay summary) | 11.5 tokens/s | 9.7 tokens/s |

UD-Q4_K_M: generation 11.7 tokens/s at 1K (a step 0.4 ms shorter than UD-Q4_K_XL's in the same session: its experts'
1.80 GB against 1.86), prompts as fast as UD-Q4_K_XL's (its last 4 layers' expert gate / up 17% faster in the prefill,
~0.5% of a chunk); its accuracy against llama.cpp's runs of its file is UD-Q4_K_XL's (the table below: KL 0.0025, 0.0016,
0.0025; greedy text 629 of 640 tokens llama.cpp's top one, the others near ties).

A step of a token streams 1.09 GB of dense weights and its 8 experts per visit (1.86 GB of tiles); each further token of
a step (drafts, or other requests with `-np`) adds the experts it routes to. Drafts (the DSpark drafter's, the
context's) keep the text the same as the model writes alone.

With the DSpark drafter (greedy, 128 tokens after ten chat prompts through the prefill, against steps of one): code
12.1 -> 14.8 tokens/s, an essay's continuation 1.19x, JSON 1.07x, math 1.02x, the other prompts (chat, SQL, a story,
explanations) 0.98 - 1.01x, where the planner leaves the drafter out after its first rounds; a 32K essay summary 9.8
-> 9.7 tokens/s. The drafter's work a round ~18 - 25 ms (its block, the head's plane, its Markov head on the host); a
prompt's drafter rows ~1 ms per 4 tokens (a 1K prompt 2.1 -> 2.4 s, 8K 12.8 -> 15.1 s, 32K +7.8 s). Its drafts accepted, teacher-forced on the model's own greedy text (of the 3 a round verifies):
code 1.8, math 2.0, prose 2.2, chat 1.0.

Accuracy against llama.cpp's CPU runs of the same file:

| | |
|---|---|
| teacher-forced decode, an essay's first 128 tokens | KL 0.0025 (llama.cpp's own prompt graph against its decode: 0.0022), argmax the same on every clear token |
| a prompt through the prefill (1024, 2560 essay tokens), then 64 teacher-forced steps | KL 0.0017, 0.0025 (llama.cpp's own prompt graph against its decode: 0.0015), argmax the same on every clear token |
| greedy text, 128 tokens after a chat, code, math, a long instruction, an essay (prompts through the prefill) | 630 of 640 tokens llama.cpp's top one, the 10 others within 1 logit of it |

Design: the module docstrings of `src/models/inkling2/`, `src/artlite/decode/attn.py` and `src/artlite/moe/`.
