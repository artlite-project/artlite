# artlite-server

Serves GGUF models as llama.cpp's `llama-server` does: an OpenAI-compatible API with llama-server's own routes and
fields, so that clients written for either work unchanged, and a web chat at `/`. With `-m` it serves that model;
without, every model of the models directory, loaded when the web chat's model picker or a request asks for it
([router mode](#several-models-router-mode)).

```bash
artlite-server -m ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf -rea on    # then open http://127.0.0.1:8080
artlite-server                                                                      # the models directory's models
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/server-dark.png">
  <img alt="The web chat on Gemma 4 E4B: a question, its collapsed reasoning, the answer and its speed" src="images/server.png" width="640">
</picture>

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Factorize x^2 + x - 6"}]}'
```

## API

| route | |
|---|---|
| `/v1/chat/completions` | chat, streamed or not, with `reasoning_content`, tool calls, `timings`, `prompt_progress`, `usage` |
| `/v1/messages`, `/v1/messages/count_tokens` | Anthropic's Messages API ([agents](#agents)) |
| `/v1/responses`, `/v1/responses/input_tokens` | OpenAI's Responses API ([agents](#agents)) |
| `/v1/completions`, `/completion` | raw prompts (OpenAI's and llama-server's shapes) |
| `/v1/models`, `/props`, `/health`, `/slots` | the model, the server's settings, its state |
| `/tokenize`, `/detokenize`, `/apply-template` | the model's tokenizer and chat template |
| `/v1/embeddings`, `/embeddings`, `/embedding` | embeddings (with `--embedding`: [below](#embeddings-and-rerank)) |
| `/v1/rerank`, `/rerank`, `/v1/reranking`, `/reranking` | a reranker's scores of documents (with `--rerank`) |
| `/classify` | a classifier's label probabilities (vLLM's shape; with `--rerank`) |
| `/v1/systemone` | a decision model's answers to questions about a state ([below](#decisions)) |

Sampling has the repetition, presence and frequency penalties, the temperature, top-k, top-p and min-p. The other
sampler fields (DRY, XTC, typical-p, mirostat, ...) are accepted and ignored, and logged when not neutral: the web UI's
settings for them have no effect. Log-probabilities come as llama-server's: chat's `logprobs: true` with
`top_logprobs` (default 20) and `/v1/completions`' `logprobs: N` give each generated token's (`logprobs.content`:
`id`, `token`, `bytes`, `logprob` and the `top_logprobs`), `/completion`'s `n_probs` gives them as
`completion_probabilities`, and `post_sampling_probs` the probabilities the sampler drew from instead (`prob`,
`top_probs`).
Images (a model with `--mmproj`: [MiniCPM-V 4.7](models/qwen35.md#images-minicpm-v-47)) come as a chat message's
`image_url` parts of data URIs, as llama-server takes them; the web UI attaches them. Not provided: infill,
`/metrics`, LoRA, model downloads; their routes answer 501.

Tool calls (OpenAI's function calling): a request's `tools` go to the model's chat template, and the calls the model
writes come back as `message.tool_calls` (streamed: `delta.tool_calls`, a call's id and name first, then its
arguments) with `finish_reason: tool_calls`; the results go back as `role: tool` messages. Qwen3.5 / 3.6 / 3.8,
MiniCPM-V 4.7 and Occamy (Qwen's XML calls), Gemma 4 and Llama 3.1 / 3.2 have them; a template without tools (Llama
3.2 1B Q3_K_L's file) ignores them. `tool_choice` takes `auto`, `none` (the tools stay in the prompt, nothing is
parsed), `required` and a named function (the answer begins with the call, after the model's thought; a named
function's call is the only one), and `parallel_tool_calls: false` ends the answer after its first call. A call's
name is always one of the request's functions: while the model writes it, only the tokens that spell one can come
(with `required` where no function fits, the model calls the closest one instead of inventing a name). The arguments
are not constrained by a grammar: a call the model writes malformed stays text, and a string argument whose schema
allows no string becomes the value it encodes (Llama 3.1 quotes numbers and lists).

## Agents

Coding agents and other clients that call tools use the API they speak: chat completions (opencode, Open WebUI, the
web chat), Anthropic's Messages API at `/v1/messages` (Claude Code, the Anthropic SDKs) or OpenAI's Responses API at
`/v1/responses` (Codex CLI, the OpenAI Agents SDK). All three run as the same chat request: tool calls, thoughts and
the prompt cache work alike.

```bash
artlite-server -m lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf -rea on -c 131072
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=none claude          # Claude Code
codex -m qwen3.6-35b-a3b -c model_provider=artlite \
    -c 'model_providers.artlite={name="artlite", base_url="http://127.0.0.1:8080/v1", wire_api="responses"}'
```

Codex needs a model name it does not know (`-m`; a server started with `-m` answers to any name): for its own models
it sends a JavaScript tool of its own instead of functions, which the server refuses. opencode takes the server as a
provider in
`~/.config/opencode/opencode.json`, without a timeout (its default ends a request after 5 minutes, and a turn that
thinks can take longer):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "artlite": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Artlite",
      "options": {"baseURL": "http://127.0.0.1:8080/v1", "timeout": false},
      "models": {
        "qwen3.6-35b-a3b": {"name": "Qwen3.6 35B-A3B", "tool_call": true, "reasoning": true,
                            "limit": {"context": 131072, "output": 32768}}
      }
    }
  },
  "model": "artlite/qwen3.6-35b-a3b",
  "small_model": "artlite/qwen3.6-35b-a3b"
}
```

- Messages API: `system`, the blocks `text`, `image` (base64), `thinking`, `tool_use` and `tool_result`, `tools`,
  `tool_choice`, `stop_sequences`, and Anthropic's event stream. `thinking` turns the model's thought on (`enabled`,
  `adaptive`) or off (`disabled`; without it, `-rea` decides); its budget is not enforced. A prompt longer than the
  context answers "prompt is too long: N tokens > M maximum", which Claude Code acts on.
- Responses API: `instructions`, the items `message`, `reasoning`, `function_call` and `function_call_output`,
  function tools (others, such as `web_search`, are left out), `tool_choice`, `reasoning.effort` (`none`: no
  thought), `max_output_tokens`, and OpenAI's event stream. `previous_response_id` continues one of the last 1024
  responses the server keeps (not those with `store: false`); `GET` and `DELETE /v1/responses/{id}` read and drop
  one. A prompt longer than the context fails with the code `context_length_exceeded`.
- On every route a `developer` message is a system message, and where a template takes a system message only first
  (Qwen's), the later ones become user messages: Claude Code and Codex send them during a conversation.
- Agents copy their context (an edit's old text, a file they rewrite, a command again): generation takes draft tokens
  from it ([artlite-cli](cli.md#text)), so on recorded agent sessions Qwen3.6-27B, Qwen3.5-9B and Llama 3.1 8B generate
  1.46-1.52x faster and MiniCPM-V 1.25x (Qwen3.6-35B-A3B and Gemma 4 guess with their drafters: +0-4% more).
- Models: Qwen3.6-35B-A3B is the one these three clients ran whole tasks on (each fixed a bug in a Python package, its
  tests run); with its drafter it generates ~30 tokens/s after a 1K-token prompt and ~21 after 32K. Occamy 1.0, a
  fine-tune of it for agentic tool use, takes 15.8 GB instead of 21. On 150 function-calling cases (BFCL's simple,
  multiple and parallel calls) Qwen3.6-35B-A3B, Gemma 4 E4B and Llama 3.1 8B call right in 131, 134 and 129, none
  malformed. Most of a session's time is generation, most of it the model's thoughts: a client's own thinking
  setting, or `-rea off`, turns them off.
- Context and memory: the clients' first prompts are 7-12K tokens (their instructions and tools), and a task's
  conversation often grows past the default 32K positions: give `-c 131072`. The weights stay in host RAM (21 GB for
  Qwen3.6-35B-A3B), with the KV cache (its 131072 positions: 2.3 GB) and the prompt cache (`--cache-ram`,
  [below](#requests)).

## Web chat

llama.cpp's own web UI (MIT license, copyright The ggml authors), prebuilt, pinned by build and checksum, downloaded
on the first start into the build cache (later starts need no network). `--path DIR` serves another build, `--no-ui`
none.

## Embeddings and rerank

With `--embedding` the server returns embeddings as llama-server does: `/v1/embeddings` (OpenAI's: `input` a text, a
list of texts or token ids; `encoding_format` `float` or `base64`; OpenAI's `dimensions`: the first N values,
normalized again) and `/embeddings` (llama-server's own: `content`). An embedding is the last layer's output after the
model's final norm, pooled by `--pooling` (else the model file's `pooling_type`): `mean` over the tokens, `cls` the
first token's, `last` the last token's, `none` every token's (`/embeddings` only), normalized by `embd_normalize`
(Euclidean by default: `--embd-normalize`). An embedding model's file sets its pooling and whether a text ends with its
end token (Qwen3-Embedding: `last`, appended); a generative model (Llama 3.x, Qwen3, Qwen3.5 / 3.6 / 3.8, MiniCPM-V
4.7) has no pooling of its own: give one. Gemma 4 has no embeddings (501).

```bash
artlite-server -m Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d '{"input": ["a text", "another"]}'
```

A request's texts run entirely as prefill chunks (no decode steps, no LM head). Several texts in one request run
packed: several of them in one chunk (up to 4096 tokens; 2048 on Qwen3.5's path), each with its own positions,
attention and Gated DeltaNet state, so they share the chunk's fixed cost. The server plans a request by the prefill's
estimates and runs a text alone where packing would not be faster (on Qwen3's path, any text longer than 512 tokens).
A prefix the texts share (a rerank query's prompt before its documents) runs once, through the prompt cache, and every
text after it sees it. Qwen3-Embedding-0.6B: 10.3K tokens/s for requests of 32 texts of 64-512 tokens (7.1K one
after another); Qwen3-Reranker-0.6B: 22 pairs/s at ~510-token pairs (16). `--rerank` serves `/v1/rerank` (Jina's shape: `query`, `documents` as strings or
`{"text": ...}`, `top_n`, Cohere's `return_documents`; TEI's with `texts`) for a reranker's file (Qwen3-Reranker): its
classifier head (`cls.output`) on the last token of its rerank prompt (the file's `rerank` template). With it,
`/classify` returns vLLM's classification of each `input` (a text or token ids): the labels' probabilities (`probs`:
the softmax of the head's outputs, a sigmoid of a single one), the likeliest `label` (the file's
`classifier.output_labels`, else its index) and `num_classes`.

The embeddings and scores come within llama.cpp's own spread of each other on retrieval and ranking (Qwen3-Embedding
0.6B and 4B: the same top-10 passages for 30 queries but one at most; Qwen3-Reranker-0.6B: the target passages ranked
first as often), not to its digits: the prefill's weights and activations are 8-bit bfp16 blocks, so an embedding is
within a cosine of 0.994-0.999 of llama.cpp's CPU result (its CPU and Vulkan builds: 0.997-0.9995; an exact fp32 model:
0.999) and a rerank score within 0.18. Model pages: [Qwen3](models/qwen3.md).

## Decisions

A decision model (a GGUF file with `{arch}.decision.*` metadata and a `systemone` template) answers `/v1/systemone`
as llama-server does (TypeSafe's System One API): a `state` (text, a JSON value or chat messages) and `questions`, each
`{"type": "choice" | "score" | "noul", "instructions", "criteria"}` (a choice's options as an object, a score's 2 to
10 levels as a list, a noul's optional `true` / `false` descriptions). Each question is one forward pass over its
prompt, no token generated: a choice's `choice`, `probabilities` and `confidence`, a score's expected level `score`,
`legend`, `probabilities` and `confidence`, a noul's probability of true `noul`; `usage.input_tokens` counts every
question's prompt. `/v1/models` reports such a model's `output_modalities` as `["decisions"]`.

```bash
artlite-server -m ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-Q8_0.gguf
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" -d '{"state": "I was charged twice for
  my order and nobody replied.", "questions": {"route": {"type": "choice", "instructions": "Which team?",
  "criteria": {"billing": null, "shipping": null}}, "angry": {"type": "noul", "instructions": "Is the customer angry?"}}}'
```

The decision types artlite-server reads: vllm-d3, pplx-decider, openjev, lev, nimble (the logits of the options' label
tokens), kev (its head's products of the last token's and each option's rows) and vllm-decision2 (a joint head over the
rows at the ends of the options and of the question), computed on the host from the prompt's output rows. Of the
published files, vllm-sr's d3-lite and d3-nano (ggml-org's Q8_0), ggml-org's Kev-4B (Q4_K_M) and Decision-2.0-Kai-0.6B
(llama.cpp's converter run on vllm-sr's checkpoint: vllm-sr's own GGUF is a fork's format) run. A vllm-d3 model reads a prompt whole (its attention is non-causal): one prefill chunk of
at most 2048 tokens (a longer prompt: 400), and it generates no text; other types' questions share the state's tokens
through the prompt cache. A request's questions run packed as embeddings' texts do (a vllm-d3 model's whole prompts;
on a Qwen3.5 model, whose state cannot be shared inside a chunk, either the state once and each question after it or
the questions' whole prompts packed, whichever the estimates say is faster). Other types (laya, clef, lfm2-d1) and images (`files`, `images`, a message's image parts)
answer 501.

## Requests

With `-m`, the model loads after the server starts; until then the API answers 503 `Loading model`, which the UI
retries. A model's requests run one at a time, in arrival order; a waiting stream gets pings. A client that goes away, the UI's Stop
button and the shutdown stop a request at its next token or prompt chunk.

A chat's next turn continues from the KV cache: the server keeps the prefix a prompt shares with the cached tokens,
rounded down to where a prefill chunk can start, and runs the rest. Other conversations stay in host RAM, as in
llama-server (`--cache-ram`: 8 GiB by default, or half of the RAM left once the model is loaded where that is less):
when a request continues one of them (an agent's subagent, a title request, another chat in the web UI), its cache is
copied back to the NPU (~15 GB/s: 30-45 ms for 32K positions of the 35B-A3B) instead of running its prompt again. The
Qwen models' Gated DeltaNet states cannot go back, so for them the server saves the states (checkpoints,
`--ctx-checkpoints`; Qwen3.6-35B-A3B: 80 MB, a few milliseconds each) at the end of each prompt and where its first
user message begins: a prompt continues from the latest one it shares (the client's own rendering of the last answer,
another answer to the same prompt), and a new conversation with the same system prompt and tools from the second. The
earlier answers keep their thoughts in the prompt, as llama-server does (`--reasoning-preserve`, on by default, where
the template can: Qwen3.6 / 3.8, Occamy, Gemma 4), so a turn extends the last one when the client sends the thoughts
back (`reasoning_content`, as the web UI does). Where they are dropped (Qwen3.5's and MiniCPM-V's templates,
`--no-reasoning-preserve`), the next user message renders the last answer from its start otherwise: the server also
saves the state where each prompt's last message begins, and a turn continues from there. A request's
`cache_prompt: false` starts from an empty cache.

## Several models (router mode)

Without `-m`, artlite-server serves the models of a directory, as llama-server's router mode does: the web chat's
model picker lists them, a request's `model` names one, and a model loads when a request asks for it.

```bash
artlite-server                                   # the models of ARTLITE_MODELS_DIR (default ~/.artlite/models)
artlite-server --models-dir DIR --models-max 2   # another directory, two models loaded at a time
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/server-models-dark.png">
  <img alt="The web chat's model picker in router mode: the loaded model on top, the models directory's files by family, size and quantization" src="images/server-models.png" width="640">
</picture>

- **Models.** Each GGUF file of the directory that Artlite runs, named `<account>/<repository>:<quantization>`
  (`lmstudio-community/Llama-3.2-1B-Instruct-GGUF:Q4_0`); its path in the directory, as `-m` takes it, names it too.
  Files Artlite does not run (another architecture, tensor types without kernels) are skipped with a log line that
  says why. Drafters (`mtp-`, `<name>-mtp-`, z-lab's `-DFlash2-GGUF` repositories) and vision encoders are not models of their own: Gemma 4 and the
  qwen35moe models (Qwen3.6 35B-A3B, Occamy 1.0) find their drafters (Qwen3.8 27B: the DFlash2 repository's, else the MTP layer in its own file), and MiniCPM-V 4.7 gets the `mmproj` file of its
  directory (`mmproj-model-f16.gguf` before the 4x one).
- **Loading.** A request for a model that is not loaded loads it and waits. `--models-max` models (default 1) are
  loaded at a time: a load first unloads the least recently used model that has no request running, or waits for
  one, so a model is never unloaded under a request. Each model runs in a server process of its own, with the
  command line's model options (`-c`, `-rea`, `--temp`, `--kv-k`, ...); unloading ends that process, which gives all
  its memory back. One process is kept started ahead, so a load does not wait for Python to start (~1.5 s). Loaded
  models take turns on the NPU: one process's dispatches run while no other's do, and another process's request waits
  for the running dispatch, or for the next piece boundary of a long prompt (up to two pieces, ~1.6 s, during a
  32K prompt of a 3B).
- **API.** `GET /v1/models` lists every model with its status: `unloaded` (`failed` and the reason when its last load
  failed), `loading`, `loaded`; `?reload=1` reads the directory again, and a running model whose file changed
  unloads. `POST /models/load` and `/models/unload` take `{"model": ...}` (an unload cancels the model's running
  request); `GET /models/sse` streams the status changes. `/props` describes the router (`role: router`);
  `/props?model=`, `/slots?model=` and the requests go to the model they name, or without a name to the one model
  loaded; `?autoload=false` does not load it.

`--models-preset FILE` sets options per model and adds models from elsewhere, in llama-server's INI format:

```ini
[*]                                        ; every model
ctx-size = 8192

[openbmb/MiniCPM-V-4.7-1B-gguf:Q4_K_M]      ; one model, by its name
mmproj = openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16-4x.gguf
max-slices = 36
load-on-startup = true

[my-llama]                                 ; a model of its own, from anywhere
model = /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf
alias = llama
```

Keys are the options' long names without dashes (`ctx-size`), their short ones (`c`) or their environment variables
(`ARTLITE_ARG_CTX_SIZE`); switches take `true` or `false` (`no-prefill = true`, `warmup = false`). An empty `mmproj`
turns a vision encoder off, `alias` adds names (comma-separated), `load-on-startup` loads the model when the server
starts. A model's option comes from the command line, else its section, else `[*]`.

## Options

Each option also reads `ARTLITE_ARG_<NAME>`, the long option's name in upper case with dashes as underscores
(`ARTLITE_ARG_CTX_SIZE=65536`); the command line overrides it. `artlite-server --help` lists every variable.

| option | |
|---|---|
| `-m FILE`, `--model` | the GGUF file (as [artlite-cli](cli.md)'s `-m`); without: router mode |
| `-a NAME`, `--alias` | the model's id in the API (default: the file name) |

Router mode (without `-m`):

| option | |
|---|---|
| `--models-dir DIR` | serve the models of this directory (default: `ARTLITE_MODELS_DIR`, `~/.artlite/models`) |
| `--models-preset FILE` | models' options and more models ([above](#several-models-router-mode)) |
| `--models-max N` | models loaded at a time, 0: any number (default 1) |
| `--models-autoload`, `--no-models-autoload` | a request loads the model it names (default on; a request's `?autoload=false`: not) |

Server:

| option | |
|---|---|
| `--host`, `--port` | the address to listen on (default 127.0.0.1:8080) |
| `--api-key KEYS`, `--api-key-file FILE` | keys clients must send (`Authorization: Bearer` or `X-Api-Key`): comma-separated, or one per line in a file (`#` starts a comment) |
| `--cors-origins LIST` | origins allowed by CORS, comma-separated (default `*`) |
| `--path DIR`, `--no-ui` | serve the web UI from this directory; no web UI |
| `-to S`, `--timeout` | seconds an idle connection is kept open (default 3600) |
| `--sse-ping-interval S` | seconds between pings while a stream waits for the NPU (default 30) |
| `--slots`, `--no-slots` | the `/slots` endpoint (default on) |

Generation:

| option | |
|---|---|
| `-c N`, `--ctx-size` | KV cache positions, rounded up to a power of two >= 4096 (default 0: 32768, at most the model's context) |
| `-n N`, `--n-predict` | tokens to generate when a request does not say (default -1: until an end-of-sequence token or a full KV cache) |
| `-np N`, `--parallel` | sequences at a time (1 only: the NPU runs one) |
| `--cache-prompt`, `--no-cache-prompt` | continue from the prefix a prompt shares with the cached tokens (default on) |
| `-cram N`, `--cache-ram` | MiB of host RAM for the prompt cache: the conversations it saves and the checkpoints (default 8192, at most half of the RAM still available once the model is loaded; -1: no limit, 0: none saved) |
| `-ctxcp N`, `--ctx-checkpoints` | checkpoints per conversation at most (Qwen3.5 / 3.6 / 3.8, Occamy, MiniCPM-V; default 32) |
| `--lookup`, `--no-lookup` | draft tokens from the prompt in the generation's steps ([artlite-cli](cli.md#text)'s; default on) |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`, `--presence-penalty`, `--frequency-penalty` | sampling defaults (default: the model's), as [artlite-cli](cli.md)'s |
| `-s SEED`, `--seed` | sampling seed (default -1: random per request) |
| `-rea on\|off\|auto`, `--reasoning` | let a model with a thinking mode think first (auto: when it has one; default off); a request's `chat_template_kwargs.enable_thinking` and the UI's toggle override it |
| `--reasoning-format auto\|deepseek\|none` | thoughts in `reasoning_content` (auto, deepseek) or in `content` (none) (default auto) |
| `--reasoning-preserve`, `--no-reasoning-preserve` | earlier answers keep their thoughts in the prompt where the template can (the templates' `preserve_thinking`; default on); a request's `chat_template_kwargs.preserve_reasoning` (or `preserve_thinking`) overrides it |
| `--chat-template-kwargs JSON` | chat template variables, under each request's own (`enable_thinking`, `preserve_reasoning`) |

NPU:

| option | |
|---|---|
| `--no-prefill` | run prompts as decode steps |
| `--kv-k single\|hilo` | the KV cache's K precision (Llama, Qwen3, Gemma 4), as [artlite-cli](cli.md)'s |
| `--dflash`, `--no-dflash` | Qwen3.8 27B's drafter, as [artlite-cli](cli.md)'s (default: its DFlash2 block drafter where the file is in the models directory, else its MTP layer) |
| `--warmup`, `--no-warmup` | run one token after loading (default on) |

Embeddings and rerank ([above](#embeddings-and-rerank)):

| option | |
|---|---|
| `--embedding`, `--embeddings` | the embedding routes |
| `--pooling none\|mean\|cls\|last\|rank` | the embeddings' pooling (default: the model file's) |
| `--rerank`, `--reranking` | the rerank routes and `/classify`: `--embedding` with pooling `rank` (a file with a classifier head) |
| `--embd-normalize N` | the embeddings' normalization: -1 none, 0 the largest magnitude to 32760, 2 Euclidean, any other p the p-norm (default 2; a request's `embd_normalize` overrides it) |

Images:

| option | |
|---|---|
| `--mmproj FILE` | the vision encoder's file (MiniCPM-V 4.7: `mmproj-model-f16.gguf`): images in chat messages |
| `--max-slices N` | an image's slices at most (default: the mmproj's, 9; the model card uses 36) |

Logging:

| option | |
|---|---|
| `-v`, `--verbose` | debug messages and every HTTP request |
| `--log-file FILE` | also write the log to this file |
