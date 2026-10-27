"""artlite-server: an OpenAI-compatible HTTP server and web chat for GGUF models on the NPU, as llama.cpp's
``llama-server`` (its options and the subset of its API that applies), for trying models more conveniently than
``artlite-cli``: one model, or the models of a directory (router mode: ``router``):

    artlite-server -m ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf    # then http://127.0.0.1:8080
    artlite-server                                                            # the models directory's models

Modules: ``main`` (the command line, uvicorn, the shutdown), ``api`` (the FastAPI routes, server-sent events, API
keys), ``agents`` (Anthropic's Messages and OpenAI's Responses APIs: their requests as chat completions, their
responses and streams, the stored responses), ``engine`` (the NPU worker, stop strings, the reasoning split,
timings), ``cache`` (the prompt cache), ``tools`` (tool calls: a template's syntax, the parsers of generated calls,
the mask of their names), ``ui`` (llama.cpp's prebuilt web UI, pinned by build and sha256; MIT license, copyright The ggml
authors); router mode: ``router`` (a model server process per model, loads and unloads, the proxy), ``child`` (a
model server under the router), ``catalog`` (the directory's models, their ids, presets).

What follows is one model's server (router mode runs one per loaded model).

Threads: uvicorn's event loop serves HTTP; one worker thread (``engine.Engine``) owns the model and makes every NPU
call. It opens and loads the model after the server starts - until then the API answers 503 ``Loading model``, which
the UI retries - and runs the requests one at a time in arrival order (the NPU runs one sequence; ``-np`` is 1): a
waiting request gets SSE pings. A request's events cross to the event loop through ``loop.call_soon_threadsafe``;
each dispatch releases the GIL, so the loop keeps serving while a step or a prompt chunk runs. Tokenizing a prompt
runs in a thread of the event loop's pool.

The prompt cache (``cache``): the KV caches keep the tokens of the last request (its prompt and the generated tokens
that ran), host RAM the conversations before it (``--cache-ram``: their KV records and states) and, for the Qwen
hybrids, checkpoints of the Gated DeltaNet states (``--ctx-checkpoints``), which cannot go back. A request continues
from what keeps the most of its prompt - the device's conversation (to the prefill's cache group where a chunk can
start, ``NpuModel.rewind``; Qwen: its end or a checkpoint), a conversation copied back from RAM - and runs the rest: a
chat's next turn runs only its new messages, an agent's subagents and side requests do not cost its main conversation
its cache. The earlier answers keep their thoughts (``--reasoning-preserve``: the templates' ``preserve_thinking``,
Qwen3.6's and Gemma 4's), so a turn extends the last one. ``cache_prompt: false`` starts from an empty cache.

Cancelling: a client that goes away, the UI's Stop (``DELETE /v1/stream?conv_id=``, matched by the chat request's
``X-Conversation-Id``) and the shutdown stop the request at its next token or prompt chunk; a dispatch is never
interrupted (one stays under the driver's 2 s).

The API (llama-server's shapes): ``/v1/chat/completions`` (streamed or not, ``reasoning_content``, tool calls,
``timings``, ``prompt_progress``, ``usage``), ``/v1/completions``, ``/completion``, ``/v1/models``, ``/props``,
``/health``, ``/slots``, ``/tokenize``, ``/detokenize``, ``/apply-template``. Sampling is the model's (penalties,
temperature, top-k, top-p, min-p); the sampler fields the NPU path does not have (DRY, XTC, typical-p, mirostat, ...)
are accepted and ignored, logged when not neutral. Tool calls: ``tools`` (the template's syntax and the parsers of
the generated calls); the engine parses them after the thoughts, and their names are masked to the functions' as the
tokens are sampled. Agent clients: ``/v1/messages`` and ``/v1/messages/count_tokens`` (Anthropic's: Claude Code),
``/v1/responses`` and ``/v1/responses/input_tokens`` (OpenAI's: Codex), on the same jobs. Not provided: embeddings,
reranking, infill, ``/metrics``, LoRA, resumable streams.
"""
