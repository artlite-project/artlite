"""artlite-server: an OpenAI-compatible HTTP server and web chat for one GGUF model on the NPU, as llama.cpp's
``llama-server`` (its options and the subset of its API that applies), for trying models more conveniently than
``artlite-cli``:

    artlite-server -m ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf    # then http://127.0.0.1:8080

Modules: ``main`` (the command line, uvicorn, the shutdown), ``api`` (the FastAPI routes, server-sent events, API
keys), ``engine`` (the NPU worker, the prompt cache, stop strings, the reasoning split, timings), ``ui`` (llama.cpp's
prebuilt web UI, pinned by build and sha256; MIT license, copyright The ggml authors).

Threads: uvicorn's event loop serves HTTP; one worker thread (``engine.Engine``) owns the model and makes every NPU
call. It opens and loads the model after the server starts - until then the API answers 503 ``Loading model``, which
the UI retries - and runs the requests one at a time in arrival order (the NPU runs one sequence; ``-np`` is 1): a
waiting request gets SSE pings. A request's events cross to the event loop through ``loop.call_soon_threadsafe``;
each dispatch releases the GIL, so the loop keeps serving while a step or a prompt chunk runs. Tokenizing a prompt
runs in a thread of the event loop's pool.

The prompt cache: the KV caches keep the tokens of the last request (its prompt and the generated tokens that ran).
The next request keeps the prefix its prompt shares with them - rounded down to the prefill's cache group, where a
chunk can start (``NpuModel.rewind``) - and runs the rest: a chat's next turn runs only its new messages. Qwen3.5 /
3.6 keeps only an exact continuation (its Gated DeltaNet states cannot go back): with thinking, whose old thoughts
the template drops, each turn runs its whole prompt again. ``cache_prompt: false`` starts from an empty cache.

Cancelling: a client that goes away, the UI's Stop (``DELETE /v1/stream?conv_id=``, matched by the chat request's
``X-Conversation-Id``) and the shutdown stop the request at its next token or prompt chunk; a dispatch is never
interrupted (one stays under the driver's 2 s).

The API (llama-server's shapes): ``/v1/chat/completions`` (streamed or not, ``reasoning_content``, ``timings``,
``prompt_progress``, ``usage``), ``/v1/completions``, ``/completion``, ``/v1/models``, ``/props``, ``/health``,
``/slots``, ``/tokenize``, ``/detokenize``, ``/apply-template``. Sampling is the model's (temperature, top-k, top-p,
min-p); the sampler fields the NPU path does not have (penalties, DRY, XTC, typical-p, mirostat, ...) are accepted
and ignored, logged when not neutral. Not provided: tool calls, images, embeddings, reranking, infill, the Responses
and Anthropic APIs, ``/metrics``, LoRA, router mode, resumable streams.
"""
