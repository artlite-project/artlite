# artlite-bench

Measures prompt processing and generation speeds of GGUF models on the NPU as llama.cpp's `llama-bench` does, with its
test names, result fields and output formats.

```console
$ artlite-bench -m bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_0.gguf -p 512 -n 128 -r 3
artlite-bench: Llama-3.2-3B-Instruct-Q4_0.gguf (Llama 3): loaded in 38.4 s: 1.93 GB of weights, 271 MB of KV cache (4096 positions, K single)
| model                          |       size |     params | backend    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --------------: | -------------------: |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | NPU        |           pp512 |       3490.08 ± 4.72 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | NPU        |           tg128 |         28.53 ± 0.04 |
```

## Tests

Per model and depth (`-d`), in llama-bench's order:

| test | |
|---|---|
| `pp<n>` | a prompt of n tokens (`-p`), as `artlite-cli` runs it: prefill chunks, then the last token's logits |
| `tg<n>` | n generated tokens (`-n`), each one decode step |
| `pp<n>+tg<m>` | both, timed together (`-pg`) |
| `... @ d<n>` | after a prompt of n tokens, which is not timed and runs again before every repetition |

Each test runs once untimed (the warmup: a prompt from position 0, one generated token), then `-r` times from an empty
KV cache; t/s = (prompt + generated tokens) / time, averaged over the repetitions. The tokens are random, starting
with the model's BOS if it has one.

Unlike llama-bench, which creates a context per test, each model is loaded once, with one KV cache sized for its
largest test (`n_ctx` in the results). `backends` is `NPU` and `gpu_info` is the NPU's name.

## Options

| option | |
|---|---|
| `-m FILES`, `--model` | GGUF files, comma-separated or repeated; a relative path is looked up in the models directory |
| `-p N`, `--n-prompt` | prompt tokens (default 512) |
| `-n N`, `--n-gen` | generated tokens (default 128) |
| `-pg PP,TG` | a prompt, then generated tokens, timed together (repeatable) |
| `-d N`, `--n-depth` | tokens in the KV cache before each test (default 0) |
| `-r N`, `--repetitions` | times to repeat each test (default 5) |
| `--delay S` | seconds of pause before each test (default 0) |
| `-o FMT`, `--output` | output format on stdout: `md` (default), `csv`, `json`, `jsonl` |
| `-oe FMT`, `--output-err` | output format on stderr: `none` (default), `md`, `csv`, `json`, `jsonl` |
| `--progress` | print test progress indicators |
| `--no-warmup` | skip the warmup run before each test |

`-p`, `-n` and `-d` take llama-bench's value lists: comma-separated values or ranges `first-last`, `first-last+step`,
`first-last*mult` (`-p 512-8192*2`). `-n 0` or `-p 0` leaves those tests out.

The README's charts are measured with it: [Benchmark methodology](methodology.md).
