"""An image prompt's time to the first token through an OpenAI-compatible server (artlite-server, llama-server,
flm serve): per image one warmup request, then REPS measured ones; prints each image's median.

    python .claude/skills/artlite-bench/image_ttft.py URL IMAGE ... [--reps 3]

A request is one user message: the image (a JPEG data URI), then "<word> <number>. Describe this image in detail.";
streamed, thinking off, greedy, 16 tokens. TTFT is the client's time from sending it to the first streamed token. No
cache serves a request: its pixels are the image's with a random 4 x 4 block in the top-left corner (Artlite keeps an
image's embeddings by its pixels; FastFlowLM's prompt cache compares token ids only, and an image's are all the same),
its text starts with a random word, and cache_prompt is false.
"""

import argparse
import base64
import io
import json
import os
import random
import statistics
import time
import urllib.request

from PIL import Image

QUESTION = "Describe this image in detail."
WORDS = ["amber", "basil", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper"]


def fresh(path: str, rng: random.Random) -> bytes:
    """The image as a JPEG whose pixels no earlier request had."""
    im = Image.open(path).convert("RGB")
    color = tuple(rng.randrange(256) for _ in range(3))
    for y in range(4):
        for x in range(4):
            im.putpixel((x, y), color)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=95)
    return out.getvalue()


def ttft(url: str, jpeg: bytes, text: str) -> tuple[float, dict]:
    """(seconds to the first streamed token, the stream's usage)"""
    image = {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": image},
                                                      {"type": "text", "text": text}]}],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 16, "temperature": 0,
            "top_k": 1, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}, "think": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    first, usage = None, {}
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content")):
                    first = time.monotonic() - t0
            usage = ev.get("usage") or usage
    return first, usage


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("images", nargs="+")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    rng = random.Random()
    for path in a.images:
        times = []
        for i in range(a.reps + 1):
            t, usage = ttft(a.url, fresh(path, rng), f"{rng.choice(WORDS)} {rng.randrange(1000, 10000)}. {QUESTION}")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached:
                raise SystemExit(f"{path}: {cached} prompt tokens cached")
            if i:
                times.append(t)
        print(f"{os.path.basename(path)}: {statistics.median(times):.3f} s (runs {', '.join(f'{t:.3f}' for t in times)};"
              f" {usage.get('prompt_tokens')} prompt tokens)", flush=True)


if __name__ == "__main__":
    main()
