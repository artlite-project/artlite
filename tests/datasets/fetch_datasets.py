"""Fetch the datasets the tests and benchmarks read (gitignored next to this script).

paul_graham_essays/: the 49 essays of the needle-in-a-haystack benchmark (also RULER's haystack), the long-prompt
corpus. Long prompts concatenate them sorted by file name and repeat the sequence when they need more tokens
(~149K Llama 3 tokens for the whole set). Pinned to one upstream commit: the tests' cached references depend on the
exact text.
"""

import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ESSAYS_REPO = "gkamradt/needle-in-a-haystack"
ESSAYS_COMMIT = "021385d68d3202e37893e9d3cd29011c569abe30"
ESSAYS_PATH = "needlehaystack/PaulGrahamEssays"


def fetch_paul_graham_essays(dest: str = os.path.join(HERE, "paul_graham_essays")) -> None:
    """Download the essays missing from dest."""
    os.makedirs(dest, exist_ok=True)
    url = f"https://api.github.com/repos/{ESSAYS_REPO}/contents/{ESSAYS_PATH}?ref={ESSAYS_COMMIT}"
    with urllib.request.urlopen(url) as r:
        names = sorted(e["name"] for e in json.load(r) if e["type"] == "file")
    fetched = 0
    for name in names:
        out = os.path.join(dest, name)
        if os.path.exists(out):
            continue
        raw = f"https://raw.githubusercontent.com/{ESSAYS_REPO}/{ESSAYS_COMMIT}/{ESSAYS_PATH}/{name}"
        with urllib.request.urlopen(raw) as r:
            data = r.read()
        with open(out + ".part", "wb") as f:
            f.write(data)
        os.replace(out + ".part", out)
        fetched += 1
    print(f"{dest}: {len(names)} essays, {fetched} fetched")


if __name__ == "__main__":
    fetch_paul_graham_essays()
