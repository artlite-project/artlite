"""Fetch the datasets the tests and benchmarks read (gitignored next to this script).

paul_graham_essays/: the 49 essays of the needle-in-a-haystack benchmark (also RULER's haystack), the long-prompt
corpus. Long prompts concatenate them sorted by file name and repeat the sequence when they need more tokens
(~149K Llama 3 tokens for the whole set). Pinned to one upstream commit: the tests' cached references depend on the
exact text.

librispeech/: the speech clips of the audio tests: LibriSpeech test-clean's first chapter in its archive (speaker
6930, chapter 75918: 21 utterances, 1.9 .. 23.3 s, 179 s together; CC BY 4.0), as 16-bit PCM WAV files and the
chapter's transcript (``6930-75918.trans.txt``). Only the archive's first members are read (a few MB of its 346).
"""

import json
import os
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ESSAYS_REPO = "gkamradt/needle-in-a-haystack"
ESSAYS_COMMIT = "021385d68d3202e37893e9d3cd29011c569abe30"
ESSAYS_PATH = "needlehaystack/PaulGrahamEssays"
LIBRISPEECH_URL = "https://openslr.elda.org/resources/12/test-clean.tar.gz"  # (an openslr mirror)
LIBRISPEECH_CHAPTER = "test-clean/6930/75918/"
LIBRISPEECH_CLIPS = 21


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


def fetch_librispeech(dest: str = os.path.join(HERE, "librispeech")) -> None:
    """The chapter's utterances as WAV files and its transcript into dest (module docstring); needs miniaudio (FLAC)."""
    import miniaudio
    import numpy as np

    from models.lfm3a.host import wav_bytes

    trans = os.path.join(dest, "6930-75918.trans.txt")
    if os.path.exists(trans) and len([f for f in os.listdir(dest) if f.endswith(".wav")]) == LIBRISPEECH_CLIPS:
        print(f"{dest}: present")
        return
    os.makedirs(dest, exist_ok=True)
    got = 0
    with urllib.request.urlopen(LIBRISPEECH_URL) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            if LIBRISPEECH_CHAPTER not in m.name or not m.isfile():
                if got:  # (the chapter's members are consecutive)
                    break
                continue
            data = tar.extractfile(m).read()
            name = os.path.basename(m.name)
            if name.endswith(".flac"):
                pcm = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=16000)
                data, name = wav_bytes(np.frombuffer(pcm.samples, np.int16) / 32768), name[:-5] + ".wav"
                got += 1
            with open(os.path.join(dest, name + ".part"), "wb") as f:
                f.write(data)
            os.replace(os.path.join(dest, name + ".part"), os.path.join(dest, name))
    assert got == LIBRISPEECH_CLIPS, got
    print(f"{dest}: {got} clips")


if __name__ == "__main__":
    fetch_paul_graham_essays()
    fetch_librispeech()
