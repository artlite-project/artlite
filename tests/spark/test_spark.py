"""The Spark-X model on the NPU (``models.spark.spark.NpuSpark``) with its DSpark drafter: a sequence's state (the
server's prompt cache: checkpoints of the model's rings and the drafter's) and a conversation continued from one."""

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import spark_dspark_of
from models.sampling import Sampling
from models.spark.spark import NpuSpark, Spark

from ..model_files import require
from ..numerics import kl_rows
from . import ref

TURN1 = "Write a Python function that reverses a string."
TURN2 = "Now make it handle None input."


@pytest.mark.slow
def test_spark_continuation() -> None:
    """A first turn and its greedy reply (DSpark rounds), a checkpoint there (the reply's last token not run yet): the
    second turn after it, then again after a restore of the checkpoint and the caches' records before it (another
    sequence ran in between; the server's prompt cache in RAM): logits and the reply's tokens bit for bit; that and the
    whole conversation from position 0 (other prefill chunks, the reply's rows as prompt tokens) against the fp32 model:
    the second turn's next-token distribution within KL 0.05 (one row: a row's KL spans 1e-8 .. 3e-2 on either path),
    the same argmax."""
    path = ref.path("1.7B")
    dpath = spark_dspark_of(path)
    require(dpath)
    host = Spark(path)
    m = NpuSpark(GGUFFile(path), capacity=2048, dspark=dpath)
    greedy = Sampling(greedy=True)
    first = [{"role": "user", "content": TURN1}]
    t1 = host.chat(first)
    reply = list(m.generate(m.forward(t1), 40, greedy, t1))  # (cut short of its end: the template closes it)
    cp = m.checkpoint()
    assert cp.position == len(t1) + len(reply) - 1
    head = host.chat_text(first) + host.tokenizer.decode(reply)
    text = host.chat_text([*first, {"role": "assistant", "content": host.tokenizer.decode(reply)},
                           {"role": "user", "content": TURN2}])
    assert text.startswith(head)
    full = t1 + reply + host.tokenizer.encode(text[len(head):])
    rest = full[cp.position:]

    def second() -> tuple[np.ndarray, list[int]]:
        logits = m.forward(rest)
        return logits.copy(), list(m.generate(logits, 40, greedy, full))

    records = m.read_records(0, cp.position)  # (the caches' records before it: the prompt cache's RAM copy)
    a = second()
    m.reset()
    m.forward(host.chat([{"role": "user", "content": "Hello there."}]))
    m.write_records(records, 0, cp.position)
    m.restore(cp.position, cp)
    b = second()
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    m.reset()
    whole = m.forward(full)
    want = ref.cpu_logits(("1.7B", "Q4_K_M", "XHToken"), full)[-1:]

    def kl(x: np.ndarray) -> float:
        return float(kl_rows(want, torch.from_numpy(np.asarray(x, np.float32))[None])[0])

    print(f"KL against the fp32 model: cached {kl(a[0]):.2e}, whole {kl(whole):.2e}; argmax {int(a[0].argmax())}, "
          f"{int(whole.argmax())}, fp32 {int(want.argmax())}")
    assert max(kl(a[0]), kl(whole)) < 0.05 and a[0].argmax() == whole.argmax()
