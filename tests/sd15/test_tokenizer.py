"""Stable Diffusion 1.5's CLIP tokenizer (models.sd15.tokenizer) against the ids of Hugging Face's CLIPTokenizer
(transformers 5.19) for the model's vocabulary: prompts, punctuation, contractions, Unicode, no HTML unescaping,
special tokens, truncation (no NPU)."""

import os

import pytest

from artlite.model_config import sd15_dir
from models.sd15.tokenizer import ClipTokenizer

BOS, EOS = 49406, 49407
# text -> its ids up to the first EOS (then EOS up to 77)
GOLDEN = [
    ("a photo of a cat", [49406, 320, 1125, 539, 320, 2368, 49407]),
    ("a photograph of an astronaut riding a horse", [49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407]),
    ("", [49406, 49407]),
    ("A Beautiful   Landscape, highly-detailed; 4k (8K) [HDR]!!",
     [49406, 320, 1215, 5727, 267, 5302, 268, 12609, 282, 275, 330, 263, 279, 330, 264, 314, 28065, 60, 748, 49407]),
    ("don't stop: it's the cat's toy, isn't it?",
     [49406, 847, 713, 1691, 281, 585, 568, 518, 2368, 568, 5988, 267, 2923, 713, 585, 286, 49407]),
    ("naïve café — “quotes” 日本語 😀",
     [49406, 1097, 35689, 563, 15304, 2005, 728, 506, 5808, 728, 507, 39121, 44353, 34002, 508, 7334, 49407]),
    ("tom &amp; jerry", [49406, 2435, 261, 6259, 282, 9164, 49407]),
    ("under_score #hashtag 3.14159 x² ½",
     [49406, 1798, 318, 4431, 258, 9238, 274, 269, 272, 275, 272, 276, 280, 343, 41175, 33613, 49407]),
    # a final capital sigma lowercases to σ; special tokens only as written
    ("ΣΑΣ <|endoftext|> <|ENDOFTEXT|>", [49406, 139, 225, 138, 109, 139, 481, 49407, 27, 347, 40786, 4160, 91, 285,
                                         49407]),
]


@pytest.fixture(scope="module")
def tok() -> ClipTokenizer:
    path = os.path.join(sd15_dir(), "tokenizer")
    if not os.path.isdir(path):
        pytest.skip(f"{path}: not present")
    return ClipTokenizer(path)


@pytest.mark.parametrize("text,ids", GOLDEN, ids=[f"golden{i}" for i in range(len(GOLDEN))])
def test_encode(tok: ClipTokenizer, text: str, ids: list[int]) -> None:
    assert tok.encode(text) == ids + [EOS] * (77 - len(ids))


def test_truncation(tok: ClipTokenizer) -> None:
    text = " ".join(f"word{i}" for i in range(60))  # 170 tokens: "word", then its digits one by one
    ids = tok.encode(text)
    assert len(ids) == 77
    assert ids[:6] == [BOS, 2653, 271, 2653, 272, 2653] and ids[-4:] == [273, 278, 2653, EOS]
    assert ids[1:-1] == tok.tokenize(text)[:75]
