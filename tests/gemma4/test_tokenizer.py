"""The Gemma 4 tokenizer and chat template on fixed examples (ids as llama.cpp's llama-tokenize; no NPU)."""

from artlite.gguf_model import GGUFFile
from artlite.model_config import gemma_gguf
from models.gemma4.tokenizer import GemmaTokenizer, chat_prompt

from ..model_files import require


def test_gemma_tokenizer() -> None:
    path = gemma_gguf("Q4_0")
    require(path)
    g = GGUFFile(path)
    tok = GemmaTokenizer(g)
    cases = {
        "Hello, world!": [9259, 236764, 1902, 236888],
        "  two  spaces\n\nand a blank line": [138, 13498, 138, 35220, 108, 624, 496, 11580, 1757],
        "naïve café — 日本語 🙂": [1789, 238527, 560, 33443, 2192, 33375, 238582, 57235],
        "def f(x):\n\treturn x**2  # square": [2063, 517, 236769, 236781, 1473, 107, 255968, 2060, 1123, 1018, 236778, 138,
                                               236865, 6281],
    }
    for text, ids in cases.items():
        assert tok.encode(text) == ids, text
        assert tok.decode(ids) == text
    prompt = chat_prompt(g, [{"role": "user", "content": "Why is the sky blue?"}])
    assert prompt == "<bos><|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n"
    assert tok.encode(prompt) == [2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107]
