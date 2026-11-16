"""Spark-X's host side (models.spark.spark, model_config.SparkConfig) on the 1.7B Q4_K_M: the config, the chat template
as llama-server's ``/apply-template`` renders it (llama.cpp 80d2c0c2c, ``--jinja``: tools, two calls and two results in
a row, thinking with an earlier reply's thought), the prompt's tokens as llama-tokenize's, the reasoning levels and the
drafter found by name (no NPU)."""

import jinja2
import pytest

from artlite.model_config import SparkConfig, spark_dspark_of
from models import open_model
from models.spark.spark import Spark

from ..llama_cpp import llama_tokenize
from . import ref

SYS = "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant."
EFFORT = ("\n`thinking_effort` guides on how much to think in your thinking channel (not including the response "
          "channel), supported values include `low` and `high`.\nNow the system is invoked with `thinking_effort={}`.")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]


@pytest.fixture(scope="module")
def model() -> Spark:
    return open_model(ref.path())


def test_config(model: Spark) -> None:
    c = model.cfg
    assert isinstance(c, SparkConfig) and (c.num_hidden_layers, c.hidden_size, c.intermediate_size) == (28, 2048, 6656)
    assert c.layers("full") == [3, 7, 11, 15, 19, 23, 27] and c.sliding_window == 512 and not c.tied
    assert (c.rot("full"), c.rot("swa"), c.rope_base, c.rope_base_swa) == (32, 128, 5e6, 1e4)
    th = c.rope_theta([0, 1, 1000], "swa")
    assert th.shape == (3, 128) and th[1, 0] == 1 and abs(th[2, 127] / 1000 - 1e4 ** (-254 / 256)) < 1e-6


def test_template(model: Spark) -> None:
    calls = [{"role": "user", "content": "Weather in Paris?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}},
                 {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lyon", "days": 1}}}]},
             {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "rain"}]
    want = (SYS + '\n\n## Tools\nYou have access to the following functions:\n<tools>\n{"name":"get_weather",'
            '"description":"Weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":'
            '{"type":"integer"}}}}\n</tools><｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Weather in Paris?'
            "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think><tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call><tool_call>"
            "get_weather<arg_key>city</arg_key><arg_value>Lyon</arg_value><arg_key>days</arg_key><arg_value>1"
            "</arg_value></tool_call><｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Tool|><tool_response>sunny"
            "</tool_response><tool_response>rain</tool_response><｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>"
            "</think>")
    assert model.chat_text(calls, tools=TOOLS) == want
    assert model.chat(calls, tools=TOOLS) == llama_tokenize(ref.path(), want)
    turn = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!", "reasoning_content": "think a"},
            {"role": "user", "content": "What is 2+2?"}]
    tail = ("<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Hi<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>{}"
            "Hello!<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>What is 2+2?<｜end▁of▁sentence｜>"
            "<｜start▁of▁sentence｜><|Bot|><think>")
    low = {"reasoning_effort": "low"}
    assert model.chat_text(turn, think=True, preserve=True, variables=low) == \
        SYS + EFFORT.format("low") + tail.format("<think>think a</think>")
    assert model.chat_text(turn, think=True) == SYS + EFFORT.format("high") + tail.format("</think>")  # (dropped)
    assert model.chat_text(turn[:1]).endswith("<|User|>Hi<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>")
    with pytest.raises(jinja2.TemplateError, match="Unexpected reasoning effort medium"):
        model.chat_text(turn, think=True, variables={"reasoning_effort": "medium"})


def test_levels_sampling_drafter(model: Spark) -> None:
    assert [model.efforts.get(e, e) for e in ("minimal", "low", "medium", "high", "xhigh")] == \
        ["low", "low", "high", "high", "high"]
    s = model.sampling()
    assert (s.temp, s.top_k, round(s.top_p, 3)) == (1.0, 1000, 0.95)  # (the file's top-k off: TOP_K_ALL)
    assert model.dspark == spark_dspark_of(ref.path()) and model.dspark.endswith("Spark-X2.6-1.7B-DSpark.gguf")
    assert open_model(ref.path(), spec_type="none").dspark is None
    assert model.encode("Hello") == llama_tokenize(ref.path(), "Hello")  # (no BOS: the file's add_bos_token)
