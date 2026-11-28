"""A request's structured output: the fields of every API that ask for one, parsed into one ``structured.Constraint``
in the request's thread (``constraint_of``; the agents' APIs carry theirs as a chat body's ``response_format``:
``agents``), compiled there before the job is queued (``ApiContext.compile``: an error is a 400 that names the field).
A field sent as null is unset.

| field | API | constraint |
|---|---|---|
| ``grammar`` | llama-server | GBNF; ``%llguidance`` first: Lark (llama.cpp built with LLAMA_LLGUIDANCE) |
| ``json_schema`` | llama-server | a JSON schema, lenient; ``{}``: any JSON object |
| ``response_format`` ``json_object`` | OpenAI, llama-server | any JSON object, or its ``schema`` (llama.cpp's), lenient |
| ``response_format`` ``json_schema`` | OpenAI, llama-server | ``json_schema.schema`` (or a flat ``schema``), exact unless ``strict`` is false |
| ``response_format`` ``structural_tag`` | vLLM | ``structures`` [{begin, schema, end}] after ``triggers`` in free text |
| ``structured_outputs`` | vLLM | one of ``json``, ``regex``, ``choice``, ``grammar``, ``json_object``, ``structural_tag``; ``whitespace_pattern``, ``disable_any_whitespace``, ``disable_additional_properties`` |
| ``guided_json``, ``guided_regex``, ``guided_choice``, ``guided_grammar``, ``guided_whitespace_pattern`` | vLLM before 0.12 | as ``structured_outputs``' (``guided_decoding_backend`` ignored) |

Two of these at once are an error (llama-server's "Cannot use both json_schema and grammar"), but as llama-server: a
``response_format`` ``json_object`` without a schema takes the top-level ``json_schema``, and on the completion routes
``grammar`` wins over ``json_schema``. An empty text (a ``grammar``, a ``response_format`` type) is unset."""

import json
from collections.abc import Mapping
from typing import Any

from llguidance.gbnf_to_lark import is_lark_syntax

from ..structured import Constraint, ConstraintError
from .errors import bad_request

RESPONSE_FORMATS = ("text", "json_object", "json_schema", "structural_tag")
VLLM_KINDS = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
GUIDED = {"guided_json": "json", "guided_regex": "regex", "guided_choice": "choice", "guided_grammar": "grammar"}


def _set(value: object) -> bool:
    return value is not None and value != ""


def constraint_of(body: Mapping[str, Any], *, route: str = "chat") -> tuple[Constraint, str] | None:
    """The constraint a request's body asks for and the field it came from, or None; ``route``: ``completion`` (raw
    prompts: llama-server's ``grammar`` wins over ``json_schema`` there). ApiError (400): two at once, a field of the
    wrong shape, a schema over the limits."""
    given = [k for k in ("grammar", "json_schema") if _set(body.get(k))]
    if route == "completion" and given == ["grammar", "json_schema"]:
        given = ["grammar"]
    rf = body.get("response_format")
    if _set(rf):
        if not isinstance(rf, Mapping):
            raise bad_request("response_format must be an object")
        # (a json_object without its own schema takes the top-level json_schema's, as llama-server does)
        if rf.get("type") not in (None, "", "text") and not (rf.get("type") == "json_object" and
                                                          not _set(rf.get("schema")) and "json_schema" in given):
            given.append("response_format")
    so = body.get("structured_outputs")
    if _set(so):
        given.append("structured_outputs")
    guided = [k for k in GUIDED if _set(body.get(k))]
    if guided:
        given.append(guided[0])
        if len(guided) > 1:
            raise bad_request(f"Cannot use both {guided[0]} and {guided[1]}")
    if len(given) > 1:
        raise bad_request(f"Cannot use both {given[0]} and {given[1]}")
    if not given:
        return None
    field = given[0]
    try:
        if field == "grammar":
            return Constraint.of_grammar(body["grammar"]), field
        if field == "json_schema":
            return Constraint.of_schema(body["json_schema"], name="json_schema"), field
        if field == "response_format":
            return _response_format(rf), field
        options = so if field == "structured_outputs" else \
            {GUIDED[field]: body[field], "whitespace_pattern": body.get("guided_whitespace_pattern")}
        return _vllm(options), field
    except ConstraintError as e:
        raise bad_request(f"{field}: {e}") from None




def _response_format(rf: Mapping[str, Any]) -> Constraint:
    kind = rf.get("type")
    if kind == "json_object":
        schema = rf.get("schema")
        return Constraint.of_schema(schema if _set(schema) else {}, name="response_format")
    if kind == "json_schema":
        inner = rf.get("json_schema")
        spec = inner if isinstance(inner, Mapping) else rf  # (the flat form: llama.cpp's README, the Responses API's)
        schema = spec.get("schema")
        if not _set(schema):
            raise ConstraintError("a json_schema needs its schema (json_schema.schema)")
        name = spec.get("name") if isinstance(spec.get("name"), str) else ""
        return Constraint.of_schema(schema, strict=spec.get("strict") is not False, name=name or "response_format")
    if kind == "structural_tag":
        return _structural_tag(rf)
    raise ConstraintError(f"response_format type must be one of {', '.join(RESPONSE_FORMATS)}, got {kind!r}")


def _structural_tag(spec: Mapping[str, Any], **options: Any) -> Constraint:
    """vLLM's structural tag: free text, and after one of ``triggers`` a structure, its ``begin``, a JSON value of its
    ``schema`` and its ``end`` (llguidance's ``StructTag``)."""
    if "format" in spec and "structures" not in spec:
        raise ConstraintError("a structural_tag's 'format' (xgrammar's) is not supported: give structures and triggers")
    structures, triggers = spec.get("structures"), spec.get("triggers")
    if not isinstance(structures, list) or not structures or not isinstance(triggers, list) or \
            not all(isinstance(t, str) and t for t in triggers):
        raise ConstraintError("a structural_tag needs structures [{begin, schema, end}] and triggers [text]")
    out = []
    for s in structures:
        if not isinstance(s, Mapping) or not isinstance(s.get("begin"), str) or not isinstance(s.get("end", ""), str):
            raise ConstraintError("a structure is {begin, schema, end}")
        trigger = next((t for t in triggers if s["begin"].startswith(t)), None)
        if trigger is None:
            raise ConstraintError(f"structure {s['begin']!r} begins with none of the triggers {triggers}")
        schema = s.get("schema")
        out.append({"trigger": trigger, "begin": s["begin"], "end": s.get("end", ""),
                    "schema": schema if _set(schema) else {}})
    return Constraint("structural_tag", json.dumps(out, ensure_ascii=False), **options)


def _vllm(so: object) -> Constraint:
    """vLLM's ``structured_outputs`` (or a request's ``guided_*`` fields as one)."""
    if not isinstance(so, Mapping):
        raise ConstraintError("structured_outputs must be an object")
    kinds = [k for k in VLLM_KINDS if _set(so.get(k))]
    if len(kinds) != 1:
        raise ConstraintError(f"one of {', '.join(VLLM_KINDS)} (" + (f"{len(kinds)} given: {', '.join(kinds)}" if kinds
                                                                      else "none given") + ")")
    kind = kinds[0]
    pattern = so.get("whitespace_pattern")
    if _set(pattern) and not isinstance(pattern, str):
        raise ConstraintError("whitespace_pattern is a regex")
    options = {"whitespace": pattern or None, "compact": bool(so.get("disable_any_whitespace")),
               "closed": bool(so.get("disable_additional_properties"))}
    value = so[kind]
    if kind == "json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as e:
                raise ConstraintError(f"json is not JSON: {e}") from None
        return Constraint.of_schema(value, name="json", **options)
    if kind == "json_object":
        return Constraint("json_object", "", **options)
    if kind == "structural_tag":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as e:
                raise ConstraintError(f"structural_tag is not JSON: {e}") from None
        if not isinstance(value, Mapping):
            raise ConstraintError("structural_tag: {structures, triggers}")
        return _structural_tag(value, **options)
    if kind == "choice":
        if not isinstance(value, list) or not value or not all(isinstance(c, str) for c in value):
            raise ConstraintError("choice is a non-empty list of texts")
        return Constraint("choice", json.dumps(value, ensure_ascii=False))
    if not isinstance(value, str):
        raise ConstraintError(f"{kind} is a text")
    if kind == "regex":
        return Constraint("regex", value)
    return Constraint("lark" if is_lark_syntax(value) else "gbnf", value)  # (llguidance's ``grammar_from("grammar")``)
