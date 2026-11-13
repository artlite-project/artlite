"""The drafter options of the tools (llama-server's names: ``--spec-type``, ``-md``, ``--spec-draft-n-max``) and the
drafter a model loads from them."""

import argparse

from artlite.gguf_model import GGUFFile
from artlite.model_config import model_path

# the drafter of a model's multi-token steps (``--spec-type``: llama-server's names, and auto: the model's own choice)
SPEC_TYPES = ("auto", "none", "draft-mtp", "draft-dflash")


def drafter_type(spec_type: str, draft_model: str | None, types: tuple[str, ...], name: str) -> str:
    """The drafter a model loads (a ``Model``'s ``spec_type``, ``draft_model`` options): ``spec_type`` (one of
    ``SPEC_TYPES``; ``types``: the drafters the model ``name`` has), or a drafter file's type (``draft_model``: by its
    architecture, ``dflash`` draft-dflash, any other draft-mtp). ValueError: a drafter the model does not have, or a
    file of another type than ``spec_type``."""
    if spec_type not in SPEC_TYPES:
        raise ValueError(f"spec_type: {spec_type!r} (one of {', '.join(SPEC_TYPES)})")
    if draft_model is not None:
        kind = "draft-dflash" if GGUFFile(draft_model).arch == "dflash" else "draft-mtp"
        if spec_type not in ("auto", kind):
            raise ValueError(f"{draft_model}: a {kind} drafter, not {spec_type}")
        spec_type = kind
    if spec_type not in ("auto", "none", *types):
        raise ValueError(f"{name} has no {spec_type} drafter (it has: {', '.join(types) or 'none'})")
    return spec_type


def draft_count(text: str) -> int:
    """``--spec-draft-n-max``'s value: 0 or more."""
    n = int(text)
    if n < 0:
        raise ValueError(f"{text}: 0 or more")
    return n


def add_spec_options(p: argparse.ArgumentParser | argparse._ArgumentGroup, *, n_max: bool = True) -> None:
    """The drafter's options (llama-server's names): ``--spec-type``, ``-md`` / ``--spec-draft-model`` and (``n_max``)
    ``--spec-draft-n-max``; ``spec_options`` reads them."""
    p.add_argument("--spec-type", choices=SPEC_TYPES, default="auto",
                   help="the drafter of the multi-token steps: auto, the model's found by file name (Qwen3.8 27B: its "
                        "DFlash2 file, else its MTP layer; Gemma 4, Qwen3.6 35B-A3B, Occamy: their MTP files); none "
                        "(the draft tokens from the context stay); draft-mtp, draft-dflash: that drafter, which must "
                        "exist (default: auto)")
    p.add_argument("-md", "--spec-draft-model", "--model-draft", metavar="FILE", type=model_path,
                   help="the drafter's GGUF file (an MTP or a DFlash2 drafter, by its architecture) in place of the one "
                        "found by name; a relative path is also looked up in the models directory")
    if n_max:
        p.add_argument("--spec-draft-n-max", type=draft_count, metavar="N",
                       help="draft tokens per step at most, the context's and the drafter's (default and at most: the "
                            "step's slots - 1, 3 or 7; 0: steps of one token, no drafter loaded)")


def spec_options(args: argparse.Namespace) -> dict[str, object]:
    """``open_model``'s options of ``add_spec_options``' (none of a drafter where a step takes no drafts:
    ``--spec-draft-n-max 0``)."""
    if getattr(args, "spec_draft_n_max", None) == 0:
        return {"spec_type": "none"}
    return {k: v for k, v in (("spec_type", args.spec_type), ("draft_model", args.spec_draft_model))
            if v not in ("auto", None)}
