"""The word layouts of the decode overlay, its models' cores and the vision core (host only): importing them runs their
checks (``operators.words``: bounds, overlaps, flag bits), their kernel constants compose per design, and the kernel
templates find every constant they name."""

import glob
import importlib
import os
import re

import pytest
from jinja2 import meta

from artlite.decode import attn as decode_attn
from artlite.decode import protocol
from artlite.operators import templating
from artlite.operators.words import LayoutError
from artlite.vision import core as vision_core
from models.lfm import conv as lfm_conv
from models.qwen35 import attn as qwen_attn
from models.qwen35 import dflash_core, gdn, words

# the layout names a template may use: the words, flags and values of the layouts (their prefixes)
_NAMES = re.compile(r"(M|F|S|B|K|C|A|G|V|MG|E|O|ST|DW|DW1|REC|ROLE)_[A-Z0-9_]+")
# what the renderers pass besides the layouts' constants (sizes and addresses of the design, not words)
_EXPLICIT = {"C_LEN", "B_G4", "V_NORM", "REC_K", "REC_V", "A_HI", "A_OBJ", "L_DONE", "L_HDONE", "L_HRDY", "H_AB", "H_DN", "H_PRM"}


def _templates() -> dict[str, dict[str, int]]:
    """Template file -> the constants its renderers pass from the layouts."""
    out = {
        "artlite/decode/design.cc.jinja": protocol.consts(),
        "artlite/decode/attn.cc.jinja": decode_attn.consts(),
        "artlite/decode/gemv.cc.jinja": {},
        "models/lfm/conv.cc.jinja": lfm_conv.STATE.consts() | decode_attn.consts(),
        "artlite/vision/core.cc.jinja": vision_core.consts(),
    }
    out |= {f"models/qwen35/{t}": qwen_attn.consts() for t in ("attn.cc.jinja", "attn.main.cc.jinja", "attn.main_one.cc.jinja",
                                                              "attn.one.cc.jinja")}
    out |= {f"models/qwen35/{t}": dflash_core.consts() for t in ("dflash_core.att.cc.jinja", "dflash_core.conv.cc.jinja")}
    root = os.path.dirname(os.path.dirname(templating.__file__))
    for path in glob.glob(os.path.join(os.path.dirname(root), "models", "qwen35", "gdn*.jinja")):
        out[f"models/qwen35/{os.path.basename(path)}"] = gdn.consts()
    return out


@pytest.mark.parametrize("name", sorted(_templates()))
def test_template_names_are_declared(name: str) -> None:
    have = _templates()[name]
    used = {v for v in meta.find_undeclared_variables(templating.JINJA_ENV.parse(templating.source(name)))
            if _NAMES.fullmatch(v)}
    assert not used - set(have) - _EXPLICIT, f"{name}: {sorted(used - set(have) - _EXPLICIT)} are declared nowhere"


@pytest.mark.parametrize("module", ["artlite.decode.design", "artlite.vision.design", "models.gemma4.decode",
                                    "models.llama.decode", "models.lfm.decode", "models.qwen35.decode",
                                    "models.qwen35.dflash_core"])
def test_layouts_import(module: str) -> None:
    """The designs' modules import (every layout of their cores is checked at import)."""
    importlib.import_module(module)


def test_decode_design_words() -> None:
    c = protocol.consts()
    # (the X metadata's header and the kinds' words: the values the kernels and the host agree on)
    assert [protocol.M_FLAGS, protocol.M_SKIP, protocol.M_SLOT, protocol.M_MODEL] == [0, 20, 23, 24]
    assert c["M_TOK_NG"] == 48 and c["M_RBASE"] == 50 and c["S_SG"] == 48 and c["C_KIND"] == 4
    # (a design takes each flag bit once, whichever model's kinds it composes)
    for kinds in ((words.GDN_IN, words.ATT, words.FLAGS), (dflash_core.CONV, dflash_core.ATT, words.FLAGS)):
        protocol.XMETA.consts(protocol.PHASE, protocol.TOKEN, protocol.MOE, *kinds)
    protocol.XMETA.consts(protocol.PHASE, protocol.TOKEN, protocol.MOE, decode_attn.ATT, decode_attn.TOKEN_ATT,
                          lfm_conv.CONV, lfm_conv.CONV_TOKEN)


def test_two_attention_flags_never_meet() -> None:
    """The attention cores of the shared design (``decode.attn``) and qwen35's name a different bit F_ATT."""
    assert decode_attn.F_ATT != words.F_ATT
    with pytest.raises(LayoutError, match="F_ATT is"):
        protocol.XMETA.consts(decode_attn.ATT, words.FLAGS)


def test_state_words_of_exclusive_features() -> None:
    protocol.STATE.together(protocol.POST_STATE, protocol.TOK_STATE)
    with pytest.raises(LayoutError, match="never meet"):
        protocol.STATE.together(protocol.POST_STATE, protocol.MOE_STATE)
    with pytest.raises(LayoutError, match="never meet"):
        protocol.STATE.together(protocol.TOK_STATE, qwen_attn.GATES)  # (the gates take the token objects' words)
    protocol.STATE.together(protocol.POST_STATE, qwen_attn.GATES)


def test_vision_command_regions() -> None:
    k = vision_core
    assert k.C_CALLS == k.C_MASKS + k.MASK_BYTES // 4 and k.C_PATS == k.C_CALLS + k.CALL * k.NCALL
    assert k.C_PHASE + 24 == k.C_CALLS + k.CONV_WORDS <= k.CMD_WORDS
    assert k.C_KMASK[0] + 64 <= k.C_WALKS and k.C_KMASK[1] + 64 <= k.CMD_WORDS
