"""llama.cpp's CPU run of a token sequence as a reference, for every architecture.

``golden`` runs a token sequence through llama.cpp on the CPU (``qwen35_dump.cpp``, built against the
llama.cpp tree of ``model_config.LLAMA_CPP_DIR``) and returns the tensors it dumped: per layer the named
intermediates of llama.cpp's graph (``src/models/qwen35.cpp``: e.g. ``attn_norm``, ``linear_attn_qkv_mixed``,
``z``, ``alpha``, ``beta``, ``conv_output_silu``, ``attn_output`` - the GDN core output, but after W_o in the
full-attention layers -, ``new_state``, ``final_output``, ``Qcur_full``, ``Kcur.1`` - after RoPE -,
``attn_pregate``, ``l_out``; a ``qwen35moe`` layer's MoE in ``src/llama-graph.cpp``'s ``build_moe_ffn`` and
``src/models/qwen35moe.cpp``: ``attn_post_norm`` - its input -, ``ffn_moe_logits``, ``ffn_moe_probs``, ``ffn_moe_topk``
- the experts' ids, int32 -, ``ffn_moe_weights_norm``, ``ffn_moe_gate`` / ``_up`` / ``_down`` (tokens, k, ...),
``ffn_moe_weighted``, ``ffn_moe_out``, ``ffn_shexp``, ``shared_expert_gate``, ``ffn_out``), and the logits of every
position; with an MTP block (``mtp``: an MTP file, or a model file with a block of its own: Qwen3.8) its block over
the same positions (``src/models/qwen35moe.cpp``'s / ``qwen35.cpp``'s ``graph_mtp``:
``mtp_eh_proj``, ``mtp_attn_out``, ``mtp_attn_residual``, ``mtp_ffn_out``, ``mtp_post_ffn``, ``h_nextn``, its logits:
``g.mtp(name, layer)``, ``g.mtp_logits``), each row fed the model's ``h_nextn`` of the row before. They are cached under ``tests/refs/<model>/golden_<hash>/`` (gitignored; ``--refresh-refs``
recomputes). llama.cpp quantizes the activations of its Q4_K / Q6_K matmuls to Q8_K, so its tensors carry that
error, and its prompt and one-token graphs differ by ~1% after a few layers: compare layers by tolerance, the model
by KL. In a ``qwen35moe`` model they differ by ~15% at the last layer and in the top 8 of 40-70% of the tokens
there: any rounding change moves a near-tied expert, and the difference grows (a 1e-7 change of the router's logits
reaches the same KL), so its layers are compared where the routing agrees.
"""

import hashlib
import json
import os

import numpy as np

from .. import common
from ..llama_cpp import lib_identity
from .runner import build_tool, cached_golden, run_tool

DUMP_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35_dump.cpp")


class Golden:
    """The tensors of one llama.cpp run (memory-mapped .npy files): ``g["attn_norm", 3]``, ``g["result_norm"]``,
    ``g.logits`` ((tokens, vocab)); ``g.meta``: what ran."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)

    def _file(self, name: str, layer: int | None) -> str:
        """"Kcur.1", 3 -> Kcur-3.1.npy (the second tensor of that name in a graph)."""
        base, dot, dup = name.rpartition(".")
        if not (dot and dup.isdigit()):
            base, dup = name, ""
        return os.path.join(self.path, (base if layer is None else f"{base}-{layer}") + (f".{dup}" if dup else "") + ".npy")

    def __contains__(self, key: str | tuple[str, int]) -> bool:
        name, layer = key if isinstance(key, tuple) else (key, None)
        return os.path.exists(self._file(name, layer))

    def __getitem__(self, key: str | tuple[str, int]) -> np.ndarray:
        name, layer = key if isinstance(key, tuple) else (key, None)
        path = self._file(name, layer)
        if not os.path.exists(path):
            have = sorted(f[:-4] for f in os.listdir(self.path) if f.endswith(".npy"))
            raise KeyError(f"{key}: not dumped (have {have})")
        return np.load(path, mmap_mode="r")

    @property
    def logits(self) -> np.ndarray:
        return self["logits"]

    def mtp(self, name: str, layer: int | None = None) -> np.ndarray:
        """A tensor of the MTP block's run (``golden(mtp=)``)."""
        return self[("mtp." + name, layer) if layer is not None else "mtp." + name]

    @property
    def mtp_logits(self) -> np.ndarray:
        return self["mtp.logits"]

    def dflash(self, name: str, layer: int | None = None) -> np.ndarray:
        """A tensor of the DFlash2 drafter's run (``golden(dflash=)``)."""
        return self[("dflash." + name, layer) if layer is not None else "dflash." + name]


def golden(model: str, tokens: np.ndarray | list[int], *, ubatch: int = 0, split: int = -1,
           layers: tuple[int | str, ...] = (), names: str = "", logits: str = "f16", states: str = "last",
           threads: int = 16, mtp: str = "", dflash: str = "", dflash_at: int = -1, dflash_names: str = "",
           dflash_random: bool = False) -> Golden:
    """llama.cpp's tensors for the token sequence (teacher forcing). ``ubatch``: tokens per graph (0: all in
    one; 1: decode graphs); ``split`` = P: the first P tokens in ubatches of ``ubatch``, then one per graph.
    ``layers``: layer indices whose tensors matching ``names`` (a regex over the names without the layer
    suffix) are dumped ("g": the tensors without a layer, "all": every layer). ``logits``: "f32", "f16" or
    "none". ``states``: the recurrent states ("new_state", ...) of the last ubatch, or "all" of them. ``mtp``: an MTP
    file whose block then runs over the same rows (its ``layers`` / ``names`` tensors, its logits). ``dflash``: a DFlash2
    file, llama.cpp's draft-dflash after the run: rows 0 .. ``dflash_at`` - 1 injected, the block at ``dflash_at``
    (``qwen35_dump.cpp``: ``g.dflash(name, layer)``, ``g["dflash.taps"]``, ``g["dflash.lattice"]``; ``dflash_names``: a regex over
    its tensors' names; ``dflash_random``: random taps, the model does not run)."""
    tokens = np.ascontiguousarray(tokens, np.int32)
    key = (common.model_key(model), hashlib.sha256(tokens.tobytes()).hexdigest(), ubatch, split,
           tuple(str(x) for x in layers), names, logits, states, lib_identity())
    if mtp:
        key += (common.model_key(mtp),)
    if dflash:
        key += (common.model_key(dflash), dflash_at, dflash_names, dflash_random)

    def run(tmp: str) -> None:
        tokens.tofile(os.path.join(tmp, "tokens.i32"))
        cmd = [build_tool("qwen35_dump", DUMP_SRC), "-m", model, "-t", os.path.join(tmp, "tokens.i32"), "-o", tmp,
               "-u", str(ubatch), "--split", str(split), "--logits", logits, "--states", states, "--threads", str(threads)]
        if layers:
            cmd += ["-l", ",".join(str(x) for x in layers), "-n", names or ".*"]
        if mtp:
            cmd += ["--mtp", mtp]
        if dflash:
            cmd += ["--dflash", dflash, "--dflash-at", str(dflash_at), "--dflash-names", dflash_names or "$^"]
            cmd += ["--dflash-random"] if dflash_random else []
        run_tool(cmd)

    return Golden(cached_golden(model, "golden", key, DUMP_SRC, run))
