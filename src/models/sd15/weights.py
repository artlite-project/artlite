"""safetensors files read without the safetensors package: an 8-byte header length, a JSON header (name -> dtype,
shape, byte range after the header), then the tensors' bytes, each read straight into its numpy array."""

import json
import struct

import numpy as np
from ml_dtypes import bfloat16

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": bfloat16, "I64": np.int64, "I32": np.int32}


def read_header(path: str) -> tuple[dict[str, dict], int]:
    """The tensors' entries (``dtype``, ``shape``, ``data_offsets``) and where their data starts in the file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def load_safetensors(path: str, prefixes: tuple[str, ...] = (), *, contains: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """The tensors of a file (``prefixes``: only those whose names start with one of them; ``contains``: or contain
    one), in file order."""
    header, base = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, e in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            if (prefixes or contains) and not (name.startswith(prefixes) if prefixes else False) and \
                    not any(c in name for c in contains):
                continue
            begin, end = e["data_offsets"]
            t = np.empty(e["shape"], DTYPES[e["dtype"]])
            f.seek(base + begin)
            if f.readinto(t.reshape(-1).view(np.uint8)) != end - begin:
                raise ValueError(f"{path}: {name}: truncated file or a byte range that does not fit the shape")
            out[name] = t
    return out
