"""safetensors files read without the safetensors package: an 8-byte header length, a JSON header (name -> dtype,
shape, byte range after the header), then the tensors' bytes, each read straight into its torch tensor."""

import json
import struct

import torch

DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32}


def read_header(path: str) -> tuple[dict[str, dict], int]:
    """The tensors' entries (``dtype``, ``shape``, ``data_offsets``) and where their data starts in the file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def load_safetensors(path: str, prefixes: tuple[str, ...] = (), *, contains: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
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
            t = torch.empty(e["shape"], dtype=DTYPES[e["dtype"]])
            f.seek(base + begin)
            if f.readinto(t.view(-1).view(torch.uint8).numpy()) != end - begin:
                raise ValueError(f"{path}: {name}: truncated file or a byte range that does not fit the shape")
            out[name] = t
    return out
