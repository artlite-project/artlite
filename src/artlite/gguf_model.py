"""GGUF model files: their metadata and tensors with their quantization fields (GGUF names), read from the file's
memory map (``GGUFFile``).

A GGUF tensor of a quantized type is a (rows, row_bytes) uint8 matrix of
ggml blocks along the row (the reduction dimension K of a linear layer).
:class:`QTensor` keeps those raw bytes and extracts the per-format fields
(scales, mins, integer codes) exactly, so NPU packers can re-lay them out
losslessly (only fp16 scales become bf16, NPU side). ``dequantize`` gives
the fp32 weights the ggml reference would compute.

Supported: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_XXS (its grid: gguf-py's table,
``iq3xxs_grid``). A tensor of another type is listed (``tensor_info``: its size unknown, None) and raises
NotImplementedError when read.

llama.cpp's converter permutes the rows of a Llama's attn_q / attn_k per head (interleaved rotary pairs):
:func:`unpermute_rows` gives the rotate-half order (the weight images take the rows in that order).
"""

import json
import math
import mmap
import struct
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from . import _native


def _f16(b: np.ndarray) -> np.ndarray:
    """(..., 2) uint8 -> (...) float32 of the little-endian fp16."""
    return np.ascontiguousarray(b).view(np.float16)[..., 0].astype(np.float32)


@dataclass
class QTensor:
    """One weight matrix (rows = output features, cols = K)."""

    name: str
    qtype: str  # "F32", "BF16", "Q4_0", ...
    rows: int
    cols: int
    raw: np.ndarray  # (rows, row_bytes) uint8 for quantized types, else (rows, cols)

    def blocks(self, block_bytes: int) -> np.ndarray:
        """(rows, n_blocks, block_bytes) view of the raw data."""
        return self.raw.reshape(self.rows, -1, block_bytes)

    def fields(self) -> dict[str, np.ndarray]:
        """Exact quantization fields of the tensor.

        Q4_0:  d (rows, cols/32) f32 (fp16 values), q (rows, cols) u8 in [0, 15]; w = d (q - 8)
        Q4_1:  d, m (rows, cols/32), q u8 in [0, 15];                              w = d q + m
        Q8_0:  d (rows, cols/32), q (rows, cols) i8;                               w = d q
        Q4_K:  d, dmin (rows, cols/256), sc, mn (rows, cols/256, 8) u8 in [0, 63],
               q u8 in [0, 15];                          w = d sc_j q - dmin mn_j (j = 32-group)
        Q5_K:  as Q4_K, q u8 in [0, 31] (the high bit from the qh plane)
        Q6_K:  d (rows, cols/256), sc (rows, cols/256, 16) i8, q u8 in [0, 63];
                                                         w = d sc_j (q - 32) (j = 16-group)
        Q2_K:  d, dmin (rows, cols/256), sc, mn (rows, cols/256, 16) u8 in [0, 15], q u8 in [0, 3];
                                                         w = d sc_j q - dmin mn_j (j = 16-group)
        Q3_K:  d (rows, cols/256), sc (rows, cols/256, 16) i8 in [-32, 31], q u8 in [0, 7] (the
               low 2 bits | 4 x the hmask bit);           w = d sc_j (q - 4) (j = 16-group)
        IQ3_XXS: d (rows, cols/256), ls (rows, cols/32) u8 in [0, 15], idx (rows, cols/4) u8 grid indices
               (``iq3xxs_grid``: 4 values per entry), signs (rows, cols/8) u8 7-bit sign fields (bit j: value j
               negative; the 8th sign their parity), q (rows, cols) i8 in [-62, 62] the signed grid values;
                                                         w = d (2 ls_j + 1) / 4 q (j = 32-group)
        BF16:  w (rows, cols) u16 bf16 bit patterns
        F16:   w (rows, cols) float16
        F32:   w (rows, cols) float32
        """
        t = self.qtype
        if t == "F32":
            return {"w": self.raw}
        if t == "BF16":
            return {"w": self._bf16().view(torch.int16).numpy().view(np.uint16)}
        if t == "F16":
            return {"w": self.raw.view(np.float16)}
        if t == "Q4_0":
            b = self.blocks(18)
            q = np.empty((self.rows, b.shape[1], 32), np.uint8)
            q[..., :16] = b[..., 2:] & 0xF
            q[..., 16:] = b[..., 2:] >> 4
            return {"d": _f16(b[..., 0:2]), "q": q.reshape(self.rows, self.cols)}
        if t == "Q4_1":
            b = self.blocks(20)
            q = np.empty((self.rows, b.shape[1], 32), np.uint8)
            q[..., :16] = b[..., 4:] & 0xF
            q[..., 16:] = b[..., 4:] >> 4
            return {"d": _f16(b[..., 0:2]), "m": _f16(b[..., 2:4]), "q": q.reshape(self.rows, self.cols)}
        if t == "Q8_0":
            b = self.blocks(34)
            return {"d": _f16(b[..., 0:2]), "q": np.ascontiguousarray(b[..., 2:]).view(np.int8).reshape(self.rows, self.cols)}
        if t in ("Q4_K", "Q5_K"):
            b = self.blocks(144 if t == "Q4_K" else 176)
            s = b[..., 4:16]
            lo, hi, top = s[..., 0:4], s[..., 4:8], s[..., 8:12]
            sc = np.concatenate([lo & 63, (top & 0xF) | ((lo >> 6) << 4)], axis=-1)
            mn = np.concatenate([hi & 63, (top >> 4) | ((hi >> 6) << 4)], axis=-1)
            qs = b[..., 16 if t == "Q4_K" else 48:].reshape(self.rows, -1, 4, 32)
            q = np.stack([qs & 0xF, qs >> 4], axis=3)  # (rows, nb, 4 chunks, 2 halves, 32)
            if t == "Q5_K":  # group g = 2 chunk + half: bit g of qh[l] is the high bit of element l
                qh = b[..., 16:48]
                hb = (qh[..., None, :] >> np.arange(8, dtype=np.uint8)[:, None]) & 1  # (rows, nb, 8, 32)
                q = q | (hb.reshape(q.shape) << 4)
            return {"d": _f16(b[..., 0:2]), "dmin": _f16(b[..., 2:4]), "sc": sc, "mn": mn,
                    "q": q.reshape(self.rows, self.cols)}
        if t == "Q6_K":
            b = self.blocks(210)
            ql = b[..., 0:128].reshape(self.rows, -1, 2, 64)
            qh = b[..., 128:192].reshape(self.rows, -1, 2, 32)
            q = np.empty((self.rows, b.shape[1], 2, 4, 32), np.uint8)
            q[..., 0, :] = (ql[..., :32] & 0xF) | (((qh >> 0) & 3) << 4)
            q[..., 1, :] = (ql[..., 32:] & 0xF) | (((qh >> 2) & 3) << 4)
            q[..., 2, :] = (ql[..., :32] >> 4) | (((qh >> 4) & 3) << 4)
            q[..., 3, :] = (ql[..., 32:] >> 4) | (((qh >> 6) & 3) << 4)
            sc = np.ascontiguousarray(b[..., 192:208]).view(np.int8)
            return {"d": _f16(b[..., 208:210]), "sc": sc, "q": q.reshape(self.rows, self.cols)}
        if t in ("Q2_K", "Q3_K"):
            # element 128 n + 32 j + 16 h + l: bits 2 j of qs[32 n + 16 h + l] (Q3_K: the high bit is bit
            # 4 n + j of hmask[16 h + l])
            b = self.blocks(84 if t == "Q2_K" else 110)
            nb = b.shape[1]
            qs = b[..., 16:80] if t == "Q2_K" else b[..., 32:96]
            j = np.arange(4, dtype=np.uint8).reshape(1, 1, 1, 4, 1, 1)
            q = (qs.reshape(self.rows, nb, 2, 1, 2, 16) >> (2 * j)) & 3  # (rows, nb, n, j, h, l)
            if t == "Q2_K":
                s = b[..., 0:16]
                return {"d": _f16(b[..., 80:82]), "dmin": _f16(b[..., 82:84]), "sc": s & 0xF, "mn": s >> 4,
                        "q": q.reshape(self.rows, self.cols)}
            n = np.arange(2, dtype=np.uint8).reshape(1, 1, 2, 1, 1, 1)
            q = q | (((b[..., 0:32].reshape(self.rows, nb, 1, 1, 2, 16) >> (4 * n + j)) & 1) << 2)
            # 6-bit scales: low nibbles of bytes 0 .. 7 (scales 0 .. 7) then their high nibbles (8 .. 15), the
            # high crumbs of scale 4 k + i in bits 2 k of byte 8 + i
            s = b[..., 96:108]
            lo = np.concatenate([s[..., :8] & 0xF, s[..., :8] >> 4], axis=-1)
            hi = (s[..., 8:12, None] >> (2 * np.arange(4, dtype=np.uint8))) & 3  # (rows, nb, i, k)
            sc = lo | (hi.transpose(0, 1, 3, 2).reshape(self.rows, nb, 16) << 4)
            return {"d": _f16(b[..., 108:110]), "sc": (sc.astype(np.int16) - 32).astype(np.int8),
                    "q": q.reshape(self.rows, self.cols)}
        if t == "IQ3_XXS":
            # 98 B per 256: d, qs[0 .. 63] the grid indices (4 values each, in k order), 8 words (one per 32 k): four
            # 7-bit sign fields (bits 7 l .., 8 values each) and ls (bits 28 .. 31)
            b = self.blocks(98)
            nb = b.shape[1]
            idx = np.ascontiguousarray(b[..., 2:66])
            words = np.ascontiguousarray(b[..., 66:98]).view("<u4")  # (rows, nb, 8)
            sg = ((words[..., None] >> np.array([0, 7, 14, 21], np.uint32)) & 0x7F).astype(np.uint8)  # (rows, nb, 8, 4)
            s8 = sg | (_PARITY7[sg] << 7)
            neg = ((s8[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool)  # (rows, nb, 8, 4, 8)
            g = iq3xxs_grid()[idx].reshape(neg.shape)
            return {"d": _f16(b[..., 0:2]), "ls": (words >> 28).astype(np.uint8).reshape(self.rows, nb * 8),
                    "idx": idx.reshape(self.rows, nb * 64), "signs": sg.reshape(self.rows, nb * 32),
                    "q": np.where(neg, -g, g).reshape(self.rows, self.cols)}
        raise ValueError(f"{self.name}: no quantization fields for {t}")

    def dequantize(self) -> torch.Tensor:
        """fp32 (rows, cols) weights, computed from :meth:`fields`."""
        t = self.qtype
        if t == "F32":
            return torch.from_numpy(self.raw.astype(np.float32))
        if t in ("F16", "BF16"):
            return torch.from_numpy(self.raw.copy()).float() if t == "F16" else self._bf16().float()
        f = self.fields()
        r = self.rows
        if t == "Q4_0":
            w = f["d"][..., None] * (f["q"].reshape(r, -1, 32).astype(np.float32) - 8.0)
        elif t == "Q4_1":
            w = f["d"][..., None] * f["q"].reshape(r, -1, 32) + f["m"][..., None]
        elif t == "Q8_0":
            w = f["d"][..., None] * f["q"].reshape(r, -1, 32).astype(np.float32)
        elif t in ("Q4_K", "Q5_K"):
            a = f["d"][..., None] * f["sc"]  # (r, nb, 8)
            b = f["dmin"][..., None] * f["mn"]
            w = a[..., None] * f["q"].reshape(r, -1, 8, 32) - b[..., None]
        elif t in ("Q6_K", "Q3_K"):
            a = f["d"][..., None] * f["sc"].astype(np.float32)  # (r, nb, 16)
            w = a[..., None] * (f["q"].reshape(r, -1, 16, 16).astype(np.float32) - (32.0 if t == "Q6_K" else 4.0))
        elif t == "Q2_K":
            a = f["d"][..., None] * f["sc"].astype(np.float32)  # (r, nb, 16)
            b = f["dmin"][..., None] * f["mn"].astype(np.float32)
            w = a[..., None] * f["q"].reshape(r, -1, 16, 16).astype(np.float32) - b[..., None]
        elif t == "IQ3_XXS":  # (db = d (0.5 + ls) 0.5: ggml's operation order, exact in fp32)
            db = (np.repeat(f["d"], 8, axis=-1) * (np.float32(0.5) + f["ls"].astype(np.float32))) * np.float32(0.5)
            w = db[..., None] * f["q"].reshape(r, -1, 32).astype(np.float32)
        else:
            raise ValueError(t)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32).reshape(r, self.cols))

    def _bf16(self) -> torch.Tensor:
        """A view of the file's bf16 data (read-only memory: callers convert or copy it)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # torch warns about non-writable arrays
            return torch.from_numpy(np.ascontiguousarray(self.raw).view(np.int16)).view(torch.bfloat16)


_PARITY7 = np.array([i.bit_count() & 1 for i in range(128)], np.uint8)  # (IQ3_XXS: a sign field's 8th sign)


def iq3xxs_grid() -> np.ndarray:
    """IQ3_XXS's grid (ggml's ``iq3xxs_grid``, from gguf-py): (256, 4) int8, entry i = the 4 values of index i, each in
    {4, 12, 20, 28, 36, 44, 52, 62}."""
    from gguf.quants import IQ3_XXS

    IQ3_XXS.init_grid()
    return IQ3_XXS.grid.reshape(256, 4).astype(np.int8)


def unpermute_rows(n_heads: int, rows: int) -> np.ndarray:
    """Row order undoing llama.cpp's rotary permutation: HF row (h, j, i) =
    GGUF row (h, i, j) with j the rotate-half, i the pair index."""
    hd = rows // n_heads
    return np.arange(rows).reshape(n_heads, hd // 2, 2).transpose(0, 2, 1).reshape(-1)


# GGUF metadata value types: scalar struct formats / numpy dtypes (8: string, 9: array)
_KV_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_KV_STRING, _KV_ARRAY = 8, 9
# ggml tensor types: name, (weights per block, bytes per block)
_GGML_TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 8: ("Q8_0", 32, 34),
               10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
               14: ("Q6_K", 256, 210), 18: ("IQ3_XXS", 256, 98), 30: ("BF16", 1, 2)}
_BLOCKS = {name: (per, nb) for name, per, nb in _GGML_TYPES.values()}


class GGUFFile:
    """A GGUF file parsed directly from its memory map (``gguf.GGUFReader`` builds a numpy
    array per metadata part: 11 s for a 262K-token vocabulary with 515K merges; this takes 0.3 s).

    ``meta``: the scalar and string values, numeric arrays as numpy arrays; string arrays
    (vocabulary, merges) are decoded on demand (``strings``). ``tensor(name)``: a QTensor over
    the mapped data (nothing is read until it is used), in GGUF naming."""

    def __init__(self, path: str):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        buf = memoryview(self.data)
        assert bytes(buf[:4]) == b"GGUF", f"{path}: not a GGUF file"
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
        assert version in (2, 3), f"{path}: GGUF version {version}"
        pos = 24
        self.meta: dict[str, object] = {}
        self._string_arrays: dict[str, tuple[int, int]] = {}  # key -> (offset of the first string, count)

        def string(pos: int) -> tuple[str, int]:
            n = int.from_bytes(buf[pos:pos + 8], "little")
            return bytes(buf[pos + 8:pos + 8 + n]).decode("utf-8", errors="replace"), pos + 8 + n

        for _ in range(n_kv):
            key, pos = string(pos)
            (vt,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            if vt == _KV_STRING:
                self.meta[key], pos = string(pos)
            elif vt == _KV_ARRAY:
                et, count = struct.unpack_from("<IQ", buf, pos)
                pos += 12
                if et == _KV_STRING:
                    self._string_arrays[key] = (pos, count)
                    for _ in range(count):
                        pos += 8 + int.from_bytes(buf[pos:pos + 8], "little")
                else:
                    dt = np.dtype(_KV_SCALAR[et])
                    self.meta[key] = np.frombuffer(buf, dt, count, pos)
                    pos += count * dt.itemsize
            else:
                fmt = _KV_SCALAR[vt]
                (self.meta[key],) = struct.unpack_from(fmt, buf, pos)
                pos += struct.calcsize(fmt)
        self.arch = str(self.meta.get("general.architecture", ""))
        infos = []
        for _ in range(n_tensors):
            name, pos = string(pos)
            (nd,) = struct.unpack_from("<I", buf, pos)
            dims = struct.unpack_from(f"<{nd}Q", buf, pos + 4)
            ttype, off = struct.unpack_from("<IQ", buf, pos + 4 + 8 * nd)
            pos += 16 + 8 * nd
            infos.append((name, dims, ttype, off))
        align = int(self.meta.get("general.alignment", 32))
        base = -(-pos // align) * align
        # name -> (dims, type, offset, bytes; None: a type gguf_model does not read, "type<N>")
        self.tensor_info: dict[str, tuple[tuple[int, ...], str, int, int | None]] = {}
        for name, dims, ttype, off in infos:
            tname, per, nb = _GGML_TYPES.get(ttype, (f"type{ttype}", 1, None))
            n = math.prod(dims)
            self.tensor_info[name] = (dims, tname, base + off, None if nb is None else n // per * nb)

    def strings(self, key: str, first: int = 0) -> list[str]:
        """A string array of the metadata (e.g. ``tokenizer.ggml.tokens``), from index ``first`` on (the ones before
        are skipped, not decoded)."""
        pos, count = self._string_arrays[key]
        buf = memoryview(self.data)
        out = []
        for i in range(count):
            n = int.from_bytes(buf[pos:pos + 8], "little")
            if i >= first:
                out.append(bytes(buf[pos + 8:pos + 8 + n]).decode("utf-8", errors="replace"))
            pos += 8 + n
        return out

    def string_count(self, key: str) -> int:
        """The length of a string array of the metadata."""
        return self._string_arrays[key][1]

    def strings_json(self, key: str, ids: np.ndarray | None = None) -> bytes:
        """The JSON text of a string array of the metadata (C++, no Python string per element): the array, or with
        ``ids`` the object {string: id} of those elements."""
        from . import _native

        pos, count = self._string_arrays[key]
        return _native.gguf_strings_json(self.data, pos, count, None if ids is None else np.asarray(ids, np.int64))

    def arch_meta(self, key: str, default: object = None) -> object:
        """Metadata value ``<architecture>.<key>``."""
        return self.meta.get(f"{self.arch}.{key}", default)

    def __contains__(self, name: str) -> bool:
        return name in self.tensor_info

    def tensor(self, name: str) -> QTensor:
        """The tensor as (rows, cols) with the ggml blocks along the columns (a vector: one row)."""
        dims, qtype, off, nbytes = self.tensor_info[name]
        if nbytes is None:
            raise NotImplementedError(f"{name}: GGUF tensor type {qtype} is not supported")
        cols = dims[0]
        rows = math.prod(dims[1:]) if len(dims) > 1 else 1
        raw = self.data[off:off + nbytes]
        if qtype in ("F32", "F16", "BF16"):
            raw = raw.view({"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[qtype]).reshape(rows, cols)
        else:
            raw = raw.reshape(rows, -1)
        return QTensor(name=name, qtype=qtype, rows=rows, cols=cols, raw=raw)

    def rows(self, name: str, idx: list[int] | np.ndarray, cols: int = 0) -> torch.Tensor:
        """Rows ``idx`` of a tensor dequantized (len(idx), cols) fp32, of its first ``cols`` columns (default all; a
        multiple of the format's block): an embedding lookup in the mapped file. In C++ (``_native.dequant_rows``: a
        decode step's lookup in a few us) the types of ``_native.dequant_supported`` (the plain ones, Q4_0, Q4_1, Q8_0,
        Q3_K .. Q6_K), the others through ``QTensor.dequantize``."""
        t = self.tensor(name)
        cols, idx = cols or t.cols, np.asarray(idx)
        per, nb = _BLOCKS[t.qtype]
        width = cols // per * nb if per > 1 else cols
        if len(idx) > 1:
            self._prefetch(self.tensor_info[name][2], t.raw.strides[0], idx, width * t.raw.itemsize)
        raw = np.ascontiguousarray(t.raw[idx, :width])
        if _native.dequant_supported(t.qtype):
            out = np.empty((len(idx), cols), np.float32)
            _native.dequant_rows(t.qtype, raw.view(np.uint8), out)
            return torch.from_numpy(out)
        return QTensor(name=name, qtype=t.qtype, rows=len(idx), cols=cols, raw=raw).dequantize()

    def _prefetch(self, off: int, stride: int, idx: np.ndarray, nbytes: int) -> None:
        """Ask for the pages of rows ``idx`` (their first ``nbytes``; rows ``stride`` bytes apart from file offset
        ``off``) at once. A table outside the NPU image (Gemma's per-layer rows, an untied ``token_embd``) is in the
        page cache only once read, and after a reboot or under memory pressure a lookup faulted its rows in one at a
        time: Gemma's per-layer rows of 1024 tokens 152 ms from disk, 10-17 ms with their reads issued together (0.7
        ms when cached)."""
        if not hasattr(mmap.mmap, "madvise"):
            return
        page = mmap.PAGESIZE
        for r in np.unique(idx):
            a = off + int(r) * stride
            self.data._mmap.madvise(mmap.MADV_WILLNEED, a // page * page, a % page + nbytes)

    def release(self, name: str | None = None) -> None:
        """Give the mapping's resident pages back (after a loader copied what it needs): they
        leave this process, stay in the system's page cache while it has room, and come back on
        the next access (Windows: nothing, the mapping has no such hint). ``name``: only the whole pages inside that
        tensor's data."""
        if not hasattr(mmap.mmap, "madvise"):
            return
        if name is None:
            self.data._mmap.madvise(mmap.MADV_DONTNEED)
            return
        _, _, off, nbytes = self.tensor_info[name]
        lo, hi = -(-off // mmap.PAGESIZE) * mmap.PAGESIZE, (off + nbytes) // mmap.PAGESIZE * mmap.PAGESIZE
        if hi > lo:
            self.data._mmap.madvise(mmap.MADV_DONTNEED, lo, hi - lo)


def bpe_tokenizer_json(gguf: GGUFFile, pattern: str, *, ignore_merges: bool = False) -> bytes:
    """A Hugging Face tokenizer.json of a GGUF byte-level BPE vocabulary (``tokenizer.ggml.model == "gpt2"``), for
    ``gigatoken.Tokenizer.from_json``: the normal tokens (type 1) and the merges; the control and user-defined
    tokens (types 3, 4) as added tokens, recognized in a text (llama.cpp's ``parse_special``); ``pattern`` the split
    regex in its tokenizer.json form (one of the schemes gigatoken implements); ``ignore_merges``: a pre-token that
    is a token is not merged (Llama 3's tokenizer.json, llama.cpp's llama3 vocabulary)."""
    assert gguf.meta.get("tokenizer.ggml.model") == "gpt2", gguf.meta.get("tokenizer.ggml.model")
    types = np.asarray(gguf.meta["tokenizer.ggml.token_type"])
    key = "tokenizer.ggml.tokens"
    added = json.loads(gguf.strings_json(key, np.flatnonzero((types == 3) | (types == 4))))  # {content: id}
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    spec = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": bool(types[i] == 3)} for t, i in added.items()],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False}, byte_level]},
        "post_processor": None,
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": ignore_merges,
                  "vocab": "\0vocab", "merges": "\0merges"},
    }
    text = json.dumps(spec).encode()
    return (text.replace(b'"\\u0000vocab"', gguf.strings_json(key, np.flatnonzero(types == 1)))
            .replace(b'"\\u0000merges"', gguf.strings_json("tokenizer.ggml.merges")))
