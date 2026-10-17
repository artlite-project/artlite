"""PNG files of RGB images, written without an imaging library: 8-bit truecolor, every row unfiltered, one
zlib-compressed IDAT chunk."""

import struct
import zlib

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunk(tag: bytes, data: bytes) -> bytes:
    """A PNG chunk: length, tag, data, CRC-32 of tag and data."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_png(rgb: np.ndarray) -> bytes:
    """The PNG file of an image (H, W, 3) uint8."""
    h, w, c = rgb.shape
    if c != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"an (H, W, 3) uint8 image expected, not {rgb.shape} {rgb.dtype}")
    rows = np.concatenate([np.zeros((h, 1), np.uint8), np.ascontiguousarray(rgb).reshape(h, w * 3)], axis=1)
    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8 bits per sample, truecolor, deflate, no filter, no interlace
    return SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + chunk(b"IEND", b"")


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an image (H, W, 3) uint8 to a PNG file."""
    data = encode_png(rgb)
    with open(path, "wb") as f:
        f.write(data)
