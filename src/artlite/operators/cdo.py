"""The configuration a compiled design leaves: the register and memory writes of aiecc's CDO files."""

import os
import struct


class CdoError(RuntimeError):
    pass


def writes(work_dir: str) -> set[int]:
    """The register and memory addresses a compiled design's configuration writes."""
    return set(values(work_dir))


def values(work_dir: str) -> dict[int, int]:
    """Address -> the value a compiled design's configuration leaves there (aiecc's CDO files: writes, masked writes
    and block writes, 64-bit addresses)."""
    out: dict[int, int] = {}
    prj = os.path.join(work_dir, "design.mlir.prj")
    for name in ("main_aie_cdo_init.bin", "main_aie_cdo_enable.bin", "main_aie_cdo_elfs.bin"):
        path = next((p for p in (os.path.join(prj, "cdo_main", name), os.path.join(prj, name)) if os.path.exists(p)), None)
        if path is None:
            raise CdoError(f"{work_dir}: no CDO {name}")
        with open(path, "rb") as f:
            b = f.read()
        w = struct.unpack(f"<{len(b) // 4}I", b[:len(b) // 4 * 4])
        i = 5  # (the header: its length, "CDO", the version, the length, the checksum)
        while i < len(w):
            api, n = w[i] & 0xFF, (w[i] >> 16) & 0xFF
            i += 1
            if n == 0xFF:
                n, i = w[i], i + 1
            p = w[i:i + n]
            if api in (5, 7, 8):  # (block write, masked write: mask, value, write: the address in two words)
                a = p[0] << 32 | p[1]
                if api == 7:
                    out[a] = out.get(a, 0) & ~p[2] | p[3] & p[2]
                else:
                    out.update({a + 4 * k: p[2 + k] for k in range(n - 2)} if api == 5 else {a: p[-1]})
            i += n
    return out
