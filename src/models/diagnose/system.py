"""What the machine is and how it is set up: the readings of ``artlite-probe --diagnose`` that need no measurement.

``root_readings`` runs first, as root (DMI memory devices, the NPU's PCI capabilities, the data fabric's DRAM map,
debugfs, the kernel log, other processes on the NPU); ``drop_privileges`` then hands the process to the user who ran
sudo; ``identity`` and ``code_state`` follow as that user."""

from __future__ import annotations

import glob
import os
import platform
import re
import struct
import subprocess
from importlib import metadata


def _read(path: str, default: str = "") -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def _run(*cmd: str, timeout: float = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


# ---------------------------------------------------------------------------
# Root readings
# ---------------------------------------------------------------------------

def npu_pci() -> str | None:
    """The NPU's PCI address (AMD 1022:1502 Phoenix / 17f0 Strix, Krackan, Halo)."""
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        if _read(f"{dev}/vendor") == "0x1022" and _read(f"{dev}/device") in ("0x1502", "0x17f0"):
            return os.path.basename(dev)
    return None


_DDR = {0x1A: "DDR4", 0x1E: "LPDDR4", 0x22: "DDR5", 0x23: "LPDDR5"}


def dmi_memory() -> list[dict[str, object]]:
    """SMBIOS type 17 (memory devices) from the DMI table (root): size, type, speed, configured speed, rank, width,
    maker, part, locator."""
    try:
        with open("/sys/firmware/dmi/tables/DMI", "rb") as f:
            data = f.read()
    except OSError:
        return []
    out, i = [], 0
    while i + 4 <= len(data):
        kind, length = data[i], data[i + 1]
        if length < 4:
            break
        end = data.find(b"\0\0", i + length)
        strings = data[i + length:end].split(b"\0") if end >= 0 else []
        rec = data[i:i + length]

        def s(off: int, rec: bytes = rec, strings: list[bytes] = strings) -> str:
            k = rec[off] if off < len(rec) else 0
            return strings[k - 1].decode(errors="replace").strip() if 0 < k <= len(strings) else ""

        if kind == 17 and length >= 0x1B:
            size = struct.unpack_from("<H", rec, 0x0C)[0]
            if size == 0x7FFF and length >= 0x20:  # (the extended size, MB)
                mb = struct.unpack_from("<I", rec, 0x1C)[0]
            else:  # (bit 15: KB units)
                mb = (size & 0x7FFF) / 1024 if size & 0x8000 else size
            if mb:
                total, width = struct.unpack_from("<HH", rec, 0x08)
                speed = struct.unpack_from("<H", rec, 0x15)[0]
                conf = struct.unpack_from("<H", rec, 0x20)[0] if length >= 0x22 else 0
                if length >= 0x58 and speed == 0xFFFF:
                    speed = struct.unpack_from("<I", rec, 0x54)[0]
                out.append({"locator": s(0x10), "bank": s(0x11), "type": _DDR.get(rec[0x12], hex(rec[0x12])),
                            "size_mb": int(mb), "speed_mts": speed, "configured_mts": conf,
                            "rank": rec[0x1B] & 0xF if length > 0x1B else 0, "data_width": width,
                            "total_width": total, "maker": s(0x17), "part": s(0x1A)})
        if kind == 127:
            break
        i = end + 2 if end >= 0 else len(data)
    return out


def pci_caps(bdf: str) -> dict[str, object]:
    """The NPU's PCI revision and extended capabilities (root reads all 4 KB of its config space): PASID, ATS, PRI."""
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as f:
            cfg = f.read()
    except OSError:
        return {}
    out: dict[str, object] = {"revision": cfg[8] if len(cfg) > 8 else None, "config_bytes": len(cfg)}
    names = {0x0F: "ATS", 0x13: "PRI", 0x1B: "PASID", 0x0D: "ACS", 0x18: "LTR", 0x1E: "L1SS"}
    found, off = [], 0x100
    while len(cfg) >= 0x104 and off and off + 4 <= len(cfg):
        hdr = struct.unpack_from("<I", cfg, off)[0]
        cap = hdr & 0xFFFF
        if cap in names:
            found.append(names[cap])
            if cap == 0x1B:  # (PASID: capability, control)
                c, ctl = struct.unpack_from("<HH", cfg, off + 4)
                out["pasid"] = {"width": (c >> 8) & 0x1F, "enabled": bool(ctl & 1)}
            if cap == 0x0F:
                out["ats_enabled"] = bool(struct.unpack_from("<H", cfg, off + 6)[0] & 0x8000)
        off = hdr >> 20
    out["extended"] = found
    return out


def _lockdown() -> str:
    m = re.search(r"\[(\w+)\]", _read("/sys/kernel/security/lockdown"))
    return m[1] if m else "unknown"


def df_dram_map() -> dict[str, object]:
    """The data fabric's DRAM address maps as drivers/ras/amd/atl reads them (D18F4: FICAA 0x8C, FICAD 0xB8; one CS
    instance): base, limit, control (hash bits), interleave (channels, first bit), the DRAM hole. Root, and a kernel
    not locked down (an indirect read writes the FICAA index register)."""
    if os.geteuid() != 0:
        return {"skipped": "not root"}
    if (lock := _lockdown()) not in ("none", "unknown"):
        return {"skipped": f"kernel lockdown {lock} (PCI config writes refused)"}
    f0, f4 = "/sys/bus/pci/devices/0000:00:18.0/config", "/sys/bus/pci/devices/0000:00:18.4/config"
    try:
        with open(f0, "rb") as f:
            f.seek(0x40)
            rev = struct.unpack("<I", f.read(4))[0]
        fd = os.open(f4, os.O_RDWR)
    except OSError as e:
        return {"skipped": f"data fabric config: {e}"}
    major, minor = (rev >> 24) & 0xF, (rev >> 16) & 0xFF
    if major != 4:
        os.close(fd)
        return {"df_revision": f"{major}.{minor}", "skipped": "not DF4 (atl's DF4 / DF4.5 layout)"}
    df45 = minor >= 5

    def read(func: int, reg: int, inst: int | None) -> int:  # (inst None: broadcast)
        ficaa = ((reg >> 2) << 1) | (func << 11) | (0 if inst is None else 1 | (inst << 16))
        os.pwrite(fd, struct.pack("<I", ficaa), 0x8C)
        return struct.unpack("<I", os.pread(fd, 4, 0xB8))[0]

    out: dict[str, object] = {"df_revision": f"{major}.{minor}", "layout": "DF4.5" if df45 else "DF4"}
    try:
        maps = []
        base_reg = 0x200 if df45 else 0xE00
        for m in range(4):
            base, limit, ctl, intlv = (read(7, base_reg + 16 * m + 4 * k, 0) for k in range(4))
            if not ctl & 1:
                continue
            hashes = [name for bit, name in ((7, "4K"), (8, "64K"), (9, "2M"), (10, "1G"), (15, "1T"))
                      if ctl & (1 << bit) and (bit not in (7, 15) or df45)]
            maps.append({"base": hex((base & 0xFFFFFFF) << 28), "limit": hex(((limit & 0xFFFFFFF) << 28) | 0xFFFFFFF),
                         "ctl": hex(ctl), "intlv": hex(intlv), "hash": hashes,
                         "first_bit": 8 + (intlv & 7), "mode": (intlv >> 4) & (0x3F if df45 else 0x1F),
                         "dies_log2": (intlv >> 12) & 3, "sockets": 2 if intlv & (1 << 18) else 1})
        out["maps"] = maps
        out["dram_hole_base"] = hex(read(7, 0x104, None) & 0xFF000000)
    except OSError as e:
        out["error"] = str(e)
    finally:
        os.close(fd)
    return out


def debugfs() -> dict[str, str]:
    """amdgpu's pm info (the SMU's enabled features, clocks) and amd_pmf's power limits (root)."""
    out = {}
    for path in sorted(glob.glob("/sys/kernel/debug/dri/*/amdgpu_pm_info"))[:1]:
        out["amdgpu_pm_info"] = _read(path)
    for path in ("/sys/kernel/debug/amd_pmf/current_power_limits",):
        if (v := _read(path)):
            out[os.path.basename(path)] = v
    return out


def kernel_log() -> list[str]:
    """This boot's kernel lines about the NPU, the IOMMU, amd_pmf and memory errors (root)."""
    text = _run("journalctl", "-k", "-b", "-o", "short-monotonic", "--no-pager") or _run("dmesg") or ""
    pat = re.compile(r"amdxdna|iommu|AMD-Vi|amd[-_]pmf|IO_PAGE_FAULT|TDR|timeout|Data Fabric|mce", re.IGNORECASE)
    host = platform.node()
    return [line.replace(f" {host} ", " ") if host else line for line in text.splitlines() if pat.search(line)][-200:]


def npu_users() -> list[str]:
    """Other processes that hold the NPU's device node open (root sees them all)."""
    out = []
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd).startswith("/dev/accel/accel"):
                pid = fd.split("/")[2]
                if int(pid) != os.getpid():
                    out.append(f"{pid} {_read(f'/proc/{pid}/comm')}")
        except (OSError, ValueError):
            pass
    return sorted(set(out))


def root_readings() -> dict[str, object]:
    bdf = npu_pci()
    return {"root": os.geteuid() == 0, "lockdown": _lockdown(), "dmi_memory": dmi_memory(),
            "npu_pci": bdf, "npu_pci_caps": pci_caps(bdf) if bdf else {}, "df_dram_map": df_dram_map(),
            "debugfs": debugfs(), "kernel_log": kernel_log(), "npu_users": npu_users()}


def drop_privileges() -> str | None:
    """Become the user who ran sudo (SUDO_UID / SUDO_GID, their groups and HOME), with an unlimited RLIMIT_MEMLOCK
    first (the NPU's pinned buffers); returns their name, or None when not run through sudo."""
    import pwd
    import resource

    if os.geteuid() != 0 or not os.environ.get("SUDO_UID"):
        return None
    uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    user = pwd.getpwuid(uid)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    os.initgroups(user.pw_name, gid)
    os.setgid(gid)
    os.setuid(uid)
    os.environ.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name)
    for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        os.environ.pop(key, None)
    return user.pw_name


# ---------------------------------------------------------------------------
# As the user
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return (_run("git", "-C", root, *args) or "").strip()


def code_state() -> dict[str, object]:
    """The checkout: its commit, the commits not upstream, uncommitted changes, the native build's state, the
    toolchain and the Python packages that matter."""
    from ..common import stale_install

    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") or "origin/main"
    local = _git("log", "--oneline", f"{upstream}..HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    versions = {}
    for pkg in ("mlir_aie", "llvm_aie", "numpy", "torch", "nanobind"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return {"commit": _git("rev-parse", "--short=9", "HEAD"), "describe": _git("describe", "--always", "--dirty"),
            "upstream": upstream, "upstream_commit": _git("rev-parse", "--short=9", upstream),
            "local_commits": local.splitlines(), "changed_files": status.splitlines(),
            "diff_stat": _git("diff", "--shortstat", "HEAD"), "stale_install": stale_install() or "",
            "python": platform.python_version(), "packages": versions}


def _cpus() -> dict[str, object]:
    """Per CPU max clock (Zen 5 vs Zen 5c cores tell apart by it), governor / EPP, amd_pstate."""
    groups: dict[str, list[int]] = {}
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"), key=lambda p: int(p.rsplit("cpu", 1)[1])):
        n = int(d.rsplit("cpu", 1)[1])
        key = " ".join(x for x in (_read(f"{d}/cpufreq/cpuinfo_max_freq"), _read(f"{d}/cpufreq/scaling_governor"),
                                   _read(f"{d}/cpufreq/energy_performance_preference")) if x)
        groups.setdefault(key, []).append(n)
    info = _read("/proc/cpuinfo")

    def field(name: str) -> str:
        m = re.search(rf"^{name}\s*:\s*(.*)$", info, re.MULTILINE)
        return m[1] if m else ""

    return {"model": field("model name"), "family": field("cpu family"), "model_id": field("model"),
            "stepping": field("stepping"), "microcode": field("microcode"), "count": os.cpu_count(),
            "amd_pstate": _read("/sys/devices/system/cpu/amd_pstate/status"),
            "boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
            "groups (max kHz, governor, EPP: CPUs)": {k: v for k, v in groups.items()}}


def _meminfo() -> dict[str, int]:
    out = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "AnonHugePages", "ShmemHugePages", "Cached",
                 "Mlocked"):
            out[k] = int(v.split()[0])
    return out


def _free_2mb_blocks() -> int:
    """Free blocks of 2 MB and more (buddyinfo orders 9 and 10, in 2 MB units)."""
    n = 0
    for line in _read("/proc/buddyinfo").splitlines():
        counts = [int(x) for x in line.split()[4:]]
        n += sum(c << (o - 9) for o, c in enumerate(counts) if o >= 9)
    return n


def _ip_versions() -> dict[str, str]:
    """The SoC's IP blocks as amdgpu's discovery table names them (MP1: the SMU; UMC; DF; GC: the iGPU)."""
    out = {}
    for d in sorted(glob.glob("/sys/class/drm/card*/device/ip_discovery/die/0/*/0"))[:64]:
        name = d.split("/")[-2]
        if name in ("MP1", "UMC", "DF", "GC", "NBIO", "SMUIO", "MP0", "VCN"):
            out[name] = ".".join(_read(f"{d}/{k}") for k in ("major", "minor", "revision"))
    return out


def identity() -> dict[str, object]:
    """The machine, its firmware, the kernel and its settings, the NPU, power."""
    dmi = {k: _read(f"/sys/class/dmi/id/{k}") for k in ("sys_vendor", "product_name", "product_version",
                                                         "board_vendor", "board_name", "bios_vendor",
                                                         "bios_version", "bios_date")}
    thp = {k: _read(f"/sys/kernel/mm/transparent_hugepage/{k}") for k in ("enabled", "defrag", "shmem_enabled")}
    thp["khugepaged/defrag"] = _read("/sys/kernel/mm/transparent_hugepage/khugepaged/defrag")
    bdf = npu_pci()
    npu: dict[str, object] = {}
    if bdf:
        d = f"/sys/bus/pci/devices/{bdf}"
        npu = {"pci": bdf, "revision": _read(f"{d}/revision"), "vbnv": _read(f"{d}/vbnv"),
               "fw_version": _read(f"{d}/fw_version"),
               "iommu_group_type": _read(f"{d}/iommu_group/type"),
               "module": _read("/sys/module/amdxdna/version") or "Linux's own",
               "module_srcversion": _read("/sys/module/amdxdna/srcversion"),
               "module_params": {os.path.basename(p): _read(p)
                                 for p in glob.glob("/sys/module/amdxdna/parameters/*")}}
        npu["runtime_pm"] = {k: _read(f"{d}/power/{k}") for k in ("control", "runtime_status",
                                                                    "autosuspend_delay_ms")}
    cpuidle = {"governor": _read("/sys/devices/system/cpu/cpuidle/current_governor")
               or _read("/sys/devices/system/cpu/cpuidle/current_governor_ro"),
               "states (name: disabled)": {_read(f"{st}/name"): _read(f"{st}/disable")
                                           for st in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cpuidle/state*"))}}
    irqs = [line.split(":")[0].strip() + ": " + " ".join(line.split()[-2:])
            for line in _read("/proc/interrupts").splitlines() if "amdxdna" in line or "xdna" in line]
    iommu = {os.path.basename(os.path.dirname(os.path.dirname(p))): _read(p)
             for p in glob.glob("/sys/class/iommu/*/amd-iommu/features")}
    supplies = {os.path.basename(p): {k: _read(f"{p}/{k}") for k in ("type", "online", "status", "capacity")}
                for p in glob.glob("/sys/class/power_supply/*")}
    return {"dmi": dmi, "kernel": platform.release(), "kernel_version": platform.version(),
            "cmdline": _read("/proc/cmdline"), "cpus": _cpus(), "memory": _meminfo(),
            "free_2mb_blocks": _free_2mb_blocks(), "thp": thp, "ip_versions": _ip_versions(), "npu": npu,
            "cpuidle": cpuidle, "npu_interrupts": irqs,
            "iommu_features": iommu,
            "platform_profile": {os.path.basename(os.path.dirname(p)): _read(p)
                                 for p in glob.glob("/sys/class/platform-profile/*/profile")}
            | {"legacy": _read("/sys/firmware/acpi/platform_profile")},
            "power_supplies": supplies,
            "amdgpu_perf_level": _read(next(iter(glob.glob("/sys/class/drm/card*/device/"
                                                           "power_dpm_force_performance_level")), ""))}


QUERY_SENSORS, QUERY_FIRMWARE_VERSION = 4, 8  # (DRM_AMDXDNA_QUERY_*)
_SENSOR = struct.Struct("<64s4I64s16sbB6x")  # (struct amdxdna_drm_query_sensor: 168 bytes)


def npu_info(dev: object) -> dict[str, object]:
    """What the driver reports about the NPU: its version, the firmware's, the power mode, the clocks it reports (Linux
    7.0's: its DPM table, not a measurement), its sensors (power, column utilization: Linux 7.1+ and Ubuntu's 7.0),
    the device heap."""
    out: dict[str, object] = {}
    for key, fn in (("driver", dev.driver_version), ("power_mode", dev.power_mode), ("clocks_mhz", dev.clocks)):
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001  (a query the driver lacks: recorded, not fatal)
            out[key] = f"{type(e).__name__}: {e}"
    try:
        out["firmware"] = ".".join(str(v) for v in struct.unpack_from("<4I", dev.get_info(QUERY_FIRMWARE_VERSION,
                                                                                        bytes(64))))
        raw = dev.get_info(QUERY_SENSORS, bytes(_SENSOR.size * 16))
        sensors = []
        for i in range(len(raw) // _SENSOR.size):
            label, inp, _max, _avg, _hi, status, units, unitm, _kind = _SENSOR.unpack_from(raw, i * _SENSOR.size)
            sensors.append({"label": label.split(b"\0")[0].decode(errors="replace"), "input": inp,
                            "units": units.split(b"\0")[0].decode(errors="replace"), "unitm": unitm,
                            "status": status.split(b"\0")[0].decode(errors="replace")})
        out["sensors"] = sensors
    except Exception as e:  # noqa: BLE001
        out["get_info"] = f"{type(e).__name__}: {e}"
    out["heap"] = "grows" if dev.grow_heap() else "one 64 MB heap BO per process" if dev.heap_fixed else "at its maximum"
    return out


def npu_sample(dev: object) -> dict[str, float]:
    """One reading of the driver's sensors and clocks for the telemetry sampler: power (mW), the busiest column's
    utilization (%), the MP-NPU and H clocks (MHz; live on Linux 7.2+ and xdna-driver)."""
    out: dict[str, float] = {}
    try:
        mp, h = dev.clocks()
        out["npu_mp_clk_mhz"], out["npu_h_clk_mhz"] = float(mp), float(h)
        raw = dev.get_info(QUERY_SENSORS, bytes(_SENSOR.size * 16))
    except Exception:  # noqa: BLE001  (a driver without the queries: no such fields)
        return out
    util = []
    for i in range(len(raw) // _SENSOR.size):
        label, inp, *_rest, unitm, _kind = _SENSOR.unpack_from(raw, i * _SENSOR.size)
        name = label.split(b"\0")[0].decode(errors="replace")
        if name == "Total Power":
            out["npu_sensor_mw"] = inp * 10.0 ** (unitm + 3)
        elif name.endswith("Utilization"):
            util.append(float(inp))
    if util:
        out["npu_column_util_max"] = max(util)
    return out
