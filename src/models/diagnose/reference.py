"""Strix Halo's numbers from the same diagnosis: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 8 x
16 GB LPDDR5 8000 MT/s, Linux 7.2.13-gentoo-dist-bin, amdxdna_accel_driver 0.18.0, firmware 1.1.2.65,
platform profile performance, plugged in (artlite-probe --diagnose)."""

HALO: dict[str, dict] = {
    "dma": {
        "2 MB": {
            "DDR read, 1 channel (GB/s)": 13.9,
            "DDR read, 1 shim: 2 channels": 26.9,
            "DDR read, 8 shims: 16 channels": 56.8,
            "DDR read, 8 shims: 1 channel each": 56.4,
            "DDR write, 16 channels": 57.6,
            "DDR read 8 + write 8 channels": 76.1,
            "gather 256 B rows 64 KB apart, 16 channels": 15.5,
            "gather 512 B rows 64 KB apart": 30.9,
            "gather 1 KB rows 64 KB apart": 56.8,
            "gather 256 B rows 64 KB + 256 B apart": 56.9,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 57.3,
            "write 132 B rows 64 KB + 132 B apart": 41.3,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.8,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.6,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 76.1,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 57.6,
            "write 132 B rows, regions at other 2 MB-page offsets": 41.3,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 57.0
        },
        "4 KB": {
            "DDR read, 1 channel (GB/s)": 13.8,
            "DDR read, 1 shim: 2 channels": 26.8,
            "DDR read, 8 shims: 16 channels": 56.8,
            "DDR read, 8 shims: 1 channel each": 56.4,
            "DDR write, 16 channels": 57.2,
            "DDR read 8 + write 8 channels": 76.0,
            "gather 256 B rows 64 KB apart, 16 channels": 15.7,
            "gather 512 B rows 64 KB apart": 30.6,
            "gather 1 KB rows 64 KB apart": 56.9,
            "gather 256 B rows 64 KB + 256 B apart": 29.2,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 12.3,
            "write 132 B rows 64 KB + 132 B apart": 12.4,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.8,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.5,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 76.1,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 34.1,
            "write 132 B rows, regions at other 2 MB-page offsets": 13.1,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 56.9
        }
    },
    "control": {
        "noop (ns per op)": 42.4,
        "noop (ns per byte)": 10.6,
        "write32 (ns per op)": 61.9,
        "write32 (ns per byte)": 2.6,
        "blockwrite 1 KB (ns per op)": 1121,
        "blockwrite 1 KB (ns per byte)": 1.1,
        "64 B DMA task + wait (ns per op)": 2070,
        "64 B DMA tasks of 8 columns + one wait (ns per op)": 598,
        "NOP dispatch, blocking (us)": 47.2,
        "NOP dispatches, 1 queued (per s)": 56878,
        "NOP dispatches, 3 queued (per s)": 57797,
        "array clock (MHz)": 1809,
        "dispatch round trip (us)": 59.3,
        "array clock, 32 cores of MMULs (MHz)": 1812,
        "array clock, 32 cores of MMULs, a busy CPU (MHz)": 1810
    },
    "pp1024": {
        "V0: 2 MB pages, whole lines": 8785,
        "V1: 4 KB pages, whole lines": 8678,
        "V2: 2 MB pages, partial lines": 8763,
        "V3: 4 KB pages, partial lines": 8722,
        "V4: V0 with a busy CPU": 8837,
        "V5: V0 at the calibrated paces": 8746
    },
    "kinds": {
        "whole program": 108,
        "AttnOperator": 10.1,
        "GemmQOperator Q4_K 2048x2304": 5.6,
        "GemmQOperator Q4_K 2048x8192": 36.0,
        "GemmQOperator Q4_K 4096x2304": 10.5,
        "GemmQdOperator Q6_K 2048x2304": 16.0,
        "PrefillNormOperator": 10.6,
        "PrepOperator": 3.5,
        "gemm qkv Q4_K 2048x3584": 8.9,
        "gemm qkv Q6_K 2048x3584": 2.5
    },
    "design_switch_us": 74.8,
    "pace_ms": {
        "0": 36.0,
        "1": 41.9,
        "2": 45.3,
        "3": 49.1,
        "4": 55.4,
        "5": 63.9,
        "6": 74.4,
        "7": 91.0
    },
    "xrt_validate": {
        "exit": 0,
        "gemm_tops": "51.3",
        "latency_us": "60.0",
        "throughput_ops": "94721.0",
        "power_mode": "default"
    }
}
