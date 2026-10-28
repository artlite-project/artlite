"""Strix Halo's numbers from the same diagnosis: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 8 x
16 GB LPDDR5 8000 MT/s, Linux 7.2.13-gentoo-dist-bin, amdxdna_accel_driver 0.18.0, firmware 1.1.2.65,
platform profile performance, plugged in (artlite-probe --diagnose)."""

HALO: dict[str, dict] = {
    "dma": {
        "2 MB": {
            "DDR read, 1 channel (GB/s)": 13.9,
            "DDR read, 1 shim: 2 channels": 27.1,
            "DDR read, 8 shims: 16 channels": 56.8,
            "DDR read, 8 shims: 1 channel each": 56.4,
            "DDR write, 16 channels": 57.6,
            "DDR read 8 + write 8 channels": 76.1,
            "gather 256 B rows 64 KB apart, 16 channels": 15.7,
            "gather 512 B rows 64 KB apart": 30.9,
            "gather 1 KB rows 64 KB apart": 56.5,
            "gather 256 B rows 64 KB + 256 B apart": 56.9,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 57.3,
            "write 132 B rows 64 KB + 132 B apart": 41.4,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.8,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.6,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 76.2,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 57.7,
            "write 132 B rows, regions at other 2 MB-page offsets": 41.4,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 57.0
        },
        "4 KB": {
            "DDR read, 1 channel (GB/s)": 13.8,
            "DDR read, 1 shim: 2 channels": 26.8,
            "DDR read, 8 shims: 16 channels": 56.8,
            "DDR read, 8 shims: 1 channel each": 56.5,
            "DDR write, 16 channels": 57.1,
            "DDR read 8 + write 8 channels": 75.9,
            "gather 256 B rows 64 KB apart, 16 channels": 15.6,
            "gather 512 B rows 64 KB apart": 30.8,
            "gather 1 KB rows 64 KB apart": 56.7,
            "gather 256 B rows 64 KB + 256 B apart": 29.1,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 12.2,
            "write 132 B rows 64 KB + 132 B apart": 12.3,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.8,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.5,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 76.0,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 34.1,
            "write 132 B rows, regions at other 2 MB-page offsets": 13.0,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 57.0
        }
    },
    "control": {
        "noop (ns per op)": 42.0,
        "noop (ns per byte)": 10.5,
        "write32 (ns per op)": 62.6,
        "write32 (ns per byte)": 2.6,
        "blockwrite 1 KB (ns per op)": 1120,
        "blockwrite 1 KB (ns per byte)": 1.1,
        "64 B DMA task + wait (ns per op)": 2082,
        "64 B DMA tasks of 8 columns + one wait (ns per op)": 596,
        "NOP dispatch, blocking (us)": 47.2,
        "NOP dispatches, 1 queued (per s)": 55651,
        "NOP dispatches, 3 queued (per s)": 57339,
        "array clock (MHz)": 1818,
        "dispatch round trip (us)": 55.4,
        "array clock, 32 cores of MMULs (MHz)": 1814,
        "array clock, 32 cores of MMULs, a busy CPU (MHz)": 1808
    },
    "pp1024": {
        "V0: 2 MB pages, whole lines": 8802,
        "V1: 4 KB pages, whole lines": 8743,
        "V2: 2 MB pages, partial lines": 8835,
        "V3: 4 KB pages, partial lines": 8718,
        "V4: V0 with a busy CPU": 8871
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
        "gemm qkv Q4_K 2048x3584": 9.0,
        "gemm qkv Q6_K 2048x3584": 2.5
    },
    "design_switch_us": 75.6,
    "duty_item_ms": {
        "100%": 2.86,
        "75%": 2.88,
        "50%": 2.93,
        "25%": 3.35
    },
    "xrt_validate": {
        "exit": 0,
        "gemm_tops": "51.3",
        "latency_us": "58.0",
        "throughput_ops": "94655.0",
        "power_mode": "default"
    }
}
