# artlite-probe

Measures the NPU's own numbers, next to those of the Strix Halo that Artlite is measured on: the array clock, the
dispatch round trip and the DRAM read and write rates of its DMA. It takes about a minute and needs no model file.
Model speeds alone cannot tell a slower clock from a narrower memory path, so a report of another machine's speeds
([Share your results](https://github.com/artlite-project/artlite/issues/5)) starts with this table.

```console
$ artlite-probe
artlite-probe
machine: ASUSTeK COMPUTER INC. ROG Flow Z13 GZ302EA_GZ302EA 1.0; AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 CPUs; 121.5 GB of memory
OS: Linux 7.2.9-gentoo-dist-bin; amdxdna_accel_driver 0.18.0 (xdna-driver), firmware 1.1.2.65
power: platform profile performance (performance), plugged in
NPU: power mode default; interrupts on CPU 0 (max 5187 MHz); device heap: grows

| | this machine | Strix Halo |
|---|---:|---:|
| array clock, measured (MHz) | 1810 | 1810 |
| dispatch round trip (us) | 49 | 49 |
| DDR read, 1 channel (GB/s) | 13.9 | 13.9 |
| DDR read, 1 shim: 2 channels | 26.9 | 26.9 |
| DDR read, 8 shims: 16 channels | 56.8 | 56.8 |
| DDR write, 16 channels | 57.6 | 57.6 |
| DDR read 8 + write 8 channels | 76.1 | 76.1 |
| gather 256 B rows 64 KB apart, 16 channels | 15.6 | 15.6 |
| gather 512 B rows 64 KB apart | 30.8 | 30.8 |
| gather 1 KB rows 64 KB apart | 56.6 | 56.6 |
| gather 256 B rows 64 KB + 256 B apart | 56.9 | 56.9 |
| 16-channel read per array cycle (B) | 31.4 | 31.4 |

build: 391fc8d2d (109)
```

| row | what it measures |
|---|---|
| array clock | one core spins until its timer has advanced by a given count of cycles; two counts timed on the host, so the dispatch cancels out |
| dispatch round trip | a dispatch of almost no work, from submit to its completion seen on the host |
| DDR read / write | shim DMA channels streaming DRAM to the memory tiles (or back) on a design with no cores: one channel, one column's two, all 16; a read and a write stream in every column at once |
| gather | 16 channels reading rows of 256 B to 1 KB, 64 KB apart: rows of 256 B or less at a stride of a multiple of 4 KB fall on one DRAM channel, a skewed stride spreads them |
| per array cycle | the 16-channel read rate over the measured clock: Strix Halo's NPU moves at most 32 B per array cycle to DRAM |

The first lines name the machine, the kernel, which `amdxdna` runs (`amdxdna_accel_driver` 0.6 / 0.8 / 0.10 are Linux
7.0 / 7.1 / 7.2's own), the NPU firmware, the platform profile and the power adapter, the NPU's power mode, the CPU
that takes the NPU's interrupts, and whether the device heap can grow (Linux before 7.2: one 64 MB heap per process).

| option | |
|---|---|
| `--seconds S` | seconds of dispatches per measurement (default 1) |
