# artlite-probe

Measures the NPU's own numbers, next to those of the Strix Halo that Artlite is measured on and of a Krackan Point and
a Strix Point laptop: the array clock, the dispatch round trip and the DRAM read and write rates of its DMA. It takes
about a minute and needs no model file.
Model speeds alone cannot tell a slower clock from a narrower memory path, so a report of another machine's speeds
([Share your results](https://github.com/artlite-project/artlite/issues/5)) starts with this table.

```console
$ artlite-probe
artlite-probe
machine: ASUSTeK COMPUTER INC. ROG Flow Z13 GZ302EA_GZ302EA 1.0; AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 CPUs; 121.5 GB of memory
OS: Linux 7.2.9-gentoo-dist-bin; amdxdna_accel_driver 0.18.0 (xdna-driver), firmware 1.1.2.65
power: platform profile performance (performance), plugged in
NPU: power mode default; interrupts on CPU 0 (max 5187 MHz); device heap: grows; host buffers: 2 MB pages

| | this machine | Strix Halo | Krackan Point | Strix Point |
|---|---:|---:|---:|---:|
| array clock, measured (MHz) | 1810 | 1810 | 1808 | 1809 |
| array clock, 32 cores of MMULs (MHz) | 1810 | 1809 | 1810 | 1808 |
| dispatch round trip (us) | 50 | 49 | 56 | 94 |
| DDR read, 1 channel (GB/s) | 13.9 | 13.9 | 14.2 | 11.5 |
| DDR read, 1 shim: 2 channels | 27.1 | 26.9 | 28.1 | 21.4 |
| DDR read, 8 shims: 16 channels | 56.8 | 56.8 | 97.8 | 53.2 |
| DDR read, 8 shims: 1 channel each | 56.5 | 56.4 | 80.1 | 47.3 |
| DDR write, 16 channels | 57.6 | 57.6 | 87.2 | 39.9 |
| DDR read 8 + write 8 channels | 76.3 | 76.1 | 88.2 | 53.4 |
| gather 256 B rows 64 KB apart, 16 channels | 15.8 | 15.6 | 28.6 | 12.5 |
| gather 512 B rows 64 KB apart | 31.0 | 30.8 | 55.9 | 25.1 |
| gather 1 KB rows 64 KB apart | 56.9 | 56.6 | 95.6 | 41.9 |
| gather 256 B rows 64 KB + 256 B apart | 57.1 | 56.9 | 89.6 | 14.0 |
| write 128 B rows 64 KB + 128 B apart, 16 channels | 57.3 | 57.2 | 73.2 | ? |
| write 132 B rows 64 KB + 132 B apart | 41.5 | 41.4 | 15.6 | ? |
| 16-channel read per array cycle (B) | 31.3 | 31.4 | 54.1 | 29.4 |

build: ef8477758 (140)
```

| row | what it measures |
|---|---|
| array clock | one core spins until its timer has advanced by a given count of cycles; two counts timed on the host, so the dispatch cancels out |
| array clock, 32 cores of MMULs | the same while every core of the array runs MMULs on random data (more NPU power than a Llama 3.2 1B prefill draws): a power or thermal limit shows here, not with one core spinning |
| dispatch round trip | a dispatch of almost no work, from submit to its completion seen on the host |
| DDR read / write | shim DMA channels streaming DRAM to the memory tiles (or back) on a design with no cores: one channel, one column's two, all 16, one per column (as the decode streams its weights); a read and a write stream in every column at once |
| gather | 16 channels reading rows of 256 B to 1 KB, 64 KB apart: rows of 256 B or less at a stride of a multiple of 4 KB fall on one DRAM channel, a skewed stride spreads them |
| write rows | 16 channels writing short rows, each next to a row written later: 128 B rows cover whole 64-byte lines, 132 B rows write parts of their first and last line, as the rows of a GEMM pass of 66 output columns would (the memory merges such a write into its line) |
| per array cycle | the 16-channel read rate over the measured clock: Strix Halo's NPU moves at most 32 B per array cycle to DRAM |

The reference columns: Strix Halo is a Ryzen AI Max+ 395 (256-bit LPDDR5X-8000, 256 GB/s; AMD's xdna-driver), Krackan
Point a Ryzen AI 7 350 laptop (128-bit LPDDR5X-7500, 120 GB/s; Linux 7.0's `amdxdna`), Strix Point a Ryzen AI 9 HX
PRO 370 laptop (two DDR5-5600 SO-DIMMs, 90 GB/s; Windows 11 and XRT, NPU driver 32.0.20102.3930), all plugged in, NPU
power mode `default`. Their arrays are the same, at the same clock, also with all 32 cores busy; what differs is the
path to DRAM. Strix Halo's NPU reaches it through a port of 32 B per array cycle (its rate follows the array clock in
every power mode), 22% of that bus; Krackan Point's reads 81% of its bus, 1.7x as much; Strix Point's 59%. Small rows
at a stride of a multiple of 4 KB cost all three alike, 71-77% below a stream at 256 B and 43-53% at 512 B, so a
layout that is fast on one is fast on the others. A write of 132 B rows that share their end lines runs 28% below whole
lines on Strix Halo and 79% below on Krackan Point (15.6 instead of 73.2 GB/s): a part of a line is a masked write,
which LPDDR5 runs as a read-modify-write inside the DRAM, at 4x a write's command spacing (tCCDMW). Artlite's prefill
writes whole lines (`?`: not measured on that machine).

Each row on a page of its own costs the NPU an address translation. Artlite's host buffers have 2 MB pages on Linux
(transparent huge pages, `madvise` or `always`; the `host buffers` line says so); Windows maps system memory for the
NPU in 4 KB pages, and there the skewed gather and the write rows run at 14-16 and 5-6 GB/s (Strix Point and a Krackan
Point laptop on Windows: some 50 million rows a second at most, whatever their size), as on Strix Halo with huge
pages off (29.3 and 12.2); what that costs the prefill: [Page-local bricks](page-local-bricks.md). The dispatch round
trip through Windows and XRT is 94 us (58 in the `turbo` power mode).

The first lines name the machine, the kernel, which `amdxdna` runs (`amdxdna_accel_driver` 0.6 / 0.8 / 0.10 are Linux
7.0 / 7.1 / 7.2's own), the NPU firmware, the platform profile and the power adapter, the NPU's power mode, the CPU
that takes the NPU's interrupts, and whether the device heap can grow (Linux before 7.2: one 64 MB heap per process).
On Windows: the NPU driver, firmware and power mode as `xrt-smi` tells them.

## Pages: `--pages`

`artlite-probe --pages` prints a second table instead (a minute and a half): what the NPU's address translation costs
rows on pages of their own. On Windows it runs in the `performance` power mode (from `default`, which runs a Krackan
Point's array at 1.4 GHz; `default` again after). Millions of rows a second, read / write; unless a row says
otherwise, rows of 256 B, each on a page of its own, a channel's pages 4 KB apart (the rows 4 KB + 256 B, so that they
spread over the DRAM channels) and visited in turn. The reference columns are Strix Halo with 2 MB pages (no
translations: the DMA's own rates for each pattern) and with 4 KB pages (transparent huge pages `never`).

| rows | what they tell | Strix Halo, 4 KB pages |
|---|---|---|
| n pages per channel, 16 channels | how many pages the translation cache holds: past them, every row pays a translation | up to 32 per channel free, from 64 on 160 / 130 |
| n pages per channel, 4 channels | per channel or shared: 4 channels with 4x the pages each | shared: 64 per channel free, 256 not; 512-1024 pages in all |
| 4096 pages per channel, 1, 2, 4, 8 channels | the rate of translations, one channel to all | 47-55 for one channel, 120-160 for 4-16 together |
| 256 pages per channel 16 KB, 64 KB, 1 MB apart | page-table locality: 8 neighbouring pages share a 64-byte line of the page table | 119 / 117 at 16 KB, 113 / 87 at 1 MB |
| 128 B rows, k a page | rows sharing a page (256 B apart) not visited before: pages next to each other or 64 KB apart; or 8 pages per channel visited again | reads: 163, 370, 406 at 1, 2, 8 a page; writes: 143, 162, 251 (back-to-back rows on a page whose translation is not back yet each pay one); visited again: 433 / 434 |
| 4 KB, 16 KB rows | rows of whole pages | as many bytes as a stream |
| 2 KB rows 9280 B apart | the prefill's reads of the decode image's tiles | as many bytes as a stream |
| 8 channels read + 8 write | whether reads and writes share the rate | 162 in all |

## Diagnosis: `--diagnose`

`sudo $(which artlite-probe) --diagnose` (about 5 minutes; Linux: root) records everything that can make Artlite
slower on a machine, so that a slower number comes with what explains it. It reads as root what only root can (the
memory devices, the NPU's PCI capabilities, the data fabric's DRAM interleaving unless the kernel is locked down, the
SMU's enabled features, this boot's kernel lines about the NPU and the IOMMU, other processes on the NPU), then runs
on as the user who ran sudo, with the `PATH` and `ARTLITE_*` settings (`ARTLITE_MODELS_DIR`, ...) they ran sudo with: their
caches and models directory. Through every step it samples what the SoC reports
(amdgpu's `gpu_metrics`: fabric and memory clocks, the NPU's clock, DRAM traffic, power, limits, temperatures; the
driver's NPU sensors where it has them; the clock of the CPU that takes the NPU's interrupts; other processes' CPU
load). It measures:

- the DMA rows above on host buffers of 2 MB pages and again of 4 KB pages, rows of channel placement (regions at
  other offsets in their 2 MB pages) and of the prefill's row shapes;
- the NPU's command path: what its control processor spends per operation of control code (no-op, write, 1 KB block
  write, a small DMA task and its wait), a dispatch, NOP dispatches back to back; the array clock under load, also
  with a busy CPU;
- Llama 3.2 1B Q4_K_M's 1024-token prompt in variants, two rounds: host buffers of 2 MB or 4 KB pages, the prefill's
  output rows in whole 64-byte lines or not, and with one CPU kept busy; for the first two the chunk's time per
  operator kind and a design switch's cost; the host's embedding lookups; the huge pages the process got;
- `xrt-smi validate` where XRT is installed.

It prints a report, its verdicts first (each candidate cause with the numbers that decide it, and how many ms of the
1B chunk's difference to Strix Halo each explains), and writes every number to `artlite-diagnose-<time>.json`: attach
that file to the report.

| option | |
|---|---|
| `--seconds S` | seconds of dispatches per measurement (default 1) |
| `--pages` | the page table instead of the hardware numbers |
| `--diagnose` | the diagnosis instead (root on Linux) |
| `--model FILE` | `--diagnose`: Llama 3.2 1B Q4_K_M's file (default: the one in the models directory) |
| `--out-dir DIR` | `--diagnose`: where the JSON file goes (default: the current directory) |
