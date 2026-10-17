"""The README's "How it works" diagram: docs/images/how-it-works.svg and how-it-works-dark.svg.

    python docs/diagram.py

The host's three paths into the NPU: designs built once per shape (cached), weights packed once per model into DRAM,
a transaction per step; the weights stream from DRAM to the array's columns through its one port (animated, unless
the viewer prefers reduced motion). charts.py's themes, font and text metrics.
"""

from xml.sax.saxutils import escape

from charts import FONT, IMAGES, THEMES, WIDTH, Theme, text_width

H = 456
HOST = (24, 24, 516, 292)  # x, y, w, h
NPU = (592, 24, 264, 292)
DRAM = (24, 344, 832, 88)
NODE_X, NODE_W, NODE_H, NODE_GAP = 40, 104, 44, 20
LANE_Y, LANE_STEP = 78, 84  # the first lane's header baseline, the lanes' pitch
LANES = [  # header, note, nodes: (title, subtitle)
    ("BUILD", "once per design, cached", [("operator", "Python class"), ("C++ + IRON", "rendered per shape"),
                                          ("Peano · aiecc", "LLVM for AI Engine"), ("ELF", "device image")]),
    ("RUN", "every decode step and prefill chunk", [("artlite-cli", "server · bench"), ("transaction", "built per step"),
                                                     ("runtime", "C++ · nanobind"), ("amdxdna", "kernel driver")]),
    ("LOAD", "once per model", [("GGUF file", "Q4_K · Q6_K · BF16"), ("re-layout", "lossless, same size")]),
]
# the AIE2P array: 8 columns of a shim tile, a memory tile and 4 cores; core row 0 level with the BUILD lane, the shim
# row below the RUN lane
COLS, TILE, PITCH = 8, 18, 22
ARRAY_X = 612
CORE_Y = (101, 123, 145, 167)
MEM_Y, SHIM_Y = 197, 227
PORT_X = ARRAY_X + (COLS * PITCH - (PITCH - TILE)) / 2  # under the array's middle
FAN_Y = 300  # where the port's stream splits into the columns
PANEL = {"": "#f6f8fa", "-dark": "#151b23"}


def node_x(j: int) -> float:
    return NODE_X + j * (NODE_W + NODE_GAP)


def node_cy(lane: int) -> float:
    return LANE_Y + lane * LANE_STEP + 10 + NODE_H / 2


def col_x(c: int) -> float:
    return ARRAY_X + TILE / 2 + c * PITCH


def text(x: float, y: float, s: str, cls: str, anchor: str = "start") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x:g}" y="{y:g}" class="{cls}"{a}>{escape(s)}</text>'


def header(x: float, y: float, name: str, note: str, cls: str = "h") -> list[str]:
    size = 12 if cls == "H" else 10.5
    w = text_width(name, size, bold=True) * 1.12  # (letter-spacing)
    return [text(x, y, name, cls), text(x + w + 8, y, note, "n")]


def pill(cx: float, cy: float, s: str) -> list[str]:
    w = text_width(s, 10) + 14
    return [f'<rect x="{cx - w / 2:g}" y="{cy - 8:g}" width="{w:g}" height="16" rx="8" class="pill"/>',
            text(cx, cy + 3.5, s, "p", "middle")]


def tile(x: float, y: float, cls: str) -> str:
    return f'<rect x="{x:g}" y="{y:g}" width="{TILE}" height="{TILE}" rx="4" class="{cls}"/>'


def check_fit() -> None:
    for _, _, nodes in LANES:
        for title, sub in nodes:
            for s, size, bold in ((title, 12, True), (sub, 10, False)):
                assert text_width(s, size, bold) <= NODE_W - 12, f"{s!r} is wider than its node"


def draw(t: Theme, panel: str) -> str:
    p: list[str] = []
    # panels: the host, the NPU, DRAM
    for x, y, w, h in (HOST, NPU, DRAM):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="panel"/>')
    p += header(HOST[0] + 16, 50, "HOST", "Python and C++", "H")
    p += header(NPU[0] + 16, 50, "NPU", "AIE2P array", "H")

    # the host's lanes
    for i, (name, note, nodes) in enumerate(LANES):
        y = LANE_Y + i * LANE_STEP
        p += header(NODE_X, y, name, note)
        for j, (title, sub) in enumerate(nodes):
            x = node_x(j)
            p.append(f'<rect x="{x:g}" y="{y + 10:g}" width="{NODE_W}" height="{NODE_H}" rx="7" class="node"/>')
            p.append(text(x + NODE_W / 2, y + 28, title, "t", "middle"))
            p.append(text(x + NODE_W / 2, y + 44, sub, "ns", "middle"))
            if j:
                p.append(f'<path d="M{x - NODE_GAP + 3:g},{y + 32:g}H{x - 3:g}" class="arrow"/>')

    # DRAM: the weight image the load packs, the KV cache, the bus
    wx, kx, by, bh = 40, 468, 362, 52
    p.append(f'<rect x="{wx}" y="{by}" width="412" height="{bh}" rx="7" class="weights"/>')
    p.append(f'<rect x="{wx}" y="{by}" width="412" height="{bh}" rx="7" fill="url(#tiles)"/>')
    p.append(text(wx + 16, by + 22, "weight image", "t"))
    p.append(text(wx + 16, by + 38, "the quantization's size, streamed every step", "s"))
    p.append(f'<rect x="{kx}" y="{by}" width="160" height="{bh}" rx="7" class="node"/>')
    p.append(text(kx + 16, by + 22, "KV cache", "t"))
    p.append(text(kx + 16, by + 38, "read and written", "s"))
    rx = DRAM[0] + DRAM[2] - 16
    p.append(text(rx, by + 12, "DRAM", "H", "end"))
    p.append(text(rx, by + 30, "LPDDR5X · 256 GB/s", "s", "end"))
    p.append(text(rx, by + 45, "shared with CPU, iGPU", "s", "end"))

    # the weights' path: the weight image and the KV cache -> a bus along DRAM -> the port -> 8 shim DMAs -> memory
    # tiles -> cores (under the tiles)
    bus, wsrc, ksrc = DRAM[1] + 9, wx + 380, kx + 80
    p.append(f'<path d="M{wsrc},{by}V{bus + 6}Q{wsrc},{bus} {wsrc + 6},{bus}H{PORT_X - 6:g}'
             f'Q{PORT_X:g},{bus} {PORT_X:g},{bus - 6}V{FAN_Y}" class="flow port"/>')
    p.append(f'<path d="M{ksrc},{by}V{bus}" class="flow"/>')
    shim_bottom = SHIM_Y + TILE
    for c in range(COLS):
        x = col_x(c)
        p.append(f'<path d="M{PORT_X:g},{FAN_Y}C{PORT_X:g},{FAN_Y - 24} {x:g},{shim_bottom + 22} {x:g},{shim_bottom}" '
                 f'class="flow"/>')
        p.append(f'<path d="M{x:g},{shim_bottom}V{CORE_Y[0] + TILE / 2:g}" class="flow col"/>')
    port_y = NPU[1] + NPU[3]
    p.append(f'<rect x="{PORT_X - 9:g}" y="{port_y - 6:g}" width="18" height="12" rx="3" class="portbox"/>')
    p.append(f'<text x="{PORT_X - 16:g}" y="{port_y + 18:g}" class="s" text-anchor="end">one 32 B/cycle port · '
             f'<tspan class="b">~57 GB/s</tspan></text>')

    # the array
    for c in range(COLS):
        x = ARRAY_X + c * PITCH
        p += [tile(x, y, "core") for y in CORE_Y]
        p.append(tile(x, MEM_Y, "mem"))
        p.append(tile(x, SHIM_Y, "shim"))
    p.append(text(PORT_X, CORE_Y[0] - 9, "8 columns", "n", "middle"))
    lx = ARRAY_X + COLS * PITCH + 6
    p.append(text(lx, (CORE_Y[0] + CORE_Y[-1] + TILE) / 2 + 4, "32 cores", "s"))
    p.append(text(lx, MEM_Y + TILE / 2 + 4, "memory", "s"))
    p.append(text(lx, SHIM_Y + TILE / 2 + 4, "shim DMA", "s"))

    # the host's arrows into the NPU and DRAM, labelled on the wire
    end = node_x(3) + NODE_W
    y0 = node_cy(0)
    p.append(f'<path d="M{end:g},{y0:g}H{ARRAY_X - 4}" class="arrow"/>')
    p += pill((end + NPU[0]) / 2, y0, "configures")
    y1, ys = node_cy(1), SHIM_Y + TILE / 2
    turn = ARRAY_X - 12
    p.append(f'<path d="M{end:g},{y1:g}H{turn - 8}Q{turn},{y1:g} {turn},{y1 + 8:g}V{ys - 8:g}'
             f'Q{turn},{ys:g} {ARRAY_X - 4},{ys:g}" class="arrow"/>')
    p += pill((end + turn - 8) / 2, y1, "dispatches")
    lx2 = node_x(1) + NODE_W / 2
    p.append(f'<path d="M{lx2:g},{node_cy(2) + NODE_H / 2:g}V{by - 4}" class="arrow"/>')
    p += pill(lx2, (HOST[1] + HOST[3] + DRAM[1]) / 2, "packs into")

    s = t.series
    style = (
        f"text{{font-family:{FONT}}}"
        f".H{{font-size:12px;font-weight:700;letter-spacing:.08em;fill:{t.ink2}}}"
        f".h{{font-size:10.5px;font-weight:700;letter-spacing:.08em;fill:{t.muted}}}"
        f".n{{font-size:10.5px;fill:{t.muted}}}"
        f".t{{font-size:12px;font-weight:600;fill:{t.ink}}}"
        f".s{{font-size:10.5px;fill:{t.ink2}}}.ns{{font-size:10px;fill:{t.ink2}}}.b{{font-weight:700;fill:{t.ink}}}"
        f".p{{font-size:10px;fill:{t.ink2}}}"
        f".panel{{fill:{panel};stroke:{t.border}}}"
        f".node{{fill:{t.surface};stroke:{t.border}}}"
        f".pill{{fill:{t.surface};stroke:{t.border}}}"
        f".weights{{fill:{t.surface};stroke:{s[2]};stroke-opacity:.6}}"
        f".arrow{{fill:none;stroke:{t.ink2};stroke-width:1.3;marker-end:url(#head)}}"
        f".core{{fill:{s[0]}}}.mem{{fill:{s[0]};fill-opacity:.45}}"
        f".shim{{fill:{t.surface};stroke:{t.muted}}}"
        f".portbox{{fill:{s[2]}}}"
        f".flow{{fill:none;stroke:{s[2]};stroke-width:1.6;stroke-dasharray:4 4;"
        f"animation:flow 0.9s linear infinite}}"
        f".port{{stroke-width:3}}.col{{stroke-width:2}}"
        "@keyframes flow{to{stroke-dashoffset:-8}}"
        "@media (prefers-reduced-motion:reduce){.flow{animation:none}}"
    )
    title = "How Artlite runs a model on the NPU"
    desc = ("The host builds each operator's design once (an operator renders C++ and an IRON design per shape, Peano "
            "and aiecc compile them into an ELF that configures the array), packs a GGUF file's weights once into a "
            "weight image in DRAM at their quantization's size, and every step dispatches a transaction through the "
            "C++ runtime and the amdxdna driver. The NPU's 8 columns of shim DMA, memory tile and 4 cores stream the "
            "weights and the KV cache from DRAM through one 32 bytes per cycle port, about 57 GB/s, of the 256 GB/s "
            "LPDDR5X that the CPU and the iGPU share.")
    defs = (f'<defs><marker id="head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0.5L8,4L0,7.5z" fill="{t.ink2}"/></marker>'
            f'<pattern id="tiles" width="12" height="12" patternUnits="userSpaceOnUse" x="{wx}" y="{by}">'
            f'<rect x="3" y="3" width="7" height="7" rx="1.5" fill="{s[2]}" fill-opacity=".14"/></pattern></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{H}" viewBox="0 0 {WIDTH} {H}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n'
            f'<desc id="d">{escape(desc)}</desc>\n<style>{style}</style>\n{defs}\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{H - 1}" rx="8" fill="{t.surface}" '
            f'stroke="{t.border}"/>\n' + "\n".join(p) + "\n</svg>\n")


def main() -> None:
    check_fit()
    for suffix, theme in THEMES.items():
        (IMAGES / f"how-it-works{suffix}.svg").write_text(draw(theme, PANEL[suffix]))


if __name__ == "__main__":
    main()
