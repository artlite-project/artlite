"""The figures of paging.md: docs/images/paging-*.svg and their -dark variants.

    python docs/paging_diagrams.py

One look for every figure, in charts.py's palette and fonts. Hierarchy by role: a panel groups (light grey), a tile
is one step darker (lighter in the dark theme) with a firmer border, both named by an uppercase eyebrow; the memories
and units in a tile are white cards, and a DMA channel or a stream port is a tab on the tile's edge, where its stream
leaves or enters. One color per kind of traffic: activations blue, the B stream
(commands and weights) green, code orange (the configuration, pages), control and locks grey. Fills are these colors
mixed into the surface (opaque: no line shows through), identifiers are monospace, coordinates keep to a 4-pixel grid.
"""

from xml.sax.saxutils import escape

from charts import FONT, IMAGES, THEMES, WIDTH, Theme, text_width

MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'
PANEL = {"": "#f6f8fa", "-dark": "#151b23"}
KINDS = {"data": 0, "code": 1, "wts": 2}  # (traffic -> its series color; "ctl": ink2)


def mono_width(s: str, size: float) -> float:
    return len(s) * size * 0.61


def luminance(c: str) -> float:
    """WCAG relative luminance of a #rrggbb color."""
    lin = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in
           (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def mix(a: str, b: str, f: float) -> str:
    """Color a over color b at opacity f, as an opaque color."""
    ca, cb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x * f + y * (1 - f)):02x}" for x, y in zip(ca, cb))


class Fig:
    """A figure's elements, drawn once for both themes (the theme is its stylesheet)."""

    def __init__(self, h: int, title: str, desc: str):
        self.w, self.h, self.title, self.desc = WIDTH, h, title, desc
        self.e: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, cls: str, rx: float = 6) -> None:
        self.e.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" class="{cls}"/>')

    def text(self, x: float, y: float, s: str, cls: str, anchor: str = "start") -> None:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        self.e.append(f'<text x="{x:g}" y="{y:g}" class="{cls}"{a}>{escape(s)}</text>')

    def path(self, d: str, cls: str) -> None:
        self.e.append(f'<path d="{d}" class="{cls}"/>')

    def panel(self, x: float, y: float, w: float, h: float, name: str = "", note: str = "") -> None:
        """A group (the NPU, DDR, one side of a comparison), named by an eyebrow at its top left."""
        self.rect(x, y, w, h, "panel", 10)
        if name:
            self.eyebrow(x + 14, y + 20, name, note)

    def tile(self, x: float, y: float, w: float, h: float, name: str = "", note: str = "") -> None:
        """A tile of the array, named by an eyebrow at its top left (none: the caller places one)."""
        self.rect(x, y, w, h, "tile", 10)
        if name:
            self.eyebrow(x + 14, y + 20, name, note)

    def eyebrow(self, x: float, y: float, name: str, note: str = "") -> None:
        self.text(x, y, name.upper(), "eb")
        if note:
            self.text(x + text_width(name.upper(), 10, bold=True) * 1.1 + 8, y, note, "en")

    def card(self, x: float, y: float, w: float, h: float, title: str = "", sub: str = "", kind: str = "",
             align: str = "middle", mono: bool = False) -> None:
        """A memory or a unit: white, or tinted by the traffic it holds; a title, a detail under it (monospace: an
        identifier or a size)."""
        self.rect(x, y, w, h, f"card {kind}".strip(), 6)
        tx = x + w / 2 if align == "middle" else x + 12
        if title:
            assert text_width(title, 11.5, True) <= w - 12, f"{title!r} is wider than its card"
            assert (mono_width(sub, 10) if mono else text_width(sub, 10)) <= w - 12, f"{sub!r} is wider than its card"
            self.text(tx, y + h / 2 + (-3 if sub else 4), title, "ti", align)
            if sub:
                self.text(tx, y + h / 2 + 12, sub, "sm" if mono else "sb", align)

    def buf(self, x: float, y: float, w: float, h: float, label: str, kind: str) -> None:
        """A buffer: a tinted block without a border."""
        self.rect(x, y, w, h, f"buf {kind}", 4)
        self.text(x + w / 2, y + h / 2 + 3.5, label, "mo", "middle")

    def port(self, cx: float, cy: float, label: str, kind: str) -> float:
        """A channel's or a port's tab centered at (cx, cy); returns its half width."""
        w = mono_width(label, 9.5) + 14
        self.rect(cx - w / 2, cy - 8, w, 16, f"port {kind}", 4)
        self.text(cx, cy + 3.4, label, "mo", "middle")
        return w / 2

    def arrow(self, kind: str, *pts: tuple[float, float]) -> None:
        """An arrow of kind ``kind`` along ``pts`` (rounded corners), its head at the last point."""
        self.path(route(pts), f"a {kind}")

    def wire(self, kind: str, *pts: tuple[float, float]) -> None:
        self.path(route(pts), f"w {kind}")

    def label(self, x: float, y: float, s: str, anchor: str = "middle", on: str = "") -> None:
        """Text on a line, cut out of it by a halo of what lies behind: the surface, a panel ("p") or a tile ("t")."""
        self.text(x, y, s, f"pl halo{on}", anchor)

    def badge(self, cx: float, cy: float, n: int | str, kind: str = "code") -> None:
        self.e.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="8.5" class="badge {kind}"/>')
        self.text(cx, cy + 3.6, str(n), f"bn {kind}", "middle")

    def steps(self, x: float, y: float, items: list[tuple[str, str, list[str]]], w: float, gap: float = 12) -> float:
        """A numbered list, per item its badge's kind, a bold lead and lines of text; returns where it ends."""
        for i, (kind, lead, lines) in enumerate(items):
            self.badge(x + 9, y, i + 1, kind)
            self.text(x + 26, y + 4, lead, "st")
            for j, s in enumerate(lines):
                assert text_width(s, 10.5) <= w - 26, s
                self.text(x + 26, y + 20 + 15 * j, s, "lb")
            y += 20 + 15 * len(lines) + gap
        return y

    def legend(self, x: float, y: float, kinds: list[tuple[str, str]], step: float = 0) -> None:
        """Arrow kinds and their meanings: in a row from (x, y), or in a column ``step`` apart."""
        for kind, s in kinds:
            self.arrow(kind, (x, y), (x + 24, y))
            self.text(x + 32, y + 3.5, s, "lb")
            if step:
                y += step
            else:
                x += 32 + text_width(s, 10.5) + 20

    def svg(self, t: Theme, panel: str) -> str:
        s = t.series
        color = {k: s[i] for k, i in KINDS.items()} | {"ctl": t.ink2}
        css = [
            f"text{{font-family:{FONT}}}",
            f".eb{{font-size:10px;font-weight:700;letter-spacing:.1em;fill:{t.muted}}}",
            f".ti{{font-size:11.5px;font-weight:600;fill:{t.ink}}}",
            f".st{{font-size:11px;font-weight:700;fill:{t.ink}}}",
            f".lb{{font-size:10.5px;fill:{t.ink2}}}",
            f".pl{{font-size:10px;fill:{t.ink2}}}.sb{{font-size:10px;fill:{t.ink2}}}",
            f".en{{font-size:10.5px;fill:{t.muted}}}.c{{font-size:10px;fill:{t.ink}}}",
            ".bn{font-size:10px;font-weight:700}",
            f".mo,.sm,.mm{{font-family:{MONO};font-size:9.5px;fill:{t.ink}}}",
            f".sm{{font-size:10px;fill:{t.ink2}}}.mm{{font-size:10px;fill:{t.muted}}}",
            f".halo,.halop,.halot{{paint-order:stroke;stroke:{t.surface};stroke-width:5px;stroke-linejoin:round}}",
            f".halop{{stroke:{panel}}}.halot{{stroke:{mix(t.ink2, panel, .08)}}}",
            f".panel{{fill:{panel};stroke:{t.border}}}",
            f".tile{{fill:{mix(t.ink2, panel, .08)};stroke:{t.axis}}}",
            f".card{{fill:{t.surface};stroke:{t.border}}}",
            ".a,.w{fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}",
            ".w.ctl,.a.ctl{stroke-width:1.3}",
            f".hatch{{fill:url(#hatch);stroke:{t.border}}}",
            f".slot{{fill:{t.surface};stroke:{s[1]};stroke-dasharray:4 3}}",
            f".axis{{fill:none;stroke:{t.axis}}}",
            f".guide{{fill:none;stroke:{t.axis};stroke-dasharray:3 3}}",
            f".resident{{fill:{mix(t.ink2, t.surface, .12)};stroke:{mix(t.ink2, t.surface, .5)}}}",
            f".gcore{{fill:{s[0]}}}.gmem{{fill:{mix(s[0], t.surface, .45)}}}.gshim{{fill:{t.surface};stroke:{t.muted}}}",
        ]
        for k, c in color.items():
            css += [f".a.{k},.w.{k}{{stroke:{c}}}", f".a.{k}{{marker-end:url(#h{k})}}",
                    f".card.{k}{{fill:{mix(c, t.surface, .1)};stroke:{mix(c, t.surface, .55)}}}",
                    f".port.{k}{{fill:{mix(c, t.surface, .16)};stroke:{c}}}",
                    f".buf.{k}{{fill:{mix(c, t.surface, .2)}}}",
                    f".solid.{k}{{fill:{c}}}",
                    f".badge.{k}{{fill:{c};stroke:{t.surface};stroke-width:2px}}",
                    f".bn.{k}{{fill:{'#0d1117' if luminance(c) > 0.25 else '#ffffff'}}}"]  # (the better contrast)
        marks = "".join(
            f'<marker id="h{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto"><path d="M1,1L9,5L1,9L3.2,5z" fill="{c}"/></marker>'
            for k, c in color.items())
        defs = (f'<defs>{marks}<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)"><rect width="5" height="5" fill="{t.surface}"/>'
                f'<line x1="0" y1="0" x2="0" y2="5" stroke="{t.border}" stroke-width="1.6"/></pattern></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">\n'
                f'<title id="t">{escape(self.title)}</title>\n<desc id="d">{escape(self.desc)}</desc>\n'
                f'<style>{"".join(css)}</style>\n{defs}\n'
                f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="8" fill="{t.surface}" '
                f'stroke="{t.border}"/>\n' + "\n".join(self.e) + "\n</svg>\n")


def route(pts: tuple[tuple[float, float], ...], r: float = 6) -> str:
    """A path through ``pts`` with its corners rounded by up to ``r``."""
    d = f"M{pts[0][0]:g},{pts[0][1]:g}"
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
        a = min(r, (abs(x1 - x0) + abs(y1 - y0)) / 2, (abs(x2 - x1) + abs(y2 - y1)) / 2)
        ux, uy = _unit(x1 - x0, y1 - y0)
        vx, vy = _unit(x2 - x1, y2 - y1)
        d += f"L{x1 - a * ux:g},{y1 - a * uy:g}Q{x1:g},{y1:g} {x1 + a * vx:g},{y1 + a * vy:g}"
    return d + f"L{pts[-1][0]:g},{pts[-1][1]:g}"


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = (dx * dx + dy * dy) ** 0.5 or 1
    return dx / n, dy / n


# ---------------------------------------------------------------------------------------------------------------------
# 1. a classic design: the code is part of the configuration


def classic() -> Fig:
    f = Fig(660, "A classic mlir-aie design",
            "aiecc compiles a design (C++ kernels and an IRON design) into main.pdi, the configuration of every tile, "
            "and insts.bin, the runtime sequence. The NPU's command processor first writes the configuration into "
            "the tiles: each core's program memory receives its ELF's code, with its data, locks, DMA buffer "
            "descriptors and stream routes. Then each run streams the input from DDR through the shim tile's and the "
            "memory tile's DMA channels into the core's data memory, and the core's output back the same way. "
            "Another operator is another design: a reset and a full configuration between them.")
    # build
    f.panel(24, 24, 232, 256, "Build", "once per design")
    f.card(40, 52, 200, 48, "kernel.cc · design.py", "C++ kernels · IRON design")
    f.card(40, 128, 200, 48, "Peano · aiecc", "compile, place, route, link")
    f.card(40, 204, 96, 48, "main.pdi", "configuration")
    f.card(144, 204, 96, 48, "insts.bin", "runtime sequence")
    f.arrow("ctl", (140, 100), (140, 126))
    f.arrow("ctl", (88, 176), (88, 202))
    f.arrow("ctl", (192, 176), (192, 202))
    f.legend(40, 348, [("code", "code: the configuration"), ("data", "data"), ("ctl", "control")], step=22)

    # the NPU: a column of tiles, the command processor
    f.panel(272, 24, 584, 468, "NPU", "one column of 8, one core of 4")
    tx, tw, xl, xr = 472, 360, 528, 776  # the column's tiles; its input and output lanes
    yc, ym, ys = 52, 252, 400  # the tiles' tops
    f.tile(tx, yc, tw, 160, "Core tile", "(c, 2)")
    f.card(tx + 16, yc + 32, 168, 48, "program memory", "16 KB · the ELF's code", "code")
    f.card(tx + 220, yc + 32, 124, 48, "core", "VLIW, vector unit")
    f.arrow("ctl", (tx + 184, yc + 56), (tx + 218, yc + 56))
    f.label(tx + 201, yc + 50, "fetch", on="t")
    f.card(tx + 16, yc + 96, 328, 40)
    f.text(tx + 180, yc + 120, "data memory · 64 KB", "sb", "middle")
    f.buf(xl - 32, yc + 104, 64, 24, "in ×2", "data")
    f.buf(xr - 32, yc + 104, 64, 24, "out ×2", "data")
    f.arrow("ctl", (tx + 264, yc + 96), (tx + 264, yc + 82))
    f.arrow("ctl", (tx + 292, yc + 82), (tx + 292, yc + 94))
    f.tile(tx, ym, tw, 108)
    f.eyebrow(600, ym + 20, "Memory tile", "(c, 1)")
    f.card(tx + 16, ym + 34, 328, 40)
    f.text(tx + 180, ym + 58, "ObjectFifo buffers · 512 KB", "sb", "middle")
    f.buf(xl - 32, ym + 42, 64, 24, "in ×2", "data")
    f.buf(xr - 32, ym + 42, 64, 24, "out ×2", "data")
    f.tile(tx, ys, tw, 64)
    f.eyebrow(600, ys + 20, "Shim tile", "(c, 0)")
    f.card(600, ys + 28, 148, 24)
    f.text(674, ys + 44, "BDs · task queues", "sb", "middle")
    # the data path: up the input lane, down the output lane, through the tiles' DMA channels
    for pts in (((xl, 520), (xl, ys + 10)), ((xl, ys - 8), (xl, ym + 118)), ((xl, ym + 100), (xl, ym + 68)),
                ((xl, ym + 42), (xl, ym + 10)), ((xl, ym - 8), (xl, yc + 170)), ((xl, yc + 152), (xl, yc + 130)),
                ((xr, yc + 128), (xr, yc + 150)), ((xr, yc + 168), (xr, ym - 10)), ((xr, ym + 8), (xr, ym + 40)),
                ((xr, ym + 66), (xr, ym + 98)), ((xr, ym + 116), (xr, ys - 10)), ((xr, ys + 8), (xr, 518))):
        f.arrow("data", *pts)
    for x, y, s in ((xl, yc + 160, "S2MM 0"), (xr, yc + 160, "MM2S 0"), (xl, ym, "MM2S 0"), (xr, ym, "S2MM 1"),
                    (xl, ym + 108, "S2MM 0"), (xr, ym + 108, "MM2S 1"), (xl, ys, "MM2S 0"), (xr, ys, "S2MM 0")):
        f.port(x, y, s, "data")
    # the command processor: 1 the configuration into every tile, 2 each run's shim tasks
    f.card(288, 270, 132, 52, "command processor", "the NPU's firmware")
    f.arrow("ctl", (192, 252), (192, 284), (286, 284))
    f.arrow("ctl", (88, 252), (88, 308), (286, 308))
    f.label(244, 288, "runs")
    f.label(244, 312, "loads")
    trunk = 444
    f.wire("code", (420, 296), (trunk, 296))
    f.wire("code", (trunk, yc + 56), (trunk, ys + 20))
    for y in (yc + 56, ym + 54, ys + 20):
        f.arrow("code", (trunk, y), (tx - 3, y))
    f.badge(trunk, 196, 1)
    f.arrow("ctl", (300, 322), (300, ys + 46), (tx - 3, ys + 46))
    f.badge(300, 412, 2, "ctl")
    f.badge(322, 356, 1)
    f.text(338, 360, "configure, once", "lb")
    f.badge(322, 382, 2, "ctl")
    f.text(338, 386, "run, every call", "lb")
    # DDR
    f.panel(272, 504, 584, 60, "DDR")
    f.card(xl - 44, 520, 88, 32, "input")
    f.card(xr - 44, 520, 88, 32, "output")
    # time
    f.panel(24, 580, 832, 64, "Time")
    f.text(38, 618, "a design", "lb")
    f.text(38, 632, "per operator", "lb")
    x = 160
    for name, w, kind in (("configure A", 92, "code"), ("run A", 150, "data"), ("configure B", 92, "code"),
                          ("run B", 116, "data"), ("configure A", 92, "code"), ("run A", 150, "data")):
        f.rect(x, 594, w - 4, 26, f"card {kind}", 5)
        f.text(x + (w - 4) / 2, 611, name, "c", "middle")
        x += w
    f.text(160 + 242 + 44, 636, "a reset and a full configuration: ~80-90 µs", "lb", "middle")
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 2. the vision overlay: one design, the operations' commands and weights in the rows' B streams

RY = {5: 108, 4: 168, 3: 228, 2: 288}  # core rows' centers (the array and the column)
MEM_Y, SHIM_Y = 356, 460  # the column's memory and shim tiles' tops


def gx(c: int) -> float:
    return 84 + 34 * c


def overlay() -> Fig:
    f = Fig(600, "The vision overlay's design",
            "One configuration runs every operation of Stable Diffusion's UNet. Each row of 8 cores receives one "
            "broadcast B stream, commands and the weights of its channel slice, from the shim tile of column 0, 2, 4 "
            "or 6. In each column the memory tile holds the activations of that column's tokens: it sends the A "
            "stream to each of its 4 cores and receives each core's outputs, the O stream; the shim tile fills the "
            "memory tile from DDR and drains it to DDR. The cores run a main loop that executes the commands; the "
            "host builds every dispatch's transaction.")
    f.panel(24, 24, 344, 552, "Array", "8 columns × 4 cores")
    f.legend(40, 60, [("wts", "B: commands, weights"), ("data", "activations")])
    mem_cy, shim_cy = MEM_Y + 32, SHIM_Y + 24
    for r, y in RY.items():  # (row r's B stream: up from its shim, along the row; on into the column, right)
        f.wire("wts", (gx(2 * (r - 2)) + 11, shim_cy), (gx(2 * (r - 2)) + 11, y))
        f.wire("wts", (gx(0) + 11, y), (gx(7) + 11, y))
    for c in range(8):
        for y in RY.values():
            f.rect(gx(c), y - 11, 22, 22, "gcore", 4)
        f.rect(gx(c), mem_cy - 11, 22, 22, "gmem", 4)
        f.rect(gx(c), shim_cy - 11, 22, 22, "gshim", 4)
        f.text(gx(c) + 11, shim_cy + 30, str(c), "mm", "middle")
    for r, y in RY.items():
        f.text(40, y + 4, f"row {r}", "en")
    f.text(40, mem_cy + 4, "memory", "en")
    f.text(40, shim_cy + 4, "shim", "en")
    f.text(40, 548, "Row r's B stream: one shim's MM2S 1,", "lb")
    f.text(40, 563, "broadcast to the row's 8 cores.", "lb")

    f.panel(384, 24, 472, 552, "One column", "enlarged")
    x0, w = 408, 232
    for r, y in RY.items():
        f.tile(x0, y - 24, w, 48)
        f.text(x0 + w / 2 + 2, y - 1, f"core (c, {r})", "ti", "middle")
        f.text(x0 + w / 2 + 2, y + 13, "main loop", "sb", "middle")
        xa = 692 + 28 * (r - 2)  # (row r's A and O lanes: nested, so that no two cross)
        f.arrow("wts", (gx(7) + 22, y), (x0 - 26, y))
        f.arrow("data", (xa, MEM_Y), (xa, y + 10), (x0 + w + 26, y + 10))
        f.arrow("data", (x0 + w + 26, y - 10), (xa + 12, y - 10), (xa + 12, MEM_Y - 2))
        f.port(x0, y, "S2MM 1", "wts")
        f.port(x0 + w, y - 10, "MM2S 0", "data")
        f.port(x0 + w, y + 10, "S2MM 0", "data")
    xf, xd, xb = 548, 628, 760  # the fill, drain and B lanes
    f.tile(x0, MEM_Y, 432, 64)
    f.eyebrow(x0 + 14, MEM_Y + 20, "Memory tile", "(c, 1)")
    f.text(x0 + 14, MEM_Y + 42, "the activations of column c's tokens", "sb")
    f.text(826, MEM_Y + 42, "A: MM2S 0-3 · O: S2MM 0-3", "sm", "end")
    f.tile(x0, SHIM_Y, 432, 48)
    f.eyebrow(x0 + 14, SHIM_Y + 28, "Shim tile", "(c, 0)")
    f.text(xb - 12, SHIM_Y + 32, "columns 0, 2, 4, 6", "sb", "end")
    f.tile(x0, 520, 432, 48)
    f.eyebrow(x0 + 14, 548, "DDR")
    f.card(xf - 66, 530, 160, 28, "activations")
    f.card(xb - 72, 530, 144, 28, "weights · commands")
    f.arrow("data", (xf, 530), (xf, SHIM_Y + 10))
    f.arrow("data", (xf, SHIM_Y - 8), (xf, MEM_Y + 74))
    f.arrow("data", (xd, MEM_Y + 72), (xd, SHIM_Y - 10))
    f.arrow("data", (xd, SHIM_Y + 8), (xd, 528))
    f.arrow("wts", (xb, 530), (xb, SHIM_Y + 10))
    f.text(xf - 12, MEM_Y + 92, "fill", "pl", "end")
    f.text(xd + 12, MEM_Y + 92, "drain", "pl")
    for x, y, s, k in ((xf, MEM_Y + 64, "S2MM 4", "data"), (xd, MEM_Y + 64, "MM2S 4", "data"),
                       (xf, SHIM_Y, "MM2S 0", "data"), (xd, SHIM_Y, "S2MM 0", "data"), (xb, SHIM_Y, "MM2S 1", "wts")):
        f.port(x, y, s, k)
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 3. a vision overlay core's program memory: the resident code and the slot its roles' pages share

RESIDENT = 8976
PAGES = [("a", "attention, heads of 80, 160", 5776), ("c", "attention, heads of 40", 5216),
         ("v", "VAE attention, a head of 512", 3360), ("gn", "GroupNorm (+ SiLU)", 4880),
         ("ln", "LayerNorm, GEGLU", 5984)]
PM_X, PM_W = 232, 512  # 16 KB


def bx(nbytes: float) -> float:
    return PM_X + nbytes * PM_W / 16384


def budget() -> Fig:
    total = RESIDENT + sum(n for *_, n in PAGES)
    f = Fig(356, "Program memory of a vision overlay core",
            f"A core has 16 KB of program memory. For Stable Diffusion its cores need {total} bytes of code in all: {RESIDENT} bytes of "
            "resident code (startup, main loop, GEMM and convolution, the pager) and five roles' code, which together "
            "do not fit. As pages, each role is linked at the slot, the 7408 bytes after the resident code, and only "
            "one is loaded at a time: attention with heads of 80 and 160 (5776 bytes), attention with heads of 40 "
            "(5216), the VAE's attention (3360), GroupNorm (4880), LayerNorm with GEGLU (5984).")
    f.eyebrow(24, 36, "Program memory of one core", "16 KB: all the code the core can run")
    for kb in range(0, 17, 4):
        x = bx(kb * 1024)
        f.path(f"M{x:g},62V68", "axis")
        f.text(x, 56, f"{kb} KB", "mm", "middle")
    f.path(f"M{bx(0):g},68H{bx(16384):g}", "axis")
    # all of it resident
    y = 84
    f.text(PM_X - 14, y + 17, "every role resident", "lb", "end")
    f.rect(bx(0), y, bx(RESIDENT) - bx(0), 26, "resident", 4)
    f.text((bx(0) + bx(RESIDENT)) / 2, y + 17, "resident", "c", "middle")
    cut, acc = bx(16384), RESIDENT
    for name, _, n in PAGES:
        a, b = bx(acc), min(bx(acc + n), cut)
        if a < cut:
            f.rect(a + 2, y, b - a - 2, 26, "card code", 4)
            f.text((a + b) / 2 + 1, y + 17, name, "mo", "middle")
        acc += n
    f.path(f"M{cut:g},{y - 6}l5,9.5l-10,9.5l10,9.5l-5,9.5", "w ctl")
    f.text(cut + 16, y + 11, f"{total} B in all:", "lb")
    f.text(cut + 16, y + 26, "2.1 × the memory", "lb")
    # resident + the slot
    y = 132
    f.text(PM_X - 14, y + 17, "with pages", "lb", "end")
    f.rect(bx(0), y, bx(RESIDENT) - bx(0), 26, "resident", 4)
    f.text((bx(0) + bx(RESIDENT)) / 2, y + 17, f"resident code · {RESIDENT} B", "c", "middle")
    f.rect(bx(RESIDENT) + 2, y, bx(16384) - bx(RESIDENT) - 2, 26, "slot", 4)
    f.text((bx(RESIDENT) + bx(16384)) / 2, y + 17, "the slot · 7408 B", "c", "middle")
    f.text(bx(0), y + 44, "startup, main loop, GEMM, convolution, the pager", "sb")
    # the pages, each linked at the slot
    y = 200
    f.text(bx(RESIDENT) + 4, y - 8, "the pages, each linked at the slot, one loaded at a time", "sb")
    for i, (name, what, n) in enumerate(PAGES):
        yy = y + i * 28
        f.text(bx(RESIDENT) - 14, yy + 15, what, "lb", "end")
        f.text(bx(RESIDENT) - 14 - text_width(what, 10.5) - 8, yy + 15, name, "mo", "end")
        f.rect(bx(RESIDENT) + 2, yy, bx(RESIDENT + n) - bx(RESIDENT) - 2, 20, "card code", 4)
        f.text(bx(RESIDENT + n) + 8, yy + 14, f"{n} B", "sm")
    for x in (bx(RESIDENT), bx(16384)):
        f.path(f"M{x:g},126V{y + 4 * 28 + 28}", "guide")
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 4. one page load (self-DMA), step by step inside a core tile

STEPS = [
    ("wts", "Arrive", ["The row's B stream brings the page:", "S2MM 1 writes two B slots, b0 and b1."]),
    ("ctl", "Recognize", ["b0's first word is a page header, not", "a command. The core waits for b1."]),
    ("ctl", "Sleep", ["The pager releases lock 7 by 2 and", "sleeps on lock 9: no fetches now."]),
    ("code", "Stream", ["MM2S 1 sends b0, then b1. The switch", "routes them into the tile's TileControl."]),
    ("code", "Write", ["TileControl writes the code into the slot,", "the state words, and last, lock 9."]),
    ("ctl", "Wake", ["The core waits for lock 8 (both slots", "sent) and frees b0, b1 for commands."]),
    ("ctl", "Call", ["The role's commands call the page", "through its entry in the state words."]),
]


def steps() -> Fig:
    f = Fig(508, "A page load inside a core tile",
            "The page arrives in the row's B stream as two B slots that the core's S2MM 1 channel writes into its data "
            "memory. The core recognizes the page's header in the first slot, releases its MM2S 1 channel and sleeps "
            "on its wake lock. MM2S 1 streams both slots into the stream switch, which routes them, as packets of "
            "stream id 7, back into the tile's own TileControl port. TileControl writes the code into the slot of "
            "program memory, the pager's state words into data memory, and finally the wake lock. The core wakes, "
            "frees the slots, and later calls the page's code through the entry in the state words.")
    f.tile(24, 24, 576, 432, "Core tile", "(c, r)")
    # the stream switch, the tile's DMA channels and TileControl
    f.rect(40, 52, 56, 388, "card", 6)
    f.text(68, 70, "stream", "sb", "middle")
    f.text(68, 83, "switch", "sb", "middle")
    f.card(120, 186, 96, 32, "TileControl", kind="code")
    f.card(120, 286, 96, 32, "MM2S 1", kind="code")
    f.card(120, 382, 96, 32, "S2MM 1", kind="wts")
    # program memory
    f.rect(248, 52, 200, 128, "card", 6)
    f.text(262, 72, "program memory · 16 KB", "ti")
    f.rect(260, 82, 176, 36, "resident", 4)
    f.text(270, 97, "resident · 8976 B", "c")
    f.text(270, 111, "main loop, GEMM, conv, pager", "sb")
    f.rect(260, 126, 176, 42, "card code", 4)
    f.text(270, 143, "the slot · 7408 B", "c")
    f.text(270, 158, "the page: a, c, v, gn or ln", "sb")
    # data memory
    f.rect(248, 228, 200, 212, "card", 6)
    f.text(262, 248, "data memory · 64 KB", "ti")
    f.buf(260, 260, 176, 26, "state words: page id, entries", "code")
    for x, s in ((260, "b0"), (352, "b1")):
        f.rect(x, 316, 84, 36, "buf wts", 4)
        f.text(x + 42, 331, f"B slot {s}", "mo", "middle")
        f.text(x + 42, 345, "5760 B", "sm", "middle")
    # the core and its locks
    f.rect(480, 52, 104, 128, "card", 6)
    f.text(532, 84, "core", "ti", "middle")
    f.text(532, 100, "main loop", "sb", "middle")
    f.text(532, 136, "asleep in", "sb", "middle")
    f.text(532, 149, "the pager", "sb", "middle")
    f.text(532, 162, "(steps 3-5)", "sb", "middle")
    f.rect(480, 228, 104, 212, "card", 6)
    f.text(494, 248, "locks", "ti")
    for y, s in ((260, "9 · wake"), (292, "7 · send"), (324, "8 · done"), (356, "2, 3 · B")):
        f.buf(492, y, 80, 24, s, "ctl")
    # 1: the B stream into the slots
    f.arrow("wts", (68, 492), (68, 398), (118, 398))
    f.text(80, 484, "the row's B stream", "pl")
    f.arrow("wts", (216, 398), (302, 398), (302, 354))
    f.arrow("wts", (302, 398), (394, 398), (394, 354))
    f.badge(240, 398, 1, "wts")
    # 2: the core reads b0's first word; 3, 6: its locks
    f.badge(262, 318, 2, "ctl")
    f.arrow("ctl", (532, 180), (532, 226))
    f.badge(552, 200, 3, "ctl")
    f.badge(572, 200, 6, "ctl")
    # 4: the slots out of MM2S 1, through the switch, into TileControl
    f.wire("code", (302, 316), (302, 302))
    f.arrow("code", (394, 316), (394, 302), (218, 302))
    f.arrow("code", (120, 302), (80, 302), (80, 202), (118, 202))
    f.badge(80, 266, 4)
    f.label(80, 236, "id 7")
    # 5: TileControl's writes
    f.arrow("code", (216, 202), (232, 202), (232, 147), (258, 147))
    f.arrow("code", (216, 202), (232, 202), (232, 273), (258, 273))
    f.arrow("code", (216, 202), (464, 202), (464, 272), (490, 272))
    f.badge(348, 202, 5)
    # 7: the core runs the page
    f.arrow("ctl", (448, 116), (478, 116))
    f.badge(463, 98, 7, "ctl")
    f.steps(624, 44, STEPS, 232, gap=11)
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 5. who writes the code: the host (a role switch) or the core itself (a page in its input stream)


def paths() -> Fig:
    f = Fig(456, "Two ways to change a row's code at run time",
            "Left: the host writes the code. It waits until the row's operations are complete, sends the code as "
            "control packets from shim tile 1 to the TileControl port of each of the row's 8 cores, and polls every "
            "core until the code is written; the row is idle meanwhile. Right: the core loads its page. The host puts "
            "the page into the row's B stream ahead of the operation that needs it; each core receives it in its B "
            "slots and streams it into its own TileControl. Nothing waits but the page's own transfer.")
    for k, ox in enumerate((24, 448)):
        own = k == 1
        kind = "wts" if own else "code"
        if own:
            f.panel(ox, 24, 408, 408, "The core loads its page", "nothing waits")
        else:
            f.panel(ox, 24, 408, 408, "The host writes the code", "the row idles from 1 to 3")
        src = 0 if own else 1  # (the sending shim's column: row 2's B stream, or the control packets)
        xs = (ox + 16, ox + 136, ox + 280)
        lane = 36 if own else 56
        f.text(ox + 264, 134, "…", "ti", "middle")
        for c, x in zip((0, 1, 7), xs):
            f.tile(x, 56, 112, 140)
            f.text(x + 12, 74, f"core ({c}, 2)", "mm")
            f.rect(x + 10, 84, 92, 26, "card code", 5)
            f.text(x + 56, 101, "program memory", "c", "middle")
            if own:
                f.rect(x + 10, 124, 92, 24, "card code", 5)
                f.text(x + 56, 140, "TileControl", "mo", "middle")
                f.buf(x + 10, 160, 52, 20, "slots", "wts")
                f.arrow("code", (x + 56, 124), (x + 56, 112))
                f.arrow("code", (x + 36, 160), (x + 36, 150))
                f.wire("wts", (x + lane, 188), (x + lane, 182))
            else:
                f.arrow("code", (x + lane, 188), (x + lane, 112))
            f.arrow(kind, (x + lane, 220), (x + lane, 206))
            f.port(x + lane, 196, "S2MM 1" if own else "TileControl", kind)
            f.tile(x, 244, 112, 40)
            f.text(x + 56, 272, f"shim ({c}, 0)", "mm", "middle")
        f.wire(kind, (xs[0] + lane, 220), (xs[2] + lane, 220))
        sx = xs[src] + 56
        f.wire(kind, (sx, 236), (sx, 220))
        f.label(ox + 264, 223, "B stream" if own else "control packets", on="p")
        f.port(sx, 244, "MM2S 1", kind)
        f.card(xs[src], 308, 112, 40, "DDR", "B run: page first" if own else "the role's code")
        f.arrow(kind, (sx, 308), (sx, 254))
        items = ([("Push", "the B run: the page, then the op's commands")] if own else
                 [("Wait", "until the row's ops are complete"), ("Push", "shim 1 sends the packets"),
                  ("Poll", "every core's marker word")])
        for i, (lead, s) in enumerate(items):
            y = 372 + 20 * i
            f.badge(ox + 25, y, i + 1, "ctl")
            f.text(ox + 42, y + 4, lead, "st")
            f.text(ox + 42 + text_width(lead, 11, True) + 6, y + 4, s, "lb")
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 6. a page as the B stream carries it: whole control packets in two slots, the wake lock last


def packets() -> Fig:
    f = Fig(312, "A page in two B slots",
            "A page is a stream of control packets: each has a stream header (packet id 7), a control header (write, "
            "1 to 4 words, the tile address) and up to 4 words. Slot b0 holds the first 240 packets of code; slot b1 "
            "the rest of the code, the state words, filler packets that rewrite the state words with the same values, "
            "and last the packet that sets the wake lock. A slot's buffer descriptor ends the stream's packet there, "
            "so no packet crosses the slots' boundary.")
    f.eyebrow(24, 36, "One packet", "6 words, 24 B: one write, applied by TileControl")
    cells = [("stream header", "id 7 · parity", 148, "card"), ("control header", "write · 4 words · address", 188, "card")]
    cells += [(f"word {i}", "code", 120, "card code") for i in range(4)]
    x = 24
    for title, sub, w, cls in cells:
        f.rect(x, 52, w - 4, 44, cls, 6)
        f.text(x + (w - 4) / 2, 70, title, "ti", "middle")
        f.text(x + (w - 4) / 2, 86, sub, "sm", "middle")
        x += w
    f.eyebrow(24, 136, "The page", "ln: 5984 B of code in 374 packets (widths schematic)")
    y, x0, w = 152, 24, 408
    parts = [  # (slot, start and end as fractions of the slot, class, label)
        (0, 0, 1, "card code", "code: packets 1-240"),
        (1, 0, 0.56, "card code", "code: packets 241-374"),
        (1, 0.56, 0.63, "buf ctl", "state"),
        (1, 0.63, 0.965, "hatch", "fillers"),
        (1, 0.965, 1, "solid code", ""),
    ]
    for k, a, b, cls, label in parts:
        sx = x0 + k * (w + 16)
        f.rect(sx + a * w + (2 if a else 0), y, (b - a) * w - (2 if a else 0), 44, cls, 4)
        if label:
            f.text(sx + (a + b) / 2 * w + 1, y + 26, label, "c" if cls != "hatch" else "c halo", "middle")
    for k in range(2):
        sx = x0 + k * (w + 16)
        f.text(sx + w / 2, y + 64, f"B slot b{k} · 5760 B", "sm", "middle")
        f.path(f"M{sx + w:g},{y - 6}V{y + 50}", "w ctl")
    f.text(x0 + 2 * w + 16, y - 10, "wake lock", "sb", "end")
    f.text(x0 + w + 8, y + 86, "a slot's BD ends with TLAST, so the packets are whole in each slot", "sb", "middle")
    y = 284
    x = 24
    for cls, s in (("card code", "code → program memory, at the slot"),
                   ("buf ctl", "state words → data memory: page id, entries"),
                   ("solid code", "wake lock := 1, last: the core wakes")):
        f.rect(x, y - 10, 14, 14, cls, 3)
        f.text(x + 22, y + 1, s, "lb")
        x += 22 + text_width(s, 10.5) + 32
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 7. the memtile transport (decode): the core asks its memory tile for a page


MT_STEPS = [
    ("code", "Load the store", ["when stale (after a configuration, a", "prefill): the stores and their BDs"]),
    ("ctl", "Request", ["a tile of a paged format whose page is", "not loaded: 3 words out, then sleep"]),
    ("ctl", "Queue", ["TileControl writes the page's BD id", "into MM2S 5's start queue"]),
    ("code", "Stream", ["MM2S 5 streams the page into the", "core's TileControl"]),
    ("code", "Write, wake", ["the slot, the state words, the wake", "lock: the core runs the page"]),
]


def memtile() -> Fig:
    f = Fig(412, "A page from the memory tile",
            "In the decode overlay, the main core of a column requests a page itself: it sends three words through its "
            "stream port, a control packet to its memory tile's TileControl that writes the page's buffer descriptor "
            "into the start queue of MM2S channel 5, and sleeps on its wake lock. MM2S 5 streams the page from the "
            "memory tile's page store into the core's TileControl, which writes the code into the slot, the state "
            "words and the wake lock. The host writes the store and its buffer descriptors before a dispatch when "
            "they are stale: after the configuration and after a prefill. The helper core pages the same way.")
    end = f.steps(24, 40, MT_STEPS, 268, gap=10)
    f.card(24, end + 4, 200, 44, "command processor", "the dispatch's prologue")
    x0, ym = 324, 236
    xr, xp = x0 + 300, x0 + 420  # the request's and the page's lanes
    # the main core
    f.tile(x0, 24, 532, 144, "Main core", "(c, 4)")
    f.rect(x0 + 16, 52, 236, 100, "card", 6)
    f.text(x0 + 30, 72, "program memory", "ti")
    f.rect(x0 + 28, 82, 212, 26, "resident", 4)
    f.text(x0 + 38, 99, "resident: Q3_K, the pager", "c")
    f.rect(x0 + 28, 116, 212, 26, "card code", 4)
    f.text(x0 + 38, 133, "slot: Q4_K, Q5_K or Q6_K", "c")
    f.buf(x0 + 272, 56, 120, 26, "state words", "code")
    f.buf(x0 + 272, 90, 120, 26, "wake lock", "code")
    f.arrow("code", (xp, 160), (xp, 69), (x0 + 394, 69))
    f.arrow("code", (xp, 103), (x0 + 394, 103))
    f.arrow("code", (xp, 160), (xp, 129), (x0 + 242, 129))
    f.badge(xp, 146, 5)
    f.text(x0 + 16, 196, "rows 3, 2 between: the helper core (c, 3)", "sb")
    f.text(x0 + 16, 210, "pages its half of each tile the same way", "sb")
    # the memory tile
    f.tile(x0, ym, 532, 156, "Memory tile", "(c, 1)")
    f.rect(x0 + 16, ym + 32, 252, 108, "card", 6)
    f.text(x0 + 30, ym + 52, "page store · 32 KB", "ti")
    for i, core in enumerate(("main", "helper")):
        y = ym + 64 + 32 * i
        f.text(x0 + 30, y + 16, core, "sb")
        for j, fmt in enumerate(("Q4_K", "Q5_K", "Q6_K")):
            f.buf(x0 + 82 + 60 * j, y, 54, 24, fmt, "code")
    f.card(xr - 32, ym + 40, 64, 26, "queue")
    # 2 the request, 3 the queue, 4 the page from the store
    f.arrow("ctl", (xr, 178), (xr, ym - 10))
    f.badge(xr, 202, 2, "ctl")
    f.arrow("ctl", (xr, ym + 8), (xr, ym + 38))
    f.badge(xr - 18, ym + 24, 3, "ctl")
    f.arrow("ctl", (xr + 32, ym + 53), (xp - 12, ym + 53), (xp - 12, ym + 10))
    f.arrow("code", (x0 + 268, ym + 112), (xp + 12, ym + 112), (xp + 12, ym + 10))
    f.text(xp + 22, ym + 84, "the page's BD:", "sb")
    f.text(xp + 22, ym + 98, "40-45", "sm")
    f.arrow("code", (xp, ym - 10), (xp, 178))
    f.badge(xp, 202, 4)
    f.port(xr, 168, "stream port", "ctl")
    f.port(xp, 168, "TileControl", "code")
    f.port(xr, ym, "TileControl", "ctl")
    f.port(xp, ym, "MM2S 5", "code")
    # 1: the host's store
    f.arrow("code", (224, end + 26), (x0 + 14, end + 26))
    f.badge(268, end + 26, 1)
    return f


# ---------------------------------------------------------------------------------------------------------------------
# 8. the vision overlay's rows in time: host role switches against pages (schematic)

OPS_SWITCH = [  # (label, units, rows that run it, kind)
    ("LN", 4, (4, 5), "op"), ("GEMM", 7, (2, 3, 4, 5), "op"), ("switch", 2, (4, 5), "switch"),
    ("self-attn", 7, (2, 3, 4, 5), "op"), ("switch", 2, (4, 5), "switch"), ("GEMM", 7, (2, 3, 4, 5), "op"),
    ("LN", 4, (4, 5), "op"), ("GEMM", 5, (2, 3, 4, 5), "op"), ("cross-attn", 7, (2, 3), "op"),
]
OPS_PAGES = [
    ("", 0.5, (2, 3, 4, 5), "page"), ("LN", 2, (2, 3, 4, 5), "op"), ("GEMM", 7, (2, 3, 4, 5), "op"),
    ("", 0.5, (2, 3, 4, 5), "page"), ("self-attn", 7, (2, 3, 4, 5), "op"), ("GEMM", 7, (2, 3, 4, 5), "op"),
    ("", 0.5, (2, 3, 4, 5), "page"), ("LN", 2, (2, 3, 4, 5), "op"), ("GEMM", 5, (2, 3, 4, 5), "op"),
    ("", 0.5, (2, 3, 4, 5), "page"), ("cross-attn", 3.5, (2, 3, 4, 5), "op"),
]
UNIT = 15.5


def timeline() -> Fig:
    f = Fig(372, "The vision overlay's rows in time",
            "A schematic of part of a transformer block on the 4 core rows. With host role switches, the LayerNorm "
            "runs on rows 4 and 5 only and cross-attention on rows 2 and 3 only, and every switch of rows 4 and 5 "
            "to attention and back waits for the array to drain. With pages, every row runs every operation, and a "
            "row's page arrives in its B stream just before the operation that needs it, about 2.7 microseconds.")
    x_start = 88
    for name, note, ops, y0 in (("Host role switches", "rows 2, 3: attention · rows 4, 5: the norms", OPS_SWITCH, 24),
                                ("Pages", "every row runs every operation", OPS_PAGES, 180)):
        f.eyebrow(24, y0 + 12, name, note)
        x = x_start
        for label, units, rows, kind in ops:
            w = units * UNIT
            for i, r in enumerate((5, 4, 3, 2)):
                y = y0 + 24 + 25 * i
                cls = {"op": "card", "switch": "solid ctl", "page": "solid code"}[kind] if r in rows else "hatch"
                f.rect(x + 1, y, w - 2, 21, cls, 4)
                if label and r in rows and kind == "op" and w > text_width(label, 10) + 8:
                    f.text(x + w / 2, y + 14.5, label, "c", "middle")
            if kind == "switch":
                f.text(x + w / 2, y0 + 24 + 100 + 12, "switch", "pl", "middle")
            x += w
        for i, r in enumerate((5, 4, 3, 2)):
            f.text(24, y0 + 39 + 25 * i, f"row {r}", "en")
    f.arrow("ctl", (x_start, 330), (x_start + 45 * UNIT, 330))
    f.text(x_start + 45 * UNIT + 8, 333.5, "time", "pl")
    y = 356
    x = x_start
    for cls, s in (("card", "operation"), ("hatch", "idle"), ("solid ctl", "host role switch: drain, write, poll"),
                   ("solid code", "page: ~2.7 µs")):
        f.rect(x, y - 10, 14, 14, cls, 3)
        f.text(x + 22, y + 1, s, "lb")
        x += 22 + text_width(s, 10.5) + 28
    f.text(856, y + 1, "schematic, not to scale", "en", "end")
    return f


FIGURES = {"classic": classic, "overlay": overlay, "budget": budget, "steps": steps, "paths": paths,
           "packets": packets, "memtile": memtile, "timeline": timeline}


def main() -> None:
    for name, fn in FIGURES.items():
        f = fn()
        for suffix, theme in THEMES.items():
            (IMAGES / f"paging-{name}{suffix}.svg").write_text(f.svg(theme, PANEL[suffix]))


if __name__ == "__main__":
    main()
