"""The README's performance charts: docs/images/<chart>.svg and <chart>-dark.svg from docs/data/*.json.

    python docs/charts.py

artlite.json holds Artlite's measurements. Every other file in docs/data is a runtime Artlite is compared with, in
the same schema; its rows are drawn next to the Artlite row of the same model whose quant comes first in their
comparable_quants (default: the row's own quant), and left out when no Artlite row matches. same_shapes_as charts a
model under another model's group (same architecture and sizes, other weights); a row's own version (one measured
with another release than its file's) is in its label, and its notes are footnotes as Artlite's. The image-ttft chart
draws the rows with image_ttft_s (per vision encoder and image) under artlite.json's image_prompts: a group per image,
its glyph the image's outline cut into its slices, all glyphs to one scale. Standard library only: the SVGs use the
platform's sans-serif font, and text widths are estimated from Helvetica's metrics.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parent
DATA = DOCS / "data"
IMAGES = DOCS / "images"

WIDTH = 880
PAD = 24
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
# color order: a runtime keeps its color whoever else is drawn
RUNTIMES = ("Artlite", "FastFlowLM", "llama.cpp", "OpenFlowLM")
NOTE_MARKS = "*†‡§¶"


@dataclass(frozen=True)
class Theme:
    surface: str
    border: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    guide: str
    # RUNTIMES' colors: categorical slots 1-3 and a dark amber for FastFlowLM's fork, all pairs validated on this
    # surface
    series: tuple[str, ...]
    tint: str  # Artlite with its drafter (MTP)


THEMES = {
    "": Theme(surface="#ffffff", border="#d1d9e0", ink="#1f2328", ink2="#59636e", muted="#818b98", grid="#eef1f4",
              axis="#c8d1da", guide="#d5dce3", series=("#2a78d6", "#eb6834", "#1baf7a", "#8f4801"),
              tint="#86b6ef"),
    "-dark": Theme(surface="#0d1117", border="#3d444d", ink="#f0f6fc", ink2="#9198a1", muted="#6e7681",
                   grid="#1b2129", axis="#3d444d", guide="#2d333b",
                   series=("#3987e5", "#d95926", "#199e70", "#8f4801"), tint="#1c5cab"),
}

# Helvetica advance widths (1/1000 em), for layout only
_WIDTHS = {c: w for chars, w in (
    ("ijl", 222), (" .,:;!ft/'|I[]·", 278), ("r-()", 333), ('*"', 389), ("cksvxyzJ?", 500),
    ("0123456789abdeghnopqu_L$†‡", 556), ("+=×<>~≈", 584), ("FTZ", 611), ("ABEKPSVXY&", 667), ("CDHNRUw", 722),
    ("GOQ", 778), ("mM", 833), ("%", 889), ("W", 944), ("—→…", 1000)) for c in chars}


def text_width(s: str, size: float, bold: bool = False) -> float:
    return sum(_WIDTHS.get(c, 600) for c in s) * size / 1000 * (1.07 if bold else 1.0)


def sig3(v: float) -> str:
    """3 significant digits, trailing zeros kept: 0.110, 1.07, 28.7, 112."""
    return f"{v:.0f}" if v >= 100 else f"{v:#.3g}"


def tokens(n: int) -> str:
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def kilo(v: float) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def times(r: float) -> str:
    return f"{r:.1f}×" if r < 10 else f"{r:.0f}×"


def speedup(r: float) -> str:
    """Artlite's speed over another runtime's."""
    if times(r) == "1.0×" or times(1 / r) == "1.0×":
        return "Artlite on par"
    return f"Artlite {times(r)} faster" if r > 1 else f"Artlite {times(1 / r)} slower"


def nice_step(span: float, target: int) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


# ---- data


@dataclass(eq=False)
class Row:
    runtime: str
    group: str  # the chart group: the model, or the model whose shapes it has
    raw: dict
    partner: Row | None = None  # a comparison row's Artlite row

    @property
    def quant(self) -> str:
        return self.raw["quant"]

    @property
    def version_quant(self) -> str:
        """The quant, after the row's own version if it has one."""
        return f"{self.raw['version'].split()[0]} {self.quant}" if "version" in self.raw else self.quant

    @property
    def label(self) -> str:
        return self.quant if self.runtime == "Artlite" else f"{self.runtime} {self.version_quant}"

    def get(self, key: str, n: int) -> float | None:
        return self.raw.get(key, {}).get(str(n))

    def mtp(self, key: str, n: int) -> float | None:
        return self.raw.get("mtp", {}).get(key, {}).get(str(n))

    def shapes_note(self) -> str | None:
        if self.group == self.raw["model"]:
            return None
        return f"{self.raw['model']}: the shapes of {self.group}, other weights"


@dataclass
class Data:
    rows: list[Row]  # Artlite's
    comparisons: list[Row]  # other runtimes' rows that have a partner
    groups: list[str]  # in artlite.json's order
    legend: dict[str, str]  # runtime -> legend label
    image_prompts: dict  # artlite.json's: the question, the vision encoders and the images of image_ttft_s

    def color(self, runtime: str) -> int:
        return RUNTIMES.index(runtime) if runtime in RUNTIMES else len(RUNTIMES) - 1

    def paired(self, row: Row) -> list[Row]:
        return [c for c in self.comparisons if c.partner is row]


def load() -> Data:
    art = json.loads((DATA / "artlite.json").read_text())
    rows = [Row("Artlite", r["model"], r) for r in art["results"]]
    groups = list(dict.fromkeys(r.group for r in rows))
    legend = {"Artlite": "Artlite"}
    others = [json.loads(p.read_text()) for p in sorted(DATA.glob("*.json")) if p.name != "artlite.json"]
    others.sort(key=lambda d: RUNTIMES.index(d["runtime"]) if d["runtime"] in RUNTIMES else len(RUNTIMES))
    comparisons = []
    for other in others:
        name = other["runtime"]
        legend[name] = f"{name} {other['version'].split()[0]}" if "version" in other else name
        for r in other["results"]:
            row = Row(name, r.get("same_shapes_as", r["model"]), r)
            candidates = [a for a in rows if a.group == row.group]
            for quant in r.get("comparable_quants", [r["quant"]]):
                row.partner = next((a for a in candidates if a.quant == quant), None)
                if row.partner:
                    comparisons.append(row)
                    break
            else:
                print(f"charts: no Artlite row to compare {name} {r['model']} {r['quant']} with", file=sys.stderr)
    return Data(rows, comparisons, groups, legend, art.get("image_prompts", {}))


# ---- SVG


@dataclass
class Notes:
    """Footnotes: a mark per distinct text, in order of first use."""
    texts: list[str] = field(default_factory=list)

    def mark(self, text: str | None) -> str:
        if text is None:
            return ""
        if text not in self.texts:
            self.texts.append(text)
        return NOTE_MARKS[self.texts.index(text)]


class Svg:
    def __init__(self, title: str, subtitle: list[str]) -> None:
        self.title, self.desc, self.parts = title, " ".join(subtitle), []
        self.height = 0.0
        self.text(PAD, PAD + 16, title, "title")
        self.y = PAD + 16  # the layout cursor
        for s in subtitle:
            self.y += 19
            self.text(PAD, self.y, s, "sub")

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, cls: str = "", anchor: str = "start", extra: str = "") -> None:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{a}{c}{extra}>{escape(s)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def legend(self, items: list[tuple[str, str, str]]) -> None:
        """items: (swatch kind: box, line or band; its class; label)."""
        x, y = PAD, self.y + 30
        for kind, cls, label in items:
            if kind == "box":
                self.add(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="3" class="{cls}"/>')
                x += 18
            elif kind == "band":
                self.add(f'<rect x="{x}" y="{y - 9}" width="20" height="12" rx="2" class="{cls}"/>')
                x += 26
            else:
                self.line(x, y - 3, x + 20, y - 3, f"l{cls}")
                self.add(f'<circle cx="{x + 10}" cy="{y - 3}" r="3.5" class="d{cls}"/>')
                x += 26
            self.text(x, y, label, "lbl")
            x += text_width(label, 12) + 22
        self.y = y

    def footer(self, notes: Notes) -> None:
        """The footnotes under the chart (how the numbers are measured: docs/methodology.md)."""
        for i, t in enumerate(notes.texts):
            self.text(PAD, self.y, f"{NOTE_MARKS[i]} {t}")
            self.y += 16
        self.height = self.y + PAD - (12 if notes.texts else 30)

    def render(self, theme: Theme) -> str:
        t = theme
        series = "".join(
            f".f{i}{{fill:{c}}}.l{i}{{stroke:{c};stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}}"
            f".d{i}{{fill:{c};stroke:{t.surface};stroke-width:1.5}}.b{i}{{fill:{c};fill-opacity:.16}}"
            for i, c in enumerate(t.series))
        style = (
            f"text{{font-family:{FONT};font-size:11px;fill:{t.ink2};font-variant-numeric:tabular-nums}}"
            f".title{{font-size:18px;font-weight:600;fill:{t.ink}}}.sub{{font-size:12.5px}}"
            f".h{{font-size:12.5px;font-weight:600;fill:{t.ink}}}.lbl{{font-size:12px;fill:{t.ink}}}"
            f".lbl2{{font-size:12px}}.strong{{font-weight:600;fill:{t.ink}}}.muted{{fill:{t.muted}}}"
            f".small{{font-size:10px}}.halo{{paint-order:stroke;stroke:{t.surface};stroke-width:3px;"
            f"stroke-linejoin:round}}.grid{{stroke:{t.grid}}}.axis{{stroke:{t.axis}}}.guide{{stroke:{t.guide}}}"
            f".bracket{{stroke:{t.ink2};fill:none}}.ft{{fill:{t.tint}}}"
            f".detail{{font-size:12px;font-weight:400;fill:{t.ink2}}}.tile{{fill:{t.grid};stroke:{t.ink2};stroke-width:1}}.cut{{stroke:{t.ink2};stroke-width:.75;fill:none}}"
            f"{series}")
        h = math.ceil(self.height)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{h}" viewBox="0 0 {WIDTH} {h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{escape(self.title)}</title>\n'
            f'<desc id="d">{escape(self.desc)}</desc>\n<style>{style}</style>\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{h - 1}" rx="8" fill="{t.surface}" '
            f'stroke="{t.border}"/>\n' + "\n".join(self.parts) + "\n</svg>\n")


def bar(x: float, y: float, w: float, h: float, cls: str, tip: str) -> str:
    """A bar from the baseline x, square there, with a 4 px rounded data end."""
    r = min(4.0, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}a{r},{r} 0 0 1 {-r},{r}'
            f'h{r - w:.1f}z" class="{cls}"><title>{escape(tip)}</title></path>')


# ---- bar charts: one row per Artlite file, comparisons right under their partner


@dataclass
class BarRow:
    label: str
    cls: str
    values: dict[int | str, float | None]  # panel -> value
    tip: str
    marks: dict[int | str, str] = field(default_factory=dict)  # panel -> footnote mark of the value
    suffix: dict[int | str, str] = field(default_factory=dict)  # panel -> muted text after the value
    partner: BarRow | None = None


@dataclass
class Heading:
    """A group of bars: its name, muted details after it, and for an image its glyph (``tiles``: width and height in
    pixels, rows and columns of its slices)."""
    text: str
    detail: str = ""
    tiles: tuple[int, int, int, int] | None = None


GLYPH = 40.0  # an image glyph's longest side in pixels: the largest image's (the others to the same scale)


def glyph(x: float, y: float, w: float, h: float, rows: int, cols: int, tip: str) -> str:
    """An image as its outline, cut into its slices."""
    cuts = "".join(f"M{x + w * c / cols:.1f},{y:.1f}v{h:.1f}" for c in range(1, cols))
    cuts += "".join(f"M{x:.1f},{y + h * r / rows:.1f}h{w:.1f}" for r in range(1, rows))
    return (f'<g><title>{escape(tip)}</title><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="1.5" class="tile"/>' + (f'<path d="{cuts}" class="cut"/>' if cuts else "") + "</g>")


def draw_bars(svg: Svg, items: list[Heading | BarRow], panels: list[tuple[int | str, str]],
              fmt: Callable[[float], str], *, lower_better: bool = False, ticks: int = 4) -> None:
    """Panels (key, heading) side by side on one scale with about ``ticks`` grid lines: a row per bar under its
    group's heading, a comparison right under its partner with how much faster Artlite is (``lower_better``: the
    values are times)."""
    bars = [it for it in items if isinstance(it, BarRow)]
    label_w = max(text_width(b.label, 12) for b in bars) + 14
    gap = 30
    plot_w = (WIDTH - 2 * PAD - label_w - (len(panels) - 1) * gap) / len(panels)
    panel_x = [PAD + label_w + i * (plot_w + gap) for i in range(len(panels))]

    def parts(b: BarRow, n: int | str) -> tuple[str, str, str]:
        """The value label: number (with its footnote mark), muted suffix, speedup."""
        v = b.values[n]
        if v is None:
            return "—" + b.marks.get(n, ""), "", ""
        pv = b.partner.values[n] if b.partner else None
        fast = speedup(v / pv if lower_better else pv / v) if pv else ""
        return fmt(v) + b.marks.get(n, ""), b.suffix.get(n, ""), fast

    def label_width(b: BarRow, n: int | str) -> float:
        num, suffix, fast = parts(b, n)
        return (text_width(num, 11) + (6 + text_width(suffix, 11) if suffix else 0)
                + (10 + text_width(fast, 11, bold=True) if fast else 0))

    # one scale for all panels, the largest that still fits every bar's label
    scale = min((plot_w - 6 - label_width(b, n)) / b.values[n] for b in bars for n, _ in panels if b.values[n])
    step = nice_step(plot_w / scale, ticks)
    grid = [i * step for i in range(int(plot_w / scale / step + 1e-9) + 1)]
    sizes = [max(h.tiles[:2]) for h in items if isinstance(h, Heading) and h.tiles]
    px_per_pixel = GLYPH / max(sizes) if sizes else 0.0
    glyph_h = max((h.tiles[1] * px_per_pixel for h in items if isinstance(h, Heading) and h.tiles), default=0.0)

    y = svg.y + 34
    for i, (_, heading) in enumerate(panels):
        svg.text(panel_x[i], y, heading, "h")
    y += 12
    rows_top = y
    body: list[str] = []
    bh = 13
    for k, it in enumerate(items):
        if isinstance(it, Heading):
            if k:
                body.append(f'<line x1="{PAD}" y1="{y + 8:.1f}" x2="{WIDTH - PAD}" y2="{y + 8:.1f}" class="grid"/>')
                y += 12
            x, band = PAD, 22.0
            if it.tiles:  # the glyphs centred in a column of the largest's width, the text beside them
                w, h, rows, cols = it.tiles
                band = glyph_h + 6
                gw, gh = w * px_per_pixel, h * px_per_pixel
                body.append(glyph(PAD + (GLYPH - gw) / 2, y + 4 + (glyph_h - gh) / 2, gw, gh, rows, cols,
                                  f"{it.text}: {w} × {h} pixels, {rows} × {cols} slices"))
                x += GLYPH + 10
            base = y + 4 + glyph_h / 2 + 4.5 if it.tiles else y + 14
            detail = f'<tspan dx="8" class="detail">{escape(it.detail)}</tspan>' if it.detail else ""
            body.append(f'<text x="{x:g}" y="{base:.1f}" class="h">{escape(it.text)}{detail}</text>')
            y += band
            continue
        if it.partner is not None:
            y -= 4  # 2 px under its partner
        base = y + bh / 2 + 4
        body.append(f'<text x="{PAD + label_w - 10:.1f}" y="{base:.1f}" text-anchor="end" '
                    f'class="{"lbl2" if it.partner else "lbl"}">{escape(it.label)}</text>')
        for i, (n, heading) in enumerate(panels):
            v = it.values[n]
            w = max(v * scale, 1.0) if v is not None else 0.0
            if v is not None:
                body.append(bar(panel_x[i], y, w, bh, it.cls, f"{it.tip}, {heading}: {fmt(v)}"))
            num, suffix, fast = parts(it, n)
            spans = escape(num)
            if suffix:
                spans += f'<tspan dx="6" class="muted">{escape(suffix)}</tspan>'
            if fast:
                spans += f'<tspan dx="10" class="strong">{escape(fast)}</tspan>'
            body.append(f'<text x="{panel_x[i] + w + 5:.1f}" y="{base:.1f}">{spans}</text>')
        y += bh + 6
    for x0 in panel_x:
        for t in grid:
            svg.line(x0 + t * scale, rows_top, x0 + t * scale, y, "grid" if t else "axis")
            svg.text(x0 + t * scale, y + 14, kilo(t), "", "middle")
    for s in body:
        svg.add(s)
    svg.y = y + 44


def bar_chart(data: Data, key: str, title: str, subtitle: list[str], contexts: tuple[int, ...],
              fmt: Callable[[float], str]) -> Svg:
    notes = Notes()
    items: list[Heading | BarRow] = []
    shown = {"Artlite"}
    mtp = False
    for g in data.groups:
        group_rows = [r for r in data.rows if r.group == g and any(r.get(key, n) is not None for n in contexts)]
        if group_rows:
            items.append(Heading(g))
        for r in group_rows:
            marks = {int(n): notes.mark(t) for n, t in r.raw.get("notes", {}).get(key, {}).items()}
            main = BarRow(r.quant, f"f{data.color('Artlite')}", {n: r.get(key, n) for n in contexts},
                          f"Artlite, {g} {r.quant}", marks)
            items.append(main)
            for c in data.paired(r):
                if any(c.get(key, n) is not None for n in contexts):
                    shown.add(c.runtime)
                    marks = {int(n): notes.mark(t) for n, t in c.raw.get("notes", {}).get(key, {}).items()}
                    items.append(BarRow(c.label + notes.mark(c.shapes_note()), f"f{data.color(c.runtime)}",
                                        {n: c.get(key, n) for n in contexts},
                                        f"{c.runtime}, {c.raw['model']} {c.quant}", marks, partner=main))
            if any(r.mtp(key, n) is not None for n in contexts):
                mtp = True
                steps = r.raw["mtp"].get("tokens_per_step", {})
                items.append(BarRow(f"{r.quant} + MTP", "ft", {n: r.mtp(key, n) for n in contexts},
                                    f"Artlite with its drafter, {g} {r.quant}",
                                    suffix={n: f"S={steps[str(n)]}" for n in contexts if str(n) in steps}))

    svg = Svg(title, subtitle)
    legend = [("box", f"f{data.color(k)}", v) for k, v in data.legend.items() if k in shown]
    if mtp:
        legend.insert(1, ("box", "ft", "Artlite with its MTP drafter, S tokens per step"))
    svg.legend(legend)
    draw_bars(svg, items, [(n, f"{tokens(n)}-token prompt") for n in contexts], fmt)
    svg.footer(notes)
    return svg


# ---- an image prompt's time to first token: a group per image, a panel per vision encoder


def image_chart(data: Data) -> Svg:
    key = "image_ttft_s"
    prompts = data.image_prompts
    encoders = list(prompts["encoders"].items())
    rows = [r for r in data.rows if r.raw.get(key)]
    notes = Notes()
    items: list[Heading | BarRow] = []
    legend = {"Artlite": "Artlite"}  # (a comparison's own version: the release its images were measured with)

    def values(row: Row, image: str) -> dict[int | str, float | None]:
        return {e: row.raw[key].get(e, {}).get(image) for e, _ in encoders}

    def marks(row: Row) -> dict[int | str, str]:
        return {e: notes.mark(t) for e, t in row.raw.get("notes", {}).get(key, {}).items()}

    for name, im in prompts["images"].items():
        (w, h), (sr, sc) = im["size"], im["slices"]
        cut = f" and {sr * sc} slices" if sr * sc else ""
        items.append(Heading(name, f"{w} × {h}: an overview{cut}", (w, h, max(sr, 1), max(sc, 1))))
        for r in rows:
            main = BarRow("Artlite" if len(rows) == 1 else f"Artlite {r.quant}", f"f{data.color('Artlite')}",
                          values(r, name), f"Artlite, {r.raw['model']} {r.quant}, {name}", marks(r))
            items.append(main)
            for c in data.paired(r):
                if c.raw.get(key):
                    legend[c.runtime] = (f"{c.runtime} {c.raw['version'].split()[0]}" if "version" in c.raw
                                         else data.legend[c.runtime])
                    label = " ".join([c.runtime, *[c.quant] * (c.quant != r.quant), *[c.raw.get("backend", "")]])
                    items.append(BarRow(label.strip(), f"f{data.color(c.runtime)}", values(c, name),
                                        f"{label.strip()}, {c.raw['model']} {c.quant}, {name}", marks(c),
                                        partner=main))

    model = " / ".join(dict.fromkeys(f"{r.raw['model']} {r.quant}" for r in rows))
    svg = Svg("Time to first token with an image, seconds", [
        (f"{model} with an image and “{prompts['question']}”: the vision encoder on each of the image's "
         "entries"),
        ("(an overview and its slices), then the prefill. From the request to the first streamed token, through each "
         "runtime's server."),
    ])
    svg.legend([("box", f"f{data.color(k)}", v) for k, v in legend.items()])
    draw_bars(svg, items, [(e, heading) for e, heading in encoders], sig3, lower_better=True, ticks=6)
    svg.footer(notes)
    return svg


# ---- time to first token: log-log small multiples


def _seg_dist(px: float, py: float, s: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - x1 - t * dx, py - y1 - t * dy)


def _box(x: float, y: float, w: float, anchor: str = "start") -> tuple[float, float, float, float]:
    """The box of an 11 px text at baseline (x, y): x0, y0, x1, y1."""
    x0 = x - w if anchor == "end" else x
    return x0 - 1, y - 10, x0 + w + 1, y + 3


def _box_segments(b: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = b
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def ttft_chart(data: Data) -> Svg:
    key = "ttft_s"
    notes = Notes()
    panels = []  # (group, the drawn Artlite row, all its Artlite rows, comparisons)
    for g in data.groups:
        arts = [r for r in data.rows if r.group == g and r.raw.get(key)]
        if arts:
            cmps = [c for c in data.comparisons if c.partner in arts and c.raw.get(key)]
            main = cmps[0].partner if cmps else arts[0]
            panels.append((g, main, arts, [c for c in cmps if c.partner is main]))

    def pts(r: Row) -> list[tuple[int, float]]:
        return sorted((int(n), t) for n, t in r.raw[key].items())

    every = [p for _, _, arts, cmps in panels for r in [*arts, *cmps] for p in pts(r)]
    nmin, nmax = min(n for n, _ in every), max(n for n, _ in every)
    x_lo, x_hi = math.log10(nmin) - 0.09, math.log10(nmax) + 0.09
    y_lo = math.log10(min(t for _, t in every)) - 0.12
    y_hi = math.log10(max(t for _, t in every)) + 0.12
    rates = [n / t for n, t in every]
    diagonals = range(math.floor(math.log10(min(rates))), math.ceil(math.log10(max(rates))) + 1)

    cols = min(4, len(panels))
    y_label_w, right_w, gap = 40, 46, 12
    plot_w = (WIDTH - 2 * PAD - y_label_w - cols * right_w - (cols - 1) * gap) / cols
    ppd = plot_w / (x_hi - x_lo)  # pixels per decade, the same on both axes
    plot_h = ppd * (y_hi - y_lo)

    svg = Svg("Time to first token", [
        ("Both axes are logarithmic and equally scaled. The 45° diagonals are constant prefill speeds, so a curve's "
         "slope is how"),
        ("its time grows with the prompt (1: linear, 2: quadratic), and the vertical gap between two curves is a "
         "speed ratio."),
    ])
    shown = {c.runtime: c for _, _, _, cmps in panels for c in cmps}
    a = data.color("Artlite")
    svg.legend([("line", str(a), "Artlite"), ("band", f"b{a}", "Artlite, the range of its quants")]
               + [("line", str(data.color(k)), data.legend[k]) for k in shown])
    top = svg.y + 22

    for i, (g, main, arts, cmps) in enumerate(panels):
        col, row = i % cols, i // cols
        px = PAD + y_label_w + col * (plot_w + right_w + gap)
        py = top + row * (plot_h + 96) + 40

        def X(n: float, px: float = px) -> float:
            return px + (math.log10(n) - x_lo) * ppd

        def Y(t: float, py: float = py) -> float:
            return py + plot_h - (math.log10(t) - y_lo) * ppd

        svg.text(px, py - 26, g, "h")
        x = px  # each curve's quant, after a dot of its color
        for r in [main, *cmps]:
            s = r.version_quant + notes.mark(r.shapes_note())
            svg.add(f'<circle cx="{x + 3.5:.1f}" cy="{py - 14:.1f}" r="3.5" class="d{data.color(r.runtime)}"/>')
            svg.text(x + 10, py - 10, s)
            x += 10 + text_width(s, 11) + 12
        for d in range(math.ceil(y_lo), math.floor(y_hi) + 1):
            svg.line(px, Y(10 ** d), px + plot_w, Y(10 ** d), "grid")
            if col == 0:
                svg.text(px - 6, Y(10 ** d) + 4, f"{10 ** d:g} s", "", "end")
        guides = []  # time = n / 10^e, clipped to the plot
        for e in diagonals:
            lo, hi = max(x_lo, y_lo + e), min(x_hi, y_hi + e)
            if hi - lo > 0.2:
                guides.append((e, X(10 ** lo), Y(10 ** (lo - e)), X(10 ** hi), Y(10 ** (hi - e))))
                svg.line(*guides[-1][1:], "guide")
        svg.line(px, py + plot_h, px + plot_w, py + plot_h, "axis")
        svg.line(px, py, px, py + plot_h, "axis")
        ticks = [2 ** k for k in range(math.ceil(math.log2(nmin)), math.floor(math.log2(nmax)) + 1)]
        every_other = ppd * math.log10(2) < 26
        for k, n in enumerate(ticks):
            svg.line(X(n), py + plot_h, X(n), py + plot_h + 4, "axis")
            if not every_other or k % 2 == 0:
                svg.text(X(n), py + plot_h + 16, tokens(n), "", "middle")

        # Artlite's quants as a band, then the curves
        obstacles: list[tuple[float, float, float, float]] = []
        common = sorted(set.intersection(*(set(map(int, r.raw[key])) for r in arts)))
        if len(arts) > 1 and len(common) > 1:
            edge = [(n, max(r.get(key, n) for r in arts)) for n in common]
            edge += [(n, min(r.get(key, n) for r in arts)) for n in reversed(common)]
            svg.add(f'<polygon points="{" ".join(f"{X(n):.1f},{Y(t):.1f}" for n, t in edge)}" class="b{a}"/>')
        labels = []  # boxes of the texts placed so far
        for r in [*cmps, main]:
            c = data.color(r.runtime)
            p = [(X(n), Y(t)) for n, t in pts(r)]
            obstacles += [(*u, *v) for u, v in pairwise(p)] + [(*u, *u) for u in p]
            svg.add(f'<polyline points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in p)}" class="l{c}"/>')
            for (n, t), (u, v) in zip(pts(r), p):
                svg.add(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="3.5" class="d{c}"><title>'
                        f'{escape(f"{r.runtime} {r.quant}, {tokens(n)} tokens: {sig3(t)} s")}</title></circle>')
            s = f"{sig3(pts(r)[-1][1])} s"
            labels.append((p[-1][0] + 7, p[-1][1] + 4, s, _box(p[-1][0] + 7, p[-1][1] + 4, text_width(s, 11))))
        labels.sort(key=lambda b: b[1])
        for k in range(1, len(labels)):  # end labels at the same length: at least a line apart
            (x0, y0, _, _), (x1, y1, s, _) = labels[k - 1], labels[k]
            if abs(x0 - x1) < 30 and y1 - y0 < 13:
                labels[k] = (x1, y0 + 13, s, _box(x1, y0 + 13, text_width(s, 11)))
        for x, y, s, _ in labels:
            svg.text(x, y, s, "halo")
        # the ratio at the shortest and the longest prompt both runtimes measured
        shared = sorted(set(main.raw[key]) & set(cmps[0].raw[key]), key=int) if cmps else []
        for n in dict.fromkeys(shared[:1] + shared[-1:]):
            x, ya, yc = X(int(n)), Y(main.raw[key][n]), Y(cmps[0].raw[key][n])
            svg.add(f'<path d="M{x - 3:.1f},{ya - 6:.1f}h6M{x:.1f},{ya - 6:.1f}V{yc + 6:.1f}M{x - 3:.1f},'
                    f'{yc + 6:.1f}h6" class="bracket"/>')
            obstacles.append((x, ya - 6, x, yc + 6))
            s = times(cmps[0].raw[key][n] / main.raw[key][n])
            w = text_width(s, 12)
            right = _box(x + 7, (ya + yc) / 2 + 4, w)
            anchor = "end" if any(_overlap(right, b) for *_, b in labels) else "start"
            lx = x - 7 if anchor == "end" else x + 7
            svg.text(lx, (ya + yc) / 2 + 4, s, "lbl halo", anchor)
            labels.append((lx, 0, s, _box(lx, (ya + yc) / 2 + 4, w, anchor)))
        obstacles += [seg for *_, b in labels for seg in _box_segments(b)]
        # each diagonal's label on the side and at the place farthest from the data and the other labels; none
        # where nothing is clear (the diagonals are at the same places in every panel)
        for e, x1, y1, x2, y2 in guides:
            s = f"{kilo(10 ** e)} tok/s"
            w, length = text_width(s, 10), math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            nx, ny = uy, -ux  # the normal on the upper side
            best = None
            for f in (k / 40 for k in range(41)):
                if f * length < w / 2 + 4 or (1 - f) * length < w / 2 + 4:
                    continue
                for base in (2, -10):  # the baseline's offset: the text above or below the line
                    cx, cy = x1 + f * (x2 - x1) + base * nx, y1 + f * (y2 - y1) + base * ny
                    samples = [(cx + d * ux + h * nx, cy + d * uy + h * ny)
                               for d in (-w / 2, -w / 4, 0, w / 4, w / 2) for h in (0, 4, 8)]
                    if not all(px <= u <= px + plot_w and py <= v <= py + plot_h for u, v in samples):
                        continue
                    score = min(_seg_dist(u, v, o) for u, v in samples for o in obstacles) + f * 0.5
                    if best is None or score > best[0]:
                        best = (score, cx, cy)
            if best and best[0] >= 4:
                _, cx, cy = best
                angle = math.degrees(math.atan2(uy, ux))
                svg.text(cx, cy, s, "muted small halo", "middle", f' transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"')
        if col == cols - 1 or i == len(panels) - 1:
            svg.text((PAD + y_label_w + px + plot_w) / 2, py + plot_h + 36, "prompt length, tokens", "muted",
                     "middle")

    svg.y = top + math.ceil(len(panels) / cols) * (plot_h + 96) + 8
    svg.footer(notes)
    return svg


def main() -> None:
    data = load()
    pairing = "Where another runtime runs the same model with comparable weights, its bar sits under Artlite's."
    charts = {
        "decode": bar_chart(
            data, "decode_tps", "Decode speed, tokens/s",
            ["Generation speed after a prompt of 1k and of 32k tokens, greedy: one bar per GGUF file.", pairing],
            (1024, 32768), lambda v: f"{v:.1f}"),
        "prefill": bar_chart(
            data, "prefill_tps", "Prefill speed, tokens/s",
            ["Prompt tokens processed per second (prompt length / time to the first token), one bar per GGUF file.",
             pairing],
            (1024, 32768), lambda v: f"{v:.0f}"),
        "ttft": ttft_chart(data),
        "image-ttft": image_chart(data),
    }
    IMAGES.mkdir(exist_ok=True)
    for name, svg in charts.items():
        for suffix, theme in THEMES.items():
            (IMAGES / f"{name}{suffix}.svg").write_text(svg.render(theme))


if __name__ == "__main__":
    main()
