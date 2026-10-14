"""Generate ManySeg logo SVGs (light/dark) with the wordmark converted to outlines."""
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(sys.argv[1])
FONT = TTFont("/usr/share/fonts/truetype/lato/Lato-Bold.ttf")
GS = FONT.getGlyphSet()
CMAP = FONT.getBestCmap()
UPM = FONT["head"].unitsPerEm
hmtx = FONT["hmtx"]

def text_path(s, x, baseline, size, tracking=0.0):
    scale = size / UPM
    pen = SVGPathPen(GS)
    for ch in s:
        g = CMAP[ord(ch)]
        tp = TransformPen(pen, (scale, 0, 0, -scale, x, baseline))
        GS[g].draw(tp)
        x += hmtx[g][0] * scale + tracking * size
    return pen.getCommands(), x

# 4x4 segmentation mask: 0 = background, 1 = class A, 2 = class B
MASK = [
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [2, 2, 1, 0],
    [2, 2, 2, 0],
]

THEMES = {
    "light": dict(ink="#0F172A", bg="#E2E8F0", a="#0D9488", b="#6366F1", seg="#6366F1",
                  layer1="#CBD5E1", layer2="#E2E8F0", plate="#F8FAFC", edge="#CBD5E1"),
    "dark":  dict(ink="#F1F5F9", bg="#334155", a="#2DD4BF", b="#818CF8", seg="#818CF8",
                  layer1="#475569", layer2="#334155", plate="#1E293B", edge="#475569"),
}

CELL, GAP, R = 22, 4, 5
INNER = 4 * CELL + 3 * GAP         # 100
PP = 6                              # plate padding around the cells
GRID = INNER + 2 * PP
OFF = 10                            # offset between stacked layers
PAD = 8

def mark(t, ox, oy):
    parts = []
    # two back layers = "many" input channels, stacked up-right
    for i, col in ((2, t["layer2"]), (1, t["layer1"])):
        parts.append(f'<rect x="{ox + i*OFF}" y="{oy - i*OFF}" width="{GRID}" height="{GRID}" rx="{R+3}" fill="{col}"/>')
    # front layer: the predicted mask on its own plate
    parts.append(f'<rect x="{ox+0.75}" y="{oy+0.75}" width="{GRID-1.5}" height="{GRID-1.5}" rx="{R+3}" fill="{t["plate"]}" stroke="{t["edge"]}" stroke-width="1.5"/>')
    ox, oy = ox + PP, oy + PP
    fill = {0: t["bg"], 1: t["a"], 2: t["b"]}
    for r, row in enumerate(MASK):
        for c, v in enumerate(row):
            parts.append(f'<rect x="{ox + c*(CELL+GAP)}" y="{oy + r*(CELL+GAP)}" width="{CELL}" height="{CELL}" rx="{R}" fill="{fill[v]}"/>')
    return "\n  ".join(parts)

def build(theme, with_text=True):
    t = THEMES[theme]
    ox, oy = PAD, PAD + 2*OFF
    m = mark(t, ox, oy)
    mark_w = PAD + GRID + 2*OFF
    h = oy + GRID + PAD
    if with_text:
        h += 6   # room for the "y"/"g" descenders
    if not with_text:
        w = mark_w + PAD
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="ManySeg">\n  {m}\n</svg>\n'
    size = 72
    baseline = oy + GRID/2 + size*0.34
    tx = mark_w + 28
    many, x = text_path("Many", tx, baseline, size, tracking=-0.01)
    seg, x = text_path("Seg", x, baseline, size, tracking=-0.01)
    w = int(x + PAD + 4)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="ManySeg">\n'
            f'  {m}\n  <path fill="{t["ink"]}" d="{many}"/>\n  <path fill="{t["seg"]}" d="{seg}"/>\n</svg>\n')

for theme in THEMES:
    (OUT / f"logo-{theme}.svg").write_text(build(theme))
    (OUT / f"icon-{theme}.svg").write_text(build(theme, with_text=False))
print("ok")
