"""Generate the README architecture diagram (light/dark SVGs, text converted to outlines).

Usage: uv run --no-project --with fonttools python scripts/make_architecture.py docs/assets
"""
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(sys.argv[1])
FONTS = {}
for weight, file in (("regular", "Lato-Regular.ttf"), ("bold", "Lato-Bold.ttf"), ("mono", None)):
    if file:
        f = TTFont(f"/usr/share/fonts/truetype/lato/{file}")
        FONTS[weight] = (f, f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm, f["hmtx"])
FONTS["mono"] = FONTS["regular"]


def text_width(s, size, weight="regular", tracking=0.0):
    _, _, cmap, upm, hmtx = FONTS[weight]
    return sum(hmtx[cmap[ord(c)]][0] * size / upm + tracking * size for c in s)


def text(s, x, y, size, fill, weight="regular", anchor="start", tracking=0.0):
    _, gs, cmap, upm, hmtx = FONTS[weight]
    if anchor == "middle":
        x -= text_width(s, size, weight, tracking) / 2
    elif anchor == "end":
        x -= text_width(s, size, weight, tracking)
    scale = size / upm
    pen = SVGPathPen(gs)
    for c in s:
        g = cmap[ord(c)]
        gs[g].draw(TransformPen(pen, (scale, 0, 0, -scale, x, y)))
        x += hmtx[g][0] * scale + tracking * size
    return f'<path fill="{fill}" d="{pen.getCommands()}"/>'


THEMES = {
    "light": dict(ink="#0F172A", muted="#64748B", line="#94A3B8", plate="#F8FAFC", edge="#CBD5E1",
                  soft="#E2E8F0", a="#0D9488", a_soft="#CCFBF1", b="#6366F1", b_soft="#E0E7FF",
                  c="#F59E0B", c_soft="#FEF3C7", d="#F43F5E", bg="#FFFFFF"),
    "dark": dict(ink="#F1F5F9", muted="#CBD5E1", line="#7C8BA1", plate="#1E293B", edge="#475569",
                 soft="#334155", a="#2DD4BF", a_soft="#134E4A", b="#818CF8", b_soft="#312E81",
                 c="#FBBF24", c_soft="#78350F", d="#FB7185", bg="#0D1117"),
}

W, H = 1000, 590
META_Y = 410
RULE_Y = 492
COLS = {"in": 95, "enc": 290, "fuse": 490, "dec": 680, "out": 885}
BRANCH_Y = {"a": 130, "b": 290}
LEVELS = 4
# fusion ports centred between the two branches (y = 210) so their curves mirror each other
FUSE_PORTS = [168, 196, 224, 252]
DEC_Y = FUSE_PORTS


def build(theme):
    t = THEMES[theme]
    el = []
    add = el.append

    # column headers
    for key, label in (("in", "INPUTS"), ("enc", "ENCODERS"), ("fuse", "FUSION"),
                       ("dec", "DECODER"), ("out", "HEADS")):
        add(text(label, COLS[key], 34, 12.5, t["muted"], "bold", "middle", tracking=0.12))

    # ---- inputs: stacked channel tiles; sar is smaller (half resolution)
    def stack(cx, cy, size, n, col, soft):
        off = 7
        x0 = cx - size / 2 - (n - 1) * off / 2
        y0 = cy - size / 2 + (n - 1) * off / 2
        for i in reversed(range(n)):
            fill = col if i == 0 else soft
            add(f'<rect x="{x0 + i*off:.1f}" y="{y0 - i*off:.1f}" width="{size}" height="{size}" '
                f'rx="7" fill="{fill}" stroke="{t["bg"]}" stroke-width="1.5"/>')

    stack(COLS["in"], BRANCH_Y["a"], 68, 4, t["a"], t["a_soft"])
    stack(COLS["in"], BRANCH_Y["b"], 42, 2, t["b"], t["b_soft"])
    add(text("optical", COLS["in"], BRANCH_Y["a"] + 62, 15, t["ink"], "bold", "middle"))
    add(text("4 ch · full res", COLS["in"], BRANCH_Y["a"] + 80, 12.5, t["muted"], anchor="middle"))
    add(text("sar", COLS["in"], BRANCH_Y["b"] + 50, 15, t["ink"], "bold", "middle"))
    add(text("2 ch · ½ res", COLS["in"], BRANCH_Y["b"] + 68, 12.5, t["muted"], anchor="middle"))

    # ---- encoders: feature pyramids (wide = shallow, narrow = deep)
    def pyramid_bars(cx, cy, widths, col):
        ys = []
        for i, w in enumerate(widths):
            y = cy - (LEVELS - 1) * 14 + i * 28
            add(f'<rect x="{cx - w/2:.1f}" y="{y - 7}" width="{w}" height="14" rx="4" fill="{col}"/>')
            ys.append((cx + w / 2, y))
        return ys

    enc_w = [104, 82, 60, 38]
    ends = {}
    for br, name, col in (("a", "convnext_tiny", t["a"]), ("b", "resnet18", t["b"])):
        cy = BRANCH_Y[br]
        # input -> encoder arrow
        x_from = COLS["in"] + (52 if br == "a" else 36)
        add(f'<path d="M{x_from} {cy} L{COLS["enc"] - 64} {cy}" stroke="{t["line"]}" stroke-width="1.5" '
            f'fill="none" marker-end="url(#arrow)"/>')
        ends[br] = pyramid_bars(COLS["enc"], cy, enc_w, col)
        add(text(name, COLS["enc"], cy + 70, 13, t["muted"], anchor="middle"))

    # multi-scale features from both encoders converge on the fusion block
    fx0, fx1 = COLS["fuse"] - 48, COLS["fuse"] + 48
    for br in ("a", "b"):
        col = t["a"] if br == "a" else t["b"]
        # straight lead-out from each bar to a common x, so every curve has the same shape and
        # the two branches mirror each other
        cx0 = COLS["enc"] + max(enc_w) / 2 + 4
        mx = (cx0 + fx0) / 2
        for (x, y), py in zip(ends[br], FUSE_PORTS):
            add(f'<path d="M{x + 4:.1f} {y} L{cx0:.1f} {y} C{mx:.1f} {y}, {mx:.1f} {py}, {fx0} {py}" '
                f'stroke="{col}" stroke-width="1.5" stroke-opacity="0.55" fill="none"/>')

    # ---- fusion block
    add(f'<rect x="{fx0}" y="138" width="{fx1 - fx0}" height="144" rx="12" fill="{t["plate"]}" '
        f'stroke="{t["edge"]}" stroke-width="1.5"/>')
    for py in FUSE_PORTS:
        add(f'<circle cx="{fx0}" cy="{py}" r="3" fill="{t["line"]}"/>')
        add(f'<circle cx="{fx1}" cy="{py}" r="3" fill="{t["line"]}"/>')
    add(text("gated", COLS["fuse"], 215, 16, t["ink"], "bold", "middle"))
    add(text("per level", COLS["fuse"], 233, 12, t["muted"], anchor="middle"))

    # ---- decoder: takes fused multi-scale features, emits full-res features
    dec_w = [104, 82, 60, 38]
    dx = COLS["dec"]
    for i, (w, y) in enumerate(zip(dec_w, DEC_Y)):
        add(f'<path d="M{fx1} {y} L{dx - w/2 - 4:.1f} {y}" stroke="{t["line"]}" stroke-width="1.5" fill="none"/>')
        add(f'<rect x="{dx - w/2:.1f}" y="{y - 7}" width="{w}" height="14" rx="4" fill="{t["ink"]}" '
            f'fill-opacity="{0.85 - i*0.15:.2f}"/>')
    add(text("unet", dx, 296, 16, t["ink"], "bold", "middle"))
    add(text("+ fpn for boundary", dx, 314, 12, t["muted"], anchor="middle"))

    # ---- heads
    hx = COLS["out"]
    # instance output: each colour is a separate object
    mask = [[0, 1, 1, 0, 2], [1, 1, 1, 0, 2], [0, 1, 0, 2, 2], [3, 3, 0, 2, 2], [3, 3, 3, 0, 0]]
    fill = {0: t["soft"], 1: t["a"], 2: t["b"], 3: t["d"]}
    n, cell, gap = 5, 11, 2.5
    gw = n * cell + (n - 1) * gap
    gx, gy = hx - gw / 2, 92
    add(f'<rect x="{gx - 5}" y="{gy - 5}" width="{gw + 10}" height="{gw + 10}" rx="8" fill="{t["plate"]}" '
        f'stroke="{t["edge"]}" stroke-width="1.5"/>')
    for r, row in enumerate(mask):
        for c, v in enumerate(row):
            add(f'<rect x="{gx + c*(cell+gap)}" y="{gy + r*(cell+gap)}" width="{cell}" height="{cell}" rx="3" fill="{fill[v]}"/>')
    add(text("water bodies", hx, gy + gw + 26, 15, t["ink"], "bold", "middle"))
    add(text("extent · boundary · distance", hx, gy + gw + 43, 12.5, t["muted"], anchor="middle"))

    px, py = hx, 282
    add(f'<rect x="{px - 34}" y="{py - 17}" width="68" height="34" rx="17" fill="{t["plate"]}" '
        f'stroke="{t["edge"]}" stroke-width="1.5"/>')
    add(text("0.37", px, py + 6, 16, t["ink"], "bold", "middle"))
    add(text("water_area", hx, py + 40, 15, t["ink"], "bold", "middle"))
    add(text("regression · per image", hx, py + 57, 12.5, t["muted"], anchor="middle"))

    # decoder -> heads
    sx = dx + 58
    add(f'<path d="M{sx} {DEC_Y[0]} C{sx + 40} {DEC_Y[0]}, {hx - 90} {gy + gw/2}, {gx - 12} {gy + gw/2}" '
        f'stroke="{t["line"]}" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>')
    add(f'<path d="M{sx} {DEC_Y[0]} C{sx + 40} {DEC_Y[0]}, {hx - 90} {py}, {px - 40} {py}" '
        f'stroke="{t["line"]}" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>')

    # ---- metadata vector: encoded once, then FiLM on the features entering the decoder
    my = META_Y
    pills = ["date", "sun elev", "lat/lon"]
    pw, ph, pg = 58, 17, 4
    px0 = COLS["in"] - pw / 2
    py0 = my - (len(pills) * ph + (len(pills) - 1) * pg) / 2
    for i, lab in enumerate(pills):
        y = py0 + i * (ph + pg)
        add(f'<rect x="{px0}" y="{y:.1f}" width="{pw}" height="{ph}" rx="8.5" fill="{t["c_soft"]}" '
            f'stroke="{t["c"]}" stroke-width="1.2"/>')
        add(text(lab, COLS["in"], y + ph / 2 + 4, 11, t["ink"], anchor="middle"))
    stack_bottom = py0 + len(pills) * ph + (len(pills) - 1) * pg
    add(text("meta", COLS["in"], stack_bottom + 22, 15, t["ink"], "bold", "middle"))
    add(text("vector · no pixels", COLS["in"], stack_bottom + 40, 12.5, t["muted"], anchor="middle"))
    ex = COLS["enc"]
    add(f'<path d="M{px0 + pw + 12} {my} L{ex - 34} {my}" stroke="{t["line"]}" stroke-width="1.5" '
        f'fill="none" marker-end="url(#arrow)"/>')
    add(f'<rect x="{ex - 30}" y="{my - 13}" width="60" height="26" rx="13" fill="{t["c"]}"/>')
    add(text("MLP", ex, my + 5, 13, t["bg"], "bold", "middle"))
    add(text("embedding", ex, my + 32, 12.5, t["muted"], anchor="middle"))
    film_x = (fx1 + dx - 52) / 2
    r = 18  # rounded elbow, matching the curved connectors elsewhere
    add(f'<path d="M{ex + 30} {my} L{film_x - r} {my} Q{film_x} {my}, {film_x} {my - r} '
        f'L{film_x} {DEC_Y[-1] + 6}" stroke="{t["c"]}" '
        f'stroke-width="1.5" fill="none"/>')
    add(f'<line x1="{film_x}" y1="{DEC_Y[0]}" x2="{film_x}" y2="{DEC_Y[-1]}" stroke="{t["c"]}" '
        f'stroke-width="1.5"/>')
    for y in DEC_Y:
        add(f'<rect x="{film_x - 5}" y="{y - 5}" width="10" height="10" rx="2" fill="{t["c"]}" '
            f'stroke="{t["bg"]}" stroke-width="1.5"/>')
    add(text("FiLM", film_x, DEC_Y[0] - 14, 12.5, t["c"], "bold", "middle"))

    # ---- swappable options under each column
    opts = {
        "in": ["any channel count", "any H × W", "metadata vectors"],
        "enc": ["any timm backbone", "CNN · hierarchical · ViT", "shared or independent"],
        "fuse": ["early · intermediate · late", "concat · add · gated", "attention · cross-attn",
                 "metadata: film · head · channels"],
        "dec": ["unet · unet++ · fastai_unet", "fpn · deeplabv3+ · upernet", "segformer",
                "shared or per head"],
        "out": ["named, any number", "segmentation · regression", "classification · uncertainty",
                "instances · points · density"],
    }
    add(f'<line x1="24" y1="{RULE_Y}" x2="{W - 24}" y2="{RULE_Y}" stroke="{t["edge"]}" stroke-width="1"/>')
    for key, lines in opts.items():
        for i, s in enumerate(lines):
            add(text(s, COLS[key], RULE_Y + 26 + i * 18, 12.5, t["muted"], anchor="middle"))

    defs = (f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 1 L9 5 L0 9 z" fill="{t["line"]}"/></marker></defs>')
    body = "\n  ".join(el)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="ManySeg architecture: inputs, encoders, fusion, metadata conditioning, decoder, heads">\n  {defs}\n  {body}\n</svg>\n')


for theme in THEMES:
    (OUT / f"architecture-{theme}.svg").write_text(build(theme))
print("ok")
