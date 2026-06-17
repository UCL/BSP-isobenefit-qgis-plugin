"""Generate the plugin icons.

Two sibling icons on the same green rounded tile and palette:
- ``icon.png`` — the simulator: faint concentric walkability rings, orange built
  blocks on the accessible ring, a central white "centrality" node (walkable access
  to centres within green space).
- ``osm_icon.png`` — the OpenStreetMap extractor: a faint map grid with an orange
  download arrow dropping into a white tray (download map data).

Run: uv run --no-project --with pillow --with numpy python scripts/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
ICON_DIR = Path(__file__).resolve().parent.parent / "isobenefit_qgis"

GREEN_TOP = (115, 192, 74)
GREEN_BOT = (62, 142, 52)
ORANGE = (197, 86, 17, 255)
WHITE = (255, 255, 255, 255)


def green_tile() -> Image.Image:
    """The shared green rounded-square tile (transparent corners)."""
    t = np.linspace(0.0, 1.0, S).reshape(S, 1)
    rows = np.array(GREEN_TOP, float) * (1 - t) + np.array(GREEN_BOT, float) * t
    bg = np.broadcast_to(rows[:, None, :], (S, S, 3)).astype(np.uint8)
    base = Image.fromarray(bg, "RGB").convert("RGBA")
    pad = int(S * 0.05)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], radius=int(S * 0.20), fill=255)
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile.paste(base, (0, 0), mask)
    return tile


def build_sim_icon() -> Image.Image:
    tile = green_tile()
    draw = ImageDraw.Draw(tile)
    cx = cy = S // 2

    # concentric walkability rings
    for rad, alpha, w in [(int(S * 0.33), 60, int(S * 0.016)), (int(S * 0.225), 95, int(S * 0.018))]:
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 255, 255, alpha), width=w)

    # built blocks clustering on the outer ring
    blk = int(S * 0.062)
    ring = int(S * 0.33)
    for ang in (35, 125, 215, 305):
        bx = cx + int(ring * math.cos(math.radians(ang)))
        by = cy + int(ring * math.sin(math.radians(ang)))
        draw.rounded_rectangle([bx - blk, by - blk, bx + blk, by + blk], radius=int(blk * 0.3), fill=ORANGE)

    # central centrality node
    nr = int(S * 0.115)
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=WHITE, outline=ORANGE, width=int(S * 0.026))
    return tile


def build_osm_icon() -> Image.Image:
    tile = green_tile()
    draw = ImageDraw.Draw(tile)
    cx = S // 2

    # faint map grid (kept inside the tile's straight edges so it never hits a corner)
    lo, hi = int(S * 0.13), int(S * 0.87)
    gw = max(2, int(S * 0.009))
    for f in (0.32, 0.5, 0.68):
        p = int(S * f)
        draw.line([(p, lo), (p, hi)], fill=(255, 255, 255, 55), width=gw)
        draw.line([(lo, p), (hi, p)], fill=(255, 255, 255, 55), width=gw)

    # white "download tray" (an open box) at the bottom
    tw = int(S * 0.030)
    tray_l, tray_r, tray_b = int(S * 0.31), int(S * 0.69), int(S * 0.77)
    tray_t = tray_b - int(S * 0.10)
    draw.line([(tray_l, tray_b), (tray_r, tray_b)], fill=WHITE, width=tw)
    draw.line([(tray_l, tray_b), (tray_l, tray_t)], fill=WHITE, width=tw)
    draw.line([(tray_r, tray_b), (tray_r, tray_t)], fill=WHITE, width=tw)

    # orange download arrow dropping into the tray
    shaft_w = int(S * 0.090)
    draw.rounded_rectangle(
        [cx - shaft_w // 2, int(S * 0.22), cx + shaft_w // 2, int(S * 0.50)],
        radius=int(shaft_w * 0.3),
        fill=ORANGE,
    )
    head_half = int(S * 0.155)
    draw.polygon(
        [(cx - head_half, int(S * 0.48)), (cx + head_half, int(S * 0.48)), (cx, int(S * 0.65))],
        fill=ORANGE,
    )
    return tile


def main() -> None:
    for name, builder in [("icon.png", build_sim_icon), ("osm_icon.png", build_osm_icon)]:
        out = ICON_DIR / name
        builder().resize((128, 128), Image.LANCZOS).save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
