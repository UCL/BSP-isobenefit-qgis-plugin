"""Generate the plugin icon (isobenefit_qgis/icon.png).

A green rounded tile (nature) with faint concentric walkability rings, orange
built blocks clustering on the accessible ring, and a central white "centrality"
node — i.e. the Isobenefit idea: walkable access to centres within green space.

Run: uv run --no-project --with pillow --with numpy python scripts/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
OUT = Path(__file__).resolve().parent.parent / "isobenefit_qgis" / "icon.png"

GREEN_TOP = (115, 192, 74)
GREEN_BOT = (62, 142, 52)
ORANGE = (197, 86, 17, 255)
WHITE = (255, 255, 255, 255)


def main() -> None:
    # vertical green gradient
    t = np.linspace(0.0, 1.0, S).reshape(S, 1)
    rows = np.array(GREEN_TOP, float) * (1 - t) + np.array(GREEN_BOT, float) * t
    bg = np.broadcast_to(rows[:, None, :], (S, S, 3)).astype(np.uint8)
    base = Image.fromarray(bg, "RGB").convert("RGBA")

    # rounded-square mask
    pad = int(S * 0.05)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], radius=int(S * 0.20), fill=255)
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile.paste(base, (0, 0), mask)

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

    tile.resize((128, 128), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
