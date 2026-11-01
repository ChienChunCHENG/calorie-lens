"""Generate PWA icons into web/public/icons/ (run: py scripts/make-icons.py). Requires Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent.parent / "public" / "icons"
out.mkdir(parents=True, exist_ok=True)
GREEN, DARK, WHITE = (31, 157, 85), (18, 110, 58), (255, 255, 255)


def draw(size: int, maskable: bool) -> Image.Image:
    S = size * 4  # supersample
    im = Image.new("RGB", (S, S), GREEN)
    d = ImageDraw.Draw(im)
    if not maskable:  # rounded-square look is applied by the OS; keep full-bleed background
        pass
    k = 0.8 if maskable else 1.0  # maskable keeps the motif inside the safe zone
    c = S / 2
    r = S * 0.34 * k
    d.ellipse([c - r, c - r, c + r, c + r], fill=WHITE)            # plate
    r2 = r * 0.72
    d.ellipse([c - r2, c - r2, c + r2, c + r2], outline=GREEN, width=int(S * 0.018 * k))
    # fork (left) and knife (right)
    w = S * 0.022 * k
    for dx in (-0.46, -0.40, -0.34):
        x = c + r * dx * 1.0 - r * 0.0
    fx = c - r * 1.28
    d.rounded_rectangle([fx - w, c - r * 0.2, fx + w, c + r * 0.85], radius=w, fill=WHITE)
    for off in (-3 * w, 0, 3 * w):
        d.rounded_rectangle([fx + off - w * 0.6, c - r * 0.85, fx + off + w * 0.6, c - r * 0.1], radius=w * 0.6, fill=WHITE)
    kx = c + r * 1.28
    d.rounded_rectangle([kx - w, c - r * 0.1, kx + w, c + r * 0.85], radius=w, fill=WHITE)
    d.rounded_rectangle([kx - w * 2.2, c - r * 0.85, kx + w * 0.4, c - r * 0.05], radius=w * 1.4, fill=WHITE)
    # lens glint
    g = r * 0.16
    d.ellipse([c - r2 * 0.5 - g, c - r2 * 0.5 - g, c - r2 * 0.5 + g, c - r2 * 0.5 + g], fill=(220, 245, 230))
    return im.resize((size, size), Image.LANCZOS)


draw(192, False).save(out / "icon-192.png")
draw(512, False).save(out / "icon-512.png")
draw(512, True).save(out / "icon-512-maskable.png")
draw(180, False).save(out / "apple-touch-icon.png")
print("wrote", sorted(p.name for p in out.iterdir()))
