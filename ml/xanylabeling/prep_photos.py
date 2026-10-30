"""Prepare iPhone photos for labelling: HEIC -> JPEG, apply rotation, STRIP ALL EXIF (incl. GPS).

    py ml/xanylabeling/prep_photos.py RAW_DIR OUT_DIR [--max-side 1600]

Keeps the sub-folder structure (YYYYMMDD_mealNN/), so the by-meal split survives.
Needs: pip install -r ml/requirements-prelabel.txt   (pillow-heif reads HEIC)
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif missing: HEIC files will be skipped")

EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-side", type=int, default=1600)
    a = ap.parse_args()
    n = 0
    for p in sorted(a.src.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        out = a.dst / p.relative_to(a.src).with_suffix(".jpg")
        if out.exists():
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}")
            continue
        im.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = Image.new("RGB", im.size)  # fresh image: no EXIF/GPS/ICC carried over
        clean.paste(im)
        clean.save(out, "JPEG", quality=92)
        n += 1
    print(f"wrote {n} images to {a.dst} (EXIF/GPS stripped)")


if __name__ == "__main__":
    main()
