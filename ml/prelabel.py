#!/usr/bin/env python3
"""prelabel.py - propose dish classes for meal photos with Claude vision.

*** LABELLING AID ONLY. *** This script runs on the developer's PC while building the
dataset. It sends photos to the Anthropic API (so keep private photos out, and strip
GPS first: see ml/CAPTURE.md). The Calorie Lens app itself stays fully offline and
never calls any API. The output is a SUGGESTION: you confirm/fix every item in
X-AnyLabeling or Roboflow; it also gives a per-class count summary showing which
classes still need more photos.

Usage (Windows):
    py -m pip install -r ml/requirements-prelabel.txt
    set ANTHROPIC_API_KEY=sk-ant-...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
    py ml/prelabel.py D:\\photos\\clean --out D:\\photos\\prelabel
    py ml/prelabel.py D:\\photos\\clean --out D:\\photos\\prelabel --dry-run   # no API call

Outputs: <out>/<relative path>.json per image, and <out>/summary.csv
(class_id, zh, en, images, items) rebuilt from all JSONs on every run.
Re-running skips images that already have a JSON (resumable); use --force to redo.
The API key is read from the environment only; it is never stored or logged.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import re
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 1024
LOCATIONS = ["top-left", "top", "top-right", "left", "center", "right",
             "bottom-left", "bottom", "bottom-right"]

# Mirror of ml/labels.yaml (binding; order = model class index). Checked against the yaml
# at startup when it is found next to this script, so drift is reported, not silent.
CLASSES = [
    ("rice_white", "白飯", "Steamed white rice", "dish"),
    ("lu_rou_fan", "滷肉飯", "Braised pork rice", "dish"),
    ("chicken_leg", "雞腿", "Chicken drumstick", "dish"),
    ("pork_chop", "排骨", "Pork chop", "dish"),
    ("braised_egg", "滷蛋", "Braised egg", "dish"),
    ("fried_egg", "荷包蛋", "Fried egg", "dish"),
    ("stirfried_greens", "炒青菜", "Stir-fried greens", "dish"),
    ("braised_tofu", "滷豆腐", "Braised tofu", "dish"),
    ("beef_noodle_soup", "牛肉麵", "Beef noodle soup", "dish"),
    ("boiled_dumplings", "水餃", "Boiled dumplings", "dish"),
    ("potstickers", "鍋貼", "Potstickers", "dish"),
    ("scallion_pancake", "蔥油餅", "Scallion pancake", "dish"),
    ("fried_chicken_cutlet", "雞排", "Fried chicken cutlet", "dish"),
    ("popcorn_chicken", "鹽酥雞", "Taiwanese popcorn chicken", "dish"),
    ("bubble_milk_tea", "珍珠奶茶", "Bubble milk tea", "dish"),
    ("container_bowl", "碗", "Bowl", "container"),
    ("container_plate", "盤", "Plate", "container"),
    ("container_bento", "便當盒", "Bento box", "container"),
    ("container_cup", "杯", "Drink cup", "container"),
]
IDS = [c[0] for c in CLASSES]

SYSTEM_TEXT = (
    "You are a labelling assistant for a Taiwanese meal photo dataset. For each photo, list "
    "which of the fixed classes below are visible, so a human can confirm them in a "
    "segmentation tool. You never invent classes.\n\n"
    "CLASSES (class_id | zh | en | type):\n"
    + "\n".join(f"{i}. {c[0]} | {c[1]} | {c[2]} | {c[3]}" for i, c in enumerate(CLASSES))
    + "\n\nRULES\n"
    "- Report each separate instance once: a mixed bowl of beef noodle soup (牛肉麵) or braised "
    "pork rice (滷肉飯) is ONE item, never split into its toppings. A pile of popcorn chicken "
    "(鹽酥雞) is one item per pile; a plate of dumplings (水餃 or 鍋貼) is one item per plate.\n"
    "- Report the container of every dish as its own item (container_bowl, container_plate, "
    "container_bento, container_cup). A bento box with several dishes is one container_bento "
    "plus one item per dish compartment (e.g. rice_white, pork_chop, braised_egg, stirfried_greens).\n"
    "- A bubble tea cup gives two items: bubble_milk_tea and container_cup.\n"
    "- Plain white rice is rice_white; rice with minced braised pork sauce poured over it is "
    "lu_rou_fan. Pork chop (排骨, bone-in fried cutlet) differs from fried_chicken_cutlet (雞排, "
    "very large flat breaded chicken breast) and from chicken_leg (drumstick/whole leg, often "
    "braised or roasted). braised_egg is brown and shiny; fried_egg has a white with a yolk.\n"
    "- boiled_dumplings are pale and wet; potstickers are pan-fried with a browned crisp bottom.\n"
    "- Anything edible that does not clearly match a class (soups, noodles other than beef noodle "
    "soup, desserts, other drinks, other dishes) goes in unknown_foods as a short English "
    "name; do NOT map it to the closest class. Ignore garnish, sauces, chopsticks and people.\n"
    "- confidence is 0-1: how sure you are that the class is right and present.\n"
    "- approx_location is one of: " + ", ".join(LOCATIONS) + ".\n\n"
    "OUTPUT: respond with ONE JSON object and nothing else (no prose, no code fences):\n"
    '{"items":[{"class_id":"<id from the list>","confidence":0.0,"approx_location":"center"}],'
    '"unknown_foods":["..."]}\n'
    'If no known class is visible: {"items":[],"unknown_foods":[...]}.'
)


def check_against_yaml() -> None:
    """Warn if ml/labels.yaml drifted from CLASSES (labels.yaml wins)."""
    y = Path(__file__).resolve().parent / "labels.yaml"
    if not y.exists():
        return
    ids = re.findall(r"-\s*\{id:\s*(\w+)", y.read_text(encoding="utf-8"))
    if ids != IDS:
        sys.exit(f"CLASSES in prelabel.py differ from {y}. Update CLASSES (yaml order is binding).")


def find_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def encode_image(path: Path) -> tuple[str, str]:
    """Resize to <=1024 px long side, return (media_type, base64). Strips EXIF."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("ascii")


def parse_reply(text: str) -> dict:
    """Strict JSON, tolerant of stray fences; validates ids and clamps values."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    items, rejected = [], []
    for it in data.get("items", []):
        cid = it.get("class_id")
        if cid not in IDS:
            rejected.append(cid)
            continue
        conf = max(0.0, min(1.0, float(it.get("confidence", 0))))
        loc = it.get("approx_location")
        items.append({"class_id": cid, "confidence": round(conf, 3),
                      "approx_location": loc if loc in LOCATIONS else "center"})
    out = {"items": items, "unknown_foods": [str(u) for u in data.get("unknown_foods", [])]}
    if rejected:
        out["rejected_class_ids"] = rejected
    return out


def call_model(client, model: str, media_type: str, b64: str, max_retries: int):
    import anthropic
    delay = 4.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM_TEXT,
                         "cache_control": {"type": "ephemeral"}}],  # static list, cached across images
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": "Label this photo. JSON only."},
                ]}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, resp.usage
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            if attempt == max_retries:
                raise
            wait = delay + random.uniform(0, 1.5)
            ra = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            print(f"  {type(e).__name__}; retry {attempt + 1}/{max_retries} in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 120)


def write_summary(out_dir: Path) -> None:
    img_ct = {i: 0 for i in IDS}
    item_ct = {i: 0 for i in IDS}
    n = 0
    for j in out_dir.rglob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "items" not in d:
            continue
        n += 1
        seen = set()
        for it in d["items"]:
            item_ct[it["class_id"]] += 1
            seen.add(it["class_id"])
        for c in seen:
            img_ct[c] += 1
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "zh", "en", "images", "items"])
        for cid, zh, en, _ in CLASSES:
            w.writerow([cid, zh, en, img_ct[cid], item_ct[cid]])
    print(f"summary.csv: {n} photos -> {out_dir / 'summary.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", type=Path, help="image file or folder (searched recursively)")
    ap.add_argument("--out", type=Path, required=True, help="output folder for per-image JSON + summary.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    ap.add_argument("--dry-run", action="store_true", help="list work and resize images; no API call")
    ap.add_argument("--force", action="store_true", help="redo images that already have a JSON")
    ap.add_argument("--limit", type=int, default=0, help="process at most N new images")
    ap.add_argument("--max-retries", type=int, default=6, help="retries on rate limit/transient errors")
    a = ap.parse_args()

    check_against_yaml()
    root = a.images if a.images.is_dir() else a.images.parent
    imgs = find_images(a.images)
    if not imgs:
        print("no images found")
        return 1
    todo = []
    for p in imgs:
        j = a.out / p.relative_to(root).with_suffix(".json")
        if a.force or not j.exists():
            todo.append((p, j))
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(imgs)} images, {len(imgs) - len(todo)} already done, {len(todo)} to do")

    client = None
    if not a.dry_run and todo:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set (never hardcode it; set it in your shell).", file=sys.stderr)
            return 2
        import anthropic
        client = anthropic.Anthropic(max_retries=0)  # we do our own backoff

    failures = 0
    for k, (p, j) in enumerate(todo, 1):
        try:
            media_type, b64 = encode_image(p)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(todo)}] {p.name}: cannot read ({e})", file=sys.stderr)
            failures += 1
            continue
        if a.dry_run:
            print(f"[{k}/{len(todo)}] DRY {p.name} -> {j} ({len(b64) * 3 // 4 // 1024} KB jpeg)")
            continue
        try:
            text, usage = call_model(client, a.model, media_type, b64, a.max_retries)
            res = parse_reply(text)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(todo)}] {p.name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue
        res["source"] = str(p.relative_to(root))
        res["model"] = a.model
        j.parent.mkdir(parents=True, exist_ok=True)
        j.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        print(f"[{k}/{len(todo)}] {p.name}: {[i['class_id'] for i in res['items']]} "
              f"unknown={res['unknown_foods']} (cache_read={cached})")

    if not a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        write_summary(a.out)
    if failures:
        print(f"{failures} failed; re-run to retry them (finished images are skipped)", file=sys.stderr)
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(main())
