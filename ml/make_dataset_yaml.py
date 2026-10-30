#!/usr/bin/env python3
"""Build the Ultralytics dataset (data.yaml + images/labels) for Calorie Lens.

Pure Python + PyYAML. No numpy / PIL / ultralytics needed.

What it does
------------
1. Reads ml/labels.yaml (binding: list order == model class index) and writes data.yaml
   with `names: {0: rice_white, 1: lu_rou_fan, ...}`.
2. Optionally merges one or more YOLO-seg exports (Roboflow "YOLOv8/YOLO11" export format,
   or a plain images/ + labels/ folder), remapping their class ids onto ours via a
   mapping YAML (ml/mappings/<name>.yaml). Source classes not in the mapping are DROPPED
   (their label lines are removed; the image stays, so it acts as a background/negative
   for those objects -- see --drop-images-with-dropped to discard such images instead).
3. Splits train/val by MEAL/SESSION, never by image, so near-duplicate frames and Roboflow
   augmentation copies of one photo never straddle the split.

Session key (first match wins)
------------------------------
  a. --session-regex (first capture group), applied to the file stem
  b. a parent folder named YYYYMMDD_mealNN (our capture layout, ml/CAPTURE.md) -> that folder
  c. "<session>__<anything>" in the file name, e.g. meal12__0007.jpg -> session "meal12"
  d. Roboflow stem with the ".rf.<hash>" suffix and "_jpg"/"_jpeg"/"_png" removed, so all
     augmented copies of one original share a session.
Sessions are namespaced by source name so two sources never collide.

Mapping YAML (ml/mappings/<name>.yaml)
--------------------------------------
    source: "Roboflow project / URL, for provenance"
    map:                     # source class (name OR integer id)  ->  our class (id string OR index)
      bento: container_bento
      "fried chicken": fried_chicken_cutlet
      3: rice_white
    drop: [fork, spoon]      # optional: classes intentionally dropped (silences the 'unmapped' warning)

Source class names are read from the source's own data.yaml (`names:` list or dict).

Examples
--------
    # just data.yaml for a dataset already in our class order
    py ml/make_dataset_yaml.py --yaml-only datasets/mine

    # merge two exports + own data, preview only
    py ml/make_dataset_yaml.py --out datasets/merged --dry-run \
        --source exports/tw_food::ml/mappings/tw_food.yaml \
        --source exports/mine                      # no mapping => already our order

    py ml/make_dataset_yaml.py --out /content/ds --source /content/mine --source /content/tw::ml/mappings/tw_food.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{8,}$", re.I)
RF_EXT_TAG = re.compile(r"_(jpg|jpeg|png)$", re.I)


# ----------------------------------------------------------------------------- labels
def load_labels(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = data["classes"]
    ids = [c["id"] for c in classes]
    if len(set(ids)) != len(ids):
        sys.exit(f"duplicate ids in {path}")
    return classes


def write_data_yaml(out_dir: Path, classes: list[dict], with_val: bool = True) -> Path:
    doc = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {i: c["id"] for i, c in enumerate(classes)},
    }
    p = out_dir / "data.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- sources
class Source:
    def __init__(self, spec: str, classes: list[dict]):
        path, _, mapping = spec.partition("::")
        self.root = Path(path)
        if not self.root.is_dir():
            sys.exit(f"source not found: {self.root}")
        self.name = re.sub(r"[^A-Za-z0-9._-]+", "-", self.root.resolve().name)
        self.our_ids = [c["id"] for c in classes]
        self.src_names = self._read_src_names()
        self.mapping_path = Path(mapping) if mapping else None
        self.cls_map: dict[int, int | None] = {}
        self._build_map()

    # Roboflow data.yaml names can be a list or a {idx: name} dict
    def _read_src_names(self) -> list[str] | None:
        for cand in ("data.yaml", "dataset.yaml"):
            f = self.root / cand
            if f.is_file():
                d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                n = d.get("names")
                if isinstance(n, dict):
                    return [str(n[k]) for k in sorted(n, key=int)]
                if isinstance(n, list):
                    return [str(x) for x in n]
        return None

    def _build_map(self):
        if self.mapping_path is None:  # identity: source already uses our class order
            self.cls_map = {i: i for i in range(len(self.our_ids))}
            return
        m = yaml.safe_load(self.mapping_path.read_text(encoding="utf-8")) or {}
        mp = m.get("map") or {}
        dropped = {str(x) for x in (m.get("drop") or [])}
        if self.src_names is None:
            # allow integer-only mappings when the source has no data.yaml
            n_src = max([int(k) for k in mp if str(k).isdigit()] + [-1]) + 1
            self.src_names = [str(i) for i in range(n_src)]
        by_name = {n: i for i, n in enumerate(self.src_names)}
        for k, tgt in mp.items():
            if isinstance(k, int) or str(k).isdigit():
                si = int(k)
            elif str(k) in by_name:
                si = by_name[str(k)]
            else:
                print(f"  [warn] {self.name}: mapping key {k!r} not in source classes {self.src_names}")
                continue
            if isinstance(tgt, int):
                ti = tgt
            elif tgt in self.our_ids:
                ti = self.our_ids.index(tgt)
            else:
                sys.exit(f"{self.mapping_path}: target {tgt!r} is not a class id in labels.yaml")
            if not 0 <= ti < len(self.our_ids):
                sys.exit(f"{self.mapping_path}: target index {ti} out of range")
            self.cls_map[si] = ti
        unmapped = [n for i, n in enumerate(self.src_names) if i not in self.cls_map and n not in dropped]
        if unmapped:
            print(f"  [info] {self.name}: unmapped source classes will be dropped: {unmapped}")

    def iter_items(self):
        """Yield (image_path, label_path_or_None, stem, rel_dir_parts). Walks the whole tree, so it handles
        Roboflow exports (train/images + train/labels), plain images/ + labels/, and our own capture layout
        (YYYYMMDD_mealNN/IMG_0001.jpg with labels next to it or in a labels/ sibling)."""
        for img in sorted(self.root.rglob("*")):
            if img.suffix.lower() not in IMG_EXT or not img.is_file():
                continue
            lbl = None
            cands = [img.with_suffix(".txt"), img.parent / "labels" / (img.stem + ".txt")]
            parts = list(img.parts)
            if "images" in parts:  # swap the LAST 'images' component for 'labels'
                k = len(parts) - 1 - parts[::-1].index("images")
                cands.insert(0, Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt"))
            for c in cands:
                if c.is_file():
                    lbl = c
                    break
            rel = img.relative_to(self.root).parent.parts
            yield img, lbl, img.stem, tuple(rel)


MEAL_DIR = re.compile(r"^\d{8}_meal\d+", re.I)
GENERIC_DIRS = {"images", "labels", "train", "valid", "val", "test"}


def session_key(src: str, stem: str, rx: re.Pattern | None, rel_dirs: tuple = ()) -> str:
    if rx:
        m = rx.search(stem)
        if m:
            return f"{src}/{m.group(1)}"
    for d in reversed(rel_dirs):  # capture layout: YYYYMMDD_mealNN/ folder (ml/CAPTURE.md)
        if MEAL_DIR.match(d):
            return f"{src}/{d}"
    if "__" in stem:
        return f"{src}/{stem.split('__', 1)[0]}"
    s = RF_SUFFIX.sub("", stem)
    s = RF_EXT_TAG.sub("", s)
    return f"{src}/{s}"


def remap_label_file(path: Path | None, cls_map: dict[int, int | None], boxes_as_polys: bool, stats: Counter):
    """Return list of output lines (strings). Updates stats."""
    if path is None:
        return [], False
    out, dropped_any = [], False
    for raw in path.read_text(encoding="utf-8").splitlines():
        t = raw.split()
        if not t:
            continue
        try:
            sc = int(float(t[0]))
        except ValueError:
            stats["bad_lines"] += 1
            continue
        tc = cls_map.get(sc)
        if tc is None:
            stats["dropped_instances"] += 1
            dropped_any = True
            continue
        vals = t[1:]
        if len(vals) == 4:  # detection box: cx cy w h
            if not boxes_as_polys:
                stats["box_lines_skipped"] += 1
                continue
            cx, cy, w, h = (float(v) for v in vals)
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            vals = [f"{v:.6f}" for v in (x0, y0, x1, y0, x1, y1, x0, y1)]
            stats["box_to_poly"] += 1
        elif len(vals) < 6 or len(vals) % 2:
            stats["bad_lines"] += 1
            continue
        out.append(" ".join([str(tc)] + vals))
    return out, dropped_any


# ----------------------------------------------------------------------------- split
def pick_val_sessions(sessions: dict[str, list], class_inst: dict[str, Counter], val_frac: float, seed: int, nc: int):
    """Deterministic session-level split. Returns the set of val session keys."""
    keys = sorted(sessions, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    total = sum(len(v) for v in sessions.values())
    target = max(1, round(total * val_frac)) if len(keys) > 1 else 0
    val, n = set(), 0
    for k in keys:
        if n >= target:
            break
        val.add(k)
        n += len(sessions[k])
    # make sure each class that has >=2 sessions shows up in val
    cls_sessions: dict[int, set] = defaultdict(set)
    for k, cnt in class_inst.items():
        for c in cnt:
            cls_sessions[c].add(k)
    for c in range(nc):
        ss = cls_sessions.get(c, set())
        if len(ss) >= 2 and not (ss & val):
            cand = min(ss, key=lambda k: len(sessions[k]))
            val.add(cand)
    return val


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(HERE / "labels.yaml"))
    ap.add_argument("--out", help="output dataset dir (required unless --yaml-only)")
    ap.add_argument("--source", action="append", default=[], metavar="DIR[::MAPPING.yaml]",
                    help="YOLO-seg export dir; optional mapping yaml after '::'. Repeatable.")
    ap.add_argument("--yaml-only", metavar="DIR", help="only write DIR/data.yaml (dataset already in our class order)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--session-regex", help="regex; group 1 = session id, applied to the file stem")
    ap.add_argument("--no-boxes-as-polys", action="store_true",
                    help="skip box-only label lines instead of converting them to rectangle polygons")
    ap.add_argument("--drop-images-with-dropped", action="store_true",
                    help="discard any image that had at least one instance of a dropped class "
                         "(avoids teaching the model that unlabeled-but-present food is background)")
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy (same drive only)")
    ap.add_argument("--dry-run", action="store_true", help="print counts per class and split sizes; write nothing")
    a = ap.parse_args(argv)

    classes = load_labels(Path(a.labels))
    nc = len(classes)

    if a.yaml_only:
        p = write_data_yaml(Path(a.yaml_only), classes)
        print(f"wrote {p} ({nc} classes)")
        return 0
    if not a.out:
        ap.error("--out is required (or use --yaml-only)")
    if not a.source:
        ap.error("at least one --source is required")
    out = Path(a.out)
    rx = re.compile(a.session_regex) if a.session_regex else None

    items = []  # dict per image
    for spec in a.source:
        src = Source(spec, classes)
        stats = Counter()
        n_img = 0
        for img, lbl, stem, rel in src.iter_items():
            lines, had_dropped = remap_label_file(lbl, src.cls_map, not a.no_boxes_as_polys, stats)
            if had_dropped and a.drop_images_with_dropped:
                stats["images_discarded"] += 1
                continue
            if not lines and lbl is not None and lbl.stat().st_size > 0:
                stats["images_all_dropped"] += 1
            items.append({
                "src": src.name, "img": img, "lines": lines, "stem": stem,
                "rel": "_".join([d for d in rel if d.lower() not in GENERIC_DIRS] + [stem]),
                "session": session_key(src.name, stem, rx, rel),
            })
            n_img += 1
        print(f"source {src.name}: {n_img} images  " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    if not items:
        sys.exit("no images found")

    sessions: dict[str, list] = defaultdict(list)
    class_inst: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        sessions[it["session"]].append(it)
        for ln in it["lines"]:
            class_inst[it["session"]][int(ln.split()[0])] += 1

    val_keys = pick_val_sessions(sessions, class_inst, a.val_frac, a.seed, nc)
    for it in items:
        it["split"] = "val" if it["session"] in val_keys else "train"

    # ---- report
    inst = {s: Counter() for s in ("train", "val")}
    imgs = {s: Counter() for s in ("train", "val")}
    n_by_split = Counter(it["split"] for it in items)
    for it in items:
        seen = set()
        for ln in it["lines"]:
            c = int(ln.split()[0])
            inst[it["split"]][c] += 1
            seen.add(c)
        for c in seen:
            imgs[it["split"]][c] += 1
    print(f"\nsessions: {len(sessions)} total, {len(val_keys)} in val")
    print(f"images: train={n_by_split['train']} val={n_by_split['val']}  (backgrounds/no-label: "
          f"{sum(1 for it in items if not it['lines'])})")
    print(f"\n{'idx':>3}  {'class id':<22}{'inst train':>11}{'inst val':>9}{'img train':>10}{'img val':>8}")
    for i, c in enumerate(classes):
        flag = ""
        if inst["train"][i] + inst["val"][i] == 0:
            flag = "  <-- NO DATA"
        elif inst["val"][i] == 0:
            flag = "  <-- not in val"
        print(f"{i:>3}  {c['id']:<22}{inst['train'][i]:>11}{inst['val'][i]:>9}{imgs['train'][i]:>10}{imgs['val'][i]:>8}{flag}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ---- write
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for it in items:
        name = re.sub(r"[^\w.\-]+", "-", f"{it['src']}__{it['rel']}")
        di = out / "images" / it["split"] / (name + it["img"].suffix.lower())
        dl = out / "labels" / it["split"] / (name + ".txt")
        if a.link:
            if di.exists():
                di.unlink()
            try:
                di.hardlink_to(it["img"])
            except OSError:
                shutil.copy2(it["img"], di)
        else:
            shutil.copy2(it["img"], di)
        dl.write_text("\n".join(it["lines"]) + ("\n" if it["lines"] else ""), encoding="utf-8")
    p = write_data_yaml(out, classes)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
