#!/usr/bin/env python3
"""Evaluate a Calorie Lens YOLO11-seg model on a HELD-OUT dataset dir (your own meals, never trained on).

    py ml/eval.py --model best.pt --data datasets/heldout --out eval_out
    py ml/eval.py --model best.pt --data datasets/merged/data.yaml --split val --out eval_out

Reports
  * box mAP50 and mask mAP50 (Ultralytics validator), plus per-class AP50 for both
  * top-1 dish accuracy per image: GT = the LARGEST dish instance in the image's label file
    (polygon area), prediction = the largest predicted dish instance (mask area) above --conf.
    Container classes are ignored for this metric. It is a proxy for "did the app name the main
    dish", which is the M3 exit criterion (>= 85% top-1 on held-out meals).
  * top1_confusion.png  (rows = GT dish, cols = predicted dish; counts + row-normalised colour)
  * metrics.json with everything, and the key-pair confusion rates below.

Key confusions to watch (the script prints how often each pair is mixed up, either direction):
  * lu_rou_fan (滷肉飯) <-> rice_white (白飯)          rice with sauce vs plain rice
  * chicken_leg (雞腿) / pork_chop (排骨) / fried_chicken_cutlet (雞排)   fried-meat look-alikes
  * boiled_dumplings (水餃) <-> potstickers (鍋貼)     same filling, different cooking
(PLAN.md M3 also names 滷肉飯 vs 肉燥飯, which is not a class yet.)

Held-out dir layouts accepted: a dir with data.yaml (uses --split, default val), or a bare
dir with images/ + labels/ (optionally images/val), or images/ + labels/ directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WATCH_PAIRS = [
    ("lu_rou_fan", "rice_white"),
    ("chicken_leg", "pork_chop"),
    ("chicken_leg", "fried_chicken_cutlet"),
    ("pork_chop", "fried_chicken_cutlet"),
    ("boiled_dumplings", "potstickers"),
]


# ------------------------------------------------------------------ pure helpers (unit-testable)
def poly_area(vals: list[float]) -> float:
    """Shoelace area of a normalised polygon [x1,y1,x2,y2,...]; 4-value boxes (cx,cy,w,h) give w*h."""
    if len(vals) == 4:
        return vals[2] * vals[3]
    xs, ys = vals[0::2], vals[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


def gt_top_dish(label_file: Path, dish_idx: set[int]) -> int | None:
    best, best_a = None, -1.0
    if not label_file.is_file():
        return None
    for ln in label_file.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        if c not in dish_idx:
            continue
        a = poly_area([float(v) for v in t[1:]])
        if a > best_a:
            best, best_a = c, a
    return best


def resolve_data(data: Path, split: str, out: Path, nc_names: dict) -> tuple[Path, Path, Path]:
    """Return (data_yaml, images_dir, labels_dir) for the requested split."""
    if data.is_file():
        d = yaml.safe_load(data.read_text(encoding="utf-8"))
        root = Path(d.get("path", data.parent))
        if not root.is_absolute():
            root = (data.parent / root).resolve()
        rel = d[split]
        imgs = (root / rel) if isinstance(rel, str) else root / rel[0]
        lbls = Path(str(imgs).replace("images", "labels"))
        return data, imgs, lbls
    for cand in (data / "images" / split, data / split / "images", data / "images"):
        if cand.is_dir():
            imgs = cand
            break
    else:
        sys.exit(f"cannot find images under {data}")
    lbls = Path(str(imgs).replace("images", "labels"))
    out.mkdir(parents=True, exist_ok=True)
    y = out / "heldout_data.yaml"
    y.write_text(yaml.safe_dump({
        "path": str(data.resolve()).replace("\\", "/"),
        "train": str(imgs.resolve()).replace("\\", "/"),
        "val": str(imgs.resolve()).replace("\\", "/"),
        "test": str(imgs.resolve()).replace("\\", "/"),
        "names": nc_names,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return y, imgs, lbls


def confusion_png(cm, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(cm)
    row = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.5)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted (largest dish)"); ax.set_ylabel("ground truth (largest dish)")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title("Top-1 dish confusion (counts; colour = row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="best.pt (seg)")
    ap.add_argument("--data", required=True, help="held-out dataset dir or data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels", default=str(HERE / "labels.yaml"))
    ap.add_argument("--imgsz", type=int, default=640, help="eval size; use 320 to mimic the web model")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence for the top-1 dish metric")
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args(argv)

    classes = yaml.safe_load(Path(a.labels).read_text(encoding="utf-8"))["classes"]
    ids = [c["id"] for c in classes]
    dish_idx = {i for i, c in enumerate(classes) if c["type"] == "dish"}
    dish_list = sorted(dish_idx)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    import numpy as np

    model = YOLO(a.model)
    if type(model.model.model[-1]).__name__.startswith("Segment"):
        model.task = "segment"  # make sure the seg validator/predictor is used
    else:
        sys.exit("eval.py needs a segmentation checkpoint (Segment head)")
    if len(model.names) != len(ids):
        sys.exit(f"model has {len(model.names)} classes, labels.yaml has {len(ids)}")
    data_yaml, imgs_dir, lbls_dir = resolve_data(Path(a.data), a.split, out, {i: n for i, n in enumerate(ids)})

    # ---- 1. mAP via the standard validator
    m = model.val(data=str(data_yaml), split=a.split if Path(a.data).is_file() else "val", imgsz=a.imgsz,
                  plots=True, project=str(out.resolve()), name="val", exist_ok=True, verbose=False)
    res = {"box_map50": float(m.box.map50), "mask_map50": float(m.seg.map50),
           "box_map50_95": float(m.box.map), "mask_map50_95": float(m.seg.map)}
    per_class = {}
    for k, ci in enumerate(m.box.ap_class_index):
        per_class[ids[int(ci)]] = {"box_ap50": float(m.box.ap50[k])}
    for k, ci in enumerate(m.seg.ap_class_index):
        per_class.setdefault(ids[int(ci)], {})["mask_ap50"] = float(m.seg.ap50[k])
    res["per_class_ap50"] = per_class
    print(f"\nbox mAP50 {res['box_map50']:.3f} | mask mAP50 {res['mask_map50']:.3f}")
    print(f"{'class':<22}{'box AP50':>9}{'mask AP50':>10}")
    for i in ids:
        if i in per_class:
            v = per_class[i]
            print(f"{i:<22}{v.get('box_ap50', float('nan')):>9.3f}{v.get('mask_ap50', float('nan')):>10.3f}")
        else:
            print(f"{i:<22}{'(no GT)':>9}")

    # ---- 2. top-1 dish accuracy per image
    pos = {c: k for k, c in enumerate(dish_list)}
    cm = np.zeros((len(dish_list), len(dish_list)), dtype=int)
    n_eval = n_ok = n_nopred = 0
    imgs = sorted(p for p in imgs_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for r in model.predict(source=[str(p) for p in imgs], imgsz=a.imgsz, conf=a.conf, stream=True, verbose=False):
        gt = gt_top_dish(lbls_dir / (Path(r.path).stem + ".txt"), dish_idx)
        if gt is None:
            continue
        n_eval += 1
        pred, best_a = None, -1.0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.int().tolist()
            if r.masks is not None:
                areas = []
                for poly in r.masks.xy:
                    p = np.asarray(poly)
                    areas.append(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
                                 if len(p) >= 3 else 0.0)
            else:
                xy = r.boxes.xywh.tolist()
                areas = [w * h for _, _, w, h in xy]
            for c, ar in zip(cls, areas):
                if c in dish_idx and ar > best_a:
                    pred, best_a = c, ar
        if pred is None:
            n_nopred += 1
            continue  # counted as wrong, not placed in the matrix
        cm[pos[gt], pos[pred]] += 1
        n_ok += int(pred == gt)
    top1 = n_ok / n_eval if n_eval else float("nan")
    res.update(top1_dish_acc=top1, top1_images=n_eval, top1_no_dish_predicted=n_nopred)
    print(f"\ntop-1 dish accuracy: {n_ok}/{n_eval} = {top1:.1%}  (no dish predicted: {n_nopred})   target >= 85%")

    names = [ids[c] for c in dish_list]
    confusion_png(cm, names, out / "top1_confusion.png")

    # ---- 3. key pairs + worst off-diagonals
    off = [(cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j]]
    off.sort(reverse=True)
    print("\nlargest top-1 confusions (GT -> predicted):")
    for n, g, p in off[:8]:
        print(f"  {g:<22}-> {p:<22}{n}")
    pairs = {}
    print("\nkey pairs (mixed up either direction / GT count of both):")
    for x, y in WATCH_PAIRS:
        if x in names and y in names:
            i, j = names.index(x), names.index(y)
            tot = int(cm[i].sum() + cm[j].sum())
            bad = int(cm[i, j] + cm[j, i])
            pairs[f"{x}|{y}"] = {"confused": bad, "total": tot}
            print(f"  {x} <-> {y}: {bad}/{tot}")
    res["key_pairs"] = pairs
    res["confusion_matrix"] = {"labels": names, "matrix": cm.tolist()}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'metrics.json'}, {out / 'top1_confusion.png'}, Ultralytics plots in {out / 'val'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
