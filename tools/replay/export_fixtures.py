"""Export golden fixtures for the Swift GoldenReplayTests.

  py tools/replay/export_fixtures.py --out DIR

Writes DIR/<case>/ (a sweep dump, see DUMP_FORMAT.md) for synthetic scenes at seed 0 and DIR/expected.json with the
ground truth plus Python's own align + volume result per item, computed on the RELOADED dump exactly like
run_replay.py does. numpy + Pillow only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dump_io  # noqa: E402
import synth  # noqa: E402
import align  # noqa: E402
import volume as volume_mod  # noqa: E402

# case name -> (synth kind, give the true plane as hint). Contract names; kinds as accepted by synth.make_scene.
CASES = [
    ("mound", "rice_mound", True),
    ("egg", "fried_egg", True),
    ("cutlet", "cutlet", True),
    ("bowl_dome", "bowl_dome", True),
    ("meal", "meal", True),
    ("meal_noplane", "meal", False),
]
SEED = 0


def _container_for_dump(c: dict) -> dict:
    """Only the manifest keys (shape, item_key, rim_diameter_cm, depth_cm); scalars are allowed by the reader."""
    out = {"shape": c["shape"], "item_key": c.get("item_key")}
    for k in ("rim_diameter_cm", "depth_cm"):
        if k in c:
            out[k] = c[k]
    return out


def _json_container(c):
    return None if c is None else {k: v for k, v in c.items()}


def export_case(out_root: str, name: str, kind: str, with_plane: bool) -> dict:
    frames, gt = synth.make_scene(kind, SEED)
    meta = {"device_model": "synthetic", "ios_version": "0", "image_size": [synth.IMG_W, synth.IMG_H],
            "depth_model": "synthetic"}
    if with_plane:
        p = gt["plane"]
        meta["plane"] = {"n": [float(x) for x in p.n], "offset": float(p.offset), "source": "synthetic"}
    meta["containers"] = {k: _container_for_dump(c) for k, c in gt["containers"].items()}
    case_dir = os.path.join(out_root, name)
    dump_io.save_dump(frames, case_dir, meta)

    # Python reference on the reloaded dump (what run_replay.py --dump does).
    frames2, man = dump_io.load_dump_with_meta(case_dir)
    containers = man.get("containers") or {}
    hint = man.get("plane_obj")
    plane, diags = align.align_frames(frames2, plane_hint=hint)
    labels, keys = {}, []
    for f in frames2:
        for k in f.masks:
            if k not in keys:
                keys.append(k)
            if k in f.labels:
                labels.setdefault(k, Counter())[f.labels[k]] += 1
    lab = {k: labels[k].most_common(1)[0][0] for k in labels}
    items = {}
    for k in keys:
        if lab.get(k, "").startswith("container_"):
            continue
        cont = containers.get(k)
        r = volume_mod.compute_item_volume(frames2, k, plane, lab.get(k, "?"), container=copy.deepcopy(cont))
        v = r.volume_ml
        items[k] = {
            "label": lab.get(k, "?"),
            "container": _json_container(cont),
            "gt_ml": float(gt["volumes_ml"][k]) if k in gt["volumes_ml"] else None,
            "py": {"method": r.method,
                   "typ": None if v is None else float(v.typ),
                   "min": None if v is None else float(v.min),
                   "max": None if v is None else float(v.max),
                   "coverage": float(r.coverage)},
        }
    return {"use_plane_hint": bool(with_plane), "kind": kind, "items": items}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    expected = {}
    t0 = time.time()
    for name, kind, with_plane in CASES:
        t1 = time.time()
        expected[name] = export_case(args.out, name, kind, with_plane)
        print(f"[{name}] kind={kind} plane_hint={with_plane} ({time.time() - t1:.1f}s)")
        for k, it in expected[name]["items"].items():
            py = it["py"]
            typ = "None" if py["typ"] is None else f"{py['typ']:.1f}"
            gtv = "None" if it["gt_ml"] is None else f"{it['gt_ml']:.1f}"
            print(f"    {k:<8} {it['label']:<22} {py['method']:<12} py_typ={typ:>8} gt={gtv:>8} cov={py['coverage']:.2f}")
    with open(os.path.join(args.out, "expected.json"), "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=1)
    print(f"wrote {len(CASES)} cases + expected.json to {args.out} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
