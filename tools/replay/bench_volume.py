"""Volume bench: synthetic kinds x seeds through align -> plane -> volume; per item-kind error / coverage stats.

  py bench_volume.py [--seeds 10] [--kinds a,b] [--true-depth] [--true-plane] [--out bench_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import align  # noqa: E402
import synth  # noqa: E402
import volume as volume_mod  # noqa: E402

KINDS = {"mound": "rice_mound", "egg": "fried_egg", "cutlet": "cutlet", "bowl_flat": "bowl_flat",
         "bowl_dome": "bowl_dome", "meal": "meal"}


def prepare_scene(synth_kind, seed, true_depth=False, true_plane=False):
    """-> (frames with depth_m, plane, gt, labels, aligned_ok)."""
    frames, gt = synth.make_scene(synth_kind, seed)
    if true_depth:
        for f, d in zip(frames, gt["depth_true"]):
            f.depth_m = d.astype(np.float32).copy()
        for f, T in zip(frames, gt["T_wc_true"]):
            f.T_wc = T
        plane = gt["plane"]
    else:
        plane, diags = align.align_frames(frames, plane_hint=gt["plane"] if true_plane else None)
        if not any(d["ok"] for d in diags):
            return None
    return frames, plane, gt


def item_labels(frames):
    labels, keys = {}, []
    for f in frames:
        for k in f.masks:
            if k not in keys:
                keys.append(k)
            if k in f.labels:
                labels.setdefault(k, Counter())[f.labels[k]] += 1
    lab = {k: labels[k].most_common(1)[0][0] for k in labels}
    return [k for k in keys if not lab.get(k, "").startswith("container_")], lab


def run_scene(scene, **kw):
    """-> list of dict rows (one per item)."""
    frames, plane, gt = scene
    items, lab = item_labels(frames)
    rows = []
    for k in items:
        r = volume_mod.compute_item_volume(frames, k, plane, lab.get(k, "?"),
                                           container=gt["containers"].get(k), **kw)
        g = gt["volumes_ml"].get(k)
        row = {"kind": gt["kind"], "seed": gt["seed"], "item": lab.get(k, k), "gt": g,
               "method": r.method, "coverage": r.coverage}
        if r.volume_ml is None:
            row.update(ok=False)
        else:
            v = r.volume_ml
            row.update(ok=True, typ=v.typ, lo=v.min, hi=v.max, err=100 * (v.typ - g) / g,
                       inside=bool(v.min <= g <= v.max), width=v.max / v.min)
        rows.append(row)
    return rows


def bench(kinds, seeds, true_depth=False, true_plane=False, **kw):
    rows = []
    for name in kinds:
        for s in seeds:
            sc = prepare_scene(KINDS[name], s, true_depth, true_plane)
            if sc is None:
                rows.append({"kind": KINDS[name], "seed": s, "item": "?", "ok": False, "method": "align_failed"})
                continue
            rows += run_scene(sc, **kw)
    return rows


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["item"])].append(r)
    out = {}
    for (kind, item), rs in by.items():
        ok = [r for r in rs if r.get("ok")]
        e = np.array([r["err"] for r in ok]) if ok else np.array([np.nan])
        out[f"{kind}/{item}"] = {
            "n": len(rs), "n_ok": len(ok), "bias_pct": float(np.mean(e)), "mape_pct": float(np.mean(np.abs(e))),
            "coverage_pct": 100.0 * float(np.mean([r["inside"] for r in ok])) if ok else float("nan"),
            "width": float(np.mean([r["width"] for r in ok])) if ok else float("nan"),
            "footprint_cov": float(np.mean([r["coverage"] for r in ok])) if ok else float("nan")}
    return out


def print_table(summ, title=""):
    if title:
        print(title)
    hdr = f"{'scene/item':<30}{'n':>3}{'bias%':>8}{'MAPE%':>8}{'interv%':>9}{'width':>7}{'fpcov':>7}"
    print(hdr)
    print("-" * len(hdr))
    for k, s in summ.items():
        print(f"{k:<30}{s['n_ok']:>3}{s['bias_pct']:>+8.1f}{s['mape_pct']:>8.1f}{s['coverage_pct']:>9.0f}"
              f"{s['width']:>7.2f}{s['footprint_cov']:>7.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--true-depth", action="store_true", help="use exact synth depth + pose + plane (algorithm only)")
    ap.add_argument("--true-plane", action="store_true")
    ap.add_argument("--exact-depth", action="store_true",
                    help="comparison only: bias/coverage with exact synth depth vs aligned depth, side by side (no file written)")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_results.json"))
    args = ap.parse_args(argv)
    if args.exact_depth:
        ks, sd = args.kinds.split(","), range(1, args.seeds + 1)
        ex = summarize(bench(ks, sd, True, True))
        al = summarize(bench(ks, sd, False, args.true_plane))
        print(f"{'scene/item':<30}{'exact%':>8}{'aligned%':>10}{'diff':>7}{'cov_ex':>8}{'cov_al':>8}")
        for k in ex:
            a = al.get(k, {"bias_pct": float('nan'), "coverage_pct": float('nan')})
            print(f"{k:<30}{ex[k]['bias_pct']:>+8.1f}{a['bias_pct']:>+10.1f}{a['bias_pct'] - ex[k]['bias_pct']:>+7.1f}"
                  f"{ex[k]['coverage_pct']:>8.0f}{a['coverage_pct']:>8.0f}")
        return 0
    rows = bench(args.kinds.split(","), range(1, args.seeds + 1), args.true_depth, args.true_plane)
    summ = summarize(rows)
    print_table(summ)
    with open(args.out, "w") as fh:
        json.dump({"summary": summ, "rows": rows}, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
