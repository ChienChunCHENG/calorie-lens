"""Replay CLI: sweep dump (or synthetic scene) -> align -> plane -> volume per item -> table.

  py run_replay.py --synth bowl_dome --seed 3 [--save-debug out_dir]
  py run_replay.py --dump path/to/dump_dir_or.zip [--save-debug out_dir]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cl_types import Plane  # noqa: E402

try:
    import plane as plane_mod  # noqa: E402
    import volume as volume_mod  # noqa: E402
except ImportError as e:
    sys.exit(f"run_replay.py needs plane.py and volume.py in {HERE} (owned by the volume agent): {e}")
import align  # noqa: E402


def _basis(n):
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.3:
        a = np.cross(n, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def debug_heightmap(frames, key, plane, cell=0.003, min_obs=3):
    """Median-over-frames height map (m) on a plane-aligned grid for one item; NaN where < min_obs samples."""
    a, b = _basis(plane.n)
    us, vs, hs = [], [], []
    for f in frames:
        m = f.masks.get(key)
        if f.depth_m is None or m is None:
            continue
        sel = m & np.isfinite(f.depth_m)
        vv, uu = np.nonzero(sel)
        d = f.depth_m[sel].astype(np.float64)
        pc = np.stack([(uu - f.K[0, 2]) * d / f.K[0, 0], -(vv - f.K[1, 2]) * d / f.K[1, 1], -d], 1)
        pw = pc @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]
        us.append(pw @ a)
        vs.append(pw @ b)
        hs.append(plane.height(pw))
    if not us:
        return None
    u, v, h = map(np.concatenate, (us, vs, hs))
    ok = (h > -0.05) & (h < 0.4)
    u, v, h = u[ok], v[ok], h[ok]
    if len(u) == 0:
        return None
    iu = np.floor((u - u.min()) / cell).astype(np.int64)
    iv = np.floor((v - v.min()) / cell).astype(np.int64)
    nv = iv.max() + 1
    key_ = iu * nv + iv
    order = np.lexsort((h, key_))
    key_s, h_s = key_[order], h[order]
    uniq, start, cnt = np.unique(key_s, return_index=True, return_counts=True)
    med = h_s[start + cnt // 2]
    out = np.full((iu.max() + 1) * nv, np.nan)
    out[uniq[cnt >= min_obs]] = med[cnt >= min_obs]
    return np.clip(out.reshape(iu.max() + 1, nv), 0, None)


def save_debug(frames, keys, plane, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for key in keys:
        hm = debug_heightmap(frames, key, plane)
        if hm is None:
            continue
        top = max(np.nanmax(hm) if np.isfinite(hm).any() else 0.0, 1e-3)
        img = np.where(np.isfinite(hm), 40 + 215 * hm / top, 0).astype(np.uint8)
        Image.fromarray(img.T).resize((img.shape[0] * 3, img.shape[1] * 3), Image.NEAREST).save(
            os.path.join(out_dir, f"height_{key}.png"))
    for i, f in enumerate(frames):
        if f.depth_m is not None:
            d = np.nan_to_num(f.depth_m, nan=0.0)
            lo, hi = np.percentile(d[d > 0], [1, 99]) if (d > 0).any() else (0, 1)
            img = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            Image.fromarray((255 * (1 - img)).astype(np.uint8)).save(os.path.join(out_dir, f"depth_{i:02d}.png"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump", help="dump directory or .zip")
    g.add_argument("--synth", help="synthetic scene kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-debug", default=None)
    ap.add_argument("--gt-plane", action="store_true", help="(synth) use the true plane instead of RANSAC on feature points")
    ap.add_argument("--ignore-plane-hint", action="store_true", help="(dump) ignore the plane stored in the manifest")
    args = ap.parse_args(argv)

    gt = None
    hint = None
    containers = {}
    if args.synth:
        import synth
        frames, gt = synth.make_scene(args.synth, args.seed)
        containers = gt["containers"]
        if args.gt_plane:
            hint = gt["plane"]
    else:
        import dump_io
        frames, man = dump_io.load_dump_with_meta(args.dump)
        containers = man.get("containers") or {}
        if not args.ignore_plane_hint:
            hint = man.get("plane_obj")

    plane, diags = align.align_frames(frames, plane_hint=hint)
    n_ok = sum(d["ok"] for d in diags)
    rm = [d["rmse_m"] * 100 for d in diags if d["ok"]]
    print(f"alignment: {n_ok}/{len(frames)} frames accepted"
          + (f", fit residual mean {np.mean(rm):.2f} cm" if rm else ""))
    for i, d in enumerate(diags):
        if not d["ok"]:
            print(f"  frame {i} rejected: {d['reason']}")
    if gt is not None:
        ang = np.degrees(np.arccos(np.clip(abs(plane.n @ gt["plane"].n), -1, 1)))
        dh = abs(gt["plane"].height(plane.n * plane.offset))
        print(f"plane: normal err {ang:.2f} deg, offset err {dh * 1000:.1f} mm")

    if n_ok == 0:
        print("no frame could be aligned; nothing to measure")
        return 1

    labels = {}
    keys = []
    for f in frames:
        for k in f.masks:
            if k not in keys:
                keys.append(k)
            if k in f.labels:
                labels.setdefault(k, Counter())[f.labels[k]] += 1
    lab = {k: labels[k].most_common(1)[0][0] for k in labels}
    items = [k for k in keys if not lab.get(k, "").startswith("container_")]

    hdr = f"{'item':<8}{'label':<22}{'method':<12}{'volume mL typ (min-max)':<28}{'GT mL':>8}{'err %':>8}{'cover':>7}"
    print(hdr)
    print("-" * len(hdr))
    for k in items:
        r = volume_mod.compute_item_volume(frames, k, plane, lab.get(k, "?"), container=containers.get(k))
        if r.volume_ml is None:
            vtxt = "fallback (failed)"
            err = ""
        else:
            v = r.volume_ml
            vtxt = f"{v.typ:.0f} ({v.min:.0f}-{v.max:.0f})"
            err = ""
        g = gt["volumes_ml"].get(k) if gt else None
        if g is not None and r.volume_ml is not None:
            err = f"{100 * (r.volume_ml.typ - g) / g:+.1f}"
        print(f"{k:<8}{lab.get(k, '?'):<22}{r.method:<12}{vtxt:<28}{(f'{g:.0f}' if g is not None else ''):>8}{err:>8}{r.coverage:>7.2f}")
        for n in r.notes:
            print(f"    note: {n}")
    if args.save_debug:
        save_debug(frames, items, plane, args.save_debug)
        print(f"debug images written to {args.save_debug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
