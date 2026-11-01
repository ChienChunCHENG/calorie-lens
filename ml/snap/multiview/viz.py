"""Visualisations -> $SP/research/mv/vis/. python viz.py <name> <dish> ...  (per-view plate ellipse, food mask, depth-height map)"""
import sys, pickle
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1]))
from common import SP
import mv

MV = SP / "research/mv"


def view_sheet(name, dish, out, ts=(1, 3), prm=None):
    prm = prm or {}; rows = []
    for cam in "ABCD":
        p = MV / f"s1_{name}/{dish}_{cam}.pkl"
        if not p.exists(): continue
        d = pickle.load(open(p, "rb")); q = MV / f"s2_{name}/{dish}_{cam}.npy"; D = np.load(q) if q.exists() else None; r = []
        for t in ts:
            if t >= len(d["idx"]): continue
            f = mv.Frame.make(d["frames"][t], d["res"][t], D[t] if D is not None else np.zeros((322, 574), np.float16), prm)
            im = d["frames"][t].copy(); hmv = np.zeros_like(im)
            if f is not None:
                fm = mv.food_mask(f, prm)
                if fm is not None: im[fm] = (0.5 * im[fm] + np.array([0, 120, 0])).astype(np.uint8)
                e = f.e; cv2.ellipse(im, ((e["cx"], e["cy"]), (e["major"], e["minor"]), e["angle"]), (255, 0, 0), 2)
                cv2.putText(im, f"{cam} tilt {f.tilt:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if D is not None:
                    r_ = mv.frame_heightmap(f, prm)
                    if r_ is not None:
                        hm = cv2.resize((np.clip(r_["hm"] / 6, 0, 1) * 255).astype(np.uint8), (200, 200), interpolation=cv2.INTER_NEAREST)
                        hmv[:200, :200] = cv2.applyColorMap(hm, cv2.COLORMAP_VIRIDIS)[..., ::-1]
                        cv2.putText(hmv, f"V {r_['vol_ml']:.0f}mL A {r_['area_cm2']:.0f}cm2 h {r_['hmean']:.1f}", (8, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            r += [cv2.resize(im, (320, 180)), cv2.resize(hmv, (320, 180))]
        if r: rows.append(np.hstack(r))
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    name = sys.argv[1]; (MV / "vis").mkdir(exist_ok=True)
    for dish in sys.argv[2:]: view_sheet(name, dish, MV / "vis" / f"{name}_{dish}.jpg")
