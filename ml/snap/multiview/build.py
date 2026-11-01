"""Stage 3 (per dish): poses, depth-heightmap volumes per view, azimuth registration, hull volumes for camera subsets.
usage: python build.py <name>  -> $SP/research/mv/b_<name>.pkl  {dish: {...}}
Options through env: MV_D_PLATE (cm, default 26), MV_FSCALE (focal multiplier, default 1)."""
import sys, os, pickle, itertools, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1]))
from common import SP
import mv, rotation, hull, geom

MV = SP / "research/mv"
CAMS = "ABCD"


def build_dish(dish, name, prm, fscale=1.0, do_hull=True, do_depth=True, subsets=None):
    views = []
    for cam in CAMS:
        p = MV / f"s1_{name}" / f"{dish}_{cam}.pkl"
        if not p.exists(): continue
        d = pickle.load(open(p, "rb"))
        D = None
        if do_depth:
            q = MV / f"s2_{name}" / f"{dish}_{cam}.npy"
            if q.exists(): D = np.load(q)
        for t in range(len(d["idx"])):
            f = mv.Frame.make(d["frames"][t], d["res"][t], D[t] if D is not None else np.zeros((322, 574), np.float16), prm, fscale)
            v = dict(cam=cam, t=t, ok=f is not None)
            if f is not None:
                v["tilt"] = f.tilt; v["ell"] = f.e; v["f"] = f
                if D is not None:
                    r = mv.frame_heightmap(f, prm)
                    v["depth"] = None if r is None else {k: r[k] for k in r if k != "hm"}
                v["rect"] = rotation.rectify(d["frames"][t], f.n, f.c, f.K)
            views.append(v)
    ok = [v for v in views if v["ok"]]
    out = dict(views=[{k: v[k] for k in v if k not in ("f", "rect")} for v in views], nviews=len(views), nok=len(ok))
    if not ok: return out
    # azimuth to the reference view (first ok view in B, C, A, D order, t = 1)
    pref = sorted(ok, key=lambda v: ("BCAD".index(v["cam"]), abs(v["t"] - 1)))[0]
    for v in ok:
        if v is pref: v["phi"], v["phis"] = 0.0, 99.0
        else: v["phi"], v["phis"] = rotation.rel_angle(pref["rect"], v["rect"])
    for v, vv in zip([x for x in views if x["ok"]], out["views"][:0]): pass
    for o, v in zip(out["views"], views):
        if v["ok"]: o["phi"], o["phis"] = v["phi"], v["phis"]
    out["ref"] = (pref["cam"], pref["t"])
    if do_hull:
        out["hull"] = {}
        sils = {}
        for v in ok:
            fm = mv.food_mask(v["f"], prm)
            if fm is not None: sils[(v["cam"], v["t"])] = (v["f"], v["phi"], fm)
        subsets = subsets or ["".join(c) for k in (1, 2, 3, 4) for c in itertools.combinations(CAMS, k)]
        for ss in subsets:
            vs = [sils[k] for k in sils if k[0] in ss]
            if len(vs) >= 2:
                vol, occ, seen = hull.carve(vs)
                out["hull"][ss] = dict(vol=vol, n=len(vs))
    return out


if __name__ == "__main__":
    name = sys.argv[1]; fs = float(os.environ.get("MV_FSCALE", 1.0)); dp = float(os.environ.get("MV_D_PLATE", 26.0))
    prm = dict(D_plate=dp, D_bowl=dp * 24 / 26)
    dishes = sorted({p.stem[:-2] for p in (MV / f"s1_{name}").glob("*.pkl")})
    tag = os.environ.get("MV_TAG", "")
    R = {}; t0 = time.time()
    for i, d in enumerate(dishes):
        R[d] = build_dish(d, name, prm, fs, do_hull=os.environ.get("MV_HULL", "1") == "1")
        if i % 10 == 0: print(i, len(dishes), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(R, open(MV / f"b_{name}{tag}.pkl", "wb")); print("done")
