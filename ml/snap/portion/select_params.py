"""Stage 4: pick container diameters on the FIT split, then calibrate the uncertainty sigma on the VALIDATION split; writes portion_params.json.
The 60-dish test set is not read here.   usage: python select_params.py [variant=gbeta] [gamma=1]"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import portion as P
from fit import *

HERE = Path(__file__).parent


def build_params(theta, Dd, s_item, s_dish=0.0):
    ks = dict(zip(P.GROUPS, theta["k"])); bs = dict(zip(P.GROUPS, theta["beta"]))
    return P.Params(dict(diam_cm=Dd, groups={g: dict(k=ks[g], beta=bs[g]) for g in P.GROUPS}, gamma=theta["gamma"], rho0=RHO0,
                         sigma_item=s_item, sigma_dish=s_dish, fallback_sigma_mult=1.5, area_cm2_min=0.5, tilt_clip=[0.5, 1.0],
                         food_lab_thr=18.0, ring=[0.80, 0.97], min_container_score=0.15, min_container_frac=0.12, container_dish_weight=0.8))


def dish_samples(D, Pm, n=2000, fallback_const=None):
    """samples per dish, using the dish's own container/areas; None if the dish has no logged items"""
    out = []
    for i, d in enumerate(D):
        its = d["F"]["items"]
        if not its: out.append(None); continue
        ar = [None if e is None else max(px * (Pm.diam(e["type"]) / e["major"]) ** 2 / min(1.0, max(0.5, e["minor"] / e["major"])), Pm.d["area_cm2_min"])
              for px, e in zip(d["px"], d["iell"])]
        out.append(P.sample_dish(V, GRP, Pm, its, ar, n=n, seed=1000 + i))
    return out


def coverage(D, S, fb_lo_hi, med_fit):
    hit = []; pred = []
    for d, s in zip(D, S):
        if s is None: lo, hi = fb_lo_hi; pred.append(med_fit)
        else: lo, hi = np.percentile(s, [10, 90]); pred.append(np.median(s))
        hit.append(lo <= d["kcal"] <= hi)
    return float(np.mean(hit)), pred


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "gbeta"; ug = (sys.argv[2] != "0") if len(sys.argv) > 2 else True
    labels = json.load(open(SP / "eval/n5k_dev/labels.json")); DA = load("dev", labels, "colour")
    Dfit = [d for d in DA if split_of(d["dish"]) == "fit"]; Dval = [d for d in DA if split_of(d["dish"]) == "val"]
    med_fit = float(np.median([d["kcal"] for d in Dfit])); fb = lambda d: typical(d) or med_fit
    q = np.percentile([d["kcal"] for d in Dfit], [10, 90]); print("fallback const", med_fit, "p10/p90", q)
    grid = {}
    # plate diameter is NOT identifiable from kcal alone (k absorbs it); fix it at the physically standard 26 cm and fit only the bowl relative to it
    for Dp in (26,):
        for Db in (12, 15, 18, 21, 24, 27):
            Dd = dict(plate=float(Dp), bowl=float(Db), tray=float(Dp)); th = fit_variant(Dfit, variant, Dd, use_gamma=ug)
            pf, _ = evaluate(Dfit, th, Dd, fb); m = metrics(pf, [d["kcal"] for d in Dfit])
            grid[(Dp, Db)] = (m["med_abs_pct"], th); print("D plate", Dp, "bowl", Db, "fit med|err| %.1f%%" % m["med_abs_pct"], flush=True)
    best = min(grid, key=lambda k: grid[k][0]); print("best on fit", best, grid[best][0])
    th = grid[best][1]; Dd = dict(plate=float(best[0]), bowl=float(best[1]), tray=float(best[0]))
    pv, _ = evaluate(Dval, th, Dd, fb); print("val (no sigma)", fmt(metrics(pv, [d["kcal"] for d in Dval])))
    # sigma calibration on validation: smallest s giving >= 80% coverage of p10-p90
    rows = []
    for s in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        Pm = build_params(th, Dd, s); S = dish_samples(Dval, Pm); cov, _ = coverage(Dval, S, q, med_fit); rows.append((s, cov)); print("sigma_item", s, "val coverage %.3f" % cov, flush=True)
    s_best = next((s for s, c in rows if c >= 0.80), rows[-1][0])
    # refine linearly between neighbours
    Pm = build_params(th, Dd, s_best); json.dump(dict(Pm.d, fallback_const_kcal=med_fit, fallback_p10_p90=q.tolist(), variant=f"{variant}+gamma" if ug else variant,
                                                      sigma_curve=rows, fit_grid={f"{a}_{b}": v[0] for (a, b), v in grid.items()}),
                                                 open(HERE / "portion_params.json", "w"), indent=1)
    print("wrote portion_params.json sigma_item", s_best, "theta", th, "D", Dd)
