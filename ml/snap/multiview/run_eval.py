"""Stage 5: A/B/C comparison. python run_eval.py dev  (choose config, fit C, calibrate)  ->  python run_eval.py test (frozen)."""
import sys, json, pickle, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from evalmv import *

which = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""
FALL = PM.d["fallback_const_kcal"]; FQ = PM.d["fallback_p10_p90"]
CONF = [("area", c) for c in ("B", "C", "BC", "ABC")] + [(m, c) for m in ("ah0.5", "ah1", "ah2", "ah4") for c in ("B", "BC", "ABC", "ABCD")] + [(m, c) for m in ("depth", "hull", "min") for c in ("A", "B", "C", "D", "BC", "ABC", "ABCD")]


def vols_for(Ds, m, c): return [volume(d.rec, m, c) for d in Ds]


def tabulate(Ds):
    """Pearson(log V, log mass) and the median |log(k V / mass)| (k = median ratio on the fit dishes) for every config"""
    mass = np.array([d.mass for d in Ds]); fit = np.array([split_of(d.id) == "fit" for d in Ds]); out = []
    for m, c in CONF:
        v = np.array([x if x else np.nan for x in vols_for(Ds, m, c)]); ok = ~np.isnan(v) & (v > 0)
        if ok.sum() < 8: continue
        r = float(np.corrcoef(np.log(v[ok]), np.log(mass[ok]))[0, 1]); k = np.exp(np.median(np.log(mass[ok & fit] / v[ok & fit])))
        lr = np.abs(np.log(k * v[ok] / mass[ok])); out.append(dict(method=m, cams=c, n=int(ok.sum()), r_log=r, med_abs_logratio=float(np.median(lr)), r_lin=float(np.corrcoef(v[ok], mass[ok])[0, 1])))
    return out


def predict_all(Ds, cfg, th, sig_C, fA=1.0, fB=1.0, with_B_fallback=True):
    """per dish: dict(A=(med,lo,hi), B=..., C=..., ok_scan). Items/recognition identical."""
    SB = samples_B(Ds); res = []
    for d, sb in zip(Ds, SB):
        sa = samples_A(d)
        if sa is None:  # no logged item: every method gets the same fallback
            r = {k: (FALL, FQ[0], FQ[1]) for k in "ABC"}; r["ok_scan"] = False; r["noitem"] = True; res.append(r); continue
        r = dict(noitem=False)
        r["A"] = tuple(np.percentile(rescale_samples(sa, fA), [50, 10, 90]))
        r["B"] = tuple(np.percentile(rescale_samples(sb, fB), [50, 10, 90])) if sb is not None else (FALL, FQ[0], FQ[1])
        v = volume(d.rec, *cfg)
        if v is None or v <= 0:
            r["C"] = r["B"]; r["ok_scan"] = False
        else:
            s = C_samples(d, v, th, sig_C, seed=hash(d.id) % 1000); r["C"] = tuple(np.percentile(s, [50, 10, 90])); r["ok_scan"] = True
        res.append(r)
    return res


def table(Ds, res, subset=None):
    true = np.array([d.kcal for d in Ds]); out = {}
    idx = np.arange(len(Ds)) if subset is None else np.where(subset)[0]
    for m in "ABC":
        med = np.array([r[m][0] for r in res]); lo = np.array([r[m][1] for r in res]); hi = np.array([r[m][2] for r in res])
        e = metrics(med[idx], true[idx]); e["cover"] = float(np.mean((lo[idx] <= true[idx]) & (true[idx] <= hi[idx])) * 100); e["p90_p10"] = float(np.median(hi[idx] / np.maximum(lo[idx], 1)))
        out[m] = e
    return out


def calibrate(Ds, cfg, th):
    """multiplier on the log-spread of A, B and sigma_C so that p10-p90 covers ~80% of the calibration dishes"""
    true = np.array([d.kcal for d in Ds])
    def cov(fn, x): r = fn(x); return float(np.mean([(r[i][1] <= true[i] <= r[i][2]) for i in range(len(Ds))]))
    best = {}
    for name, key in (("fA", "A"), ("fB", "B")):
        cs = []
        for f in np.arange(0.4, 2.01, 0.1):
            r = predict_all(Ds, cfg, th, 0.5, **{name: f}); cs.append((abs(np.mean([(r[i][key][1] <= true[i] <= r[i][key][2]) for i in range(len(Ds))]) - 0.8), f))
        best[name] = min(cs)[1]
    cs = []
    for s in np.arange(0.2, 1.21, 0.1):
        r = predict_all(Ds, cfg, th, s); cs.append((abs(np.mean([(r[i]["C"][1] <= true[i] <= r[i]["C"][2]) for i in range(len(Ds))]) - 0.8), s))
    best["sC"] = min(cs)[1]
    return best


def pairs(Ds, res, dom, min_ratio=1.5):
    P = []
    for i, j in itertools.combinations(range(len(Ds)), 2):
        a, b = Ds[i], Ds[j]
        if a.id not in dom or b.id not in dom or dom[a.id][0] != dom[b.id][0]: continue
        hi, lo = (i, j) if a.mass >= b.mass else (j, i)
        if Ds[hi].mass / max(Ds[lo].mass, 1) < min_ratio or abs(Ds[hi].kcal - Ds[lo].kcal) < 1e-6: continue
        P.append((hi, lo))
    out = {}
    for m in "ABC":
        ok = []; ratio = []
        for hi, lo in P:
            ph, pl = res[hi][m][0], res[lo][m][0]; th_, tl = Ds[hi].kcal, Ds[lo].kcal
            truth_hi = th_ > tl   # does the heavier dish truly have more kcal?
            ok.append(0.5 if abs(ph - pl) < 1e-6 else float((ph > pl) == truth_hi)); ratio.append((ph / max(pl, 1e-6)) / (th_ / tl))
        out[m] = dict(n=len(P), acc=float(np.mean(ok) * 100) if ok else float("nan"), med_ratio=float(np.median(ratio)) if ratio else float("nan"), med_abs_logratio=float(np.median(np.abs(np.log(ratio)))) if ratio else float("nan"))
    return out, P


if __name__ == "__main__":
    Ds = load_dishes(which, TAG); dom = dominant_ingredient()
    print(which, len(Ds), "dishes with video;", sum(1 for d in Ds if d.rec and d.rec["nok"] > 0), "with >=1 pose")
    if which == "dev":
        tab = tabulate(Ds)
        for t in sorted(tab, key=lambda t: -t["r_log"]): print(f"{t['method']:6s} {t['cams']:5s} n={t['n']:3d} r_log={t['r_log']:.2f} r_lin={t['r_lin']:.2f} med|logratio|={t['med_abs_logratio']:.2f}")
        json.dump(tab, open(MV / f"tab_dev{TAG}.json", "w"), indent=1)


def fit_cfg(Ds, cfg):
    Dfit = [d for d in Ds if split_of(d.id) == "fit"]
    return fit_C(Dfit, [volume(d.rec, *cfg) for d in Dfit], grid_b=(0.6, 0.8, 1.0))


def report(Ds, cfg, th, cal, tag, subs=None):
    res = predict_all(Ds, cfg, th, cal["sC"], cal["fA"], cal["fB"])
    true = np.array([d.kcal for d in Ds]); big = true >= 300; scan = np.array([r["ok_scan"] for r in res])
    out = dict(cfg=cfg, th=th, cal=cal, all=table(Ds, res), meal=table(Ds, res, big), scan_ok=table(Ds, res, scan), n=len(Ds), n_meal=int(big.sum()), scan_fail=float(1 - scan.mean()))
    pr, P = pairs(Ds, res, dom); out["pairs"] = pr; out["n_pairs"] = len(P)
    # volume signal on scanned dishes
    v = np.array([volume(d.rec, *cfg) or np.nan for d in Ds]); m = np.array([d.mass for d in Ds]); ok = ~np.isnan(v) & (v > 0)
    out["vol_r_log"] = float(np.corrcoef(np.log(v[ok]), np.log(m[ok]))[0, 1])
    # what the true weighed mass would give with the same density treatment (oracle)
    return out, res, P


if __name__ == "__main__":
    pass


def main():
    Ds = load_dishes(which, TAG)
    FR = {}
    if which == "dev":
        cfgs = {"best_area_h": ("ah2", "ABC"), "depth_only": ("depth", "ABCD"), "hull_only": ("hull", "ABC")}
        for name, cfg in cfgs.items():
            th = fit_cfg(Ds, cfg); Dval = [d for d in Ds if split_of(d.id) == "val"]
            cal = calibrate(Ds, cfg, th)   # calibrated on all dev dishes (fit+val): 3 numbers only
            o, res, P = report(Dval, cfg, th, cal, name); o_all, _, _ = report(Ds, cfg, th, cal, name)
            FR[name] = dict(val=o, dev_all=o_all)
            print("==", name, cfg, "theta(k,beta,gamma)=", th, "cal", cal)
            for part in ("val", "dev_all"):
                x = FR[name][part]; print(" ", part, "n", x["n"], "meal", x["n_meal"], "pairs", x["n_pairs"], "scan_fail", round(x["scan_fail"], 3), "vol r_log", round(x["vol_r_log"], 2))
                for k in "ABC": print("   ", k, {a: round(b, 2) for a, b in x["all"][k].items()}, "| meal", {a: round(b, 1) for a, b in x["meal"][k].items() if a in ("med", "w25")}, "| pairs", {a: round(b, 2) for a, b in x["pairs"][k].items() if a != "n"})
        # arc / frames table for the chosen config: camera subsets and frames per camera
        rows = []
        for cams in ("B", "C", "BC", "ABC", "ABCD"):
            for ts in ((1,), (1, 2), (0, 1, 2, 3)):
                cfg = ("ah2", cams); th = fit_cfg(Ds, cfg)
                vs = [volume(d.rec, "ah2", cams, ts) for d in Ds]
                Dv = [d for d in Ds if split_of(d.id) == "val"]
                pred = [C_median_kcal(d, volume(d.rec, "ah2", cams, ts), th) for d in Dv]
                okv = [(p, d.kcal) for p, d in zip(pred, Dv) if p]; m = metrics([a for a, _ in okv], [b for _, b in okv])
                arc = float(np.median([arc_span(d.rec, cams, ts) for d in Ds if d.rec]))
                rows.append(dict(cams=cams, frames_per_cam=len(ts), views=len(cams) * len(ts), arc_deg_median=arc, val_med_err=m["med"], val_w25=m["w25"], n=m["n"]))
                print(rows[-1])
        FR["arc_table"] = rows
        FR["chosen"] = dict(cfg=("ah2", "ABC"), th=fit_cfg(Ds, ("ah2", "ABC")), cal=calibrate(Ds, ("ah2", "ABC"), fit_cfg(Ds, ("ah2", "ABC"))))
        pickle.dump(FR, open(MV / f"fr_dev{TAG}.pkl", "wb"))
    else:
        FR = pickle.load(open(MV / "fr_dev.pkl", "rb")); ALL = {}
        for name in ("best_area_h", "depth_only", "hull_only"):
            v = FR[name]["val"]; o, res, P = report(Ds, tuple(v["cfg"]), v["th"], v["cal"], "test"); ALL[name] = (o, res, P)
            print("== TEST", name, "n", o["n"], "meal", o["n_meal"], "pairs", o["n_pairs"], "scan_fail", round(o["scan_fail"], 3), "vol r_log", round(o["vol_r_log"], 2))
            for part in ("all", "meal", "scan_ok"):
                for k in "ABC": print(part, k, {a: round(b, 2) for a, b in o[part][k].items()})
            for k in "ABC": print("pairs", k, {a: round(b, 2) for a, b in o["pairs"][k].items()})
        pickle.dump((ALL, [(d.id, d.kcal, d.mass) for d in Ds]), open(MV / f"fr_test{TAG}.pkl", "wb"))


dom = dominant_ingredient()
main()
