"""Stage 4: fit/validate/test of volume -> kcal for methods A (typical portion), B (single-photo plate ruler), C (multi-view size), same recognition.
Library + CLI:  python evalmv.py dev      (fit on fit split, report fit/val)   ;  python evalmv.py test (frozen params from dev; run once)"""
import sys, os, json, pickle, hashlib, math, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parents[1] / "portion")); sys.path.insert(0, str(Path(__file__).parent))
from common import SP, MODELS
import portion as PT
from fit import load as pload, split_of, V as VOC, GRP, K100, TYP
from select_params import dish_samples
from snaplib import mixture

MV = SP / "research/mv"
PM = PT.Params.load(Path(__file__).parents[1] / "portion/portion_params.json")
RHO0 = 1.2


def dominant_ingredient():
    """dish -> (dominant ingredient name by kcal share, its share, total grams)"""
    out = {}
    for fn in ("meta_cafe1.csv", "meta_cafe2.csv"):
        for row in csv.reader(open(MV / fn, encoding="utf-8")):
            if len(row) < 6: continue
            ing = []
            for k in range(6, len(row) - 6, 7):
                try: ing.append((row[k + 1], float(row[k + 2]), float(row[k + 3])))
                except ValueError: pass
            if not ing: continue
            tot = sum(x[2] for x in ing) or 1
            n, g, kc = max(ing, key=lambda x: x[2])
            out[row[0]] = (n, kc / tot, float(row[2]))
    return out


def labels(which):
    p = SP / ("eval/n5k_dev/labels.json" if which == "dev" else "eval/n5k/labels.json")
    return [dict(dish=x["dish"], kcal=x["kcal"], mass_g=x["mass_g"], ingr=x["ingr"]) for x in json.load(open(p))]


def dens(c, gamma): return max(K100[c] / 100.0, 0.02) ** gamma * RHO0 ** (1 - gamma)


class Dish:
    """one dish: truth, recognition (identical for A, B, C), and the multi-view record."""
    def __init__(self, d, rec):
        self.id = d["dish"]; self.kcal = d["kcal"]; self.mass = d["mass"]; self.F = d["F"]; self.rec = rec; self.D = d
        its = d["F"]["items"]; self.items = its
        px = np.array(d["px"], float) if len(its) else np.zeros(0)
        self.share = px / px.sum() if len(its) and px.sum() > 0 else (np.ones(len(its)) / max(len(its), 1))

    def kcal_per_g(self, gamma):
        """share-weighted expected kcal per gram over items and candidates (same weights as the shipped mixture)"""
        s = 0.0
        for sh, it in zip(self.share, self.items):
            w = np.array([w for _, w in it["cand"]]); w = w / w.sum()
            s += sh * sum(wi * dens(c, gamma) for wi, (c, _) in zip(w, it["cand"]))
        return s


def pick_views(rec, cams="ABCD", ts=(0, 1, 2, 3), method="depth"):
    vs = [v for v in rec["views"] if v["ok"] and v["cam"] in cams and v["t"] in ts]
    if method == "depth": vs = [v for v in vs if v.get("depth")]
    return vs


def volume(rec, method="depth", cams="ABCD", ts=(0, 1, 2, 3), agg="median", q=None):
    """dish volume (mL) or None."""
    if rec is None or rec["nok"] == 0: return None
    if method == "depth":
        vs = pick_views(rec, cams, ts, "depth")
        if not vs: return None
        x = np.array([v["depth"]["vol_ml"] for v in vs])
        return float(np.median(x)) if agg == "median" else float(np.percentile(x, q))
    if method == "hull":
        h = rec.get("hull", {}).get("".join(sorted(cams)))
        return None if h is None else h["vol"]
    if method.startswith("ah"):  # footprint area * (mean depth height + h0 cm): h0 = typical thickness prior
        h0 = float(method[2:]); vs = pick_views(rec, cams, ts, "depth")
        return float(np.median([v["depth"]["area_cm2"] * (max(v["depth"]["hmean"], 0) + h0) for v in vs])) if vs else None
    if method == "area":
        vs = pick_views(rec, cams, ts, "depth")
        return float(np.median([v["depth"]["area_cm2"] for v in vs])) if vs else None
    if method == "min":  # hull as an upper bound for the depth volume
        a, b = volume(rec, "depth", cams, ts), volume(rec, "hull", cams, ts)
        return a if b is None else (b if a is None else min(a, b))
    raise ValueError(method)


def arc_span(rec, cams, ts=(0, 1, 2, 3)):
    """azimuth span (deg) of the selected views (smallest arc containing them, from the registered angles)."""
    vs = [v for v in rec["views"] if v["ok"] and v["cam"] in cams and v["t"] in ts and "phi" in v]
    if len(vs) < 2: return 0.0
    a = np.sort(np.mod([v["phi"] for v in vs], 360)); gaps = np.diff(np.r_[a, a[0] + 360])
    return float(360 - gaps.max())


def fit_C(Dfit, vols, grid_g=np.arange(0.2, 1.01, 0.1), grid_b=(1.0,)):
    """-> (k, beta, gamma) minimising median |err| on the fit dishes (L1 grid) with items and a volume."""
    best = None
    use = [(d, v) for d, v in zip(Dfit, vols) if v and d.items]
    y = np.array([d.kcal for d, _ in use]); lv = np.log([v for _, v in use])
    for g in grid_g:
        s = np.array([d.kcal_per_g(g) for d, _ in use])
        for b in grid_b:
            res = np.log(y) - np.log(s) - b * lv; k = float(np.exp(np.median(res)))
            e = np.median(np.abs(k * s * np.exp(b * lv) / y - 1))
            if best is None or e < best[0]: best = (e, k, b, g)
    return best[1:]


def C_median_kcal(d, v, th):
    k, b, g = th
    if v is None or not d.items: return None
    return k * (v ** b) * d.kcal_per_g(g)


def C_samples(d, v, th, sigma, n=2000, seed=0):
    """samples of dish kcal: grams = k V^b with a shared log-normal error sigma; kcal/g drawn per item candidate."""
    k, b, g = th; rng = np.random.default_rng(seed + 7)
    G = k * v ** b * np.exp(sigma * rng.standard_normal(n)); tot = np.zeros(n)
    for sh, it in zip(d.share, d.items):
        w = np.array([w for _, w in it["cand"]]); w = w / w.sum()
        c = rng.choice(len(w), size=n, p=w); dd = np.array([dens(cc, g) for cc, _ in it["cand"]])
        tot += G * sh * dd[c]
    return tot


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float); e = np.abs(pred - true) / true
    r = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 2 else float("nan")
    return dict(n=len(true), med=float(np.median(e) * 100), w25=float(np.mean(e <= 0.25) * 100), pearson=r)


# ---------------------------------------------------------------------------------------------------------------------------------------
def load_dishes(which, tag=""):
    lab = labels(which)
    DA = pload(which, lab, "colour")
    B = pickle.load(open(MV / f"b_{which}{tag}.pkl", "rb"))
    return [Dish(d, B.get(d["dish"])) for d in DA if d["dish"] in B]


def samples_A(d, n=4000):
    its = d.F["items"]
    if not its: return None
    tot = np.zeros(n)
    for k, it in enumerate(its): tot += mixture(VOC, [w for _, w in it["cand"]], [c for c, _ in it["cand"]], seed=1234 + k)
    return tot


def samples_B(Ds):
    return dish_samples([d.D for d in Ds], PM)


def interval_stats(samples, true, lo_q=10, hi_q=90):
    lo = np.array([np.percentile(s, lo_q) for s in samples]); hi = np.array([np.percentile(s, hi_q) for s in samples]); med = np.array([np.percentile(s, 50) for s in samples])
    return med, lo, hi


def rescale_samples(s, f):
    """widen/narrow a sample cloud in log space about its median by factor f"""
    m = np.median(s); return m * np.exp(f * np.log(np.maximum(s, 1e-6) / m))
