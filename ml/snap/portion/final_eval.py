"""Stage 5: evaluate portion_params.json against baselines. `python final_eval.py val` (dev validation split) or `python final_eval.py test` (the 60-dish
held-out set; run ONCE, after the parameters are frozen). Writes $SP/research/portion/eval_<which>.json."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import portion as P
from fit import *
from select_params import dish_samples
from snaplib import mixture

which = sys.argv[1]
Pm = P.Params.load(Path(__file__).parent / "portion_params.json")
if which == "val":
    DA = load("dev", json.load(open(SP / "eval/n5k_dev/labels.json")), "colour"); D = [d for d in DA if split_of(d["dish"]) == "val"]
else:
    lab = [dict(dish=x["dish"], kcal=x["kcal"], mass_g=x["mass_g"], ingr=x["ingr"]) for x in json.load(open(SP / "eval/n5k/labels.json"))]
    D = load("test", lab, "colour")
ingr = {x["dish"]: x.get("ingr") for x in json.load(open(SP / ("eval/n5k_dev/labels.json" if which == "val" else "eval/n5k/labels.json")))}
fitlab = json.load(open(SP / "eval/n5k_dev/labels.json")); DF = [d for d in load("dev", fitlab, "colour") if split_of(d["dish"]) == "fit"]
med_fit = Pm.d["fallback_const_kcal"]; q = Pm.d["fallback_p10_p90"]
true = np.array([d["kcal"] for d in D]); n = len(D)
has_items = np.array([bool(d["F"]["items"]) for d in D]); has_cont = np.array([all(e is not None for e in d["iell"]) for d in D])


def interval_cov(lo, hi): return float(np.mean([(l <= t <= h) for l, h, t in zip(lo, hi, true)]) * 100)


def width(lo, hi): return float(np.median(np.array(hi) / np.maximum(np.array(lo), 1)))


R = {}
# (A) portion method
S = dish_samples(D, Pm)
pa, lo_a, hi_a = [], [], []
for s in S:
    if s is None: pa.append(med_fit); lo_a.append(q[0]); hi_a.append(q[1])
    else: l, m, h = np.percentile(s, [10, 50, 90]); pa.append(m); lo_a.append(l); hi_a.append(h)
# (B) current typical-portion pipeline (shipped mixture); dishes with no logged item fall back to the fit median like the others
pb, lo_b, hi_b = [], [], []
for d in D:
    its = d["F"]["items"]
    if not its: pb.append(med_fit); lo_b.append(q[0]); hi_b.append(q[1]); continue
    tot = np.zeros(4000)
    for k, it in enumerate(its): tot += mixture(V, [w for _, w in it["cand"]], [c for c, _ in it["cand"]], seed=1234 + k)
    l, m, h = np.percentile(tot, [10, 50, 90]); pb.append(m); lo_b.append(l); hi_b.append(h)
R["portion"] = (pa, lo_a, hi_a); R["typical"] = (pb, lo_b, hi_b); R["const350"] = ([350.0] * n, [q[0]] * n, [q[1]] * n); R["const_fitmedian"] = ([med_fit] * n, [q[0]] * n, [q[1]] * n)
out = dict(n=n, no_items=int((~has_items).sum()), no_container=int((has_items & ~has_cont).sum()), methods={})
print(which, "n", n, "dishes with no logged item (fallback const)", out["no_items"], "items but no container", out["no_container"])
for k, (p, lo, hi) in R.items():
    p = np.array(p); m = metrics(p, true); m["coverage_p10_p90"] = interval_cov(lo, hi); m["median_width_ratio_p90_over_p10"] = width(lo, hi)
    sub = has_items; ms = metrics(p[sub], true[sub]); m["logged_only"] = dict(n=int(sub.sum()), med_abs_pct=ms["med_abs_pct"], within25=ms["within25"], coverage=float(np.mean([(lo[i] <= true[i] <= hi[i]) for i in np.where(sub)[0]]) * 100))
    out["methods"][k] = m
    print(f"{k:16s} {fmt(m)}  cov {m['coverage_p10_p90']:.0f}%  width x{m['median_width_ratio_p90_over_p10']:.1f} | logged-only n={ms and int(sub.sum())}: med {ms['med_abs_pct']:.1f}% cov {m['logged_only']['coverage']:.0f}%")
# failure cases for the portion method
p = np.array(R["portion"][0]); err = (p - true) / true; order = np.argsort(-np.abs(err))[:10]; fails = []
for i in order:
    d = D[i]; its = d["F"]["items"]
    top = [(V[it["cand"][0][0]]["en"], round(it["cand"][0][1], 2)) for it in its]
    fails.append(dict(dish=d["dish"], true_kcal=round(d["kcal"]), pred=round(p[i]), true_mass=d["mass"], err_pct=round(err[i] * 100), plate=(d["ell"] or {}).get("type"), items=top,
                      food_px=sum(d["px"]), ingr=(ingr.get(d["dish"]) or [])[:6]))
out["failures"] = fails
for f in fails: print(f)
json.dump(out, open(RP / f"eval_{which}.json", "w"), indent=1, default=float)
