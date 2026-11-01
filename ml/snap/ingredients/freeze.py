"""Freeze params on dev: fit on FIT, calibrate interval on VAL, then refit on FIT+VAL with the same hyperparameters. Writes params.json.
Shippable config: plate-normalised area, mix 0.5, temp 1, soft labels, b fixed 1 (grams linear in area), grouped K, fitted oil allowance."""
import json, pickle, math
import numpy as np, torch
from common_i import *
from fit_params import *
CFG = dict(mix=0.5, temp=1.0, hard=False, norm="plate")
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = dev_labels()
fitL = [x for x in d if split(x["dish"]) == "fit"]; valL = [x for x in d if split(x["dish"]) == "val"]
Xf, Tf, Yf, _, _ = build(segs, fitL, **CFG); Xv, Tv, Yv, idv, _ = build(segs, valL, **CFG)
THR = 0.004
m = fit(Xf, Yf, fix_b=1.0, thr=THR)
with torch.no_grad(): pf = m(Xf, THR).numpy(); pv = m(Xv, THR).numpy()
print("fit", score(pf, Yf)); print("val", score(pv, Yv))
# optional shrinkage toward the fit median in log space (alpha fitted on FIT)
lc = np.log(np.median(Yf)); al = np.linspace(0.5, 1.0, 26)
best = min(al, key=lambda a: np.median(np.abs(np.exp(a * np.log(np.maximum(pf, 5)) + (1 - a) * lc) - Yf) / Yf))
sv = np.exp(best * np.log(np.maximum(pv, 5)) + (1 - best) * lc); print("shrink alpha", best, "val", score(sv, Yv))
# interval: empirical quantiles of ln(true/pred) on VAL; honest 2-fold check
r = np.log(Yv / np.maximum(pv, 5)); q = [float(np.quantile(r, .1)), float(np.quantile(r, .9))]
print("resid q10,q90", q, "val coverage (in-sample)", float(((r >= q[0]) & (r <= q[1])).mean()))
idx = np.random.default_rng(0).permutation(len(r)); h1, h2 = idx[:len(r) // 2], idx[len(r) // 2:]
cov = [float(((r[b] >= np.quantile(r[a], .1)) & (r[b] <= np.quantile(r[a], .9))).mean()) for a, b in ((h1, h2), (h2, h1))]; print("2-fold coverage", cov)
# refit on fit+val
Xa = np.concatenate([Xf, Xv]); Ya = np.concatenate([Yf, Yv]); ma = fit(Xa, Ya, fix_b=1.0, thr=THR)
P = dict(CFG, grouped=True, thr=THR, groups=GROUPS, lk=[float(v) for v in ma.lk.data], b=1.0, oil=float(torch.relu(ma.oil)), resid_q=q, fit_n=len(Ya), val_cov_2fold=cov)
json.dump(P, open(Path(__file__).parent / "params.json", "w"), indent=1)
print({g: round(math.exp(v)) for g, v in zip(GROUPS, P["lk"])}, "oil", P["oil"])
