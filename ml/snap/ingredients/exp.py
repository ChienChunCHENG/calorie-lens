"""Dev experiments (fit split -> validation split). Writes dev_results.json and params.json (frozen for final_eval.py).
Nothing here touches the 60 test dishes."""
import json, pickle, math, sys
import numpy as np, torch
from common_i import *
import fit_params as F
from fit_params import *

segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = dev_labels()
fitL = [x for x in d if split(x["dish"]) == "fit" and x["dish"] in segs]; valL = [x for x in d if split(x["dish"]) == "val" and x["dish"] in segs]
print("fit", len(fitL), "val", len(valL))
R = {}


def run(tag, kw, grouped=True, fix_b=None, fix_oil=None, thr=0.004, lam=0.05):
    Xf, Tf, Yf, _, _ = build(segs, fitL, **kw); Xv, Tv, Yv, idv, _ = build(segs, valL, **kw)
    m = fit(Xf, Yf, grouped=grouped, fix_b=fix_b, fix_oil=fix_oil, thr=thr, lam=lam)
    with torch.no_grad(): pv = m(Xv, thr).numpy()
    s = score(pv, Yv, tag); s["recall"] = recall(Xv, Tv, thr); s["b"] = float(m.b); s["oil"] = float(torch.relu(m.oil))
    R[tag] = s; print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "recall"}, "recall(kcal-weighted, mean, prec)", [round(x, 3) for x in s["recall"]], flush=True)
    return m, (Xf, Yf, Xv, Yv, idv, pv)


base = dict(mix=0.5, temp=1.0)
for nm in ("plate", "image"):
    for mix in (0.0, 0.5, 1.0): run(f"norm={nm} mix={mix}", dict(mix=mix, temp=1.0, norm=nm))
    run(f"norm={nm} hard", dict(mix=0.5, temp=1.0, hard=True, norm=nm))
    run(f"norm={nm} temp=0.5", dict(mix=0.5, temp=0.5, norm=nm))
    run(f"norm={nm} b=1", dict(mix=0.5, temp=1.0, norm=nm), fix_b=1.0)
    run(f"norm={nm} oil=0", dict(mix=0.5, temp=1.0, norm=nm), fix_oil=0.0)
    run(f"norm={nm} ungrouped", dict(mix=0.5, temp=1.0, norm=nm), grouped=False)
json.dump(R, open(Path(__file__).parent / "dev_results.json", "w"), indent=1)
