"""Qualitative sanity on the 22 Taiwanese photos and the user's negatives (neg: gate = shipped whole-image neg gate AND >=1 food component)."""
import json, pickle, sys
import numpy as np, torch
from common_i import *
import fit_params as F
from final_eval import dish_level, predict
P = json.load(open(Path(__file__).parent / "params.json"))
for name in ("tw", "neg"):
    segs = pickle.load(open(RI / f"seg_{name}.pkl", "rb")); print("==", name)
    logged = 0
    for k, s in segs.items():
        kc, a = predict(s, P)
        _, dn, pneg, top = dish_level(s["ew"])
        gate_ok = pneg <= 0.1 and top >= 0.05
        comps = [(F.rows[i]["id"], round(float(a[i]), 3)) for i in np.argsort(-a)[:4] if a[i] >= P["thr"]]
        logs = gate_ok and len(comps) > 0; logged += logs
        print(f"{k:28s} kcal {kc:5.0f} log={int(logs)} (pneg {pneg:.2f}, dish-level says '{dn[:28]}') comps {comps}")
    print(name, "logged", logged, "of", len(segs))
