"""Save 6 example pairs (test + dev) with images and the A/B/C numbers -> $SP/research/mv/pairs/"""
import sys, json, pickle, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1]))
from common import SP
MV = SP / "research/mv"; out = MV / "pairs"; out.mkdir(exist_ok=True)
ALL, info = pickle.load(open(MV / "fr_test.pkl", "rb")); o, res, P = ALL["hull_only"] if len(sys.argv) > 1 else ALL["best_area_h"]
true = {i: (d, k, m) for i, (d, k, m) in enumerate(info)}
ex = []
srt = sorted(P, key=lambda p: -(info[p[0]][2] / info[p[1]][2])); sel, used = [], set()
for p in srt[::max(1, len(srt) // 12)]:   # spread over the ratio range, no dish twice (not cherry-picked: every k-th pair)
    if p[0] in used or p[1] in used: continue
    sel.append(p); used |= set(p)
for hi, lo in sel[:6]:
    row = {}
    for role, i in (("heavier", hi), ("lighter", lo)):
        d, k, m = info[i]; src = SP / "eval/n5k" / f"{d}.png"; shutil.copy(src, out / f"{d}.png")
        row[role] = dict(dish=d, image=f"{d}.png", true_kcal=round(k), true_mass_g=m, A=round(res[i]["A"][0]), B=round(res[i]["B"][0]), C=round(res[i]["C"][0]),
                         C_p10_p90=[round(res[i]["C"][1]), round(res[i]["C"][2])])
    h, l = row["heavier"], row["lighter"]
    row["true_ratio"] = round(h["true_kcal"] / l["true_kcal"], 2); row["est_ratio"] = {m: round(h[m] / max(l[m], 1), 2) for m in "ABC"}
    ex.append(row)
json.dump(ex, open(out / "pairs.json", "w"), indent=1); print(json.dumps(ex, indent=1)[:2500])
