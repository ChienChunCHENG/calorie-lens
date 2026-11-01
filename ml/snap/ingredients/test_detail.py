import csv, json, pickle, numpy as np
from common_i import *
import fit_params as F
from final_eval import predict, test, segs
P = json.load(open(Path(__file__).parent / "params.json"))
meta = {}
for f in ("meta1.csv", "meta2.csv"):
    for r in csv.reader(open(SP / "eval/n5k_dev" / f)):
        ing = [dict(name=r[i + 1], g=float(r[i + 2]), kcal=float(r[i + 3])) for i in range(6, len(r) - 6, 7)]
        meta[r[0]] = dict(kcal=float(r[1]), ingr=ing)
T, A, rows_ = [], [], []
for d in test:
    k, a = predict(segs[d["dish"]], P); t = np.zeros(F.nK)
    for i in meta[d["dish"]]["ingr"]:
        kk = F.n2k.get(i["name"])
        if kk: t[F.kid[kk]] += i["kcal"]
    T.append(t); A.append(a)
    top = [(F.rows[j]["id"], round(float(a[j]), 3)) for j in np.argsort(-a)[:3] if a[j] >= P["thr"]]
    tt = [(F.rows[j]["id"], round(float(t[j]))) for j in np.argsort(-t)[:3] if t[j] > 0]
    rows_.append((d["dish"], d["kcal"], k, top, tt))
T, A = np.array(T), np.array(A)
print("test kcal-weighted ingredient recall (area>=thr):", F.recall(A, T, P["thr"]))
rows_.sort(key=lambda r: -abs(r[2] - r[1]) / r[1])
for r in rows_[:6]: print("OVER/WORST", r[0], round(r[1]), round(r[2]), "pred", r[3], "true", r[4])
under = sorted(rows_, key=lambda r: (r[2] - r[1]) / r[1])[:4]
for r in under: print("UNDER", r[0], round(r[1]), round(r[2]), "pred", r[3], "true", r[4])
