"""ONE-TIME final evaluation on the 60 Nutrition5k test dishes with params.json frozen on dev. usage: python final_eval.py
Also: shipped-style dish-level baseline on the same cached whole-image embedding, constant-350 baseline, TW and negatives sanity."""
import sys, json, pickle, math
import numpy as np, torch
from common_i import *
import fit_params as F, ingr, ingr_vocab as V
from common import MODELS

test = json.load(open(SP / "eval/n5k/labels.json"))
segs = pickle.load(open(RI / "seg_test.pkl", "rb"))


def predict(seg, P, mix=None, hard=None):
    a, s = F.features(seg, mix=P["mix"], temp=P["temp"], hard=P["hard"], norm=P["norm"])
    m = F.Model(P["grouped"]); m.lk.data = torch.tensor(P["lk"]); m.b.data = torch.tensor(P["b"]); m.oil.data = torch.tensor(P["oil"])
    with torch.no_grad(): k = float(m(a[None], P["thr"])[0])
    return k, a


# ---- shipped dish-level (whole image) for comparison: top-1 typical kcal from the shipped vocab
V0 = json.load(open(MODELS / "vocab.json", encoding="utf-8")); M0 = np.frombuffer((MODELS / "vocab_emb.bin").read_bytes(), "<f2").astype(np.float32).reshape(len(V0), -1)
NEG0 = np.array([bool(v.get("neg")) for v in V0])


def dish_level(ew):
    z = 64.25 * (M0 @ ew); p = np.exp(z - z.max()); p /= p.sum()
    pneg = p[NEG0].sum(); pf = np.where(NEG0, 0, p); i = int(pf.argmax()); v = V0[i]
    return v["portion_g"][1] * v["kcal100"] / 100, V0[i]["en"], float(pneg), float(pf.max())


def report(name, pred, Y, lo=None, hi=None):
    e = np.abs(pred - Y) / Y; big = Y >= 300
    out = dict(name=name, n=len(Y), med=float(np.median(e)), w25=float((e <= .25).mean()), r=float(np.corrcoef(pred, Y)[0, 1]),
               med_big=float(np.median(e[big])), n_big=int(big.sum()), w25_big=float((e[big] <= .25).mean()))
    if lo is not None:
        c = (Y >= lo) & (Y <= hi); out["cov"] = float(c.mean()); out["cov_big"] = float(c[big].mean())
    return out


if __name__ == "__main__":
    out = {}
    for tag, pf in (("ingredient_plate_norm (shippable)", "params.json"), ("ingredient_image_norm (fixed camera only)", "params_image.json")):
        P = json.load(open(Path(__file__).parent / pf))
        Y, pr, ids, ds = [], [], [], []
        for d in test:
            k, a = predict(segs[d["dish"]], P); Y.append(d["kcal"]); pr.append(k); ids.append(d["dish"]); ds.append(dish_level(segs[d["dish"]]["ew"])[0])
        Y, pr, ds = map(np.array, (Y, pr, ds))
        q = P["resid_q"]; lo, hi = pr * math.exp(q[0]), pr * math.exp(q[1])
        out[tag] = report(tag, pr, Y, lo, hi)
        out[tag]["per_dish"] = [dict(dish=i, true=float(y), pred=float(p), lo=float(l), hi=float(h)) for i, y, p, l, h in zip(ids, Y, pr, lo, hi)]
        print(out[tag] | {"per_dish": "..."})
    out["dish_whole_image_top1"] = report("dish-level (whole image, top-1 typical)", ds, Y)
    out["const350"] = report("constant 350", np.full(len(Y), 350.0) + 1e-6 * np.arange(len(Y)), Y)
    json.dump(out, open(Path(__file__).parent / "test_results.json", "w"), indent=1)
    print(out["dish_whole_image_top1"]); print(out["const350"])
    k0 = "ingredient_plate_norm (shippable)"; pd_ = out[k0]["per_dish"]
    for r in sorted(pd_, key=lambda r: -abs(r["pred"] - r["true"]) / r["true"])[:8]: print("WORST", r["dish"], "true", round(r["true"]), "pred", round(r["pred"]))
