"""Stage 3: fit the portion model on the dev FIT split (70%), choose on the VALIDATION split (30%); the 60-dish test set is NOT touched here
(see final_eval.py). Split = md5(dish id) % 10 < 7 -> fit.

usage: python fit.py            -> prints variant table, writes $SP/research/portion/fit_result.json
"""
import sys, json, pickle, hashlib, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
from common import SP, MODELS
import portion as P

RP = SP / "research/portion"
V = json.load(open(MODELS / "vocab.json", encoding="utf-8"))
GRP = [P.group_of(v) for v in V]; GI = np.array([P.GROUPS.index(g) for g in GRP]); K100 = np.array([v["kcal100"] for v in V], float)
RHO0 = 1.2  # kcal/g, median dish density on the fit split
TYP = np.array([v["portion_g"][1] for v in V], float)


def split_of(d): return "fit" if int(hashlib.md5(d.encode()).hexdigest(), 16) % 10 < 7 else "val"


def load(name, labels, area_mode="colour"):
    FE = pickle.load(open(RP / f"feat_{name}.pkl", "rb")); AN = pickle.load(open(RP / f"an_{name}.pkl", "rb"))
    out = []
    for d in labels:
        if d["dish"] not in FE: continue
        F, A = FE[d["dish"]], AN[d["dish"]]
        iell = A["item_ell"]; first = next((e for e in iell if e), None)
        out.append(dict(dish=d["dish"], kcal=d["kcal"], mass=d["mass_g"], F=F, A=A, ell=first, iell=iell, px=A["area_px"]))
    return out


def flatten(D, Dd):
    """rows = (dish, item, candidate) for a vectorised fit. area_cm2 = px * (Dd[type]/major)^2 / tilt. Items without a container get weight 0 and
    contribute their typical-portion kcal as a fixed term per dish (`fixed`)."""
    r = dict(dish=[], a=[], w=[], g=[], k100=[]); fixed = np.zeros(len(D))
    for di, d in enumerate(D):
        for ii, it in enumerate(d["F"]["items"]):
            e = d["iell"][ii]
            for c, w in it["cand"]:
                r["dish"].append(di); r["g"].append(GI[c]); r["k100"].append(K100[c])
                if e is None: r["w"].append(0.0); r["a"].append(1.0); fixed[di] += w * TYP[c] * K100[c] / 100.0
                else:
                    tilt = min(1.0, max(0.5, e["minor"] / e["major"]))
                    r["w"].append(w); r["a"].append(max(d["px"][ii] * (Dd.get(e["type"], Dd["plate"]) / e["major"]) ** 2 / tilt, 0.5))
    return {k: np.array(v) for k, v in r.items()}, fixed


def fit_variant(Dfit, variant, Dd, steps=1500, seed=0, use_gamma=False):
    rows, fixed = flatten(Dfit, Dd); fixed = torch.tensor(fixed)
    ok = np.array([any(e is not None for e in d["iell"]) for d in Dfit])
    a = torch.tensor(rows["a"]); w = torch.tensor(rows["w"]); g = torch.tensor(rows["g"]); k100 = torch.tensor(rows["k100"]); di = torch.tensor(rows["dish"])
    y = torch.tensor([math.log(d["kcal"]) for d in Dfit]); okt = torch.tensor(ok)
    lk = torch.zeros(8, requires_grad=True) if variant in ("kgroup", "full") else torch.zeros(1, requires_grad=True)
    lb = torch.zeros(8, requires_grad=True) if variant == "full" else torch.zeros(1, requires_grad=True)
    train_beta = variant in ("gbeta", "full")
    lg = torch.ones(1, requires_grad=True)  # gamma: density shrinkage, kcal/g = (kcal100/100)^gamma * RHO0^(1-gamma)
    opt = torch.optim.Adam([lk] + ([lb] if train_beta else []) + ([lg] if use_gamma else []), lr=0.03)

    def pred_log(lk, lb):
        kk = lk.exp().expand(8) if lk.numel() == 1 else lk.exp()
        bb = (1 + lb).expand(8) if lb.numel() == 1 else 1 + lb  # beta = 1 + lb
        if not train_beta: bb = torch.ones(8)
        gr = kk[g] * a ** bb[g]
        dens = (k100 / 100.0).clamp_min(0.02) ** lg * RHO0 ** (1 - lg) if use_gamma else k100 / 100.0
        s = torch.zeros(len(Dfit), dtype=torch.float64).index_add(0, di, w * gr * dens) + fixed
        return torch.log(s.clamp_min(1.0))

    for it in range(steps):
        opt.zero_grad()
        e = (pred_log(lk, lb) - y)[okt].abs()
        loss = e.mean() + 0.01 * ((lb ** 2).sum() if train_beta else 0) + (0.02 * ((lk - lk.mean()) ** 2).sum() if lk.numel() > 1 else 0)
        loss.backward(); opt.step()
        if train_beta:
            with torch.no_grad(): lb.clamp_(-0.8, 0.5)
    kk = lk.detach().exp().expand(8).numpy(); bb = (1 + lb.detach()).expand(8).numpy() if train_beta else np.ones(8)
    return dict(k=kk.tolist(), beta=bb.tolist(), gamma=float(lg.detach()) if use_gamma else 1.0)


def evaluate(D, theta, Dd, fallback):
    """kcal per dish with the portion model; dishes with no logged item use `fallback(d)`; items without a container use the typical portion."""
    out = np.zeros(len(D)); used = np.zeros(len(D), bool)
    for i, d in enumerate(D):
        its = d["F"]["items"]
        if not its: out[i] = fallback(d); continue
        s = 0.0
        for ii, it in enumerate(its):
            e = d["iell"][ii]
            if e is None: s += sum(w * TYP[c] * K100[c] / 100.0 for c, w in it["cand"]); continue
            tilt = min(1.0, max(0.5, e["minor"] / e["major"])); a = max(d["px"][ii] * (Dd.get(e["type"], Dd["plate"]) / e["major"]) ** 2 / tilt, 0.5)
            for c, w in it["cand"]:
                gi = GI[c]; dens = max(K100[c] / 100.0, 0.02) ** theta.get("gamma", 1.0) * RHO0 ** (1 - theta.get("gamma", 1.0)); s += w * theta["k"][gi] * a ** theta["beta"][gi] * dens
        out[i] = max(s, 1.0); used[i] = True
    return out, used


def typical(d): return sum(sum(w * TYP[c] * K100[c] / 100.0 for c, w in it["cand"]) for it in d["F"]["items"])


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = np.abs(pred - true) / true
    return dict(med_abs_pct=float(np.median(err) * 100), within25=float((err <= 0.25).mean() * 100), mean_abs_pct=float(err.mean() * 100),
                pearson=float(np.corrcoef(pred, true)[0, 1]), pearson_log=float(np.corrcoef(np.log(np.maximum(pred, 1)), np.log(true))[0, 1]),
                bias_log=float(np.median(np.log(np.maximum(pred, 1) / true))))


def fmt(m): return f"med|err| {m['med_abs_pct']:5.1f}%  +-25% {m['within25']:5.1f}%  r {m['pearson']:.2f}  rlog {m['pearson_log']:.2f}  biaslog {m['bias_log']:+.2f}"


if __name__ == "__main__":
    labels = json.load(open(SP / "eval/n5k_dev/labels.json"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "colour"
    DA = load("dev", labels, mode)
    Dfit = [d for d in DA if split_of(d["dish"]) == "fit"]; Dval = [d for d in DA if split_of(d["dish"]) == "val"]
    print(len(DA), "dishes; fit", len(Dfit), "val", len(Dval), "area mode", mode)
    for nm, DD in (("fit", Dfit), ("val", Dval)):
        print(nm, "items w/o container:", sum(any(e is None for e in d["iell"]) for d in DD), "no items:", sum(len(d["F"]["items"]) == 0 for d in DD))
    med_fit = float(np.median([d["kcal"] for d in Dfit])); print("fit-split median kcal", med_fit)
    for nm, DD in (("fit", Dfit), ("val", Dval)):
        t = [d["kcal"] for d in DD]
        print(f"{nm}: typical     ", fmt(metrics([typical(d) or 350 for d in DD], t)))
        print(f"{nm}: const 350   ", fmt(metrics([350] * len(DD), t)))
        print(f"{nm}: const fitmed", fmt(metrics([med_fit] * len(DD), t)))
    res = {}
    Dd = dict(plate=26.0, bowl=16.0, tray=26.0); fb = lambda d: typical(d) or med_fit
    for var in ("glob", "kgroup", "gbeta", "full"):
        for ug in (False, True):
            th = fit_variant(Dfit, var, Dd, use_gamma=ug)
            pf, _ = evaluate(Dfit, th, Dd, fb); pv, uv = evaluate(Dval, th, Dd, fb)
            mf, mv = metrics(pf, [d["kcal"] for d in Dfit]), metrics(pv, [d["kcal"] for d in Dval])
            print(f"{var:7s} gamma={ug!s:5s} (g={th['gamma']:.2f}) fit {fmt(mf)} | val {fmt(mv)}")
            res[f"{var}_{ug}"] = dict(theta=th, fit=mf, val=mv)
    json.dump(res, open(RP / f"fit_result_{mode}.json", "w"), indent=1)
