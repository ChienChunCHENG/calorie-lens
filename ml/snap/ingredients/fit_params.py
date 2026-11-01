"""Fit the few portion/oil parameters on the FIT split and report validation. usage: python fit_params.py [--final]
Model: kcal = oil + sum_k K[group_k] * a_k^b * kcal_g_k   where a_k = soft-assigned area of component k as a fraction of the plate."""
import sys, json, pickle, math, itertools
import numpy as np, torch
from common_i import *
import ingr, ingr_vocab as V

torch.set_num_threads(2)
rows = ingr.load_vocab(); nK = len(rows)
E = ingr.text_embs(rows, V.BG); n2k = V.n5k_map(); kid = {r["id"]: i for i, r in enumerate(rows)}
GROUPS = sorted({r["group"] for r in rows}); gidx = np.array([GROUPS.index(r["group"]) for r in rows])
kcal_g = np.array([r["kcal_g"] for r in rows])
NONVIS = np.array([r["group"] == "oil" for r in rows])


def truth(d):
    t = np.zeros(nK)
    for i in d["ingr"]:
        k = n2k.get(i["name"])
        if k: t[kid[k]] += i["kcal"]
    return t


def features(seg, mix=0.5, temp=1.0, hard=False, norm="plate"):
    """returns comp area vector [nK] (fraction of plate)"""
    p = ingr.label_items(seg, E_eff, nK, mix, temp)
    A = ingr.areas(seg, None, nK, norm=norm)
    if len(A) == 0: return np.zeros(nK), 0.0
    pf = p[:, :nK].copy()
    if hard:
        h = np.zeros_like(pf); top = p.argmax(1)
        for i, t in enumerate(top):
            if t < nK: h[i, t] = 1
        pf = h
    return (A[:, None] * pf).sum(0), float(A.sum())


E_eff = E.copy()
E_eff[:nK][NONVIS] = -E_eff[:nK][NONVIS]  # oil is invisible: push its text row away (never wins)


def build(segs, labels, **kw):
    X, T, Y, ids, tot = [], [], [], [], []
    for d in labels:
        if d["dish"] not in segs: continue
        a, s = features(segs[d["dish"]], **kw); X.append(a); T.append(truth(d)); Y.append(d["kcal"]); ids.append(d["dish"]); tot.append(s)
    return np.array(X), np.array(T), np.array(Y), ids, np.array(tot)


class Model(torch.nn.Module):
    def __init__(s, grouped=True):
        super().__init__(); s.grouped = grouped
        s.lk = torch.nn.Parameter(torch.full((len(GROUPS),), math.log(200.0))); s.b = torch.nn.Parameter(torch.tensor(1.0))
        s.oil = torch.nn.Parameter(torch.tensor(20.0)); s.g0 = torch.nn.Parameter(torch.tensor(math.log(200.0)))

    def forward(s, X, thr=0.004):
        X = torch.as_tensor(X, dtype=torch.float32)
        k = torch.exp(s.lk[gidx]) if s.grouped else torch.exp(s.g0).expand(nK)
        a = torch.clamp(X, min=1e-6); g = k * a ** s.b * (X >= thr)
        return (g * torch.as_tensor(kcal_g, dtype=torch.float32)).sum(1) + torch.relu(s.oil)


def fit(X, Y, grouped=True, iters=800, lam=0.05, fix_b=None, fix_oil=None, thr=0.004):
    m = Model(grouped); Yt = torch.as_tensor(Y, dtype=torch.float32)
    if fix_b is not None: m.b.data.fill_(fix_b); m.b.requires_grad_(False)
    if fix_oil is not None: m.oil.data.fill_(fix_oil); m.oil.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.03)
    for it in range(iters):
        opt.zero_grad(); pr = m(X, thr)
        r = torch.log(torch.clamp(pr, min=5) / Yt)
        loss = torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=0.3) + (lam * ((m.lk - m.lk.mean()) ** 2).mean() if grouped else 0)
        loss.backward(); opt.step()
    return m


def score(pred, Y, name=""):
    e = np.abs(pred - Y) / Y
    big = Y >= 300
    return dict(name=name, n=len(Y), med=float(np.median(e)), w25=float((e <= .25).mean()), r=float(np.corrcoef(pred, Y)[0, 1]),
                med_big=float(np.median(e[big])) if big.any() else None, n_big=int(big.sum()))


def recall(X, T, thr=0.004):
    f = (X >= thr); rec = (T * f).sum(1) / np.maximum(T.sum(1), 1e-9)
    tr = T > 0; pr = f.sum(1)
    prec = (f & tr).sum(1) / np.maximum(pr, 1)
    return float((T * f).sum() / T.sum()), float(rec.mean()), float(prec.mean())


if __name__ == "__main__":
    segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = dev_labels()
    fitL = [x for x in d if split(x["dish"]) == "fit"]; valL = [x for x in d if split(x["dish"]) == "val"]
    print("seg dishes", len(segs))
    for kw in [dict(mix=0.5, temp=1.0), dict(mix=1.0, temp=1.0), dict(mix=0.0, temp=1.0), dict(mix=0.5, temp=1.0, hard=True)]:
        Xf, Tf, Yf, _, _ = build(segs, fitL, **kw); Xv, Tv, Yv, _, _ = build(segs, valL, **kw)
        if len(Yf) < 20: print("too few"); break
        mm = fit(Xf, Yf)
        with torch.no_grad(): pv = mm(Xv).numpy(); pf = mm(Xf).numpy()
        print(kw, "n fit/val", len(Yf), len(Yv), "kcal-recall(all,mean,prec) val", recall(Xv, Tv))
        print("  fit", score(pf, Yf), "\n  val", score(pv, Yv), "b", float(mm.b), "oil", float(torch.relu(mm.oil)))
    c = np.median(Yf); print("const baseline val med |err|", float(np.median(np.abs(c - Yv) / Yv)))
