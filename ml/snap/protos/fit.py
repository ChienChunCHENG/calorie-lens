"""Fit the prototype combination rule on a VALIDATION split carved out of the training pools (never on the test sets).
Training pool = Food-101 train photos (whole) + Open Images train box crops. Split 70/30 by image-id hash: `tr` builds prototypes/probe, `dev`
scores candidate rules. Metric on dev: top-1 / top-5 over all food rows (accepted = vocab ids of the concept), plus text-only baseline.
usage: python fit.py  -> prints grids, writes $SP/research/protos/fit.json (chosen parameters)"""
import sys, json, pickle, itertools, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SP, MODELS
import protolib as pl

P = SP / "research/protos"
torch = None


def load_text():
    T = pickle.load(open(SP / "snap/text.pkl", "rb"))[(("a photo of {}.",), "custom")].astype(np.float32)
    return T


def load_pool():
    Xs, ys, ps, src = [], [], [], []
    for nm in ("f101", "oi", "neg"):
        f = P / f"emb_{nm}.npz"
        if not f.exists(): continue
        z = np.load(f); Xs.append(z["emb"].astype(np.float32)); ys.append(z["concept"]); ps.append(z["path"]); src += [nm] * len(z["concept"])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps), np.array(src)


def topk_acc(S, y, rows, isneg, ks=(1, 5)):
    """S: (n, R) scores; y concept idx; rows[k] accepted rows"""
    S = np.where(isneg[None], -9, S); order = np.argsort(-S, axis=1)[:, :max(ks)]
    out = []
    for k in ks:
        out.append(np.mean([bool(set(order[i, :k]) & set(rows[y[i]])) for i in range(len(y))]))
    return out


def probe_train(X, y, T, rc, rows, isneg, reg, steps=150, lr=0.02, scale=64.25, seed=0):
    import torch as th
    th.set_num_threads(3); th.manual_seed(seed)
    K = len(pl.CONCEPTS); trained = np.where(rc >= 0)[0]
    Tt = th.tensor(T); Xt = th.tensor(X); yt = th.tensor(y)
    D = th.zeros(K, T.shape[1], requires_grad=True)
    rck = th.tensor(rc[trained]); tr_idx = th.tensor(trained)
    mask_pos = th.zeros(len(pl.CONCEPTS), T.shape[0], dtype=th.bool)
    for k, r in enumerate(rows): mask_pos[k, r] = True
    opt = th.optim.Adam([D], lr=lr)
    for s in range(steps):
        M = Tt.clone(); M[tr_idx] = th.nn.functional.normalize(Tt[tr_idx] + D[rck], dim=-1)
        logits = scale * Xt @ M.T
        lse = th.logsumexp(logits, 1)
        pos = th.logsumexp(logits.masked_fill(~mask_pos[yt], -1e9), 1)
        loss = (lse - pos).mean() + reg * (D ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        M = Tt.clone(); M[tr_idx] = th.nn.functional.normalize(Tt[tr_idx] + D[rck], dim=-1)
    return M.numpy(), float(loss)


def main():
    T = load_text(); ids, isneg = pl.load_ids(MODELS); rows, rc = pl.concept_rows(ids)
    X, y, paths, src = load_pool(); dev = pl.split_dev(paths, 0.3)
    print("pool", X.shape, "dev", dev.sum(), "concepts with train imgs", len(set(y[~dev])), flush=True)
    Xtr, ytr, Xd, yd = X[~dev], y[~dev], X[dev], y[dev]
    # only evaluate dev images of concepts that have >= 1 training image
    has = np.isin(yd, np.unique(ytr)); Xd, yd, srcd = Xd[has], yd[has], src[dev][has]
    res = {}
    def rep(name, S):
        a = topk_acc(S, yd, rows, isneg); gen = [cm_ for cm_ in range(len(pl.CONCEPTS))]
        f1 = np.mean([bool(np.argmax(np.where(isneg, -9, S[i])) in rows[yd[i]]) for i in range(len(yd)) if srcd[i] == "f101"]) if (srcd == "f101").any() else float("nan")
        o1 = np.mean([bool(np.argmax(np.where(isneg, -9, S[i])) in rows[yd[i]]) for i in range(len(yd)) if srcd[i] == "oi"]) if (srcd == "oi").any() else float("nan")
        print(f"{name:40s} top1 {a[0]:.3f} top5 {a[1]:.3f} | f101 {f1:.3f} oi {o1:.3f}", flush=True); res[name] = dict(top1=a[0], top5=a[1], f101=f1, oi=o1); return a[0]
    rep("text", Xd @ T.T)
    mu = pl.unit(np.stack([Xtr[ytr == k].mean(0) if (ytr == k).any() else np.zeros(512) for k in range(len(pl.CONCEPTS))]) + 1e-9)
    best = {}
    # (1) merged class-mean (zero cost)
    for lam in (0.25, 0.5, 1, 2, 4):
        M = pl.merged_mean(T, rc, mu, lam); s = rep(f"merged_mean lam={lam}", Xd @ M.T)
        if s > best.get("merged_mean", (0,))[0]: best["merged_mean"] = (s, dict(lam=lam))
    # (2) kNN/prototype additive: a*(q-c)
    for m in (1, 3, 5):
        for a in (0.1, 0.2, 0.4, 0.8):
            for c in (0.4, 0.5, 0.6, 0.7):
                t = pl.Tip(T, rc, Xtr, ytr, a, c, m)
                S = np.stack([t @ x for x in Xd[:600]]) if False else None
                # vectorised: compute all sims once
                sp = Xd @ t.P.T; g = np.where(t.mask[None], sp[:, t.idx0], -9.0)
                mm = min(m, g.shape[2]); top = -np.partition(-g, mm - 1, axis=2)[:, :, :mm] if mm < g.shape[2] else g
                cnt = np.minimum(t.mask.sum(1), m).clip(1); q = np.where(top > -8, top, 0).sum(2) / cnt
                S = Xd @ T.T; ok = t.has[t.rk]
                S[:, t.trained[ok]] += a * (q[:, t.rk[ok]] - c)
                s = rep(f"tip m={m} a={a} c={c}", S)
                if s > best.get("tip", (0,))[0]: best["tip"] = (s, dict(m=m, a=a, c=c))
    # (3) text-initialised probe
    for reg in (1e-4, 1e-3, 1e-2):
        t0 = time.time(); M, l = probe_train(Xtr, ytr, T, rc, rows, isneg, reg)
        s = rep(f"probe reg={reg} (loss {l:.2f}, {time.time() - t0:.0f}s)", Xd @ M.T)
        if s > best.get("probe", (0,))[0]: best["probe"] = (s, dict(reg=reg))
    json.dump(dict(best={k: v[1] for k, v in best.items()}, dev=res), open(P / "fit.json", "w"), indent=1)
    print("BEST", best)


if __name__ == "__main__":
    main()
