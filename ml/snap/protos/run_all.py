"""Final comparison on the TEST sets using parameters from fit.json (fit on the validation split). Final prototypes/probe use the whole training pool.
Writes $SP/research/protos/final_results.json and prints the tables."""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import eval_all as ea, protolib as pl, fit, class_map as cm
from common import SP, MODELS


def kmeans_reduce(X, y, kmax, seed=0):
    rng = np.random.default_rng(seed); outX, outy = [], []
    for k in np.unique(y):
        A = X[y == k]
        if len(A) <= kmax: outX.append(A); outy += [k] * len(A); continue
        C = A[rng.choice(len(A), kmax, replace=False)]
        for _ in range(15):
            a = np.argmax(A @ C.T, 1)
            for j in range(kmax):
                if (a == j).any(): C[j] = pl.unit(A[a == j].mean(0))
        outX.append(C); outy += [k] * kmax
    return np.concatenate(outX).astype(np.float32), np.array(outy)


def build_variants():
    T = fit.load_text(); ids, isneg = pl.load_ids(MODELS); rows, rc = pl.concept_rows(ids)
    X, y, paths, src = fit.load_pool(); best = json.load(open(SP / "research/protos/fit.json"))["best"]
    K = len(pl.CONCEPTS)
    mu = pl.unit(np.stack([X[y == k].mean(0) if (y == k).any() else np.zeros(512) for k in range(K)]) + 1e-9)
    V = {"text": pl.TextOnly(T)}
    V["merged_mean"] = pl.Merged(pl.merged_mean(T, rc, mu, best["merged_mean"]["lam"]))
    b = best["tip"]
    V["tip_all"] = pl.Tip(T, rc, X, y, b["a"], b["c"], b["m"])
    Xr, yr = kmeans_reduce(X, y, 16)
    V["tip_k16"] = pl.Tip(T, rc, Xr, yr, b["a"], b["c"], min(b["m"], 3))
    M, _ = fit.probe_train(X, y, T, rc, rows, isneg, best["probe"]["reg"])
    V["probe"] = pl.Merged(M)
    return V, dict(T=T, rc=rc, X=X, y=y, Xr=Xr, yr=yr, mu=mu, probeM=M, best=best, isneg=isneg, ids=ids)


def main():
    ea.setup(); V, aux = build_variants(); out = {}
    for name, sc in V.items():
        t0 = time.time(); R = ea.run_pipeline(sc); s = ea.summarize(R)
        f1 = ea.food101_extra(sc); oi, grp = ea.oi_heldout(sc)
        out[name] = dict(summary=s, f101_classes=f1, oi_concepts=oi, oi_groups=grp, raw=R)
        n = sum(v[0] for v in oi.values())
        print(name, {k: round(v, 3) for k, v in s.items()}, "| OI top1 %.3f top5 %.3f n=%d" % (sum(v[1] for v in oi.values()) / n, sum(v[2] for v in oi.values()) / n, n), f"{time.time() - t0:.0f}s", flush=True)
    ea.save_cache()
    json.dump(out, open(SP / "research/protos/final_results.json", "w"), default=float)
    np.savez(SP / "research/protos/final_aux.npz", **{k: v for k, v in aux.items() if isinstance(v, np.ndarray)})


if __name__ == "__main__":
    main()
