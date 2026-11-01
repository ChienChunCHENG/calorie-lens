"""Final test-set evaluation of the chosen rule (params from grid2.py) with prototypes from the WHOLE training pool."""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import eval_all as ea, protolib as pl, fit
from run_all import kmeans_reduce
from build_protos import PARAMS
from common import SP

ea.setup(); T = fit.load_text(); ids, isneg = pl.load_ids(); rows, rc = pl.concept_rows(ids); X, y, _, _ = fit.load_pool()
V = {"text": pl.TextOnly(T)}
for K in (16, 32, 0):
    Xr, yr = kmeans_reduce(X, y, K) if K else (X, y)
    V[f"proto_K{K or 'all'}"] = pl.Tip(T, rc, Xr, yr, PARAMS["a"], PARAMS["c"], PARAMS["m"], relu=True)
out = {}
for n, sc in V.items():
    R = ea.run_pipeline(sc); s = ea.summarize(R); f1 = ea.food101_extra(sc); oi, grp = ea.oi_heldout(sc)
    out[n] = dict(summary=s, f101_classes=f1, oi_concepts=oi, oi_groups=grp, raw=R)
    N = sum(v[0] for v in oi.values()); print(n, {k: round(v, 3) for k, v in s.items()}, "OI %.3f %.3f" % (sum(v[1] for v in oi.values()) / N, sum(v[2] for v in oi.values()) / N), flush=True)
ea.save_cache(); json.dump(out, open(SP / "research/protos/final_results.json", "w"), default=float)
