"""Round 2: thresholded boost  score_i = text_i + a*max(q_k - c, 0)  (q_k = mean top-m cosine to concept-k prototypes), with negative-object prototypes
for the neg_* rows. Prototypes from the TRAIN part of the pool only; dev = held-out 30 % of the pool (concept images); guards = neg_extra tune half
(15 images) + drinks + UEC recall (not the user's negatives or the held-out half; those are reported afterwards, not used to choose)."""
import sys, json, itertools
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import eval_all as ea, protolib as pl, fit, class_map as cm
from common import SP, MODELS

ea.setup(); T = fit.load_text(); ids, isneg = pl.load_ids(); rows, rc = pl.concept_rows(ids)
X, y, paths, src = fit.load_pool(); dev = pl.split_dev(paths, 0.3)
tr = ~dev; res = []
isneg_c = np.array([cm.C[k]["grp"] == "neg" for k in pl.CONCEPTS])
dm = dev & ~isneg_c[y]; Xd, yd = X[dm], y[dm]
base = ea.run_pipeline(pl.TextOnly(T)); b = ea.summarize(base); print("base", b, flush=True)
for m, a, c in itertools.product((3,), (1.0, 2.0, 4.0, 8.0), (0.5, 0.6, 0.65, 0.7, 0.75, 0.8)):
    t = pl.Tip(T, rc, X[tr], y[tr], a, c, m, relu=True)
    S = np.stack([t @ x for x in Xd]); acc = fit.topk_acc(S, yd, rows, isneg)
    R = ea.run_pipeline(t); s = ea.summarize(R)
    row = dict(m=m, a=a, c=c, dev_top1=acc[0], dev_top5=acc[1], **{k: s[k] for k in ("negU", "negX_tune", "negX_held", "drinks_logged", "uec_recall", "uec_prec", "f101_whole_top1", "f101_item_top1", "tw_item_top1")})
    res.append(row); print({k: round(v, 3) for k, v in row.items()}, flush=True)
ea.save_cache(); json.dump(dict(base=b, grid=res), open(SP / "research/protos/grid2.json", "w"), default=float)
