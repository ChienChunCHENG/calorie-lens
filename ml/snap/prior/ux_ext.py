import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_prior import *
term = np.array([v.get("lp", 0.0) for v in V]); (E, meta), imgs = load("ux_img"), None
E, meta = load("ux_img")
for n in ("fish_chips","schnitzel","steak","burger_fries","salad","sandwich","western_breakfast","croissant_coffee","pasta","cake_slice"):
    for k, x in enumerate(meta):
        if os.path.basename(x["path"])[:-4] != n or x["kind"]!="box": continue
        s = np.where(NEG, -1e9, scores(E[k:k+1], term, 0.5)[0]); i = np.argsort(-s)[:3]
        s0 = np.where(NEG, -1e9, scores(E[k:k+1], term, 0.0)[0]); i0 = np.argsort(-s0)[:1]
        print(n, [int(v) for v in x["box"]], "| no-prior top1:", V[i0[0]]["en"], "| prior top3:", "; ".join(V[j].get("display", V[j]["en"]) for j in i))
