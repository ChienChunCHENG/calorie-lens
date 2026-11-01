"""Build out/vocab_prior.json + params.json.  python build_prior.py  (needs research members.json: see build_members.py; NHANES files auto-download)
vocab_prior.json: array aligned row-for-row with web/public/models/vocab.json:
  {id, prior (sum over food rows = 1, neg rows 0), lp (centred log prior = ln prior - mean over food rows of ln prior; neg rows 0 = neutral), display? (top-800 rows)}"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from prior_lib import *
import names

PARAMS = dict(**{"lambda": 0.5}, temper=1.0, alpha=20.0, agg="rep", gate="raw (gate uses the prior-free softmax; prior only reorders food rows)", n_display=800)


def main():
    V = load_vocab(); R = load_raw(); M = members(); cnt = nhanes_counts()
    neg = np.array([bool(v.get("neg")) for v in V])
    c, kind, euro = row_counts(PARAMS["agg"])
    p = prior_from_counts(c, neg, PARAMS["alpha"], PARAMS["temper"]); lp = logprior_term(p, neg)
    order = [i for i in np.argsort(-c, kind="stable") if not neg[i]][:PARAMS["n_display"]]
    disp = {}
    for i in order:
        m = re.search(r"FNDDS 2021-2023 #(\d+)", R[i]["source"]); top = ""
        if m: top = sorted(M[m.group(1)], key=lambda t: -cnt.get(t[0], 0))[0][1]
        d = names.display_for(V[i]["id"], V[i]["en"], top, kind[i])
        if d: disp[i] = d
    out = []
    for i, v in enumerate(V):
        o = dict(id=v["id"], prior=float(f"{p[i]:.6e}"), lp=round(float(lp[i]), 4))
        if i in disp: o["display"] = disp[i]
        out.append(o)
    json.dump(out, open(HERE / "out/vocab_prior.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    json.dump(PARAMS, open(HERE / "params.json", "w"), indent=1)
    print("rows", len(out), "display", len(disp), "prior sum", p.sum(), "kinds", {k: kind.count(k) for k in set(kind)})
    json.dump(dict(counts=c.tolist(), kind=kind, euro=euro), open(WORK / "counts.json", "w"))


if __name__ == "__main__":
    main()
