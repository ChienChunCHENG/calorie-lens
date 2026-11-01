import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evalmv import *
for tag in ("", "_f085", "_f115", "_d22", "_d30"):
    if not (MV / f"b_dev{tag}.pkl").exists(): continue
    Ds = load_dishes("dev", tag); m = np.array([d.mass for d in Ds]); out = []
    for meth in ("ah2", "depth"):
        v = np.array([volume(d.rec, meth, "ABC") or np.nan for d in Ds]); ok = ~np.isnan(v) & (v > 0)
        out.append(f"{meth}: r_log={np.corrcoef(np.log(v[ok]), np.log(m[ok]))[0,1]:.2f}")
    Dfit = [d for d in Ds if split_of(d.id) == "fit"]; Dv = [d for d in Ds if split_of(d.id) == "val"]
    th = fit_C(Dfit, [volume(d.rec, "ah2", "ABC") for d in Dfit], grid_b=(0.6, 0.8, 1.0))
    p = [(C_median_kcal(d, volume(d.rec, "ah2", "ABC"), th), d.kcal) for d in Dv]; p = [x for x in p if x[0]]
    mm = metrics([a for a, _ in p], [b for _, b in p])
    print(tag or "base", " ".join(out), f"val median err {mm['med']:.1f}% (n={mm['n']})")
