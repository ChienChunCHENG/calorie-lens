import pickle, sys, numpy as np
from fit_params import *
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = {x["dish"]: x for x in dev_labels()}
for k in list(segs)[:12]:
    a, s = features(segs[k]); x = d[k]
    top = np.argsort(-a)[:5]
    print(k, round(x["kcal"]), "truth:", [(i["name"], round(i["g"])) for i in sorted(x["ingr"], key=lambda i: -i["kcal"])[:4]])
    print("   pred:", [(rows[t]["id"], round(a[t], 3)) for t in top if a[t] > 0.002], "plate-area-sum", round(s, 2))
