import pickle, numpy as np
from fit_params import *
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = [x for x in dev_labels() if x["dish"] in segs]
mass = np.array([x["mass_g"] for x in d]); lm = np.log(mass)
rows_ = []
for x in d:
    s = segs[x["dish"]]; its = s["items"]; h, w = its[0]["m"].shape if its else (120, 160)
    plate, found = ingr.plate_mask(s, (h, w)); pa = plate.sum() / (h * w)
    p = ingr.label_items(s, E_eff, nK, 0.5, 1.0); nb = p.shape[1] - nK
    A = ingr.areas(s, None, nK) * pa  # abs fraction of image
    foodp = p[:, :nK].sum(1)  # prob of food vs BG
    rows_.append((pa, A.sum(), (A * foodp).sum(), found, len(its)))
r = np.array(rows_, float)
print("plate found", r[:, 3].mean(), "plate frac of image: mean", r[:, 0].mean(), "std", r[:, 0].std())
for nm, v in [("plate area", r[:, 0]), ("painted abs", r[:, 1]), ("food-weighted abs", r[:, 2]), ("food-weighted / plate", r[:, 2] / r[:, 0])]:
    print(nm, "corr log-mass", np.corrcoef(np.log(np.maximum(v, 1e-4)), lm)[0, 1])
# truth: how about image-level 'non-white' pixel area? 
