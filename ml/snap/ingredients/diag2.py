import pickle, numpy as np
from fit_params import *
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = [x for x in dev_labels() if x["dish"] in segs]
X, T, Y, ids, tot = build(segs, d, mix=0.5, temp=1.0)
mass = np.array([x["mass_g"] for x in d]); print(len(d))
dens = Y / mass; print("kcal/g median", np.median(dens), "iqr", np.percentile(dens, [25, 75]))
e = lambda p: np.median(np.abs(p - Y) / Y)
print("oracle mass x median density:", e(mass * np.median(dens)))
print("const median:", e(np.full(len(Y), np.median(Y))))
la = np.log(np.maximum(X.sum(1), 1e-4)); lm = np.log(mass)
print("corr log painted-area vs log mass", np.corrcoef(la, lm)[0, 1], " tot-area (before label)", np.corrcoef(np.log(np.maximum(tot, 1e-4)), lm)[0, 1])
# oracle-label + area: give true per-comp kcal share? use true composition kcal density: kcal_density_true = Y/mass ; pred = mass_from_area * true density
sl = np.polyfit(np.log(np.maximum(tot, 1e-4)), lm, 1); mp = np.exp(np.polyval(sl, np.log(np.maximum(tot, 1e-4))))
print("mass-from-area err median", np.median(np.abs(mp - mass) / mass), "slope", sl)
print("area-mass + TRUE density:", e(mp * dens), " area-mass + median dens:", e(mp * np.median(dens)))
print("true mass + predicted composition density?")
# predicted composition density: sum area*kcal_g / sum area
cd = (X * kcal_g).sum(1) / np.maximum(X.sum(1), 1e-6)
print("corr predicted density vs true density", np.corrcoef(cd, dens)[0, 1], " log", np.corrcoef(np.log(np.maximum(cd,.05)), np.log(dens))[0, 1])
print("true mass x pred density err", e(mass * np.clip(cd, .1, 5)), " true mass x median dens", e(mass*np.median(dens)))
