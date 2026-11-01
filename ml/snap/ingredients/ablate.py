"""Dev-val ablations: how much do labels buy?  (a) dish-level whole-image top-1 typical kcal, (b) constant, (c) label-free area-only, (d) component model (shippable cfg)."""
import pickle, numpy as np, torch, json
from common_i import *
from fit_params import *
from final_eval import dish_level
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); d = dev_labels()
fitL = [x for x in d if split(x["dish"]) == "fit"]; valL = [x for x in d if split(x["dish"]) == "val"]
out = {}
Xf, Tf, Yf, idf, _ = build(segs, fitL, mix=.5, temp=1., norm="plate"); Xv, Tv, Yv, idv, _ = build(segs, valL, mix=.5, temp=1., norm="plate")
e = lambda p: float(np.median(np.abs(p - Yv) / Yv))
out["const_median_fit"] = e(np.full(len(Yv), np.median(Yf))); out["const350"] = e(np.full(len(Yv), 350.))
out["dish_level_whole_image"] = e(np.array([dish_level(segs[i]["ew"])[0] for i in idv]))
# area-only: kcal = K * (sum area)^b + c, labels ignored (all rows 1.14 kcal/g, one K)
af, av = Xf.sum(1, keepdims=True), Xv.sum(1, keepdims=True)
lf = torch.tensor(np.log(np.maximum(af[:, 0], 1e-3)), dtype=torch.float32); y = torch.tensor(Yf, dtype=torch.float32)
w = torch.tensor([5., 0.7, 3.], requires_grad=True); o = torch.optim.Adam([w], lr=.05)
for _ in range(1500):
    o.zero_grad(); p = torch.exp(w[0] + w[1] * lf) + torch.relu(w[2]); r = torch.log(torch.clamp(p, min=5) / y); torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=.3).backward(); o.step()
pv = np.exp(w[0].item() + w[1].item() * np.log(np.maximum(av[:, 0], 1e-3))) + max(w[2].item(), 0); out["area_only_no_labels"] = e(pv)
m = fit(Xf, Yf, fix_b=1.0)
with torch.no_grad(): out["component_model"] = e(m(Xv).numpy())
# oracle labels on the same areas: replace predicted composition density by true density (upper bound of label quality)
out["note"] = "median |err|/true on dev-val (n=%d)" % len(Yv)
print(json.dumps(out, indent=1)); json.dump(out, open(Path(__file__).parent / "ablations.json", "w"), indent=1)
