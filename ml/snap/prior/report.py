"""Final numbers + before/after examples -> $SP/research/prior/report.md (pasted into RESULTS.md)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_prior import *
P = json.load(open(HERE / "params.json")); VP = json.load(open(HERE / "out/vocab_prior.json", encoding="utf-8"))
term = np.array([x["lp"] for x in VP]); lam = P["lambda"]; DISP = {i: x["display"] for i, x in enumerate(VP) if "display" in x}
D = Data(); b = evaluate(D, np.zeros(N), 0.0); a = evaluate(D, term, lam, "raw")
L = []
def pc(x): return f"{100*x:.1f}"
L.append("| set (n) | top-1 before | top-1 after | top-3 before | top-3 after |\n|---|---|---|---|---|")
for k, nm in (("oi_val_all", "OI-val all mapped crops"), ("oi_val_strict", "OI-val specific classes"), ("oi_test_all", "OI-test all"), ("oi_test_strict", "OI-test specific"), ("food101", "Food-101 mappable")):
    L.append(f"| {nm} ({b[k][2]}) | {pc(b[k][0])} | {pc(a[k][0])} | {pc(b[k][1])} | {pc(a[k][1])} |")
L.append("\n| images logging anything | before | after (gate raw) |\n|---|---|---|")
for k, nm in (("gate_neg_user", "user negatives (10)"), ("gate_neg_ux", "uxset negatives (80)"), ("gate_neg_extra", "extra negatives (30)"), ("gate_drinks", "drinks (39)")):
    L.append(f"| {nm} | {b[k][0]}/{b[k][1]} | {a[k][0]}/{a[k][1]} |")
pg = evaluate(D, term, lam, "prior", True)
L.append("\nIf instead the gate used the prior-adjusted softmax with neutral (0) neg terms: " + ", ".join(f"{k[5:]} {pg[k][0]}/{pg[k][1]}" for k in ("gate_neg_user", "gate_neg_ux", "gate_neg_extra", "gate_drinks")))
# ux examples
(E, meta), imgs = D.regions["ux_img"]
def top(k, t, l):
    s = scores(E[k:k + 1], t, l)[0]; s = np.where(NEG, -1e9, s); i = np.argsort(-s)[:3]
    p = np.exp(s - s.max()); p /= p[~NEG].sum() if False else p.sum()
    return [(DISP.get(j, V[j]["en"]) if t is term else V[j]["en"]) for j in i], [round(float(p[j]), 2) for j in i]
L.append("\n| image | region | before (en, top-3) | after (display, top-3) |\n|---|---|---|---|")
for k, x in enumerate(meta):
    n = os.path.basename(x["path"])[:-4]
    if n in ("steak", "dark_steak", "burger_fries", "blurry_burger", "sushi", "fish_chips", "pizza_slice", "pasta", "schnitzel", "salad", "sandwich", "us_breakfast", "tacos", "western_breakfast", "croissant_coffee", "paella", "cake_slice", "oatmeal", "falafel_hummus", "snacks"):
        tb, _ = top(k, np.zeros(N), 0.0); ta, _ = top(k, term, lam)
        L.append(f"| {n} | {x['kind']} {[int(v) for v in x['box']]} | {'; '.join(tb)} | {'; '.join(ta)} |")
open(SP / "research/prior/report.md", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
