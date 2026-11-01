"""Scores eval_web.py output (what the real web app logged) per group."""
import json, re, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import SP  # noqa: E402
from evalsets import FOOD101_MAP, TW_ACCEPT  # noqa: E402

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else SP / "eval/results_web.json", encoding="utf-8"))
V = {v["id"]: v for v in json.load(open(Path(__file__).parents[2] / "web/public/models/vocab.json", encoding="utf-8"))}
g = lambda name: [r for r in R if r["group"] == name]
pct = lambda a, b: f"{a}/{b} ({100 * a / max(b, 1):.0f}%)"
logged = lambda r: bool(r.get("items"))

for grp in ("neg_user", "neg_extra"):
    rs = g(grp); bad = [r for r in rs if logged(r)]
    print(f"{grp:10s} logged any kcal: {pct(len(bad), len(rs))}", [(Path(r['path']).name, [i['nameEn'] for i in r['items']]) for r in bad])
rs = g("drinks"); print(f"drinks     logged: {pct(sum(map(logged, rs)), len(rs))}")

def name_hits(r, ok):  # top-1 of any item / any candidate of any item
    t1 = any(ok(i["cands"][0]["id"]) for i in r.get("items", []))
    tk = any(ok(c["id"]) for i in r.get("items", []) for c in i["cands"])
    return t1, tk

rs = g("tw"); h = [name_hits(r, lambda cid, r=r: re.search(TW_ACCEPT[r["label"]], V[cid]["en"], re.I) is not None) for r in rs]
print(f"tw         logged {pct(sum(map(logged, rs)), len(rs))}; top-1 right {pct(sum(a for a, _ in h), len(rs))}; in candidates {pct(sum(b for _, b in h), len(rs))}")
rs = [r for r in g("food101") if r["label"] in FOOD101_MAP]
h = [name_hits(r, lambda cid, r=r: cid in FOOD101_MAP[r["label"]]) for r in rs]
print(f"food101    (classes with a vocab entry) logged {pct(sum(map(logged, rs)), len(rs))}; top-1 right {pct(sum(a for a, _ in h), len(rs))}; in candidates {pct(sum(b for _, b in h), len(rs))}")

rs = g("n5k"); lg = [r for r in rs if logged(r)]
err = [abs(r["total"]["p50"] - r["kcal"]) / r["kcal"] for r in lg]
cov = [r["total"]["p10"] <= r["kcal"] <= r["total"]["p90"] for r in lg]
ratio = [r["total"]["p50"] / r["kcal"] for r in lg]
print(f"n5k        logged {pct(len(lg), len(rs))}; median |err| {100 * st.median(err):.0f}%; within 25%: {pct(sum(e <= .25 for e in err), len(lg))}; "
      f"true kcal inside logged 10-90% range: {pct(sum(cov), len(lg))}; median logged/true {st.median(ratio):.2f}")
ms = [r["ms"] for r in R if "ms" in r][1:]
print(f"latency (desktop Chromium, wasm, 1st excluded): median {st.median(ms):.0f} ms, p90 {sorted(ms)[int(.9 * len(ms))]} ms; errors {sum('error' in r for r in R)}")
