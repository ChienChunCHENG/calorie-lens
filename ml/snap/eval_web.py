"""Runs the REAL web app (built site + vite preview, headless Chromium) over the evaluation photos and records what it
logged. Same code and models the phone loads; only the browser differs (desktop Chromium, wasm backend).

    cd web && npm run build -- --base=/calorie-lens/ && npx vite preview --base=/calorie-lens/ --port 4173
    python ml/snap/eval_web.py --out results_web.json [--limit-food101 60]

Sets come from evalsets.py plus Nutrition5k (scratchpad/eval/n5k/labels.json: overhead photos with measured kcal).
"""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import SP  # noqa: E402
from evalsets import sets  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4173/calorie-lens/")
    ap.add_argument("--out", default=str(SP / "eval/results_web.json"))
    ap.add_argument("--limit-food101", type=int, default=60)
    a = ap.parse_args()

    S = sets()
    random.seed(0)
    by_class = {}
    for x in S["food101"]:
        by_class.setdefault(x["label"], []).append(x)
    f101 = [v[0] for v in by_class.values()][: a.limit_food101]
    n5k = json.load(open(SP / "eval/n5k/labels.json"))
    jobs = ([("neg_user", x) for x in S["neg_user"]] + [("neg_extra", x) for x in S["neg_extra"]]
            + [("drinks", x) for x in S["drinks"]] + [("tw", x) for x in S["tw"]] + [("food101", x) for x in f101]
            + [("n5k", dict(path=str(SP / "eval/n5k" / m["file"]), kcal=m["kcal"], mass_g=m["mass_g"], ingr=m["ingr"])) for m in n5k])

    out = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(viewport={"width": 390, "height": 844}).new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(a.url)
        pg.wait_for_function("document.querySelectorAll('input[type=file][accept=\"image/*\"]').length >= 2")
        pick = pg.locator('input[type=file][accept="image/*"]:not([capture])').first
        n = 0
        for grp, x in jobs:
            t0 = time.time()
            pick.set_input_files(x["path"])
            n += 1
            pg.wait_for_function(f"window.__clLast && window.__clLast.n >= {n}", timeout=300_000)
            r = pg.evaluate("window.__clLast")
            rec = {"group": grp, **{k: v for k, v in x.items() if k != "boxes"}, "wall_s": round(time.time() - t0, 2)}
            if "error" in r:
                rec["error"] = r["error"]
            else:
                res = r["result"]
                rec.update(ms=r["ms"], backend=r["backend"], nBoxes=res["nBoxes"], whole=res["whole"], skipped=res["skipped"],
                           total=res["total"], items=[{k: it[k] for k in ("name", "nameEn", "unsure", "box", "p10", "p50", "p90", "cands")}
                                                     for it in res["items"]])
            out.append(rec)
            print(f"{grp:9s} {Path(x['path']).name[:34]:34s} items {len(rec.get('items', []))} "
                  f"total {rec.get('total')} {rec.get('ms', '')}ms {rec.get('error', '')}", flush=True)
        b.close()
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("page errors:", errs[:5])


if __name__ == "__main__":
    main()
