"""TFDA long-format CSV -> data/tfda_pivot.json (one object per food, per 100 g).

Energy analyte: '熱量' (TFDA's Atwater 4/9/4 on total carbohydrate) is used as kcal.
'修正熱量' (fibre-adjusted modified energy; differs from 熱量 in about half of the foods)
is kept as kcal_modified for reference only. 熱量 matches 4*P+9*F+4*C(total carb) within ~5 kcal on average,
while 修正熱量 averages ~10 kcal lower, so 熱量 is the more conservative (higher) choice.
TFDA gives no n/sd for energy itself, so kcal sd is propagated from the macro sds
(sqrt((4*sd_p)^2+(9*sd_f)^2+(4*sd_c)^2), missing sds treated as 0) and n is the
minimum n across the macros that have one.

Usage: py -I -X utf8 build_tfda_pivot.py
"""
import csv, json, math, pathlib, zipfile

HERE = pathlib.Path(__file__).resolve().parent
ZIP = HERE / "sources" / "tfda_export_20.zip"
OUT = HERE / "tfda_pivot.json"
AN = {"熱量": "kcal", "修正熱量": "kcal_modified", "粗蛋白": "protein",
      "粗脂肪": "fat", "總碳水化合物": "carb"}


def num(s):
    s = (s or "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main():
    foods = {}
    with zipfile.ZipFile(ZIP) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            import io
            rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
            for r in rd:
                fid = r["整合編號"]
                o = foods.setdefault(fid, {
                    "tfda_id": fid, "name": r["樣品名稱"], "alias": r["俗名"],
                    "english": r["樣品英文名稱"], "category": r["食品分類"],
                    "description": r["內容物描述"],
                    "per100g": {}, "n": {}, "sd": {}})
                k = AN.get(r["分析項"])
                if not k:
                    continue
                o["per100g"][k] = num(r["每100克含量"])
                if k in ("protein", "fat", "carb"):
                    o["n"][k] = int(num(r["樣本數"])) if num(r["樣本數"]) is not None else None
                    o["sd"][k] = num(r["標準差"])
    out = []
    for o in foods.values():
        sds = [o["sd"].get(k) for k in ("protein", "fat", "carb")]
        ns = [v for v in o["n"].values() if v]
        if any(s is not None for s in sds):
            w = {"protein": 4, "fat": 9, "carb": 4}
            o["sd"]["kcal"] = round(math.sqrt(sum((w[k] * (o["sd"].get(k) or 0)) ** 2 for k in w)), 2)
        else:
            o["sd"]["kcal"] = None
        o["n"]["kcal"] = min(ns) if ns else None
        out.append(o)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for o in out if all(o["per100g"].get(k) is not None for k in ("kcal", "protein", "fat", "carb")))
    print(f"foods={len(out)} complete_macros={ok} -> {OUT}")


if __name__ == "__main__":
    main()
