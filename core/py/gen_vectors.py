"""Generate spec/vectors.json from the reference implementation."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import range_engine as R


def T(a, b, c):
    return {"min": a, "typ": b, "max": c}


FRAC = {"full": 1.0, "less": 0.7, "half": 0.5, "light": 0.3, "none": 0.0}

RICE = {"id": "rice", "kind": "solid", "density_g_ml": T(0.6, 0.7, 0.8),
        "kcal_100g": T(125, 140, 170), "oil_g_per_100g": {"min": 0, "max": 0},
        "default_portion_g": T(150, 200, 280)}
FRIED = {"id": "fried_cutlet", "kind": "solid", "density_g_ml": T(0.5, 0.8, 1.2),
         "kcal_100g": T(220, 280, 380), "oil_g_per_100g": {"min": 4, "max": 12},
         "default_portion_g": T(150, 220, 320)}
NOOIL = {"id": "noodle", "kind": "solid", "density_g_ml": T(0.8, 0.9, 1.0),
         "kcal_100g": T(90, 100, 115), "default_portion_g": T(300, 450, 600)}
FIXED = {"id": "fixed", "kind": "solid", "density_g_ml": T(0.8, 0.8, 0.8),
         "kcal_100g": T(200, 200, 200), "oil_g_per_100g": {"min": 0, "max": 0},
         "default_portion_g": T(100, 100, 100)}
SKEW = {"id": "skew", "kind": "solid", "density_g_ml": T(0.9, 1.3, 1.4),
        "kcal_100g": T(100, 100, 100)}
WIDE = {"id": "wide", "kind": "solid", "density_g_ml": T(0.3, 0.8, 1.6),
        "kcal_100g": T(100, 250, 500), "default_portion_g": T(50, 150, 450)}
TEA = {"id": "tea", "kind": "drink", "base_kcal_per_700ml": T(120, 180, 260),
       "toppings_kcal": T(100, 150, 220)}
PLAIN = {"id": "plain_tea", "kind": "drink", "base_kcal_per_700ml": T(0, 0, 0),
         "toppings_kcal": T(0, 0, 0)}

cases = []


def add(name, fn, inp, expect_error=False):
    c = {"name": name, "fn": fn, "input": inp}
    try:
        if fn == "estimateSolid":
            r = R.estimate_solid(inp["dish"], inp["volumeMl"])
        elif fn == "estimateDrink":
            r = R.estimate_drink(inp["drink"], inp["sizeMl"], inp["sugarLevel"],
                                 inp["fullSugarGPer700"], inp["sugarFraction"])
        else:
            r = R.combine_meal(inp["items"])
        if expect_error:
            raise SystemExit("case %s should have failed" % name)
        c["expected"] = {"lo": r["lo"], "med": r["med"], "hi": r["hi"],
                         "flags": r.get("flags", [])}
    except ValueError:
        if not expect_error:
            raise
        c["expected"] = {"error": True}
    cases.append(c)


def solid(name, dish, vol, **kw):
    add(name, "estimateSolid", {"dish": dish, "volumeMl": vol}, **kw)


def drink(name, d, size, level, **kw):
    add(name, "estimateDrink", {"drink": d, "sizeMl": size, "sugarLevel": level,
                                "fullSugarGPer700": 60, "sugarFraction": FRAC}, **kw)


solid("rice_typical", RICE, T(200, 280, 380))
solid("fried_wide_with_oil", FRIED, T(150, 300, 600))
solid("rice_null_volume_default_portion", RICE, None)
solid("fried_null_volume_default_portion_oil", FRIED, None)
solid("floor_zero_width_all_factors", FIXED, T(250, 250, 250))
solid("zero_width_null_volume", FIXED, None)
solid("no_oil_field_noodle", NOOIL, T(400, 500, 650))
solid("typ_not_geometric_mean_skewed", SKEW, T(100, 100, 100))
solid("low_confidence_wide", WIDE, T(100, 400, 900))
solid("low_confidence_null_volume", WIDE, None)
solid("narrow_inputs_floor_inactive", RICE, T(279, 280, 281))
solid("narrow_noodle_floor_inactive", NOOIL, T(499, 500, 501))
for lvl in ["full", "less", "half", "light", "none"]:
    drink("drink_%s_700" % lvl, TEA, 700, lvl)
drink("drink_full_500", TEA, 500, "full")
drink("drink_none_500", TEA, 500, "none")
drink("drink_zero_everything_none", PLAIN, 700, "none")
drink("drink_plain_tea_full_sugar", PLAIN, 700, "full")

meal = [R.estimate_solid(RICE, T(200, 280, 380)), R.estimate_solid(FRIED, T(150, 300, 600)),
        R.estimate_solid(NOOIL, None), R.estimate_drink(TEA, 700, "half", 60, FRAC)]
add("meal_four_items", "combineMeal", {"items": meal})
add("meal_single_item", "combineMeal", {"items": [meal[0]]})
add("meal_empty", "combineMeal", {"items": []})

solid("err_min_zero", RICE, T(0, 280, 380), expect_error=True)
solid("err_negative_min", RICE, T(-1, 280, 380), expect_error=True)
solid("err_min_gt_typ", RICE, T(300, 280, 380), expect_error=True)
bad = dict(RICE)
bad["kcal_100g"] = T(0, 140, 170)
solid("err_kcal_min_zero", bad, T(200, 280, 380), expect_error=True)
drink("err_unknown_sugar_level", TEA, 700, "extra", expect_error=True)
drink("err_nonpositive_size", TEA, 0, "full", expect_error=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec", "vectors.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"version": 1, "cases": cases}, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote %d cases" % len(cases))
