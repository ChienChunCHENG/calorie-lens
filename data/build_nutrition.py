"""Merge data/dishes.yaml with data/priors/*.yaml into data/nutrition.json.

Usage:  py -X utf8 data/build_nutrition.py

Robust to missing/partial priors: absent fields become null and a warning is
printed.  Hard errors (exit 1): missing dish ids, kind mismatch, or any
min <= typ <= max violation.
"""
import datetime
import json
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
PRIORS = HERE / "priors"
EXPECTED = {  # spec/data_contract.md
    "rice_white": "solid", "lu_rou_fan": "solid", "chicken_leg": "solid", "pork_chop": "solid",
    "braised_egg": "solid", "fried_egg": "solid", "stirfried_greens": "solid", "braised_tofu": "solid",
    "beef_noodle_soup": "solid", "boiled_dumplings": "solid", "potstickers": "solid",
    "scallion_pancake": "solid", "fried_chicken_cutlet": "solid", "popcorn_chicken": "solid",
    "bubble_milk_tea": "drink",
}
warnings, errors = [], []


def warn(msg):
    warnings.append(msg)
    print("WARNING:", msg)


def err(msg):
    errors.append(msg)
    print("ERROR:", msg)


def load(name):
    p = PRIORS / name
    if not p.exists():
        warn(f"priors/{name} missing")
        return {}
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        warn(f"priors/{name} unreadable: {e}")
        return {}
    if not isinstance(d, dict):
        warn(f"priors/{name} is not a mapping")
        return {}
    return d


def lookup(data, key, *containers):
    """Find data[key], also looking inside wrapper sections like data['dishes']."""
    if key in data and isinstance(data[key], dict):
        return data[key]
    for c in containers:
        sub = data.get(c)
        if isinstance(sub, dict) and isinstance(sub.get(key), dict):
            return sub[key]
    return None


def check_triple(label, t, keys=("min", "typ", "max")):
    if t is None:
        return
    try:
        vals = [t[k] for k in keys]
    except (KeyError, TypeError):
        err(f"{label}: malformed range {t}")
        return
    if any(v is None for v in vals):
        warn(f"{label}: has null in {t}")
        return
    if not all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
        err(f"{label}: ordering violated {t}")


def as_triple(t):
    return dict(t) if isinstance(t, dict) else None


def main():
    dishes_doc = yaml.safe_load((HERE / "dishes.yaml").read_text(encoding="utf-8"))
    dishes = dishes_doc["dishes"]
    density = load("density.yaml")
    oil = load("oil.yaml")
    portions = load("portions.yaml")
    containers = load("containers.yaml")
    drinks = load("drinks.yaml")

    methods = oil.get("methods") or {}
    dish_method = oil.get("dish_method") or {}

    for i, kind in EXPECTED.items():
        if i not in dishes:
            err(f"dish {i} missing from dishes.yaml")
        elif dishes[i].get("kind") != kind:
            err(f"dish {i}: kind {dishes[i].get('kind')!r} != contract {kind!r}")
    for i in dishes:
        if i not in EXPECTED:
            warn(f"dish {i} in dishes.yaml is not a contract id")

    out_dishes = []
    for i in EXPECTED:
        d = dishes.get(i)
        if d is None:
            continue
        kind = d["kind"]
        e = {"id": i, "zh": d["zh"], "en": d["en"], "kind": kind}
        sources = list(d.get("source") or [])

        # density (solids only)
        dens = None
        if kind == "solid":
            ds = lookup(density, i, "dishes", "densities", "solids")
            if ds is None:
                warn(f"{i}: no density prior")
            else:
                dens = as_triple(ds.get("density_g_ml"))
                if dens is None:
                    warn(f"{i}: density prior has no density_g_ml")
                if "bulk" in ds:
                    e["bulk"] = ds["bulk"]
                sources += [f"density: {s}" for s in (ds.get("source") or [])]
        e["density_g_ml"] = dens

        e["kcal_100g"] = as_triple(d["kcal_100g"])
        e["macros_100g"] = d.get("macros_100g")

        # oil adder
        oil_rng = None
        if kind == "solid":
            m = dish_method.get(i)
            applies = True
            if isinstance(m, dict):  # {method, adder_applies, reason}
                applies = m.get("adder_applies", True)
                m = m.get("method")
            if m is None:
                warn(f"{i}: no oil dish_method")
            elif not applies:
                e["oil_method"] = m  # dish kcal already includes its cooking fat
            elif m not in methods:
                warn(f"{i}: oil method {m!r} not defined in oil.yaml methods")
            else:
                spec = methods[m].get("oil_g_per_100g") if isinstance(methods[m], dict) else None
                if spec is None:
                    warn(f"{i}: oil method {m!r} has no oil_g_per_100g")
                else:
                    oil_rng = {"min": spec.get("min"), "max": spec.get("max")}
                    e["oil_method"] = m
        e["oil_g_per_100g"] = oil_rng

        # portions
        pd = lookup(portions, i, "dishes", "portions")
        if pd is None:
            warn(f"{i}: no portions prior")
            e["default_portion_g"], e["container_prior"] = None, None
        else:
            e["default_portion_g"] = as_triple(pd.get("default_portion_g"))
            e["container_prior"] = pd.get("container_prior")
            if e["default_portion_g"] is None:
                warn(f"{i}: portions prior has no default_portion_g")
            sources += [f"portion: {s}" for s in (pd.get("source") or [])]

        e["tfda_refs"] = d.get("tfda_refs") or []
        e["origin"] = d.get("origin")
        e["source"] = sources
        e["confidence"] = d.get("confidence")
        e["method_notes"] = d.get("method_notes")
        if kind == "drink":
            e["kcal_note"] = "indicative only; use top-level drinks for drink kcal"

        check_triple(f"{i}.kcal_100g", e["kcal_100g"])
        check_triple(f"{i}.density_g_ml", e["density_g_ml"])
        check_triple(f"{i}.default_portion_g", e["default_portion_g"])
        check_triple(f"{i}.oil_g_per_100g", e["oil_g_per_100g"], ("min", "max"))
        if e["kcal_100g"] and e["kcal_100g"]["min"] <= 0:
            err(f"{i}: kcal min must be > 0")
        out_dishes.append(e)

    # top-level containers / drinks (pass through)
    cont = containers.get("containers", containers) if containers else None
    if cont:
        for cid, c in cont.items():
            if isinstance(c, dict):
                check_triple(f"container {cid}.capacity_ml", c.get("capacity_ml"))
    else:
        cont = None
    drk = drinks or None
    if drk:
        for k, v in drk.items():
            if isinstance(v, dict):
                for kk in ("base_kcal_per_700ml", "toppings_kcal"):
                    check_triple(f"drinks.{k}.{kk}", v.get(kk))
        if "sugar_levels" not in drk:
            warn("drinks.yaml has no sugar_levels")
        if "full_sugar_g_per_700ml" not in drk:
            warn("drinks.yaml has no full_sugar_g_per_700ml")
        if "bubble_milk_tea" not in drk and not (isinstance(drk.get("drinks"), dict) and "bubble_milk_tea" in drk["drinks"]):
            warn("drinks.yaml has no entry for bubble_milk_tea")

    result = {
        "version": 1,
        "generated": datetime.date.today().isoformat(),
        "schema_note": dishes_doc.get("schema_note"),
        "dishes": out_dishes,
        "containers": cont,
        "drinks": drk,
    }
    (HERE / "nutrition.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote nutrition.json: {len(out_dishes)} dishes, {len(warnings)} warnings, {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
