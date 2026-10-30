"""Validate data/priors/*.yaml: parse, ids match the data contract, and min <= typ <= max everywhere.
Run: py data/priors/check_priors.py"""
import sys
from pathlib import Path
import yaml

D = Path(__file__).parent
IDS = ["rice_white", "lu_rou_fan", "chicken_leg", "pork_chop", "braised_egg", "fried_egg", "stirfried_greens",
       "braised_tofu", "beef_noodle_soup", "boiled_dumplings", "potstickers", "scallion_pancake",
       "fried_chicken_cutlet", "popcorn_chicken", "bubble_milk_tea"]
METHODS = ["stir_fry", "deep_fry_breaded", "pan_fry", "braise", "boil", "steam", "none"]
CONTAINERS = ["bowl_small", "bowl_medium", "bowl_large", "bento_box", "plate_26cm", "cup_500", "cup_700"]
errs = []


def load(n):
    return yaml.safe_load((D / n).read_text(encoding="utf-8"))


def walk(node, path):
    if isinstance(node, dict):
        if {"min", "max"} <= node.keys():
            lo, hi = node["min"], node["max"]
            typ = node.get("typ", lo)
            if not all(isinstance(v, (int, float)) for v in (lo, typ, hi)):
                errs.append(f"{path}: non-numeric")
            elif not lo <= typ <= hi:
                errs.append(f"{path}: min<=typ<=max violated {lo},{typ},{hi}")
            return
        for k, v in node.items():
            walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]")


den, oil, por, con, dr = (load(f) for f in
                          ["density.yaml", "oil.yaml", "portions.yaml", "containers.yaml", "drinks.yaml"])
for name, doc in [("density", den), ("oil", oil), ("portions", por), ("containers", con), ("drinks", dr)]:
    walk(doc, name)

solids = [i for i in IDS if i != "bubble_milk_tea"]
if sorted(den["solids"]) != sorted(solids):
    errs.append("density ids mismatch")
for i, e in den["solids"].items():
    if not isinstance(e["bulk"], bool) or e["confidence"] not in ("high", "med", "low") or not e["source"]:
        errs.append(f"density.{i}: bulk/confidence/source")
if sorted(oil["methods"]) != sorted(METHODS):
    errs.append("oil methods mismatch")
if sorted(oil["dish_method"]) != sorted(IDS):
    errs.append("dish_method ids mismatch")
for i, e in oil["dish_method"].items():
    if e["method"] not in METHODS or not isinstance(e["adder_applies"], bool):
        errs.append(f"dish_method.{i}")
if sorted(por["portions"]) != sorted(IDS):
    errs.append("portions ids mismatch")
for i, e in por["portions"].items():
    for c in e["container_prior"]:
        if c not in CONTAINERS:
            errs.append(f"portions.{i}: unknown container {c}")
if sorted(con["containers"]) != sorted(CONTAINERS):
    errs.append("containers ids mismatch")
if not set(dr["sugar_levels"]) == {"full", "less", "half", "light", "none"}:
    errs.append("sugar_levels keys")
v = [dr["sugar_levels"][k] for k in ("full", "less", "half", "light", "none")]
if v != sorted(v, reverse=True) or v[0] != 1.0:
    errs.append("sugar_levels not monotone / full != 1")
if "bubble_milk_tea" not in dr["drinks"]:
    errs.append("drinks missing bubble_milk_tea")

print("\n".join(errs) if errs else "OK: all priors parse and satisfy min<=typ<=max")
sys.exit(1 if errs else 0)
