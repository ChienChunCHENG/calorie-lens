import json, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import SP
RI = SP / "research/ingr"
def h(d): return int(hashlib.md5(d.encode()).hexdigest(), 16)
def split(d): return "val" if (h(d) >> 1) % 3 == 0 else "fit"
def dev_labels():
    ids = set(json.load(open(RI / "dev_ids.json")))
    return [x for x in json.load(open(SP / "eval/n5k_dev/labels.json")) if x["dish"] in ids]
