"""Fetch Food-101 TRAIN images via the HF datasets-server rows API (openly licensed research data). Never touches the validation images.
Row order in the HF train split = blocks of 750 rows, one class per block (block order permuted). Output: $SP/research/protos/f101/<class>/<n>.jpg
usage: python fetch_food101.py [per_class=64]"""
import json, os, sys, time, random, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SP
OUT = SP / "research/protos/f101"; OUT.mkdir(parents=True, exist_ok=True)
API = "https://datasets-server.huggingface.co/rows?dataset=ethz/food101&config=default&split=train&offset={}&length={}"


def get(url, tries=8):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "calorie-lens-research"}), timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500: time.sleep(5 * (t + 1)); continue
            raise
        except Exception: time.sleep(3 * (t + 1))
    raise RuntimeError(url)


def rows(off, n): return json.loads(get(API.format(off, n)))["rows"]


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rng = random.Random(0)
    bm = OUT / "blocks.json"
    if bm.exists(): blocks = json.load(open(bm))
    else:
        blocks = {}
        for b in range(101):
            blocks[b] = rows(b * 750 + 375, 1)[0]["row"]["label"]; print("block", b, blocks[b], flush=True); time.sleep(0.2)
        json.dump(blocks, open(bm, "w"))
    names = json.loads(get(API.format(0, 1)))["features"][1]["type"]["names"]
    for b, lab in blocks.items():
        b = int(b); cls = names[lab]; d = OUT / cls; d.mkdir(exist_ok=True)
        have = len(list(d.glob("*.jpg")))
        if have >= per: continue
        start = b * 750 + rng.randrange(0, 750 - 100)
        rs = rows(start, 100); rng.shuffle(rs)
        for i, r in enumerate(rs[: per]):
            p = d / f"{start + i}.jpg"
            if p.exists(): continue
            try: p.write_bytes(get(r["row"]["image"]["src"]))
            except Exception as e: print("fail", cls, e); continue
            time.sleep(0.05)
        print(cls, len(list(d.glob("*.jpg"))), flush=True)


if __name__ == "__main__": main()
