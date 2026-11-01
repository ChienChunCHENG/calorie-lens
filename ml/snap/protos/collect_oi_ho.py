"""Held-out Open Images TEST crops (validation + test splits; never used for training prototypes). Excludes every image id found in $SP/uxset
(images + manifest.json) so the UX test set stays clean. Writes $SP/research/protos/oi_ho/boxes.json and downloads images."""
import csv, json, random, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SP
import class_map as cm
from collect_oi import ux_ids
OUT = SP / "research/protos/oi_ho"; (OUT / "img").mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def main():
    names = {r[1]: r[0] for r in csv.reader(open(SP / "uxset/raw/oidv7-class-descriptions-boxable.csv", encoding="utf8"))}
    want = {names[o]: (o, k) for k, d in cm.C.items() for o in d["oi"]}
    ux = ux_ids(); print("excluding", len(ux), "ux ids")
    pool = {m: [] for m in want}
    for split in ("validation", "test"):
        for r in csv.DictReader(open(SP / f"uxset/raw/{split}-annotations-bbox.csv", encoding="utf8")):
            m = r["LabelName"]
            if m not in pool or r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" or r["IsTruncated"] == "1" or r["ImageID"] in ux: continue
            x1, x2, y1, y2 = float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])
            if (x2 - x1) * (y2 - y1) < 0.04 or max(x2 - x1, y2 - y1) < 0.2: continue
            pool[m].append((split, r["ImageID"], x1, y1, x2, y2))
    rng = random.Random(2); out = []
    for m, rows in pool.items():
        rng.shuffle(rows); seen = set(); k = 0
        for sp, iid, *b in rows:
            if iid in seen: continue
            seen.add(iid); out.append(dict(id=iid, split=sp, oi=want[m][0], concept=want[m][1], box=b)); k += 1
            if k >= N: break
        print(want[m][0], len(rows), k)
    json.dump(out, open(OUT / "boxes.json", "w"))

    def dl(o):
        p = OUT / "img" / f"{o['id']}.jpg"
        if p.exists(): return 1
        for t in range(4):
            try: p.write_bytes(urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/{o['split']}/{o['id']}.jpg", timeout=30).read()); return 1
            except Exception: time.sleep(1 + t)
        return 0
    with cf.ThreadPoolExecutor(8) as ex: print("downloaded", sum(ex.map(dl, out)), "/", len(out))


if __name__ == "__main__": main()
