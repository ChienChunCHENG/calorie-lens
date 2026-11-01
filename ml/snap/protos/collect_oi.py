"""Collect Open Images V6 TRAIN box crops for the concepts in class_map.py (OI class -> vocab concept).
Reads train-bbox.csv (https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv), keeps clean boxes (not group, not depiction,
not truncated, area >= 4% of image, long side >= 0.2), samples N per OI class (distinct images), downloads the JPEGs from the public S3 bucket.
Writes $SP/research/protos/oi/train_boxes.json (image id, class, box) and images to oi/img/. Training split only: the UX test set uses
validation/test (checked: no overlap by construction; ids in $SP/uxset are additionally excluded)."""
import csv, json, random, sys, os, time, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SP
import class_map as cm
OI = SP / "research/protos/oi"; (OI / "img").mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 70


def ux_ids():
    s = set()
    for d in (SP / "uxset").glob("*"):
        if d.is_dir():
            s |= {p.stem for p in d.rglob("*.jpg")}
    m = SP / "uxset/manifest.json"
    if m.exists():
        def walk(o):
            if isinstance(o, str): s.add(Path(o).stem)
            elif isinstance(o, dict):
                for k, v in o.items(): s.add(str(k)); walk(v)
            elif isinstance(o, list): [walk(v) for v in o]
        walk(json.load(open(m)))
    return s


def main():
    names = {r[1]: r[0] for r in csv.reader(open(SP / "uxset/raw/oidv7-class-descriptions-boxable.csv", encoding="utf8"))}
    want = {}  # MID -> (OI name, concept)
    for k, d in cm.C.items():
        for o in d["oi"]: want[names[o]] = (o, k)
    pool = {m: [] for m in want}
    bx = OI / "train_boxes.json"
    if not bx.exists():
        for r in csv.DictReader(open(OI / "train-bbox.csv", encoding="utf8")):
            m = r["LabelName"]
            if m not in pool or r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" or r["IsTruncated"] == "1": continue
            x1, x2, y1, y2 = float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])
            if (x2 - x1) * (y2 - y1) < 0.04 or max(x2 - x1, y2 - y1) < 0.2: continue
            pool[m].append((r["ImageID"], x1, y1, x2, y2))
        rng = random.Random(1); ux = ux_ids(); out = []
        for m, rows in pool.items():
            rng.shuffle(rows); seen = set(); k = 0
            for iid, *b in rows:
                if iid in seen or iid in ux: continue
                seen.add(iid); out.append(dict(id=iid, oi=want[m][0], concept=want[m][1], box=b)); k += 1
                if k >= N: break
            print(want[m][0], len(rows), k, flush=True)
        json.dump(out, open(bx, "w"))
    out = json.load(open(bx))
    ids = sorted({o["id"] for o in out})

    def dl(i):
        p = OI / "img" / f"{i}.jpg"
        if p.exists(): return 1
        for t in range(4):
            try: p.write_bytes(urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/train/{i}.jpg", timeout=30).read()); return 1
            except Exception: time.sleep(1 + t)
        return 0
    with cf.ThreadPoolExecutor(8) as ex: ok = sum(ex.map(dl, ids))
    print("downloaded", ok, "/", len(ids))


if __name__ == "__main__": main()
