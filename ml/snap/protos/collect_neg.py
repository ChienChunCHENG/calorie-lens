"""Negative-object prototypes: Open Images validation/test boxes for the non-food classes in class_map (grp 'neg'); excludes uxset ids and any
image used in oi_ho/boxes.json. Writes oi_neg/boxes.json + images, then emb_neg.npz (same embedding as embed_train.oi)."""
import csv, json, random, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from common import SP
import class_map as cm
from collect_oi import ux_ids
import embed_train as et
OUT = SP / "research/protos/oi_neg"; (OUT / "img").mkdir(parents=True, exist_ok=True)
N = 30
names = {r[1]: r[0] for r in csv.reader(open(SP / "uxset/raw/oidv7-class-descriptions-boxable.csv", encoding="utf8"))}
want = {names[o]: (o, k) for k, d in cm.C.items() if d["grp"] == "neg" for o in d["oi"]}
ex = ux_ids() | {b["id"] for b in json.load(open(SP / "research/protos/oi_ho/boxes.json"))}
pool = {m: [] for m in want}
for split in ("validation", "test"):
    for r in csv.DictReader(open(SP / f"uxset/raw/{split}-annotations-bbox.csv", encoding="utf8")):
        m = r["LabelName"]
        if m not in pool or r["IsGroupOf"] == "1" or r["ImageID"] in ex: continue
        x1, x2, y1, y2 = float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])
        if (x2 - x1) * (y2 - y1) < 0.04: continue
        pool[m].append((split, r["ImageID"], x1, y1, x2, y2))
rng = random.Random(3); out = []
for m, rows in pool.items():
    rng.shuffle(rows); seen = set(); k = 0
    for sp, iid, *b in rows:
        if iid in seen: continue
        seen.add(iid); out.append(dict(id=iid, split=sp, oi=want[m][0], concept=want[m][1], box=b)); k += 1
        if k >= N: break
def dl(o):
    p = OUT / "img" / f"{o['id']}.jpg"
    if p.exists(): return 1
    for t in range(4):
        try: p.write_bytes(urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/{o['split']}/{o['id']}.jpg", timeout=30).read()); return 1
        except Exception: time.sleep(1 + t)
    return 0
with cf.ThreadPoolExecutor(8) as e: print("downloaded", sum(e.map(dl, out)), len(out))
out = [o for o in out if (OUT / "img" / f"{o['id']}.jpg").exists()]
sess = et.session(); cache = {}
def mk(i):
    b = out[i]
    if b["id"] not in cache: cache.clear(); cache[b["id"]] = et.load_img(OUT / "img" / f"{b['id']}.jpg")
    im = cache[b["id"]]; x1, y1, x2, y2 = b["box"]
    return et.crop_box(im, [x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], 0.2)
E, ok = et.embed_batches(sess, mk, len(out), tag="neg")
c2i = {k: i for i, k in enumerate(et.CONCEPTS)}
np.savez(SP / "research/protos/emb_neg.npz", emb=E[ok].astype(np.float16), concept=np.array([c2i[out[i]["concept"]] for i in range(len(out)) if ok[i]]),
         cls=np.array([out[i]["oi"] for i in range(len(out)) if ok[i]]), path=np.array(["neg_" + out[i]["id"] for i in range(len(out)) if ok[i]]))
print("saved", ok.sum())
