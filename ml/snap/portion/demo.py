"""Transfer sanity check: run the portion method on arbitrary photos; prints container + grams per item and writes an overlay sheet.
usage: python demo.py <name: wiki|neg> <image dir> <out.jpg>"""
import sys, json, pickle, glob, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import portion as P
from common import SP, MODELS
from snaplib import load_img
name, src, outp = sys.argv[1:4]
RP = SP / "research/portion"; V = json.load(open(MODELS / "vocab.json", encoding="utf-8")); GRP = [P.group_of(v) for v in V]
Pm = P.Params.load(Path(__file__).parent / "portion_params.json")
FE = pickle.load(open(RP / f"feat_{name}.pkl", "rb")); tiles = []
for stem, F in FE.items():
    p = glob.glob(os.path.join(src, stem + ".*"))[0]; img = np.asarray(load_img(p)).copy(); a = P.analyze(img, F)
    line = f"{stem}: items={len(F['items'])}"; ells = a["item_ell"]; drawn = set()
    for e in ells:
        if e is None or id(e) in drawn: continue
        drawn.add(id(e)); fg, _ = P.food_pixels(img, e); img[fg] = (0.5 * img[fg] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.ellipse(img, ((e["cx"], e["cy"]), (e["major"], e["minor"]), e["angle"]), (0, 255, 0), 3)
        line += f"\n  container={e['type']} major={e['major']:.0f}px ({100*e['major']/max(img.shape[:2]):.0f}% of long edge) minor/major={e['minor']/e['major']:.2f}"
    ars = [None if e is None else P.area_cm2(px, e, Pm.diam(e["type"])) for px, e in zip(a["area_px"], ells)]
    for it, ar in zip(F["items"], ars):
        c0 = it["cand"][0][0]
        if ar is None: line += f"\n    {V[c0]['en']} (p={it['cand'][0][1]:.2f}): no container -> typical {V[c0]['portion_g'][1]} g"
        else:
            g = Pm.grams(GRP[c0], ar); line += f"\n    {V[c0]['en']} (p={it['cand'][0][1]:.2f}, grp {GRP[c0]}): area {ar:.0f} cm2 -> {g:.0f} g -> {g*Pm.density(V[c0]['kcal100']):.0f} kcal (typical portion {V[c0]['portion_g'][1]} g)"
        cv2.rectangle(img, tuple(int(v) for v in it["box"][:2]), tuple(int(v) for v in it["box"][2:]), (255, 255, 0), 3)
    print(line)
    h = 300; tiles.append(cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)))
W = 4; rows = []
for i in range(0, len(tiles), W):
    r = tiles[i:i + W]; r += [np.zeros_like(tiles[0])] * (W - len(r)); r = [cv2.resize(t, (400, 300)) for t in r]; rows.append(np.hstack(r))
cv2.imwrite(outp, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
