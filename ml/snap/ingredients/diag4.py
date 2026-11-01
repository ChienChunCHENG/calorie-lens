import pickle, numpy as np, collections
from fit_params import *
from PIL import Image
segs = pickle.load(open(RI / "seg_dev.pkl", "rb")); ks = list(segs)
c = collections.Counter(); fr = collections.defaultdict(list)
for k in ks:
    s = segs[k]
    if not s["cont"]: continue
    b = max(s["cont"], key=lambda c: c["score"]); c[b["cls"]] += 1; fr[b["cls"]].append((b["mask"] > 0).mean())
print(c, {k: (np.mean(v), np.std(v)) for k, v in fr.items()})
W = Image.new("RGB", (4 * 320, 3 * 240))
import glob
for i, k in enumerate(ks[40:52]):
    im = np.asarray(Image.open(SP / "eval/n5k_dev" / (k + ".png")).convert("RGB").resize((160, 120))).copy()
    s = segs[k]; b = max(s["cont"], key=lambda c: c["score"]); m = b["mask"] > 0
    im[~m] = (im[~m] * 0.4).astype(np.uint8)
    W.paste(Image.fromarray(im).resize((320, 240)), ((i % 4) * 320, (i // 4) * 240))
W.save(RI / "hull.jpg")
