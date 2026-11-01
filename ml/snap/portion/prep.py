"""Stage 2: container ellipse + food-pixel areas for every image (3 area modes) -> $SP/research/portion/an_<name>.pkl
usage: python prep.py <name> <image dir>   (name: dev | test | wiki)"""
import sys, pickle, glob, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from common import SP
import portion as P
from snaplib import load_img

RP = SP / "research/portion"

if __name__ == "__main__":
    name, src = sys.argv[1], sys.argv[2]
    FE = pickle.load(open(RP / f"feat_{name}.pkl", "rb")); out = {}
    for k, (stem, F) in enumerate(FE.items()):
        p = next(iter(glob.glob(os.path.join(src, stem + ".*"))))
        img = np.asarray(load_img(p))
        a = P.analyze(img, F); out[stem] = dict(area_px=a["area_px"], food_px=a["food_px"], item_ell=a["item_ell"])
    pickle.dump(out, open(RP / f"an_{name}.pkl", "wb")); print("done", len(out))
