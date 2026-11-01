"""Build the web-ready prototype files. Pipeline (each step caches under $SP/research/protos/):
  python fetch_food101.py 32      # Food-101 TRAIN photos via HF datasets-server (never the validation images)
  python collect_oi.py 70         # Open Images V6 train box crops (class_map oi lists)
  python collect_oi_ho.py 25      # Open Images val/test held-out crops (TEST only, uxset ids excluded)
  python collect_neg.py           # non-food object crops for the neg_* rows (OI val/test, excl. held-out + uxset ids)
  python embed_train.py f101 ; python embed_train.py oi     # shipped clip_image.onnx, runtime preprocessing
  python fit.py ; python grid2.py # parameters chosen on the validation split
  python final_run.py             # test-set evaluation of text vs prototype variants -> $SP/research/protos/final_results.json
  python build_protos.py [K=16]   # this file: reduce to <=K prototypes per concept (k-means) and write out/protos_emb.bin + out/protos.json
Output (candidate web files, NOT copied to web/public/models):
  out/protos_emb.bin   float16 LE, n x 512, each row L2-normalised, rows sorted by concept
  out/protos.json      {dim, a, c, m, relu, concepts:[{name, vocab_ids, neg}], offsets:[start row of each concept, +n]}  vocab ids, not row numbers,
                       so it stays valid when vocab.json gains rows."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import protolib as pl, fit, class_map as cm
from run_all import kmeans_reduce

PARAMS = dict(a=1.0, c=0.65, m=3, relu=True)   # chosen in grid2.py (see RESULTS.md)


def build(K=16):
    X, y, paths, src = fit.load_pool()
    Xr, yr = kmeans_reduce(X, y, K) if K else (X, y)
    o = np.argsort(yr, kind="stable"); Xr, yr = Xr[o], yr[o]
    out = HERE / "out"; out.mkdir(exist_ok=True)
    Xr.astype("<f2").tofile(out / "protos_emb.bin")
    counts = np.bincount(yr, minlength=len(pl.CONCEPTS)); offs = np.concatenate([[0], np.cumsum(counts)]).tolist()
    concepts = [dict(name=k, vocab_ids=cm.C[k]["vocab"], neg=cm.C[k]["grp"] == "neg") for k in pl.CONCEPTS]
    json.dump(dict(dim=512, n=int(len(yr)), K=K, **PARAMS, concepts=concepts, offsets=offs), open(out / "protos.json", "w"))
    print("rows", len(yr), "MB", len(yr) * 1024 / 1e6)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
