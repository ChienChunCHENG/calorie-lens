"""Candidate web files: out/vocab.json (old rows + 36 new rows, with display/prior/lp) and text-embedding variants.
  python extend.py build   -> out/vocab.json, $SP/research/prior/emb_{A,B,C}.bin (A = en only (current recipe), B = display only, C = mean(en, display))
Text recipe exactly as ml/snap/build_vocab.py: "a photo of {}." template(s) from clip.json, MobileCLIP-S1 datacompdr (reparameterized), L2-normalised, fp16.
Rows without display keep their en embedding in B and C. Negative rows are copied unchanged (their hand-written prompts)."""
import sys, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from prior_lib import *
import newrows

OUT = HERE / "out"; TMP = WORK


def build_vocab_json():
    V = load_vocab(); old_prior = json.load(open(OUT / "vocab_prior.json", encoding="utf-8")); P = json.load(open(HERE / "params.json"))
    assert [v["id"] for v in V] == [x["id"] for x in old_prior]
    neg_old = np.array([bool(v.get("neg")) for v in V]); c_old, _, _ = row_counts(P["agg"])
    c_new = np.array([TIERS[r[7]] for r in newrows.ROWS], dtype=float)
    V2 = copy.deepcopy(V)
    for i, x in enumerate(old_prior):
        if "display" in x: V2[i]["display"] = x["display"]
    for r in newrows.ROWS:
        V2.append(dict(id=r[0], en=r[1], display=r[2], kcal100=r[3], portion_g=r[4], cat=r[5], est=(not r[6].startswith("FNDDS")) or None))
        if V2[-1]["est"] is None: del V2[-1]["est"]
    ids = [v["id"] for v in V2]; assert len(set(ids)) == len(ids), "duplicate ids"
    neg = np.concatenate([neg_old, np.zeros(len(c_new), bool)]); c = np.concatenate([c_old, c_new])
    p = prior_from_counts(c, neg, P["alpha"], P["temper"]); lp = logprior_term(p, neg)
    for i, v in enumerate(V2): v["prior"] = float(f"{p[i]:.6e}"); v["lp"] = round(float(lp[i]), 4)
    json.dump(V2, open(OUT / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    json.dump({r[0]: r[6] for r in newrows.ROWS}, open(OUT / "new_rows_sources.json", "w", encoding="utf-8"), indent=1)
    return V2


def build_emb(V2):
    import torch; torch.set_num_threads(3)
    from common import load_clip
    from snaplib import encode_texts
    cfg = json.load(open(MODELS / "clip.json")); tpls = cfg.get("templates") or [cfg["template"]]
    model, _, tok = load_clip(); n_old = len(load_vocab())
    M0 = np.frombuffer((MODELS / "vocab_emb.bin").read_bytes(), "<f2").astype(np.float32).reshape(n_old, -1)

    def ens(strings):  # mean over templates of normalised embeddings, then normalise (same as snaplib.text_matrix)
        flat = [t.format(s) for s in strings for t in tpls]; e = encode_texts(model, tok, flat).reshape(len(strings), len(tpls), -1).mean(1)
        return e / np.linalg.norm(e, axis=1, keepdims=True)
    # sanity: re-encoding a few old en strings must reproduce the shipped rows (fp16 rounding aside)
    chk = [i for i in (0, 500, 1500, 2200)]; e = ens([V2[i]["en"] for i in chk]); print("recipe check cos", [round(float(e[k] @ M0[i]), 5) for k, i in enumerate(chk)])
    new_idx = list(range(n_old, len(V2))); A = np.concatenate([M0, ens([V2[i]["en"] for i in new_idx])])
    di = [i for i, v in enumerate(V2) if "display" in v and not v.get("neg")]
    D = ens([V2[i]["display"] for i in di])
    B = A.copy(); C = A.copy(); B[di] = D; m = A[di] + D; C[di] = m / np.linalg.norm(m, axis=1, keepdims=True)
    print("rows with display", len(di), "mean cos(en,display)", float(np.mean(np.sum(A[di] * D, 1))))
    for nm, X in (("A", A), ("B", B), ("C", C)):
        X = X / np.linalg.norm(X, axis=1, keepdims=True); (TMP / f"emb_{nm}.bin").write_bytes(X.astype("<f2").tobytes())


if __name__ == "__main__":
    V2 = build_vocab_json(); print("rows", len(V2)); build_emb(V2)
