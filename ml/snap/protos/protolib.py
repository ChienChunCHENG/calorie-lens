"""Scoring variants that combine MobileCLIP text embeddings (vocab_emb) with image prototypes. Every scorer behaves like the text matrix used by
tune.py: `scorer @ e` (e = unit image embedding, float32 [512]) returns one effective cosine-like score per vocab row; the caller multiplies by
logit_scale (64.25) and softmaxes exactly as before, so gate/mixture code is unchanged."""
import json
from pathlib import Path
import numpy as np
import class_map as cm

CONCEPTS = list(cm.C)


def load_ids(models_dir=None):
    """row order of the EVALUATION vocab (ml/snap snaplib.build_vocab = research vocab + NEGS, 2,322 rows, matches the cached text matrix).
    build_protos.py maps by id onto whatever vocab.json is currently shipped."""
    import snaplib
    v = snaplib.build_vocab()
    return [x["id"] for x in v], np.array([bool(x.get("neg")) for x in v])


def concept_rows(ids):
    id2i = {x: i for i, x in enumerate(ids)}
    rows = [[id2i[v] for v in cm.C[k]["vocab"]] for k in CONCEPTS]
    rc = -np.ones(len(ids), int)
    for k, r in enumerate(rows):
        for i in r: rc[i] = k
    return rows, rc


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class TextOnly:
    def __init__(self, T): self.M = T
    def __matmul__(self, e): return self.M @ e


class Merged:
    """Zero-cost variant: a replacement text/vocab matrix (rows re-normalised)."""
    def __init__(self, M): self.M = M.astype(np.float32)
    def __matmul__(self, e): return self.M @ e


def merged_mean(T, rc, mu, lam):
    """row_i <- normalise(T_i + lam * mu_{concept(i)}) where mu = class-mean image embedding (unit). Rows without a concept are unchanged."""
    M = T.copy()
    for i in np.where(rc >= 0)[0]:
        M[i] = unit(T[i] + lam * mu[rc[i]])
    return M


class Tip:
    """score_i = T_i.e + a * (q_k - c) for rows of a concept k (q_k = mean of the top-m cosine similarities between e and the concept's prototypes),
    score_i = T_i.e otherwise.  `gate_free`: untrained rows get nothing (no penalty)."""
    def __init__(self, T, rc, P, pc, a, c, m=1, ncon=None, relu=False):
        self.T, self.rc, self.a, self.c, self.m, self.relu = T, rc, a, c, m, relu
        K = ncon or len(CONCEPTS); self.K = K
        order = np.argsort(pc, kind="stable"); self.P = P[order].astype(np.float32); pcs = pc[order]
        counts = np.bincount(pcs, minlength=K); self.has = counts > 0
        mx = counts.max(); idx = -np.ones((K, mx), int); st = 0
        for k in range(K):
            idx[k, :counts[k]] = np.arange(st, st + counts[k]); st += counts[k]
        self.idx = idx; self.mask = idx >= 0; self.idx0 = np.where(self.mask, idx, 0)
        self.trained = np.where(rc >= 0)[0]; self.rk = rc[self.trained]

    def q(self, e):
        s = self.P @ e; g = np.where(self.mask, s[self.idx0], -9.0)
        m = min(self.m, g.shape[1])
        top = -np.partition(-g, m - 1, axis=1)[:, :m] if m < g.shape[1] else g
        cnt = np.minimum(self.mask.sum(1), m).clip(1)
        top = np.where(top > -8, top, 0.0)
        return top.sum(1) / cnt

    def __matmul__(self, e):
        out = self.T @ e; q = self.q(e)
        ok = self.has[self.rk]
        d = q[self.rk[ok]] - self.c
        out[self.trained[ok]] += self.a * (np.maximum(d, 0) if self.relu else d)
        return out


def split_dev(paths, frac=0.3):
    """deterministic train/dev split by hash of the image identity"""
    import hashlib
    return np.array([int(hashlib.md5(str(p).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac for p in paths])
