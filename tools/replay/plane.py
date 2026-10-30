"""Table-plane fitting (numpy only, Swift-portable). See VOLUME_ALGO.md section 1."""
from __future__ import annotations

import numpy as np

try:
    from cl_types import Plane
except ImportError:  # imported as a package
    from .cl_types import Plane

_MAX_SCORE_PTS = 5000


def _orient(n: np.ndarray, offset: float, cam_centers_w):
    """Flip (n, offset) so that the camera centres (or +Y world) are on the positive side."""
    if cam_centers_w is not None and len(cam_centers_w) > 0:
        c = np.asarray(cam_centers_w, dtype=np.float64).reshape(-1, 3)
        up = float(np.mean(c @ n - offset)) > 0.0
    else:
        up = n[1] >= 0.0
    return (n, offset) if up else (-n, -offset)


def _lsq_plane(pts: np.ndarray):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n / np.linalg.norm(n), c


def fit_plane_ransac(points_w: np.ndarray, thresh_m: float = 0.005, iters: int = 500,
                     seed: int = 0, cam_centers_w: np.ndarray | None = None) -> Plane:
    """3-point RANSAC + SVD least-squares refine on the inliers. Normal oriented UP."""
    P = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    rng = np.random.default_rng(seed)
    S = P if len(P) <= _MAX_SCORE_PTS else P[rng.choice(len(P), _MAX_SCORE_PTS, replace=False)]

    idx = rng.integers(0, len(S), size=(iters, 3))
    a, b, c = S[idx[:, 0]], S[idx[:, 1]], S[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-9
    if not ok.any():
        raise ValueError("degenerate (collinear) points")
    nrm = nrm[ok] / ln[ok, None]
    off = np.einsum("ij,ij->i", nrm, a[ok])
    # (iters, N) distance matrix; N <= 5000 so this stays small
    dist = np.abs(S @ nrm.T - off[None, :])
    inl = dist < thresh_m
    # score: inlier count, ties broken by smaller clipped residual
    score = inl.sum(axis=0) - 0.5 * np.minimum(dist, thresh_m).mean(axis=0) / thresh_m
    best = int(np.argmax(score))
    n, off0 = nrm[best], off[best]

    # refine on ALL points: two rounds of (inliers -> SVD)
    for _ in range(3):
        mask = np.abs(P @ n - off0) < thresh_m
        if mask.sum() < 3:
            break
        n, cen = _lsq_plane(P[mask])
        off0 = float(n @ cen)
    mask = np.abs(P @ n - off0) < thresh_m
    rmse = float(np.sqrt(np.mean((P[mask] @ n - off0) ** 2))) if mask.any() else 0.0
    n, off0 = _orient(n, float(off0), cam_centers_w)
    return Plane(n=n, offset=float(off0), inlier_rmse_m=rmse)
