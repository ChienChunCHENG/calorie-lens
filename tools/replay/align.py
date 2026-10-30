"""Relative depth -> metric depth, per frame.

Model: 1/d = s * rel + t   (s > 0)

Why disparity space: Depth Anything V2 predicts affine-invariant *inverse* depth, so rel is (up to noise) exactly
affine in 1/d. Fitting there is a plain linear least squares problem (no division by a small fitted number, no
blow-up of far pixels), and the unknowns (s, t) are well conditioned as long as the anchors span a range of rel.
Anchors are (1/d_metric, rel) pairs:
  (a) table: for background pixels (outside every mask, dilated by a few px) the metric depth is the ray-plane
      intersection with the table plane; (b) ARKit feature points projected into the frame (rel sampled bilinearly).
Robustness: RANSAC (2-point minimal fits, s > 0, inlier test in *metric depth* so the threshold means centimetres)
followed by iterated weighted LS on the inliers with a shrinking threshold. Frames with too few anchors, too small a
rel spread, too high a residual or a bad inlier fraction are rejected (depth_m stays None).

align_frames(frames, plane_hint=None) fills frame.depth_m in place and returns (plane, diags).
"""
from __future__ import annotations

import numpy as np

from cl_types import Frame, Plane

BG_DILATE_PX = 4
MAX_BG_ANCHORS = 2000
FP_WEIGHT = 4.0            # a feature point counts as this many background pixels
MIN_ANCHORS = 40
MIN_INLIER_FRAC = 0.5
MAX_RMSE_M = 0.015         # inlier RMSE in metric depth above which the frame is rejected
MIN_REL_SPREAD = 0.04      # (p95-p5 of anchor rel) / median rel
MIN_DISP = 1.0 / 5.0       # pixels whose fitted disparity is below this (> 5 m) are invalid (NaN)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask
    for ax in (0, 1):
        acc = out.copy()
        for k in range(1, r + 1):
            sh = np.roll(out, k, axis=ax)
            sh2 = np.roll(out, -k, axis=ax)
            acc |= sh | sh2
        out = acc
    return out


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = img.shape
    u0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    fu, fv = u - u0, v - v0
    a = img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu
    b = img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


def _ray_dirs(K, H, W):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones((H, W))], -1)


def _wls(x, y, w):
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum()
    if vx < 1e-12:
        return None
    s = (w * (x - mx) * (y - my)).sum() / vx
    return s, my - s * mx


def _depth_resid(x, y, s, t):
    """Metric-depth residual of anchor disparities y under the fit (inf where the fit has non-positive disparity)."""
    pred = s * x + t
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(1.0 / pred - 1.0 / y)
    return np.where(pred > 1e-3, r, np.inf)


def robust_affine_fit(x, y, w, rng, ransac_thr=0.02, iters=300):
    """Fit y = s x + t (y = anchor disparity, x = rel). Returns (s, t, inlier_mask) or None."""
    n = len(x)
    if n < 3:
        return None
    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(x[i] - x[j]) < 1e-6:
            continue
        s = (y[i] - y[j]) / (x[i] - x[j])
        if s <= 0:
            continue
        t = y[i] - s * x[i]
        inl = _depth_resid(x, y, s, t) < ransac_thr
        score = w[inl].sum()
        if score > best_score:
            best, best_score = (s, t), score
    if best is None:
        return None
    s, t = best
    inl = None
    for thr in (ransac_thr, 0.015, 0.01, 0.01):
        inl = _depth_resid(x, y, s, t) < thr
        if inl.sum() < 3:
            return None
        r = _wls(x[inl], y[inl], w[inl])
        if r is None or r[0] <= 0:
            return None
        s, t = r
    inl = _depth_resid(x, y, s, t) < 0.015
    return s, t, inl


def estimate_plane(frames: list[Frame], seed: int = 0) -> Plane:
    try:
        from plane import fit_plane_ransac
    except ImportError as e:  # pragma: no cover
        raise ImportError("plane.py (fit_plane_ransac) is required to estimate the table plane from feature points; "
                          "pass plane_hint=Plane(...) to skip it") from e
    pts = [f.feature_points_w for f in frames if f.feature_points_w is not None and len(f.feature_points_w)]
    if not pts:
        raise ValueError("no feature points in any frame and no plane_hint given")
    plane = fit_plane_ransac(np.concatenate(pts).astype(np.float64), seed=seed)
    return _orient_up(plane, frames)


def _orient_up(plane: Plane, frames: list[Frame]) -> Plane:
    cams = np.array([f.T_wc[:3, 3] for f in frames])
    if np.mean(cams @ plane.n - plane.offset) < 0:
        return Plane(n=-plane.n, offset=-plane.offset, inlier_rmse_m=plane.inlier_rmse_m)
    return plane


def align_frame(f: Frame, plane: Plane, rng: np.random.Generator) -> dict:
    """Fill f.depth_m (or leave None) and return a diagnostics dict."""
    H, W = f.rel_depth.shape
    rel = f.rel_depth.astype(np.float64)
    R, o = f.T_wc[:3, :3], f.T_wc[:3, 3]
    diag = {"ok": False, "reason": "", "s": None, "t": None, "n_bg": 0, "n_fp": 0, "n_inl": 0,
            "inlier_frac": 0.0, "rmse_m": None}

    # (a) table-plane anchors on background pixels
    union = np.zeros((H, W), bool)
    for m in f.masks.values():
        union |= m
    bg = ~_dilate(union, BG_DILATE_PX)
    bg[:2], bg[-2:], bg[:, :2], bg[:, -2:] = False, False, False, False
    Dw = _ray_dirs(f.K, H, W) @ R.T
    ndot = Dw @ plane.n
    with np.errstate(divide="ignore", invalid="ignore"):
        d_plane = (plane.offset - o @ plane.n) / ndot
    ok = bg & np.isfinite(d_plane) & (ndot < -1e-3) & (d_plane > 0.1) & (d_plane < 3.0)
    idx = np.flatnonzero(ok.ravel())
    if len(idx) > MAX_BG_ANCHORS:
        idx = rng.choice(idx, MAX_BG_ANCHORS, replace=False)
    xs = [rel.ravel()[idx]]
    ds = [d_plane.ravel()[idx]]
    ws = [np.ones(len(idx))]
    diag["n_bg"] = len(idx)

    # (b) feature points
    fp = f.feature_points_w
    if fp is not None and len(fp):
        pc = (np.asarray(fp, np.float64) - o) @ R          # world -> camera
        d = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f.K[0, 0] * pc[:, 0] / d + f.K[0, 2]
            v = -f.K[1, 1] * pc[:, 1] / d + f.K[1, 2]
        ins = (d > 0.05) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if ins.any():
            xs.append(_bilinear(rel, u[ins], v[ins]))
            ds.append(d[ins])
            ws.append(np.full(int(ins.sum()), FP_WEIGHT))
            diag["n_fp"] = int(ins.sum())

    x, d_a, w = np.concatenate(xs), np.concatenate(ds), np.concatenate(ws)
    if len(x) < MIN_ANCHORS:
        diag["reason"] = f"too few anchors ({len(x)})"
        return diag
    spread = (np.percentile(x, 95) - np.percentile(x, 5)) / max(abs(np.median(x)), 1e-9)
    diag["rel_spread"] = float(spread)
    if spread < MIN_REL_SPREAD:
        diag["reason"] = f"anchors span too little relative depth ({spread:.3f})"
        return diag
    y = 1.0 / d_a
    fit = robust_affine_fit(x, y, w, rng)
    if fit is None:
        diag["reason"] = "fit failed"
        return diag
    s, t, inl = fit
    res = _depth_resid(x, y, s, t)
    n_inl = int(inl.sum())
    rmse = float(np.sqrt(np.average(res[inl] ** 2, weights=w[inl]))) if n_inl else None
    diag.update(s=float(s), t=float(t), n_inl=n_inl, inlier_frac=float(w[inl].sum() / w.sum()), rmse_m=rmse)
    if n_inl < MIN_ANCHORS * 0.75:
        diag["reason"] = f"too few inliers ({n_inl})"
    elif diag["inlier_frac"] < MIN_INLIER_FRAC:
        diag["reason"] = f"inlier fraction {diag['inlier_frac']:.2f}"
    elif rmse > MAX_RMSE_M:
        diag["reason"] = f"residual {rmse * 100:.1f} cm"
    elif s <= 0:
        diag["reason"] = "non-positive scale"
    if diag["reason"]:
        return diag

    disp = s * rel + t
    depth = np.full((H, W), np.nan, np.float32)
    good = disp > MIN_DISP
    depth[good] = (1.0 / disp[good]).astype(np.float32)
    f.depth_m = depth
    diag["ok"] = True
    return diag


def align_frames(frames: list[Frame], plane_hint: Plane | None = None, seed: int = 0):
    """Align every frame in place. Returns (plane_used, [diag per frame])."""
    plane = _orient_up(plane_hint, frames) if plane_hint is not None else estimate_plane(frames, seed)
    rng = np.random.default_rng(seed)
    diags = []
    for f in frames:
        f.depth_m = None
        diags.append(align_frame(f, plane, rng))
    return plane, diags
