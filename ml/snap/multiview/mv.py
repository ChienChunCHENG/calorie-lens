"""Multi-view food size from a short video, plain-browser constraints (no ARKit / LiDAR / poses). Reference implementation.

Pipeline (see SPEC.md for the exact numbers a TypeScript port needs):
  per frame   1. plate ellipse   : YOLOE-seg 'plate/bowl/tray' mask -> ellipse (border-clipped contour points dropped)       [stage1.py, geom.py]
              2. camera pose     : circle of known diameter -> plane normal n + centre c (closed form, 2-fold ambiguity -> camera above table, small roll)
              3. food footprint  : pixels inside the plate ellipse that differ from the plate colour (portion.food_pixels) + YOLOE 'food' mask near the plate
              4. monocular depth : Depth Anything V2 small -> relative inverse depth; affine-aligned to the plate plane (robust fit on plate pixels
                                   without food) -> metric depth of the food pixels -> height above the plate plane -> height map on the plate grid (cm)
              5. frame volume    : sum of heights over the footprint cells (mL)
  per video   6. aggregation     : median (or a high quantile) of the frame volumes across the selected frames
              7. grams/kcal      : grams = k * V^beta ; kcal = grams * density-shrunk kcal/g of the recognised food (portion.Params) ; MC range
Alternative (visual hull): see hull.py (needs the azimuth between frames, from rotation.py).
"""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "portion"))
import geom
import portion as PT

W, H = 640, 360
CELL = 0.5  # cm
GRID = 64    # 32 cm / 0.5 cm: plate-centred grid covering +-16 cm


def unpack(m): return np.unpackbits(m)[:H * W].reshape(H, W).astype(bool)


def choose_plate(res, min_area_frac=0.04):
    """best plate candidate of a stage-1 frame result: largest valid ellipse among candidates with score >= 0.15, not heavily border-clipped."""
    best = None
    for p in res["plates"]:
        e = p["ell"]
        if e is None: continue
        s = p["score"] * (1.0 if p["cls"] >= 14 else 0.8)
        if s < 0.15 or p["area"] < min_area_frac * H * W or e["clip"] > 0.45: continue
        if e["minor"] < 0.15 * e["major"]: continue
        sc = s * p["area"]
        if best is None or sc > best[0]: best = (sc, p)
    return None if best is None else best[1]


def plate_diam(p, D_plate=26.0, D_bowl=24.0):
    return D_bowl if p["cls"] in (15, 2) else D_plate


def robust_affine(d, y, iters=8, w_lo=0.0):
    """y ~ a*d + b with Huber IRLS. Returns (a, b)."""
    A = np.stack([d, np.ones_like(d)], 1); w = np.ones(len(d))
    for _ in range(iters):
        sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        r = y - A @ sol; s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12)))
    return sol


class Frame:
    """all per-frame quantities. Build with Frame.make(img, res, disp, params)."""

    @staticmethod
    def make(img, res, disp, prm, fscale=1.0, diam=None):
        p = choose_plate(res)
        if p is None: return None
        e = p["ell"]; K = geom.K_of(W, H, prm.get("hfov", 69.0), fscale)
        D = diam if diam else plate_diam(p, prm.get("D_plate", 26.0), prm.get("D_bowl", 24.0))
        pose = geom.best_pose(e, K, D)
        if pose is None: return None
        n, c, tilt = pose
        f = Frame(); f.img, f.res, f.e, f.K, f.n, f.c, f.tilt, f.D, f.disp, f.cls = img, res, e, K, n, c, tilt, D, disp, p["cls"]
        return f


def food_mask(f, prm=None):
    """footprint of the food in the image: plate-colour difference inside the ellipse (portion.food_pixels, thr 18) minus thin structures (plate rim bevels,
    printed rings: opening with a `food_open`-px disc) plus the YOLOE food mask in a band around the ellipse (food rising above the rim of an obliquely seen plate)."""
    prm = prm or {}
    fg, ref = PT.food_pixels(f.img, f.e, thr=prm.get("food_thr", 18.0))
    if ref is None: return None
    k = int(prm.get("food_open", 15))
    if k > 1: fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    fg &= PT.ellipse_mask((H, W), dict(f.e), prm.get("fg_shrink", 0.95))
    ym = unpack(f.res["food"])
    em = PT.ellipse_mask((H, W), dict(f.e), 1.25)
    out = fg | (ym & em & ~PT.ellipse_mask((H, W), dict(f.e), 0.97) & (prm.get("band", 1) > 0))
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def frame_heightmap(f, prm):
    """-> dict(vol_ml, area_cm2, hmean, hmax, fit_rmse, slope_ok, hm (GRID x GRID cm), cnt) or None"""
    fm = food_mask(f, prm)
    if fm is None or fm.sum() < 200: return None
    d = cv2.resize(f.disp.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    Zp, r = geom.plane_depth_map(f.n, f.c, f.K, (H, W))
    ell_in = PT.ellipse_mask((H, W), dict(f.e), prm.get("fit_shrink", 0.95))
    fd = cv2.dilate(fm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    sel = ell_in & ~fd & (Zp > 5) & (Zp < 300)
    # the plate colour reference ring pixels only: exclude pixels differing from the plate colour (already done by ~fd); thin out
    ys, xs = np.nonzero(sel)
    if len(ys) < 300: return None
    if len(ys) > 6000:
        k = np.random.default_rng(0).choice(len(ys), 6000, replace=False); ys, xs = ys[k], xs[k]
    dd = d[ys, xs]; y = 1.0 / Zp[ys, xs]
    # normalise d for conditioning
    mu, sd = dd.mean(), dd.std() + 1e-6
    a, b = robust_affine((dd - mu) / sd, y)
    pred = a * (dd - mu) / sd + b; rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    relrmse = rmse / float(np.mean(y))
    a_ok = float(abs(a) * sd / (np.mean(y)))  # how much 1/Z varies per unit of the disparity spread: conditioning
    # food pixels -> metric points
    fy, fx = np.nonzero(fm)
    inv = a * (d[fy, fx] - mu) / sd + b
    ok = inv > 1.0 / 300
    fy, fx, inv = fy[ok], fx[ok], inv[ok]
    Z = 1.0 / inv
    P = r[fy, fx] * Z[:, None]
    h = (P - f.c) @ f.n
    A, B = geom.plane_basis(f.n)
    base = P - np.outer(h, f.n) - f.c
    gx, gy = base @ A, base @ B
    hm = np.clip(h, -3, prm.get("hmax", 12.0))
    ci = np.floor(gx / CELL + GRID / 2).astype(int); cj = np.floor(gy / CELL + GRID / 2).astype(int)
    inside = (ci >= 0) & (ci < GRID) & (cj >= 0) & (cj < GRID)
    ci, cj, hm = ci[inside], cj[inside], hm[inside]
    if len(ci) < 50: return None
    hmap, cnt = grid_median(ci, cj, hm)
    foot = cnt > 0
    # close small holes (occlusion by food itself / sparse sampling at grazing angles), inpaint with neighbour mean
    foot_c = cv2.morphologyEx(foot.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    hfill = inpaint(hmap, foot, foot_c)
    hfill = np.maximum(hfill, 0)
    vol = float(hfill[foot_c].sum() * CELL * CELL)
    area = float(foot_c.sum() * CELL * CELL)
    return dict(vol_ml=vol, area_cm2=area, hmean=float(hfill[foot_c].mean()), hmax=float(np.percentile(hfill[foot_c], 95)), relrmse=relrmse, cond=a_ok,
                tilt=f.tilt, hm=hfill * foot_c)


def grid_median(ci, cj, hm):
    key = cj * GRID + ci
    order = np.argsort(key); key, hm = key[order], hm[order]
    u, start, cnt = np.unique(key, return_index=True, return_counts=True)
    med = np.array([np.median(hm[s:s + c]) for s, c in zip(start, cnt)])
    out = np.zeros(GRID * GRID); n = np.zeros(GRID * GRID, int)
    out[u] = med; n[u] = cnt
    return out.reshape(GRID, GRID), n.reshape(GRID, GRID)


def inpaint(h, known, target, iters=10):
    h = h.copy(); kn = known.copy()
    for _ in range(iters):
        need = target & ~kn
        if not need.any(): break
        k = kn.astype(np.float32); s = cv2.blur(h * k, (3, 3)); c = cv2.blur(k, (3, 3))
        fill = need & (c > 0)
        h[fill] = s[fill] / c[fill]; kn = kn | fill
    return h
