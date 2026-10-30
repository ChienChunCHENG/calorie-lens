"""Per-item food volume from metric depth + masks + table plane (numpy only, Swift-portable).

Spec: VOLUME_ALGO.md.  Entry point: compute_item_volume().
"""
from __future__ import annotations

import math

import numpy as np

try:
    from cl_types import Frame, Plane, Triple, VolumeResult
except ImportError:  # imported as a package
    from .cl_types import Frame, Plane, Triple, VolumeResult

H_MAX = 0.25            # clamp on height above table (m)
MIN_VALID_CELLS = 20
FILL_ITERS = 12         # inpainting passes (12 x 3 mm = 36 mm reach)
FP_FLOOR = 0.0
FP_SAT = 0.5            # footprint weight = min(1, presence frequency / FP_SAT)
HULL_VOTE = 0.8         # table-level projection vote threshold (see _vote_footprint)
SUB_N = 3               # table-level vote sampled at SUB_N x SUB_N points per cell
BASE_THICK_M = 0.005    # bowl base thickness: cavity floor sits this far above the table
SCALE_ERR = 0.06        # +-6 % linear depth scale error (treated as 90 % interval)
Z90 = 1.645
DRINK_LABELS = {"bubble_milk_tea", "container_cup", "black_tea", "green_tea", "soy_milk"}
DEFAULT_BOWL = {"rim_diameter_cm": {"min": 12.5, "typ": 13.5, "max": 14.5},
                "depth_cm": {"min": 6.0, "typ": 6.5, "max": 7.5}}


# ---------------------------------------------------------------- helpers
def _tri(x, scale=1.0):
    """container field (triple dict or scalar) -> (min, typ, max) * scale."""
    if isinstance(x, dict):
        return x["min"] * scale, x["typ"] * scale, x["max"] * scale
    return x * scale, x * scale, x * scale


def plane_basis(n: np.ndarray):
    """Deterministic orthonormal (a, b) spanning the plane."""
    ref = np.zeros(3)
    ref[int(np.argmin(np.abs(n)))] = 1.0
    a = np.cross(n, ref)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return a, b


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """4-neighbour binary erosion, px iterations (border pixels count as background)."""
    m = mask.astype(bool)
    for _ in range(px):
        p = np.pad(m, 1, constant_values=False)
        m = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def unproject_world(depth, K, T_wc, sel):
    """World points for pixels where sel (bool HxW) is True. ARKit convention (cl_types)."""
    vv, uu = np.nonzero(sel)
    d = depth[vv, uu].astype(np.float64)
    x = (uu - K[0, 2]) * d / K[0, 0]
    y = -(vv - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, -d, np.ones_like(d)], axis=1)
    return (pc @ T_wc.T)[:, :3]


def _nb4(a):
    """Sum over the 4-neighbourhood (zero padded)."""
    p = np.pad(a, 1)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _fill(H, Wt, footprint, iters=FILL_ITERS, fpw=None):
    """Inpaint unobserved footprint cells (holes AND the eroded edge ring), nothing outside the footprint.
    Each pass: every unknown footprint cell with >= 1 known 4-neighbour takes the mean height of its known
    4-neighbours (simultaneous update), weight 1. Known = observed cells (Wt > 0) or cells filled in earlier
    passes. Returns (H, Wt)."""
    H = np.where(Wt > 0, H, 0.0)
    Wt = Wt.copy()
    for _ in range(iters):
        known = Wt > 0
        cnt = _nb4(known.astype(np.float64))
        cand = footprint & ~known & (cnt > 0)
        if not cand.any():
            break
        sm = _nb4(np.where(known, H, 0.0))
        H = np.where(cand, sm / np.maximum(cnt, 1), H)
        Wt = np.where(cand, fpw if fpw is not None else 1.0, Wt)
    return H, Wt


def _vote_footprint(cand, frames, item_key, plane, a_ax, b_ax, cu, cv, cell, vote, sub=SUB_N):
    """Footprint area fraction per cell in [0,1] by projection voting against the UN-eroded masks.
    The cell is sampled at sub x sub points (sub-cell positions) at table height (h = 0). A sample point is
    'in' when it projects inside the item mask in >= `vote` of the frames that see it (in the image, in
    front of the camera). Frames where the item's own body occludes the point still vote 'in' (the mask covers
    it), which is right: the table point lies under the item. Cell fraction = mean over its sample points.
    Only cells in `cand` are evaluated."""
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    iu, iv = np.nonzero(cand)
    pts = []
    for du in offs:
        for dv in offs:
            pu = cu[iu, iv] + du * cell
            pv = cv[iu, iv] + dv * cell
            pts.append(pu[:, None] * a_ax + pv[:, None] * b_ax + plane.offset * plane.n)
    P = np.concatenate(pts)                                  # (sub*sub*M, 3)
    ins = np.zeros(len(P))
    see = np.zeros(len(P))
    for f in frames:
        m = f.masks[item_key]
        Hh, Ww = m.shape
        pc = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(f.T_wc).T)[:, :3]
        d = -pc[:, 2]
        ok = d > 1e-6
        dd = np.where(ok, d, 1.0)
        u = np.rint(f.K[0, 0] * pc[:, 0] / dd + f.K[0, 2]).astype(int)
        v = np.rint(f.K[1, 2] - f.K[1, 1] * pc[:, 1] / dd).astype(int)
        ok &= (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
        see += ok
        ins += ok & m[np.clip(v, 0, Hh - 1), np.clip(u, 0, Ww - 1)].astype(bool)
    pin = (see > 0) & (ins >= vote * np.maximum(see, 1))
    out = np.zeros(cand.shape)
    out[iu, iv] = pin.reshape(sub * sub, -1).mean(axis=0)
    return out


def _median_grid(keys, hs, ncell, shape, min_obs):
    """Per-cell median of hs over observations with key; cells with < min_obs -> NaN."""
    G = np.full(ncell, np.nan)
    if len(keys) == 0:
        return G.reshape(shape), np.zeros(shape)
    order = np.lexsort((hs, keys))
    k, h = keys[order], hs[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[starts, len(k)])
    med = 0.5 * (h[starts + (counts - 1) // 2] + h[starts + counts // 2])
    keep = counts >= min_obs
    G[k[starts][keep]] = med[keep]
    C = np.zeros(ncell)
    C[k[starts][keep]] = counts[keep]
    return G.reshape(shape), C.reshape(shape)


FRAME_REDUCE = "median"   # "max" per spec biases +15 % under 5 mm noise; see VOLUME_ALGO.md
WEIGHT_SAT = 0.6        # only for observed cells outside the voted footprint


def _frame_max_per_cell(keys, hs):
    order = np.lexsort((hs, keys))
    k, h = keys[order], hs[order]
    last = np.flatnonzero(np.r_[k[1:] != k[:-1], True])
    if FRAME_REDUCE == "max":
        return k[last], h[last]
    first = np.r_[0, last[:-1] + 1]
    cnt = last - first + 1
    if FRAME_REDUCE == "median":
        return k[last], 0.5 * (h[first + (cnt - 1) // 2] + h[first + cnt // 2])
    # "upper": mean of top half
    c = np.r_[0.0, np.cumsum(h)]
    top = np.maximum(cnt // 2, 1)
    return k[last], (c[last + 1] - c[last + 1 - top]) / top


def _cap_floor(r, rim_r, depth, base):
    """Cavity floor height above the table at horizontal distance r (spherical cap)."""
    Rs = (rim_r ** 2 + depth ** 2) / (2.0 * depth)
    rr = np.minimum(r, rim_r)
    return base + Rs - np.sqrt(np.maximum(Rs ** 2 - rr ** 2, 0.0))


def _volume_m3(H, Wt, cell, cu, cv, bowl):
    """bowl: None or (centre_u, centre_v, rim_r, depth, base). Cells weighted by Wt in [0,1]."""
    valid = Wt > 0
    h, w = H[valid], Wt[valid]
    if bowl is None:
        return float((w * h).sum() * cell * cell)
    c_u, c_v, rim_r, depth, base = bowl
    r = np.hypot(cu[valid] - c_u, cv[valid] - c_v)
    return float((w * np.maximum(h - _cap_floor(r, rim_r, depth, base), 0.0)).sum() * cell * cell)


def _sigma90(lo, hi):
    return math.log(max(hi, 1e-12) / max(lo, 1e-12)) / 3.29


# ---------------------------------------------------------------- main
def compute_item_volume(frames: list, item_key: str, plane, label: str, container: dict | None = None,
                        cell_m: float = 0.003, min_obs: int = 3, *, erode_px: int = 1, fp_sat: float = FP_SAT, fp_floor: float = FP_FLOOR, hull_vote: float = HULL_VOTE, fill: bool = True,
                        n_boot: int = 50, seed: int = 0) -> VolumeResult:
    notes: list[str] = []

    # ---- clear cup / drink: never trust depth
    shape = (container or {}).get("shape")
    if label in DRINK_LABELS or shape == "cup":
        return VolumeResult(item_key, None, "cup_prior", 0, 0.0,
                            ["depth unreliable through clear plastic"])

    # ---- gather per-frame world points
    a_ax, b_ax = plane_basis(plane.n)
    pts = []          # per frame: (pu, pv, h, is_eroded)
    pframes = []      # the Frame of each pts entry (for footprint voting)
    pmasks = []
    cont_uv = []      # container-mask points (plane coords)
    bowl_label_seen = False
    for f in frames:
        if f.depth_m is None:
            continue
        valid = np.isfinite(f.depth_m) & (f.depth_m > 0)
        if f.conf is not None:
            valid &= f.conf >= 0.3
        m = f.masks.get(item_key)
        if m is not None:
            sel = m.astype(bool) & valid
            if sel.any():
                er = erode(m, erode_px)[sel]
                P = unproject_world(f.depth_m, f.K, f.T_wc, sel)
                pts.append((P @ a_ax, P @ b_ax, plane.height(P), er))
                pframes.append(f)
        for ck, cl in f.labels.items():
            if cl == "container_bowl" and ck != item_key and ck in f.masks:
                bowl_label_seen = True
                cs = f.masks[ck].astype(bool) & valid
                if cs.any():
                    Pc = unproject_world(f.depth_m, f.K, f.T_wc, cs)
                    cont_uv.append(np.stack([Pc @ a_ax, Pc @ b_ax], axis=1))
    if not pts:
        return VolumeResult(item_key, None, "failed", 0, 0.0, ["no frames with depth and mask"])

    # reject garbage depth: far from the plane or far from the item in-plane
    allu = np.concatenate([p[0] for p in pts])
    allv = np.concatenate([p[1] for p in pts])
    mu, mv = np.median(allu), np.median(allv)
    cleaned, cframes = [], []
    for (pu, pv, h, er), fr in zip(pts, pframes):
        ok = (h > -0.05) & (h < 0.5) & (np.hypot(pu - mu, pv - mv) < 0.5)
        if ok.any():
            cleaned.append((pu[ok], pv[ok], np.clip(h[ok], 0.0, H_MAX), er[ok]))
            cframes.append(fr)
    if not cleaned:
        return VolumeResult(item_key, None, "failed", 0, 0.0, ["all depth points implausible"])
    pts, pframes = cleaned, cframes

    u0 = min(p[0].min() for p in pts) - 3 * cell_m
    v0 = min(p[1].min() for p in pts) - 3 * cell_m
    nu = int(math.ceil((max(p[0].max() for p in pts) - u0) / cell_m)) + 4
    nv = int(math.ceil((max(p[1].max() for p in pts) - v0) / cell_m)) + 4
    shape2 = (nu, nv)
    ncell = nu * nv

    red = []                       # per-frame (keys, h) after erosion + max-per-cell
    fp_er_keys = []
    pres = np.zeros(ncell)             # number of frames whose UN-eroded mask samples touch the cell
    for pu, pv, h, er in pts:
        k = (np.floor((pu - u0) / cell_m).astype(np.int64) * nv
             + np.floor((pv - v0) / cell_m).astype(np.int64))
        pres[np.unique(k)] += 1
        if er.any():
            fp_er_keys.append(k[er])
            red.append(_frame_max_per_cell(k[er], h[er]))
    if not red:
        return VolumeResult(item_key, None, "failed", 0, 0.0, ["mask empty after erosion"])
    fp_er = np.zeros(ncell, bool)
    fp_er[np.concatenate(fp_er_keys)] = True
    footprint = np.zeros(shape2, bool)
    fpw = np.zeros(shape2)
    n_fp = int(fp_er.sum())

    I, J = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    cu = u0 + (I + 0.5) * cell_m
    cv = v0 + (J + 0.5) * cell_m

    def grid(sel_frames):
        keys = np.concatenate([red[i][0] for i in sel_frames])
        hs = np.concatenate([red[i][1] for i in sel_frames])
        G, C = _median_grid(keys, hs, ncell, shape2, min_obs)
        Wt = np.where(C > 0, np.minimum(C / (WEIGHT_SAT * len(sel_frames)), 1.0), 0.0)
        Wt = np.where((C > 0) & footprint, fpw, Wt)        # voted footprint: area fraction
        return G, Wt

    G, W0 = grid(range(len(red)))
    n_valid = int((W0 > 0).sum())
    if n_valid < MIN_VALID_CELLS:
        return VolumeResult(item_key, None, "failed", n_valid, 0.0,
                            [f"only {n_valid} cells with >= {min_obs} observations"])
    coverage = min(1.0, n_valid / max(n_fp, 1))
    # footprint membership = presence frequency of the UN-eroded mask in the cell, saturating at fp_sat
    pf = pres.reshape(shape2) / len(pts)
    fpw = np.clip((pf - fp_floor) / (fp_sat - fp_floor), 0.0, 1.0)
    if hull_vote > 0:
        fpw = fpw * _vote_footprint(fpw > 0, pframes, item_key, plane, a_ax, b_ax, cu, cv, cell_m, hull_vote)
    footprint = fpw > 0
    G, W0 = grid(range(len(red)))
    Gf, Wf = _fill(G, W0, footprint, FILL_ITERS if fill else 0, fpw)
    validf = Wf > 0

    # ---- bowl?
    is_bowl = shape == "bowl" or (shape is None and bowl_label_seen)
    bowl_typ = None
    method = "heightmap"
    cont_sigma = 0.0
    if is_bowl:
        method = "bowl_cavity"
        cont = container if (container and "rim_diameter_cm" in container) else None
        cuv = np.concatenate(cont_uv) if cont_uv else None
        if cont is None:
            cont = dict(DEFAULT_BOWL)
            if cuv is not None:
                lo, hi = np.percentile(cuv, 2, axis=0), np.percentile(cuv, 98, axis=0)
                d_est = float(np.mean(hi - lo)) * 100.0
                cont["rim_diameter_cm"] = {"min": d_est * 0.95, "typ": d_est, "max": d_est * 1.05}
                notes.append("rim diameter estimated from container mask")
            else:
                notes.append("no container dims: generic medium-bowl prior used")
        d_min, d_typ, d_max = _tri(cont["rim_diameter_cm"], 0.01)
        h_min, h_typ, h_max = _tri(cont["depth_cm"], 0.01)
        if cuv is not None:
            centre = 0.5 * (np.percentile(cuv, 2, axis=0) + np.percentile(cuv, 98, axis=0))
        else:
            centre = np.array([cu[validf].mean(), cv[validf].mean()])

        def bowl_fn(rim_d, depth):
            return (centre[0], centre[1], 0.5 * rim_d, depth, BASE_THICK_M)

        bowl_typ = bowl_fn(d_typ, h_typ)
        vs = [_volume_m3(Gf, Wf, cell_m, cu, cv, bowl_fn(d, h))
              for d in (d_min, d_max) for h in (h_min, h_max)]
        vs.append(_volume_m3(Gf, Wf, cell_m, cu, cv, bowl_typ))
        if min(vs) > 0:
            cont_sigma = _sigma90(min(vs), max(vs))
            if max(vs) / min(vs) > 2.0:
                notes.append("food volume very sensitive to container dims")
    elif shape in ("plate", "box"):
        notes.append(f"container shape {shape}: plain heightmap")

    v_typ = _volume_m3(Gf, Wf, cell_m, cu, cv, bowl_typ)
    if v_typ <= 0:
        return VolumeResult(item_key, None, "failed", n_valid, coverage, ["non-positive volume"])

    # ---- bootstrap over frames
    nf = len(red)
    if nf >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            Gb, vb = grid(rng.integers(0, nf, size=nf))
            if (vb > 0).sum() < MIN_VALID_CELLS:
                continue
            Gb, vb = _fill(Gb, vb, footprint, FILL_ITERS if fill else 0, fpw)
            v = _volume_m3(Gb, vb, cell_m, cu, cv, bowl_typ)
            if v > 0:
                vals.append(v)
        if len(vals) >= 10:
            p5, p95 = np.percentile(vals, [5, 95])
            sigma_boot = _sigma90(p5, p95)
        else:
            sigma_boot = 0.25
            notes.append("bootstrap unstable (few frames): default spread used")
    else:
        sigma_boot = 0.25
        notes.append("single frame: default spread used")

    # ---- interval (log space; PLAN section 5)
    sigma_scale = _sigma90((1 - SCALE_ERR) ** 3, (1 + SCALE_ERR) ** 3)
    sigma_boot = min(sigma_boot, 0.4)   # cap: bootstrap on a handful of cells is meaningless
    sigma2 = sigma_boot ** 2 + sigma_scale ** 2 + cont_sigma ** 2
    if coverage < 0.5:
        sigma2 += (0.5 * math.log(0.5 / max(coverage, 0.2))) ** 2 + 0.1 ** 2
        notes.append(f"low footprint coverage {coverage:.2f}: interval widened, volume likely underestimated")
    sigma = math.sqrt(sigma2)
    typ_ml = v_typ * 1e6
    tri = Triple(typ_ml * math.exp(-Z90 * sigma), typ_ml, typ_ml * math.exp(Z90 * sigma))
    return VolumeResult(item_key, tri, method, n_valid, float(coverage), notes)
