"""Synthetic ground-truth sweeps for the replay pipeline.

Everything is analytic: solids have a closed-form volume, a point-in-solid test (for brute-force voxel checks) and
ray intersections (vectorised), so exact metric depth and masks come from a numpy ray caster. Relative depth is
then *simulated* the way Depth Anything V2 behaves (affine disparity + noise + smoothing + edge bleeding).

Local table frame: origin on the table, +z = table normal (up), x/y in the table plane. World frame = slightly
tilted copy of it (Y roughly up, like ARKit gravity-aligned world).

make_scene(kind, seed) -> (frames, gt)   kinds: see KINDS.
gt keys: kind, seed, volumes_ml{key: mL}, plane (true Plane), labels{key: class id}, containers{food_key: dict},
         container_keys[list], solids{key: Solid}, depth_true[list of (H,W) float32], T_wc_true[list], note{key: str}
"""
from __future__ import annotations

import math

import numpy as np

from cl_types import Frame, Plane

KINDS = ("rice_mound", "fried_egg", "cutlet", "bowl_flat", "bowl_dome", "clear_cup", "meal")
IMG_W, IMG_H = 256, 192
FX = FY = 200.0
N_FRAMES = 12
INF = np.inf


# ----------------------------------------------------------------------------- ray primitives
def _sphere_roots(o, D, c, R):
    """Both roots (t1<=t2) of |o + tD - c| = R for unnormalised D; nan if miss."""
    oc = o - c
    a = np.einsum("ij,ij->i", D, D)
    b = D @ oc
    cc = oc @ oc - R * R
    disc = b * b - a * cc
    s = np.sqrt(np.where(disc >= 0, disc, np.nan))
    return (-b - s) / a, (-b + s) / a


def _valid(t, p_ok):
    return np.where(np.isfinite(t) & (t > 1e-6) & p_ok, t, INF)


def _plane_z(o, D, z, region):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - o[2]) / D[:, 2]
    p = o + t[:, None] * D
    return _valid(t, region(p))


def _cyl_hits(o, D, cx, cy, r, z0, z1, caps=("top",)):
    """Vertical cylinder side + selected caps. Returns list of t arrays."""
    dx, dy = D[:, 0], D[:, 1]
    ox, oy = o[0] - cx, o[1] - cy
    a = dx * dx + dy * dy
    b = dx * ox + dy * oy
    c = ox * ox + oy * oy - r * r
    disc = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1, t2 = (-b - s) / a, (-b + s) / a
    out = []
    for t in (t1, t2):
        z = o[2] + t * D[:, 2]
        out.append(_valid(t, (z >= z0) & (z <= z1)))
    inside = lambda p: (p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2 <= r * r
    if "top" in caps:
        out.append(_plane_z(o, D, z1, inside))
    if "bottom" in caps:
        out.append(_plane_z(o, D, z0, inside))
    return out


class Solid:
    """A solid with analytic volume, point membership and ray hits."""

    def __init__(self, key, label, vol_ml, contains, bbox, hits, opaque=True):
        self.key, self.label, self.vol_ml = key, label, vol_ml
        self.contains, self.bbox, self._hits, self.opaque = contains, bbox, hits, opaque

    def hit_t(self, o, D):
        ts = self._hits(o, D)
        return np.min(np.stack(ts), axis=0)


def _cap_vol(R, h):
    return math.pi * h * h * (3 * R - h) / 3.0


def make_mound(key, label, cx, cy, a, h):
    R = (a * a + h * h) / (2 * h)
    c = np.array([cx, cy, h - R])

    def contains(P):
        return (np.sum((P - c) ** 2, axis=1) <= R * R) & (P[:, 2] >= 0)

    def hits(o, D):
        return [_valid(t, (o[2] + t * D[:, 2]) >= -1e-9) for t in _sphere_roots(o, D, c, R)]

    vol = math.pi * h * (3 * a * a + h * h) / 6.0 * 1e6
    return Solid(key, label, vol, contains, ((cx - a, cy - a, 0), (cx + a, cy + a, h)), hits)


def make_cylinder(key, label, cx, cy, r, h, opaque=True):
    def contains(P):
        return ((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2 <= r * r) & (P[:, 2] >= 0) & (P[:, 2] <= h)

    def hits(o, D):
        return _cyl_hits(o, D, cx, cy, r, 0.0, h, caps=("top",))

    return Solid(key, label, math.pi * r * r * h * 1e6, contains, ((cx - r, cy - r, 0), (cx + r, cy + r, h)), hits, opaque)


def make_box(key, label, cx, cy, sx, sy, h, yaw):
    cs, sn = math.cos(yaw), math.sin(yaw)
    Rm = np.array([[cs, -sn], [sn, cs]])  # box->local; local->box uses Rm.T

    def to_box(P):
        q = (P[:, :2] - [cx, cy]) @ Rm  # = Rm.T applied to row vectors
        return q

    def contains(P):
        q = to_box(P)
        return (np.abs(q[:, 0]) <= sx / 2) & (np.abs(q[:, 1]) <= sy / 2) & (P[:, 2] >= 0) & (P[:, 2] <= h)

    def hits(o, D):
        oq = (o[:2] - [cx, cy]) @ Rm
        Dq = D[:, :2] @ Rm
        lo = np.stack([np.full(len(D), -sx / 2), np.full(len(D), -sy / 2), np.zeros(len(D))], 1)
        hi = np.stack([np.full(len(D), sx / 2), np.full(len(D), sy / 2), np.full(len(D), h)], 1)
        o3 = np.array([oq[0], oq[1], o[2]])
        D3 = np.concatenate([Dq, D[:, 2:3]], 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (lo - o3) / D3
            t_hi = (hi - o3) / D3
        tn = np.nanmax(np.minimum(t_lo, t_hi), axis=1)
        tf = np.nanmin(np.maximum(t_lo, t_hi), axis=1)
        return [_valid(tn, tn < tf)]

    ext = 0.5 * math.hypot(sx, sy)
    return Solid(key, label, sx * sy * h * 1e6, contains, ((cx - ext, cy - ext, 0), (cx + ext, cy + ext, h)), hits)


def make_bowl_with_food(food_key, food_label, bowl_key, fill: str):
    """Spherical-cap cavity bowl (rim radius 6 cm, depth 4.5 cm) + food. fill: 'flat' | 'dome'."""
    a, D_, zb, wall = 0.06, 0.045, 0.005, 0.004   # zb = cavity floor height (volume.py assumes ~5 mm)
    Ri = (a * a + D_ * D_) / (2 * D_)
    Ro = Ri + wall
    zr = zb + D_
    c = np.array([0.0, 0.0, zb + Ri])
    r_rim_out2 = Ro * Ro - (zr - c[2]) ** 2

    def b_hits(o, D):
        out = []
        for t in _sphere_roots(o, D, c, Ro):
            z = o[2] + t * D[:, 2]
            out.append(_valid(t, (z <= zr) & (z >= zb - wall + 1e-4)))
        for t in _sphere_roots(o, D, c, Ri):
            z = o[2] + t * D[:, 2]
            out.append(_valid(t, z <= zr))
        out.append(_plane_z(o, D, zr, lambda p: (p[:, 0] ** 2 + p[:, 1] ** 2 >= a * a - 1e-12) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= r_rim_out2)))
        out += _cyl_hits(o, D, 0.0, 0.0, 0.03, 0.0, zb - wall + 5e-4, caps=())
        return out

    bowl = Solid(bowl_key, "container_bowl", _cap_vol(Ri, D_) * 1e6, lambda P: np.zeros(len(P), bool),
                 ((-0.07, -0.07, 0), (0.07, 0.07, zr)), b_hits)  # vol_ml = capacity (brim), not shell volume

    if fill == "flat":
        f = 0.030
        zf = zb + f
        reg = lambda p: np.sum((p - c) ** 2, axis=1) < Ri * Ri
        vol = _cap_vol(Ri, f)

        def contains(P):
            return (np.sum((P - c) ** 2, axis=1) <= Ri * Ri) & (P[:, 2] <= zf)

        def f_hits(o, D):
            return [_plane_z(o, D, zf, reg)]

        top = zf
    else:
        hd = 0.025
        Rd = (a * a + hd * hd) / (2 * hd)
        cd = np.array([0.0, 0.0, zr + hd - Rd])
        vol = _cap_vol(Ri, D_) + _cap_vol(Rd, hd)

        def contains(P):
            in_cav = (np.sum((P - c) ** 2, axis=1) <= Ri * Ri) & (P[:, 2] <= zr)
            in_dome = (np.sum((P - cd) ** 2, axis=1) <= Rd * Rd) & (P[:, 2] >= zr)
            return in_cav | in_dome

        def f_hits(o, D):
            return [_valid(t, (o[2] + t * D[:, 2]) >= zr - 1e-9) for t in _sphere_roots(o, D, cd, Rd)]

        top = zr + hd
    food = Solid(food_key, food_label, vol * 1e6, contains, ((-a, -a, zb), (a, a, top)), f_hits)
    cont = {"shape": "bowl", "item_key": bowl_key, "label": "container_bowl",
            "rim_diameter_cm": 2 * a * 100, "depth_cm": D_ * 100, "clear": False}
    return bowl, food, cont


# ----------------------------------------------------------------------------- helpers
def _blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = a.astype(np.float64)
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[ax]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * (p[i:i + n] if ax == 0 else p[:, i:i + n])
        out = acc
    return out


def simulate_relative_depth(depth, masks, rng):
    """Depth Anything-like output: rel = a * (1/depth) + b with smoothing, noise, a smooth non-affine warp and
    bleeding across mask borders. Larger = nearer."""
    disp = 1.0 / depth
    disp = _blur(disp, 0.8)
    # edge bleeding: replace disparity in a ring around each mask border by a strongly blurred version
    heavy = _blur(disp, 3.0)
    for m in masks:
        mb = _blur(m.astype(np.float64), 2.0)
        ring = (mb > 0.05) & (mb < 0.95)
        w = ring * 0.7
        disp = disp * (1 - w) + heavy * w
    # noise: spatially correlated (1.2 %) + pixel (0.4 %) + very smooth warp (1 %), multiplicative
    n1 = _blur(rng.standard_normal(disp.shape), 1.5)
    n1 /= max(n1.std(), 1e-9)
    n2 = _blur(rng.standard_normal(disp.shape), 25.0)
    n2 /= max(n2.std(), 1e-9)
    disp = disp * (1 + 0.012 * n1 + 0.004 * rng.standard_normal(disp.shape) + 0.010 * n2)
    a = rng.uniform(3.0, 9.0)
    b = rng.uniform(-2.0, 3.0)
    return (a * disp + b).astype(np.float32), (a, b)


def _look_at(pos, target, up):
    f = target - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    upc = np.cross(right, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, upc, -f, pos
    return T


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def intrinsics(w=IMG_W, h=IMG_H):
    return np.array([[FX, 0, w / 2.0], [0, FY, h / 2.0], [0, 0, 1.0]])


# ----------------------------------------------------------------------------- scene assembly
def _build_solids(kind):
    S, containers, container_keys, notes = [], {}, [], {}
    if kind == "rice_mound":
        S.append(make_mound("rice", "rice_white", 0, 0, 0.05, 0.035))
    elif kind == "fried_egg":
        S.append(make_cylinder("egg", "fried_egg", 0, 0, 0.045, 0.01))
    elif kind == "cutlet":
        S.append(make_box("cutlet", "fried_chicken_cutlet", 0, 0, 0.11, 0.07, 0.02, math.radians(25)))
    elif kind in ("bowl_flat", "bowl_dome"):
        bowl, food, cont = make_bowl_with_food("rice", "rice_white", "bowl", "flat" if kind == "bowl_flat" else "dome")
        S += [bowl, food]
        containers["rice"] = cont
        container_keys.append("bowl")
    elif kind == "clear_cup":
        r, h_cup, fill = 0.04, 0.10, 0.08
        cup = make_cylinder("cup", "bubble_milk_tea", 0, 0, r, h_cup, opaque=False)
        cup.vol_ml = math.pi * r * r * fill * 1e6   # liquid volume (cup drawn transparent)
        cup.contains = lambda P: ((P[:, 0] ** 2 + P[:, 1] ** 2 <= r * r) & (P[:, 2] >= 0) & (P[:, 2] <= fill))
        S.append(cup)
        containers["cup"] = {"shape": "cup", "item_key": "cup", "label": "container_cup", "clear": True,
                             "diameter_cm": 2 * r * 100, "height_cm": h_cup * 100, "fill_fraction": fill / h_cup}
        notes["cup"] = "clear cup: depth sees through to the table on purpose; GT = liquid volume"
    elif kind == "meal":
        S.append(make_mound("rice", "rice_white", -0.10, 0.02, 0.05, 0.035))
        S.append(make_cylinder("egg", "fried_egg", 0.10, 0.05, 0.045, 0.01))
        S.append(make_box("cutlet", "fried_chicken_cutlet", 0.0, -0.12, 0.11, 0.07, 0.02, math.radians(-20)))
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    return S, containers, container_keys, notes


def make_scene(kind: str, seed: int, n_frames: int = N_FRAMES, width: int = IMG_W, height: int = IMG_H):
    rng = np.random.default_rng(seed)
    solids, containers, container_keys, notes = _build_solids(kind)

    # table frame -> world: slight tilt (1-3 deg), table 0.5-0.8 m below the world origin
    tilt = math.radians(rng.uniform(1.0, 3.0))
    az = rng.uniform(0, 2 * math.pi)
    n = np.array([math.sin(tilt) * math.cos(az), math.cos(tilt), math.sin(tilt) * math.sin(az)])
    ex = np.cross([0.3, 0.0, 1.0], n)
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    E = np.stack([ex, ey, n], axis=1)       # local -> world rotation
    O = np.array([rng.uniform(-0.1, 0.1), -rng.uniform(0.5, 0.8), rng.uniform(-0.6, -0.3)])
    plane = Plane(n=n.copy(), offset=float(n @ O), inlier_rmse_m=0.0)

    K = intrinsics(width, height)
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    Dc = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones_like(uu, float)], -1).reshape(-1, 3)

    # camera arc, ~2 s
    r0 = 0.50 if kind == "meal" else rng.uniform(0.34, 0.44)
    phi0 = rng.uniform(0, 2 * math.pi)
    direction = rng.choice([-1, 1])
    arc = math.radians(rng.uniform(55, 80))
    el0 = math.radians(rng.uniform(52, 62))
    el_amp = math.radians(rng.uniform(3, 8))
    el_ph = rng.uniform(0, 2 * math.pi)
    target_l = np.array([0, 0, 0.02])
    ts = np.linspace(0, 2.0, n_frames) + rng.normal(0, 0.005, n_frames).cumsum() * 0
    s_all = ts / 2.0

    frames, depth_true_l, T_true_l = [], [], []
    for i, s in enumerate(s_all):
        phi = phi0 + direction * arc * (s - 0.5)
        el = np.clip(el0 + el_amp * math.sin(2 * math.pi * 0.7 * s + el_ph), math.radians(45), math.radians(70))
        r = r0 * (1 - 0.12 * s) * (1 + 0.02 * math.sin(7 * s))
        pos_l = target_l + r * np.array([math.cos(el) * math.cos(phi), math.cos(el) * math.sin(phi), math.sin(el)])
        T_l = _look_at(pos_l, target_l, np.array([0, 0, 1.0]))
        # to world
        T_w = np.eye(4)
        T_w[:3, :3] = E @ T_l[:3, :3]
        T_w[:3, 3] = O + E @ T_l[:3, 3]

        # ray cast in the local frame
        o_l = pos_l
        D_l = (T_l[:3, :3] @ Dc.T).T
        t_tab = _plane_z(o_l, D_l, 0.0, lambda p: np.ones(len(p), bool))
        t_all = [t_tab]
        owner = [0]
        for k, sd in enumerate(solids, start=1):
            t_all.append(sd.hit_t(o_l, D_l))
            owner.append(k)
        # opaque winners (clear solids don't write depth)
        t_op = np.stack([t if (j == 0 or solids[j - 1].opaque) else np.full_like(t, INF) for j, t in enumerate(t_all)])
        win = np.argmin(t_op, axis=0)
        depth = t_op[win, np.arange(len(win))]
        masks = {}
        for j, sd in enumerate(solids, start=1):
            if sd.opaque:
                m = win == j
            else:
                m = t_all[j] < depth + 1e-9   # silhouette of the transparent solid
            masks[sd.key] = m.reshape(height, width)
        # bowl silhouette includes the food inside it
        for ck in container_keys:
            for sd in solids:
                if sd.key not in container_keys:
                    masks[ck] = masks[ck] | masks[sd.key]
        depth = depth.reshape(height, width)

        rel, _ab = simulate_relative_depth(depth, list(masks.values()), rng)

        # feature points: mostly table, few on food, 3 mm noise, in (noise-free) world coords
        npts = int(rng.integers(60, 121))
        fg = np.zeros(depth.shape, bool)
        for sd in solids:
            if sd.opaque and sd.key not in container_keys:
                fg |= masks[sd.key]
        fg_idx, bg_idx = np.flatnonzero(fg.ravel()), np.flatnonzero(~fg.ravel())
        n_fg = min(len(fg_idx), int(round(npts * 0.12)))
        pick = np.concatenate([rng.choice(bg_idx, npts - n_fg, replace=False),
                               rng.choice(fg_idx, n_fg, replace=False) if n_fg else np.empty(0, int)])
        P_l = o_l + depth.ravel()[pick, None] * D_l[pick]
        fp_w = (O + P_l @ E.T) + rng.normal(0, 0.003, (len(pick), 3))

        # noisy pose given to the algorithm: ~2 mm / 0.2 deg
        T_n = T_w.copy()
        T_n[:3, 3] += rng.normal(0, 0.002, 3)
        ang = np.deg2rad(0.2) * rng.normal(0, 1, 3) / math.sqrt(3)
        T_n[:3, :3] = _rodrigues(ang) @ T_w[:3, :3]

        frames.append(Frame(t=float(ts[i]), T_wc=T_n, K=K.copy(), rel_depth=rel, depth_m=None,
                            masks=masks, labels={sd.key: sd.label for sd in solids},
                            feature_points_w=fp_w.astype(np.float32)))
        depth_true_l.append(depth.astype(np.float32))
        T_true_l.append(T_w)

    labels = {sd.key: sd.label for sd in solids}
    volumes = {sd.key: sd.vol_ml for sd in solids if sd.key not in container_keys}
    gt = {"kind": kind, "seed": seed, "volumes_ml": volumes, "plane": plane, "labels": labels,
          "containers": containers, "container_keys": container_keys, "solids": {s.key: s for s in solids},
          "depth_true": depth_true_l, "T_wc_true": T_true_l, "notes": notes,
          "table_to_world": (E, O)}
    return frames, gt


def voxel_volume_ml(solid: Solid, cell=0.001) -> float:
    """Brute-force volume of a solid by counting voxel centres (used to validate the analytic volumes)."""
    lo, hi = solid.bbox
    axes = [np.arange(lo[i] + cell / 2, hi[i], cell) for i in range(3)]
    total = 0
    X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
    for z in axes[2]:
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
        total += int(solid.contains(P).sum())
    return total * cell ** 3 * 1e6


if __name__ == "__main__":
    for k in KINDS:
        fr, gt = make_scene(k, 0)
        print(k, {a: round(b, 1) for a, b in gt["volumes_ml"].items()}, len(fr), "frames")
