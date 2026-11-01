"""Geometry: pose of a plate (circle of known diameter) from its image ellipse, plane back-projection, camera model.
Camera coords: x right, y down, z forward (OpenCV). Plane: n . P = dd with n pointing TOWARD the camera? We store (n, c) with c = circle centre in camera
coords and n = unit normal pointing from the table towards the camera (table up). Plane points satisfy n . (P - c) = 0.
"""
import numpy as np


def K_of(W, H, hfov_deg=69.0, fscale=1.0):
    f = fscale * (W / 2) / np.tan(np.radians(hfov_deg / 2))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def ellipse_conic(e):
    """e: dict cx, cy, major, minor (full axis lengths px), angle deg (of the major axis, cv2 convention) -> 3x3 symmetric Q, x^T Q x = 0."""
    t = np.radians(e["angle"]); a, b = e["major"] / 2, e["minor"] / 2
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    # cv2.fitEllipse: angle is rotation of the first axis (width) from the x axis; callers pass major with matching angle
    A = R @ np.diag([1 / a ** 2, 1 / b ** 2]) @ R.T
    c = np.array([e["cx"], e["cy"]])
    Q = np.zeros((3, 3)); Q[:2, :2] = A; Q[:2, 2] = -A @ c; Q[2, :2] = -c @ A; Q[2, 2] = c @ A @ c - 1
    return Q


def pose_from_ellipse(e, K, diam_cm):
    """Closed-form circle pose (Kanatani). Returns the two candidate solutions [(n, c)], c in cm. n points to the camera."""
    Q = ellipse_conic(e)
    Qc = K.T @ Q @ K
    Qc = (Qc + Qc.T) / 2
    w, V = np.linalg.eigh(Qc)
    if np.sum(w > 0) == 1: w = -w  # want two positive, one negative
    if np.sum(w > 0) != 2: return []
    order = np.argsort(-w)  # descending: l1 >= l2 > 0 > l3
    l = w[order]; E = V[:, order]
    if not (l[0] >= l[1] > 0 > l[2]): return []
    l1, l2, l3 = l
    sols = []
    for sgn in (+1, -1):
        n_e = np.array([np.sqrt((l1 - l2) / (l1 - l3)), 0.0, sgn * np.sqrt((l2 - l3) / (l1 - l3))])
        n = E @ n_e
        # circle centre direction & scale: intersect plane n.P = d (d=1) with the cone sum(l_i P_i^2)=0 (eigenframe), P = d*n + s*u + t*v
        u = E @ np.array([0, 1.0, 0]); v = np.cross(n, u)
        # quadratic form in P: P^T Qd P with Qd = E diag(l) E^T (sign as flipped)
        Qd = E @ np.diag(l) @ E.T
        # P = n + s u + t v: P^T Qd P = n.Qd.n + 2 s u.Qd.n + 2 t v.Qd.n + s^2 uQu + 2st uQv + t^2 vQv = 0  -> circle (uQu = vQv, uQv = 0)
        quu, qvv, quv = u @ Qd @ u, v @ Qd @ v, u @ Qd @ v
        gu, gv, g0 = u @ Qd @ n, v @ Qd @ n, n @ Qd @ n
        M = np.array([[quu, quv], [quv, qvv]]); s0 = -np.linalg.solve(M, [gu, gv])
        rho2 = (s0 @ M @ s0 - g0) / quu if abs(quu) > 1e-12 else None
        if rho2 is None or rho2 <= 0: continue
        rho = np.sqrt(rho2)  # radius when plane is n.P = 1
        d = (diam_cm / 2) / rho
        c = d * (n + s0[0] * u + s0[1] * v)
        if c[2] < 0: c = -c; n = -n  # in front of the camera
        nn = n if n @ (-c) > 0 else -n  # towards the camera
        sols.append((nn / np.linalg.norm(nn), c))
    return sols


def best_pose(e, K, diam_cm):
    """Resolve the 2-fold ambiguity: the camera is held roughly level (small roll), so the table normal has a small x component (and points up in the image:
    n_y < 0). Returns (n, c, tilt_deg) with tilt = angle between n and the viewing ray to the plate (0 = looking straight down)."""
    s = pose_from_ellipse(e, K, diam_cm)
    if not s: return None
    s = [x for x in s if x[0][1] < 0.05] or s  # normal pointing up in the image (camera above the table)
    n, c = min(s, key=lambda x: abs(x[0][0]))
    ray = c / np.linalg.norm(c)
    return n, c, float(np.degrees(np.arccos(np.clip(n @ -ray, -1, 1))))


def plane_basis(n):
    a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    return a, b


def plane_depth_map(n, c, K, shape):
    """depth Z (cm along the optical axis) of the plane at every pixel; inf/neg where the ray misses (parallel / behind)."""
    h, w = shape
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    r = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)  # z = 1
    den = r @ n
    Z = (n @ c) / np.where(np.abs(den) < 1e-6, 1e-6, den)
    return Z, r
