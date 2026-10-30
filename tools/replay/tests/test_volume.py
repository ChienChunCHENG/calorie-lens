"""Self-contained tests for plane.py / volume.py (own tiny analytic ray-caster; no synth.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cl_types import Frame, Plane  # noqa: E402
from plane import fit_plane_ransac  # noqa: E402
from volume import compute_item_volume, _fill  # noqa: E402

W, H = 200, 150
K = np.array([[220.0, 0, 100.0], [0, 220.0, 75.0], [0, 0, 1.0]])
TABLE = Plane(n=np.array([0.0, 1.0, 0.0]), offset=0.0)


def look_at(eye, target):
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def rays(T):
    vv, uu = np.mgrid[0:H, 0:W]
    d_c = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones_like(uu, float)], -1)
    return T[:3, 3], d_c @ T[:3, :3].T      # origin, (H,W,3) dirs; depth = t because d_c.z = -1


def cast_box(T, lo, hi):
    o, d = rays(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tn = np.minimum(t1, t2).max(-1)
    tf = np.maximum(t1, t2).min(-1)
    hit = (tn < tf) & (tn > 0)
    return np.where(hit, tn, 0.0), hit


def cast_disc(T, centre_xz, radius, h):
    o, d = rays(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (h - o[1]) / d[..., 1]
    p = o + d * t[..., None]
    hit = (t > 0) & (np.hypot(p[..., 0] - centre_xz[0], p[..., 2] - centre_xz[1]) < radius)
    return np.where(hit, t, 0.0), hit


def poses(centre, n=6, dist=0.40, elev_deg=(50, 62)):
    out = []
    for i in range(n):
        az = math.radians(-40 + 80 * i / (n - 1))
        el = math.radians(elev_deg[i % 2])
        eye = np.array(centre) + dist * np.array([math.cos(el) * math.sin(az), math.sin(el),
                                                   math.cos(el) * math.cos(az)])
        out.append(look_at(eye, np.array(centre)))
    return out


def make_frames(caster, centre, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    fr = []
    for i, T in enumerate(poses(centre)):
        depth, mask = caster(T)
        depth = depth + noise * rng.standard_normal(depth.shape)
        fr.append(Frame(t=i * 0.2, T_wc=T, K=K, depth_m=depth.astype(np.float32),
                        masks={"food": mask}, labels={"food": "x"}))
    return fr


class TestVolume(unittest.TestCase):
    def setUp(self):
        self.box_lo = np.array([-0.04, 0.0, -0.03])
        self.box_hi = np.array([0.04, 0.04, 0.03])
        self.box_true_ml = 8 * 6 * 4.0       # 192 mL
        self.caster = lambda T: cast_box(T, self.box_lo, self.box_hi)

    def test_box_clean(self):
        fr = make_frames(self.caster, (0, 0.02, 0))
        r = compute_item_volume(fr, "food", TABLE, "rice_white")
        self.assertEqual(r.method, "heightmap")
        err = r.volume_ml.typ / self.box_true_ml - 1
        print(f"\n[box clean] typ={r.volume_ml.typ:.1f} true={self.box_true_ml} err={err*100:+.2f}% "
              f"tri=({r.volume_ml.min:.0f},{r.volume_ml.max:.0f}) cov={r.coverage:.2f} cells={r.n_cells}")
        self.assertLess(abs(err), 0.05)

    def test_box_noisy_interval(self):
        fr = make_frames(self.caster, (0, 0.02, 0), noise=0.005)
        r = compute_item_volume(fr, "food", TABLE, "rice_white")
        v = r.volume_ml
        print(f"\n[box 5mm] typ={v.typ:.1f} true={self.box_true_ml} err={(v.typ/self.box_true_ml-1)*100:+.2f}% "
              f"interval=({v.min:.0f},{v.max:.0f}) notes={r.notes}")
        self.assertLessEqual(v.min, self.box_true_ml)
        self.assertGreaterEqual(v.max, self.box_true_ml)

    def test_failed_without_depth(self):
        fr = make_frames(self.caster, (0, 0.02, 0))
        for f in fr:
            f.depth_m = None
        r = compute_item_volume(fr, "food", TABLE, "rice_white")
        self.assertEqual(r.method, "failed")
        self.assertIsNone(r.volume_ml)

    def test_bowl_cavity(self):
        a, Hd, base = 0.0675, 0.065, 0.005
        Rs = (a * a + Hd * Hd) / (2 * Hd)
        fill = 0.04                                  # food depth in the cavity
        hs = base + fill                             # surface height above table
        rs = math.sqrt(2 * Rs * fill - fill * fill)
        true_ml = math.pi * fill ** 2 * (3 * Rs - fill) / 3 * 1e6
        cast = lambda T: cast_disc(T, (0.0, 0.0), rs, hs)
        fr = make_frames(cast, (0, hs, 0), noise=0.003)
        cont = {"shape": "bowl",
                "rim_diameter_cm": {"min": 12.5, "typ": 13.5, "max": 14.5},
                "depth_cm": {"min": 6.0, "typ": 6.5, "max": 7.5}}
        r = compute_item_volume(fr, "food", TABLE, "rice_white", container=cont)
        naive = compute_item_volume(fr, "food", TABLE, "rice_white", container=None)
        self.assertEqual(r.method, "bowl_cavity")
        err = r.volume_ml.typ / true_ml - 1
        nerr = naive.volume_ml.typ / true_ml - 1
        print(f"\n[bowl] true={true_ml:.1f} cavity={r.volume_ml.typ:.1f} ({err*100:+.1f}%) "
              f"interval=({r.volume_ml.min:.0f},{r.volume_ml.max:.0f}) | naive={naive.volume_ml.typ:.1f} "
              f"({nerr*100:+.1f}%) notes={r.notes}")
        self.assertLess(abs(err), 0.10)
        self.assertGreater(nerr, 0.5)                # naive demonstrably overcounts
        self.assertGreater(naive.volume_ml.typ, 1.5 * r.volume_ml.typ)

    def test_cup_prior(self):
        fr = make_frames(self.caster, (0, 0.02, 0))
        r = compute_item_volume(fr, "food", TABLE, "bubble_milk_tea")
        self.assertEqual(r.method, "cup_prior")
        self.assertIsNone(r.volume_ml)
        self.assertTrue(any("clear plastic" in n for n in r.notes))

    def test_low_coverage_widens(self):
        fr = make_frames(self.caster, (0, 0.02, 0))
        full = compute_item_volume(fr, "food", TABLE, "rice_white")
        # frames 2..5 only see the right part of the item -> most footprint cells have < min_obs observations
        for f in fr[2:]:
            m = f.masks["food"].copy()
            m[:, :W // 2 + 20] = False
            f.masks["food"] = m
        r = compute_item_volume(fr, "food", TABLE, "rice_white")
        print(f"\n[coverage] full cov={full.coverage:.2f} width={full.volume_ml.max/full.volume_ml.min:.2f} | "
              f"partial cov={r.coverage:.2f} width={r.volume_ml.max/r.volume_ml.min:.2f}")
        self.assertLess(r.coverage, 0.5)
        self.assertTrue(any("coverage" in n for n in r.notes))
        self.assertGreater(r.volume_ml.max / r.volume_ml.min, 1.2 * full.volume_ml.max / full.volume_ml.min)

    def test_ransac_outliers(self):
        rng = np.random.default_rng(3)
        n = np.array([0.05, 1.0, -0.03])
        n /= np.linalg.norm(n)
        off = 0.012
        a = np.cross(n, [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        N = 3000
        uv = rng.uniform(-0.4, 0.4, (N, 2))
        P = uv[:, :1] * a + uv[:, 1:] * b + (n * off) + 0.001 * rng.standard_normal((N, 1)) * n
        k = int(0.3 * N)
        P[:k] = rng.uniform(-0.4, 0.4, (k, 3)) + n * 0.1       # 30 % outliers
        cams = np.array([[0.0, 0.5, 0.3], [0.1, 0.45, 0.35]])
        pl = fit_plane_ransac(P, cam_centers_w=cams)
        ang = math.degrees(math.acos(min(1.0, abs(float(pl.n @ n)))))
        print(f"\n[ransac] normal err={ang:.3f} deg offset err={abs(pl.offset-off)*1000:.2f} mm rmse={pl.inlier_rmse_m*1000:.2f} mm")
        self.assertLess(ang, 0.5)
        self.assertLess(abs(pl.offset - off), 0.002)
        self.assertGreater(float(pl.n @ n), 0.99)               # oriented toward cameras
        # flip check: cameras below plane -> normal flips
        pl2 = fit_plane_ransac(P, cam_centers_w=-cams + 2 * off * n)
        self.assertLess(float(pl2.n @ n), 0)


class TestFillAndBench(unittest.TestCase):
    def test_fill_only_inside_footprint(self):
        H = np.zeros((9, 9))
        W = np.zeros((9, 9))
        H[2:7, 2:7], W[2:7, 2:7] = 0.02, 1.0
        W[4, 4] = 0.0                               # hole
        foot = np.zeros((9, 9), bool)
        foot[1:8, 1:8] = True                       # footprint = observed block + 1 cell ring
        Hf, Wf = _fill(H, W, foot, iters=5)
        self.assertAlmostEqual(Hf[4, 4], 0.02)
        self.assertTrue(np.all(Wf[1:8, 1:8] > 0))   # ring and hole filled
        self.assertAlmostEqual(Hf[1, 3], 0.02)
        self.assertTrue(np.all(Wf[0, :] == 0) and np.all(Wf[:, 8] == 0))   # nothing outside footprint

    def test_bench_subset_meal(self):
        """Fast slice of bench_volume: 3 seeds of the synthetic meal through align -> plane -> volume.
        Pooled over the 3 items the mean signed error must be within 7 %; with the exact synth depth the
        algorithm alone must be within 5 % per item (separates volume.py from alignment error)."""
        import bench_volume as B
        rows = B.bench(["meal"], [1, 2, 3])
        errs = [r["err"] for r in rows if r.get("ok")]
        self.assertEqual(len(errs), 9)
        print(f"\n[bench meal x3 aligned] pooled bias {np.mean(errs):+.1f}%")
        self.assertLessEqual(abs(np.mean(errs)), 7.0)
        rows = B.bench(["meal"], [1, 2, 3], true_depth=True)
        for item in {r["item"] for r in rows}:
            e = [r["err"] for r in rows if r["item"] == item and r.get("ok")]
            print(f"[bench meal x3 true depth] {item}: {np.mean(e):+.1f}%")
            self.assertLessEqual(abs(np.mean(e)), 5.0, item)


if __name__ == "__main__":
    unittest.main()
