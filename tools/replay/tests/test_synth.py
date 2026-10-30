import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dump_io  # noqa: E402
import synth  # noqa: E402


class TestSynth(unittest.TestCase):
    def test_analytic_volume_matches_voxels(self):
        for kind in synth.KINDS:
            _, gt = synth.make_scene(kind, 1)
            for key, s in gt["solids"].items():
                if key in gt["container_keys"]:
                    continue
                v = synth.voxel_volume_ml(s)
                self.assertLess(abs(v / s.vol_ml - 1), 0.02, f"{kind}/{key}: analytic {s.vol_ml:.2f} voxel {v:.2f}")
                self.assertAlmostEqual(gt["volumes_ml"][key], s.vol_ml)

    def test_raycast_consistent_with_solids(self):
        """Hit point nudged 0.5 mm along the ray lies inside the item for (nearly) all its mask pixels."""
        for kind in ("rice_mound", "cutlet", "bowl_flat", "bowl_dome", "meal"):
            frames, gt = synth.make_scene(kind, 2)
            E, O = gt["table_to_world"]
            f, dt, Tw = frames[5], gt["depth_true"][5], gt["T_wc_true"][5]
            K = f.K
            for key, m in f.masks.items():
                if key in gt["container_keys"]:
                    continue
                vv, uu = np.nonzero(m)
                d = dt[vv, uu].astype(float)
                dc = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones(len(uu))], 1)
                pw = Tw[:3, 3] + (dc * d[:, None]) @ Tw[:3, :3].T
                dirw = (dc @ Tw[:3, :3].T)
                dirw /= np.linalg.norm(dirw, axis=1, keepdims=True)
                pl = (pw + 0.0005 * dirw - O) @ E
                frac = gt["solids"][key].contains(pl).mean()
                self.assertGreater(frac, 0.97, f"{kind}/{key}: {frac:.3f}")

    def test_plane_and_features(self):
        frames, gt = synth.make_scene("meal", 4)
        pl = gt["plane"]
        self.assertAlmostEqual(float(np.linalg.norm(pl.n)), 1.0)
        self.assertEqual(len(frames), 12)
        for f in frames:
            self.assertEqual(f.rel_depth.shape, (192, 256))
            self.assertGreater(f.rel_depth.min(), 0)
            self.assertTrue(60 <= len(f.feature_points_w) <= 120)
            self.assertGreater(float(np.mean(pl.height(f.T_wc[:3, 3]))), 0.25)
        h = np.concatenate([pl.height(f.feature_points_w) for f in frames])
        self.assertLess(np.median(np.abs(h)), 0.005)   # mostly table points, ~3 mm noise

    def test_dump_round_trip(self):
        frames, gt = synth.make_scene("bowl_flat", 5)
        with tempfile.TemporaryDirectory() as d:
            dump_io.save_dump(frames, d, meta={"device_model": "iPhone14,5", "ios_version": "17.4",
                                               "image_size": [1920, 1440], "depth_model": "DepthAnythingV2SmallF16",
                                               "plane": {"n": list(gt["plane"].n), "offset": gt["plane"].offset, "source": "gt"},
                                               "containers": gt["containers"]})
            back, man = dump_io.load_dump_with_meta(d)
            self.assertEqual(man["version"], 1)
            self.assertEqual(man["containers"]["rice"]["shape"], "bowl")
            self.assertAlmostEqual(man["plane_obj"].offset, gt["plane"].offset)
            self.assertEqual(len(back), len(frames))
            for a, b in zip(frames, back):
                np.testing.assert_allclose(a.T_wc, b.T_wc, atol=1e-12)
                np.testing.assert_allclose(a.K, b.K)
                np.testing.assert_allclose(a.rel_depth, b.rel_depth, rtol=1e-3)
                self.assertEqual(a.t, b.t)
                self.assertEqual(set(a.masks), set(b.masks))
                for k in a.masks:
                    np.testing.assert_array_equal(a.masks[k], b.masks[k])
                self.assertEqual(a.labels, b.labels)
                np.testing.assert_array_equal(a.feature_points_w, b.feature_points_w)
            # zip form
            import shutil
            z = shutil.make_archive(os.path.join(d, "..", "ziptest_dump"), "zip", d)
            try:
                self.assertEqual(len(dump_io.load_dump(z)), len(frames))
            finally:
                os.remove(z)


if __name__ == "__main__":
    unittest.main()
