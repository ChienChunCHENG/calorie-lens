import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import align  # noqa: E402
import synth  # noqa: E402


def _errors(frames, gt):
    allerr, fooderr = [], []
    for f, dt in zip(frames, gt["depth_true"]):
        if f.depth_m is None:
            continue
        e = np.abs(f.depth_m - dt)
        ok = np.isfinite(e)
        allerr.append(e[ok])
        fg = np.zeros(dt.shape, bool)
        for k, m in f.masks.items():
            if k not in gt["container_keys"]:
                fg |= m
        fooderr.append(e[ok & fg])
    return np.concatenate(allerr), np.concatenate(fooderr)


class TestAlign(unittest.TestCase):
    def test_metric_accuracy_known_plane(self):
        for kind in synth.KINDS:
            frames, gt = synth.make_scene(kind, 11)
            plane, diags = align.align_frames(frames, plane_hint=gt["plane"])
            self.assertEqual(sum(d["ok"] for d in diags), len(frames), kind)
            e, ef = _errors(frames, gt)
            self.assertLess(np.median(e), 0.01, f"{kind}: median {np.median(e) * 1000:.1f} mm")
            self.assertLess(np.median(ef), 0.01, f"{kind} food: median {np.median(ef) * 1000:.1f} mm")

    def test_metric_accuracy_ransac_plane(self):
        try:
            import plane  # noqa: F401
        except ImportError:
            self.skipTest("plane.py missing")
        frames, gt = synth.make_scene("meal", 12)
        pl, diags = align.align_frames(frames)
        self.assertLess(abs(float(pl.height(gt["plane"].n * gt["plane"].offset))), 0.003)
        e, ef = _errors(frames, gt)
        self.assertLess(np.median(e), 0.01)
        self.assertLess(np.median(ef), 0.01)

    def test_feature_points_only_when_table_hidden(self):
        """Masks covering most of the image: still fits using feature points + remaining table."""
        frames, gt = synth.make_scene("rice_mound", 13)
        _, diags = align.align_frames(frames, plane_hint=gt["plane"])
        self.assertTrue(all(d["n_fp"] > 20 for d in diags))

    def test_rejects_garbage(self):
        frames, gt = synth.make_scene("rice_mound", 14)
        rng = np.random.default_rng(0)
        frames[3].rel_depth = rng.uniform(1, 10, frames[3].rel_depth.shape).astype(np.float32)
        frames[4].rel_depth = np.full_like(frames[4].rel_depth, 7.0)
        _, diags = align.align_frames(frames, plane_hint=gt["plane"])
        self.assertFalse(diags[3]["ok"])
        self.assertFalse(diags[4]["ok"])
        self.assertIsNone(frames[3].depth_m)
        self.assertEqual(sum(d["ok"] for d in diags), len(frames) - 2)

    def test_outlier_robustness(self):
        """Gross feature-point outliers do not break the fit."""
        frames, gt = synth.make_scene("rice_mound", 15)
        for f in frames:
            f.feature_points_w = f.feature_points_w.copy()
            f.feature_points_w[:10] += np.array([0.3, 0.2, -0.3], np.float32)
        align.align_frames(frames, plane_hint=gt["plane"])
        e, _ = _errors(frames, gt)
        self.assertLess(np.median(e), 0.01)


if __name__ == "__main__":
    unittest.main()
