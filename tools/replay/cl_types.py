"""Shared types for the sweep-replay / volume pipeline (binding contract between align.py, volume.py, synth.py).

Conventions follow ARKit so the Swift port is a transliteration:
  * World and camera frames are right-handed, metres. Camera frame: +X right, +Y up, camera looks along -Z.
  * Image pixel coords (u, v): u right, v DOWN, origin top-left, pixel centres at integer + 0.5 is NOT assumed
    (use u, v = column, row indices directly, as ARKit's intrinsics do).
  * K is the 3x3 intrinsics AT DEPTH-MAP RESOLUTION (scale ARKit's full-res intrinsics by depth_w/img_w, depth_h/img_h).
  * Unprojection of pixel (u, v) with metric depth d (distance along the viewing axis, > 0):
        x_c = (u - cx) * d / fx
        y_c = -(v - cy) * d / fy
        z_c = -d
    then  p_world = T_wc @ [x_c, y_c, z_c, 1].
  * T_wc is camera-to-world (ARKit `frame.camera.transform`).
  * Plane: unit normal n pointing UP (away from the table, towards the camera side) and offset so that
        signed height h(p) = n . p - offset   (> 0 above the table).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    t: float                                  # seconds since sweep start
    T_wc: np.ndarray                          # (4,4) float64 camera-to-world
    K: np.ndarray                             # (3,3) float64 intrinsics at depth resolution
    rel_depth: np.ndarray | None = None       # (H,W) float32 RELATIVE depth model output (e.g. Depth Anything: larger = nearer, affine-invariant disparity)
    depth_m: np.ndarray | None = None         # (H,W) float32 METRIC depth (filled by align.py, or LiDAR in future)
    conf: np.ndarray | None = None            # (H,W) float32 in [0,1], optional per-pixel confidence
    masks: dict[str, np.ndarray] = field(default_factory=dict)   # item_key -> (H,W) bool at depth resolution
    labels: dict[str, str] = field(default_factory=dict)         # item_key -> class id (ml/labels.yaml)
    feature_points_w: np.ndarray | None = None  # (N,3) ARKit rawFeaturePoints in WORLD coords visible this frame


@dataclass
class Plane:
    n: np.ndarray        # (3,) unit normal, pointing up
    offset: float        # n . p = offset on the plane
    inlier_rmse_m: float = 0.0

    def height(self, p_w: np.ndarray) -> np.ndarray:
        """Signed height above the plane for (...,3) world points."""
        return p_w @ self.n - self.offset


@dataclass
class Triple:
    min: float
    typ: float
    max: float

    def as_dict(self) -> dict:
        return {"min": self.min, "typ": self.typ, "max": self.max}


@dataclass
class VolumeResult:
    item_key: str
    volume_ml: Triple | None       # None => failed; caller falls back to default portion
    method: str                    # 'heightmap' | 'bowl_cavity' | 'cup_prior' | 'failed'
    n_cells: int = 0               # grid cells with enough observations
    coverage: float = 0.0          # fraction of the item's footprint cells observed >= min_obs times
    notes: list[str] = field(default_factory=list)
