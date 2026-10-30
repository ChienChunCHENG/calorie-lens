# Sweep dump format, version 1

Written by the Swift app (Share -> zip), read by `dump_io.load_dump(path)` (directory or `.zip`).
Nothing here needs more than `JSONEncoder`, `Data.write`, `CGImage` PNG/JPEG encoding and `ZipArchive`/`NSFileCoordinator`
zipping. No npz, no protobuf. All binary files are little-endian, row-major, no header.

```
sweep_YYYYMMDD_HHMMSS/
  manifest.json
  rel_00.f16 ...          raw float16 relative depth, H*W values
  mask_00_<key>.png ...   8-bit grayscale PNG, 0 or 255, depth resolution
  fp_00.f32 ...           raw float32 N*3 world points (x,y,z)
  rgb_00.jpg ...          optional keyframe image (full resolution, for human debugging only)
  depth_00.f32 ...        optional metric depth (float32 H*W), debugging / future LiDAR; ignored by replay
```
A zip may contain the files at its root or inside one top-level folder.

## Conventions (same as `cl_types.py`)
* World/camera: right-handed, metres; camera +X right, +Y up, looks along -Z (ARKit `frame.camera.transform`).
* `T_wc`: camera-to-world 4x4 **row-major**: `T_wc[r][c]` is at index `4*r + c`. simd matrices are column-major, so in
  Swift write `for r in 0..<4 { for c in 0..<4 { out.append(Double(m[c][r])) } }` (`m[c]` = column c).
* `K`: 3x3 row-major (9 values) **at depth-map resolution**: `fx, 0, cx, 0, fy, cy, 0, 0, 1`, with pixel coordinates
  u = column, v = row (top-left origin, v down). Scale ARKit's full-res intrinsics by depth_w/img_w and depth_h/img_h.
* Plane: `n` unit normal pointing up, `offset` with `n . p = offset` on the plane (`height = n.p - offset`).

## manifest.json
```json
{
  "version": 1,
  "device_model": "iPhone14,5",
  "ios_version": "17.4",
  "image_size": [1920, 1440],              // full-res RGB (w, h); informational
  "depth_model": "DepthAnythingV2SmallF16",
  "depth_semantics": "disparity_larger_is_nearer",
  "plane": {"n": [0.01, 0.99, 0.02], "offset": -0.71, "source": "arkit"},   // optional, null if none
  "containers": {                           // optional, keyed by FOOD item key
    "rice": {"shape": "bowl", "item_key": "bowl", "rim_diameter_cm": 12.0, "depth_cm": 4.5}
  },
  "frames": [
    {
      "index": 0,
      "t": 0.0,                             // seconds since sweep start
      "T_wc": [16 numbers],
      "K": [9 numbers],
      "rel":  {"file": "rel_00.f16", "shape": [392, 518]},          // [H, W]
      "masks": {"rice": {"file": "mask_00_rice.png", "label": "rice_white"},
                "bowl": {"file": "mask_00_bowl.png", "label": "container_bowl"}},
      "feature_points": {"file": "fp_00.f32", "count": 87},
      "rgb": "rgb_00.jpg"                   // or null
    }
  ]
}
```
* Item keys match `[A-Za-z0-9_-]+` (they go into file names). Labels are class ids from `ml/labels.yaml`.
  Container masks (`container_*` labels) may overlap the food masks. All masks are already un-letterboxed and resized
  to the depth-map resolution (same H, W as `rel`).
* Per-frame `K`/shape may differ between frames in principle; replay does not assume they are equal.
* `fp_NN.f32` holds only the `rawFeaturePoints` visible in that keyframe (world coordinates); N may be 0 (empty file).

## Why float16 raw for relative depth (not PNG-16)
The model output is unnormalised (arbitrary scale/offset per frame), so PNG-16 would need a per-frame min/max
quantisation (lossy, extra metadata, extra code). Core ML hands Swift a Float16 `MLMultiArray`; writing its bytes
straight to disk is one `Data(bytes:count:)` call and loses nothing. float16 keeps ~0.05 % relative precision, i.e.
under 0.5 mm at 40 cm, far below the model noise. Size: 518x392x2 = 406 kB per frame. If the model output is Float32,
convert to Float16 before writing (values up to 65504 are fine).

## Replay
`py run_replay.py --dump sweep.zip [--save-debug out]` or `py run_replay.py --synth bowl_dome --seed 3`.
