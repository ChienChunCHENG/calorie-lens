# Volume algorithm: Swift-portable spec

Reference implementation: `plane.py`, `volume.py` (numpy only). Conventions: `cl_types.py` (ARKit camera, metres, plane normal up, `h(p) = n.p - offset`).

## 1. Table plane (`fit_plane_ransac`)
1. Input: world points (ARKit plane-raycast samples / feature points / background depth), optional camera centres.
2. Score on <= 5000 random points. 500 iterations: sample 3 points, normal = normalise(cross(b-a, c-a)), skip if |cross| < 1e-9. Inlier = |n.p - d| < thresh (5 mm). Best = most inliers, ties by smaller clipped residual.
3. Refine on ALL points: 3 rounds of (inliers of current plane -> centroid + SVD, normal = right singular vector of the smallest singular value). `inlier_rmse_m` = RMS residual of final inliers.
4. Orientation: flip so mean camera height > 0 when `cam_centers_w` is given, else so that n.y >= 0.

## 2. Heightmap with median fusion (`compute_item_volume`, method `heightmap`)
Per frame with `depth_m` and a mask for `item_key`:
1. Valid pixels = mask & depth finite & > 0 (& conf >= 0.3 if conf given). The *eroded* mask (4-neighbour, `erode_px` = 1) selects the pixels used for HEIGHTS; the un-eroded mask is used for the FOOTPRINT only.
2. Unproject (cl_types formula), transform to world, `h = plane.height(p)`. Discard h outside (-0.05, 0.5) or > 0.5 m in-plane from the median (garbage depth). Clamp h to [0, 0.25].
3. Plane coords: `pu = p.a`, `pv = p.b`, with (a, b) an orthonormal basis of the plane (`a = normalise(cross(n, e))`, e = world axis least aligned with n; `b = cross(n, a)`). Cell = floor((pu-u0)/cell_m), floor((pv-v0)/cell_m); `cell_m = 3 mm`.
4. Per-frame reduction: one height per (frame, cell) = median of the eroded-mask samples in the cell (decision D1).
5. Fuse: per-cell median across frames (mean of the two middle values for even counts). "Observed" cells have >= `min_obs` (3) frames.
6. **Footprint weight `fpw` per cell in [0,1]** (the footprint is NOT taken from the eroded height samples, which lose a ring and are laterally noisy):
   a. Presence: `pf = (# frames whose UN-eroded mask samples touch the cell) / F`. `fpw = clip((pf - FP_FLOOR) / (FP_SAT - FP_FLOOR), 0, 1)` with FP_FLOOR = 0, FP_SAT = 0.5 (a cell present in half the frames counts fully; rarer cells count fractionally, which suppresses lateral depth-noise dilation).
   b. Table-level projection vote (hull test), only for cells with `fpw > 0`: take SUB_N x SUB_N = 3 x 3 sample points inside the cell on the table plane (h = 0), project each into every frame (`pc = inv(T_wc) p`, `d = -pc.z > 0`, `u = fx*pc.x/d + cx`, `v = cy - fy*pc.y/d`, nearest pixel). A sample is "in" if it lies in the item's un-eroded mask in >= `HULL_VOTE` (0.8) of the frames in which it projects inside the image. A table point under the item is covered by the mask (the item occludes it); a table point beside the item is not (except in the item's shadow for a few views), so depth-noise smear cells beyond the true edge are rejected. `fpw *= fraction of the 9 samples that are "in"`.
   c. `footprint = fpw > 0`.
7. Cell weight: observed cell inside the footprint -> `fpw`; observed cell outside the footprint (e.g. the mask was truncated in most frames) -> `min(1, nframes_obs / (0.6 F))` as before.
8. **Inpaint** (`_fill`), `FILL_ITERS` = 12 passes: each pass, every footprint cell with no height and >= 1 known 4-neighbour takes the mean height of its known 4-neighbours (simultaneous update) and weight `fpw`. This closes interior holes AND regrows the ring lost to erosion / the < min_obs edge cells, strictly inside the footprint. Cells outside the footprint are never filled.
9. `V = sum(w * h) * cell_m^2`, reported in mL.
10. `coverage = observed_cells(>= min_obs) / cells touched by the eroded masks in any frame` (unchanged definition; it measures how much of the item was actually seen, not the inpainted share). `n_cells` = observed cells.
11. Failure: no depth/mask frames, or < 20 observed cells -> `volume_ml=None`, method `failed`.
12. Bootstrap resamples frames but keeps the footprint and `fpw` fixed (they come from the full data).

Diagnosis behind these steps (`bench_volume.py`, synthetic, 6 seeds, bias %, columns mound / egg / cutlet / meal-rice / meal-egg / meal-cutlet; exact synth depth = "true", aligned depth = "aligned"): old code true -9.7 -0.8 -0.3 -6.0 -4.3 -2.8; removing the 1 px erosion alone +5 / +4 on mound/meal-rice, the weight saturation (0.6 F) about +3 on mound; per-frame max instead of median pushes noisy cutlets +11 %. New code, true depth: -2.2 +1.1 +2.1 -0.4 +0.2 +1.2. The residual with aligned depth (meal rice/egg about -8/-10 %) is alignment (per-frame disparity fit makes object depth about 0.6 mm too far, 6 % of a 1 cm egg), not volume.py.

## 3. Interval (log space, PLAN section 5)
- sigma_boot: 50 bootstrap resamples of frames (with replacement), full pipeline re-run on each; `sigma_boot = ln(p95/p5)/3.29`, capped at 0.4. Fewer than 2 frames: 0.25 with a note.
- sigma_scale: depth scale error +-6 % on length. Volume scales as s^3, so the 90 % interval is [0.94^3, 1.06^3] = [0.831, 1.191] (about -17 % / +19 %); `sigma_scale = ln(1.191/0.831)/3.29 = 0.110`.
- sigma_cont (bowl only): spread of food volume over the four corner combinations of the container's rim-diameter and depth ranges.
- `sigma = sqrt(sigma_boot^2 + sigma_scale^2 + sigma_cont^2)`; min/max = typ * exp(-/+ 1.645 sigma). `typ` = full-data volume.
- coverage < 0.5: `sigma^2 += (0.5 ln(0.5/max(cov, 0.2)))^2 + 0.1^2` and the note "low footprint coverage ... volume likely underestimated". (typ is not corrected upward.)

## 4. Bowl cavity (method `bowl_cavity`)
Used when `container['shape'] == 'bowl'`, or when no container shape is given and a `container_bowl` mask is present in the frames (then missing dims come from the container-mask extent, else the bowl_medium default, with a note).
- Spherical cap with rim radius `a = rim_diameter/2` and interior depth `Hd`: sphere radius `Rs = (a^2 + Hd^2) / (2 Hd)`.
- Cavity floor height above the table at horizontal distance r from the bowl centre: `floor(r) = base + Rs - sqrt(Rs^2 - min(r, a)^2)`, `base = 5 mm`. Rim height = base + Hd.
- Food volume = `sum over cells w * max(h_surface - floor(r), 0) * cell^2` (w, h after the footprint weighting and inpainting of section 2). Where the food heaps above the rim the dome is counted automatically (floor is capped at rim height beyond r = a).
- Centre: midpoint of the 2nd/98th percentile extents of the container-mask points in plane coords, else the centroid of the observed food cells.
- Container uncertainty enters via sigma_cont. Uses typ dims for `typ`.

## 5. Clear cup
Label in {bubble_milk_tea, container_cup, black_tea, green_tea, soy_milk} or container shape `cup`: no depth is used; `method='cup_prior'`, `volume_ml=None`, note "depth unreliable through clear plastic". The caller applies cup size x fill level.

## 6. Decisions to review
- D1 Per-frame reduction is **median**, not max. With 5 mm depth noise and about 2-3 samples per 3 mm cell, max biased a box +16 % (median +4 %). Max remains better for steep walls; revisit with real data.
- D2 The old edge-ring regrow (fill unobserved ring cells with >= 2 neighbours) inflated volume 8-20 % because it was not bounded by a footprint. The new inpaint is bounded by the voted footprint and weighted by `fpw`. FILL is flat (neighbour mean): it overestimates slightly on steep mound edges; the mound still measures -2 to -3 %.
- D3 The +-6 % scale term is added as an independent log-normal factor, but a scale error is common to all items in a meal (PLAN section 5 caveat).
- D4 Coverage < 0.5 widens the interval but does not extrapolate the unseen footprint.
- D5 The plate/box container only adds a note; no rim-based model yet.
- D6 Footprint parameters (FP_SAT 0.5, HULL_VOTE 0.8, FILL_ITERS 12) were tuned on the synth bench to remove the bias with exact depth. Under strong i.i.d. 5 mm pixel noise (test_box_noisy_interval) the box measures about +8 % (was +4 %): smear cells at full wall height enter the footprint. Raising FP_FLOOR to 0.3-0.4 fixes that case but costs the mound -6 %; revisit with real data.
- D7 Interval unchanged (bootstrap + 6 % scale term + container term); it still achieved 97-100 % coverage on the synth bench with mean max/min of 1.44-1.51, so no recalibration was made. The bias remaining from alignment lies inside it.

## Alignment (align.py)
Diagnosis of the aligned-depth bias (meal rice -8.1 %, egg -9.7 % before; exact depth -3..+1 %), synthetic, per-frame statistics:
- Regression dilution is real but small: fitted/true slope = 0.99 (rel noise variance is about 0.4 % of the anchor rel variance), worth about 0.15 mm at 0.5 m. Reverse regression / Deming (delta = var(err 1/d)/var(err rel)) moved the food-minus-table bias by only 0.1-0.3 mm, and an oracle fit on all pixels shows the same bias.
- Anchor conditioning is fine: table anchors span 1/d = 1.5-3.6 (single item, p5-p95; 1.5-2.6 for the meal), the food top is only 0.15-0.25 1/m from the table mean. The slope is not the problem; 10-20 % of table anchors are never near food.
- The actual cause is edge bleed: simulated (and real) depth models blend food borders with a blurred neighbourhood, so food pixels within about 2 px of the mask border read 2-3 mm too far (about 35 % of an egg's pixels). No affine fit can undo that. Interior food pixels are within 0.0-0.5 mm.
- Per-frame food bias has sd about 3 mm (pose noise 2 mm + smooth warp), so a 10-seed mean of one item has a standard error of about 0.3 mm; bench biases of 3 % on a 1 cm egg are partly sampling noise.

Fixes, in order of effect:
1. `fix_mask_edges`: the outer `EDGE_RING_PX` = 2 px of every non-container mask get their rel replaced by a local weighted-plane extrapolation of the mask interior (Gaussian-normalised moments, sigma 3 px, slope ridge 0.5). Food-pixel depth bias +1.1 -> about +0.1 mm. Ring 3 over-corrects (-0.9 mm) and inflates cutlets/mounds.
2. Deming fit (`DEMING_DELTA` = 0.01, closed form, replaces OLS in the RANSAC refinement). Hypothesis (a)/(b) is a minor improvement only (0.1-0.2 mm).
3. `_local_offset`: smooth additive 1/d offset from inlier residuals (Gaussian kernel 30 px, ridge 5), absorbing the slow non-affine warp near the item. Small effect on the mean, no effect on the per-frame spread.
RANSAC, inlier tests in metric depth and all rejection gates are unchanged; everything is closed-form numpy (Swift-portable). `bench_volume.py --exact-depth` prints exact-vs-aligned bias side by side.

Residual: the rice-in-meal / egg-in-meal bias (-5.9 / -5.7 %) is at the limit of what the rel noise allows (oracle bg-only affine fit with true plane and pose gives -4.7 / -5.1 %). Cutlet alone (+7.9 %) is volume.py's noise response (exact depth +2 %, any oracle alignment +8 %); raising the edge ring cures egg/rice but worsens cutlets, so the trade-off is left at ring 2.
