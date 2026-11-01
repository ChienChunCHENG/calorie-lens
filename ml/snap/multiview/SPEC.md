# Multi-view size from a short video: what the browser would compute (NOT recommended to port; see RESULTS.md: FAIL)

Reference code: `geom.py` (pose), `mv.py` (frame pipeline), `rotation.py` (azimuth), `hull.py` (carving), `evalmv.py` (volume -> kcal). Written down so a port is cheap IF a better dataset changes the verdict.

## 1. Frames
- Live scan at <= 640x360 working size. Keep every frame's plate ellipse (cheap) and run the heavy steps only on key frames.
- Key-frame selection (proposal; evaluated only crudely): coverage ring of azimuth bins (12 bins of 30 deg) filled from the registered rotation (section 4); take a frame when its azimuth is >= 20 deg from all kept frames
  and its plate is valid (section 2); stop at 8 key frames or when 120 deg is covered (about 3 s). The dataset could not validate the stop rule (videos span 17-68 deg).

## 2. Plate ellipse and pose per frame
1. YOLOE-seg locator (the 17-prompt research export `loc17.onnx`; rows 14-16 plate/bowl/tray, 1-3 dish/plate/bowl of food x 0.8), conf >= 0.10, NMS iou 0.5; keep score >= 0.15, mask area >= 4% of the frame,
   keep the best candidate by score x mask area. Mask = sigmoid(coefs . protos) cropped from the letterbox, > 0.5.
2. Ellipse = least-squares fit (cv2.fitEllipse) to the biggest contour of the closed mask (5x5) after DROPPING contour points within 2 px of the image border; reject if < 30 remaining points,
   if > 45% of the contour touched the border, or if minor < 0.15 major.
3. Intrinsics: f = (W/2) / tan(hfov/2), hfov = 69 deg, principal point at the centre.
4. Circle pose (Kanatani, closed form, `geom.pose_from_ellipse`): conic Q of the ellipse; Qc = K^T Q K (symmetric); eigen-decompose, make two eigenvalues positive and one negative (flip sign), sort l1 >= l2 > 0 > l3, eigenvectors E.
   Normal candidates n = E [ sqrt((l1-l2)/(l1-l3)), 0, +-sqrt((l2-l3)/(l1-l3)) ]. For each: tangent basis u = E[:,1], v = n x u; plane n.P = 1 intersected with the cone P^T Qc P = 0 is a circle
   (P = n + s u + t v: quu, qvv, quv, gu, gv, g0 as in the code), centre (s0,t0) = -M^-1 g, radius rho at d = 1; true distance d = (D/2)/rho, centre c = d (n + s0 u + t0 v); flip so c.z > 0 and n points to the camera.
5. Two-fold ambiguity: keep the solution with n_y < 0.05 (table normal points up in the image) and the smaller |n_x| (camera roughly level). With an IMU (DeviceMotion: gravity) use n = gravity-up directly; not evaluated.
6. D = 26 cm plate / 24 cm bowl (class 15 / row 2), user-editable "my plate is X cm". tilt = angle(n, -c/|c|).
Verified on synthetic circles (exact to 3 digits, `test_geom.py`). On Nutrition5k 92-95% of views give a pose; correctness is much lower (cut-off plates, wrong plate).

## 3. Food footprint (silhouette)
- `portion.food_pixels` on the frame: Lab (L 0-100, a, b centred) after 5x5 blur; plate colour = peak of the (12, 8, 8)-binned Lab histogram of the ring 0.80-0.97 of the ellipse; food = Lab distance > 18 inside the 0.97 ellipse.
- Opening with a 15 px elliptical kernel (kills rim bevels / printed rings; also kills thin food), intersect with the 0.95 ellipse, union the YOLOE `food` mask (row 0) inside the 1.25 ellipse and outside the 0.97 one (food above the rim),
  final 3x3 opening. `mv.food_mask`.

## 4. Azimuth between frames (`rotation.py`)
- Rectify: homography from plate-plane cm coordinates (x, y; basis A = normalize(n x z_or_x), B = n x A) to pixels, M = K [A B c]; render a 128x128 top-down image covering +-14 cm (0.22 cm/px).
- Angle: polar resample (180 angles x 40 radii, r = 0.15-0.95 of the half size), grey, per-radius zero mean, unit std; circular cross-correlation over the angular axis via FFT; peak with parabolic refinement.
  Output phi: the content of frame b is rotated by +phi (angle measured x -> y of the rectified image) relative to frame a. Peak z-score reported as confidence.
- Works within one video (small angles, z 2-3); across cameras with different elevations it failed (z about 1.7).

## 5. Volume, three variants (all evaluated; none good)
a. Depth heightmap per frame (`mv.frame_heightmap`): Depth Anything V2 small at 322x574 -> relative inverse depth d. Fit 1/Z_plane = a d + b (Huber IRLS, 8 iterations, d standardised) on plate pixels inside the 0.95 ellipse that are > 9 px from the food mask,
   <= 6000 sampled pixels; Z_plane = (n.c)/(n.r) with r = K^-1 (u, v, 1). Food pixels: 1/Z = a d + b, P = Z r, height h = (P - c).n, base = P - h n, plane coordinates (x, y) in cm; cell 0.5 cm on a 64x64 grid;
   per-cell median of h clipped to [-3, 12]; footprint = cells hit, closed with a 5x5 morphological close; holes inpainted by 3x3 neighbour mean (10 iterations); volume = sum(max(h,0)) x 0.25 mL; area = footprint cells x 0.25 cm2.
   Per scan: median over the selected frames.
b. Footprint x thickness (`ah2`, dev-selected): V = median over views of area_cm2 x (max(mean height,0) + 2 cm). It is basically a tilt-corrected area from several views.
c. Visual hull (`hull.carve`): voxels 0.5 cm, x,y in [-14, 14], z in (0, 10) cm in the reference view's plate frame. Voxel (x, y, z) in view v: (x', y') = Rot(phi_v)(x, y); P = c_v + x' A_v + y' B_v + z n_v; pixel = K P. Keep when it projects inside the
   (2 px dilated) food silhouette in >= 75% of the views where it projects in the image and in at least 2 views, inside the 14 cm disc. Volume = count x 0.125 mL.

## 6. Grams and kcal (`evalmv.py`)
- grams = k V^beta (dev fit; chosen method: k = 0.311, beta = 1, gamma = 0.4). Item share = food-pixel share of the item in the overhead photo (portion/ `assign_to_boxes`).
- kcal/g per candidate: max(kcal100/100, 0.02)^gamma x 1.2^(1-gamma); mixture over the shipped candidates with their weights. Total = grams x sum_items share x E[kcal/g].
- Range: Monte Carlo, shared log-normal error on grams sigma = 0.8 (calibrated to 80% coverage on dev), candidate drawn per item; report p10/p50/p90. Median p90/p10 = 8.3x (B: 4.8x).
- Fallback: no usable view -> the single-photo portion estimate (B).

## 7. Cost (reference CPU, 3-4 threads)
YOLOE-seg 0.4-0.5 s/frame (shipped, 20 MB); Depth Anything V2 small about 1.1 s/frame, about 50 MB fp16; everything else under 0.3 s per frame set. Masks-only variant (b without depth heights, or hull): about 5 s for 12 views.

## 8. Parity
Reference outputs to compare against a TS port: `$SP/research/mv/b_dev.pkl` / `b_test.pkl` (per-view ellipse, tilt, volume, area, azimuth; per-set hull volume). Tolerances: pose tilt 2 deg, area 5%, volume 10%.
