# Multi-view food size in a plain browser: results (Nutrition5k side-angle videos)

## A/B/C table (held-out test; recognition identical for A, B, C)
Verdict against the agreed bar (median |err| <= 30% AND p10-p90 coverage >= 75%) in the realistic setting (<= 120 deg arc, <= 8 frames): **FAIL**. Multi-view size did NOT beat the single-photo plate ruler
and gives no demonstrated UX gain over it. The volume signal we can extract from these videos is as weak as the area signal (r = 0.44 for the best volume proxy vs 0.40 for area alone).

Test set: 57 of the 60 dishes have side-angle videos (3 have none; dropped for ALL methods, so B shows 43.2% here vs the official 42.6% on 60). 26 are meal-sized (>= 300 kcal). 23 same-dominant-ingredient pairs.
Same logged items and candidate foods for A, B, C (the shipped pipeline on the overhead photo); only the portion/size step differs. Dishes with no logged item get the same fallback (249 kcal, 64-562) in all three.
Spread of A, B, C each calibrated on the 145 dev dishes to ~80% p10-p90 coverage (A spread x1.8, B x0.9, C sigma 0.8); coverage on test came out 88 / 77 / 81%.

| method | median abs err (all) | within +-25% (all) | median err, meals >=300 kcal | within +-25%, meals | p10-p90 coverage | median p90/p10 | pair order correct | median est-ratio / true-ratio |
|---|---|---|---|---|---|---|---|---|
| A typical portion (no size) | 48.3% | 24.6% | 48.6% | 15.4% | 87.7% | 9.8x | 52.2% (ties = 0.5) | 0.50 |
| B single photo, plate ruler (ml/snap/portion) | **43.2%** | **31.6%** | **33.5%** | **38.5%** | 77.2% | **4.8x** | 69.6% | 0.71 |
| C multi-view, dev-selected (footprint area x (depth height + 2 cm), cams A+B+C, 12 views) | 53.6% | 24.6% | 41.5% | 30.8% | 80.7% | 8.3x | 56.5% | 0.60 |
| C-hull (visual hull, cams A+B+C, registered azimuths) | 49.6% | 28.1% | 38.6% | 38.5% | 84.2% | 13.0x | 73.9% | 0.76 |
| C-depth (pure depth-integrated volume, 4 cams x 4 frames) | 55.0% | 28.1% | 46.7% | 26.9% | 80.7% | 8.8x | 56.5% | 0.38 |

Reading it: C (dev-selected) is worse than B on every accuracy metric and 1.7x wider at equal coverage. The hull variant has the best pair ordering (73.9%) but n = 23 pairs (one standard error is about 9 points), the widest
range (13x) and 49.6% median error; it was NOT the dev-selected variant (selection = highest dev volume-vs-mass correlation); it was run on test as a pre-declared secondary, so its pair number is not evidence of a gain.
"Pair order correct": of dish pairs with the same dominant ingredient (by kcal share, from the metadata CSV) and true mass ratio >= 1.5x, the share where the method ranks the dishes in the true kcal order; ties count 0.5.
"Median est-ratio / true-ratio" = predicted kcal ratio (heavier/lighter) divided by the true kcal ratio; 1.0 is perfect; all methods under-separate (0.4-0.8).
Pairs on dev (230 pairs from 145 dishes; params fitted on the fit part so slightly optimistic): A 51.1%, B 64.1%, C(chosen) 69.4%, C-hull 67.6%, C-depth 59.8%. Validation split only (36 dishes, 19 pairs): A 42.1, B 68.4, C 68.4, hull 84.2.
Example pairs (6; every k-th pair along the ratio range, no dish twice, not hand-picked): `$SP/research/mv/pairs/` (`pairs.json` + both PNGs per pair, A/B/C numbers).

## Development results (145 dev dishes, split by md5 as in portion/: fit 109 / validation 36)
Validation (C parameters k, beta, gamma fitted on the fit split; spread calibrated on all dev):

| method | median err | within 25% | meals >= 300 (n=12) median | coverage | p90/p10 | Pearson |
|---|---|---|---|---|---|---|
| A | 51.7% | 22.2% | 43.8% | 77.8% | 8.8x | 0.02 |
| B | 29.2% | 44.4% | 35.8% | 80.6% | 4.9x | 0.43 |
| C chosen | 43.0% | 27.8% | 37.1% | 91.7% | 8.1x | 0.29 |
| C-hull | 34.6% | 30.6% | 33.6% | 83.3% | 13.0x | 0.25 |
| C-depth | 49.5% | 36.1% | 47.9% | 77.8% | 8.8x | 0.23 |

All 145 dev, median err: A 57.7 / B 39.1 / C 48.1 / hull 44.9 / depth 50.3; meals (n=59): 57.8 / 35.1 / 38.9 / 35.7 / 51.7. B's 29.2% on 36 validation dishes is a lucky sample (its 60-dish test number is 42.6%): do not read it as B passing.

### Volume signal (dev, 145 dishes): volume vs TRUE mass, one constant density (k from the fit split)
| volume source | Pearson r (log V, log mass) | median abs log ratio |
|---|---|---|
| footprint area x (depth height + 2 cm), cams ABC (dev-selected) | 0.44 | 0.41 (about +-50%) |
| footprint area only (no height), cam B | 0.40 | 0.44 |
| depth-integrated volume, 4 cams | 0.35 | 0.64 |
| depth-integrated volume, cam B alone | 0.25 | 1.31 |
| visual hull, camera subsets (registered) | 0.27-0.35 | 0.48-0.62 |
| min(depth, hull) | 0.20-0.31 | 0.55-1.29 |

Depth adds about +0.04 correlation over plain area; the hull is no better than area. Monocular depth heights are tiny and noisy: median per-view mean height 0.75 cm; the plane-fit residual of the depth is 7.5% of 1/Z (median), the same size as the signal (a 3 cm mound at 40 cm is about 7.5%).
For reference, the portion agent's oracle (TRUE mass x constant density) is 33.5% median kcal error; our mass accuracy (r 0.44, +-50%) is far from that oracle.

### By arc and frames (validation median err, dev-selected method; arc = azimuth span from our rotation registration: indicative only)
| cameras | frames per camera | views | median arc (deg) | val median err (%) | within 25% (%) |
|---|---|---|---|---|---|
| B | 1 / 2 / 4 | 1 / 2 / 4 | 0 / 17 / 43 | 50.6 / 51.6 / 45.2 | 28 / 16 / 28 |
| C | 1 / 2 / 4 | 1 / 2 / 4 | 0 / 24 / 68 | 45.2 / 43.9 / 45.0 | 24 / 27 / 26 |
| B+C | 1 / 2 / 4 | 2 / 4 / 8 | 98 / 151 / 181 | 30.7 / 34.3 / 38.3 | 42 / 33 / 30 |
| A+B+C | 1 / 2 / 4 | 3 / 6 / 12 | 159 / 195 / 229 | 37.0 / 43.6 / 39.4 | 30 / 30 / 30 |
| A+B+C+D | 1 / 2 / 4 | 4 / 8 / 16 | 182 / 229 / 254 | 43.5 / 40.3 / 41.8 | 30 / 30 / 36 |

n = 29-33 validation dishes per row (one dish = 3 points): differences between rows are noise; error does not fall with more views or wider arc. One video only gives arcs <= ~70 deg; larger arcs only by mixing cameras, whose
relative azimuth we could NOT register reliably (caveats). 360 deg and clean 120 deg single-sweep settings are NOT testable on this data and are not reported. The 3/5/8/12-frame sets map to 3 views (A,B,C x 1), 6 (x 2), 8 (B+C x 4), 12 (A+B+C x 4).

### Sensitivity (dev; Pearson r of log volume vs log mass for the dev-selected method, same cameras)
| setting | r(log V, log mass) area x (h+2cm) | r depth-only | val median err (k refitted) |
|---|---|---|---|
| base (f from 69 deg FOV, plate 26 cm) | 0.44 | 0.29 | 39.4% |
| focal x0.85 | 0.43 | 0.30 | 39.3% |
| focal x1.15 | 0.44 | 0.29 | 37.9% |
| plate 22 cm | 0.43 | 0.30 | 35.4% |
| plate 30 cm | 0.44 | 0.29 | 41.9% |

The signal is insensitive to focal error and plate diameter because a global k absorbs the scale; the kcal differences (35-42%) are within the noise of 33 validation dishes. (Validation numbers here use the ABC set with k refitted per setting, hence 39.4% vs the 43.0% in the main table, which used the calibration/median path.) This is not a sign the method is robust: the ceiling is the weak volume signal, not scale.

## Scan cost and failures
- Frames/arc: 3-12 views evaluated; error is flat from 1 to 16 views. Nothing here supports "more frames / wider arc improves size".
- Scan failure (no plate pose at all): 0 of 145 dev dishes and 0 of 57 test dishes; per-view pose success 94.6% dev, 92.0% test (valid ellipse, <45% of the contour clipped by the frame). "Pose found" is NOT "pose correct": by eye a large share of views
  have a wrong or partial plate (cut-off plate, a second plate chosen, square/oval plate, reflections); see failure cases. The raw log field "scan_fail" (10-12%) is the share of dishes with no logged item (gate), identical for A, B, C; it is not a plate failure.
- Compute per scan (desktop CPU, 3-4 threads, shared machine; browser WASM would be about 2-4x slower): decode + YOLOE-seg (plate/bowl/tray + food, 640 px) about 0.4-0.5 s/frame; Depth Anything V2 small at 322x574 about 1.1 s/frame;
  pose + food mask + heightmap about 0.15 s/frame; hull carving about 0.1-0.3 s per camera set. A 12-view scan: about 5 s without depth, about 20 s with it. Depth Anything V2 small = about 25 M params, about 50 MB as fp16 weights (not exported to ONNX here). The 20 MB locator is already shipped.
- Depth: about +0.04 correlation for about 3x the compute and 50 MB: not worth it on this evidence.

## Failure cases (test)
- dish_1563567722 (576 g, 645 kcal; C 144, B 349, A 327): the large plate is cut by the frame in cams B/C/D; the ellipse lands on a second small plate (B) or a fragment (A, D); heights are 0. A plate ruler cannot work when the plate is not fully in view.
  (`$SP/research/mv/vis/test_dish_1563567722.jpg`)
- dish_1565812146 (93 g, 73 kcal; C 154) and dish_1565811529 (104 g, 117 kcal; C 196): small portions get area from rim bevels/reflections; C inflates them, so the true 8.8x kcal ratio against dish_1563567722 is estimated as 0.94x (A 1.8x, B 2.8x).
- dish_1566414754 vs dish_1567023184 (533 g vs 195 g, true kcal ratio 3.9x): C 0.87x, B 1.9x, A 2.2x. dish_1558638292 vs dish_1558640450 (2.4x true): A 1.13, B 1.17, C 1.18.
- Square/oval plates (many in Nutrition5k): the circle model gives the right TILT (a square is affinely a circle) but only an "equivalent diameter"; ovals (dev dish_1558460908) give a wrong aspect and cm scale.
- Printed rings and bevels on paper/rim plates are counted as food by the plate-colour test (dev dish_1558380414, 52 g of almonds: footprint 160-200 cm2); a 15 px opening only removes part (`$SP/research/mv/vis/dev_dish_1558380414.jpg`).

## Caveats
1. Not a phone: the side videos are 4 fixed cameras at different elevations watching a plate on a turntable; each video is about 1.4 s (33-40 frames) and the plate turns only about 17-68 deg (measured). The cameras are not time-synchronised
   and our azimuth registration between cameras is weak (cross-correlation peak z = 1.7 median; camera-to-camera angles scatter across dishes), so the arc column and the hull are partly noise. A real phone sweep (continuous, known intrinsics, IMU gravity) could behave better or worse: unknown.
2. Intrinsics: one 69 deg horizontal FOV for all cameras (real lenses differ, images may be cropped). Plate diameter 26 cm (bowl 24) as in portion/; Nutrition5k plates are often square, oval or paper.
3. Oblique views: median tilt 60 deg from the plate normal (30 deg elevation); 10% of views are above 69 deg where ellipse and depth are poor.
4. Dev is 145 dishes (36 validation): the method was chosen by correlation on all 145, parameters fitted on 109; validation numbers are noisy (+-8 points). Test is 57 dishes: about +-7 points on a median, +-9 on pair accuracy (23 pairs).
5. Depth Anything V2 small ran as-is (fp32 PyTorch, 322x574). A larger model or metric depth might do better; not tried. The test set was evaluated once, with frozen dev parameters, for the three pre-declared variants.
6. A and B use the portion agent's frozen parameters (fitted on its own dev split, which contains our dev dishes).

## Honest conclusion
Size from several angles with browser-feasible methods is no better than size from one overhead photo on this data. The limit is the same as before: mass is poorly constrained by what is seen (area r about 0.4; height from monocular depth is noise-level),
plus density uncertainty (0.5-2.3 kcal/g). The plate-ellipse pose needs the whole round plate in view, which these videos (and probably many close phone sweeps) do not give. No variant approached the 30% bar; do not build the live-scan UI on this evidence.
Open options: a user-entered plate size and a small/medium/large tap, a REAL phone dataset with weighed food, depth from phones with LiDAR (native path already exists).

## Reproduce
```
stage1.py (decode, YOLOE plate/food masks) -> stage2.py (Depth Anything V2) -> build.py (poses, depth heightmaps, rotation registration, hull) -> run_eval.py dev | test (test once) -> pairs_export.py, viz.py
```
Intermediates in `$SP/research/mv/` (s1_*, s2_*, b_*.pkl, fr_dev.pkl, fr_test.pkl, vis/, pairs/).
