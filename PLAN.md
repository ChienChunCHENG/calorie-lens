# Calorie Lens on iPhone: Build Plan

*Drafted 2026-10-30. Research by 7 parallel Sonnet agents; planning and QC by Opus.*

## 0. Goal and hard constraints

**Goal:** a personal calorie lens running on your own iPhone. You point the camera at a Taiwanese meal and sweep for about 2 seconds, with no shutter button. Each food item then gets a label that follows it, with a **calorie range** (`滷肉飯 450–600 kcal`). A total sits at the bottom. Everything runs on-device and offline.

**Your setup (confirmed):**

| Constraint | Consequence |
|---|---|
| Windows 11, **no Mac** | No local Xcode. Native builds run on a **GitHub Actions macOS runner**, and you install with **Sideloadly** from Windows. |
| **Non-Pro iPhone** (no LiDAR) | No metric depth sensor. Portion size comes from ARKit's metric camera tracking, the table plane, monocular depth and container priors, so ranges must be wider. |
| **Free Apple ID** only | Sideloaded apps expire after **7 days** and need re-signing (Sideloadly can auto-refresh). Max 3 sideloaded apps. No TestFlight, push or iCloud, none of which we need. |

**What this changes from the earlier Android/ARCore plan:** the pipeline idea (detect, measure, look up, show ranges) carries over unchanged. The capture hardware is weaker (no LiDAR, no ARCore Depth API), so the accuracy targets in §7 are relaxed for v1.

---

## 1. Strategy: two tracks, one shared core

```
                 ┌──────────────────────────── shared core ────────────────────────────┐
                 │  ml/      one trained YOLO11n-seg  ──► ONNX (web)  +  Core ML (iOS)  │
                 │  data/    nutrition.json (TFDA + sugar table + densities + priors)   │
                 │  spec/    range-engine spec + golden test vectors (TS and Swift)      │
                 └───────────────┬──────────────────────────────────┬──────────────────┘
                                 │                                  │
     Track W: Web Lens (Safari)  ▼                                  ▼  Track N: Native Lens (Swift)
     ─ playable in ~1 week                                         ─ playable in ~4–6 weeks
     ─ zero install, instant iteration                             ─ ARKit metric tracking + planes
     ─ 2D labels follow boxes (like                                ─ 2 s sweep, multi-frame volume
       Google Translate camera mode)                               ─ world-anchored labels
     ─ portion = container prior × fill                            ─ Core ML on the Neural Engine
     ─ no true depth                                               ─ built in CI, sideloaded
```

**Why two tracks:**
- **Track W** gets something on your phone this week, with no Mac, no signing and no 7-day expiry. Iteration takes seconds. It proves the parts that don't need depth: the dish model, the nutrition data, the range logic and the UX.
- **Track N** is the real calorie lens. Only native ARKit gives metric camera poses and plane detection on a non-LiDAR iPhone, and that is what makes sweep-based volume possible. The cost is a slower loop: about 5 to 8 minutes per CI build, no debugger, and re-signing every 7 days.
- The shared core means none of Track W's work is thrown away.

---

## 2. Repository layout

```
calorie-len/
  PLAN.md
  data/
    sources/            raw downloads (TFDA CSV, FAO density xlsx, MOHW sugar table); git-ignored if large
    build_nutrition.py  → data/nutrition.json  (pivot TFDA long format, merge priors)
    dishes.yaml         hand-curated per-dish entries (min/typ/max + citations)
  spec/
    range_engine.md     the math (§5) — single source of truth
    vectors.json        golden input→output cases both ports must pass
  ml/
    train_seg.ipynb     Colab: fine-tune YOLO11n-seg, export ONNX
    labels.yaml         class list (zh/en/id)
    eval.py             held-out mAP + per-class confusion
  web/                  Vite + TypeScript + onnxruntime-web
  ios/
    project.yml         XcodeGen spec (no .xcodeproj committed)
    Sources/CalorieLens/...
  tools/replay/         Python: replay native sweep dumps → volume algorithm dev on Windows
  .github/workflows/
    ios.yml             build unsigned IPA artifact
    coreml-export.yml   .pt → .mlpackage on the macOS runner
```

**Repo visibility:** a **public** repo gets unlimited free macOS CI minutes. A private repo gets 2,000 minutes a month, but macOS counts 10×, so that is about 200 macOS minutes, or 25 to 40 builds. Keep **meal photos and the dataset out of git** either way (privacy). Store them in Roboflow or Google Drive.

---

## 3. Milestones

Time estimates assume part-time work. Each milestone has an exit check you can test on the phone.

### M0: De-risk the toolchain (days 1–2) ⚠ do this first
1. Create the GitHub repo. Add a minimal SwiftUI "hello camera" app (`ios/project.yml` plus one view showing an `ARView` and the tracking state).
2. Add `.github/workflows/ios.yml`. It runs on macos-26, installs XcodeGen, then runs `xcodebuild -sdk iphoneos CODE_SIGNING_ALLOWED=NO`. It zips `Payload/CalorieLens.app` into a `.ipa` and uploads that as an artifact.
3. On Windows, install **iTunes and iCloud from apple.com**, not the Microsoft Store builds. Then install **Sideloadly** and install the IPA with your free Apple ID.
4. On the iPhone, turn on **Settings › Privacy & Security › Developer Mode**, then trust the developer profile.
5. In parallel, scaffold `web/` with Vite. Expose it with `cloudflared tunnel --url http://localhost:5173` and open the HTTPS URL in Safari. Confirm that `getUserMedia({video:{facingMode:'environment'}})` shows the rear camera.

**Exit:** both apps open the camera on your iPhone, and AR tracking reaches `.normal` in the native app.
**If the Sideloadly path fails:** fall back to [xtool](https://github.com/xtool-org/xtool) under WSL. It is less mature, and Core ML compiling is uncertain. Otherwise go Track W-only until you have a Mac.

### M1: Nutrition core (days 2–5)
1. Download the TFDA 食品營養成分資料集 CSV: data.gov.tw/dataset/8543, export URL `https://data.fda.gov.tw/data/opendata/export/20/csv`, under the open government data licence. It is in **long format**, one row per food per nutrient. Pivot it to kcal, protein, fat and carbs per 100 g, and keep `樣本數` and `標準差` as the source for min/max.
2. Grep the TFDA file for prepared dishes (滷肉飯, 牛肉麵, 鹽酥雞). Expect mostly raw ingredients. Prepared dishes go in `dishes.yaml`, assembled from component ingredients plus the 國健署 food exchange table (食物代換表), with `confidence: low` wherever they are estimates.
3. Densities: use the FAO/INFOODS Density DB v2 (xlsx) and USDA FoodData Central cup weights (density = g/cup ÷ 236.6). Store **bulk** density for loose items such as greens, fried pieces and noodles.
4. Drinks: use the MOHW 700 mL sugar table (珍奶 全糖 61.6 g / 半糖 44.8 g / 微糖 39.2 g). Model a drink as `sugar_g×4 + base + toppings`. Do not use density × kcal for drinks.
5. Oil adders per cooking method go in as min/max ranges. Sourcing here is weak: only fried chicken has a cited figure (about 15% fat uptake), so the ranges should be wide.
6. Write `spec/range_engine.md` and `vectors.json` (§5). Implement the engine in TypeScript first.

**Exit:** `nutrition.json` covers the first 15 classes (§4), and the TS engine passes all vectors.

### M2: Web Lens v0, playable (days 4–8)
1. Bootstrap without a custom model: COCO YOLO11n (boxes only, ONNX, `imgsz=320`). It finds `bowl`, `cup` and `dining table`, plus a few foods. You **tap a box** and pick a dish from chips (top 15 dishes). This is fully offline and playable right away.
2. Portion without depth: the container prior.
   - Pick the container once: 小碗 / 大碗 / 便當盒 / 26 cm 盤 / 500 or 700 mL 杯.
   - Portion = container capacity × fill estimate, with the fill estimate from food-mask area divided by container area. Use a wide default fill of 0.6–1.0 until masks exist.
3. Inference loop: `onnxruntime-web` with `executionProviders: ['webgpu','wasm']` and a toggle in the UI, at about 5–10 Hz with latest-frame-wins. Use IoU tracking with smoothing so labels follow boxes between inferences.
4. UI:
   - Labels show `name  lo–hi kcal`, plus a total bar.
   - A sugar and ice picker (甜度 / 冰塊) appears automatically for any cup.
   - A "no numbers" toggle.
   - A wellness disclaimer.
5. Deploy to Cloudflare Pages. Its `_headers` file allows COOP/COEP, which you need later for WASM threads.

**Exit:** you can point Safari at lunch, tap items, see ranges and a total, and it runs for 5 minutes without crashing. That last check matters because of the WebKit WebGPU memory-leak bugs on iOS 26; keep the WASM fallback.

### M3: Dataset and custom model (weeks 2–4, overlaps M2 and M4)
1. **Collect:** photograph your own meals from overhead and at 45°, in different lighting, bowls and shops. Target about 1,000 to 1,500 images for 40 classes, at **150 to 300 instances per class**.
2. **Seed data:** Roboflow Universe "taiwan food" and "bento" sets (check each licence), plus UEC-Food256 and FoodSeg103 for warm-up. Both of those are non-commercial, which is fine for personal use.
3. **Label:** Roboflow Smart Polygon or X-AnyLabeling (local, Windows), both SAM-assisted. *Optional speed-up:* have a vision LLM (e.g. Claude) propose dish names per photo, and you confirm or fix them. This helps labelling only; the app stays offline.
4. **Train:** YOLO11n-seg in Colab (free T4) or on your own NVIDIA GPU. Start with 15 classes, then grow to 40.
5. **Export:**
   - ONNX for the web, from Colab.
   - Core ML from `.github/workflows/coreml-export.yml` on the macOS runner, with Python 3.11 or 3.12 pinned: `yolo export format=coreml nms=True quantize=16`. The macOS runner can also *validate* the `.mlpackage`, which Linux and Colab cannot.
   - Check early: whether segment masks come out decoded with `nms=True`, or whether you must do coefficients × prototypes in Swift and TS yourself.
6. **Evaluate** on a held-out set of *your own* meals (never seen in training): mask mAP50 and per-class confusion. The usual confusions are 滷肉飯 vs 肉燥飯 and 雞腿 vs 排骨.

**Exit:** the 15-class model reaches ≥85% top-1 on held-out meals and has replaced tap-to-pick in Track W.

### M4: Native Lens: sweep, volume, AR labels (weeks 3–6)
Build in this order. Each step is a sideloadable build.

1. **ARKit session.** Use `ARWorldTrackingConfiguration` with `planeDetection = [.horizontal]` and `sceneReconstruction` off (it needs LiDAR). Gate on `trackingState == .normal`.
2. **Sweep state machine.**
   - Idle → *plane found and food detected* → **Capturing** (2 s, slow arc) → Processing → Labels.
   - Keep 10 to 15 sharp keyframes. Reject frames with motion blur or a high angular rate.
   - Prompt "slower" or "move back" when the camera is too close or moving too fast.
3. **Detector.** Run the YOLO11n-seg Core ML model through Vision or `MLModel` on each keyframe. Decode masks in Swift.
4. **Depth without LiDAR.** Use Apple's `coreml-depth-anything-v2-small` (float16, about 50 MB, about 31–34 ms on the Neural Engine). It outputs *relative* depth, so it needs aligning:
   - **Metric anchor 1:** the ARKit table plane, raycast through background pixels. Fit scale and shift by least squares on the table region outside the food masks.
   - **Metric anchor 2:** ARKit `rawFeaturePoints` that land in the frame.
   - **Sanity check:** the container's rim diameter must match the chosen container class.
5. **Volume.**
   - Unproject masked pixels to world space using the ARKit intrinsics, scaled to the depth-map resolution. Undo the YOLO letterbox before resizing the masks.
   - Compute heights above the plane on a 3 mm grid aligned to the plane. Take the **per-cell median across keyframes**, and ignore cells with fewer than 3 observations.
   - Volume = Σ height × cell area.
   - **Bowls:** fit a rim circle and use a container cavity model (spherical cap or frustum per container class). Food volume = cavity below the food surface.
   - **Soup:** flat liquid level inside the cavity.
   - **Clear cups (bubble tea):** never trust depth. Use cup-size class × fill level, labelled *estimated from cup size*.
   - If volume fails for any reason, fall back to the class's default portion range and widen it.
6. **World-anchored labels.** Raycast each mask centroid to an `ARAnchor`. Every frame, `camera.projectPoint` gives a 2D position for a **SwiftUI overlay** label, smoothed with a low-pass filter. Do not use 3D text meshes, because CJK glyphs render poorly.
7. **Debugging without Xcode.** This is essential.
   - An **in-app log viewer**, plus a crash handler that writes the last crash to disk and shows it on the next launch.
   - **Sweep dump export:** keyframes (JPEG), depth outputs, poses, intrinsics, masks and the plane, as one zip you share to the PC.
   - `tools/replay/` re-runs the volume algorithm in **Python on Windows**. Develop and tune the algorithm there, then port it to Swift. That way not every change costs a CI round trip.

**Exit:** on your phone, a 2-second sweep over a 3-item meal produces anchored labels with ranges within 3 seconds of the sweep ending.

### M5: Home evaluation (week 6, then continuous)
1. Use a kitchen scale (1 g) and a measuring cup. Use water displacement (sealed bag) for solid items and leveled-rice fills to check bowl volumes.
2. Build a **20-dish set**: flat, domed, in a bowl, soup, stacked, and a clear cup. Do 3 to 4 sweeps each at different angles and lighting.
3. Log, per item and per meal:
   - per-item volume MAPE;
   - meal kcal MAPE;
   - **signed bias**;
   - **range coverage** (% of true values inside the shown range);
   - mean range width.
4. Tune range widths on one half of the set and report coverage on the other half, so the ranges are not overfit.

### M6: Polish and safety (week 7+)
- Ranges by default and a "no numbers" mode. Frame the app as general wellness, with no medical or insulin-dosing claims.
- For drinks, always prompt for 甜度 and 冰塊, defaulting to 半糖. 董氏基金會 testing found that over 60% of "半糖" drinks are really 7–8 分糖, so bias the default upward.
- Local-only meal log on the phone. No cloud.

---

## 4. First label set (start with 15, grow to ~40)

**First 15**, chosen for frequency and visual distinctness:

| | | | |
|---|---|---|---|
| 白飯 | 滷肉飯 | 雞腿 | 排骨 |
| 滷蛋 | 荷包蛋 | 炒青菜 | 滷豆腐 |
| 牛肉麵 | 水餃 | 鍋貼 | 蔥油餅 |
| 雞排 | 鹽酥雞 | 珍珠奶茶 | |

**Next 25:**

| Group | Dishes |
|---|---|
| Staples | 蛋炒飯, 雞肉飯, 稀飯, 麵線, 陽春麵, 炒麵, 小籠包 |
| Breakfast and handheld | 蛋餅, 飯糰, 割包 |
| Sides and mains | 高麗菜, 滷白菜, 酸菜, 香腸, 魚排 |
| Snacks | 蚵仔煎, 甜不辣, 臭豆腐, 滷味, 地瓜, 肉圓 |
| Drinks and dessert | 紅茶, 綠茶, 豆漿, 豆花 |

Mixed bowls such as 牛肉麵 and 滷肉飯 are **one item each**. Do not try to segment their components.

---

## 5. Range engine (shared spec)

For each solid item:

```
mass_med  = V_med · ρ_typ
kcal_med  = mass_med · e_typ/100
σ_i       = ln(max_i / min_i) / 3.29          # per factor, treating min/max as a 90% interval
σ_total   = sqrt(σ_V² + σ_ρ² + σ_e²)          # log-normal, factors assumed independent
lo, hi    = kcal_med · exp(∓1.645 · σ_total)
lo += oil_min·9 ;  hi += oil_max·9            # oil added as an interval, not in log space
lo = min(lo, 0.85·kcal_med) ; hi = max(hi, 1.15·kcal_med)   # ±15 % floor — inputs are partly unverified
```

- **Drinks:** `kcal = sugar_g(level, size)·4 + base(milk/creamer) + toppings`. Each term has a min/max, and they combine by interval.
- **Volume failed:** use `default_portion_g` min/max in place of V·ρ, and flag the label.
- **Meal total:** sum the medians, and combine the half-widths in quadrature. Show the meal range, not the sum of the item ranges. *Caveat:* quadrature assumes item errors are independent, but a shared scale error (one bad plane fit) moves every item the same way. If meal coverage in M5 falls below 80%, switch to a linear sum of half-widths for the volume term.
- **Calibration:** after M5, scale σ_V per capture mode so that coverage is ≥80%.

`nutrition.json` entry (schema):

```json
{ "id":"rice_white", "zh":"白飯", "en":"Steamed white rice", "kind":"solid",
  "density_g_ml":{"min":0.60,"typ":0.70,"max":0.80},
  "kcal_100g":{"min":125,"typ":140,"max":170},
  "macros_100g":{"p":2.5,"f":0.3,"c":31},
  "oil_g_per_100g":{"min":0,"max":0},
  "default_portion_g":{"min":150,"typ":200,"max":280},
  "container_prior":["bowl_small","bowl_large","bento"],
  "source":["TFDA row <id>","食物代換表"], "confidence":"med" }
```

The numbers above are placeholders from research and need checking against the TFDA rows in M1.

---

## 6. How the build fans out to agents

This follows the working rule for this repo: **Sonnet subagents do the building and research, and Opus plans, integrates and does QC.** The workstreams below are independent once `spec/` is agreed, so they can run in parallel. Your ceiling is 16 agents; each wave uses 4 to 6, which leaves room for research agents spun up when a wave hits an unknown.

| Wave | Parallel Sonnet agents | Opus QC gate |
|---|---|---|
| 1 (M0, M1) | ① CI workflow + XcodeGen skeleton · ② web scaffold + camera · ③ TFDA ingest + `build_nutrition.py` · ④ density/sugar/oil tables + `dishes.yaml` · ⑤ range engine TS + vectors | Review the spec, run the vectors, check the citations in `dishes.yaml` |
| 2 (M2, M3) | ⑥ web lens UI + tracking · ⑦ Colab training notebook + export · ⑧ Core ML export workflow · ⑨ labelling guide + class taxonomy | Test on the phone, check the model eval |
| 3 (M4) | ⑩ ARKit session + sweep state machine · ⑪ Core ML seg decode (Swift) · ⑫ depth alignment + replay tool (Python) · ⑬ volume + bowl model (Python, then Swift) · ⑭ SwiftUI label overlay + log viewer · ⑮ range engine Swift port | Check the Swift port against the vectors, plus replay-vs-device parity |
| 4 (M5) | ⑯ eval scripts + report | Final accuracy review against §7 |

---

## 7. Targets (relaxed for a non-LiDAR iPhone)

| Metric | Web Lens (M2/M3) | Native Lens (M4) | Original research target (depth phones) |
|---|---|---|---|
| Top-1 dish recognition (own held-out meals) | ≥85% | ≥85% | ≥90% |
| Per-item volume MAPE | n/a (container prior) | **≤25%** | ≤15% |
| Meal kcal MAPE | ≤35% | **≤25%** | ≤20% |
| Signed bias | within ±15% | **within ±10%** | within ±10% |
| Range coverage | ≥80% | **≥80%** | ≥80% |
| Time to labels | ≤1 s after tap | ≤3 s after sweep ends | ≤2 s |

The reasoning: a non-LiDAR phone combining monocular depth with plane scale is expected to reach roughly 20–35% per-item volume error, an estimate rather than a published figure. Density and recipe variance add about 10% even with perfect volume. **Coverage and bias matter more than MAPE**: published apps underestimate by about a third, so an honest, unbiased range already beats them.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sideloadly or free-account signing breaks after an iOS update | Find out in M0 on day 1. Fallbacks are xtool on WSL, or the web track only. |
| 7-day expiry interrupts use | Run Sideloadly auto-refresh on the same Wi-Fi, or re-sign weekly (one click). |
| iOS 26 WebGPU memory leak or ONNX failures in Safari | Keep the `wasm` fallback toggle, and run a 5-minute soak test in M2. |
| PWA standalone camera bugs (rotation, stream stops) | Use the Safari tab, not the home-screen PWA. Avoid route or hash changes. |
| No Taiwanese food dataset off the shelf | Your own photos plus SAM-assisted labelling. Start with 15 classes. |
| Monocular depth too noisy for flat foods | Container priors and class height priors as a backstop. Widen ranges for flat items. |
| Clear bubble-tea cups break depth | Never use depth for drinks. Use cup size × fill level × sugar level. |
| Hidden oil and braised fat underestimated | Oil adder intervals per cooking method, biased toward the high side. |
| AGPL-3.0 (Ultralytics) | Fine for personal use. Revisit before any distribution. |
| No debugger | In-app logs, crash file, sweep dumps and the Python replay tool. |

---

## 9. Verify first (unconfirmed research claims)

1. Sideloadly with a free Apple ID on iOS 26 in Windows. No 2026 breakage was found, but that is absence of evidence. **M0 settles it.**
2. Whether Ultralytics Core ML segment export with `nms=True` outputs decoded masks.
3. Whether YOLO11n-seg runs fast enough in Safari 26: the WebGPU vs WASM speed on your exact iPhone is unknown.
4. Latency of Depth Anything V2 Small on your iPhone. Apple's figures are 31–34 ms on a 12 Pro Max and 15 Pro Max.
5. Which prepared Taiwanese dishes the TFDA CSV actually contains.
6. The oil-adder numbers for stir-fry and braise, which currently have no citable source.

---

## 10. What changes if your hardware changes

- **You get a Mac:** use Xcode with direct USB install and a debugger. Keep the free 7-day signing. Track N iterates much faster (seconds instead of a 5–8 minute CI round trip). Nothing else changes.
- **You get a Pro iPhone (LiDAR):** swap the depth source to `.smoothedSceneDepth` and its confidence map, and drop the scale alignment. Tighten targets to the original column in §7. The rest of the pipeline is unchanged.
- **You pay the $99 developer fee:** get 1-year signing and TestFlight. Mostly a convenience.
