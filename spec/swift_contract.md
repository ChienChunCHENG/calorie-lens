# Swift contract (Wave 4, native lens M4)

Binding for every Swift agent. Types live in `ios/Packages/CalorieCore/Sources/CalorieCore/Types.swift`
(Opus-owned; do not edit, ask in your report if something is missing). Conventions = `tools/replay/cl_types.py`.

Nobody can compile locally (Windows, no Mac). CI compiles on a GitHub macOS runner with Xcode 26:
`swift test --package-path ios/Packages/CalorieCore` (macOS host, arm64), then `xcodegen` + `xcodebuild` for the app.
So: write conservative Swift 5.9 (language mode 5), XCTest, Foundation/simd/Accelerate/ImageIO/CoreGraphics only in the
package. No macros, no `@Observable`, no Swift Testing, no third-party packages. Every public symbol below must exist
with exactly this signature (others may add private helpers and extra public API freely). Mark mutable shared state
`@MainActor` or keep it in value types; avoid `Sendable` warnings becoming errors (do not enable strict concurrency).

## Layout and ownership

| Path | Owner |
|---|---|
| `ios/Packages/CalorieCore/Package.swift`, `Sources/CalorieCore/Types.swift`, `ios/project.yml`, `.github/workflows/ios.yml` | Opus |
| `Sources/CalorieCore/RangeEngine.swift`, `NutritionData.swift`, `Tests/CalorieCoreTests/RangeEngineTests.swift`, `NutritionDataTests.swift` | S1 |
| `Sources/CalorieCore/PlaneFit.swift`, `Align.swift`, `HeightSegment.swift`, `Tests/.../AlignTests.swift`, `HeightSegmentTests.swift`, `Tests/.../SynthScene.swift` | S2 |
| `Sources/CalorieCore/Volume.swift`, `Tests/.../VolumeTests.swift` | S3 |
| `Sources/CalorieCore/DumpIO.swift`, `Tests/.../DumpIOTests.swift`, `GoldenReplayTests.swift`, `tools/replay/export_fixtures.py` | S4 |
| `ios/Sources/CalorieLens/ARSessionController.swift`, `ARViewContainer.swift`, `SweepController.swift`, `DepthEstimator.swift`, `Detector.swift` | S5 |
| `ios/Sources/CalorieLens/ContentView.swift`, `MealModel.swift`, `LabelOverlay.swift`, `ItemSheet.swift`, `MealBar.swift` | S6 |

Tests that need repo files locate the repo root from `#filePath` (walk up until a directory containing `spec/` exists).
The golden fixtures are generated in CI by `py tools/replay/export_fixtures.py --out "$CL_FIXTURES"` before
`swift test`; tests needing them `throw XCTSkip` when the env var `CL_FIXTURES` is unset.

## S1: range engine + nutrition data (mirror `core/ts/rangeEngine.ts`, `web/src/meal.ts`)

```swift
public enum SugarLevel: String, CaseIterable, Codable { case full, less, half, light, none }
public enum Ice: String, CaseIterable, Codable { case normal, less, none }
public struct OilRange: Equatable, Codable { public var min: Double; public var max: Double }
public struct DishEntry { public var id: String; public var kind: String /* "solid"|"drink" */
    public var densityGml: Triple?; public var kcal100g: Triple; public var oilGPer100g: OilRange?; public var defaultPortionG: Triple? }
public struct DrinkEntry { public var id: String; public var baseKcalPer700ml: Triple; public var toppingsKcal: Triple }
public struct ItemEstimate: Equatable { public var id: String; public var lo: Double; public var med: Double; public var hi: Double; public var flags: [String] }
public struct MealRange: Equatable { public var lo: Double; public var med: Double; public var hi: Double }
public enum RangeEngineError: Error { case invalid(String) }
public enum RangeEngine {
    public static func sigmaOf(_ mn: Double, _ mx: Double) throws -> Double
    public static func estimateSolid(_ dish: DishEntry, volumeMl: Triple?) throws -> ItemEstimate
    public static func estimateDrink(_ drink: DrinkEntry, sizeMl: Double, sugar: SugarLevel,
                                     fullSugarGPer700: Double, sugarFraction: [String: Double]) throws -> ItemEstimate
    public static func combineMeal(_ items: [ItemEstimate]) -> MealRange
    public static func round5(_ x: Double) -> Double
    public static func formatRange(_ lo: Double, _ hi: Double) -> String   // "1310–2105" (en dash)
}
public struct DishInfo { public var entry: DishEntry; public var zh: String; public var en: String; public var containerPrior: [String] }
public struct ContainerInfo { public var id: String; public var shape: String; public var zh: String; public var capacityMl: Triple
    public var rimDiameterCm: Triple?; public var depthCm: Triple?
    public var spec: ContainerSpec? }       // ContainerSpec for the volume step (nil for unknown shapes)
public struct MealData {
    public var dishes: [String: DishInfo]; public var dishOrder: [String]; public var containers: [String: ContainerInfo]
    public var drinks: [String: DrinkEntry]; public var fullSugarGPer700: Double; public var sugarLevels: [String: Double]
    public static func load(jsonData: Data) throws -> MealData     // data/nutrition.json
}
public struct MealItemChoice: Equatable { public var dishId: String; public var containerId: String; public var sugar: SugarLevel; public var ice: Ice }
public enum MealLogic {
    public static let unknownContainer = "unknown"
    public static let fill = Triple(min: 0.6, typ: 0.8, max: 1.0)
    public static let cupMl: [String: Double] = ["cup_500": 500, "cup_700": 700]
    public static func volumeFor(containerId: String, data: MealData) -> Triple?            // web volumeFor
    public static func containerChoices(data: MealData, dishId: String, current: String?) -> [String]
    public static func defaultChoice(data: MealData, dishId: String, preferBowl: Bool) -> MealItemChoice
    /// measuredVolumeMl (native depth) wins over the container prior for solids; drinks ignore it.
    public static func estimate(data: MealData, choice: MealItemChoice, measuredVolumeMl: Triple?) throws -> ItemEstimate
    public static func safeEstimate(data: MealData, choice: MealItemChoice, measuredVolumeMl: Triple?) -> ItemEstimate?
}
```
Tests: every case of `spec/vectors.json` (same tolerances as `core/ts/rangeEngine.test.ts`), plus MealData loads
`data/nutrition.json` and reproduces web numbers (small-bowl rice median 328, small 滷肉飯 451, 700 mL 珍奶 half 584).

## S2: plane fit, alignment, class-agnostic segmentation (port `plane.py`, `align.py`)

```swift
public enum PlaneFit {
    public static func fitRANSAC(_ points: [SIMD3<Double>], threshM: Double = 0.005, iters: Int = 500,
                                 camCentersW: [SIMD3<Double>]? = nil, seed: UInt64 = 0) -> Plane?
}
public enum Align {
    public static func fixMaskEdges(rel: FloatGrid, masks: [String: Mask], labels: [String: String]) -> FloatGrid
    public static func alignFrame(_ f: inout SweepFrame, plane: Plane, rng: inout SplitMix64) -> AlignDiag
    /// Orients planeHint up (or fits one from feature points); fills depthM in place. nil if no plane obtainable.
    public static func alignFrames(_ frames: inout [SweepFrame], planeHint: Plane?, seed: UInt64 = 0) -> (plane: Plane, diags: [AlignDiag])?
}
public struct SegmentedObject { public var key: String; public var centroidW: SIMD3<Double>; public var topW: SIMD3<Double>
    public var footprintAreaM2: Double; public var maxHeightM: Double; public var nFrames: Int }
public enum HeightSegment {
    /// Needs aligned frames (depthM). Pixels with height above the plane > minHeightM form 4-connected components
    /// (after a 1-px open); components >= minAreaM2 (plane-projected) are clustered ACROSS frames by plane-coordinate
    /// centroid distance < mergeDistM; clusters seen in >= max(2, 30% of aligned frames) frames become objects
    /// "obj0", "obj1", ... (sorted by area, largest first). Writes their masks into frames[i].masks with label "unknown".
    public static func segment(_ frames: inout [SweepFrame], plane: Plane, minHeightM: Double = 0.006,
                               minAreaM2: Double = 0.0009, mergeDistM: Double = 0.03) -> [SegmentedObject]
}
```
Native flow without a detector: `alignFrames` with no masks (RANSAC treats food as outliers) -> `segment` ->
`alignFrames` again (now masks exclude food from the anchors) -> `Volume`. `Tests/.../SynthScene.swift` is a tiny Swift
renderer (flat table + spherical-cap mound + box, analytic ray casting, a ring of 8 cameras, rel = 1/d affine-scrambled
+ noise) shared by the S2/S3 tests; keep its public-to-tests API documented at the top of the file.

## S3: volume (port `volume.py` exactly per `tools/replay/VOLUME_ALGO.md`)

```swift
public enum Volume {
    public static let drinkLabels: Set<String>     // bubble_milk_tea, container_cup, black_tea, green_tea, soy_milk
    public static func computeItemVolume(frames: [SweepFrame], itemKey: String, plane: Plane, label: String,
                                         container: ContainerSpec?, cellM: Double = 0.003, minObs: Int = 3,
                                         nBoot: Int = 50, seed: UInt64 = 0) -> VolumeResult
}
```
Bowl centre from the container mask (`container.itemKey`, or any `container_bowl`-labelled mask), defaults as Python.
Performance budget: <= 1.5 s on an A15 for 12 frames 518x392, 50 bootstraps (precompute per-frame cell samples once,
bootstrap only re-fuses).

## S4: dumps + golden replay

```swift
public struct DumpMeta { public var deviceModel: String?; public var iosVersion: String?; public var imageSize: [Int]?
    public var depthModel: String?; public var plane: Plane?; public var planeSource: String?
    public var containers: [String: ContainerSpec]; public init() }
public enum DumpIO {
    public static func write(frames: [SweepFrame], meta: DumpMeta, rgbJPEGs: [Data?]?, to dir: URL) throws
    public static func read(_ dir: URL) throws -> (frames: [SweepFrame], meta: DumpMeta)     // directory only
    public static func zip(_ dir: URL) throws -> URL      // NSFileCoordinator .forUploading, returns a .zip in tmp
}
```
Exactly `tools/replay/DUMP_FORMAT.md` (Float16 little-endian rel, 8-bit PNG masks, f32 feature points, row-major
T_wc/K). Python `dump_io.load_dump` must read what Swift writes and vice versa. `export_fixtures.py` writes synth dumps
(`mound`, `egg`, `cutlet`, `bowl_dome`, `meal`, seed 0, plus a no-plane-hint variant) and `expected.json` with ground
truth and Python's own align+volume result per item. `GoldenReplayTests` runs Swift align+volume on each dump and checks
it against both (tolerances justified in comments; Swift and numpy RNGs differ).

## S5: app capture pipeline (ARKit, Core ML)

```swift
struct MeasuredItem: Identifiable { let id: String /*item key*/; var label: String? /*detector class id*/
    var centroidW: SIMD3<Float>; var topW: SIMD3<Float>; var volume: VolumeResult }
@MainActor final class SweepController: ObservableObject {
    enum Phase: Equatable { case idle, needPlane, sweeping(progress: Double), processing, done, failed(String) }
    @Published private(set) var phase: Phase
    @Published private(set) var guidance: String
    @Published private(set) var items: [MeasuredItem]
    @Published private(set) var lastDumpURL: URL?          // zip, ready for ShareLink
    init(session: ARSessionController)
    func start(); func cancel()
    func remeasure(itemKey: String, container: ContainerSpec?) async -> VolumeResult?   // re-run Volume on kept frames
}
// ARSessionController additions (S5 owns it): `var currentCamera: ARCamera?` (main queue, never retain ARFrame),
// `@Published private(set) var frameTick: Int` (throttled to <= 30 Hz), `weak var sweep: SweepController?` frame tap.
```
Everything in the SENSOR (landscape) orientation of `capturedImage`: depth model input, masks, K scaled by
depth_w/img_w, depth_h/img_h. Plane hint from the horizontal `ARPlaneAnchor` under the screen centre (normal = anchor
y axis, oriented up). Keyframes: tracking `.normal`, >= 8 deg view change or >= 4 cm move, 8-12 frames, about 5 s.
Depth: `DepthAnythingV2SmallF16.mlmodelc` loaded by URL from the bundle (absent model => phase `.failed` with a clear
message, never a crash); verify on device that output is disparity-like (log the sign of corr(rel, 1/d_featurepoints)).
Detector: optional `CalorieLensDetector.mlmodelc` + `labels.json`; absent => HeightSegment path. Each sweep is written
with DumpIO (no RGB by default; a debug toggle adds keyframe JPEGs) so `tools/replay/run_replay.py` can replay it.
Heavy work off the main actor; log every step to `LogStore.shared`.

## S6: app UI (mirror the web lens)

`MealModel` (ObservableObject, @MainActor): one entry per `MeasuredItem` (+ manually added drinks), each with a
`MealItemChoice?`; estimate via `MealLogic.safeEstimate(..., measuredVolumeMl: item.volume.volumeMl)`; meal total via
`RangeEngine.combineMeal`. Loads `nutrition.json` from the app bundle (`Bundle.main.url(forResource: "nutrition", withExtension: "json")`).
`LabelOverlay`: SwiftUI labels at `currentCamera.projectPoint(topW, orientation: .portrait, viewportSize:)`, refreshed
on `frameTick`, hidden when behind the camera; shows 中文 name + `formatRange` kcal (or "tap to label"), flags as small
badges ("portion_default", "low_confidence"). Tap opens `ItemSheet` (dish / container / 甜度 / 冰塊 chips, the same
rules as `web/src/sheet.ts`; choosing a bowl container calls `sweep.remeasure` with the container's `spec`).
`MealBar`: Scan button (drives `SweepController`), meal total range, "no numbers" toggle, add-drink button, share dump,
Logs. Disclaimer text as in the web lens. Keep the existing HUD/logs/crash alert working.
