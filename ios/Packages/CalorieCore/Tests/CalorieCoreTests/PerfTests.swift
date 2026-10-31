// Release-mode timing of the native no-detector pipeline at device resolution (518 x 392, 12 keyframes), exactly the
// order SweepController.process runs it: align (no masks) -> HeightSegment -> align (with masks) -> Volume per item.
// Skipped unless CL_PERF=1; CI runs it with `swift test -c release --filter PerfTests`. The CI runner is an Apple
// M-series Mac; an A15/A16 phone core is roughly 1.5-2x slower single-threaded, so budget ~2x the printed numbers.
import Foundation
import XCTest
import simd
import CalorieCore

final class PerfTests: XCTestCase {
    func testNativePipelineAtDeviceResolution() throws {
        guard ProcessInfo.processInfo.environment["CL_PERF"] == "1" else {
            throw XCTSkip("set CL_PERF=1 (and build with -c release) to run the timing benchmark")
        }
        var cfg = SynthScene.Config()
        cfg.width = 518
        cfg.height = 392
        cfg.nFrames = 12
        cfg.featurePointsPerFrame = 150
        let scene = SynthScene.make(cfg)
        var frames = scene.frames

        func ms(_ t: Date) -> Double { Date().timeIntervalSince(t) * 1000 }

        let t0 = Date()
        guard let first = Align.alignFrames(&frames, planeHint: scene.plane) else {
            XCTFail("align #1 failed"); return
        }
        let align1 = ms(t0)

        let t1 = Date()
        let objs = HeightSegment.segment(&frames, plane: first.plane)
        let seg = ms(t1)

        let t2 = Date()
        guard let second = Align.alignFrames(&frames, planeHint: first.plane) else {
            XCTFail("align #2 failed"); return
        }
        let align2 = ms(t2)

        var lines: [String] = []
        var volTotal = 0.0
        for o in objs {
            let tv = Date()
            let r = Volume.computeItemVolume(frames: frames, itemKey: o.key, plane: second.plane, label: "unknown",
                                             container: nil, nBoot: 50)
            let dt = ms(tv)
            volTotal += dt
            let truth = simd_distance(SIMD2(o.centroidW.x, o.centroidW.z), SynthScene.moundCenterXZ) <
                simd_distance(SIMD2(o.centroidW.x, o.centroidW.z), SynthScene.boxCenterXZ)
                ? SynthScene.moundVolumeMl : SynthScene.boxVolumeMl
            let v = r.volumeMl
            lines.append(String(format: "  %@ %@ typ %.1f mL (%.1f-%.1f) truth %.1f mL  err %+.1f%%  %.0f ms",
                                o.key, r.method.rawValue, v?.typ ?? -1, v?.min ?? -1, v?.max ?? -1, truth,
                                v.map { 100 * ($0.typ - truth) / truth } ?? 0, dt))
            if let v = v { XCTAssertLessThan(abs(v.typ - truth) / truth, 0.25, "\(o.key) volume far from truth") }
        }

        let total = align1 + seg + align2 + volTotal
        print("---- PerfTests (518x392, 12 frames, nBoot 50) ----")
        print(String(format: "align#1 %.0f ms (%.1f ms/frame), segment %.0f ms, align#2 %.0f ms, volume %.0f ms, total %.0f ms",
                     align1, align1 / 12, seg, align2, volTotal, total))
        print("align ok: #1 \(first.diags.filter { $0.ok }.count)/12, #2 \(second.diags.filter { $0.ok }.count)/12; objects \(objs.count)")
        for l in lines { print(l) }
        XCTAssertEqual(objs.count, 2, "expected mound + box")
    }
}
