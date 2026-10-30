// Tests for PlaneFit and Align on the analytic SynthScene (see SynthScene.swift for the API).
import XCTest
import simd
import CalorieCore

fileprivate func alignTestUniform(_ rng: inout SplitMix64) -> Double {
    return Double(rng.next() >> 11) * (1.0 / 9007199254740992.0)
}

fileprivate func alignTestGauss(_ rng: inout SplitMix64) -> Double {
    let u1 = max(alignTestUniform(&rng), 1e-12)
    let u2 = alignTestUniform(&rng)
    return (-2.0 * log(u1)).squareRoot() * cos(2.0 * Double.pi * u2)
}

fileprivate func alignTestMedian(_ a: [Double]) -> Double {
    if a.isEmpty { return Double.infinity }
    let s = a.sorted()
    let m = s.count / 2
    return s.count % 2 == 1 ? s[m] : 0.5 * (s[m - 1] + s[m])
}

final class AlignTests: XCTestCase {

    // MARK: plane fit

    private func syntheticPlanePoints(n trueN: SIMD3<Double>, offset: Double, seed: UInt64) -> [SIMD3<Double>] {
        var rng = SplitMix64(seed: seed)
        let e = SIMD3<Double>(1, 0, 0)
        let a = simd_normalize(simd_cross(trueN, e))
        let b = simd_cross(trueN, a)
        var pts: [SIMD3<Double>] = []
        for _ in 0..<1500 {
            let u = (alignTestUniform(&rng) - 0.5) * 0.6
            let v = (alignTestUniform(&rng) - 0.5) * 0.6
            let h = 0.0015 * alignTestGauss(&rng)
            pts.append(a * u + b * v + trueN * (offset + h))
        }
        for _ in 0..<400 {       // "food" outliers 1..8 cm above the table
            let u = (alignTestUniform(&rng) - 0.5) * 0.1
            let v = (alignTestUniform(&rng) - 0.5) * 0.1
            let h = 0.01 + 0.07 * alignTestUniform(&rng)
            pts.append(a * u + b * v + trueN * (offset + h))
        }
        return pts
    }

    func testPlaneRecoveryFromNoisyPoints() {
        let trueN = simd_normalize(SIMD3<Double>(0.12, 0.98, -0.08))
        let offset = 0.35
        let pts = syntheticPlanePoints(n: trueN, offset: offset, seed: 42)
        let cams = [trueN * (offset + 0.4), trueN * (offset + 0.35) + SIMD3<Double>(0.1, 0, 0)]
        guard let p = PlaneFit.fitRANSAC(pts, camCentersW: cams, seed: 3) else { return XCTFail("no plane") }
        XCTAssertGreaterThan(simd_dot(p.n, trueN), cos(1.0 * Double.pi / 180.0))
        XCTAssertEqual(p.offset, offset, accuracy: 0.002)
        XCTAssertEqual(p.inlierRMSEm, 0.0015, accuracy: 0.0008)
        XCTAssertEqual(simd_length(p.n), 1.0, accuracy: 1e-9)

        // without camera centres the normal is oriented to +Y
        guard let q = PlaneFit.fitRANSAC(pts, seed: 4) else { return XCTFail("no plane") }
        XCTAssertGreaterThanOrEqual(q.n.y, 0)
        XCTAssertGreaterThan(simd_dot(q.n, trueN), cos(1.0 * Double.pi / 180.0))

        // a plane whose true normal points down is flipped up by the camera orientation
        let down = PlaneFit.fitRANSAC(pts.map { -$0 }, camCentersW: cams.map { -$0 }, seed: 3)
        XCTAssertNotNil(down)
        if let d = down {
            XCTAssertGreaterThan(simd_dot(d.n, -trueN), cos(1.0 * Double.pi / 180.0))
            XCTAssertEqual(d.offset, offset, accuracy: 0.002)
        }
    }

    func testPlaneFitDegenerateInput() {
        XCTAssertNil(PlaneFit.fitRANSAC([]))
        XCTAssertNil(PlaneFit.fitRANSAC([SIMD3<Double>(0, 0, 0), SIMD3<Double>(1, 0, 0)]))
        var line: [SIMD3<Double>] = []
        for i in 0..<50 { line.append(SIMD3<Double>(Double(i) * 0.01, 0.2, 0.1)) }
        XCTAssertNil(PlaneFit.fitRANSAC(line))
        let nan = [SIMD3<Double>(Double.nan, 0, 0), SIMD3<Double>(0, 1, 0), SIMD3<Double>(1, 1, 0)]
        XCTAssertNil(PlaneFit.fitRANSAC(nan))
    }

    // MARK: fixMaskEdges

    func testFixMaskEdgesReplacesRingByInteriorExtrapolation() {
        let W = 80
        let H = 60
        func clean(_ u: Int, _ v: Int) -> Float { return Float(10.0 + 0.05 * Double(u) + 0.02 * Double(v)) }
        var rel = FloatGrid(width: W, height: H, repeating: 0)
        var mask = Mask(width: W, height: H, repeating: false)
        for v in 0..<H { for u in 0..<W { rel[u, v] = clean(u, v) } }
        func isRing(_ u: Int, _ v: Int) -> Bool {
            return u < 22 || u >= 58 || v < 17 || v >= 43
        }
        for v in 15..<45 {
            for u in 20..<60 {
                mask[u, v] = true
                if isRing(u, v) { rel[u, v] += 5.0 }          // bleed
            }
        }
        rel[5, 5] = 99.0                                     // outside the mask: must stay
        let fixed = Align.fixMaskEdges(rel: rel, masks: ["food": mask], labels: ["food": "unknown"])
        for v in 0..<H {
            for u in 0..<W {
                if mask[u, v] {
                    XCTAssertEqual(Double(fixed[u, v]), Double(clean(u, v)), accuracy: 0.15, "pixel \(u),\(v)")
                } else {
                    XCTAssertEqual(fixed[u, v], rel[u, v])
                }
            }
        }
        // container masks are skipped exactly
        let kept = Align.fixMaskEdges(rel: rel, masks: ["bowl": mask], labels: ["bowl": "container_bowl"])
        XCTAssertEqual(kept.data, rel.data)
    }

    // MARK: alignFrames on the synthetic scene

    private func makeScene(masks: Bool, frames: Int = 6, mound: Bool = true, box: Bool = true) -> SynthScene {
        var cfg = SynthScene.Config()
        cfg.nFrames = frames
        cfg.includeMasks = masks
        cfg.withMound = mound
        cfg.withBox = box
        return SynthScene.make(cfg)
    }

    func testAlignRecoversDepthWithMasks() {
        let scene = makeScene(masks: true)
        var frames = scene.frames
        guard let res = Align.alignFrames(&frames, planeHint: scene.plane, seed: 5) else { return XCTFail("nil") }
        XCTAssertEqual(res.diags.count, frames.count)
        XCTAssertGreaterThan(simd_dot(res.plane.n, SIMD3<Double>(0, 1, 0)), 0.999)
        for (i, d) in res.diags.enumerated() {
            XCTAssertTrue(d.ok, "frame \(i): \(d.reason)")
            XCTAssertLessThan(d.rmseM ?? 1.0, 0.004)
            XCTAssertGreaterThan(d.nBg, 100)
            XCTAssertGreaterThan(d.nFp, 10)
            guard let depth = frames[i].depthM else { XCTFail("no depth \(i)"); continue }
            let table = scene.interiorErrors(frame: i, id: SynthScene.idTable, erodePx: 8, depth: depth)
            let mound = scene.interiorErrors(frame: i, id: SynthScene.idMound, erodePx: 4, depth: depth)
            let box = scene.interiorErrors(frame: i, id: SynthScene.idBox, erodePx: 3, depth: depth)
            XCTAssertGreaterThan(table.count, 1000)
            XCTAssertLessThan(alignTestMedian(table), 0.002, "table frame \(i)")
            XCTAssertGreaterThan(mound.count, 20)
            XCTAssertLessThan(alignTestMedian(mound), 0.003, "mound frame \(i)")
            XCTAssertGreaterThan(box.count, 0)
            XCTAssertLessThan(alignTestMedian(box), 0.003, "box frame \(i)")
        }
    }

    func testAlignEstimatesPlaneAndWorksWithoutMasks() {
        let scene = makeScene(masks: false)
        var frames = scene.frames
        guard let res = Align.alignFrames(&frames, planeHint: nil, seed: 7) else { return XCTFail("no plane") }
        XCTAssertGreaterThan(simd_dot(res.plane.n, SIMD3<Double>(0, 1, 0)), cos(1.0 * Double.pi / 180.0))
        XCTAssertEqual(res.plane.offset, 0.0, accuracy: 0.003)
        for (i, d) in res.diags.enumerated() {
            XCTAssertTrue(d.ok, "frame \(i): \(d.reason)")
            guard let depth = frames[i].depthM else { XCTFail("no depth \(i)"); continue }
            let table = scene.interiorErrors(frame: i, id: SynthScene.idTable, erodePx: 8, depth: depth)
            XCTAssertLessThan(alignTestMedian(table), 0.003, "table frame \(i)")
        }
        // deterministic for a fixed seed
        var again = scene.frames
        _ = Align.alignFrames(&again, planeHint: nil, seed: 7)
        XCTAssertEqual(again[0].depthM?.data, frames[0].depthM?.data)
    }

    func testAlignFramesWithoutPlaneOrFeaturePointsReturnsNil() {
        let scene = makeScene(masks: false, frames: 2)
        var frames = scene.frames
        for i in 0..<frames.count { frames[i].featurePointsW = [] }
        XCTAssertNil(Align.alignFrames(&frames, planeHint: nil))
    }

    func testAlignFramesOrientsHintUp() {
        let scene = makeScene(masks: true, frames: 2)
        var frames = scene.frames
        guard let res = Align.alignFrames(&frames, planeHint: scene.plane.flipped, seed: 1) else { return XCTFail("nil") }
        XCTAssertGreaterThan(res.plane.n.y, 0.99)
        XCTAssertEqual(res.plane.offset, 0.0, accuracy: 1e-12)
        XCTAssertTrue(res.diags.allSatisfy { $0.ok })
    }

    // MARK: rejection gates

    func testRejectsConstantRelativeDepth() {
        let scene = makeScene(masks: false, frames: 1)
        var f = scene.frames[0]
        f.relDepth = FloatGrid(width: f.width, height: f.height, repeating: 5.0)
        var rng = SplitMix64(seed: 1)
        let d = Align.alignFrame(&f, plane: scene.plane, rng: &rng)
        XCTAssertFalse(d.ok)
        XCTAssertTrue(d.reason.contains("too little"), d.reason)
        XCTAssertNil(f.depthM)
        XCTAssertNotNil(d.relSpread)
    }

    func testRejectsFrameWithoutAnchors() {
        let scene = makeScene(masks: false, frames: 1)
        var f = scene.frames[0]
        f.featurePointsW = []
        var rng = SplitMix64(seed: 1)
        let far = Plane(n: SIMD3<Double>(0, 1, 0), offset: -5.0)     // table "5 m below the camera": no usable anchors
        let d = Align.alignFrame(&f, plane: far, rng: &rng)
        XCTAssertFalse(d.ok)
        XCTAssertTrue(d.reason.hasPrefix("too few anchors"), d.reason)
        XCTAssertNil(f.depthM)
    }

    func testRejectsUncorrelatedRelativeDepth() {
        let scene = makeScene(masks: false, frames: 1)
        var f = scene.frames[0]
        var rng = SplitMix64(seed: 99)
        var g = f.relDepth
        for i in 0..<g.data.count { g.data[i] = Float(alignTestUniform(&rng) * 10.0) }
        f.relDepth = g
        var rng2 = SplitMix64(seed: 2)
        let d = Align.alignFrame(&f, plane: scene.plane, rng: &rng2)
        XCTAssertFalse(d.ok, "random rel must not be accepted")
        XCTAssertFalse(d.reason.isEmpty)
        XCTAssertNil(f.depthM)
    }

    func testRobustAffineFitRejectsOutliersAndKeepsPositiveScale() {
        var rng = SplitMix64(seed: 11)
        var x: [Double] = []
        var y: [Double] = []
        var w: [Double] = []
        for i in 0..<400 {
            let xi = 5.0 + 5.0 * alignTestUniform(&rng)
            var yi = 0.2 * xi + 1.0                       // disparity 2..3 1/m
            if i % 5 == 0 { yi += 0.6 }                   // 20 % gross outliers
            x.append(xi); y.append(yi); w.append(1.0)
        }
        guard let fit = Align.robustAffineFit(x: x, y: y, w: w, rng: &rng) else { return XCTFail("nil") }
        XCTAssertEqual(fit.s, 0.2, accuracy: 0.005)
        XCTAssertEqual(fit.t, 1.0, accuracy: 0.05)
        XCTAssertGreaterThan(fit.inliers.filter { $0 }.count, 300)
        XCTAssertNil(Align.robustAffineFit(x: [1, 2], y: [1, 2], w: [1, 1], rng: &rng))
    }
}
