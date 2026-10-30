// Volume tests on exact analytic metric depth (own tiny ray caster, independent of Align / SynthScene).
// Scene: y-up table plane y = 0, ring of 8 cameras (radius 0.2 m, height 0.3 m) looking at (0, 0.01, 0),
// 160x120 depth maps, f = 200 px. Reference numbers were cross-checked against tools/replay/volume.py on the same scene
// (box 47.9 vs 48.0 mL, mound 54.8 vs 54.5 mL, bowl 230 vs 242.8 mL).
import XCTest
import simd
import CalorieCore

private typealias VolCaster = (_ o: SIMD3<Double>, _ d: SIMD3<Double>) -> Double?

private enum VolTestScene {
    static let width = 160
    static let height = 120
    static let focal = 200.0
    static let plane = Plane(n: SIMD3<Double>(0, 1, 0), offset: 0.0)

    static func cameraT(_ i: Int, of n: Int) -> simd_double4x4 {
        let ang = 2.0 * Double.pi * Double(i) / Double(n)
        let pos = SIMD3<Double>(0.2 * cos(ang), 0.3, 0.2 * sin(ang))
        let tgt = SIMD3<Double>(0.0, 0.01, 0.0)
        let z = simd_normalize(pos - tgt)
        let x = simd_normalize(simd_cross(SIMD3<Double>(0.0, 1.0, 0.0), z))
        let y = simd_cross(z, x)
        return simd_double4x4(columns: (SIMD4<Double>(x.x, x.y, x.z, 0.0),
                                        SIMD4<Double>(y.x, y.y, y.z, 0.0),
                                        SIMD4<Double>(z.x, z.y, z.z, 0.0),
                                        SIMD4<Double>(pos.x, pos.y, pos.z, 1.0)))
    }

    /// Axis-aligned box 6 x 4 x 2 cm (x, z, height) standing on the table, centred at the origin. 48 mL.
    static func box(_ o: SIMD3<Double>, _ d: SIMD3<Double>) -> Double? {
        let lo = SIMD3<Double>(-0.03, 0.0, -0.02)
        let hi = SIMD3<Double>(0.03, 0.02, 0.02)
        var tn = -Double.infinity
        var tfar = Double.infinity
        for a in 0..<3 {
            let t1 = (lo[a] - o[a]) / d[a]
            let t2 = (hi[a] - o[a]) / d[a]
            tn = max(tn, min(t1, t2))
            tfar = min(tfar, max(t1, t2))
        }
        if tn <= tfar && tn > 0 { return tn }
        return nil
    }

    static func sphere(_ o: SIMD3<Double>, _ d: SIMD3<Double>, centre c: SIMD3<Double>, radius r: Double,
                       accept: (SIMD3<Double>) -> Bool) -> Double? {
        let oc = o - c
        let a = simd_dot(d, d)
        let b = 2.0 * simd_dot(d, oc)
        let cc = simd_dot(oc, oc) - r * r
        let disc = b * b - 4.0 * a * cc
        if disc <= 0 { return nil }
        let t = (-b - disc.squareRoot()) / (2.0 * a)
        if t <= 0 { return nil }
        return accept(o + d * t) ? t : nil
    }

    /// Spherical-cap mound: base radius 4 cm, height 2 cm (R = 5 cm).
    static func mound(_ o: SIMD3<Double>, _ d: SIMD3<Double>) -> Double? {
        let r = 0.05
        return sphere(o, d, centre: SIMD3<Double>(0.0, 0.02 - r, 0.0), radius: r, accept: { $0.y >= 0.0 })
    }
    static let moundTruthMl = Double.pi * 0.02 * (3.0 * 0.04 * 0.04 + 0.02 * 0.02) / 6.0 * 1.0e6

    /// Food heap in a bowl cavity: dome of radius 15 cm, top at 5.2 cm, food radius 5 cm.
    static let domeR = 0.15
    static let domeTop = 0.052
    static func dome(_ o: SIMD3<Double>, _ d: SIMD3<Double>) -> Double? {
        let yc = domeTop - domeR
        return sphere(o, d, centre: SIMD3<Double>(0.0, yc, 0.0), radius: domeR,
                      accept: { $0.y >= yc && ($0.x * $0.x + $0.z * $0.z).squareRoot() <= 0.05 })
    }
    /// Rim disc (radius 6 cm at 5.5 cm height) standing in for the bowl body seen by the container mask.
    static func lid(_ o: SIMD3<Double>, _ d: SIMD3<Double>) -> Double? {
        if !(d.y < 0) { return nil }
        let t = (0.055 - o.y) / d.y
        if t <= 0 { return nil }
        let p = o + d * t
        return (p.x * p.x + p.z * p.z).squareRoot() <= 0.06 ? t : nil
    }
    static func bowlTruthMl() -> Double {
        let a = 0.06, depth = 0.05, base = 0.005
        let rs = (a * a + depth * depth) / (2.0 * depth)
        let step = 0.0005
        var sum = 0.0
        for i in 0..<250 {
            let x = -0.0625 + (Double(i) + 0.5) * step
            for j in 0..<250 {
                let z = -0.0625 + (Double(j) + 0.5) * step
                let r = (x * x + z * z).squareRoot()
                if r > 0.05 { continue }
                let top = domeTop - (domeR - (domeR * domeR - r * r).squareRoot())
                let rr = min(r, a)
                let fl = base + rs - max(rs * rs - rr * rr, 0.0).squareRoot()
                sum += max(top - fl, 0.0)
            }
        }
        return sum * step * step * 1.0e6
    }

    static func frames(food: VolCaster, lid: VolCaster? = nil, withDepth: Bool = true) -> [SweepFrame] {
        var out: [SweepFrame] = []
        let K = Intrinsics(fx: focal, fy: focal, cx: Double(width) / 2.0, cy: Double(height) / 2.0)
        for i in 0..<8 {
            let T = cameraT(i, of: 8)
            let o = SIMD3<Double>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
            let ax = SIMD3<Double>(T.columns.0.x, T.columns.0.y, T.columns.0.z)
            let ay = SIMD3<Double>(T.columns.1.x, T.columns.1.y, T.columns.1.z)
            let az = SIMD3<Double>(T.columns.2.x, T.columns.2.y, T.columns.2.z)
            var depth = FloatGrid(width: width, height: height, repeating: Float.nan)
            var foodMask = Mask(width: width, height: height, repeating: false)
            var contMask = Mask(width: width, height: height, repeating: false)
            for v in 0..<height {
                for u in 0..<width {
                    let dx = (Double(u) - K.cx) / focal
                    let dy = -(Double(v) - K.cy) / focal
                    let d = ax * dx + ay * dy + az * (-1.0)
                    var tTable = Double.infinity
                    if d.y < 0 { tTable = -o.y / d.y }
                    let tf = food(o, d)
                    var tl: Double? = nil
                    if let l = lid { tl = l(o, d) }
                    var best = tTable
                    if let t = tf {
                        best = t
                        foodMask[u, v] = true
                        contMask[u, v] = true
                    } else if let t = tl {
                        best = min(tTable, t)
                        contMask[u, v] = true
                    }
                    depth[u, v] = Float(best)
                }
            }
            var masks: [String: Mask] = ["food": foodMask]
            var labels: [String: String] = ["food": "unknown"]
            if lid != nil {
                masks["bowl"] = contMask
                labels["bowl"] = "container_bowl"
            }
            out.append(SweepFrame(t: Double(i) * 0.5, T_wc: T, K: K,
                                  relDepth: FloatGrid(width: width, height: height, repeating: 0),
                                  depthM: withDepth ? depth : nil, masks: masks, labels: labels))
        }
        return out
    }
}

final class VolumeTests: XCTestCase {

    private func checkHeightmap(_ r: VolumeResult, truth: Double, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(r.method, .heightmap, file: file, line: line)
        XCTAssertGreaterThan(r.coverage, 0.8, file: file, line: line)
        XCTAssertGreaterThan(r.nCells, 100, file: file, line: line)
        guard let v = r.volumeMl else {
            XCTFail("no volume; notes: \(r.notes)", file: file, line: line)
            return
        }
        XCTAssertEqual(v.typ, truth, accuracy: 0.10 * truth, file: file, line: line)
        XCTAssertLessThanOrEqual(v.min, truth, file: file, line: line)
        XCTAssertGreaterThanOrEqual(v.max, truth, file: file, line: line)
        XCTAssertLessThan(v.min, v.typ, file: file, line: line)
        XCTAssertGreaterThan(v.max, v.typ, file: file, line: line)
    }

    func testBoxHeightmap() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.box(o, d) })
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: nil, nBoot: 10)
        checkHeightmap(r, truth: 48.0)
    }

    func testMoundHeightmap() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.mound(o, d) })
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: nil, nBoot: 10)
        checkHeightmap(r, truth: VolTestScene.moundTruthMl)
    }

    func testPlateContainerAddsNote() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.box(o, d) })
        let spec = ContainerSpec(shape: .plate)
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: spec, nBoot: 10)
        XCTAssertEqual(r.method, .heightmap)
        XCTAssertTrue(r.notes.contains("container shape plate: plain heightmap"), "\(r.notes)")
    }

    func testBowlCavity() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.dome(o, d) },
                                         lid: { o, d in VolTestScene.lid(o, d) })
        let spec = ContainerSpec(shape: .bowl, itemKey: "bowl",
                                 rimDiameterCm: Triple(min: 11.5, typ: 12.0, max: 12.5),
                                 depthCm: Triple(min: 4.5, typ: 5.0, max: 5.5))
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: spec, nBoot: 10)
        XCTAssertEqual(r.method, .bowlCavity)
        guard let v = r.volumeMl else {
            XCTFail("no volume; notes: \(r.notes)")
            return
        }
        let truth = VolTestScene.bowlTruthMl()
        XCTAssertEqual(v.typ, truth, accuracy: 0.15 * truth)
        XCTAssertLessThan(v.min, v.typ)
        XCTAssertGreaterThan(v.max, v.typ)
        XCTAssertGreaterThan(r.coverage, 0.8)
    }

    func testBowlWithoutDimsUsesLabelAndEstimatesRim() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.dome(o, d) },
                                         lid: { o, d in VolTestScene.lid(o, d) })
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: nil, nBoot: 10)
        XCTAssertEqual(r.method, .bowlCavity)
        XCTAssertTrue(r.notes.contains("rim diameter estimated from container mask"), "\(r.notes)")
        XCTAssertNotNil(r.volumeMl)
    }

    func testDrinkLabelIsCupPrior() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.box(o, d) })
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "bubble_milk_tea", container: nil, nBoot: 10)
        XCTAssertEqual(r.method, .cupPrior)
        XCTAssertNil(r.volumeMl)
        XCTAssertEqual(r.notes, ["depth unreliable through clear plastic"])
        let r2 = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                          label: "unknown", container: ContainerSpec(shape: .cup), nBoot: 10)
        XCTAssertEqual(r2.method, .cupPrior)
        XCTAssertNil(r2.volumeMl)
    }

    func testNoDepthFails() {
        let frames = VolTestScene.frames(food: { o, d in VolTestScene.box(o, d) }, withDepth: false)
        let r = Volume.computeItemVolume(frames: frames, itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: nil, nBoot: 10)
        XCTAssertEqual(r.method, .failed)
        XCTAssertNil(r.volumeMl)
        XCTAssertEqual(r.notes, ["no frames with depth and mask"])
    }

    func testSingleFrameUsesDefaultSpread() {
        let all = VolTestScene.frames(food: { o, d in VolTestScene.box(o, d) })
        let r = Volume.computeItemVolume(frames: [all[0]], itemKey: "food", plane: VolTestScene.plane,
                                         label: "unknown", container: nil, minObs: 1, nBoot: 10)
        XCTAssertTrue(r.notes.contains("single frame: default spread used"), "\(r.notes)")
    }
}
