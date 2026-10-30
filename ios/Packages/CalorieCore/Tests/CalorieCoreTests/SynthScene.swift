// Tiny analytic renderer shared by the S2 / S3 tests (test target only).
//
// API (all internal to the test target):
//   var cfg = SynthScene.Config()          // width 160, height 120, nFrames 8, withMound, withBox, includeMasks, ...
//   let scene = SynthScene.make(cfg)
//   scene.frames        [SweepFrame]  relDepth = A*(1/d) + B + noise (per-frame random A in 8...14, B in -6...6,
//                                     noise std relNoise*A); depthM nil; featurePointsW = random visible surface points
//                                     (table and objects) with featureNoiseM Gaussian noise; masks only if
//                                     cfg.includeMasks ("mound" / "box", label "unknown")
//   scene.truthDepth    [FloatGrid]   exact metric z-depth per frame (distance along the viewing axis)
//   scene.truthId       [Grid<UInt8>] 0 table, 1 mound, 2 box (SynthScene.idTable / idMound / idBox)
//   scene.plane         Plane         y-up world, table at y = 0: n = (0, 1, 0), offset 0
//   scene.interiorErrors(frame:id:erodePx:depth:)  |depth - truth| over pixels whose (2*erodePx+1)^2 neighbourhood
//                                     lies fully inside region `id` (use id 0 for the table away from objects/borders)
//   SynthScene.moundTopAreaM2 / moundVolumeMl / boxAreaM2 / boxVolumeMl and centres: ground truth.
//
// Geometry (metres, y up, ARKit conventions as in Types.swift): table plane y = 0; spherical-cap mound centred at
// (-0.07, 0) with base radius 4.5 cm and height 3 cm; axis-aligned box centred at (0.08, 0) of 6 x 5 cm and 4 cm
// height. Ring of cameras at 0.4 m from (0, 0.01, 0), looking down 50 degrees, azimuth offset 0.3 steps,
// horizontal fov 60 degrees, K at the 160x120 render resolution.
import Foundation
import simd
import CalorieCore

fileprivate func synthUniform(_ rng: inout SplitMix64) -> Double {
    return Double(rng.next() >> 11) * (1.0 / 9007199254740992.0)
}

fileprivate func synthGauss(_ rng: inout SplitMix64) -> Double {
    let u1 = max(synthUniform(&rng), 1e-12)
    let u2 = synthUniform(&rng)
    return (-2.0 * log(u1)).squareRoot() * cos(2.0 * Double.pi * u2)
}

struct SynthScene {
    struct Config {
        var width = 160
        var height = 120
        var nFrames = 8
        var withMound = true
        var withBox = true
        var includeMasks = false
        var relNoise = 0.004            // std of the noise in disparity units (1/m), before the affine scramble
        var featurePointsPerFrame = 80
        var featureNoiseM = 0.001
        var radiusM = 0.4
        var elevationDeg = 50.0
        var hfovDeg = 60.0
        var seed: UInt64 = 1
    }

    static let idTable: UInt8 = 0
    static let idMound: UInt8 = 1
    static let idBox: UInt8 = 2

    static let moundCenterXZ = SIMD2<Double>(-0.07, 0.0)
    static let moundBaseRadiusM = 0.045
    static let moundHeightM = 0.03
    static let boxCenterXZ = SIMD2<Double>(0.08, 0.0)
    static let boxHalfXZ = SIMD2<Double>(0.03, 0.025)
    static let boxHeightM = 0.04

    static var moundTopAreaM2: Double { Double.pi * moundBaseRadiusM * moundBaseRadiusM }
    static var moundVolumeMl: Double {
        let a = moundBaseRadiusM
        let h = moundHeightM
        return Double.pi * h * (3 * a * a + h * h) / 6.0 * 1e6
    }
    static var boxAreaM2: Double { 4.0 * boxHalfXZ.x * boxHalfXZ.y }
    static var boxVolumeMl: Double { boxAreaM2 * boxHeightM * 1e6 }

    var config: Config
    var plane: Plane
    var frames: [SweepFrame]
    var truthDepth: [FloatGrid]
    var truthId: [Grid<UInt8>]

    // MARK: ray casting

    /// Nearest hit of origin + t * dir (dir unnormalised, camera-style so that t is the z-depth).
    static func cast(origin o: SIMD3<Double>, dir d: SIMD3<Double>, withMound: Bool, withBox: Bool) -> (t: Double, id: UInt8)? {
        var bestT = Double.infinity
        var bestId = idTable
        var hit = false
        // table y = 0
        if d.y < -1e-12 {
            let t = -o.y / d.y
            if t > 0 { bestT = t; bestId = idTable; hit = true }
        }
        // spherical cap: sphere radius Rs, centre (cx, h - Rs, cz); only the part with y >= 0 exists
        if withMound {
            let a = moundBaseRadiusM
            let h = moundHeightM
            let rs = (a * a + h * h) / (2.0 * h)
            let c = SIMD3<Double>(moundCenterXZ.x, h - rs, moundCenterXZ.y)
            let oc = o - c
            let qa = simd_dot(d, d)
            let qb = simd_dot(oc, d)
            let qc = simd_dot(oc, oc) - rs * rs
            let disc = qb * qb - qa * qc
            if disc > 0 {
                let t = (-qb - disc.squareRoot()) / qa
                if t > 0 && t < bestT {
                    let py = o.y + t * d.y
                    if py >= 0 { bestT = t; bestId = idMound; hit = true }
                }
            }
        }
        // axis-aligned box (slab method)
        if withBox {
            let lo = SIMD3<Double>(boxCenterXZ.x - boxHalfXZ.x, 0.0, boxCenterXZ.y - boxHalfXZ.y)
            let hi = SIMD3<Double>(boxCenterXZ.x + boxHalfXZ.x, boxHeightM, boxCenterXZ.y + boxHalfXZ.y)
            var t0 = 0.0
            var t1 = Double.infinity
            var miss = false
            for axis in 0..<3 {
                let oa = o[axis]
                let da = d[axis]
                if abs(da) < 1e-14 {
                    if oa < lo[axis] || oa > hi[axis] { miss = true; break }
                } else {
                    var ta = (lo[axis] - oa) / da
                    var tb = (hi[axis] - oa) / da
                    if ta > tb { swap(&ta, &tb) }
                    if ta > t0 { t0 = ta }
                    if tb < t1 { t1 = tb }
                    if t0 > t1 { miss = true; break }
                }
            }
            if !miss && t0 > 0 && t0 < bestT {
                bestT = t0
                bestId = idBox
                hit = true
            }
        }
        return hit ? (t: bestT, id: bestId) : nil
    }

    // MARK: construction

    static func make(_ config: Config = Config()) -> SynthScene {
        var rng = SplitMix64(seed: config.seed)
        let W = config.width
        let H = config.height
        let n = W * H
        let fx = (Double(W) / 2.0) / tan(config.hfovDeg * Double.pi / 180.0 / 2.0)
        let K = Intrinsics(fx: fx, fy: fx, cx: Double(W) / 2.0, cy: Double(H) / 2.0)
        let elev = config.elevationDeg * Double.pi / 180.0
        let ringR = config.radiusM * cos(elev)
        let camH = config.radiusM * sin(elev)
        let target = SIMD3<Double>(0.0, 0.01, 0.0)
        let worldUp = SIMD3<Double>(0.0, 1.0, 0.0)

        var frames: [SweepFrame] = []
        var truthDepth: [FloatGrid] = []
        var truthId: [Grid<UInt8>] = []

        for k in 0..<config.nFrames {
            let phi = 2.0 * Double.pi * (Double(k) + 0.3) / Double(max(config.nFrames, 1))
            let cpos = SIMD3<Double>(ringR * cos(phi), camH, ringR * sin(phi))
            let fwd = simd_normalize(target - cpos)
            let right = simd_normalize(simd_cross(fwd, worldUp))
            let up = simd_cross(right, fwd)
            let T = simd_double4x4(columns: (
                SIMD4<Double>(right.x, right.y, right.z, 0.0),
                SIMD4<Double>(up.x, up.y, up.z, 0.0),
                SIMD4<Double>(-fwd.x, -fwd.y, -fwd.z, 0.0),
                SIMD4<Double>(cpos.x, cpos.y, cpos.z, 1.0)))

            var depth = [Float](repeating: Float.nan, count: n)
            var ids = [UInt8](repeating: 0, count: n)
            for v in 0..<H {
                let ry = -(Double(v) - K.cy) / K.fy
                for u in 0..<W {
                    let rx = (Double(u) - K.cx) / K.fx
                    let dir = right * rx + up * ry + fwd
                    if let hit = cast(origin: cpos, dir: dir, withMound: config.withMound, withBox: config.withBox) {
                        depth[v * W + u] = Float(hit.t)
                        ids[v * W + u] = hit.id
                    }
                }
            }

            let scaleA = 8.0 + 6.0 * synthUniform(&rng)
            let offsetB = -6.0 + 12.0 * synthUniform(&rng)
            var rel = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let d = Double(depth[i])
                if d.isFinite && d > 0 {
                    rel[i] = Float(scaleA * (1.0 / d) + offsetB + scaleA * config.relNoise * synthGauss(&rng))
                }
            }

            var fps: [SIMD3<Double>] = []
            if W > 6 && H > 6 {
                for _ in 0..<config.featurePointsPerFrame {
                    let u = Int.random(in: 2..<(W - 2), using: &rng)
                    let v = Int.random(in: 2..<(H - 2), using: &rng)
                    let d = Double(depth[v * W + u])
                    if !(d.isFinite && d > 0) { continue }
                    let rx = (Double(u) - K.cx) / K.fx
                    let ry = -(Double(v) - K.cy) / K.fy
                    var p = cpos + right * (rx * d) + up * (ry * d) + fwd * d
                    p.x += config.featureNoiseM * synthGauss(&rng)
                    p.y += config.featureNoiseM * synthGauss(&rng)
                    p.z += config.featureNoiseM * synthGauss(&rng)
                    fps.append(p)
                }
            }

            var masks: [String: Mask] = [:]
            var labels: [String: String] = [:]
            if config.includeMasks {
                for (name, id) in [("mound", idMound), ("box", idBox)] {
                    var data = [Bool](repeating: false, count: n)
                    var hit = false
                    for i in 0..<n where ids[i] == id { data[i] = true; hit = true }
                    if hit {
                        masks[name] = Mask(width: W, height: H, data: data)
                        labels[name] = "unknown"
                    }
                }
            }

            frames.append(SweepFrame(t: Double(k) * 0.5, T_wc: T, K: K,
                                     relDepth: FloatGrid(width: W, height: H, data: rel),
                                     depthM: nil, conf: nil, masks: masks, labels: labels, featurePointsW: fps))
            truthDepth.append(FloatGrid(width: W, height: H, data: depth))
            truthId.append(Grid<UInt8>(width: W, height: H, data: ids))
        }
        return SynthScene(config: config, plane: Plane(n: SIMD3<Double>(0, 1, 0), offset: 0.0),
                          frames: frames, truthDepth: truthDepth, truthId: truthId)
    }

    // MARK: evaluation helpers

    /// |depth - truth| (metres) over pixels whose (2e+1)^2 neighbourhood lies fully inside region `id`
    /// (neighbourhoods leaving the image do not count, so image borders are excluded too). Invalid depth is skipped.
    func interiorErrors(frame i: Int, id: UInt8, erodePx e: Int, depth: FloatGrid) -> [Double] {
        let g = truthId[i]
        let W = g.width
        let H = g.height
        var hOK = [Bool](repeating: false, count: W * H)
        for v in 0..<H {
            for u in 0..<W {
                if u - e < 0 || u + e >= W { continue }
                var ok = true
                for dx in -e...e where g[u + dx, v] != id { ok = false; break }
                hOK[v * W + u] = ok
            }
        }
        var errs: [Double] = []
        for v in 0..<H {
            if v - e < 0 || v + e >= H { continue }
            for u in 0..<W {
                var ok = true
                for dy in -e...e where !hOK[(v + dy) * W + u] { ok = false; break }
                if !ok { continue }
                let d = Double(depth[u, v])
                if d.isFinite { errs.append(abs(d - Double(truthDepth[i][u, v]))) }
            }
        }
        return errs
    }
}
