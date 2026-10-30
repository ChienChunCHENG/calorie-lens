// Table-plane fitting. Port of tools/replay/plane.py (fit_plane_ransac); spec: tools/replay/VOLUME_ALGO.md section 1.
//
// 3-point RANSAC scored on <= 5000 points, then 3 rounds of (inliers -> centroid + smallest-eigenvector of the 3x3
// covariance) on ALL points. Python uses an SVD; the right singular vector of the smallest singular value equals the
// eigenvector of the smallest covariance eigenvalue, which we get with a tiny Jacobi solver (no LAPACK needed).
// Returns nil where Python raises (< 3 finite points, or every sampled triple collinear).
import Foundation
import simd

public enum PlaneFit {
    static let maxScorePts = 5000

    public static func fitRANSAC(_ points: [SIMD3<Double>], threshM: Double = 0.005, iters: Int = 500,
                                 camCentersW: [SIMD3<Double>]? = nil, seed: UInt64 = 0) -> Plane? {
        var P: [SIMD3<Double>] = []
        P.reserveCapacity(points.count)
        for p in points where p.x.isFinite && p.y.isFinite && p.z.isFinite {
            P.append(p)
        }
        if P.count < 3 { return nil }
        var rng = SplitMix64(seed: seed)

        var S: [SIMD3<Double>] = P
        if P.count > maxScorePts {
            let pick = sampleWithoutReplacement(count: P.count, k: maxScorePts, rng: &rng)
            S = pick.map { P[$0] }
        }
        let m = S.count

        var found = false
        var bestScore = -Double.infinity
        var bestN = SIMD3<Double>(0, 0, 0)
        var bestOff = 0.0
        for _ in 0..<max(iters, 0) {
            let ia = Int.random(in: 0..<m, using: &rng)
            let ib = Int.random(in: 0..<m, using: &rng)
            let ic = Int.random(in: 0..<m, using: &rng)
            let a = S[ia]
            let cr = simd_cross(S[ib] - a, S[ic] - a)
            let ln = simd_length(cr)
            if !(ln > 1e-9) { continue }
            let nrm = cr / ln
            let off = simd_dot(nrm, a)
            var cnt = 0
            var sumClip = 0.0
            for p in S {
                let dd = abs(simd_dot(p, nrm) - off)
                if dd < threshM { cnt += 1 }
                sumClip += min(dd, threshM)
            }
            let score = Double(cnt) - 0.5 * (sumClip / Double(m)) / threshM
            if score > bestScore {
                bestScore = score
                bestN = nrm
                bestOff = off
                found = true
            }
        }
        if !found { return nil }

        var n = bestN
        var off0 = bestOff
        for _ in 0..<3 {
            var sel: [SIMD3<Double>] = []
            for p in P where abs(simd_dot(p, n) - off0) < threshM { sel.append(p) }
            if sel.count < 3 { break }
            let fit = lsqPlane(sel)
            n = fit.n
            off0 = simd_dot(n, fit.centroid)
        }
        var sumSq = 0.0
        var cnt = 0
        for p in P {
            let r = simd_dot(p, n) - off0
            if abs(r) < threshM {
                sumSq += r * r
                cnt += 1
            }
        }
        let rmse = cnt > 0 ? (sumSq / Double(cnt)).squareRoot() : 0.0
        let oriented = orient(n: n, offset: off0, camCentersW: camCentersW)
        return Plane(n: oriented.n, offset: oriented.offset, inlierRMSEm: rmse)
    }

    /// Flip (n, offset) so the camera centres (or +Y world when none are given) are on the positive side.
    static func orient(n: SIMD3<Double>, offset: Double, camCentersW: [SIMD3<Double>]?) -> (n: SIMD3<Double>, offset: Double) {
        let up: Bool
        if let cams = camCentersW, !cams.isEmpty {
            var sum = 0.0
            for c in cams { sum += simd_dot(c, n) - offset }
            up = (sum / Double(cams.count)) > 0.0
        } else {
            up = n.y >= 0.0
        }
        if up { return (n, offset) }
        return (-n, -offset)
    }

    /// Centroid and unit normal (smallest covariance eigenvector) of a point set.
    static func lsqPlane(_ pts: [SIMD3<Double>]) -> (n: SIMD3<Double>, centroid: SIMD3<Double>) {
        var c = SIMD3<Double>(0, 0, 0)
        for p in pts { c += p }
        c /= Double(pts.count)
        var cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0
        for p in pts {
            let d = p - c
            cxx += d.x * d.x; cxy += d.x * d.y; cxz += d.x * d.z
            cyy += d.y * d.y; cyz += d.y * d.z; czz += d.z * d.z
        }
        let cov: [[Double]] = [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]]
        let n = smallestEigenvector(cov)
        return (n, c)
    }

    /// Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (cyclic-by-largest Jacobi).
    static func smallestEigenvector(_ input: [[Double]]) -> SIMD3<Double> {
        var a = input
        var v: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for _ in 0..<64 {
            var p = 0
            var q = 1
            var mx = abs(a[0][1])
            if abs(a[0][2]) > mx { p = 0; q = 2; mx = abs(a[0][2]) }
            if abs(a[1][2]) > mx { p = 1; q = 2; mx = abs(a[1][2]) }
            let scale = abs(a[0][0]) + abs(a[1][1]) + abs(a[2][2])
            if mx <= 1e-15 * scale || mx == 0 { break }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            let sgn: Double = theta >= 0 ? 1.0 : -1.0
            let t = sgn / (abs(theta) + (theta * theta + 1.0).squareRoot())
            let cs = 1.0 / (t * t + 1.0).squareRoot()
            let sn = t * cs
            let app = a[p][p]
            let aqq = a[q][q]
            let apq = a[p][q]
            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = 0
            a[q][p] = 0
            let r = 3 - p - q
            let arp = a[r][p]
            let arq = a[r][q]
            a[r][p] = cs * arp - sn * arq
            a[p][r] = a[r][p]
            a[r][q] = cs * arq + sn * arp
            a[q][r] = a[r][q]
            for k in 0..<3 {
                let vkp = v[k][p]
                let vkq = v[k][q]
                v[k][p] = cs * vkp - sn * vkq
                v[k][q] = sn * vkp + cs * vkq
            }
        }
        var imin = 0
        if a[1][1] < a[imin][imin] { imin = 1 }
        if a[2][2] < a[imin][imin] { imin = 2 }
        let vec = SIMD3<Double>(v[0][imin], v[1][imin], v[2][imin])
        let len = simd_length(vec)
        return len > 0 ? vec / len : SIMD3<Double>(0, 1, 0)
    }

    /// k distinct indices from 0..<count (partial Fisher-Yates), in random order.
    static func sampleWithoutReplacement(count: Int, k: Int, rng: inout SplitMix64) -> [Int] {
        let kk = min(k, count)
        if kk <= 0 { return [] }
        var idx = Array(0..<count)
        for i in 0..<kk {
            let j = Int.random(in: i..<count, using: &rng)
            if i != j { idx.swapAt(i, j) }
        }
        return Array(idx[0..<kk])
    }
}
