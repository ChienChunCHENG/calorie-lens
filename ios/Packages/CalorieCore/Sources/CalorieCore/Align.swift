// Relative depth -> metric depth, per frame. Port of tools/replay/align.py (read its header for the model):
//
//     1/d = s * rel + t   (s > 0)
//
// Anchors = (1/d_metric, rel) pairs from (a) table-plane ray intersections on background pixels (outside every mask,
// dilated by BG_DILATE_PX) and (b) ARKit feature points projected into the frame. RANSAC in metric depth, shrinking
// threshold iterated Deming WLS, smooth local disparity offset field, rejection gates (depthM stays nil on failure).
//
// Deviations from the Python (all numerically negligible, see the report):
//   * mask dilation does not wrap around the image border (numpy.roll does); anchors are cut 2 px from the border anyway.
//   * Gaussian-normalised moments in fixMaskEdges are evaluated only inside the bounding box of each mask (+3 sigma),
//     which is exact because all moment images are zero outside the mask.
//   * the local offset field is evaluated with the separable form of the Gaussian kernel (exact, ~100x fewer exp calls).
//   * randomness comes from SplitMix64 (not numpy), so results match Python to tolerance only.
//   * masks of the wrong size (not width x height) are ignored; masks are visited in sorted-key order.
import Foundation
import simd

public enum Align {
    public static let bgDilatePx = 4
    public static let maxBgAnchors = 2000
    public static let fpWeight = 4.0
    public static let minAnchors = 40
    public static let minInlierFrac = 0.5
    public static let maxRmseM = 0.015
    public static let minRelSpread = 0.04
    public static let demingDelta = 0.01
    public static let localSigmaPx = 30.0
    public static let localRidge = 5.0
    public static let edgeRingPx = 2
    public static let edgeSigmaPx = 3.0
    public static let edgeSlopeRidge = 0.5
    public static let minDisp = 1.0 / 5.0

    // MARK: public API

    public static func fixMaskEdges(rel: FloatGrid, masks: [String: Mask], labels: [String: String]) -> FloatGrid {
        if edgeRingPx <= 0 || masks.isEmpty { return rel }
        let w = rel.width
        let h = rel.height
        var d = [Double](repeating: 0, count: w * h)
        for i in 0..<(w * h) { d[i] = Double(rel.data[i]) }
        let fixed = AlignKernels.fixMaskEdges(d, w: w, h: h, masks: masks, labels: labels,
                                              ring: edgeRingPx, sigma: edgeSigmaPx)
        var out = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) { out[i] = Float(fixed[i]) }
        return FloatGrid(width: w, height: h, data: out)
    }

    /// Robust fit y = s x + t (y = anchor disparity, x = rel). Mirrors Python robust_affine_fit.
    public static func robustAffineFit(x: [Double], y: [Double], w: [Double], rng: inout SplitMix64,
                                       ransacThr: Double = 0.02, iters: Int = 300) -> (s: Double, t: Double, inliers: [Bool])? {
        let n = x.count
        if n < 3 { return nil }
        var dA = [Double](repeating: 0, count: n)       // 1/y = metric depth of each anchor
        for i in 0..<n { dA[i] = 1.0 / y[i] }

        var haveBest = false
        var bestS = 0.0
        var bestT = 0.0
        var bestScore = -1.0
        for _ in 0..<iters {
            let i = Int.random(in: 0..<n, using: &rng)
            let j = Int.random(in: 0..<n, using: &rng)
            let dx = x[i] - x[j]
            if abs(dx) < 1e-6 { continue }
            let s = (y[i] - y[j]) / dx
            if !(s > 0) { continue }
            let t = y[i] - s * x[i]
            var score = 0.0
            for k in 0..<n {
                let pred = s * x[k] + t
                if pred > 1e-3 {
                    if abs(1.0 / pred - dA[k]) < ransacThr { score += w[k] }
                }
            }
            if score > bestScore {
                bestScore = score
                bestS = s
                bestT = t
                haveBest = true
            }
        }
        if !haveBest { return nil }
        var s = bestS
        var t = bestT
        let thresholds: [Double] = [ransacThr, 0.015, 0.01, 0.01]
        for thr in thresholds {
            var xs: [Double] = []
            var ys: [Double] = []
            var ws: [Double] = []
            for k in 0..<n {
                let pred = s * x[k] + t
                if pred > 1e-3 && abs(1.0 / pred - dA[k]) < thr {
                    xs.append(x[k]); ys.append(y[k]); ws.append(w[k])
                }
            }
            if xs.count < 3 { return nil }
            guard let r = AlignKernels.wls(xs, ys, ws), r.s > 0 else { return nil }
            s = r.s
            t = r.t
        }
        var inl = [Bool](repeating: false, count: n)
        for k in 0..<n {
            let pred = s * x[k] + t
            inl[k] = pred > 1e-3 && abs(1.0 / pred - dA[k]) < 0.015
        }
        return (s, t, inl)
    }

    /// Fills f.depthM (leaves it untouched on failure) and returns diagnostics.
    public static func alignFrame(_ f: inout SweepFrame, plane: Plane, rng: inout SplitMix64) -> AlignDiag {
        var diag = AlignDiag()
        let W = f.width
        let H = f.height
        let n = W * H
        if W < 5 || H < 5 {
            diag.reason = "image too small"
            return diag
        }

        // relative depth with food-mask borders re-estimated
        var rel = [Double](repeating: 0, count: n)
        for i in 0..<n { rel[i] = Double(f.relDepth.data[i]) }
        rel = AlignKernels.fixMaskEdges(rel, w: W, h: H, masks: f.masks, labels: f.labels,
                                        ring: edgeRingPx, sigma: edgeSigmaPx)

        let T = f.T_wc
        let c0 = SIMD3<Double>(T.columns.0.x, T.columns.0.y, T.columns.0.z)
        let c1 = SIMD3<Double>(T.columns.1.x, T.columns.1.y, T.columns.1.z)
        let c2 = SIMD3<Double>(T.columns.2.x, T.columns.2.y, T.columns.2.z)
        let o = SIMD3<Double>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        let K = f.K
        let nrm = plane.n
        // camera-frame plane normal m = R^T n ; for a camera-frame ray (rx, ry, -1): n . (R ray) = m . ray
        let m0 = simd_dot(c0, nrm)
        let m1 = simd_dot(c1, nrm)
        let m2 = simd_dot(c2, nrm)
        let num = plane.offset - simd_dot(o, nrm)

        // (a) table-plane anchors on background pixels
        var union = [Bool](repeating: false, count: n)
        for (_, m) in f.masks where m.width == W && m.height == H {
            for i in 0..<n where m.data[i] { union[i] = true }
        }
        let grown = AlignKernels.dilate(union, w: W, h: H, r: bgDilatePx)
        var cand: [Int] = []
        cand.reserveCapacity(n / 2)
        for v in 2..<(H - 2) {
            let ry = -(Double(v) - K.cy) / K.fy
            for u in 2..<(W - 2) {
                let idx = v * W + u
                if grown[idx] { continue }
                let rx = (Double(u) - K.cx) / K.fx
                let nd = m0 * rx + m1 * ry - m2
                if !(nd < -1e-3) { continue }
                let dp = num / nd
                if dp > 0.1 && dp < 3.0 { cand.append(idx) }
            }
        }
        var chosen = cand
        if cand.count > maxBgAnchors {
            let pick = PlaneFit.sampleWithoutReplacement(count: cand.count, k: maxBgAnchors, rng: &rng)
            chosen = pick.map { cand[$0] }
        }
        var xs: [Double] = []
        var us: [Double] = []
        var vs: [Double] = []
        var ds: [Double] = []
        var ws: [Double] = []
        let cap = chosen.count + f.featurePointsW.count
        xs.reserveCapacity(cap); us.reserveCapacity(cap); vs.reserveCapacity(cap)
        ds.reserveCapacity(cap); ws.reserveCapacity(cap)
        for idx in chosen {
            let u = idx % W
            let v = idx / W
            let rx = (Double(u) - K.cx) / K.fx
            let ry = -(Double(v) - K.cy) / K.fy
            let nd = m0 * rx + m1 * ry - m2
            xs.append(rel[idx])
            us.append(Double(u))
            vs.append(Double(v))
            ds.append(num / nd)
            ws.append(1.0)
        }
        diag.nBg = chosen.count

        // (b) feature points
        var nFp = 0
        for p in f.featurePointsW {
            let dv = p - o
            let pcx = simd_dot(c0, dv)
            let pcy = simd_dot(c1, dv)
            let d = -simd_dot(c2, dv)
            if !(d > 0.05) { continue }
            let u = K.fx * pcx / d + K.cx
            let v = -K.fy * pcy / d + K.cy
            if u >= 0 && u <= Double(W - 1) && v >= 0 && v <= Double(H - 1) {
                xs.append(AlignKernels.bilinear(rel, w: W, h: H, u: u, v: v))
                us.append(u)
                vs.append(v)
                ds.append(d)
                ws.append(fpWeight)
                nFp += 1
            }
        }
        diag.nFp = nFp

        let nA = xs.count
        if nA < minAnchors {
            diag.reason = "too few anchors (\(nA))"
            return diag
        }
        let sortedX = xs.sorted()
        let med = AlignKernels.percentile(sortedX, 50.0)
        let spread = (AlignKernels.percentile(sortedX, 95.0) - AlignKernels.percentile(sortedX, 5.0)) / max(abs(med), 1e-9)
        diag.relSpread = spread
        if spread < minRelSpread {
            diag.reason = "anchors span too little relative depth (\(String(format: "%.3f", spread)))"
            return diag
        }

        var y = [Double](repeating: 0, count: nA)
        for i in 0..<nA { y[i] = 1.0 / ds[i] }
        guard let fit = robustAffineFit(x: xs, y: y, w: ws, rng: &rng) else {
            diag.reason = "fit failed"
            return diag
        }
        let s = fit.s
        let t = fit.t
        let inl = fit.inliers

        var nInl = 0
        var wInl = 0.0
        var wAll = 0.0
        var sumSq = 0.0
        for i in 0..<nA {
            wAll += ws[i]
            if inl[i] {
                nInl += 1
                wInl += ws[i]
                let pred = s * xs[i] + t
                let r = abs(1.0 / pred - ds[i])      // inliers have pred > 1e-3
                sumSq += ws[i] * r * r
            }
        }
        let rmse: Double? = nInl > 0 && wInl > 0 ? (sumSq / wInl).squareRoot() : nil
        diag.s = s
        diag.t = t
        diag.nInl = nInl
        diag.inlierFrac = wAll > 0 ? wInl / wAll : 0.0
        diag.rmseM = rmse
        if Double(nInl) < Double(minAnchors) * 0.75 {
            diag.reason = "too few inliers (\(nInl))"
        } else if diag.inlierFrac < minInlierFrac {
            diag.reason = "inlier fraction \(String(format: "%.2f", diag.inlierFrac))"
        } else if (rmse ?? Double.infinity) > maxRmseM {
            diag.reason = "residual \(String(format: "%.1f", (rmse ?? 0) * 100)) cm"
        } else if !(s > 0) {
            diag.reason = "non-positive scale"
        }
        if !diag.reason.isEmpty { return diag }

        // local additive disparity offset from the inlier residuals
        var offsetField: [Double]? = nil
        if localSigmaPx > 0 {
            var iu: [Double] = []
            var iv: [Double] = []
            var ir: [Double] = []
            var iw: [Double] = []
            for i in 0..<nA where inl[i] {
                iu.append(us[i]); iv.append(vs[i])
                ir.append(y[i] - (s * xs[i] + t))
                iw.append(ws[i])
            }
            offsetField = AlignKernels.localOffset(us: iu, vs: iv, r: ir, w: iw, W: W, H: H)
        }

        var depth = [Float](repeating: Float.nan, count: n)
        for i in 0..<n {
            var disp = s * rel[i] + t
            if let off = offsetField { disp += off[i] }
            if disp > minDisp {
                depth[i] = Float(1.0 / disp)
            }
        }
        f.depthM = FloatGrid(width: W, height: H, data: depth)
        diag.ok = true
        return diag
    }

    /// Orients planeHint up (or fits one from feature points); fills depthM in place. nil if no plane obtainable.
    public static func alignFrames(_ frames: inout [SweepFrame], planeHint: Plane?, seed: UInt64 = 0) -> (plane: Plane, diags: [AlignDiag])? {
        var plane: Plane
        if let hint = planeHint {
            plane = orientUp(hint, frames)
        } else {
            var pts: [SIMD3<Double>] = []
            for f in frames { pts.append(contentsOf: f.featurePointsW) }
            if pts.isEmpty { return nil }
            guard let fitted = PlaneFit.fitRANSAC(pts, seed: seed) else { return nil }
            plane = orientUp(fitted, frames)
        }
        var rng = SplitMix64(seed: seed)
        var diags: [AlignDiag] = []
        diags.reserveCapacity(frames.count)
        for i in 0..<frames.count {
            frames[i].depthM = nil
            let d = alignFrame(&frames[i], plane: plane, rng: &rng)
            diags.append(d)
        }
        return (plane, diags)
    }

    static func orientUp(_ plane: Plane, _ frames: [SweepFrame]) -> Plane {
        if frames.isEmpty { return plane }
        var sum = 0.0
        for f in frames { sum += simd_dot(f.cameraCenterW, plane.n) - plane.offset }
        if sum / Double(frames.count) < 0 { return plane.flipped }
        return plane
    }
}

// MARK: - numeric kernels (internal namespace so other files can reuse them without name clashes)

enum AlignKernels {
    static func gaussKernel(_ sigma: Double) -> [Double] {
        let r = Int((3.0 * sigma).rounded(.up))
        var k = [Double](repeating: 0, count: 2 * r + 1)
        var sum = 0.0
        for i in 0..<(2 * r + 1) {
            let x = Double(i - r) / sigma
            k[i] = exp(-0.5 * x * x)
            sum += k[i]
        }
        for i in 0..<k.count { k[i] /= sum }
        return k
    }

    /// Separable Gaussian blur with edge replication (== numpy pad mode "edge").
    static func gaussBlur(_ src: [Double], w: Int, h: Int, kernel: [Double]) -> [Double] {
        let r = (kernel.count - 1) / 2
        let kc = kernel.count
        var tmp = [Double](repeating: 0, count: w * h)
        var out = [Double](repeating: 0, count: w * h)
        var pad = [Double](repeating: 0, count: w + 2 * r)
        src.withUnsafeBufferPointer { sp in
            tmp.withUnsafeMutableBufferPointer { tp in
                pad.withUnsafeMutableBufferPointer { pp in
                    kernel.withUnsafeBufferPointer { kp in
                        for y in 0..<h {
                            let row = y * w
                            let left = sp[row]
                            let right = sp[row + w - 1]
                            for i in 0..<r {
                                pp[i] = left
                                pp[r + w + i] = right
                            }
                            for x in 0..<w { pp[r + x] = sp[row + x] }
                            for x in 0..<w {
                                var acc = 0.0
                                for k in 0..<kc { acc += kp[k] * pp[x + k] }
                                tp[row + x] = acc
                            }
                        }
                    }
                }
            }
        }
        tmp.withUnsafeBufferPointer { tp in
            out.withUnsafeMutableBufferPointer { op in
                kernel.withUnsafeBufferPointer { kp in
                    for y in 0..<h {
                        let orow = y * w
                        for k in 0..<kc {
                            var yy = y + k - r
                            if yy < 0 { yy = 0 }
                            if yy > h - 1 { yy = h - 1 }
                            let wk = kp[k]
                            let irow = yy * w
                            for x in 0..<w { op[orow + x] += wk * tp[irow + x] }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Square (Chebyshev) dilation by r px, no wrap-around.
    static func dilate(_ m: [Bool], w: Int, h: Int, r: Int) -> [Bool] {
        if r <= 0 { return m }
        var tmp = [Bool](repeating: false, count: w * h)
        var out = [Bool](repeating: false, count: w * h)
        var pre = [Int32](repeating: 0, count: max(w, h) + 1)
        for y in 0..<h {
            pre[0] = 0
            for x in 0..<w { pre[x + 1] = pre[x] + (m[y * w + x] ? 1 : 0) }
            for x in 0..<w {
                let lo = max(0, x - r)
                let hi = min(w - 1, x + r)
                tmp[y * w + x] = pre[hi + 1] - pre[lo] > 0
            }
        }
        for x in 0..<w {
            pre[0] = 0
            for y in 0..<h { pre[y + 1] = pre[y] + (tmp[y * w + x] ? 1 : 0) }
            for y in 0..<h {
                let lo = max(0, y - r)
                let hi = min(h - 1, y + r)
                out[y * w + x] = pre[hi + 1] - pre[lo] > 0
            }
        }
        return out
    }

    /// Square erosion by r px; the image border is not treated as an object border.
    static func erode(_ m: [Bool], w: Int, h: Int, r: Int) -> [Bool] {
        if r <= 0 { return m }
        var inv = [Bool](repeating: false, count: w * h)
        for i in 0..<(w * h) { inv[i] = !m[i] }
        let d = dilate(inv, w: w, h: h, r: r)
        var out = [Bool](repeating: false, count: w * h)
        for i in 0..<(w * h) { out[i] = m[i] && !d[i] }
        return out
    }

    /// numpy.percentile (linear interpolation) of an ascending-sorted array.
    static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        let n = sorted.count
        if n == 0 { return 0 }
        if n == 1 { return sorted[0] }
        let pos = q / 100.0 * Double(n - 1)
        let lo = Int(pos.rounded(.down))
        let hi = min(lo + 1, n - 1)
        let fr = pos - Double(lo)
        return sorted[lo] * (1.0 - fr) + sorted[hi] * fr
    }

    static func bilinear(_ img: [Double], w: Int, h: Int, u: Double, v: Double) -> Double {
        let u0 = min(max(Int(u.rounded(.down)), 0), w - 2)
        let v0 = min(max(Int(v.rounded(.down)), 0), h - 2)
        let fu = u - Double(u0)
        let fv = v - Double(v0)
        let a = img[v0 * w + u0] * (1 - fu) + img[v0 * w + u0 + 1] * fu
        let b = img[(v0 + 1) * w + u0] * (1 - fu) + img[(v0 + 1) * w + u0 + 1] * fu
        return a * (1 - fv) + b * fv
    }

    /// Weighted Deming fit y = s x + t (Python _wls).
    static func wls(_ x: [Double], _ y: [Double], _ w: [Double]) -> (s: Double, t: Double)? {
        var sw = 0.0, swx = 0.0, swy = 0.0
        for i in 0..<x.count {
            sw += w[i]; swx += w[i] * x[i]; swy += w[i] * y[i]
        }
        if !(sw > 0) { return nil }
        let mx = swx / sw
        let my = swy / sw
        var sxx = 0.0, syy = 0.0, sxy = 0.0
        for i in 0..<x.count {
            let dx = x[i] - mx
            let dy = y[i] - my
            sxx += w[i] * dx * dx
            syy += w[i] * dy * dy
            sxy += w[i] * dx * dy
        }
        if sxx < 1e-12 || sxy <= 0 { return nil }
        let d = Align.demingDelta
        let q = syy - d * sxx
        let s = (q + (q * q + 4.0 * d * sxy * sxy).squareRoot()) / (2.0 * sxy)
        return (s, my - s * mx)
    }

    /// Smooth additive disparity offset field (Python _local_offset), evaluated on a coarse grid with the separable
    /// Gaussian kernel and bilinearly upsampled to W x H.
    static func localOffset(us: [Double], vs: [Double], r: [Double], w: [Double], W: Int, H: Int, grid: Int = 8) -> [Double] {
        let nx = (W + grid + grid - 1) / grid        // len(arange(0, W + grid, grid))
        let ny = (H + grid + grid - 1) / grid
        var num = [Double](repeating: 0, count: nx * ny)
        var den = [Double](repeating: Align.localRidge, count: nx * ny)
        let inv = 1.0 / (2.0 * Align.localSigmaPx * Align.localSigmaPx)
        var ex = [Double](repeating: 0, count: nx)
        var ey = [Double](repeating: 0, count: ny)
        let g = Double(grid)
        for i in 0..<us.count {
            for j in 0..<nx {
                let dx = Double(j) * g - us[i]
                ex[j] = exp(-dx * dx * inv)
            }
            for k in 0..<ny {
                let dy = Double(k) * g - vs[i]
                ey[k] = exp(-dy * dy * inv)
            }
            let wi = w[i]
            let ri = r[i]
            ex.withUnsafeBufferPointer { exp_ in
                num.withUnsafeMutableBufferPointer { np in
                    den.withUnsafeMutableBufferPointer { dp in
                        for k in 0..<ny {
                            let ky = ey[k] * wi
                            let base = k * nx
                            for j in 0..<nx {
                                let kk = ky * exp_[j]
                                np[base + j] += kk * ri
                                dp[base + j] += kk
                            }
                        }
                    }
                }
            }
        }
        var cg = [Double](repeating: 0, count: nx * ny)
        for i in 0..<(nx * ny) { cg[i] = num[i] / den[i] }
        var out = [Double](repeating: 0, count: W * H)
        for v in 0..<H {
            let fvv = Double(v) / g
            let v0 = min(max(Int(fvv.rounded(.down)), 0), ny - 2)
            let fv = fvv - Double(v0)
            for u in 0..<W {
                let fuu = Double(u) / g
                let u0 = min(max(Int(fuu.rounded(.down)), 0), nx - 2)
                let fu = fuu - Double(u0)
                let a = cg[v0 * nx + u0] * (1 - fu) + cg[v0 * nx + u0 + 1] * fu
                let b = cg[(v0 + 1) * nx + u0] * (1 - fu) + cg[(v0 + 1) * nx + u0 + 1] * fu
                out[v * W + u] = a * (1 - fv) + b * fv
            }
        }
        return out
    }

    @inline(__always)
    static func det3(_ a: Double, _ b: Double, _ c: Double,
                     _ d: Double, _ e: Double, _ f: Double,
                     _ g: Double, _ h: Double, _ i: Double) -> Double {
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// Python fix_mask_edges on a Double buffer (ring/sigma explicit).
    static func fixMaskEdges(_ rel: [Double], w: Int, h: Int, masks: [String: Mask], labels: [String: String],
                             ring: Int, sigma: Double) -> [Double] {
        if ring <= 0 || masks.isEmpty { return rel }
        var out = rel
        let kernel = gaussKernel(sigma)
        let kr = (kernel.count - 1) / 2
        for key in masks.keys.sorted() {
            guard let m = masks[key], m.width == w, m.height == h else { continue }
            if (labels[key] ?? "").hasPrefix("container_") { continue }
            let inner = erode(m.data, w: w, h: h, r: ring)
            var nInner = 0
            var nTarget = 0
            var minX = w, maxX = -1, minY = h, maxY = -1
            for y in 0..<h {
                for x in 0..<w {
                    let i = y * w + x
                    if m.data[i] {
                        if x < minX { minX = x }
                        if x > maxX { maxX = x }
                        if y < minY { minY = y }
                        if y > maxY { maxY = y }
                        if inner[i] { nInner += 1 } else { nTarget += 1 }
                    }
                }
            }
            if nInner < 20 || nTarget == 0 { continue }
            let x0 = max(0, minX - kr)
            let x1 = min(w - 1, maxX + kr)
            let y0 = max(0, minY - kr)
            let y1 = min(h - 1, maxY + kr)
            let cw = x1 - x0 + 1
            let ch = y1 - y0 + 1
            let cn = cw * ch
            var v0 = [Double](repeating: 0, count: cn)
            var vx = [Double](repeating: 0, count: cn)
            var vy = [Double](repeating: 0, count: cn)
            var vxx = [Double](repeating: 0, count: cn)
            var vxy = [Double](repeating: 0, count: cn)
            var vyy = [Double](repeating: 0, count: cn)
            var vr = [Double](repeating: 0, count: cn)
            var vrx = [Double](repeating: 0, count: cn)
            var vry = [Double](repeating: 0, count: cn)
            for cy in 0..<ch {
                for cx in 0..<cw {
                    let gi = (y0 + cy) * w + (x0 + cx)
                    if inner[gi] && rel[gi].isFinite {
                        let ci = cy * cw + cx
                        let X = Double(x0 + cx)
                        let Y = Double(y0 + cy)
                        let R = rel[gi]
                        v0[ci] = 1
                        vx[ci] = X
                        vy[ci] = Y
                        vxx[ci] = X * X
                        vxy[ci] = X * Y
                        vyy[ci] = Y * Y
                        vr[ci] = R
                        vrx[ci] = R * X
                        vry[ci] = R * Y
                    }
                }
            }
            let S = gaussBlur(v0, w: cw, h: ch, kernel: kernel)
            let Sx = gaussBlur(vx, w: cw, h: ch, kernel: kernel)
            let Sy = gaussBlur(vy, w: cw, h: ch, kernel: kernel)
            let Sxx = gaussBlur(vxx, w: cw, h: ch, kernel: kernel)
            let Sxy = gaussBlur(vxy, w: cw, h: ch, kernel: kernel)
            let Syy = gaussBlur(vyy, w: cw, h: ch, kernel: kernel)
            let Sr = gaussBlur(vr, w: cw, h: ch, kernel: kernel)
            let Srx = gaussBlur(vrx, w: cw, h: ch, kernel: kernel)
            let Sry = gaussBlur(vry, w: cw, h: ch, kernel: kernel)
            for cy in 0..<ch {
                for cx in 0..<cw {
                    let gi = (y0 + cy) * w + (x0 + cx)
                    if !m.data[gi] || inner[gi] { continue }
                    let ci = cy * cw + cx
                    let s0 = S[ci]
                    if !(s0 > 0.05) { continue }
                    let px = Double(x0 + cx)
                    let py = Double(y0 + cy)
                    let sx = Sx[ci] - px * s0
                    let sy = Sy[ci] - py * s0
                    let sr = Sr[ci]
                    let sxx = Sxx[ci] - 2 * px * Sx[ci] + px * px * s0
                    let syy = Syy[ci] - 2 * py * Sy[ci] + py * py * s0
                    let sxy = Sxy[ci] - px * Sy[ci] - py * Sx[ci] + px * py * s0
                    let srx = Srx[ci] - px * sr
                    let sry = Sry[ci] - py * sr
                    let lam = Align.edgeSlopeRidge * max(s0, 1e-9)
                    let a11 = sxx + lam
                    let a22 = syy + lam
                    let detA = det3(s0, sx, sy, sx, a11, sxy, sy, sxy, a22)
                    if !(abs(detA) > 1e-300) { continue }
                    let detN = det3(sr, sx, sy, srx, a11, sxy, sry, sxy, a22)
                    let c0 = detN / detA
                    if c0.isFinite { out[gi] = c0 }
                }
            }
        }
        return out
    }
}
