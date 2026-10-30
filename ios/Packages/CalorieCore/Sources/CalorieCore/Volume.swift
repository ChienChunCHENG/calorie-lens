// Per-item food volume from metric depth + masks + table plane.
// Port of tools/replay/volume.py (spec: tools/replay/VOLUME_ALGO.md); constants and note texts are identical.
// Differences from the Python reference: the bootstrap uses SplitMix64 (not numpy's PCG64), so intervals differ
// slightly; the bootstrap re-fuses precomputed per-frame cell medians instead of re-gathering points.
import Foundation
import simd

public enum Volume {
    public static let drinkLabels: Set<String> = ["bubble_milk_tea", "container_cup", "black_tea", "green_tea", "soy_milk"]

    static let hMax = 0.25                 // clamp on height above table (m)
    static let minValidCells = 20
    static let fillIters = 12              // inpainting passes (12 x 3 mm = 36 mm reach)
    static let fpFloor = 0.0
    static let fpSat = 0.5                 // footprint weight = min(1, presence frequency / fpSat)
    static let hullVote = 0.8              // table-level projection vote threshold
    static let subN = 3                    // vote sampled at subN x subN points per cell
    static let baseThickM = 0.005          // bowl base thickness
    static let scaleErr = 0.06             // +-6 % linear depth scale error (90 % interval)
    static let z90 = 1.645
    static let weightSat = 0.6             // only for observed cells outside the voted footprint
    static let erodePx = 1
    static let defaultBowlRim = Triple(min: 12.5, typ: 13.5, max: 14.5)
    static let defaultBowlDepth = Triple(min: 6.0, typ: 6.5, max: 7.5)

    // MARK: - small helpers

    /// Deterministic orthonormal (a, b) spanning the plane.
    static func planeBasis(_ n: SIMD3<Double>) -> (SIMD3<Double>, SIMD3<Double>) {
        var idx = 0
        var best = abs(n.x)
        if abs(n.y) < best { best = abs(n.y); idx = 1 }
        if abs(n.z) < best { idx = 2 }
        var ref = SIMD3<Double>(0, 0, 0)
        ref[idx] = 1.0
        var a = simd_cross(n, ref)
        a = a / simd_length(a)
        let b = simd_cross(n, a)
        return (a, b)
    }

    /// 4-neighbour binary erosion, `px` iterations (border pixels count as background).
    static func erode(_ mask: [Bool], width: Int, height: Int, px: Int) -> [Bool] {
        var m = mask
        if px <= 0 { return m }
        for _ in 0..<px {
            var out = [Bool](repeating: false, count: m.count)
            for v in 1..<max(height - 1, 1) {
                for u in 1..<max(width - 1, 1) {
                    let i = v * width + u
                    if m[i] && m[i - 1] && m[i + 1] && m[i - width] && m[i + width] { out[i] = true }
                }
            }
            m = out
        }
        return m
    }

    static func medianOf(_ a: [Double]) -> Double {
        let s = a.sorted()
        let n = s.count
        if n == 0 { return 0 }
        if n % 2 == 1 { return s[n / 2] }
        return 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    /// numpy-style linear-interpolation percentile, p in [0, 100].
    static func percentileOf(_ a: [Double], _ p: Double) -> Double {
        let s = a.sorted()
        let n = s.count
        if n == 0 { return 0 }
        let pos = p / 100.0 * Double(n - 1)
        let lo = Int(pos.rounded(.down))
        let hi = min(lo + 1, n - 1)
        let fr = pos - Double(lo)
        return s[lo] + (s[hi] - s[lo]) * fr
    }

    static func sigma90(_ lo: Double, _ hi: Double) -> Double {
        return log(max(hi, 1e-12) / max(lo, 1e-12)) / 3.29
    }

    /// Plane coordinates (pu, pv), height h of the selected pixels, row-major order.
    /// Returns the pixel indices too so callers can look up per-pixel flags.
    static func unprojectPlane(depth: FloatGrid, K: Intrinsics, T: simd_double4x4, sel: [Bool],
                               aAx: SIMD3<Double>, bAx: SIMD3<Double>, plane: Plane)
        -> (idx: [Int], pu: [Double], pv: [Double], h: [Double]) {
        var idx: [Int] = []
        var pu: [Double] = []
        var pv: [Double] = []
        var hh: [Double] = []
        let w = depth.width
        for i in 0..<sel.count where sel[i] {
            let u = i % w
            let v = i / w
            let d = Double(depth.data[i])
            let x = (Double(u) - K.cx) * d / K.fx
            let y = -(Double(v) - K.cy) * d / K.fy
            let p4 = T * SIMD4<Double>(x, y, -d, 1.0)
            let p = SIMD3<Double>(p4.x, p4.y, p4.z)
            idx.append(i)
            pu.append(simd_dot(p, aAx))
            pv.append(simd_dot(p, bAx))
            hh.append(plane.height(p))
        }
        return (idx, pu, pv, hh)
    }

    private struct PtsFrame {
        var pu: [Double]
        var pv: [Double]
        var h: [Double]
        var er: [Bool]
    }

    /// Per-cell observations of all frames in CSR layout: cell c (sorted by key) owns entries start[c]..<start[c+1].
    private struct CellObs {
        var keys: [Int] = []
        var start: [Int] = [0]
        var frame: [Int] = []
        var h: [Double] = []
    }

    // MARK: - fusion / fill / volume

    /// Per-cell median over the frame multiset `mult` (mult[f] = how often frame f is drawn); cells with fewer than
    /// `minObs` observations are unobserved. Weight: voted footprint -> fpw, else min(1, count / (0.6 nSel)).
    private static func fuse(_ obs: CellObs, mult: [Int], nSel: Int, ncell: Int, minObs: Int,
                             footprint: [Bool], fpw: [Double]) -> (g: [Double], w: [Double], nValid: Int) {
        var g = [Double](repeating: 0, count: ncell)
        var w = [Double](repeating: 0, count: ncell)
        var nValid = 0
        var buf: [Double] = []
        buf.reserveCapacity(64)
        let denom = weightSat * Double(nSel)
        for ci in 0..<obs.keys.count {
            buf.removeAll(keepingCapacity: true)
            for e in obs.start[ci]..<obs.start[ci + 1] {
                let c = mult[obs.frame[e]]
                if c > 0 {
                    let hv = obs.h[e]
                    for _ in 0..<c { buf.append(hv) }
                }
            }
            let cnt = buf.count
            if cnt == 0 || cnt < minObs { continue }
            buf.sort()
            let med = 0.5 * (buf[(cnt - 1) / 2] + buf[cnt / 2])
            let k = obs.keys[ci]
            g[k] = med
            let wt: Double
            if footprint[k] {
                wt = fpw[k]
            } else {
                wt = min(Double(cnt) / denom, 1.0)
            }
            w[k] = wt
            if wt > 0 { nValid += 1 }
        }
        return (g, w, nValid)
    }

    /// Inpaint unobserved footprint cells from known 4-neighbours (simultaneous updates), weight fpw.
    private static func fill(g: [Double], w: [Double], footprint: [Bool], fpw: [Double], footCells: [Int],
                             nu: Int, nv: Int, iters: Int) -> (h: [Double], w: [Double]) {
        var h = g
        var wt = w
        for k in 0..<h.count where !(wt[k] > 0) { h[k] = 0 }
        if iters <= 0 { return (h, wt) }
        var upd: [(Int, Double)] = []
        for _ in 0..<iters {
            upd.removeAll(keepingCapacity: true)
            for k in footCells {
                if wt[k] > 0 { continue }
                let iu = k / nv
                let iv = k - iu * nv
                var cnt = 0
                var sm = 0.0
                if iu > 0 && wt[k - nv] > 0 { cnt += 1; sm += h[k - nv] }
                if iu < nu - 1 && wt[k + nv] > 0 { cnt += 1; sm += h[k + nv] }
                if iv > 0 && wt[k - 1] > 0 { cnt += 1; sm += h[k - 1] }
                if iv < nv - 1 && wt[k + 1] > 0 { cnt += 1; sm += h[k + 1] }
                if cnt > 0 { upd.append((k, sm / Double(cnt))) }
            }
            if upd.isEmpty { break }
            for (k, val) in upd {
                h[k] = val
                wt[k] = fpw[k]
            }
        }
        return (h, wt)
    }

    private typealias BowlParams = (cu: Double, cv: Double, rimR: Double, depth: Double, base: Double)

    private static func volumeM3(h: [Double], w: [Double], nv: Int, u0: Double, v0: Double, cell: Double,
                                 bowl: BowlParams?) -> Double {
        var total = 0.0
        if let b = bowl {
            let rs = (b.rimR * b.rimR + b.depth * b.depth) / (2.0 * b.depth)
            for k in 0..<h.count where w[k] > 0 {
                let iu = k / nv
                let iv = k - iu * nv
                let cu = u0 + (Double(iu) + 0.5) * cell
                let cv = v0 + (Double(iv) + 0.5) * cell
                let r = (cu - b.cu) * (cu - b.cu) + (cv - b.cv) * (cv - b.cv)
                let rr = min(r.squareRoot(), b.rimR)
                let floorH = b.base + rs - (max(rs * rs - rr * rr, 0.0)).squareRoot()
                total += w[k] * max(h[k] - floorH, 0.0)
            }
        } else {
            for k in 0..<h.count where w[k] > 0 { total += w[k] * h[k] }
        }
        return total * cell * cell
    }

    /// Footprint area fraction for each candidate cell by table-level projection voting (see volume.py).
    private static func voteFootprint(cand: [Int], frames: [SweepFrame], itemKey: String, plane: Plane,
                                      aAx: SIMD3<Double>, bAx: SIMD3<Double>, u0: Double, v0: Double,
                                      nv: Int, cell: Double, vote: Double) -> [Double] {
        let sub = subN
        let nS = sub * sub
        let m = cand.count
        var pts = [SIMD3<Double>](repeating: SIMD3<Double>(0, 0, 0), count: m * nS)
        let base = plane.n * plane.offset
        for (ci, k) in cand.enumerated() {
            let iu = k / nv
            let iv = k - iu * nv
            let cu = u0 + (Double(iu) + 0.5) * cell
            let cv = v0 + (Double(iv) + 0.5) * cell
            for ia in 0..<sub {
                let du = (Double(ia) + 0.5) / Double(sub) - 0.5
                for ib in 0..<sub {
                    let dv = (Double(ib) + 0.5) / Double(sub) - 0.5
                    let pu = cu + du * cell
                    let pv = cv + dv * cell
                    pts[ci * nS + ia * sub + ib] = aAx * pu + bAx * pv + base
                }
            }
        }
        var ins = [Double](repeating: 0, count: m * nS)
        var see = [Double](repeating: 0, count: m * nS)
        for f in frames {
            guard let mask = f.masks[itemKey] else { continue }
            let mw = mask.width
            let mh = mask.height
            let invT = f.T_wc.inverse
            for i in 0..<pts.count {
                let pc = invT * SIMD4<Double>(pts[i].x, pts[i].y, pts[i].z, 1.0)
                let d = -pc.z
                if !(d > 1e-6) { continue }
                let uf = (f.K.fx * pc.x / d + f.K.cx).rounded(.toNearestOrEven)
                let vf = (f.K.cy - f.K.fy * pc.y / d).rounded(.toNearestOrEven)
                if !(uf >= 0 && vf >= 0 && uf < Double(mw) && vf < Double(mh)) { continue }
                let u = Int(uf)
                let v = Int(vf)
                see[i] += 1
                if mask.data[v * mw + u] { ins[i] += 1 }
            }
        }
        var out = [Double](repeating: 0, count: m)
        for ci in 0..<m {
            var acc = 0.0
            for s in 0..<nS {
                let i = ci * nS + s
                if see[i] > 0 && ins[i] >= vote * max(see[i], 1.0) { acc += 1.0 }
            }
            out[ci] = acc / Double(nS)
        }
        return out
    }

    // MARK: - main

    public static func computeItemVolume(frames: [SweepFrame], itemKey: String, plane: Plane, label: String,
                                         container: ContainerSpec?, cellM: Double = 0.003, minObs: Int = 3,
                                         nBoot: Int = 50, seed: UInt64 = 0) -> VolumeResult {
        var notes: [String] = []
        let cell = cellM

        // ---- clear cup / drink: never trust depth
        let shape: ContainerSpec.Shape? = container?.shape
        if drinkLabels.contains(label) || shape == .cup {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .cupPrior, nCells: 0, coverage: 0.0,
                                notes: ["depth unreliable through clear plastic"])
        }

        // ---- gather per-frame world points
        let (aAx, bAx) = planeBasis(plane.n)
        var pts: [PtsFrame] = []
        var pframes: [SweepFrame] = []
        var contU: [Double] = []
        var contV: [Double] = []
        var bowlLabelSeen = false
        for f in frames {
            guard let depth = f.depthM else { continue }
            let w = depth.width
            let hgt = depth.height
            let n = w * hgt
            if n == 0 { continue }
            var valid = [Bool](repeating: false, count: n)
            let confData: [Float]? = (f.conf != nil && f.conf!.data.count == n) ? f.conf!.data : nil
            for i in 0..<n {
                let d = depth.data[i]
                var ok = d.isFinite && d > 0
                if ok, let cd = confData, !(cd[i] >= 0.3) { ok = false }
                valid[i] = ok
            }
            if let m = f.masks[itemKey], m.width == w, m.height == hgt {
                var sel = [Bool](repeating: false, count: n)
                var hasAny = false
                for i in 0..<n where m.data[i] && valid[i] { sel[i] = true; hasAny = true }
                if hasAny {
                    let eroded = erode(m.data, width: w, height: hgt, px: erodePx)
                    let up = unprojectPlane(depth: depth, K: f.K, T: f.T_wc, sel: sel, aAx: aAx, bAx: bAx, plane: plane)
                    var er = [Bool](repeating: false, count: up.idx.count)
                    for j in 0..<up.idx.count { er[j] = eroded[up.idx[j]] }
                    pts.append(PtsFrame(pu: up.pu, pv: up.pv, h: up.h, er: er))
                    pframes.append(f)
                }
            }
            var contKeys = Set<String>()
            for (ck, cl) in f.labels where cl == "container_bowl" && ck != itemKey && f.masks[ck] != nil {
                bowlLabelSeen = true
                contKeys.insert(ck)
            }
            if let ck = container?.itemKey, ck != itemKey, f.masks[ck] != nil { contKeys.insert(ck) }
            for ck in contKeys {
                guard let cm = f.masks[ck], cm.width == w, cm.height == hgt else { continue }
                var cs = [Bool](repeating: false, count: n)
                var hasAny = false
                for i in 0..<n where cm.data[i] && valid[i] { cs[i] = true; hasAny = true }
                if hasAny {
                    let up = unprojectPlane(depth: depth, K: f.K, T: f.T_wc, sel: cs, aAx: aAx, bAx: bAx, plane: plane)
                    contU.append(contentsOf: up.pu)
                    contV.append(contentsOf: up.pv)
                }
            }
        }
        if pts.isEmpty {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .failed, nCells: 0, coverage: 0.0,
                                notes: ["no frames with depth and mask"])
        }

        // ---- reject garbage depth: far from the plane or far from the item in-plane
        var allU: [Double] = []
        var allV: [Double] = []
        for p in pts { allU.append(contentsOf: p.pu); allV.append(contentsOf: p.pv) }
        let mu = medianOf(allU)
        let mv = medianOf(allV)
        allU = []
        allV = []
        var cleaned: [PtsFrame] = []
        var cframes: [SweepFrame] = []
        for (p, fr) in zip(pts, pframes) {
            var q = PtsFrame(pu: [], pv: [], h: [], er: [])
            for j in 0..<p.pu.count {
                let hv = p.h[j]
                let dx = p.pu[j] - mu
                let dy = p.pv[j] - mv
                if hv > -0.05 && hv < 0.5 && (dx * dx + dy * dy).squareRoot() < 0.5 {
                    q.pu.append(p.pu[j])
                    q.pv.append(p.pv[j])
                    q.h.append(min(max(hv, 0.0), hMax))
                    q.er.append(p.er[j])
                }
            }
            if !q.pu.isEmpty {
                cleaned.append(q)
                cframes.append(fr)
            }
        }
        if cleaned.isEmpty {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .failed, nCells: 0, coverage: 0.0,
                                notes: ["all depth points implausible"])
        }
        pts = cleaned
        pframes = cframes

        var minU = Double.infinity, minV = Double.infinity
        var maxU = -Double.infinity, maxV = -Double.infinity
        for p in pts {
            for x in p.pu { minU = min(minU, x); maxU = max(maxU, x) }
            for x in p.pv { minV = min(minV, x); maxV = max(maxV, x) }
        }
        let u0 = minU - 3 * cell
        let v0 = minV - 3 * cell
        let nu = Int(((maxU - u0) / cell).rounded(.up)) + 4
        let nv = Int(((maxV - v0) / cell).rounded(.up)) + 4
        let ncell = nu * nv

        // ---- per-frame cell keys, presence, per-frame median per cell
        var pres = [Double](repeating: 0, count: ncell)
        var stamp = [Int](repeating: -1, count: ncell)
        var fpEr = [Bool](repeating: false, count: ncell)
        var entries: [(key: Int, frame: Int, h: Double)] = []
        var nf = 0
        for (fi, p) in pts.enumerated() {
            var pairs: [(Int, Double)] = []
            for j in 0..<p.pu.count {
                let iu = Int(((p.pu[j] - u0) / cell).rounded(.down))
                let iv = Int(((p.pv[j] - v0) / cell).rounded(.down))
                let k = iu * nv + iv
                if stamp[k] != fi {
                    stamp[k] = fi
                    pres[k] += 1
                }
                if p.er[j] {
                    pairs.append((k, p.h[j]))
                    fpEr[k] = true
                }
            }
            if pairs.isEmpty { continue }
            pairs.sort { (a, b) in a.0 != b.0 ? a.0 < b.0 : a.1 < b.1 }
            var s = 0
            while s < pairs.count {
                var e = s
                while e + 1 < pairs.count && pairs[e + 1].0 == pairs[s].0 { e += 1 }
                let cnt = e - s + 1
                let med = 0.5 * (pairs[s + (cnt - 1) / 2].1 + pairs[s + cnt / 2].1)
                entries.append((key: pairs[s].0, frame: nf, h: med))
                s = e + 1
            }
            nf += 1
        }
        if nf == 0 {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .failed, nCells: 0, coverage: 0.0,
                                notes: ["mask empty after erosion"])
        }
        var nFp = 0
        for b in fpEr where b { nFp += 1 }

        entries.sort { $0.key < $1.key }
        var obs = CellObs()
        obs.frame.reserveCapacity(entries.count)
        obs.h.reserveCapacity(entries.count)
        for (i, e) in entries.enumerated() {
            if i == 0 || e.key != entries[i - 1].key {
                if i != 0 { obs.start.append(i) }
                obs.keys.append(e.key)
            }
            obs.frame.append(e.frame)
            obs.h.append(e.h)
        }
        obs.start.append(entries.count)
        entries = []

        let noFoot = [Bool](repeating: false, count: ncell)
        let noW = [Double](repeating: 0, count: ncell)
        let allMult = [Int](repeating: 1, count: nf)

        let first = fuse(obs, mult: allMult, nSel: nf, ncell: ncell, minObs: minObs, footprint: noFoot, fpw: noW)
        let nValid = first.nValid
        if nValid < minValidCells {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .failed, nCells: nValid, coverage: 0.0,
                                notes: ["only \(nValid) cells with >= \(minObs) observations"])
        }
        let coverage = min(1.0, Double(nValid) / Double(max(nFp, 1)))

        // ---- footprint weights: presence frequency of the un-eroded mask, then table-level vote
        var fpw = [Double](repeating: 0, count: ncell)
        let nPts = Double(pts.count)
        for k in 0..<ncell where pres[k] > 0 {
            let pf = pres[k] / nPts
            fpw[k] = min(max((pf - fpFloor) / (fpSat - fpFloor), 0.0), 1.0)
        }
        if hullVote > 0 {
            var cand: [Int] = []
            for k in 0..<ncell where fpw[k] > 0 { cand.append(k) }
            let frac = voteFootprint(cand: cand, frames: pframes, itemKey: itemKey, plane: plane, aAx: aAx, bAx: bAx,
                                     u0: u0, v0: v0, nv: nv, cell: cell, vote: hullVote)
            for (ci, k) in cand.enumerated() { fpw[k] = fpw[k] * frac[ci] }
        }
        var footprint = [Bool](repeating: false, count: ncell)
        var footCells: [Int] = []
        for k in 0..<ncell where fpw[k] > 0 {
            footprint[k] = true
            footCells.append(k)
        }
        let full = fuse(obs, mult: allMult, nSel: nf, ncell: ncell, minObs: minObs, footprint: footprint, fpw: fpw)
        let filled = fill(g: full.g, w: full.w, footprint: footprint, fpw: fpw, footCells: footCells,
                          nu: nu, nv: nv, iters: fillIters)
        let gf = filled.h
        let wf = filled.w

        // ---- bowl?
        let isBowl = shape == .bowl || (shape == nil && bowlLabelSeen)
        var bowlTyp: BowlParams? = nil
        var method = VolumeMethod.heightmap
        var contSigma = 0.0
        if isBowl {
            method = .bowlCavity
            var rimTri: Triple
            if let r = container?.rimDiameterCm {
                rimTri = r
            } else {
                rimTri = defaultBowlRim
                if !contU.isEmpty {
                    let dU = percentileOf(contU, 98) - percentileOf(contU, 2)
                    let dV = percentileOf(contV, 98) - percentileOf(contV, 2)
                    let dEst = (0.5 * (dU + dV)) * 100.0
                    rimTri = Triple(min: dEst * 0.95, typ: dEst, max: dEst * 1.05)
                    notes.append("rim diameter estimated from container mask")
                } else {
                    notes.append("no container dims: generic medium-bowl prior used")
                }
            }
            let depthTri = container?.depthCm ?? defaultBowlDepth
            let dMin = rimTri.min * 0.01, dTyp = rimTri.typ * 0.01, dMax = rimTri.max * 0.01
            let hMin = depthTri.min * 0.01, hTyp = depthTri.typ * 0.01, hMax2 = depthTri.max * 0.01
            var centreU = 0.0
            var centreV = 0.0
            if !contU.isEmpty {
                centreU = 0.5 * (percentileOf(contU, 2) + percentileOf(contU, 98))
                centreV = 0.5 * (percentileOf(contV, 2) + percentileOf(contV, 98))
            } else {
                var su = 0.0, sv = 0.0
                var cnt = 0.0
                for k in 0..<ncell where wf[k] > 0 {
                    let iu = k / nv
                    let iv = k - iu * nv
                    su += u0 + (Double(iu) + 0.5) * cell
                    sv += v0 + (Double(iv) + 0.5) * cell
                    cnt += 1
                }
                if cnt > 0 { centreU = su / cnt; centreV = sv / cnt }
            }
            func bowlFn(_ rimD: Double, _ depth: Double) -> BowlParams {
                return (cu: centreU, cv: centreV, rimR: 0.5 * rimD, depth: depth, base: baseThickM)
            }
            let typ = bowlFn(dTyp, hTyp)
            bowlTyp = typ
            var vs: [Double] = []
            for d in [dMin, dMax] {
                for hd in [hMin, hMax2] {
                    vs.append(volumeM3(h: gf, w: wf, nv: nv, u0: u0, v0: v0, cell: cell, bowl: bowlFn(d, hd)))
                }
            }
            vs.append(volumeM3(h: gf, w: wf, nv: nv, u0: u0, v0: v0, cell: cell, bowl: typ))
            let vmin = vs.min() ?? 0
            let vmax = vs.max() ?? 0
            if vmin > 0 {
                contSigma = sigma90(vmin, vmax)
                if vmax / vmin > 2.0 { notes.append("food volume very sensitive to container dims") }
            }
        } else if shape == .plate || shape == .box {
            notes.append("container shape \(shape!.rawValue): plain heightmap")
        }

        let vTyp = volumeM3(h: gf, w: wf, nv: nv, u0: u0, v0: v0, cell: cell, bowl: bowlTyp)
        if !(vTyp > 0) {
            return VolumeResult(itemKey: itemKey, volumeMl: nil, method: .failed, nCells: nValid, coverage: coverage,
                                notes: ["non-positive volume"])
        }

        // ---- bootstrap over frames (footprint and fpw stay fixed)
        var sigmaBoot = 0.25
        if nf >= 2 && nBoot > 0 {
            var rng = SplitMix64(seed: seed)
            var vals: [Double] = []
            for _ in 0..<nBoot {
                var mult = [Int](repeating: 0, count: nf)
                for _ in 0..<nf {
                    let j = Int.random(in: 0..<nf, using: &rng)
                    mult[j] += 1
                }
                let fb = fuse(obs, mult: mult, nSel: nf, ncell: ncell, minObs: minObs, footprint: footprint, fpw: fpw)
                if fb.nValid < minValidCells { continue }
                let fl = fill(g: fb.g, w: fb.w, footprint: footprint, fpw: fpw, footCells: footCells,
                              nu: nu, nv: nv, iters: fillIters)
                let v = volumeM3(h: fl.h, w: fl.w, nv: nv, u0: u0, v0: v0, cell: cell, bowl: bowlTyp)
                if v > 0 { vals.append(v) }
            }
            if vals.count >= 10 {
                sigmaBoot = sigma90(percentileOf(vals, 5), percentileOf(vals, 95))
            } else {
                sigmaBoot = 0.25
                notes.append("bootstrap unstable (few frames): default spread used")
            }
        } else {
            sigmaBoot = 0.25
            notes.append("single frame: default spread used")
        }

        // ---- interval (log space)
        let sigmaScale = sigma90(pow(1.0 - scaleErr, 3.0), pow(1.0 + scaleErr, 3.0))
        sigmaBoot = min(sigmaBoot, 0.4)
        var sigma2 = sigmaBoot * sigmaBoot + sigmaScale * sigmaScale + contSigma * contSigma
        if coverage < 0.5 {
            let t = 0.5 * log(0.5 / max(coverage, 0.2))
            sigma2 += t * t + 0.1 * 0.1
            notes.append("low footprint coverage \(String(format: "%.2f", coverage)): interval widened, volume likely underestimated")
        }
        let sigma = sigma2.squareRoot()
        let typMl = vTyp * 1e6
        let tri = Triple(min: typMl * exp(-z90 * sigma), typ: typMl, max: typMl * exp(z90 * sigma))
        return VolumeResult(itemKey: itemKey, volumeMl: tri, method: method, nCells: nValid, coverage: coverage,
                            notes: notes)
    }
}
