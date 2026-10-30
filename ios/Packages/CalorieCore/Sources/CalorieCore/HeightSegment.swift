// Class-agnostic food segmentation by height above the table plane (native flow without a detector).
//
// Flow: alignFrames (no masks) -> HeightSegment.segment -> alignFrames again (masks exclude food from the anchors).
//
// Per aligned frame (depthM != nil):
//   1. unproject every valid pixel, height h = plane.height(p); candidate = minHeightM < h < 0.30 m (the cap rejects
//      garbage depth such as walls) and the pixel is not covered by an existing NON-"objN" mask;
//   2. 1-px morphological open (3x3 square erode then dilate; removes speckle and 1-2 px slivers);
//   3. 4-connected components; each gets a plane-projected area and a plane-coordinate centroid.
//      Area = sum over pixels of the exact orthographic area of the pixel's quad on the plane,
//      |det(P(u+1,v) - P(u,v), P(u,v+1) - P(u,v))| in plane coordinates (vertical depth jumps cancel, so side walls
//      of a box count ~0 and only top surfaces contribute). Where a neighbour is invalid or across a depth
//      discontinuity (> 1.5 cm) it falls back to (d/fx)(d/fy)/|n . ray| (the pixel footprint cut by a horizontal plane).
//      This is deliberately NOT the plain (d/fx)(d/fy)/|cos| formula, which over-counts the camera-facing walls of tall
//      objects by a strip h/tan(elevation) wide.
//   4. components with area >= minAreaM2 are clustered ACROSS frames greedily (largest first) by centroid distance
//      in plane coordinates (< mergeDistM, against the running mean centroid of the cluster).
//   5. clusters seen in >= max(2, ceil(30 % of aligned frames)) distinct frames become "obj0", "obj1", ... sorted by
//      median per-frame area (largest first). Their pixels are written to frames[i].masks[key] with label "unknown".
//
// Existing masks: segmentation masks from a previous call (keys obj<digits> labelled "unknown") are removed first so
// re-running is idempotent. Any other mask (detector output) is never modified and its pixels are excluded from the
// candidates, so a hybrid detector + height flow does not double count.
//
// Reported geometry: footprintAreaM2 = median per-frame summed area of the cluster; maxHeightM = median over frames
// of the 98th percentile of the component's heights; centroidW = plane-coordinate centroid (mean over member
// components) on the table lifted by maxHeightM / 2; topW = same point lifted by maxHeightM along the plane normal.
import Foundation
import simd

public struct SegmentedObject {
    public var key: String
    public var centroidW: SIMD3<Double>
    public var topW: SIMD3<Double>
    public var footprintAreaM2: Double
    public var maxHeightM: Double
    public var nFrames: Int

    public init(key: String, centroidW: SIMD3<Double>, topW: SIMD3<Double>, footprintAreaM2: Double,
                maxHeightM: Double, nFrames: Int) {
        self.key = key; self.centroidW = centroidW; self.topW = topW
        self.footprintAreaM2 = footprintAreaM2; self.maxHeightM = maxHeightM; self.nFrames = nFrames
    }
}

fileprivate struct SegComp {
    var frame: Int
    var label: Int
    var area: Double
    var cu: Double
    var cv: Double
    var hTop: Double
}

fileprivate struct SegCluster {
    var sumU = 0.0
    var sumV = 0.0
    var members: [Int] = []          // indices into the component array
}

public enum HeightSegment {
    static let maxHeightCapM = 0.30
    static let jumpM = 0.015

    public static func segment(_ frames: inout [SweepFrame], plane: Plane, minHeightM: Double = 0.006,
                               minAreaM2: Double = 0.0009, mergeDistM: Double = 0.03) -> [SegmentedObject] {
        let nrm = plane.n
        // orthonormal in-plane basis (a, b)
        var imin = 0
        if abs(nrm.y) < abs(nrm[imin]) { imin = 1 }
        if abs(nrm.z) < abs(nrm[imin]) { imin = 2 }
        var e = SIMD3<Double>(0, 0, 0)
        e[imin] = 1.0
        let ba = simd_normalize(simd_cross(nrm, e))
        let bb = simd_cross(nrm, ba)

        var nAligned = 0
        for f in frames where f.depthM != nil { nAligned += 1 }

        var comps: [SegComp] = []
        var labelMaps: [[Int32]] = Array(repeating: [], count: frames.count)
        var labelCounts: [Int] = Array(repeating: 0, count: frames.count)

        for fi in 0..<frames.count {
            // drop our own earlier output
            var stale: [String] = []
            for k in frames[fi].masks.keys where isSegmentKey(k, labels: frames[fi].labels) { stale.append(k) }
            for k in stale {
                frames[fi].masks[k] = nil
                frames[fi].labels[k] = nil
            }
            guard let depth = frames[fi].depthM else { continue }
            let W = frames[fi].width
            let H = frames[fi].height
            if depth.width != W || depth.height != H || W < 4 || H < 4 { continue }
            let n = W * H
            let K = frames[fi].K
            let T = frames[fi].T_wc
            let c0 = SIMD3<Double>(T.columns.0.x, T.columns.0.y, T.columns.0.z)
            let c1 = SIMD3<Double>(T.columns.1.x, T.columns.1.y, T.columns.1.z)
            let c2 = SIMD3<Double>(T.columns.2.x, T.columns.2.y, T.columns.2.z)
            let o = SIMD3<Double>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
            let m0 = simd_dot(c0, nrm)
            let m1 = simd_dot(c1, nrm)
            let m2 = simd_dot(c2, nrm)

            var taken = [Bool](repeating: false, count: n)
            for (_, m) in frames[fi].masks where m.width == W && m.height == H {
                for i in 0..<n where m.data[i] { taken[i] = true }
            }

            var dep = [Double](repeating: Double.nan, count: n)
            var hgt = [Double](repeating: Double.nan, count: n)
            var pu = [Double](repeating: Double.nan, count: n)
            var pv = [Double](repeating: Double.nan, count: n)
            var cand = [Bool](repeating: false, count: n)
            for v in 0..<H {
                let ry = -(Double(v) - K.cy) / K.fy
                for u in 0..<W {
                    let idx = v * W + u
                    let d = Double(depth.data[idx])
                    if !(d.isFinite && d > 0) { continue }
                    let rx = (Double(u) - K.cx) / K.fx
                    let p = o + c0 * (rx * d) + c1 * (ry * d) - c2 * d
                    let hh = simd_dot(p, nrm) - plane.offset
                    dep[idx] = d
                    hgt[idx] = hh
                    pu[idx] = simd_dot(p, ba)
                    pv[idx] = simd_dot(p, bb)
                    if hh > minHeightM && hh < maxHeightCapM && !taken[idx] { cand[idx] = true }
                }
            }
            let opened = AlignKernels.dilate(AlignKernels.erode(cand, w: W, h: H, r: 1), w: W, h: H, r: 1)

            var lab = [Int32](repeating: -1, count: n)
            var stack: [Int] = []
            var nextLabel = 0
            for start in 0..<n {
                if !opened[start] || lab[start] >= 0 { continue }
                let L = Int32(nextLabel)
                stack.removeAll(keepingCapacity: true)
                stack.append(start)
                lab[start] = L
                var sumA = 0.0
                var sumAU = 0.0
                var sumAV = 0.0
                var heights: [Double] = []
                while let cur = stack.popLast() {
                    let x = cur % W
                    let y = cur / W
                    if x > 0 {
                        let nb = cur - 1
                        if opened[nb] && lab[nb] < 0 { lab[nb] = L; stack.append(nb) }
                    }
                    if x < W - 1 {
                        let nb = cur + 1
                        if opened[nb] && lab[nb] < 0 { lab[nb] = L; stack.append(nb) }
                    }
                    if y > 0 {
                        let nb = cur - W
                        if opened[nb] && lab[nb] < 0 { lab[nb] = L; stack.append(nb) }
                    }
                    if y < H - 1 {
                        let nb = cur + W
                        if opened[nb] && lab[nb] < 0 { lab[nb] = L; stack.append(nb) }
                    }
                    // plane-projected area of this pixel
                    let d = dep[cur]
                    var area = Double.nan
                    if x + 1 < W && y + 1 < H {
                        let i1 = cur + 1
                        let i2 = cur + W
                        let d1 = dep[i1]
                        let d2 = dep[i2]
                        if d1.isFinite && d2.isFinite && abs(d1 - d) < jumpM && abs(d2 - d) < jumpM {
                            let ax = pu[i1] - pu[cur]
                            let ay = pv[i1] - pv[cur]
                            let bx = pu[i2] - pu[cur]
                            let by = pv[i2] - pv[cur]
                            area = abs(ax * by - ay * bx)
                        }
                    }
                    if !area.isFinite {
                        let rx = (Double(x) - K.cx) / K.fx
                        let ry = -(Double(y) - K.cy) / K.fy
                        let q = max(abs(m0 * rx + m1 * ry - m2), 0.1)
                        area = (d / K.fx) * (d / K.fy) / q
                    }
                    sumA += area
                    sumAU += area * pu[cur]
                    sumAV += area * pv[cur]
                    heights.append(hgt[cur])
                }
                if sumA >= minAreaM2 && sumA > 0 {
                    heights.sort()
                    let hTop = AlignKernels.percentile(heights, 98.0)
                    comps.append(SegComp(frame: fi, label: nextLabel, area: sumA, cu: sumAU / sumA,
                                         cv: sumAV / sumA, hTop: hTop))
                }
                nextLabel += 1
            }
            labelMaps[fi] = lab
            labelCounts[fi] = nextLabel
        }

        if comps.isEmpty { return [] }

        // greedy cross-frame clustering, largest component first
        let order = comps.indices.sorted { (i: Int, j: Int) -> Bool in
            if comps[i].area != comps[j].area { return comps[i].area > comps[j].area }
            return i < j
        }
        var clusters: [SegCluster] = []
        for ci in order {
            let c = comps[ci]
            var best = -1
            var bestDist = mergeDistM
            for k in 0..<clusters.count {
                let cnt = Double(clusters[k].members.count)
                let du = clusters[k].sumU / cnt - c.cu
                let dv = clusters[k].sumV / cnt - c.cv
                let dist = (du * du + dv * dv).squareRoot()
                if dist < bestDist {
                    bestDist = dist
                    best = k
                }
            }
            if best < 0 {
                var nc = SegCluster()
                nc.sumU = c.cu
                nc.sumV = c.cv
                nc.members = [ci]
                clusters.append(nc)
            } else {
                clusters[best].sumU += c.cu
                clusters[best].sumV += c.cv
                clusters[best].members.append(ci)
            }
        }

        let minFrames = max(2, Int((0.3 * Double(nAligned)).rounded(.up)))
        struct Valid {
            var cluster: Int
            var area: Double
            var maxH: Double
            var nFrames: Int
        }
        var valid: [Valid] = []
        for (k, cl) in clusters.enumerated() {
            var areaByFrame: [Int: Double] = [:]
            var topByFrame: [Int: Double] = [:]
            for mi in cl.members {
                let c = comps[mi]
                areaByFrame[c.frame, default: 0.0] += c.area
                topByFrame[c.frame] = max(topByFrame[c.frame] ?? -Double.infinity, c.hTop)
            }
            if areaByFrame.count < minFrames { continue }
            let areas = areaByFrame.values.sorted()
            let tops = topByFrame.values.sorted()
            valid.append(Valid(cluster: k, area: median(areas), maxH: median(tops), nFrames: areaByFrame.count))
        }
        if valid.isEmpty { return [] }
        valid.sort { (a: Valid, b: Valid) -> Bool in
            if a.area != b.area { return a.area > b.area }
            return a.cluster < b.cluster
        }

        var objects: [SegmentedObject] = []
        var rankOfCluster: [Int: Int] = [:]
        for (rank, v) in valid.enumerated() {
            let cl = clusters[v.cluster]
            let cnt = Double(cl.members.count)
            let cu = cl.sumU / cnt
            let cv = cl.sumV / cnt
            let onPlane = ba * cu + bb * cv + nrm * plane.offset
            objects.append(SegmentedObject(key: "obj\(rank)",
                                           centroidW: onPlane + nrm * (v.maxH * 0.5),
                                           topW: onPlane + nrm * v.maxH,
                                           footprintAreaM2: v.area,
                                           maxHeightM: v.maxH,
                                           nFrames: v.nFrames))
            rankOfCluster[v.cluster] = rank
        }

        // masks
        var rankOfLabel: [[Int]] = []
        for fi in 0..<frames.count { rankOfLabel.append([Int](repeating: -1, count: labelCounts[fi])) }
        for (k, cl) in clusters.enumerated() {
            guard let rank = rankOfCluster[k] else { continue }
            for mi in cl.members {
                let c = comps[mi]
                rankOfLabel[c.frame][c.label] = rank
            }
        }
        for fi in 0..<frames.count {
            let lab = labelMaps[fi]
            if lab.isEmpty { continue }
            let W = frames[fi].width
            let H = frames[fi].height
            var perRank = [[Bool]?](repeating: nil, count: objects.count)
            for idx in 0..<lab.count {
                let l = Int(lab[idx])
                if l < 0 { continue }
                let r = rankOfLabel[fi][l]
                if r < 0 { continue }
                if perRank[r] == nil { perRank[r] = [Bool](repeating: false, count: W * H) }
                perRank[r]![idx] = true
            }
            for r in 0..<objects.count {
                if let data = perRank[r] {
                    frames[fi].masks[objects[r].key] = Mask(width: W, height: H, data: data)
                    frames[fi].labels[objects[r].key] = "unknown"
                }
            }
        }
        return objects
    }

    static func isSegmentKey(_ key: String, labels: [String: String]) -> Bool {
        if !key.hasPrefix("obj") { return false }
        let rest = key.dropFirst(3)
        if rest.isEmpty { return false }
        for ch in rest where !(ch >= "0" && ch <= "9") { return false }
        return (labels[key] ?? "unknown") == "unknown"
    }

    static func median(_ sorted: [Double]) -> Double {
        if sorted.isEmpty { return 0 }
        let m = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[m] }
        return 0.5 * (sorted[m - 1] + sorted[m])
    }
}
