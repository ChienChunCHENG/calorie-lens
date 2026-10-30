// Shared types for the sweep / alignment / volume pipeline. Binding contract: spec/swift_contract.md.
// Transliteration of tools/replay/cl_types.py (ARKit conventions):
//   * world and camera frames right-handed, metres; camera +X right, +Y up, looks along -Z.
//   * pixel (u, v) = (column, row), top-left origin, v down, in the SENSOR (landscape) orientation of ARFrame.capturedImage.
//   * K is at DEPTH-MAP resolution. Unproject: x = (u-cx)d/fx, y = -(v-cy)d/fy, z = -d; p_w = T_wc * [x y z 1].
//   * Plane normal points up; height(p) = n.p - offset (> 0 above the table).
import Foundation
import simd

/// Row-major 2D grid. `self[u, v]` is column u, row v (index v*width + u).
public struct Grid<T> {
    public let width: Int
    public let height: Int
    public var data: [T]

    public init(width: Int, height: Int, repeating value: T) {
        self.width = width
        self.height = height
        self.data = Array(repeating: value, count: width * height)
    }

    public init(width: Int, height: Int, data: [T]) {
        precondition(data.count == width * height, "Grid data count \(data.count) != \(width)x\(height)")
        self.width = width
        self.height = height
        self.data = data
    }

    @inline(__always) public subscript(u: Int, v: Int) -> T {
        get { data[v * width + u] }
        set { data[v * width + u] = newValue }
    }

    @inline(__always) public func contains(u: Int, v: Int) -> Bool { u >= 0 && v >= 0 && u < width && v < height }
}

/// Float grid; NaN marks an invalid pixel.
public typealias FloatGrid = Grid<Float>
public typealias Mask = Grid<Bool>

/// Pinhole intrinsics (pixels). Use `scaled` to go from ARKit's full-resolution K to depth-map resolution.
public struct Intrinsics: Equatable, Codable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double

    public init(fx: Double, fy: Double, cx: Double, cy: Double) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
    }

    /// Row-major 3x3 as stored in dumps: fx, 0, cx, 0, fy, cy, 0, 0, 1.
    public init(rowMajor k: [Double]) {
        precondition(k.count == 9)
        self.init(fx: k[0], fy: k[4], cx: k[2], cy: k[5])
    }

    public var rowMajor: [Double] { [fx, 0, cx, 0, fy, cy, 0, 0, 1] }

    public func scaled(sx: Double, sy: Double) -> Intrinsics {
        Intrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy)
    }
}

/// One keyframe of a sweep (Python `Frame`).
public struct SweepFrame {
    public var t: Double                         // seconds since sweep start
    public var T_wc: simd_double4x4              // camera-to-world (ARKit frame.camera.transform, as Double)
    public var K: Intrinsics                     // at depth-map resolution
    public var relDepth: FloatGrid               // RELATIVE depth, disparity-like: larger = nearer, affine in 1/d
    public var depthM: FloatGrid?                // METRIC depth (filled by Align), NaN = invalid
    public var conf: FloatGrid?                  // optional per-pixel confidence in [0, 1]
    public var masks: [String: Mask]             // item key -> mask at depth resolution
    public var labels: [String: String]          // item key -> class id (ml/labels.yaml) or "unknown"
    public var featurePointsW: [SIMD3<Double>]   // ARKit rawFeaturePoints (world) visible in this frame

    public init(t: Double, T_wc: simd_double4x4, K: Intrinsics, relDepth: FloatGrid, depthM: FloatGrid? = nil,
                conf: FloatGrid? = nil, masks: [String: Mask] = [:], labels: [String: String] = [:],
                featurePointsW: [SIMD3<Double>] = []) {
        self.t = t; self.T_wc = T_wc; self.K = K; self.relDepth = relDepth; self.depthM = depthM
        self.conf = conf; self.masks = masks; self.labels = labels; self.featurePointsW = featurePointsW
    }

    public var width: Int { relDepth.width }
    public var height: Int { relDepth.height }
    public var cameraCenterW: SIMD3<Double> { SIMD3(T_wc.columns.3.x, T_wc.columns.3.y, T_wc.columns.3.z) }
}

/// Table plane: unit normal pointing up, n.p = offset on the plane.
public struct Plane: Equatable, Codable {
    public var n: SIMD3<Double>
    public var offset: Double
    public var inlierRMSEm: Double

    public init(n: SIMD3<Double>, offset: Double, inlierRMSEm: Double = 0) {
        self.n = n; self.offset = offset; self.inlierRMSEm = inlierRMSEm
    }

    @inline(__always) public func height(_ p: SIMD3<Double>) -> Double { simd_dot(p, n) - offset }

    public var flipped: Plane { Plane(n: -n, offset: -offset, inlierRMSEm: inlierRMSEm) }
}

/// 90 % interval with a typical value (min <= typ <= max).
public struct Triple: Equatable, Codable {
    public var min: Double
    public var typ: Double
    public var max: Double

    public init(min: Double, typ: Double, max: Double) {
        self.min = min; self.typ = typ; self.max = max
    }

    public init(scalar x: Double) { self.init(min: x, typ: x, max: x) }

    public func scaled(_ k: Double) -> Triple { Triple(min: min * k, typ: typ * k, max: max * k) }
}

/// Container geometry passed to the volume step (Python `container` dict). Dimensions in cm.
public struct ContainerSpec: Equatable {
    public enum Shape: String { case bowl, plate, box, cup }
    public var shape: Shape
    public var itemKey: String?            // key of the container's own mask, if any
    public var rimDiameterCm: Triple?
    public var depthCm: Triple?

    public init(shape: Shape, itemKey: String? = nil, rimDiameterCm: Triple? = nil, depthCm: Triple? = nil) {
        self.shape = shape; self.itemKey = itemKey; self.rimDiameterCm = rimDiameterCm; self.depthCm = depthCm
    }
}

public enum VolumeMethod: String, Codable {
    case heightmap
    case bowlCavity = "bowl_cavity"
    case cupPrior = "cup_prior"
    case failed
}

public struct VolumeResult {
    public var itemKey: String
    public var volumeMl: Triple?           // nil => failed or cup prior; caller falls back to container/default portion
    public var method: VolumeMethod
    public var nCells: Int
    public var coverage: Double
    public var notes: [String]

    public init(itemKey: String, volumeMl: Triple?, method: VolumeMethod, nCells: Int = 0, coverage: Double = 0,
                notes: [String] = []) {
        self.itemKey = itemKey; self.volumeMl = volumeMl; self.method = method
        self.nCells = nCells; self.coverage = coverage; self.notes = notes
    }
}

/// Per-frame alignment diagnostics (Python align_frame's dict).
public struct AlignDiag {
    public var ok = false
    public var reason = ""
    public var s: Double?
    public var t: Double?
    public var nBg = 0
    public var nFp = 0
    public var nInl = 0
    public var inlierFrac = 0.0
    public var rmseM: Double?
    public var relSpread: Double?

    public init() {}
}

/// Deterministic RNG so the bootstrap / RANSAC are reproducible on device and in tests.
/// (Does NOT reproduce numpy's streams: compare Swift vs Python results with tolerances, never bit-exact.)
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

extension simd_double4x4 {
    /// From 16 row-major numbers (dump format: T[r][c] at 4r + c).
    public init(rowMajor m: [Double]) {
        precondition(m.count == 16)
        self.init(columns: (SIMD4(m[0], m[4], m[8], m[12]), SIMD4(m[1], m[5], m[9], m[13]),
                            SIMD4(m[2], m[6], m[10], m[14]), SIMD4(m[3], m[7], m[11], m[15])))
    }

    public var rowMajor: [Double] {
        var out: [Double] = []
        out.reserveCapacity(16)
        for r in 0..<4 { for c in 0..<4 { out.append(self[c][r]) } }
        return out
    }

    /// From ARKit's Float transform.
    public init(_ m: simd_float4x4) {
        self.init(columns: (SIMD4<Double>(m.columns.0), SIMD4<Double>(m.columns.1),
                            SIMD4<Double>(m.columns.2), SIMD4<Double>(m.columns.3)))
    }
}
