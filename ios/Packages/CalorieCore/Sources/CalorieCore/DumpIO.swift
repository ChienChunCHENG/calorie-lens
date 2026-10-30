// Sweep dump reader / writer. Exactly tools/replay/DUMP_FORMAT.md (version 1); Python dump_io.py reads what this
// writes and vice versa.
//
// All binary files are little-endian. Bytes are assembled / decoded explicitly (shift and or), so the code does not
// depend on the host endianness. Float16 is used for rel_NN.f16 (available on arm64 macOS and on iOS).
import Foundation
import simd
import CoreGraphics
import ImageIO

public struct DumpMeta {
    public var deviceModel: String? = nil
    public var iosVersion: String? = nil
    public var imageSize: [Int]? = nil
    public var depthModel: String? = nil
    public var plane: Plane? = nil
    public var planeSource: String? = nil
    public var containers: [String: ContainerSpec] = [:]
    public init() {}
}

public enum DumpError: Error, CustomStringConvertible {
    case invalidKey(String)
    case invalid(String)

    public var description: String {
        switch self {
        case .invalidKey(let k): return "DumpIO: item key '\(k)' must match [A-Za-z0-9_-]+"
        case .invalid(let m): return "DumpIO: \(m)"
        }
    }
}

public enum DumpIO {
    public static let formatVersion = 1

    // MARK: write

    /// Writes `frames` into the directory `dir` (created if needed). `rgbJPEGs[i]` is the optional keyframe image of
    /// frame i. With `includeDepth` the metric `depthM` grids (if present) are written as depth_NN.f32 (debug only).
    public static func write(frames: [SweepFrame], meta: DumpMeta, rgbJPEGs: [Data?]?, includeDepth: Bool = false,
                             to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        var entries: [[String: Any]] = []

        for (i, f) in frames.enumerated() {
            let idx = twoDigits(i)
            let w = f.relDepth.width
            let h = f.relDepth.height
            guard w > 0, h > 0 else { throw DumpError.invalid("frame \(i) has an empty relDepth grid") }
            let tw = f.T_wc.rowMajor
            if !tw.allSatisfy({ $0.isFinite }) { throw DumpError.invalid("frame \(i) has a non-finite T_wc") }
            let kk = f.K.rowMajor
            if !kk.allSatisfy({ $0.isFinite }) { throw DumpError.invalid("frame \(i) has a non-finite K") }
            if !f.t.isFinite { throw DumpError.invalid("frame \(i) has a non-finite t") }

            var e: [String: Any] = ["index": i, "t": f.t, "T_wc": tw, "K": kk]

            // relative depth, float16 little-endian
            let relName = "rel_\(idx).f16"
            try encodeHalf(f.relDepth.data).write(to: dir.appendingPathComponent(relName))
            e["rel"] = ["file": relName, "shape": [h, w]] as [String: Any]

            // masks (sorted for a deterministic manifest)
            var maskEntries: [String: Any] = [:]
            for key in f.masks.keys.sorted() {
                guard isValidKey(key) else { throw DumpError.invalidKey(key) }
                guard let m = f.masks[key] else { continue }
                guard m.width == w, m.height == h else {
                    throw DumpError.invalid("mask '\(key)' of frame \(i) is \(m.width)x\(m.height), rel is \(w)x\(h)")
                }
                let mName = "mask_\(idx)_\(key).png"
                try encodeMaskPNG(m).write(to: dir.appendingPathComponent(mName))
                var me: [String: Any] = ["file": mName]
                if let lab = f.labels[key] { me["label"] = lab } else { me["label"] = NSNull() }
                maskEntries[key] = me
            }
            e["masks"] = maskEntries

            // feature points, float32 little-endian
            let fpName = "fp_\(idx).f32"
            var fpFloats: [Float] = []
            fpFloats.reserveCapacity(f.featurePointsW.count * 3)
            for p in f.featurePointsW {
                fpFloats.append(Float(p.x)); fpFloats.append(Float(p.y)); fpFloats.append(Float(p.z))
            }
            try encodeFloat32(fpFloats).write(to: dir.appendingPathComponent(fpName))
            e["feature_points"] = ["file": fpName, "count": f.featurePointsW.count] as [String: Any]

            // optional rgb
            if let all = rgbJPEGs, i < all.count, let jpg = all[i] {
                let rName = "rgb_\(idx).jpg"
                try jpg.write(to: dir.appendingPathComponent(rName))
                e["rgb"] = rName
            } else {
                e["rgb"] = NSNull()
            }

            // optional metric depth (debug)
            if includeDepth, let d = f.depthM, d.width == w, d.height == h {
                let dName = "depth_\(idx).f32"
                try encodeFloat32(d.data).write(to: dir.appendingPathComponent(dName))
                e["depth"] = ["file": dName, "shape": [h, w]] as [String: Any]
            }
            entries.append(e)
        }

        var manifest: [String: Any] = [
            "version": formatVersion,
            "depth_semantics": "disparity_larger_is_nearer",
            "frames": entries,
        ]
        manifest["device_model"] = orNull(meta.deviceModel)
        manifest["ios_version"] = orNull(meta.iosVersion)
        manifest["image_size"] = orNull(meta.imageSize)
        manifest["depth_model"] = orNull(meta.depthModel)
        if let p = meta.plane {
            if !(p.n.x.isFinite && p.n.y.isFinite && p.n.z.isFinite && p.offset.isFinite) {
                throw DumpError.invalid("plane is not finite")
            }
            manifest["plane"] = ["n": [p.n.x, p.n.y, p.n.z], "offset": p.offset,
                                 "source": meta.planeSource ?? "unknown"] as [String: Any]
        } else {
            manifest["plane"] = NSNull()
        }
        var conts: [String: Any] = [:]
        for key in meta.containers.keys.sorted() {
            guard isValidKey(key) else { throw DumpError.invalidKey(key) }
            guard let c = meta.containers[key] else { continue }
            var ce: [String: Any] = ["shape": c.shape.rawValue]
            if let ik = c.itemKey { ce["item_key"] = ik }
            if let r = c.rimDiameterCm { ce["rim_diameter_cm"] = tripleDict(r) }
            if let d = c.depthCm { ce["depth_cm"] = tripleDict(d) }
            conts[key] = ce
        }
        manifest["containers"] = conts

        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
    }

    // MARK: read

    /// Reads a dump DIRECTORY (manifest.json at its root).
    public static func read(_ dir: URL) throws -> (frames: [SweepFrame], meta: DumpMeta) {
        let manifestData = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        guard let root = try JSONSerialization.jsonObject(with: manifestData, options: []) as? [String: Any] else {
            throw DumpError.invalid("manifest.json is not a JSON object")
        }
        guard let version = intValue(root["version"]), version == formatVersion else {
            throw DumpError.invalid("unsupported dump version \(String(describing: root["version"]))")
        }

        var meta = DumpMeta()
        meta.deviceModel = root["device_model"] as? String
        meta.iosVersion = root["ios_version"] as? String
        meta.depthModel = root["depth_model"] as? String
        if let sz = root["image_size"] as? [Any] {
            meta.imageSize = sz.compactMap { intValue($0) }
        }
        if let pl = root["plane"] as? [String: Any], let nArr = pl["n"] as? [Any], nArr.count == 3,
           let off = doubleValue(pl["offset"]) {
            let n = nArr.compactMap { doubleValue($0) }
            if n.count == 3 {
                meta.plane = Plane(n: SIMD3<Double>(n[0], n[1], n[2]), offset: off, inlierRMSEm: 0)
                meta.planeSource = pl["source"] as? String
            }
        }
        if let conts = root["containers"] as? [String: Any] {
            for (key, v) in conts {
                guard let c = v as? [String: Any], let shapeStr = c["shape"] as? String,
                      let shape = ContainerSpec.Shape(rawValue: shapeStr) else { continue }
                meta.containers[key] = ContainerSpec(shape: shape, itemKey: c["item_key"] as? String,
                                                     rimDiameterCm: parseTriple(c["rim_diameter_cm"]),
                                                     depthCm: parseTriple(c["depth_cm"]))
            }
        }

        guard let entries = root["frames"] as? [[String: Any]] else {
            throw DumpError.invalid("manifest has no frames array")
        }
        var frames: [SweepFrame] = []
        for (i, e) in entries.enumerated() {
            guard let relInfo = e["rel"] as? [String: Any], let relFile = relInfo["file"] as? String,
                  let shapeArr = relInfo["shape"] as? [Any], shapeArr.count == 2,
                  let h = intValue(shapeArr[0]), let w = intValue(shapeArr[1]), w > 0, h > 0 else {
                throw DumpError.invalid("frame \(i): bad rel entry")
            }
            guard let t = doubleValue(e["t"]),
                  let twArr = e["T_wc"] as? [Any], twArr.count == 16,
                  let kArr = e["K"] as? [Any], kArr.count == 9 else {
                throw DumpError.invalid("frame \(i): bad t / T_wc / K")
            }
            let tw = twArr.compactMap { doubleValue($0) }
            let kk = kArr.compactMap { doubleValue($0) }
            guard tw.count == 16, kk.count == 9 else { throw DumpError.invalid("frame \(i): non-numeric T_wc / K") }

            let relBytes = [UInt8](try Data(contentsOf: dir.appendingPathComponent(relFile)))
            guard relBytes.count == w * h * 2 else {
                throw DumpError.invalid("\(relFile): \(relBytes.count) bytes, expected \(w * h * 2)")
            }
            let rel = FloatGrid(width: w, height: h, data: decodeHalf(relBytes))

            var masks: [String: Mask] = [:]
            var labels: [String: String] = [:]
            if let me = e["masks"] as? [String: Any] {
                for (key, v) in me {
                    guard let m = v as? [String: Any], let file = m["file"] as? String else {
                        throw DumpError.invalid("frame \(i): bad mask entry '\(key)'")
                    }
                    let png = try Data(contentsOf: dir.appendingPathComponent(file))
                    masks[key] = try decodeMaskPNG(png, width: w, height: h, name: file)
                    if let lab = m["label"] as? String { labels[key] = lab }
                }
            }

            var fps: [SIMD3<Double>] = []
            if let fpInfo = e["feature_points"] as? [String: Any], let fpFile = fpInfo["file"] as? String {
                let fb = [UInt8](try Data(contentsOf: dir.appendingPathComponent(fpFile)))
                guard fb.count % 12 == 0 else { throw DumpError.invalid("\(fpFile): size not a multiple of 12") }
                let fl = decodeFloat32(fb)
                fps.reserveCapacity(fl.count / 3)
                var j = 0
                while j + 2 < fl.count {
                    fps.append(SIMD3<Double>(Double(fl[j]), Double(fl[j + 1]), Double(fl[j + 2])))
                    j += 3
                }
            }

            var depth: FloatGrid? = nil
            if let dInfo = e["depth"] as? [String: Any], let dFile = dInfo["file"] as? String {
                let db = [UInt8](try Data(contentsOf: dir.appendingPathComponent(dFile)))
                if db.count == w * h * 4 { depth = FloatGrid(width: w, height: h, data: decodeFloat32(db)) }
            }

            frames.append(SweepFrame(t: t, T_wc: simd_double4x4(rowMajor: tw), K: Intrinsics(rowMajor: kk),
                                     relDepth: rel, depthM: depth, conf: nil, masks: masks, labels: labels,
                                     featurePointsW: fps))
        }
        return (frames, meta)
    }

    // MARK: zip

    /// Zips the directory with NSFileCoordinator (.forUploading) and returns a copy `<dirname>.zip` in the temporary
    /// directory (the coordinator's own file is deleted when the accessor returns). The archive contains one
    /// top-level folder named like `dir`, which Python's loader accepts.
    public static func zip(_ dir: URL) throws -> URL {
        let coordinator = NSFileCoordinator(filePresenter: nil)
        var coordError: NSError? = nil
        var result: Result<URL, Error>? = nil
        let name = dir.lastPathComponent + ".zip"
        coordinator.coordinate(readingItemAt: dir, options: [.forUploading], error: &coordError) { zipURL in
            do {
                let dest = FileManager.default.temporaryDirectory.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: zipURL, to: dest)
                result = .success(dest)
            } catch {
                result = .failure(error)
            }
        }
        if let ce = coordError { throw ce }
        guard let r = result else { throw DumpError.invalid("zip: coordinator produced no archive") }
        switch r {
        case .success(let url): return url
        case .failure(let err): throw err
        }
    }

    // MARK: helpers

    private static func orNull(_ x: Any?) -> Any { return x ?? NSNull() }

    private static func twoDigits(_ i: Int) -> String { i < 10 ? "0\(i)" : "\(i)" }

    private static func isValidKey(_ key: String) -> Bool {
        if key.isEmpty { return false }
        for s in key.unicodeScalars {
            let v = s.value
            let ok = (v >= 48 && v <= 57) || (v >= 65 && v <= 90) || (v >= 97 && v <= 122) || v == 95 || v == 45
            if !ok { return false }
        }
        return true
    }

    private static func tripleDict(_ t: Triple) -> [String: Any] {
        return ["min": t.min, "typ": t.typ, "max": t.max]
    }

    /// Accepts a scalar (min = typ = max) or {"min", "typ", "max"}.
    private static func parseTriple(_ v: Any?) -> Triple? {
        guard let v = v else { return nil }
        if let d = v as? [String: Any] {
            guard let mn = doubleValue(d["min"]), let ty = doubleValue(d["typ"]), let mx = doubleValue(d["max"]) else {
                return nil
            }
            return Triple(min: mn, typ: ty, max: mx)
        }
        if let x = doubleValue(v) { return Triple(scalar: x) }
        return nil
    }

    private static func doubleValue(_ v: Any?) -> Double? {
        if let n = v as? NSNumber { return n.doubleValue }
        return nil
    }

    private static func intValue(_ v: Any?) -> Int? {
        if let n = v as? NSNumber { return n.intValue }
        return nil
    }

    private static func encodeHalf(_ values: [Float]) -> Data {
        var bytes = [UInt8](repeating: 0, count: values.count * 2)
        for (i, v) in values.enumerated() {
            let bits = Float16(v).bitPattern
            bytes[2 * i] = UInt8(truncatingIfNeeded: bits)
            bytes[2 * i + 1] = UInt8(truncatingIfNeeded: bits >> 8)
        }
        return Data(bytes)
    }

    private static func decodeHalf(_ bytes: [UInt8]) -> [Float] {
        let n = bytes.count / 2
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let bits = UInt16(bytes[2 * i]) | (UInt16(bytes[2 * i + 1]) << 8)
            out[i] = Float(Float16(bitPattern: bits))
        }
        return out
    }

    private static func encodeFloat32(_ values: [Float]) -> Data {
        var bytes = [UInt8](repeating: 0, count: values.count * 4)
        for (i, v) in values.enumerated() {
            let bits = v.bitPattern
            bytes[4 * i] = UInt8(truncatingIfNeeded: bits)
            bytes[4 * i + 1] = UInt8(truncatingIfNeeded: bits >> 8)
            bytes[4 * i + 2] = UInt8(truncatingIfNeeded: bits >> 16)
            bytes[4 * i + 3] = UInt8(truncatingIfNeeded: bits >> 24)
        }
        return Data(bytes)
    }

    private static func decodeFloat32(_ bytes: [UInt8]) -> [Float] {
        let n = bytes.count / 4
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var bits = UInt32(bytes[4 * i])
            bits |= UInt32(bytes[4 * i + 1]) << 8
            bits |= UInt32(bytes[4 * i + 2]) << 16
            bits |= UInt32(bytes[4 * i + 3]) << 24
            out[i] = Float(bitPattern: bits)
        }
        return out
    }

    /// 8-bit grayscale PNG, 255 where the mask is set, else 0.
    private static func encodeMaskPNG(_ mask: Mask) throws -> Data {
        let w = mask.width
        let h = mask.height
        var pixels = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) where mask.data[i] { pixels[i] = 255 }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            throw DumpError.invalid("PNG: cannot create data provider")
        }
        guard let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: CGColorRenderingIntent.defaultIntent) else {
            throw DumpError.invalid("PNG: cannot create CGImage")
        }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out as CFMutableData, "public.png" as CFString, 1, nil) else {
            throw DumpError.invalid("PNG: cannot create image destination")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw DumpError.invalid("PNG: encoding failed") }
        return out as Data
    }

    /// Decodes any PNG to 8-bit gray and thresholds at > 127.
    private static func decodeMaskPNG(_ data: Data, width: Int, height: Int, name: String) throws -> Mask {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw DumpError.invalid("\(name): cannot decode PNG")
        }
        guard image.width == width, image.height == height else {
            throw DumpError.invalid("\(name): PNG is \(image.width)x\(image.height), expected \(width)x\(height)")
        }
        var buf = [UInt8](repeating: 0, count: width * height)
        let drawn: Bool = buf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw DumpError.invalid("\(name): cannot create gray bitmap context") }
        return Mask(width: width, height: height, data: buf.map { $0 > 127 })
    }
}
