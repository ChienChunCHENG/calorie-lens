import CalorieCore
import CoreImage
import CoreML
import CoreVideo
import Foundation

// MARK: - Shared helpers (also used by Detector)

/// Resizes a camera pixel buffer to an exact size as 32BGRA with Core Image. The image is STRETCHED (no letterbox):
/// the full sensor frame maps onto the full model input, so the camera intrinsics only need a pure per-axis scale
/// (fx, cx by outW/imgW; fy, cy by outH/imgH). Orientation is untouched (sensor / landscape).
final class PixelBufferResizer {
    private let context = CIContext(options: nil)

    func resize(_ src: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        if srcW <= 0 || srcH <= 0 || width <= 0 || height <= 0 { return nil }

        var out: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &out)
        guard status == kCVReturnSuccess, let dst = out else { return nil }

        let sx = CGFloat(width) / CGFloat(srcW)
        let sy = CGFloat(height) / CGFloat(srcH)
        let image = CIImage(cvPixelBuffer: src).transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        context.render(image, to: dst, bounds: CGRect(x: 0, y: 0, width: width, height: height),
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return dst
    }
}

enum MLArrayReader {
    /// IEEE half (bit pattern) to Float, without relying on the Float16 type.
    @inline(__always) static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        var exp = Int32((h >> 10) & 0x1F)
        var man = UInt32(h & 0x3FF)
        if exp == 0 {
            if man == 0 { return Float(bitPattern: sign) }
            exp = 1
            while (man & 0x400) == 0 {
                man <<= 1
                exp -= 1
            }
            man &= 0x3FF
        } else if exp == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (man << 13))
        }
        let e = UInt32(exp + 112)
        return Float(bitPattern: sign | (e << 23) | (man << 13))
    }

    /// All elements in logical row-major order, honouring strides and the data type.
    static func floats(_ ma: MLMultiArray) -> [Float] {
        let count = ma.count
        if count <= 0 { return [] }
        let shape = ma.shape.map { $0.intValue }
        let strides = ma.strides.map { $0.intValue }
        let dt = ma.dataType
        var out = [Float](repeating: 0, count: count)
        ma.withUnsafeBytes { raw in
            guard raw.baseAddress != nil else { return }
            var counter = [Int](repeating: 0, count: shape.count)
            var off = 0
            for i in 0..<count {
                var value: Float = 0
                switch dt {
                case .float32:
                    value = raw.load(fromByteOffset: off * 4, as: Float.self)
                case .float16:
                    value = halfToFloat(raw.load(fromByteOffset: off * 2, as: UInt16.self))
                case .double:
                    value = Float(raw.load(fromByteOffset: off * 8, as: Double.self))
                case .int32:
                    value = Float(raw.load(fromByteOffset: off * 4, as: Int32.self))
                default:
                    value = 0
                }
                out[i] = value
                var d = shape.count - 1
                while d >= 0 {
                    counter[d] += 1
                    off += strides[d]
                    if counter[d] < shape[d] { break }
                    off -= strides[d] * shape[d]
                    counter[d] = 0
                    d -= 1
                }
            }
        }
        return out
    }

    static func describe(_ d: MLFeatureDescription) -> String {
        var s = "\(d.name): type=\(d.type.rawValue)"
        if let ic = d.imageConstraint {
            s += " image \(ic.pixelsWide)x\(ic.pixelsHigh) pixelFormat=\(ic.pixelFormatType)"
        }
        if let mc = d.multiArrayConstraint {
            s += " multiArray shape=\(mc.shape.map { $0.intValue }) dtype=\(mc.dataType.rawValue)"
        }
        return s
    }
}

// MARK: - Depth Anything V2 Small

/// Apple's coreml-depth-anything-v2-small (DepthAnythingV2SmallF16).
///
/// Model facts (HF apple/coreml-depth-anything-v2-small, model.mlmodel protobuf):
///   input  "image": RGB image, width 518 x height 392 (sensor landscape frame stretched onto it); the model
///          doc string says the larger dimension should be a multiple of 14.
///   output "depth": grayscale Float16 image (OneComponent16Half) 518 x 392; relative, unnormalised.
/// We read the real names / sizes from `modelDescription` at runtime and log them. The camera frame is stretched to
/// the model size (see PixelBufferResizer), so K is scaled by (W_out / imgW, H_out / imgH).
final class DepthEstimator: @unchecked Sendable {
    static let modelName = "DepthAnythingV2SmallF16"

    enum DepthError: LocalizedError {
        case modelMissing
        case loadFailed(String)
        case badInput(String)
        case badOutput(String)

        var errorDescription: String? {
            switch self {
            case .modelMissing: return "Depth model missing"
            case .loadFailed(let s): return "Depth model failed to load: " + s
            case .badInput(let s): return "Depth input error: " + s
            case .badOutput(let s): return "Depth output error: " + s
            }
        }
    }

    private struct Loaded {
        let model: MLModel
        let inputName: String
        let outputName: String
        let width: Int
        let height: Int
    }

    private let lock = NSLock()
    private var loaded: Loaded?
    private let resizer = PixelBufferResizer()

    static func modelURL() -> URL? {
        if let u = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") { return u }
        return Bundle.main.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models")
    }

    static var isModelBundled: Bool { modelURL() != nil }

    /// Loads the model now (throws a descriptive error if absent). Safe to call from any thread.
    func prepare() throws {
        _ = try loadedModel()
    }

    private func loadedModel() throws -> Loaded {
        lock.lock()
        defer { lock.unlock() }
        if let l = loaded { return l }
        guard let url = DepthEstimator.modelURL() else { throw DepthError.modelMissing }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let t0 = Date()
        let model: MLModel
        do {
            model = try MLModel(contentsOf: url, configuration: cfg)
        } catch {
            throw DepthError.loadFailed(error.localizedDescription)
        }
        let desc = model.modelDescription
        for (_, d) in desc.inputDescriptionsByName {
            LogStore.shared.log("depth model input  " + MLArrayReader.describe(d))
        }
        for (_, d) in desc.outputDescriptionsByName {
            LogStore.shared.log("depth model output " + MLArrayReader.describe(d))
        }

        var inName = "image"
        var inW = 518
        var inH = 392
        var foundImage = false
        for (name, d) in desc.inputDescriptionsByName {
            if let ic = d.imageConstraint {
                inName = name
                if ic.pixelsWide > 0 { inW = ic.pixelsWide }
                if ic.pixelsHigh > 0 { inH = ic.pixelsHigh }
                foundImage = true
                break
            }
        }
        if !foundImage { throw DepthError.loadFailed("model has no image input") }

        var outName = "depth"
        if desc.outputDescriptionsByName[outName] == nil {
            guard let first = desc.outputDescriptionsByName.keys.sorted().first else {
                throw DepthError.loadFailed("model has no outputs")
            }
            outName = first
        }
        let loadMs = Int(Date().timeIntervalSince(t0) * 1000)
        LogStore.shared.log("depth model loaded in \(loadMs) ms (in=\(inName) \(inW)x\(inH), out=\(outName))")
        let l = Loaded(model: model, inputName: inName, outputName: outName, width: inW, height: inH)
        loaded = l
        return l
    }

    /// Relative depth (disparity-like as far as the model goes; the caller verifies the sign) at model output
    /// resolution, in the sensor orientation of `pixelBuffer`.
    func estimate(pixelBuffer: CVPixelBuffer) throws -> FloatGrid {
        let l = try loadedModel()
        guard let resized = resizer.resize(pixelBuffer, width: l.width, height: l.height) else {
            throw DepthError.badInput("could not resize camera frame to \(l.width)x\(l.height)")
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [l.inputName: MLFeatureValue(pixelBuffer: resized)])
        let out = try l.model.prediction(from: provider)
        guard let fv = out.featureValue(for: l.outputName) else {
            throw DepthError.badOutput("missing output \(l.outputName)")
        }
        if let ob = fv.imageBufferValue {
            return try DepthEstimator.grid(from: ob)
        }
        if let ma = fv.multiArrayValue {
            return try DepthEstimator.grid(from: ma)
        }
        throw DepthError.badOutput("output \(l.outputName) is neither image nor multi-array")
    }

    static func grid(from ob: CVPixelBuffer) throws -> FloatGrid {
        CVPixelBufferLockBaseAddress(ob, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(ob, .readOnly) }
        let w = CVPixelBufferGetWidth(ob)
        let h = CVPixelBufferGetHeight(ob)
        let fmt = CVPixelBufferGetPixelFormatType(ob)
        let bpr = CVPixelBufferGetBytesPerRow(ob)
        guard w > 0, h > 0, let base = CVPixelBufferGetBaseAddress(ob) else {
            throw DepthError.badOutput("empty output pixel buffer")
        }
        var data = [Float](repeating: 0, count: w * h)
        if fmt == kCVPixelFormatType_OneComponent16Half {
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt16.self)
                for x in 0..<w { data[y * w + x] = MLArrayReader.halfToFloat(row[x]) }
            }
        } else if fmt == kCVPixelFormatType_OneComponent32Float {
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: Float.self)
                for x in 0..<w { data[y * w + x] = row[x] }
            }
        } else if fmt == kCVPixelFormatType_OneComponent8 {
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { data[y * w + x] = Float(row[x]) / 255.0 }
            }
        } else {
            throw DepthError.badOutput("unsupported output pixel format \(fmt)")
        }
        return FloatGrid(width: w, height: h, data: data)
    }

    static func grid(from ma: MLMultiArray) throws -> FloatGrid {
        let shape = ma.shape.map { $0.intValue }
        guard shape.count >= 2 else { throw DepthError.badOutput("multi-array rank < 2") }
        let h = shape[shape.count - 2]
        let w = shape[shape.count - 1]
        let all = MLArrayReader.floats(ma)
        guard w > 0, h > 0, all.count >= w * h else { throw DepthError.badOutput("multi-array shape \(shape)") }
        let data = all.count == w * h ? all : Array(all[0..<(w * h)])
        return FloatGrid(width: w, height: h, data: data)
    }
}
