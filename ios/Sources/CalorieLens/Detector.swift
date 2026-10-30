import CalorieCore
import CoreML
import CoreVideo
import Foundation

/// One detection, with a mask at depth-map resolution in the SENSOR orientation.
struct DetectedObject {
    var label: String
    var classIndex: Int
    var score: Float
    var mask: Mask
}

/// Optional YOLO(11)-seg Core ML detector (CalorieLensDetector.mlmodelc + labels.json in the bundle).
/// Absent or unexpected layout => `isAvailable == false` and the pipeline uses HeightSegment.
///
/// Supported output layouts (decided from the real outputs at the first prediction):
///  1. Raw ultralytics segmentation (nms=False): a rank-3 array [1, 4+nc+nm, anchors] (or transposed) with
///     cx,cy,w,h (pixels of the model input, or normalised) + class scores + mask coefficients, and a rank-4
///     prototype array [1, nm, ph, pw]. Decoded here (threshold, class-aware NMS, coeff . protos, crop to box).
///  2. Ultralytics Core ML NMS pipeline (nms=True): "confidence" [N, nc] and "coordinates" [N, 4] (normalised
///     cx,cy,w,h). That layout carries no mask output, so masks are the detection BOXES (still fine for the
///     height-map volume: cells at table height contribute nothing).
/// Anything else logs the output shapes and disables the detector.
/// The camera frame is stretched onto the model input (no letterbox), so mask coordinates map linearly.
final class Detector: @unchecked Sendable {
    static let modelName = "CalorieLensDetector"
    private static let scoreThreshold: Float = 0.35
    private static let iouThreshold: Float = 0.5
    private static let maxDetections = 8

    private struct Loaded {
        let model: MLModel
        let inputName: String
        let width: Int
        let height: Int
    }

    private let lock = NSLock()
    private var available: Bool
    private var loaded: Loaded?
    private var names: [Int: String] = [:]
    private let resizer = PixelBufferResizer()
    private var loggedLayout = false

    var isAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return available
    }

    init() {
        let hasModel = Detector.modelURL() != nil
        let labels = Detector.loadLabels()
        names = labels
        available = hasModel && !labels.isEmpty
        if !available {
            LogStore.shared.log("detector: not bundled (model=\(hasModel), labels=\(labels.count)); using HeightSegment")
        }
    }

    static func modelURL() -> URL? {
        if let u = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") { return u }
        return Bundle.main.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models")
    }

    private static func loadLabels() -> [Int: String] {
        var url = Bundle.main.url(forResource: "labels", withExtension: "json")
        if url == nil { url = Bundle.main.url(forResource: "labels", withExtension: "json", subdirectory: "Models") }
        guard let u = url, let data = try? Data(contentsOf: u),
              let obj = try? JSONSerialization.jsonObject(with: data) else { return [:] }
        var out: [Int: String] = [:]
        if let dict = obj as? [String: Any] {
            if let names = dict["names"] as? [String: Any] {
                for (k, v) in names { if let i = Int(k), let s = v as? String { out[i] = s } }
            } else if let names = dict["names"] as? [Any] {
                for (i, v) in names.enumerated() { if let s = v as? String { out[i] = s } }
            }
        } else if let arr = obj as? [Any] {
            for (i, v) in arr.enumerated() { if let s = v as? String { out[i] = s } }
        }
        return out
    }

    private func disable(_ why: String) {
        lock.lock()
        available = false
        lock.unlock()
        LogStore.shared.log("detector disabled: " + why)
    }

    /// Loads the model (background thread). Failure disables the detector instead of throwing.
    func prepare() {
        _ = loadedModel()
    }

    private func loadedModel() -> Loaded? {
        lock.lock()
        if let l = loaded { lock.unlock(); return l }
        let ok = available
        lock.unlock()
        if !ok { return nil }
        guard let url = Detector.modelURL() else { disable("model missing"); return nil }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let t0 = Date()
        let model: MLModel
        do {
            model = try MLModel(contentsOf: url, configuration: cfg)
        } catch {
            disable("load failed: " + error.localizedDescription)
            return nil
        }
        let desc = model.modelDescription
        for (_, d) in desc.inputDescriptionsByName { LogStore.shared.log("detector input  " + MLArrayReader.describe(d)) }
        for (_, d) in desc.outputDescriptionsByName { LogStore.shared.log("detector output " + MLArrayReader.describe(d)) }

        var inName: String?
        var w = 640
        var h = 640
        for (name, d) in desc.inputDescriptionsByName {
            if let ic = d.imageConstraint {
                inName = name
                if ic.pixelsWide > 0 { w = ic.pixelsWide }
                if ic.pixelsHigh > 0 { h = ic.pixelsHigh }
                break
            }
        }
        guard let input = inName else { disable("no image input"); return nil }
        let loadMs = Int(Date().timeIntervalSince(t0) * 1000)
        LogStore.shared.log("detector loaded in \(loadMs) ms (\(input) \(w)x\(h), \(names.count) classes)")
        let l = Loaded(model: model, inputName: input, width: w, height: h)
        lock.lock()
        loaded = l
        lock.unlock()
        return l
    }

    /// Runs the detector on a camera frame. Masks are returned at (outWidth x outHeight), sensor orientation.
    func detect(pixelBuffer: CVPixelBuffer, outWidth: Int, outHeight: Int) -> [DetectedObject] {
        guard let l = loadedModel() else { return [] }
        guard let resized = resizer.resize(pixelBuffer, width: l.width, height: l.height) else { return [] }
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [l.inputName: MLFeatureValue(pixelBuffer: resized)])
            let out = try l.model.prediction(from: provider)
            return decode(out, inputW: l.width, outW: outWidth, outH: outHeight)
        } catch {
            disable("prediction failed: " + error.localizedDescription)
            return []
        }
    }

    // MARK: Decoding

    private func label(for cls: Int) -> String {
        if let s = names[cls] { return s }
        return "class\(cls)"
    }

    private struct Candidate {
        var cls: Int
        var score: Float
        var x0: Float   // normalised 0...1
        var y0: Float
        var x1: Float
        var y1: Float
        var coeffs: [Float]
    }

    private func decode(_ out: MLFeatureProvider, inputW: Int, outW: Int, outH: Int) -> [DetectedObject] {
        var arrays: [String: MLMultiArray] = [:]
        for name in out.featureNames {
            if let a = out.featureValue(for: name)?.multiArrayValue { arrays[name] = a }
        }
        if !loggedLayout {
            loggedLayout = true
            for (name, a) in arrays.sorted(by: { $0.key < $1.key }) {
                LogStore.shared.log("detector result \(name) shape=\(a.shape.map { $0.intValue }) dtype=\(a.dataType.rawValue)")
            }
        }

        // Layout 2: NMS pipeline.
        if let conf = arrays["confidence"], let coords = arrays["coordinates"] {
            return decodePipeline(conf: conf, coords: coords, outW: outW, outH: outH)
        }

        // Layout 1: raw seg head + prototypes.
        var protos: MLMultiArray?
        var pred: MLMultiArray?
        for (_, a) in arrays {
            let nonOne = a.shape.map { $0.intValue }.filter { $0 != 1 }.count
            if a.shape.count == 4 && a.shape[1].intValue <= 64 && a.shape[2].intValue > 8 { protos = a } else if nonOne == 2 && pred == nil { pred = a }
        }
        if let p = pred, let pr = protos {
            return decodeRaw(pred: p, protos: pr, outW: outW, outH: outH)
        }
        let desc = arrays.map { "\($0.key)=\($0.value.shape.map { $0.intValue })" }.joined(separator: ", ")
        disable("unexpected output layout: " + desc)
        return []
    }

    private func boxMask(_ x0: Float, _ y0: Float, _ x1: Float, _ y1: Float, outW: Int, outH: Int) -> Mask {
        var m = Mask(width: outW, height: outH, repeating: false)
        let u0 = max(0, Int((x0 * Float(outW)).rounded(.down)))
        let u1 = min(outW - 1, Int((x1 * Float(outW)).rounded(.up)))
        let v0 = max(0, Int((y0 * Float(outH)).rounded(.down)))
        let v1 = min(outH - 1, Int((y1 * Float(outH)).rounded(.up)))
        if u1 < u0 || v1 < v0 { return m }
        for v in v0...v1 { for u in u0...u1 { m[u, v] = true } }
        return m
    }

    private func decodePipeline(conf: MLMultiArray, coords: MLMultiArray, outW: Int, outH: Int) -> [DetectedObject] {
        let cs = conf.shape.map { $0.intValue }
        guard let nc = cs.last, nc > 0 else { return [] }
        let cf = MLArrayReader.floats(conf)
        let co = MLArrayReader.floats(coords)
        let n = cf.count / nc
        guard co.count >= n * 4 else { return [] }
        var result: [DetectedObject] = []
        for i in 0..<n {
            var best = 0
            var bestS: Float = -1
            for c in 0..<nc where cf[i * nc + c] > bestS { bestS = cf[i * nc + c]; best = c }
            if bestS < Detector.scoreThreshold { continue }
            let cx = co[i * 4], cy = co[i * 4 + 1], w = co[i * 4 + 2], h = co[i * 4 + 3]
            let m = boxMask(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, outW: outW, outH: outH)
            result.append(DetectedObject(label: label(for: best), classIndex: best, score: bestS, mask: m))
            if result.count >= Detector.maxDetections { break }
        }
        return result
    }

    private func decodeRaw(pred: MLMultiArray, protos: MLMultiArray, outW: Int, outH: Int) -> [DetectedObject] {
        let ps = pred.shape.map { $0.intValue }.filter { $0 != 1 }
        let prs = protos.shape.map { $0.intValue }
        guard ps.count == 2, prs.count == 4 else { return [] }
        let nm = prs[1], ph = prs[2], pw = prs[3]
        let channelFirst = ps[0] < ps[1]
        let nCh = channelFirst ? ps[0] : ps[1]
        let nA = channelFirst ? ps[1] : ps[0]
        let nc = nCh - 4 - nm
        guard nc > 0, nm > 0, nA > 0 else {
            disable("raw layout channels \(nCh) do not fit 4 + classes + \(nm)")
            return []
        }
        let f = MLArrayReader.floats(pred)
        @inline(__always) func at(_ c: Int, _ a: Int) -> Float { channelFirst ? f[c * nA + a] : f[a * nCh + c] }

        // Candidates.
        var cands: [Candidate] = []
        var maxCoord: Float = 0
        var raw: [(cx: Float, cy: Float, w: Float, h: Float, cls: Int, score: Float, a: Int)] = []
        for a in 0..<nA {
            var best = 0
            var bestS: Float = -1
            for c in 0..<nc {
                let s = at(4 + c, a)
                if s > bestS { bestS = s; best = c }
            }
            if bestS < Detector.scoreThreshold { continue }
            let cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a)
            maxCoord = max(maxCoord, cx, cy, w, h)
            raw.append((cx, cy, w, h, best, bestS, a))
        }
        // Coordinates are either normalised (<= ~1) or pixels of the model input.
        let inv: Float = maxCoord <= 2.0 ? 1.0 : 1.0 / Float(inputSizeGuess)
        for r in raw {
            var coeffs = [Float](repeating: 0, count: nm)
            for k in 0..<nm { coeffs[k] = at(4 + nc + k, r.a) }
            cands.append(Candidate(cls: r.cls, score: r.score,
                                   x0: (r.cx - r.w / 2) * inv, y0: (r.cy - r.h / 2) * inv,
                                   x1: (r.cx + r.w / 2) * inv, y1: (r.cy + r.h / 2) * inv, coeffs: coeffs))
        }
        cands.sort { $0.score > $1.score }

        // Class-aware greedy NMS.
        var kept: [Candidate] = []
        for c in cands {
            var suppressed = false
            for k in kept where k.cls == c.cls && Detector.iou(k, c) > Detector.iouThreshold {
                suppressed = true
                break
            }
            if !suppressed { kept.append(c) }
            if kept.count >= Detector.maxDetections { break }
        }
        if kept.isEmpty { return [] }

        let pf = MLArrayReader.floats(protos)
        guard pf.count >= nm * ph * pw else { return [] }
        var result: [DetectedObject] = []
        for c in kept {
            // Mask at prototype resolution, cropped to the box.
            var pm = [Bool](repeating: false, count: ph * pw)
            let px0 = max(0, Int((c.x0 * Float(pw)).rounded(.down)))
            let px1 = min(pw - 1, Int((c.x1 * Float(pw)).rounded(.up)))
            let py0 = max(0, Int((c.y0 * Float(ph)).rounded(.down)))
            let py1 = min(ph - 1, Int((c.y1 * Float(ph)).rounded(.up)))
            if px1 >= px0 && py1 >= py0 {
                for py in py0...py1 {
                    for px in px0...px1 {
                        var logit: Float = 0
                        let idx = py * pw + px
                        for k in 0..<nm { logit += c.coeffs[k] * pf[k * ph * pw + idx] }
                        if logit > 0 { pm[idx] = true }
                    }
                }
            }
            var mask = Mask(width: outW, height: outH, repeating: false)
            var any = false
            for v in 0..<outH {
                let py = min(ph - 1, v * ph / outH)
                for u in 0..<outW {
                    let px = min(pw - 1, u * pw / outW)
                    if pm[py * pw + px] { mask[u, v] = true; any = true }
                }
            }
            if any {
                result.append(DetectedObject(label: label(for: c.cls), classIndex: c.cls, score: c.score, mask: mask))
            }
        }
        return result
    }

    private var inputSizeGuess: Int {
        lock.lock()
        defer { lock.unlock() }
        return loaded?.width ?? 640
    }

    private static func iou(_ a: Candidate, _ b: Candidate) -> Float {
        let ix0 = max(a.x0, b.x0), iy0 = max(a.y0, b.y0)
        let ix1 = min(a.x1, b.x1), iy1 = min(a.y1, b.y1)
        let iw = max(0, ix1 - ix0), ih = max(0, iy1 - iy0)
        let inter = iw * ih
        let areaA = max(0, a.x1 - a.x0) * max(0, a.y1 - a.y0)
        let areaB = max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0)
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }
}
