import ARKit
import CalorieCore
import Foundation
import simd

/// One measured food item (key = item key in the sweep frames' masks).
struct MeasuredItem: Identifiable {
    let id: String
    var label: String?
    var centroidW: SIMD3<Float>
    var topW: SIMD3<Float>
    var volume: VolumeResult
}

// MARK: - Background-work value types

/// CVPixelBuffer is not Sendable; the buffer is only used on the worker task, briefly.
private final class PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
    init(_ b: CVPixelBuffer) { buffer = b }
}

private final class KeyframeOutput: @unchecked Sendable {
    let rel: FloatGrid
    let detections: [DetectedObject]
    let depthMs: Int
    let detectMs: Int
    init(rel: FloatGrid, detections: [DetectedObject], depthMs: Int, detectMs: Int) {
        self.rel = rel
        self.detections = detections
        self.depthMs = depthMs
        self.detectMs = detectMs
    }
}

/// Everything copied out of the ARFrame at keyframe time (value types only).
private struct KeyframeMeta {
    var t: Double
    var T_wc: simd_double4x4
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    var imgW: Int
    var imgH: Int
    var featurePoints: [SIMD3<Double>]
}

private final class ProcessInput: @unchecked Sendable {
    var frames: [SweepFrame]
    var detections: [[DetectedObject]]
    var planeHint: Plane?
    var imageSize: [Int]
    init(frames: [SweepFrame], detections: [[DetectedObject]], planeHint: Plane?, imageSize: [Int]) {
        self.frames = frames
        self.detections = detections
        self.planeHint = planeHint
        self.imageSize = imageSize
    }
}

private final class ProcessOutput: @unchecked Sendable {
    var frames: [SweepFrame] = []
    var plane: Plane?
    var items: [MeasuredItem] = []
    var dumpZip: URL?
    var error: String?
}

// MARK: - SweepController

/// Drives one capture sweep: waits for a table plane, collects 8-12 keyframes (depth from Core ML, optional
/// detector masks), then aligns, segments and measures volumes off the main thread and writes a replayable dump.
/// All geometry stays in the SENSOR (landscape) orientation of ARFrame.capturedImage.
@MainActor
final class SweepController: ObservableObject {
    enum Phase: Equatable {
        case idle
        case needPlane
        case sweeping(progress: Double)
        case processing
        case done
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var guidance: String = ""
    @Published private(set) var items: [MeasuredItem] = []
    @Published private(set) var lastDumpURL: URL?

    private static let minKeyframes = 8
    private static let maxKeyframes = 12
    private static let sweepSeconds = 5.0
    private static let timeoutSeconds = 20.0
    private static let minKeyframeGapSeconds = 0.25
    private static let viewChangeDegrees: Float = 8
    private static let translationMeters: Float = 0.04

    private let session: ARSessionController
    private let estimator = DepthEstimator()
    private let detector = Detector()

    private var runID = 0
    private var planeHint: Plane?
    private var sweepStartTS: TimeInterval = 0
    private var lastIngestTS: TimeInterval = 0
    private var lastKeyPos: SIMD3<Float>?
    private var lastKeyFwd: SIMD3<Float>?
    private var lastKeyTS: TimeInterval = 0
    private var planePollCounter = 0
    private var depthBusy = false
    private var imageSize: [Int] = [0, 0]
    private var useDetector = false
    private var collectedFrames: [SweepFrame] = []
    private var collectedDetections: [[DetectedObject]] = []

    // Kept for remeasure.
    private var keptFrames: [SweepFrame] = []
    private var keptPlane: Plane?

    init(session: ARSessionController) {
        self.session = session
        session.sweep = self
    }

    // MARK: Public API

    func start() {
        switch phase {
        case .needPlane, .sweeping, .processing:
            return
        default:
            break
        }
        runID += 1
        let myRun = runID
        resetCollection()
        items = []

        if !DepthEstimator.isModelBundled {
            LogStore.shared.log("sweep: depth model missing (\(DepthEstimator.modelName).mlmodelc not in bundle)")
            fail("Depth model missing / 缺少深度模型")
            return
        }
        useDetector = detector.isAvailable
        LogStore.shared.log("sweep: start (detector=\(useDetector ? "yes" : "no, HeightSegment"))")
        phase = .needPlane
        setGuidance("Point at the table, then hold still / 請對準桌面並保持穩定")

        // Warm up the models off the main thread so the first keyframe is not slow.
        let est = estimator
        let det = detector
        let useDet = useDetector
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                try est.prepare()
                if useDet { det.prepare() }
            } catch {
                let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                LogStore.shared.log("sweep: model warm-up failed: " + msg)
                await self?.warmupFailed(run: myRun, message: msg)
            }
        }
    }

    func cancel() {
        runID += 1
        resetCollection()
        depthBusy = false
        phase = .idle
        setGuidance("")
        LogStore.shared.log("sweep: cancelled")
    }

    /// Re-runs the volume step for one item on the kept frames (e.g. after the user picks a bowl container).
    func remeasure(itemKey: String, container: ContainerSpec?) async -> VolumeResult? {
        guard let plane = keptPlane, !keptFrames.isEmpty,
              let index = items.firstIndex(where: { $0.id == itemKey }) else { return nil }
        let frames = keptFrames
        let label = items[index].label ?? "unknown"
        let t0 = Date()
        let result = await Task.detached(priority: .userInitiated) { () -> VolumeResult in
            return Volume.computeItemVolume(frames: frames, itemKey: itemKey, plane: plane, label: label,
                                            container: container)
        }.value
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        LogStore.shared.log("sweep: remeasure \(itemKey) -> \(result.method.rawValue) in \(ms) ms")
        if let i = items.firstIndex(where: { $0.id == itemKey }) {
            items[i].volume = result
        }
        return result
    }

    // MARK: Frame tap (called by ARSessionController on the main queue)

    func ingest(frame: ARFrame) {
        switch phase {
        case .needPlane:
            handleNeedPlane(frame)
        case .sweeping:
            handleSweeping(frame)
        default:
            return
        }
    }

    // MARK: State handling

    private func resetCollection() {
        planeHint = nil
        lastKeyPos = nil
        lastKeyFwd = nil
        lastKeyTS = 0
        planePollCounter = 0
        collectedFrames = []
        collectedDetections = []
        imageSize = [0, 0]
    }

    private func setGuidance(_ s: String) {
        if guidance != s { guidance = s }
    }

    private func fail(_ message: String) {
        phase = .failed(message)
        setGuidance(message)
        LogStore.shared.log("sweep: failed: " + message)
    }

    private func warmupFailed(run: Int, message: String) {
        guard run == runID else { return }
        switch phase {
        case .needPlane, .sweeping:
            runID += 1
            fail(message)
        default:
            break
        }
    }

    private func handleNeedPlane(_ frame: ARFrame) {
        lastIngestTS = frame.timestamp
        planePollCounter += 1
        if planePollCounter % 6 != 0 { return }
        guard let hint = session.horizontalPlaneHint() else {
            setGuidance("Looking for the table surface... move the phone slowly / 正在尋找桌面，請慢慢移動手機")
            return
        }
        planeHint = hint
        sweepStartTS = frame.timestamp
        LogStore.shared.log(String(format: "sweep: plane n=(%.3f, %.3f, %.3f) offset=%.3f",
                                   hint.n.x, hint.n.y, hint.n.z, hint.offset))
        phase = .sweeping(progress: 0)
        setGuidance("Slowly circle the plate / 繞著餐點慢慢移動")
        handleSweeping(frame)
    }

    private func handleSweeping(_ frame: ARFrame) {
        lastIngestTS = frame.timestamp
        let elapsed = frame.timestamp - sweepStartTS

        if elapsed > SweepController.timeoutSeconds {
            if collectedFrames.count >= 4 && !depthBusy {
                LogStore.shared.log("sweep: timeout with \(collectedFrames.count) keyframes, processing anyway")
                beginProcessing()
            } else if collectedFrames.count < 4 {
                runID += 1
                fail("Not enough views, try again / 視角不足，請重試")
            }
            return
        }

        var tracking = false
        if case .normal = frame.camera.trackingState { tracking = true }
        if !tracking {
            setGuidance("Tracking unstable, hold steady / 追蹤不穩，請稍微停住")
            return
        }
        setGuidance("Slowly circle the plate / 繞著餐點慢慢移動")

        // Keyframe test (copy plain values only; the ARFrame is never retained).
        let t = frame.camera.transform
        let pos = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        let fwdRaw = SIMD3<Float>(-t.columns.2.x, -t.columns.2.y, -t.columns.2.z)
        let fwdLen = simd_length(fwdRaw)
        let fwd = fwdLen > 1e-6 ? fwdRaw / fwdLen : fwdRaw

        var isKey = false
        if let lp = lastKeyPos, let lf = lastKeyFwd {
            let c = max(-1, min(1, simd_dot(fwd, lf)))
            let angle = acos(c) * 180 / Float.pi
            let moved = simd_length(pos - lp)
            isKey = angle >= SweepController.viewChangeDegrees || moved >= SweepController.translationMeters
            if frame.timestamp - lastKeyTS < SweepController.minKeyframeGapSeconds { isKey = false }
        } else {
            isKey = true
        }

        if isKey && !depthBusy && collectedFrames.count < SweepController.maxKeyframes {
            dispatchKeyframe(frame, pos: pos, fwd: fwd)
        }

        updateProgress(elapsed: elapsed)
        finishIfReady(elapsed: elapsed)
    }

    private func updateProgress(elapsed: Double) {
        guard case .sweeping(let old) = phase else { return }
        let byCount = Double(collectedFrames.count) / Double(SweepController.minKeyframes)
        let byTime = elapsed / SweepController.sweepSeconds
        let p = max(0, min(0.99, min(byCount, byTime)))
        if abs(p - old) >= 0.02 { phase = .sweeping(progress: p) }
    }

    private func finishIfReady(elapsed: Double) {
        guard case .sweeping = phase, !depthBusy else { return }
        let n = collectedFrames.count
        if n >= SweepController.maxKeyframes || (n >= SweepController.minKeyframes && elapsed >= SweepController.sweepSeconds) {
            beginProcessing()
        }
    }

    private func dispatchKeyframe(_ frame: ARFrame, pos: SIMD3<Float>, fwd: SIMD3<Float>) {
        let cam = frame.camera
        let tf = cam.transform
        let T = simd_double4x4(columns: (SIMD4<Double>(Double(tf.columns.0.x), Double(tf.columns.0.y), Double(tf.columns.0.z), Double(tf.columns.0.w)),
                                         SIMD4<Double>(Double(tf.columns.1.x), Double(tf.columns.1.y), Double(tf.columns.1.z), Double(tf.columns.1.w)),
                                         SIMD4<Double>(Double(tf.columns.2.x), Double(tf.columns.2.y), Double(tf.columns.2.z), Double(tf.columns.2.w)),
                                         SIMD4<Double>(Double(tf.columns.3.x), Double(tf.columns.3.y), Double(tf.columns.3.z), Double(tf.columns.3.w))))
        let k = cam.intrinsics   // simd_float3x3, column-major: fx = k[0][0], fy = k[1][1], cx = k[2][0], cy = k[2][1]
        let res = cam.imageResolution
        var fps: [SIMD3<Double>] = []
        if let pts = frame.rawFeaturePoints?.points {
            fps.reserveCapacity(pts.count)
            for p in pts { fps.append(SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z))) }
        }
        let pb = frame.capturedImage
        let meta = KeyframeMeta(t: frame.timestamp - sweepStartTS, T_wc: T,
                                fx: Double(k[0][0]), fy: Double(k[1][1]), cx: Double(k[2][0]), cy: Double(k[2][1]),
                                imgW: CVPixelBufferGetWidth(pb), imgH: CVPixelBufferGetHeight(pb),
                                featurePoints: fps)
        _ = res
        let box = PixelBufferBox(pb)
        lastKeyPos = pos
        lastKeyFwd = fwd
        lastKeyTS = frame.timestamp
        depthBusy = true
        let myRun = runID
        let est = estimator
        let det = detector
        let useDet = useDetector

        Task.detached(priority: .userInitiated) { [weak self] in
            let t0 = Date()
            do {
                let rel = try est.estimate(pixelBuffer: box.buffer)
                let t1 = Date()
                var dets: [DetectedObject] = []
                if useDet && det.isAvailable {
                    dets = det.detect(pixelBuffer: box.buffer, outWidth: rel.width, outHeight: rel.height)
                }
                let t2 = Date()
                let out = KeyframeOutput(rel: rel, detections: dets,
                                         depthMs: Int(t1.timeIntervalSince(t0) * 1000),
                                         detectMs: Int(t2.timeIntervalSince(t1) * 1000))
                await self?.keyframeFinished(run: myRun, meta: meta, output: out, error: nil)
            } catch {
                await self?.keyframeFinished(run: myRun, meta: meta, output: nil, error: error)
            }
        }
    }

    private func keyframeFinished(run: Int, meta: KeyframeMeta, output: KeyframeOutput?, error: Error?) {
        guard run == runID else { return }
        depthBusy = false
        guard case .sweeping = phase else { return }

        if let error = error {
            runID += 1
            if let de = error as? DepthEstimator.DepthError, case .modelMissing = de {
                fail("Depth model missing / 缺少深度模型")
            } else {
                let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                fail("Depth failed: " + msg)
            }
            return
        }
        guard let out = output else { return }

        let sx = Double(out.rel.width) / Double(max(1, meta.imgW))
        let sy = Double(out.rel.height) / Double(max(1, meta.imgH))
        let K = Intrinsics(fx: meta.fx, fy: meta.fy, cx: meta.cx, cy: meta.cy).scaled(sx: sx, sy: sy)
        let frame = SweepFrame(t: meta.t, T_wc: meta.T_wc, K: K, relDepth: out.rel,
                               featurePointsW: meta.featurePoints)
        collectedFrames.append(frame)
        collectedDetections.append(out.detections)
        imageSize = [meta.imgW, meta.imgH]
        LogStore.shared.log("sweep: keyframe \(collectedFrames.count) t=\(String(format: "%.2f", meta.t))s depth \(out.depthMs) ms detect \(out.detectMs) ms (\(out.detections.count) dets, \(meta.featurePoints.count) fps, rel \(out.rel.width)x\(out.rel.height))")

        let elapsed = lastIngestTS - sweepStartTS
        updateProgress(elapsed: elapsed)
        finishIfReady(elapsed: elapsed)
    }

    // MARK: Processing

    private func beginProcessing() {
        guard collectedFrames.count >= 2 else {
            runID += 1
            fail("Not enough views, try again / 視角不足，請重試")
            return
        }
        phase = .processing
        setGuidance("Measuring... / 計算中…")
        let myRun = runID
        let input = ProcessInput(frames: collectedFrames, detections: collectedDetections,
                                 planeHint: planeHint, imageSize: imageSize)
        LogStore.shared.log("sweep: processing \(collectedFrames.count) keyframes")
        Task.detached(priority: .userInitiated) { [weak self] in
            let out = SweepController.process(input)
            await self?.processingFinished(run: myRun, output: out)
        }
    }

    private func processingFinished(run: Int, output: ProcessOutput) {
        guard run == runID, phase == .processing else { return }
        if let e = output.error {
            fail(e)
            return
        }
        items = output.items
        keptFrames = output.frames
        keptPlane = output.plane
        if let z = output.dumpZip { lastDumpURL = z }
        phase = .done
        setGuidance(output.items.isEmpty ? "No food found / 找不到餐點" : "Done / 完成")
        LogStore.shared.log("sweep: done, \(output.items.count) item(s)")
    }

    // MARK: Heavy work (runs on a background task; no actor state touched)

    nonisolated private static func process(_ inp: ProcessInput) -> ProcessOutput {
        let result = ProcessOutput()
        let tStart = Date()
        func ms(_ since: Date) -> Int { Int(Date().timeIntervalSince(since) * 1000) }
        var frames = inp.frames

        // 1. Depth semantics: relDepth must be disparity-like (larger = nearer).
        if let c = disparityCorrelation(frames) {
            LogStore.shared.log(String(format: "sweep: corr(rel, 1/d) over feature points = %.3f", c))
            if c < -0.3 {
                LogStore.shared.log("sweep: WARNING depth output looks like depth, not disparity; inverting sign")
                for i in 0..<frames.count {
                    var g = frames[i].relDepth
                    for j in 0..<g.data.count { g.data[j] = -g.data[j] }
                    frames[i].relDepth = g
                }
            }
        } else {
            LogStore.shared.log("sweep: corr(rel, 1/d) unavailable (too few feature points); assuming disparity")
        }

        // 2. First alignment (no masks: food acts as outliers).
        let t1 = Date()
        guard let first = Align.alignFrames(&frames, planeHint: inp.planeHint) else {
            result.error = "Could not fit the table plane / 找不到桌面"
            return result
        }
        var plane = first.plane
        LogStore.shared.log("sweep: align#1 \(first.diags.filter { $0.ok }.count)/\(frames.count) frames ok, plane n=(\(fmt(plane.n.x)), \(fmt(plane.n.y)), \(fmt(plane.n.z))) off=\(fmt(plane.offset)) in \(ms(t1)) ms")

        // 3. Segmentation: detector masks (associated across frames) or class-agnostic height segmentation.
        var segmented: [String: SegmentedObject] = [:]
        let haveDets = inp.detections.contains { !$0.isEmpty }
        let t2 = Date()
        if haveDets {
            let n = associateDetections(&frames, detections: inp.detections, plane: plane)
            LogStore.shared.log("sweep: detector association -> \(n) object(s) in \(ms(t2)) ms")
        } else {
            let objs = HeightSegment.segment(&frames, plane: plane)
            for o in objs { segmented[o.key] = o }
            LogStore.shared.log("sweep: HeightSegment -> \(objs.count) object(s) in \(ms(t2)) ms")
        }

        // 4. Second alignment with masks excluding the food.
        let t3 = Date()
        if let second = Align.alignFrames(&frames, planeHint: plane) {
            plane = second.plane
            LogStore.shared.log("sweep: align#2 \(second.diags.filter { $0.ok }.count)/\(frames.count) frames ok in \(ms(t3)) ms")
        } else {
            LogStore.shared.log("sweep: align#2 failed, keeping first alignment")
        }

        // 5. Volumes.
        var keys = Set<String>()
        var labelOf: [String: String] = [:]
        for f in frames {
            for k in f.masks.keys {
                keys.insert(k)
                if labelOf[k] == nil, let l = f.labels[k] { labelOf[k] = l }
            }
        }
        var items: [MeasuredItem] = []
        for key in keys.sorted() {
            let label = labelOf[key] ?? "unknown"
            if label.hasPrefix("container_") { continue }
            let tv = Date()
            let vol = Volume.computeItemVolume(frames: frames, itemKey: key, plane: plane, label: label, container: nil)
            var centroid: SIMD3<Float>
            var top: SIMD3<Float>
            if let o = segmented[key] {
                centroid = f3(o.centroidW)
                top = f3(o.topW)
            } else if let ct = maskCentroidAndTop(frames, key: key, plane: plane) {
                centroid = f3(ct.centroid)
                top = f3(ct.top)
            } else {
                LogStore.shared.log("sweep: item \(key) has no valid depth pixels, skipped")
                continue
            }
            LogStore.shared.log("sweep: item \(key) label=\(label) \(vol.method.rawValue) volume=\(vol.volumeMl.map { String(format: "%.0f/%.0f/%.0f mL", $0.min, $0.typ, $0.max) } ?? "nil") cells=\(vol.nCells) cov=\(fmt(vol.coverage)) in \(ms(tv)) ms")
            items.append(MeasuredItem(id: key, label: label == "unknown" ? nil : label,
                                      centroidW: centroid, topW: top, volume: vol))
            _ = centroid; _ = top
        }

        // 6. Dump (no RGB) + zip for sharing.
        let td = Date()
        do {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let root = docs.appendingPathComponent("sweeps", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = "yyyyMMdd_HHmmss"
            let dir = root.appendingPathComponent("sweep_" + df.string(from: Date()), isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            var meta = DumpMeta()
            meta.deviceModel = deviceModel()
            let v = ProcessInfo.processInfo.operatingSystemVersion
            meta.iosVersion = "\(v.majorVersion).\(v.minorVersion)"
            meta.imageSize = inp.imageSize
            meta.depthModel = DepthEstimator.modelName
            meta.plane = plane
            meta.planeSource = "arkit"
            try DumpIO.write(frames: frames, meta: meta, rgbJPEGs: nil, to: dir)
            result.dumpZip = try DumpIO.zip(dir)
            LogStore.shared.log("sweep: dump written to \(dir.lastPathComponent), zip ready in \(ms(td)) ms")
        } catch {
            LogStore.shared.log("sweep: dump failed: " + error.localizedDescription)
        }

        result.frames = frames
        result.plane = plane
        result.items = items
        LogStore.shared.log("sweep: processing total \(ms(tStart)) ms")
        return result
    }

    nonisolated private static func fmt(_ x: Double) -> String { String(format: "%.3f", x) }

    nonisolated private static func f3(_ v: SIMD3<Double>) -> SIMD3<Float> {
        SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z))
    }

    nonisolated private static func deviceModel() -> String {
        var sys = utsname()
        _ = uname(&sys)
        let mirror = Mirror(reflecting: sys.machine)
        var s = ""
        for child in mirror.children {
            if let v = child.value as? Int8, v != 0 { s.append(Character(UnicodeScalar(UInt8(bitPattern: v)))) }
        }
        return s
    }

    /// Mean over frames of corr(rel, 1/d) at the ARKit feature points projected into each frame.
    nonisolated private static func disparityCorrelation(_ frames: [SweepFrame]) -> Double? {
        var sum = 0.0
        var cnt = 0
        for f in frames {
            let inv = simd_inverse(f.T_wc)
            var xs: [Double] = []
            var ys: [Double] = []
            for p in f.featurePointsW.prefix(600) {
                let pc = inv * SIMD4<Double>(p.x, p.y, p.z, 1)
                let d = -pc.z
                if d < 0.05 { continue }
                let u = f.K.cx + f.K.fx * pc.x / d
                let v = f.K.cy - f.K.fy * pc.y / d
                let ui = Int(u.rounded())
                let vi = Int(v.rounded())
                if !f.relDepth.contains(u: ui, v: vi) { continue }
                let r = Double(f.relDepth[ui, vi])
                if !r.isFinite { continue }
                xs.append(r)
                ys.append(1.0 / d)
            }
            if xs.count < 10 { continue }
            let n = Double(xs.count)
            let mx = xs.reduce(0, +) / n
            let my = ys.reduce(0, +) / n
            var sxy = 0.0, sxx = 0.0, syy = 0.0
            for i in 0..<xs.count {
                let dx = xs[i] - mx, dy = ys[i] - my
                sxy += dx * dy
                sxx += dx * dx
                syy += dy * dy
            }
            if sxx > 1e-12 && syy > 1e-12 {
                sum += sxy / (sxx * syy).squareRoot()
                cnt += 1
            }
        }
        return cnt > 0 ? sum / Double(cnt) : nil
    }

    nonisolated private static func worldPoint(_ f: SweepFrame, u: Int, v: Int, d: Double) -> SIMD3<Double> {
        let x = (Double(u) - f.K.cx) * d / f.K.fx
        let y = -(Double(v) - f.K.cy) * d / f.K.fy
        let pw = f.T_wc * SIMD4<Double>(x, y, -d, 1)
        return SIMD3<Double>(pw.x, pw.y, pw.z)
    }

    nonisolated private static func maskPoints(_ f: SweepFrame, mask: Mask, step: Int) -> [SIMD3<Double>] {
        guard let dm = f.depthM, dm.width == mask.width, dm.height == mask.height else { return [] }
        var out: [SIMD3<Double>] = []
        var v = 0
        while v < mask.height {
            var u = 0
            while u < mask.width {
                if mask[u, v] {
                    let d = Double(dm[u, v])
                    if d.isFinite && d > 0.05 { out.append(worldPoint(f, u: u, v: v, d: d)) }
                }
                u += step
            }
            v += step
        }
        return out
    }

    /// Centroid (mean of masked points) and top point (mean of the highest 5 % above the plane) over all frames.
    nonisolated private static func maskCentroidAndTop(_ frames: [SweepFrame], key: String, plane: Plane)
        -> (centroid: SIMD3<Double>, top: SIMD3<Double>)? {
        var pts: [SIMD3<Double>] = []
        for f in frames {
            if let m = f.masks[key] { pts.append(contentsOf: maskPoints(f, mask: m, step: 3)) }
        }
        if pts.count < 10 { return nil }
        var sum = SIMD3<Double>(0, 0, 0)
        for p in pts { sum += p }
        let centroid = sum / Double(pts.count)
        let sorted = pts.sorted { plane.height($0) < plane.height($1) }
        let start = min(sorted.count - 1, Int(Double(sorted.count) * 0.95))
        var tsum = SIMD3<Double>(0, 0, 0)
        for i in start..<sorted.count { tsum += sorted[i] }
        let top = tsum / Double(sorted.count - start)
        return (centroid, top)
    }

    /// Clusters per-frame detections into objects by (label, plane-projected world centroid), then rewrites
    /// frames[i].masks / labels with keys "d0", "d1", ... Returns the number of objects kept.
    nonisolated private static func associateDetections(_ frames: inout [SweepFrame], detections: [[DetectedObject]],
                                                        plane: Plane) -> Int {
        struct Cluster {
            var label: String
            var sum: SIMD3<Double>
            var n: Int
            var members: [(frame: Int, det: Int)]
            var frameSet: Set<Int>
        }
        var clusters: [Cluster] = []
        for (fi, dets) in detections.enumerated() where fi < frames.count {
            for (di, det) in dets.enumerated() {
                let pts = maskPoints(frames[fi], mask: det.mask, step: 4)
                if pts.count < 15 { continue }
                var s = SIMD3<Double>(0, 0, 0)
                for p in pts { s += p }
                var c = s / Double(pts.count)
                c -= plane.n * plane.height(c)   // project onto the table
                var bestIdx = -1
                var bestDist = 0.05
                for (ci, cl) in clusters.enumerated() where cl.label == det.label {
                    let mean = cl.sum / Double(cl.n)
                    let d = simd_length(mean - c)
                    if d < bestDist { bestDist = d; bestIdx = ci }
                }
                if bestIdx >= 0 {
                    clusters[bestIdx].sum += c
                    clusters[bestIdx].n += 1
                    clusters[bestIdx].members.append((fi, di))
                    clusters[bestIdx].frameSet.insert(fi)
                } else {
                    clusters.append(Cluster(label: det.label, sum: c, n: 1, members: [(fi, di)], frameSet: [fi]))
                }
            }
        }
        let minFrames = min(2, frames.count)
        let kept = clusters.filter { $0.frameSet.count >= minFrames }.sorted { $0.members.count > $1.members.count }
        for (k, cl) in kept.enumerated() {
            let key = "d\(k)"
            for m in cl.members {
                let det = detections[m.frame][m.det]
                if var existing = frames[m.frame].masks[key] {
                    if existing.data.count == det.mask.data.count {
                        for i in 0..<existing.data.count where det.mask.data[i] { existing.data[i] = true }
                        frames[m.frame].masks[key] = existing
                    }
                } else {
                    frames[m.frame].masks[key] = det.mask
                }
                frames[m.frame].labels[key] = cl.label
            }
        }
        return kept.count
    }
}
