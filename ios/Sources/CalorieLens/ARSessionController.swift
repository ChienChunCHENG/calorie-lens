import ARKit
import CalorieCore
import Foundation
import simd

/// Owns the ARSessionDelegate role and publishes values for the HUD.
/// Delegate callbacks arrive on the main queue (see `attach`).
final class ARSessionController: NSObject, ObservableObject, ARSessionDelegate {
    @Published private(set) var trackingText = "notAvailable"
    @Published private(set) var planeCount = 0
    @Published private(set) var fps = 0.0
    /// Incremented at most ~30 times per second; views observe it to refresh overlays from `currentCamera`.
    @Published private(set) var frameTick = 0

    /// Latest camera (main queue only). ARCamera is a small value-like object; the ARFrame itself is never retained.
    private(set) var currentCamera: ARCamera?

    /// Frame tap for the sweep state machine (main queue). Weak to avoid a retain cycle with the view model.
    weak var sweep: SweepController?

    private weak var arSession: ARSession?
    private var planeIDs = Set<UUID>()
    private var planeAnchors = [UUID: ARPlaneAnchor]()
    private var frameCount = 0
    private var fpsWindowStart: TimeInterval = 0
    private var lastLoggedTracking = ""
    private var lastTickTimestamp: TimeInterval = 0

    func attach(_ session: ARSession) {
        arSession = session
        session.delegateQueue = DispatchQueue.main
        session.delegate = self
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        currentCamera = frame.camera

        let text = ARSessionController.describe(frame.camera.trackingState)
        if text != trackingText { trackingText = text }
        if text != lastLoggedTracking {
            lastLoggedTracking = text
            LogStore.shared.log("tracking: " + text)
        }

        frameCount += 1
        let now = frame.timestamp
        if fpsWindowStart == 0 { fpsWindowStart = now }
        let elapsed = now - fpsWindowStart
        if elapsed >= 1.0 {
            fps = Double(frameCount) / elapsed
            frameCount = 0
            fpsWindowStart = now
        }

        // <= 30 Hz tick (at 60 fps this fires on every second frame).
        if now - lastTickTimestamp >= 0.030 {
            lastTickTimestamp = now
            frameTick &+= 1
        }

        if let tap = sweep {
            // Delegate queue is main, so assuming main-actor isolation is sound.
            MainActor.assumeIsolated {
                tap.ingest(frame: frame)
            }
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal {
                planeIDs.insert(plane.identifier)
                planeAnchors[plane.identifier] = plane
            }
        }
        publishPlaneCount()
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal {
                planeAnchors[plane.identifier] = plane
            }
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            planeIDs.remove(anchor.identifier)
            planeAnchors.removeValue(forKey: anchor.identifier)
        }
        publishPlaneCount()
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        LogStore.shared.log("AR session failed: " + error.localizedDescription)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        LogStore.shared.log("AR session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        LogStore.shared.log("AR session interruption ended")
    }

    // MARK: Plane hint

    /// Table plane for the sweep. Prefers the horizontal plane anchor hit by a ray through the image centre
    /// (the optical axis); otherwise the largest horizontal anchor (ties: the lowest).
    /// Normal = anchor y axis oriented up, offset = n . anchor position. Main queue only.
    func horizontalPlaneHint() -> Plane? {
        var chosen: ARPlaneAnchor?

        if let session = arSession, let cam = currentCamera {
            let t = cam.transform
            let origin = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
            let dir = SIMD3<Float>(-t.columns.2.x, -t.columns.2.y, -t.columns.2.z)
            let query = ARRaycastQuery(origin: origin, direction: dir,
                                       allowing: .existingPlaneGeometry, alignment: .horizontal)
            for hit in session.raycast(query) {
                if let anchor = hit.anchor as? ARPlaneAnchor, anchor.alignment == .horizontal {
                    chosen = anchor
                    break
                }
            }
        }

        if chosen == nil {
            var bestArea: Float = -1
            var bestY: Float = 0
            for anchor in planeAnchors.values {
                let area = anchor.planeExtent.width * anchor.planeExtent.height
                let y = anchor.transform.columns.3.y
                if area > bestArea + 1e-4 || (abs(area - bestArea) <= 1e-4 && y < bestY) {
                    bestArea = area
                    bestY = y
                    chosen = anchor
                }
            }
        }

        guard let anchor = chosen else { return nil }
        let m = anchor.transform
        var n = SIMD3<Double>(Double(m.columns.1.x), Double(m.columns.1.y), Double(m.columns.1.z))
        let len = simd_length(n)
        if len < 1e-6 { return nil }
        n = n / len
        if n.y < 0 { n = -n }
        let pos = SIMD3<Double>(Double(m.columns.3.x), Double(m.columns.3.y), Double(m.columns.3.z))
        return Plane(n: n, offset: simd_dot(n, pos), inlierRMSEm: 0)
    }

    // MARK: Helpers

    private func publishPlaneCount() {
        if planeIDs.count != planeCount {
            planeCount = planeIDs.count
            LogStore.shared.log("horizontal planes: \(planeCount)")
        }
    }

    private static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "notAvailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited(initializing)"
            case .excessiveMotion: return "limited(excessiveMotion)"
            case .insufficientFeatures: return "limited(insufficientFeatures)"
            case .relocalizing: return "limited(relocalizing)"
            @unknown default: return "limited(unknown)"
            }
        }
    }
}
