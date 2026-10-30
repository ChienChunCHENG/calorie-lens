import ARKit
import Foundation

/// Owns the ARSessionDelegate role and publishes values for the HUD.
final class ARSessionController: NSObject, ObservableObject, ARSessionDelegate {
    @Published private(set) var trackingText = "notAvailable"
    @Published private(set) var planeCount = 0
    @Published private(set) var fps = 0.0

    private var planeIDs = Set<UUID>()
    private var frameCount = 0
    private var fpsWindowStart: TimeInterval = 0
    private var lastLoggedTracking = ""

    func attach(_ session: ARSession) {
        session.delegateQueue = DispatchQueue.main
        session.delegate = self
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
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
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal {
                planeIDs.insert(plane.identifier)
            }
        }
        publishPlaneCount()
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            planeIDs.remove(anchor.identifier)
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
