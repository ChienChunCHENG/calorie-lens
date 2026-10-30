import SwiftUI
import RealityKit
import ARKit

/// Full-screen AR camera. World tracking + horizontal planes only (no LiDAR features).
struct ARViewContainer: UIViewRepresentable {
    let controller: ARSessionController

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        controller.attach(view.session)

        if ARWorldTrackingConfiguration.isSupported {
            let config = ARWorldTrackingConfiguration()
            config.planeDetection = [.horizontal]
            view.session.run(config)
            LogStore.shared.log("AR session started")
        } else {
            LogStore.shared.log("ARWorldTracking not supported on this device")
        }
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    static func dismantleUIView(_ uiView: ARView, coordinator: ()) {
        uiView.session.pause()
    }
}
