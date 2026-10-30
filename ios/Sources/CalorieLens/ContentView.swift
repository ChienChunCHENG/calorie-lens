import SwiftUI

struct ContentView: View {
    @StateObject private var controller = ARSessionController()
    @State private var showLogs = false
    @State private var crashReport = ""
    @State private var showCrash = false

    var body: some View {
        ZStack(alignment: .top) {
            ARViewContainer(controller: controller)
                .ignoresSafeArea()
            HStack(alignment: .top) {
                HUDView(controller: controller)
                Spacer()
                Button("Logs") { showLogs = true }
                    .font(.footnote.bold())
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
        }
        .sheet(isPresented: $showLogs) {
            LogView()
        }
        .alert("Previous crash", isPresented: $showCrash) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(crashReport)
        }
        .onAppear {
            if let report = CrashReporter.takePendingReport() {
                crashReport = report
                showCrash = true
            }
        }
    }
}

struct HUDView: View {
    @ObservedObject var controller: ARSessionController

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Tracking: \(controller.trackingText)")
            Text("Planes: \(controller.planeCount)")
            Text(String(format: "FPS: %.0f", controller.fps))
        }
        .font(.system(size: 12, design: .monospaced))
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}
