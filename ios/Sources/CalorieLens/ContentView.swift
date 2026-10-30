import SwiftUI

/// Owns the session controller, the sweep controller (which needs the session) and the meal model.
@MainActor
final class AppModel: ObservableObject {
    let session: ARSessionController
    let sweep: SweepController
    let meal: MealModel

    init() {
        let s = ARSessionController()
        session = s
        sweep = SweepController(session: s)
        meal = MealModel()
    }
}

struct SheetTarget: Identifiable {
    let id: String
}

struct ContentView: View {
    @StateObject private var app = AppModel()

    var body: some View {
        LensView(session: app.session, sweep: app.sweep, meal: app.meal)
    }
}

struct LensView: View {
    @ObservedObject var session: ARSessionController
    @ObservedObject var sweep: SweepController
    @ObservedObject var meal: MealModel

    @State private var showLogs = false
    @State private var crashReport = ""
    @State private var showCrash = false
    @State private var sheetTarget: SheetTarget? = nil

    var body: some View {
        ZStack {
            ARViewContainer(controller: session)
                .ignoresSafeArea()
            LabelOverlay(session: session, meal: meal) { id in
                sheetTarget = SheetTarget(id: id)
            }
            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    HUDView(controller: session)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                Spacer()
                MealBar(sweep: sweep,
                        meal: meal,
                        onAddDrink: {
                            if let id = meal.addManualDrink() {
                                sheetTarget = SheetTarget(id: id)
                            }
                        },
                        onSelect: { id in
                            sheetTarget = SheetTarget(id: id)
                        },
                        onLogs: { showLogs = true })
            }
        }
        .onReceive(sweep.$items) { items in
            meal.sync(with: items)
        }
        .sheet(isPresented: $showLogs) {
            LogView()
        }
        .sheet(item: $sheetTarget) { target in
            ItemSheet(meal: meal, sweep: sweep, entryId: target.id)
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
