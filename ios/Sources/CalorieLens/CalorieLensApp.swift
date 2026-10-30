import SwiftUI

@main
struct CalorieLensApp: App {
    init() {
        CrashReporter.install()
        LogStore.shared.log("App launched")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
