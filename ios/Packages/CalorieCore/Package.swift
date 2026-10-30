// swift-tools-version:5.9
// Pure-Swift core (no ARKit/UIKit): range engine, nutrition data, plane fit, depth alignment, volume, sweep dumps.
// Builds for iOS (linked by the app) and macOS (so `swift test` runs on the CI runner without a simulator).
import PackageDescription

let package = Package(
    name: "CalorieCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "CalorieCore", targets: ["CalorieCore"]),
    ],
    targets: [
        .target(name: "CalorieCore", path: "Sources/CalorieCore"),
        .testTarget(name: "CalorieCoreTests", dependencies: ["CalorieCore"], path: "Tests/CalorieCoreTests"),
    ]
)
