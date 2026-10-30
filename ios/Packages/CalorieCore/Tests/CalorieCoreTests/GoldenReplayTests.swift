import XCTest
import simd
@testable import CalorieCore

/// Replays the Python-exported synthetic dumps (tools/replay/export_fixtures.py) through the Swift pipeline
/// (DumpIO.read -> Align.alignFrames -> Volume.computeItemVolume) and compares with Python's own result and with the
/// analytic ground truth. Skipped unless CL_FIXTURES points at the export directory.
///
/// Tolerance rationale (numbers from tools/replay/bench_results.json, 10 seeds per scene):
///  * Python's own error vs ground truth: |bias| 3-8 % (worst: cutlet +7.9 %, rice in the 3-item meal -5.9 %); in the
///    seed-0 fixtures Python's meal rice is already -10 % off ground truth. Hence |swift - gt| / gt < 20 %.
///  * Swift and numpy use different RNG streams (RANSAC, bootstrap), so results are not bit-exact. With a plane hint
///    the plane is deterministic (only the per-frame affine RANSAC differs), so typ should agree to a few percent:
///    |swift - py| / py < 10 %.
///  * Without a plane hint (meal_noplane) the plane comes from a RANSAC over feature points; a 2 mm plane
///    difference changes a 1 cm high egg by ~20 %, so use looser bounds: 15 % vs Python, 25 % vs ground truth.
///  * The method string must match exactly (heightmap vs bowl_cavity).
final class GoldenReplayTests: XCTestCase {
    private func pad(_ s: String, _ n: Int) -> String {
        return s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
    }

    private func num(_ v: Any?) -> Double? {
        if let n = v as? NSNumber { return n.doubleValue }
        return nil
    }

    func testGoldenReplay() throws {
        guard let fx = ProcessInfo.processInfo.environment["CL_FIXTURES"], !fx.isEmpty else {
            throw XCTSkip("CL_FIXTURES not set")
        }
        let root = URL(fileURLWithPath: fx, isDirectory: true)
        let expData = try Data(contentsOf: root.appendingPathComponent("expected.json"))
        guard let expected = try JSONSerialization.jsonObject(with: expData, options: []) as? [String: Any] else {
            XCTFail("expected.json is not an object")
            return
        }

        var table: [String] = []
        table.append("case          item     method(sw/py)           swift_typ  py_typ     gt_ml   sw-py%   sw-gt%  cov")

        for caseName in expected.keys.sorted() {
            guard let c = expected[caseName] as? [String: Any],
                  let items = c["items"] as? [String: Any] else {
                XCTFail("\(caseName): bad expected entry")
                continue
            }
            let useHint = (c["use_plane_hint"] as? Bool) ?? ((c["use_plane_hint"] as? NSNumber)?.boolValue ?? false)
            let tolPy = useHint ? 0.10 : 0.15
            let tolGt = useHint ? 0.20 : 0.25

            let dir = root.appendingPathComponent(caseName, isDirectory: true)
            let loaded = try DumpIO.read(dir)
            var frames = loaded.frames
            let meta = loaded.meta
            guard let aligned = Align.alignFrames(&frames, planeHint: useHint ? meta.plane : nil) else {
                XCTFail("\(caseName): alignFrames returned nil")
                continue
            }
            let plane = aligned.plane
            let okFrames = aligned.diags.filter { $0.ok }.count
            XCTAssertGreaterThanOrEqual(okFrames, frames.count / 2, "\(caseName): too few aligned frames")

            for key in items.keys.sorted() {
                guard let it = items[key] as? [String: Any], let py = it["py"] as? [String: Any] else {
                    XCTFail("\(caseName)/\(key): bad expected item")
                    continue
                }
                let label = (it["label"] as? String) ?? "unknown"
                let pyMethod = (py["method"] as? String) ?? "?"
                let pyTyp = num(py["typ"])
                let gt = num(it["gt_ml"])

                let result = Volume.computeItemVolume(frames: frames, itemKey: key, plane: plane, label: label,
                                                      container: meta.containers[key], nBoot: 20)
                let swTyp = result.volumeMl?.typ

                var dPy = "n/a"
                var dGt = "n/a"
                if let s = swTyp, let p = pyTyp, p > 0 { dPy = String(format: "%+.1f", 100.0 * (s - p) / p) }
                if let s = swTyp, let g = gt, g > 0 { dGt = String(format: "%+.1f", 100.0 * (s - g) / g) }
                let nums = String(format: "%9.1f %9.1f %9.1f", swTyp ?? -1.0, pyTyp ?? -1.0, gt ?? -1.0)
                table.append(pad(caseName, 13) + " " + pad(key, 8) + " "
                             + pad("\(result.method.rawValue)/\(pyMethod)", 22) + " " + nums + " "
                             + pad(dPy, 8) + " " + pad(dGt, 8) + " " + String(format: "%.2f", result.coverage))

                XCTAssertEqual(result.method.rawValue, pyMethod, "\(caseName)/\(key): method")
                if let p = pyTyp {
                    guard let s = swTyp else {
                        XCTFail("\(caseName)/\(key): Swift returned no volume, Python \(p) mL")
                        continue
                    }
                    XCTAssertLessThan(abs(s - p) / p, tolPy, "\(caseName)/\(key): swift \(s) vs python \(p)")
                    if let g = gt {
                        XCTAssertLessThan(abs(s - g) / g, tolGt, "\(caseName)/\(key): swift \(s) vs ground truth \(g)")
                    }
                }
            }
        }
        print("---- GoldenReplay (volumes in mL) ----")
        for line in table { print(line) }
    }
}
