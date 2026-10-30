import XCTest
import Foundation
import CalorieCore

/// Walks up from this file until a directory containing spec/vectors.json exists.
private func rangeEngineRepoRoot(_ file: String = #filePath) throws -> URL {
    var dir = URL(fileURLWithPath: file).deletingLastPathComponent()
    for _ in 0..<12 {
        let probe = dir.appendingPathComponent("spec").appendingPathComponent("vectors.json")
        if FileManager.default.fileExists(atPath: probe.path) { return dir }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    throw XCTSkip("repo root (spec/vectors.json) not found above " + file)
}

private func reNum(_ any: Any?) -> Double? {
    guard let n = any as? NSNumber else { return nil }
    return n.doubleValue
}

private func reTriple(_ any: Any?) -> Triple? {
    guard let d = any as? [String: Any] else { return nil }
    guard let mn = reNum(d["min"]), let ty = reNum(d["typ"]), let mx = reNum(d["max"]) else { return nil }
    return Triple(min: mn, typ: ty, max: mx)
}

private func reEstimate(_ any: Any) -> ItemEstimate? {
    guard let d = any as? [String: Any] else { return nil }
    guard let lo = reNum(d["lo"]), let med = reNum(d["med"]), let hi = reNum(d["hi"]) else { return nil }
    let id = (d["id"] as? String) ?? ""
    let flags = (d["flags"] as? [String]) ?? []
    return ItemEstimate(id: id, lo: lo, med: med, hi: hi, flags: flags)
}

private struct VectorResult {
    var lo: Double
    var med: Double
    var hi: Double
    var flags: [String]?   // nil for combineMeal
}

final class RangeEngineTests: XCTestCase {

    /// Runs one vector; throws when the engine rejects the input (or the input cannot even be represented).
    private func runVector(fn: String, input: [String: Any]) throws -> VectorResult {
        if fn == "estimateSolid" {
            guard let dd = input["dish"] as? [String: Any] else { throw RangeEngineError.invalid("no dish") }
            var oil: OilRange? = nil
            if let o = dd["oil_g_per_100g"] as? [String: Any] {
                if let omin = reNum(o["min"]), let omax = reNum(o["max"]) {
                    oil = OilRange(min: omin, max: omax)
                }
            }
            guard let kcal = reTriple(dd["kcal_100g"]) else { throw RangeEngineError.invalid("no kcal") }
            let dish = DishEntry(id: (dd["id"] as? String) ?? "", kind: (dd["kind"] as? String) ?? "solid",
                                 densityGml: reTriple(dd["density_g_ml"]), kcal100g: kcal, oilGPer100g: oil,
                                 defaultPortionG: reTriple(dd["default_portion_g"]))
            let r = try RangeEngine.estimateSolid(dish, volumeMl: reTriple(input["volumeMl"]))
            return VectorResult(lo: r.lo, med: r.med, hi: r.hi, flags: r.flags)
        }
        if fn == "estimateDrink" {
            guard let dd = input["drink"] as? [String: Any] else { throw RangeEngineError.invalid("no drink") }
            guard let base = reTriple(dd["base_kcal_per_700ml"]), let top = reTriple(dd["toppings_kcal"]) else {
                throw RangeEngineError.invalid("bad drink")
            }
            let drink = DrinkEntry(id: (dd["id"] as? String) ?? "", baseKcalPer700ml: base, toppingsKcal: top)
            guard let levelName = input["sugarLevel"] as? String, let level = SugarLevel(rawValue: levelName) else {
                // A level outside the Swift enum is the "unknown sugar level" error case.
                throw RangeEngineError.invalid("unknown sugar level")
            }
            var fractions: [String: Double] = [:]
            if let fd = input["sugarFraction"] as? [String: Any] {
                for (k, v) in fd {
                    if let x = reNum(v) { fractions[k] = x }
                }
            }
            guard let size = reNum(input["sizeMl"]), let full = reNum(input["fullSugarGPer700"]) else {
                throw RangeEngineError.invalid("bad drink inputs")
            }
            let r = try RangeEngine.estimateDrink(drink, sizeMl: size, sugar: level, fullSugarGPer700: full,
                                                  sugarFraction: fractions)
            return VectorResult(lo: r.lo, med: r.med, hi: r.hi, flags: r.flags)
        }
        if fn == "combineMeal" {
            var items: [ItemEstimate] = []
            if let arr = input["items"] as? [Any] {
                for a in arr {
                    if let e = reEstimate(a) { items.append(e) }
                }
            }
            let m = RangeEngine.combineMeal(items)
            return VectorResult(lo: m.lo, med: m.med, hi: m.hi, flags: nil)
        }
        throw RangeEngineError.invalid("unknown fn " + fn)
    }

    private func assertClose(_ a: Double, _ b: Double, _ what: String) {
        let tol = 1e-9 * Swift.max(abs(a), abs(b)) + 1e-12
        XCTAssertTrue(abs(a - b) <= tol, what + ": \(a) vs \(b)")
    }

    func testVectors() throws {
        let root = try rangeEngineRepoRoot()
        let data = try Data(contentsOf: root.appendingPathComponent("spec").appendingPathComponent("vectors.json"))
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let top = parsed as? [String: Any], let cases = top["cases"] as? [[String: Any]] else {
            XCTFail("vectors.json: expected {cases: [...]}")
            return
        }
        XCTAssertGreaterThanOrEqual(cases.count, 20)
        var errorCases = 0
        for c in cases {
            let name = (c["name"] as? String) ?? "?"
            guard let fn = c["fn"] as? String, let input = c["input"] as? [String: Any],
                  let expected = c["expected"] as? [String: Any] else {
                XCTFail("malformed case " + name)
                continue
            }
            let expectsError: Bool = (expected["error"] as? NSNumber)?.boolValue ?? false
            if expectsError {
                errorCases += 1
                var threw = false
                do {
                    _ = try runVector(fn: fn, input: input)
                } catch {
                    threw = true
                }
                XCTAssertTrue(threw, name + ": expected an error")
                continue
            }
            do {
                let r = try runVector(fn: fn, input: input)
                guard let elo = reNum(expected["lo"]), let emed = reNum(expected["med"]),
                      let ehi = reNum(expected["hi"]) else {
                    XCTFail(name + ": expected lo/med/hi missing")
                    continue
                }
                assertClose(r.lo, elo, name + " lo")
                assertClose(r.med, emed, name + " med")
                assertClose(r.hi, ehi, name + " hi")
                if let flags = r.flags {
                    let eflags = (expected["flags"] as? [String]) ?? []
                    XCTAssertEqual(flags, eflags, name + " flags")
                }
            } catch {
                XCTFail(name + ": unexpected error \(error)")
            }
        }
        XCTAssertGreaterThanOrEqual(errorCases, 4)
    }

    func testFormatRange() {
        XCTAssertEqual(RangeEngine.formatRange(447.4, 602.6), "445\u{2013}605")
        XCTAssertEqual(RangeEngine.formatRange(1310.0, 2105.0), "1310\u{2013}2105")
        XCTAssertEqual(RangeEngine.formatRange(0.0, 2.4), "0\u{2013}0")
    }

    func testRound5() {
        XCTAssertEqual(RangeEngine.round5(447.4), 445.0)
        XCTAssertEqual(RangeEngine.round5(447.5), 450.0)
        XCTAssertEqual(RangeEngine.round5(2.5), 5.0)
    }

    func testSigmaOf() throws {
        XCTAssertEqual(try RangeEngine.sigmaOf(100.0, 100.0), 0.0)
        XCTAssertEqual(try RangeEngine.sigmaOf(100.0, 200.0), Foundation.log(2.0) / 3.29, accuracy: 1e-12)
        XCTAssertThrowsError(try RangeEngine.sigmaOf(0.0, 10.0))
        XCTAssertThrowsError(try RangeEngine.sigmaOf(-1.0, 10.0))
    }
}
