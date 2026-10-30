// Calorie Lens range engine: Swift port of core/ts/rangeEngine.ts (see spec/range_engine.md).
// Mirrors the TS arithmetic operation for operation so spec/vectors.json matches to ~1e-9.
import Foundation

public enum SugarLevel: String, CaseIterable, Codable { case full, less, half, light, none }
public enum Ice: String, CaseIterable, Codable { case normal, less, none }

public struct OilRange: Equatable, Codable {
    public var min: Double
    public var max: Double
    public init(min: Double, max: Double) { self.min = min; self.max = max }
}

public struct DishEntry {
    public var id: String
    public var kind: String   // "solid" | "drink"
    public var densityGml: Triple?
    public var kcal100g: Triple
    public var oilGPer100g: OilRange?
    public var defaultPortionG: Triple?

    public init(id: String, kind: String, densityGml: Triple?, kcal100g: Triple, oilGPer100g: OilRange?,
                defaultPortionG: Triple?) {
        self.id = id
        self.kind = kind
        self.densityGml = densityGml
        self.kcal100g = kcal100g
        self.oilGPer100g = oilGPer100g
        self.defaultPortionG = defaultPortionG
    }
}

public struct DrinkEntry {
    public var id: String
    public var baseKcalPer700ml: Triple
    public var toppingsKcal: Triple

    public init(id: String, baseKcalPer700ml: Triple, toppingsKcal: Triple) {
        self.id = id
        self.baseKcalPer700ml = baseKcalPer700ml
        self.toppingsKcal = toppingsKcal
    }
}

public struct ItemEstimate: Equatable {
    public var id: String
    public var lo: Double
    public var med: Double
    public var hi: Double
    public var flags: [String]

    public init(id: String, lo: Double, med: Double, hi: Double, flags: [String]) {
        self.id = id
        self.lo = lo
        self.med = med
        self.hi = hi
        self.flags = flags
    }
}

public struct MealRange: Equatable {
    public var lo: Double
    public var med: Double
    public var hi: Double

    public init(lo: Double, med: Double, hi: Double) {
        self.lo = lo
        self.med = med
        self.hi = hi
    }
}

public enum RangeEngineError: Error { case invalid(String) }

public enum RangeEngine {
    static let z90: Double = 1.645
    static let span90: Double = 3.29
    static let floorLo: Double = 0.85
    static let floorHi: Double = 1.15
    static let lowConfSigma: Double = 0.35
    static let sugarVariance: Double = 0.25
    static let kcalPerGSugar: Double = 4.0
    static let kcalPerGOil: Double = 9.0

    private static func finite(_ x: Double, _ what: String) throws -> Double {
        if !x.isFinite { throw RangeEngineError.invalid(what + " must be a finite number") }
        return x
    }

    private static func checkedTriple(_ t: Triple?, _ what: String, positive: Bool) throws -> Triple {
        guard let t = t else { throw RangeEngineError.invalid(what + " missing") }
        let mn = try finite(t.min, what + ".min")
        let ty = try finite(t.typ, what + ".typ")
        let mx = try finite(t.max, what + ".max")
        if !(mn <= ty && ty <= mx) { throw RangeEngineError.invalid(what + " must satisfy min<=typ<=max") }
        if positive && mn <= 0 { throw RangeEngineError.invalid(what + ".min must be > 0") }
        if !positive && mn < 0 { throw RangeEngineError.invalid(what + ".min must be >= 0") }
        return Triple(min: mn, typ: ty, max: mx)
    }

    public static func sigmaOf(_ mn: Double, _ mx: Double) throws -> Double {
        if !(mn > 0) { throw RangeEngineError.invalid("min must be > 0") }
        if mn == mx { return 0.0 }
        return Foundation.log(mx / mn) / span90
    }

    private static func applyFloor(_ lo: Double, _ med: Double, _ hi: Double) -> (Double, Double) {
        return (Swift.min(lo, floorLo * med), Swift.max(hi, floorHi * med))
    }

    public static func estimateSolid(_ dish: DishEntry, volumeMl: Triple?) throws -> ItemEstimate {
        var flags: [String] = []
        let e = try checkedTriple(dish.kcal100g, "kcal_100g", positive: true)
        let sE = try sigmaOf(e.min, e.max)
        let massMed: Double
        let sig2: Double
        if let vol = volumeMl {
            let v = try checkedTriple(vol, "volumeMl", positive: true)
            let d = try checkedTriple(dish.densityGml, "density_g_ml", positive: true)
            massMed = v.typ * d.typ
            let sV = try sigmaOf(v.min, v.max)
            let sD = try sigmaOf(d.min, d.max)
            sig2 = sV * sV + sD * sD + sE * sE
        } else {
            let p = try checkedTriple(dish.defaultPortionG, "default_portion_g", positive: true)
            massMed = p.typ
            let sP = try sigmaOf(p.min, p.max)
            sig2 = sP * sP + sE * sE
            flags.append("portion_default")
        }
        var med: Double = (massMed * e.typ) / 100.0
        let sigma: Double = sig2.squareRoot()
        var lo: Double = med * Foundation.exp(-z90 * sigma)
        var hi: Double = med * Foundation.exp(z90 * sigma)
        if let oil = dish.oilGPer100g {
            let oMin = try finite(oil.min, "oil.min")
            let oMax = try finite(oil.max, "oil.max")
            if oMin < 0 || oMax < oMin { throw RangeEngineError.invalid("oil must satisfy 0<=min<=max") }
            lo += (oMin * massMed * kcalPerGOil) / 100.0
            hi += (oMax * massMed * kcalPerGOil) / 100.0
            med += (((oMin + oMax) / 2.0) * massMed * kcalPerGOil) / 100.0
        }
        let floored = applyFloor(lo, med, hi)
        lo = floored.0
        hi = floored.1
        if sigma > lowConfSigma { flags.append("low_confidence") }
        return ItemEstimate(id: dish.id, lo: lo, med: med, hi: hi, flags: flags)
    }

    public static func estimateDrink(_ drink: DrinkEntry, sizeMl: Double, sugar: SugarLevel,
                                     fullSugarGPer700: Double, sugarFraction: [String: Double]) throws -> ItemEstimate {
        let size = try finite(sizeMl, "sizeMl")
        if size <= 0 { throw RangeEngineError.invalid("sizeMl must be > 0") }
        let full = try finite(fullSugarGPer700, "fullSugarGPer700")
        if full < 0 { throw RangeEngineError.invalid("fullSugarGPer700 must be >= 0") }
        guard let rawFrac = sugarFraction[sugar.rawValue] else {
            throw RangeEngineError.invalid("unknown sugar level")
        }
        let frac = try finite(rawFrac, "sugar fraction")
        if frac < 0 { throw RangeEngineError.invalid("sugar fraction must be >= 0") }
        let b = try checkedTriple(drink.baseKcalPer700ml, "base_kcal_per_700ml", positive: false)
        let t = try checkedTriple(drink.toppingsKcal, "toppings_kcal", positive: false)
        let sKcal: Double = ((full * frac * size) / 700.0) * kcalPerGSugar
        let k: Double = size / 700.0
        let lo0: Double = sKcal * (1.0 - sugarVariance) + b.min * k + t.min
        let med: Double = sKcal + b.typ * k + t.typ
        let hi0: Double = sKcal * (1.0 + sugarVariance) + b.max * k + t.max
        let floored = applyFloor(lo0, med, hi0)
        return ItemEstimate(id: drink.id, lo: floored.0, med: med, hi: floored.1, flags: ["drink_sugar_assumed"])
    }

    public static func combineMeal(_ items: [ItemEstimate]) -> MealRange {
        var med = 0.0
        var dl = 0.0
        var du = 0.0
        for i in items {
            med += i.med
            let a = i.med - i.lo
            let b = i.hi - i.med
            dl += a * a
            du += b * b
        }
        return MealRange(lo: med - dl.squareRoot(), med: med, hi: med + du.squareRoot())
    }

    public static func round5(_ x: Double) -> Double {
        return (x / 5.0 + 0.5).rounded(.down) * 5.0
    }

    /// "1310-2105" with an en dash (U+2013).
    public static func formatRange(_ lo: Double, _ hi: Double) -> String {
        let a: Double = round5(lo) + 0.0
        let b: Double = round5(hi) + 0.0
        return String(format: "%.0f", a) + "\u{2013}" + String(format: "%.0f", b)
    }
}
