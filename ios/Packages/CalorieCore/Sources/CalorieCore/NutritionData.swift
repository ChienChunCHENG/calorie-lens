// Meal logic: Swift port of web/src/meal.ts plus the native rule that a measured volume overrides the container prior.
import Foundation

public struct DishInfo {
    public var entry: DishEntry
    public var zh: String
    public var en: String
    public var containerPrior: [String]

    public init(entry: DishEntry, zh: String, en: String, containerPrior: [String]) {
        self.entry = entry
        self.zh = zh
        self.en = en
        self.containerPrior = containerPrior
    }
}

public struct ContainerInfo {
    public var id: String
    public var shape: String
    public var zh: String
    public var capacityMl: Triple
    public var rimDiameterCm: Triple?
    public var depthCm: Triple?

    public init(id: String, shape: String, zh: String, capacityMl: Triple, rimDiameterCm: Triple?, depthCm: Triple?) {
        self.id = id
        self.shape = shape
        self.zh = zh
        self.capacityMl = capacityMl
        self.rimDiameterCm = rimDiameterCm
        self.depthCm = depthCm
    }

    /// ContainerSpec for the volume step; nil for unknown shapes. (nutrition.json shapes: bowl, plate, box, cup.)
    public var spec: ContainerSpec? {
        switch shape {
        case "bowl":
            return ContainerSpec(shape: .bowl, rimDiameterCm: rimDiameterCm, depthCm: depthCm)
        case "plate":
            return ContainerSpec(shape: .plate, rimDiameterCm: rimDiameterCm, depthCm: depthCm)
        case "box", "bento_box":
            return ContainerSpec(shape: .box, rimDiameterCm: rimDiameterCm, depthCm: depthCm)
        case "cup":
            return ContainerSpec(shape: .cup, rimDiameterCm: rimDiameterCm, depthCm: depthCm)
        default:
            return nil
        }
    }
}

public struct MealData {
    public var dishes: [String: DishInfo]
    public var dishOrder: [String]
    public var containers: [String: ContainerInfo]
    public var drinks: [String: DrinkEntry]
    public var fullSugarGPer700: Double
    public var sugarLevels: [String: Double]

    public init(dishes: [String: DishInfo], dishOrder: [String], containers: [String: ContainerInfo],
                drinks: [String: DrinkEntry], fullSugarGPer700: Double, sugarLevels: [String: Double]) {
        self.dishes = dishes
        self.dishOrder = dishOrder
        self.containers = containers
        self.drinks = drinks
        self.fullSugarGPer700 = fullSugarGPer700
        self.sugarLevels = sugarLevels
    }

    private static func number(_ any: Any?) -> Double? {
        guard let n = any as? NSNumber else { return nil }
        return n.doubleValue
    }

    private static func tripleOrNil(_ any: Any?) -> Triple? {
        guard let d = any as? [String: Any] else { return nil }
        guard let mn = number(d["min"]), let ty = number(d["typ"]), let mx = number(d["max"]) else { return nil }
        return Triple(min: mn, typ: ty, max: mx)
    }

    private static func requiredTriple(_ any: Any?, _ what: String) throws -> Triple {
        guard let t = tripleOrNil(any) else {
            throw RangeEngineError.invalid("nutrition.json: bad or missing " + what)
        }
        return t
    }

    /// Parses data/nutrition.json (only the parts the app reads).
    public static func load(jsonData: Data) throws -> MealData {
        let rootAny: Any = try JSONSerialization.jsonObject(with: jsonData, options: [])
        guard let root = rootAny as? [String: Any] else {
            throw RangeEngineError.invalid("nutrition.json: root is not an object")
        }

        guard let dishArray = root["dishes"] as? [[String: Any]] else {
            throw RangeEngineError.invalid("nutrition.json: dishes must be an array")
        }
        var dishes: [String: DishInfo] = [:]
        var dishOrder: [String] = []
        for d in dishArray {
            guard let id = d["id"] as? String else {
                throw RangeEngineError.invalid("nutrition.json: dish without id")
            }
            let kind: String = (d["kind"] as? String) ?? "solid"
            var oil: OilRange? = nil
            if let o = d["oil_g_per_100g"] as? [String: Any] {
                if let omin = number(o["min"]), let omax = number(o["max"]) {
                    oil = OilRange(min: omin, max: omax)
                }
            }
            let kcal: Triple = try requiredTriple(d["kcal_100g"], id + ".kcal_100g")
            let entry = DishEntry(id: id, kind: kind, densityGml: tripleOrNil(d["density_g_ml"]), kcal100g: kcal,
                                  oilGPer100g: oil, defaultPortionG: tripleOrNil(d["default_portion_g"]))
            let prior: [String] = (d["container_prior"] as? [String]) ?? []
            let zh: String = (d["zh"] as? String) ?? ""
            let en: String = (d["en"] as? String) ?? ""
            dishes[id] = DishInfo(entry: entry, zh: zh, en: en, containerPrior: prior)
            dishOrder.append(id)
        }

        var containers: [String: ContainerInfo] = [:]
        if let cdict = root["containers"] as? [String: Any] {
            for (cid, cany) in cdict {
                guard let c = cany as? [String: Any] else { continue }
                let cap: Triple = try requiredTriple(c["capacity_ml"], cid + ".capacity_ml")
                let shape: String = (c["shape"] as? String) ?? ""
                let zh: String = (c["zh"] as? String) ?? ""
                containers[cid] = ContainerInfo(id: cid, shape: shape, zh: zh, capacityMl: cap,
                                                rimDiameterCm: tripleOrNil(c["rim_diameter_cm"]),
                                                depthCm: tripleOrNil(c["depth_cm"]))
            }
        }

        guard let drinkRoot = root["drinks"] as? [String: Any] else {
            throw RangeEngineError.invalid("nutrition.json: drinks missing")
        }
        var drinks: [String: DrinkEntry] = [:]
        if let ddict = drinkRoot["drinks"] as? [String: Any] {
            for (did, dany) in ddict {
                guard let dd = dany as? [String: Any] else { continue }
                let base: Triple = try requiredTriple(dd["base_kcal_per_700ml"], did + ".base_kcal_per_700ml")
                let top: Triple = try requiredTriple(dd["toppings_kcal"], did + ".toppings_kcal")
                drinks[did] = DrinkEntry(id: did, baseKcalPer700ml: base, toppingsKcal: top)
            }
        }
        let fullSugar: Double = try requiredTriple(drinkRoot["full_sugar_g_per_700ml"], "full_sugar_g_per_700ml").typ
        var sugarLevels: [String: Double] = [:]
        if let sl = drinkRoot["sugar_levels"] as? [String: Any] {
            for (k, v) in sl {
                if let x = number(v) { sugarLevels[k] = x }
            }
        }
        return MealData(dishes: dishes, dishOrder: dishOrder, containers: containers, drinks: drinks,
                        fullSugarGPer700: fullSugar, sugarLevels: sugarLevels)
    }
}

public struct MealItemChoice: Equatable {
    public var dishId: String
    public var containerId: String
    public var sugar: SugarLevel
    public var ice: Ice

    public init(dishId: String, containerId: String, sugar: SugarLevel, ice: Ice) {
        self.dishId = dishId
        self.containerId = containerId
        self.sugar = sugar
        self.ice = ice
    }
}

public enum MealLogic {
    public static let unknownContainer = "unknown"
    public static let fill = Triple(min: 0.6, typ: 0.8, max: 1.0)
    public static let cupMl: [String: Double] = ["cup_500": 500, "cup_700": 700]
    public static let defaultSugar: SugarLevel = .half

    /// Bowls and cups: capacity x fill fraction (element-wise). Plate, bento box, unknown: nil (engine uses default portion).
    public static func volumeFor(containerId: String, data: MealData) -> Triple? {
        guard let c = data.containers[containerId] else { return nil }
        if c.shape != "bowl" && c.shape != "cup" { return nil }
        let cap = c.capacityMl
        return Triple(min: cap.min * fill.min, typ: cap.typ * fill.typ, max: cap.max * fill.max)
    }

    /// Containers offered for a dish: its container_prior (known ones only), plus `current` if set and known.
    public static func containerChoices(data: MealData, dishId: String, current: String?) -> [String] {
        var ids: [String] = []
        if let dish = data.dishes[dishId] {
            for id in dish.containerPrior {
                if data.containers[id] != nil { ids.append(id) }
            }
        }
        if let cur = current {
            if !cur.isEmpty && data.containers[cur] != nil && !ids.contains(cur) {
                ids.append(cur)
            }
        }
        return ids
    }

    public static func defaultChoice(data: MealData, dishId: String, preferBowl: Bool) -> MealItemChoice {
        let dish: DishInfo? = data.dishes[dishId]
        var containerId: String = unknownContainer
        if let first = dish?.containerPrior.first { containerId = first }
        if preferBowl, let d = dish, d.entry.kind != "drink" {
            let currentShape: String? = data.containers[containerId]?.shape
            if currentShape != "bowl" {
                var found: String? = nil
                for id in d.containerPrior {
                    if data.containers[id]?.shape == "bowl" {
                        found = id
                        break
                    }
                }
                containerId = found ?? "bowl_small"
            }
        }
        return MealItemChoice(dishId: dishId, containerId: containerId, sugar: defaultSugar, ice: .normal)
    }

    /// measuredVolumeMl (native depth) wins over the container prior for solids; drinks ignore it.
    public static func estimate(data: MealData, choice: MealItemChoice, measuredVolumeMl: Triple?) throws -> ItemEstimate {
        guard let dish = data.dishes[choice.dishId] else {
            throw RangeEngineError.invalid("unknown dish " + choice.dishId)
        }
        if dish.entry.kind == "drink" {
            var size: Double = 700.0
            if let s = cupMl[choice.containerId] {
                size = s
            } else if let first = dish.containerPrior.first, let s2 = cupMl[first] {
                size = s2
            }
            guard let drink = data.drinks[dish.entry.id] else {
                throw RangeEngineError.invalid("unknown drink " + dish.entry.id)
            }
            return try RangeEngine.estimateDrink(drink, sizeMl: size, sugar: choice.sugar,
                                                 fullSugarGPer700: data.fullSugarGPer700,
                                                 sugarFraction: data.sugarLevels)
        }
        var volume: Triple? = measuredVolumeMl
        if volume == nil {
            volume = volumeFor(containerId: choice.containerId, data: data)
        }
        return try RangeEngine.estimateSolid(dish.entry, volumeMl: volume)
    }

    /// Never throws (the UI should still show the name).
    public static func safeEstimate(data: MealData, choice: MealItemChoice, measuredVolumeMl: Triple?) -> ItemEstimate? {
        return try? estimate(data: data, choice: choice, measuredVolumeMl: measuredVolumeMl)
    }
}
