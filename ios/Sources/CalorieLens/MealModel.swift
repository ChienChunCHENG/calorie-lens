import Foundation
import CalorieCore

/// UI-side meal state: one entry per measured item (keyed by MeasuredItem.id) plus manually added drinks.
/// All estimates are derived on demand from `data`, the user's choice and the (possibly re-measured) volume.
@MainActor
final class MealModel: ObservableObject {
    struct Entry: Identifiable {
        let id: String
        var measured: MeasuredItem?          // nil => manually added drink (no measurement)
        var choice: MealItemChoice?          // nil => not labelled yet
        var volumeOverride: VolumeResult?    // set after a bowl re-measure

        var isManualDrink: Bool { measured == nil }
    }

    struct Option: Identifiable {
        let id: String
        let zh: String
        let sub: String
    }

    @Published private(set) var entries: [Entry] = []
    @Published var noNumbers = false
    @Published private(set) var loadError: String?

    private(set) var data: MealData?
    private var manualCounter = 0

    /// Detector / sweep labels that mean "this is a drink".
    static let drinkLabels: Set<String> = [
        "bubble_milk_tea", "container_cup", "black_tea", "green_tea", "soy_milk",
        "cup", "wine glass", "bottle"
    ]
    static let bowlLabels: Set<String> = ["bowl", "container_bowl"]

    init() {
        loadData()
    }

    private func loadData() {
        guard let url = Bundle.main.url(forResource: "nutrition", withExtension: "json") else {
            loadError = "nutrition.json is missing from the app bundle"
            LogStore.shared.log("MealModel: " + (loadError ?? ""))
            return
        }
        do {
            let raw = try Data(contentsOf: url)
            let loaded = try MealData.load(jsonData: raw)
            data = loaded
            LogStore.shared.log("MealModel: loaded nutrition.json (\(loaded.dishOrder.count) dishes)")
        } catch {
            loadError = "Could not load nutrition.json: " + error.localizedDescription
            LogStore.shared.log("MealModel: " + (loadError ?? ""))
        }
    }

    // MARK: Sync with the sweep

    /// Keeps user choices for item keys that already exist; new keys get auto-assigned when the label is a dish id.
    func sync(with items: [MeasuredItem]) {
        var old: [String: Entry] = [:]
        for e in entries { old[e.id] = e }
        var next: [Entry] = []
        for item in items {
            if var existing = old[item.id], existing.measured != nil {
                existing.measured = item
                next.append(existing)
            } else {
                var fresh = Entry(id: item.id, measured: item, choice: nil, volumeOverride: nil)
                fresh.choice = autoChoice(label: item.label)
                next.append(fresh)
            }
        }
        for e in entries where e.measured == nil {
            next.append(e)
        }
        entries = next
    }

    /// Called when a new scan starts: drops measured entries, keeps manual drinks.
    func clearMeasured() {
        entries = entries.filter { $0.measured == nil }
    }

    private func autoChoice(label: String?) -> MealItemChoice? {
        guard let d = data, let label = label, d.dishes[label] != nil else { return nil }
        return MealLogic.defaultChoice(data: d, dishId: label, preferBowl: false)
    }

    // MARK: Lookups

    func entry(_ id: String) -> Entry? {
        entries.first(where: { $0.id == id })
    }

    private func index(of id: String) -> Int? {
        entries.firstIndex(where: { $0.id == id })
    }

    func dishInfo(_ e: Entry) -> DishInfo? {
        guard let d = data, let c = e.choice else { return nil }
        return d.dishes[c.dishId]
    }

    func dishZh(_ e: Entry) -> String? {
        dishInfo(e)?.zh
    }

    func isDrinkDish(_ e: Entry) -> Bool {
        dishInfo(e)?.entry.kind == "drink"
    }

    /// True when the sheet should only list drinks.
    func isDrinkOnly(_ e: Entry) -> Bool {
        if e.isManualDrink { return true }
        if let label = e.measured?.label, MealModel.drinkLabels.contains(label) { return true }
        return isDrinkDish(e)
    }

    func effectiveVolume(_ e: Entry) -> VolumeResult? {
        if let o = e.volumeOverride { return o }
        return e.measured?.volume
    }

    func estimate(_ e: Entry) -> ItemEstimate? {
        guard let d = data, let c = e.choice else { return nil }
        var vol: Triple? = nil
        if !isDrinkDish(e), let r = effectiveVolume(e) {
            vol = r.volumeMl
        }
        return MealLogic.safeEstimate(data: d, choice: c, measuredVolumeMl: vol)
    }

    /// "1310–2105" or nil when the item has no estimate.
    func rangeText(_ e: Entry) -> String? {
        guard let est = estimate(e) else { return nil }
        return RangeEngine.formatRange(est.lo, est.hi)
    }

    func flags(_ e: Entry) -> [String] {
        estimate(e)?.flags ?? []
    }

    /// "measured 230 mL" for heightmap / bowl-cavity volumes of solids.
    func volumeText(_ e: Entry) -> String? {
        if isDrinkDish(e) { return nil }
        guard let r = effectiveVolume(e), let v = r.volumeMl else { return nil }
        switch r.method {
        case .heightmap, .bowlCavity:
            return "measured \(Int(RangeEngine.round5(v.typ))) mL"
        default:
            return nil
        }
    }

    var labelledCount: Int {
        entries.filter { $0.choice != nil }.count
    }

    var total: MealRange? {
        var ests: [ItemEstimate] = []
        for e in entries {
            if let est = estimate(e) { ests.append(est) }
        }
        if ests.isEmpty { return nil }
        return RangeEngine.combineMeal(ests)
    }

    // MARK: Sheet options

    func dishOptions(drinkOnly: Bool) -> [Option] {
        guard let d = data else { return [] }
        var out: [Option] = []
        for id in d.dishOrder {
            guard let info = d.dishes[id] else { continue }
            if drinkOnly && info.entry.kind != "drink" { continue }
            out.append(Option(id: id, zh: info.zh, sub: info.en))
        }
        return out
    }

    func containerOptions(_ e: Entry) -> [Option] {
        guard let d = data, let c = e.choice else { return [] }
        var out: [Option] = []
        for id in MealLogic.containerChoices(data: d, dishId: c.dishId, current: c.containerId) {
            guard let info = d.containers[id] else { continue }
            out.append(Option(id: id, zh: info.zh, sub: id))
        }
        if !isDrinkDish(e) {
            out.append(Option(id: MealLogic.unknownContainer, zh: "不確定", sub: "unknown (default portion)"))
        }
        return out
    }

    func containerSpec(_ containerId: String) -> ContainerSpec? {
        data?.containers[containerId]?.spec
    }

    // MARK: Mutations

    func setDish(_ id: String, dishId: String) {
        guard let d = data, d.dishes[dishId] != nil, let i = index(of: id) else { return }
        if entries[i].choice?.dishId == dishId { return }
        var prefer = false
        if let label = entries[i].measured?.label { prefer = MealModel.bowlLabels.contains(label) }
        entries[i].choice = MealLogic.defaultChoice(data: d, dishId: dishId, preferBowl: prefer)
        entries[i].volumeOverride = nil
    }

    func setContainer(_ id: String, containerId: String) {
        guard let i = index(of: id), var c = entries[i].choice else { return }
        c.containerId = containerId
        entries[i].choice = c
        if containerSpec(containerId)?.shape != .bowl {
            entries[i].volumeOverride = nil
        }
    }

    func setSugar(_ id: String, sugar: SugarLevel) {
        guard let i = index(of: id), var c = entries[i].choice else { return }
        c.sugar = sugar
        entries[i].choice = c
    }

    func setIce(_ id: String, ice: Ice) {
        guard let i = index(of: id), var c = entries[i].choice else { return }
        c.ice = ice
        entries[i].choice = c
    }

    func setVolumeOverride(_ id: String, _ result: VolumeResult) {
        guard let i = index(of: id) else { return }
        entries[i].volumeOverride = result
    }

    func removeLabel(_ id: String) {
        guard let i = index(of: id) else { return }
        if entries[i].isManualDrink {
            entries.remove(at: i)
        } else {
            entries[i].choice = nil
            entries[i].volumeOverride = nil
        }
    }

    /// Adds a manual drink (default: first drink in the dish list). Returns its entry id, nil if data is unavailable.
    func addManualDrink() -> String? {
        guard let d = data else { return nil }
        guard let dishId = d.dishOrder.first(where: { d.dishes[$0]?.entry.kind == "drink" }) else { return nil }
        manualCounter += 1
        let id = "manual_drink_\(manualCounter)"
        var e = Entry(id: id, measured: nil, choice: nil, volumeOverride: nil)
        e.choice = MealLogic.defaultChoice(data: d, dishId: dishId, preferBowl: false)
        entries.append(e)
        return id
    }
}
