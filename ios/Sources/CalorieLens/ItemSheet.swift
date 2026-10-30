import SwiftUI
import CalorieCore

/// Bottom sheet with chips for dish / container / sugar / ice (same rules as web/src/sheet.ts).
struct ItemSheet: View {
    @ObservedObject var meal: MealModel
    let sweep: SweepController
    let entryId: String
    @Environment(\.dismiss) private var dismiss

    struct SugarChoice: Identifiable {
        let level: SugarLevel
        let zh: String
        var id: String { level.rawValue }
    }
    struct IceChoice: Identifiable {
        let level: Ice
        let zh: String
        var id: String { level.rawValue }
    }

    static let sugars: [SugarChoice] = [
        SugarChoice(level: SugarLevel.full, zh: "全糖"),
        SugarChoice(level: SugarLevel.less, zh: "少糖"),
        SugarChoice(level: SugarLevel.half, zh: "半糖"),
        SugarChoice(level: SugarLevel.light, zh: "微糖"),
        SugarChoice(level: SugarLevel.none, zh: "無糖")
    ]
    static let ices: [IceChoice] = [
        IceChoice(level: Ice.normal, zh: "正常冰"),
        IceChoice(level: Ice.less, zh: "少冰"),
        IceChoice(level: Ice.none, zh: "去冰")
    ]

    static func sugarZh(_ s: SugarLevel) -> String {
        for c in sugars where c.level == s { return c.zh }
        return s.rawValue
    }

    private let columns = [GridItem(.adaptive(minimum: 104), spacing: 8, alignment: .top)]

    var body: some View {
        NavigationStack {
            ScrollView {
                if let entry = meal.entry(entryId) {
                    content(entry)
                        .padding(16)
                } else {
                    Text("Item removed")
                        .padding(16)
                }
            }
            .navigationTitle(titleText)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .frame(minHeight: 44)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private var titleText: String {
        if let e = meal.entry(entryId) {
            if let zh = meal.dishZh(e) { return zh }
            return e.isManualDrink ? "Drink 飲料" : "Item 品項"
        }
        return "Item"
    }

    @ViewBuilder
    private func content(_ entry: MealModel.Entry) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            if let err = meal.loadError {
                Text(err).font(.footnote).foregroundStyle(Color.red)
            }
            if !meal.noNumbers, let range = meal.rangeText(entry) {
                Text("\(range) kcal")
                    .font(.title3.bold())
            }
            if let v = meal.volumeText(entry) {
                Text(v).font(.footnote).foregroundStyle(Color.secondary)
            }

            let drinkOnly = meal.isDrinkOnly(entry)
            sectionTitle(drinkOnly ? "Drink 飲料" : "Dish 菜色")
            LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                ForEach(meal.dishOptions(drinkOnly: drinkOnly)) { opt in
                    chip(main: opt.zh, sub: opt.sub, on: entry.choice?.dishId == opt.id) {
                        meal.setDish(entry.id, dishId: opt.id)
                    }
                }
            }

            if let choice = entry.choice {
                let isDrink = meal.isDrinkDish(entry)
                sectionTitle(isDrink ? "Cup 杯型" : "Container 容器")
                LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                    ForEach(meal.containerOptions(entry)) { opt in
                        chip(main: opt.zh, sub: opt.sub, on: choice.containerId == opt.id) {
                            pickContainer(entry, containerId: opt.id)
                        }
                    }
                }

                if isDrink {
                    sectionTitle("Sugar 甜度")
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                        ForEach(ItemSheet.sugars) { s in
                            chip(main: s.zh, sub: s.level.rawValue, on: choice.sugar == s.level) {
                                meal.setSugar(entry.id, sugar: s.level)
                            }
                        }
                    }
                    sectionTitle("Ice 冰塊 (display only)")
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                        ForEach(ItemSheet.ices) { i in
                            chip(main: i.zh, sub: nil, on: choice.ice == i.level) {
                                meal.setIce(entry.id, ice: i.level)
                            }
                        }
                    }
                }

                Button {
                    meal.removeLabel(entry.id)
                    dismiss()
                } label: {
                    Text("Remove label 移除標記")
                        .font(.body.bold())
                        .frame(maxWidth: .infinity, minHeight: 44)
                }
                .buttonStyle(.bordered)
                .tint(Color.red)
            }
        }
    }

    @MainActor
    private func pickContainer(_ entry: MealModel.Entry, containerId: String) {
        meal.setContainer(entry.id, containerId: containerId)
        guard entry.measured != nil, let spec = meal.containerSpec(containerId), spec.shape == .bowl else { return }
        let key = entry.id
        Task { @MainActor in
            if let r = await sweep.remeasure(itemKey: key, container: spec) {
                meal.setVolumeOverride(key, r)
            }
        }
    }

    private func sectionTitle(_ s: String) -> some View {
        Text(s)
            .font(.subheadline.bold())
            .foregroundStyle(Color.secondary)
    }

    private func chip(main: String, sub: String?, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Text(main)
                    .font(.body.bold())
                    .multilineTextAlignment(.center)
                if let sub = sub {
                    Text(sub)
                        .font(.caption2)
                        .opacity(0.75)
                        .multilineTextAlignment(.center)
                }
            }
            .foregroundStyle(on ? Color.white : Color.primary)
            .frame(maxWidth: .infinity, minHeight: 44)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(on ? Color.accentColor : Color.gray.opacity(0.18),
                        in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }
}
