import SwiftUI
import CalorieCore

/// Bottom bar: scan control, guidance, meal total, toggles, share and logs.
struct MealBar: View {
    @ObservedObject var sweep: SweepController
    @ObservedObject var meal: MealModel
    let onAddDrink: () -> Void
    let onSelect: (String) -> Void
    let onLogs: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            statusBlock
            totalBlock
            drinkRow
            actionRow
            Text("Estimates are ranges, not medical advice. 熱量為估計範圍，非醫療建議。")
                .font(.caption2)
                .foregroundStyle(Color.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .environment(\.colorScheme, .dark)
        .padding(.horizontal, 8)
        .padding(.bottom, 4)
    }

    // MARK: Status

    private var isBusy: Bool {
        switch sweep.phase {
        case .idle, .done, .failed: return false
        default: return true
        }
    }

    @ViewBuilder
    private var statusBlock: some View {
        VStack(spacing: 4) {
            if case .sweeping(let progress) = sweep.phase {
                ProgressView(value: min(max(progress, 0), 1))
            } else if case .processing = sweep.phase {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Processing 處理中")
                        .font(.footnote)
                }
            }
            if case .failed(let message) = sweep.phase {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(Color.red)
                    .multilineTextAlignment(.center)
            } else if !sweep.guidance.isEmpty {
                Text(sweep.guidance)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
            }
            if let err = meal.loadError {
                Text(err)
                    .font(.footnote)
                    .foregroundStyle(Color.red)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: Total

    private var totalText: String {
        if meal.entries.isEmpty { return "Scan or add a drink 掃描或新增飲料" }
        if meal.noNumbers { return "\(meal.labelledCount) item(s), kcal hidden" }
        if let t = meal.total { return "Meal \(RangeEngine.formatRange(t.lo, t.hi)) kcal" }
        return "Tap a label to identify items 點選標記"
    }

    private var totalBlock: some View {
        Text(totalText)
            .font(.title3.bold())
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
    }

    // MARK: Manual drinks

    @ViewBuilder
    private var drinkRow: some View {
        let drinks = meal.entries.filter { $0.isManualDrink }
        if !drinks.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(drinks) { d in
                        Button {
                            onSelect(d.id)
                        } label: {
                            VStack(spacing: 0) {
                                Text(meal.dishZh(d) ?? "Drink")
                                    .font(.footnote.bold())
                                if !meal.noNumbers, let r = meal.rangeText(d) {
                                    Text("\(r) kcal")
                                        .font(.caption2)
                                }
                            }
                            .padding(.horizontal, 12)
                            .frame(minHeight: 44)
                            .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: Buttons

    private var actionRow: some View {
        HStack(spacing: 8) {
            scanButton
            barButton("+ Drink", systemImage: nil, action: onAddDrink)
            barButton(meal.noNumbers ? "Show kcal" : "Hide kcal", systemImage: nil) {
                meal.noNumbers.toggle()
            }
            if let url = sweep.lastDumpURL {
                ShareLink(item: url) {
                    Image(systemName: "square.and.arrow.up")
                        .frame(minWidth: 44, minHeight: 44)
                        .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
                }
            }
            barButton("Logs", systemImage: nil, action: onLogs)
        }
    }

    private var scanButton: some View {
        let busy = isBusy
        let processing: Bool = {
            if case .processing = sweep.phase { return true }
            return false
        }()
        return Button {
            if busy {
                sweep.cancel()
            } else {
                meal.clearMeasured()
                sweep.start()
            }
        } label: {
            Text(busy ? "Cancel 取消" : "Scan 掃描")
                .font(.body.bold())
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .foregroundStyle(Color.white)
                .padding(.horizontal, 12)
                .frame(minWidth: 44, minHeight: 44)
                .background(busy ? Color.red.opacity(0.8) : Color.accentColor,
                            in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .disabled(processing)
    }

    private func barButton(_ title: String, systemImage: String?, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.footnote.bold())
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .foregroundStyle(Color.white)
                .padding(.horizontal, 10)
                .frame(minWidth: 44, minHeight: 44)
                .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }
}
