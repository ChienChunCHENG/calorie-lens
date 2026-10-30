import SwiftUI
import ARKit
import simd
import CalorieCore

/// Floating labels anchored to each measured item's top point; re-projected on every `frameTick`.
struct LabelOverlay: View {
    @ObservedObject var session: ARSessionController
    @ObservedObject var meal: MealModel
    let onSelect: (String) -> Void

    private struct Placed: Identifiable {
        let id: String
        let entry: MealModel.Entry
        let point: CGPoint
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                ForEach(placements(in: geo.size)) { p in
                    LabelChip(meal: meal, entry: p.entry) { onSelect(p.id) }
                        .position(x: clampX(p.point.x, width: geo.size.width),
                                  y: max(48, p.point.y - 34))
                }
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
        }
        .ignoresSafeArea()
    }

    private func clampX(_ x: CGFloat, width: CGFloat) -> CGFloat {
        let half: CGFloat = 80
        return min(max(x, half), max(half, width - half))
    }

    private func placements(in size: CGSize) -> [Placed] {
        // Reading frameTick ties this computation to the per-frame refresh.
        let tick = session.frameTick
        _ = tick
        guard let camera = session.currentCamera, size.width > 0, size.height > 0 else { return [] }
        let inverse = camera.transform.inverse
        var out: [Placed] = []
        for e in meal.entries {
            guard let item = e.measured else { continue }
            let w = item.topW
            let cam = inverse * SIMD4<Float>(w.x, w.y, w.z, 1)
            // Camera looks along -Z, so visible points have z < 0.
            if !(cam.z < 0) { continue }
            let pt = camera.projectPoint(w, orientation: .portrait, viewportSize: size)
            if pt.x < 0 || pt.y < 0 || pt.x > size.width || pt.y > size.height { continue }
            if !pt.x.isFinite || !pt.y.isFinite { continue }
            out.append(Placed(id: e.id, entry: e, point: pt))
        }
        return out
    }
}

private struct LabelChip: View {
    @ObservedObject var meal: MealModel
    let entry: MealModel.Entry
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                if let zh = meal.dishZh(entry) {
                    Text(zh)
                        .font(.subheadline.bold())
                    if !meal.noNumbers, let range = meal.rangeText(entry) {
                        Text("\(range) kcal")
                            .font(.footnote)
                    }
                    let badges = badgeTexts()
                    if !badges.isEmpty {
                        HStack(spacing: 4) {
                            ForEach(badges, id: \.self) { b in
                                Text(b)
                                    .font(.caption2)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 1)
                                    .background(Color.white.opacity(0.22), in: Capsule())
                            }
                        }
                    }
                    if let v = meal.volumeText(entry) {
                        Text(v)
                            .font(.caption2)
                            .opacity(0.85)
                    }
                } else {
                    Text("Tap to label 點我標記")
                        .font(.subheadline.bold())
                }
            }
            .foregroundStyle(Color.white)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(minWidth: 44, minHeight: 44)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .environment(\.colorScheme, .dark)
        }
        .buttonStyle(.plain)
    }

    private func badgeTexts() -> [String] {
        var out: [String] = []
        if meal.isDrinkDish(entry), let c = entry.choice {
            out.append(ItemSheet.sugarZh(c.sugar))
        }
        if meal.noNumbers { return out }
        for f in meal.flags(entry) {
            if f == "portion_default" { out.append("default portion") }
            else if f == "low_confidence" { out.append("low confidence") }
        }
        return out
    }
}
