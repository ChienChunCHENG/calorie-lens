import XCTest
import Foundation
import CalorieCore

/// Walks up from this file until a directory containing spec/vectors.json exists.
private func nutritionRepoRoot(_ file: String = #filePath) throws -> URL {
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

final class NutritionDataTests: XCTestCase {

    private func loadData() throws -> MealData {
        let root = try nutritionRepoRoot()
        let url = root.appendingPathComponent("data").appendingPathComponent("nutrition.json")
        let bytes = try Data(contentsOf: url)
        return try MealData.load(jsonData: bytes)
    }

    func testLoadsStructure() throws {
        let data = try loadData()
        XCTAssertEqual(data.dishOrder.count, 15)
        XCTAssertEqual(data.dishOrder.first, "rice_white")
        XCTAssertEqual(data.dishes.count, data.dishOrder.count)
        XCTAssertNotNil(data.dishes["lu_rou_fan"])
        XCTAssertNotNil(data.drinks["bubble_milk_tea"])
        XCTAssertEqual(data.fullSugarGPer700, 61.6, accuracy: 1e-9)
        XCTAssertEqual(data.sugarLevels["half"] ?? -1.0, 0.727, accuracy: 1e-9)
        XCTAssertEqual(data.containers.count, 7)
        XCTAssertEqual(data.dishes["rice_white"]?.containerPrior ?? [], ["bowl_small", "bowl_medium", "bento_box"])
        XCTAssertNil(data.dishes["rice_white"]?.entry.oilGPer100g)
        XCTAssertNil(data.dishes["bubble_milk_tea"]?.entry.densityGml)
    }

    func testContainerSpecs() throws {
        let data = try loadData()
        XCTAssertEqual(data.containers["bowl_small"]?.spec?.shape, ContainerSpec.Shape.bowl)
        XCTAssertEqual(data.containers["plate_26cm"]?.spec?.shape, ContainerSpec.Shape.plate)
        XCTAssertEqual(data.containers["bento_box"]?.spec?.shape, ContainerSpec.Shape.box)
        XCTAssertEqual(data.containers["cup_700"]?.spec?.shape, ContainerSpec.Shape.cup)
        XCTAssertEqual(data.containers["bowl_small"]?.spec?.rimDiameterCm?.typ ?? -1.0, 11.5, accuracy: 1e-9)
        XCTAssertEqual(data.containers["bowl_small"]?.spec?.depthCm?.typ ?? -1.0, 6.0, accuracy: 1e-9)
    }

    func testVolumeFor() throws {
        let data = try loadData()
        let v = MealLogic.volumeFor(containerId: "bowl_small", data: data)
        XCTAssertNotNil(v)
        XCTAssertEqual(v?.min ?? -1.0, 250.0 * 0.6, accuracy: 1e-9)
        XCTAssertEqual(v?.typ ?? -1.0, 320.0 * 0.8, accuracy: 1e-9)
        XCTAssertEqual(v?.max ?? -1.0, 420.0, accuracy: 1e-9)
        XCTAssertNil(MealLogic.volumeFor(containerId: "plate_26cm", data: data))
        XCTAssertNil(MealLogic.volumeFor(containerId: "bento_box", data: data))
        XCTAssertNil(MealLogic.volumeFor(containerId: MealLogic.unknownContainer, data: data))
    }

    func testWebParityNumbers() throws {
        let data = try loadData()
        // Values produced by web/src/meal.ts estimateItem on the same nutrition.json.
        let rice = MealLogic.safeEstimate(
            data: data,
            choice: MealItemChoice(dishId: "rice_white", containerId: "bowl_small", sugar: .half, ice: .normal),
            measuredVolumeMl: nil)
        XCTAssertNotNil(rice)
        XCTAssertEqual(rice?.med ?? -1.0, 327.936, accuracy: 1.0)
        XCTAssertEqual(rice?.lo ?? -1.0, 186.667, accuracy: 1.0)
        XCTAssertEqual(rice?.hi ?? -1.0, 576.117, accuracy: 1.0)
        XCTAssertEqual(rice?.flags ?? ["x"], [])

        let lrf = MealLogic.safeEstimate(
            data: data,
            choice: MealItemChoice(dishId: "lu_rou_fan", containerId: "bowl_small", sugar: .half, ice: .normal),
            measuredVolumeMl: nil)
        XCTAssertNotNil(lrf)
        XCTAssertEqual(lrf?.med ?? -1.0, 450.877, accuracy: 1.0)
        XCTAssertEqual(lrf?.flags ?? [], ["low_confidence"])

        let tea = MealLogic.safeEstimate(
            data: data,
            choice: MealItemChoice(dishId: "bubble_milk_tea", containerId: "cup_700", sugar: .half, ice: .normal),
            measuredVolumeMl: nil)
        XCTAssertNotNil(tea)
        XCTAssertEqual(tea?.med ?? -1.0, 584.133, accuracy: 1.0)
        XCTAssertEqual(tea?.lo ?? -1.0, 434.350, accuracy: 1.0)
        XCTAssertEqual(tea?.hi ?? -1.0, 753.916, accuracy: 1.0)
        XCTAssertEqual(tea?.flags ?? [], ["drink_sugar_assumed"])
    }

    func testMeasuredVolumeOverridesContainer() throws {
        let data = try loadData()
        let choice = MealItemChoice(dishId: "rice_white", containerId: "bowl_small", sugar: .half, ice: .normal)
        let measured = Triple(min: 200.0, typ: 280.0, max: 380.0)
        let est = try MealLogic.estimate(data: data, choice: choice, measuredVolumeMl: measured)
        // rice_white: density typ 0.7, kcal typ 183 -> 280 * 0.7 * 183 / 100
        XCTAssertEqual(est.med, 280.0 * 0.7 * 183.0 / 100.0, accuracy: 1e-6)
        XCTAssertFalse(est.flags.contains("portion_default"))

        // Plate has no volume prior: default portion unless a measured volume is supplied.
        let plate = MealItemChoice(dishId: "rice_white", containerId: "plate_26cm", sugar: .half, ice: .normal)
        let noVol = try MealLogic.estimate(data: data, choice: plate, measuredVolumeMl: nil)
        XCTAssertTrue(noVol.flags.contains("portion_default"))
        let withVol = try MealLogic.estimate(data: data, choice: plate, measuredVolumeMl: measured)
        XCTAssertFalse(withVol.flags.contains("portion_default"))
    }

    func testDrinkIgnoresMeasuredVolume() throws {
        let data = try loadData()
        let choice = MealItemChoice(dishId: "bubble_milk_tea", containerId: "cup_500", sugar: .none, ice: .less)
        let a = try MealLogic.estimate(data: data, choice: choice, measuredVolumeMl: nil)
        let b = try MealLogic.estimate(data: data, choice: choice, measuredVolumeMl: Triple(min: 1.0, typ: 2.0, max: 3.0))
        XCTAssertEqual(a, b)
    }

    func testUnknownDishIsSafe() throws {
        let data = try loadData()
        let choice = MealItemChoice(dishId: "no_such_dish", containerId: "bowl_small", sugar: .half, ice: .normal)
        XCTAssertNil(MealLogic.safeEstimate(data: data, choice: choice, measuredVolumeMl: nil))
        XCTAssertThrowsError(try MealLogic.estimate(data: data, choice: choice, measuredVolumeMl: nil))
    }

    func testContainerChoices() throws {
        let data = try loadData()
        XCTAssertEqual(MealLogic.containerChoices(data: data, dishId: "lu_rou_fan", current: nil),
                       ["bowl_small", "bowl_medium"])
        XCTAssertEqual(MealLogic.containerChoices(data: data, dishId: "lu_rou_fan", current: "plate_26cm"),
                       ["bowl_small", "bowl_medium", "plate_26cm"])
        XCTAssertEqual(MealLogic.containerChoices(data: data, dishId: "lu_rou_fan", current: "bowl_small"),
                       ["bowl_small", "bowl_medium"])
        XCTAssertEqual(MealLogic.containerChoices(data: data, dishId: "lu_rou_fan", current: "unknown"),
                       ["bowl_small", "bowl_medium"])
    }

    func testDefaultChoice() throws {
        let data = try loadData()
        let rice = MealLogic.defaultChoice(data: data, dishId: "rice_white", preferBowl: false)
        XCTAssertEqual(rice.containerId, "bowl_small")
        XCTAssertEqual(rice.sugar, SugarLevel.half)
        XCTAssertEqual(rice.ice, Ice.normal)
        let chicken = MealLogic.defaultChoice(data: data, dishId: "chicken_leg", preferBowl: false)
        XCTAssertEqual(chicken.containerId, "bento_box")
        let chickenBowl = MealLogic.defaultChoice(data: data, dishId: "chicken_leg", preferBowl: true)
        XCTAssertEqual(chickenBowl.containerId, "bowl_small")
        let tea = MealLogic.defaultChoice(data: data, dishId: "bubble_milk_tea", preferBowl: true)
        XCTAssertEqual(tea.containerId, "cup_700")
        let missing = MealLogic.defaultChoice(data: data, dishId: "nope", preferBowl: false)
        XCTAssertEqual(missing.containerId, MealLogic.unknownContainer)
    }

    func testCombineMealThroughLogic() throws {
        let data = try loadData()
        let c1 = MealItemChoice(dishId: "rice_white", containerId: "bowl_small", sugar: .half, ice: .normal)
        let c2 = MealItemChoice(dishId: "bubble_milk_tea", containerId: "cup_700", sugar: .half, ice: .normal)
        var ests: [ItemEstimate] = []
        for c in [c1, c2] {
            if let e = MealLogic.safeEstimate(data: data, choice: c, measuredVolumeMl: nil) { ests.append(e) }
        }
        XCTAssertEqual(ests.count, 2)
        let m = RangeEngine.combineMeal(ests)
        XCTAssertEqual(m.med, 327.936 + 584.133, accuracy: 1.0)
        XCTAssertLessThan(m.lo, m.med)
        XCTAssertGreaterThan(m.hi, m.med)
    }
}
