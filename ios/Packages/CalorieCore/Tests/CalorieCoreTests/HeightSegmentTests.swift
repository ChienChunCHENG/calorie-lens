// Tests for the class-agnostic height segmentation on the analytic SynthScene (see SynthScene.swift).
import XCTest
import simd
import CalorieCore

fileprivate func segTestMedian(_ a: [Double]) -> Double {
    if a.isEmpty { return Double.infinity }
    let s = a.sorted()
    let m = s.count / 2
    return s.count % 2 == 1 ? s[m] : 0.5 * (s[m - 1] + s[m])
}

final class HeightSegmentTests: XCTestCase {

    private func makeScene(masks: Bool, frames: Int = 6, mound: Bool = true, box: Bool = true) -> SynthScene {
        var cfg = SynthScene.Config()
        cfg.nFrames = frames
        cfg.includeMasks = masks
        cfg.withMound = mound
        cfg.withBox = box
        return SynthScene.make(cfg)
    }

    private func distXZ(_ p: SIMD3<Double>, _ c: SIMD2<Double>) -> Double {
        let dx = p.x - c.x
        let dz = p.z - c.y
        return (dx * dx + dz * dz).squareRoot()
    }

    /// Full native flow: align (no masks) -> segment -> align again with the segmentation masks.
    func testFindsTwoObjectsAndSupportsRealignment() {
        let scene = makeScene(masks: false)
        var frames = scene.frames
        guard let first = Align.alignFrames(&frames, planeHint: scene.plane, seed: 3) else { return XCTFail("align") }
        XCTAssertTrue(first.diags.allSatisfy { $0.ok })

        let objs = HeightSegment.segment(&frames, plane: first.plane)
        XCTAssertEqual(objs.count, 2)
        guard objs.count == 2 else { return }
        XCTAssertEqual(Set(objs.map { $0.key }), Set(["obj0", "obj1"]))
        XCTAssertEqual(objs[0].key, "obj0")
        XCTAssertEqual(objs[1].key, "obj1")
        XCTAssertGreaterThanOrEqual(objs[0].footprintAreaM2, objs[1].footprintAreaM2)

        var moundObj: SegmentedObject? = nil
        var boxObj: SegmentedObject? = nil
        for o in objs {
            if distXZ(o.centroidW, SynthScene.moundCenterXZ) < 0.02 { moundObj = o }
            if distXZ(o.centroidW, SynthScene.boxCenterXZ) < 0.02 { boxObj = o }
        }
        guard let mo = moundObj, let bo = boxObj else { return XCTFail("objects not at the expected places: \(objs)") }
        XCTAssertNotEqual(mo.key, bo.key)

        // Plane-projected VISIBLE area per frame: the far flank of the steep mound is hidden, so the mound comes out
        // below the full 6.4 cm^2 disc; the box top is 30 cm^2.
        XCTAssertGreaterThan(mo.footprintAreaM2, 0.0030)
        XCTAssertLessThan(mo.footprintAreaM2, 0.0070)
        XCTAssertGreaterThan(bo.footprintAreaM2, 0.0022)
        XCTAssertLessThan(bo.footprintAreaM2, 0.0042)
        XCTAssertEqual(mo.maxHeightM, SynthScene.moundHeightM, accuracy: 0.006)
        XCTAssertEqual(bo.maxHeightM, SynthScene.boxHeightM, accuracy: 0.006)
        XCTAssertEqual(mo.topW.y, mo.maxHeightM, accuracy: 1e-9)
        XCTAssertEqual(mo.centroidW.y, 0.5 * mo.maxHeightM, accuracy: 1e-9)
        XCTAssertGreaterThanOrEqual(mo.nFrames, 4)
        XCTAssertGreaterThanOrEqual(bo.nFrames, 4)

        // masks written into the frames, labelled "unknown", and located on the right truth object
        for (i, f) in frames.enumerated() {
            for o in [mo, bo] {
                guard let m = f.masks[o.key] else { XCTFail("frame \(i) lacks \(o.key)"); continue }
                XCTAssertEqual(f.labels[o.key], "unknown")
                let truthId = (o.key == mo.key) ? SynthScene.idMound : SynthScene.idBox
                var n = 0
                var good = 0
                for v in 0..<m.height {
                    for u in 0..<m.width where m[u, v] {
                        n += 1
                        if scene.truthId[i][u, v] == truthId { good += 1 }
                    }
                }
                XCTAssertGreaterThan(n, 50, "frame \(i) \(o.key)")
                XCTAssertGreaterThan(Double(good) / Double(max(n, 1)), 0.9, "frame \(i) \(o.key)")
            }
        }

        // second alignment pass with the masks (food excluded from the anchors) is still accurate
        guard let second = Align.alignFrames(&frames, planeHint: first.plane, seed: 4) else { return XCTFail("align 2") }
        for (i, d) in second.diags.enumerated() {
            XCTAssertTrue(d.ok, "frame \(i): \(d.reason)")
            guard let depth = frames[i].depthM else { XCTFail("no depth"); continue }
            let mound = scene.interiorErrors(frame: i, id: SynthScene.idMound, erodePx: 4, depth: depth)
            XCTAssertLessThan(segTestMedian(mound), 0.003, "mound frame \(i)")
        }

        // segment is idempotent: previous obj* masks are replaced, not accumulated or treated as detector masks
        let again = HeightSegment.segment(&frames, plane: second.plane)
        XCTAssertEqual(again.count, 2)
        XCTAssertEqual(Set(frames[0].masks.keys), Set(["obj0", "obj1"]))
    }

    func testEmptyTableHasNoObjects() {
        let scene = makeScene(masks: false, frames: 4, mound: false, box: false)
        var frames = scene.frames
        guard let res = Align.alignFrames(&frames, planeHint: scene.plane, seed: 2) else { return XCTFail("align") }
        let objs = HeightSegment.segment(&frames, plane: res.plane)
        XCTAssertTrue(objs.isEmpty, "\(objs)")
        for f in frames { XCTAssertTrue(f.masks.isEmpty) }
    }

    func testDetectorMasksAreNeverTouched() {
        let scene = makeScene(masks: true, frames: 4)
        var frames = scene.frames
        guard let res = Align.alignFrames(&frames, planeHint: scene.plane, seed: 2) else { return XCTFail("align") }
        let before = frames.map { $0.masks }
        let objs = HeightSegment.segment(&frames, plane: res.plane)
        XCTAssertTrue(objs.isEmpty, "everything above the table is already covered by detector masks: \(objs)")
        for (i, f) in frames.enumerated() {
            XCTAssertEqual(Set(f.masks.keys), Set(["mound", "box"]))
            XCTAssertEqual(f.masks["mound"]?.data, before[i]["mound"]?.data)
            XCTAssertEqual(f.masks["box"]?.data, before[i]["box"]?.data)
            XCTAssertEqual(f.labels["mound"], "unknown")
        }
    }

    func testFramesWithoutDepthAreIgnored() {
        let scene = makeScene(masks: false, frames: 3)
        var frames = scene.frames          // never aligned: depthM == nil everywhere
        let objs = HeightSegment.segment(&frames, plane: scene.plane)
        XCTAssertTrue(objs.isEmpty)
    }
}
