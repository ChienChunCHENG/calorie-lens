import XCTest
import simd
@testable import CalorieCore

final class DumpIOTests: XCTestCase {
    private func tempDir(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
        return url
    }

    private func makeFrame(index: Int) -> SweepFrame {
        let w = 8
        let h = 6
        // Values exactly representable in Float16 (multiples of 1/8, magnitude < 2048).
        var rel = FloatGrid(width: w, height: h, repeating: 0)
        for v in 0..<h { for u in 0..<w { rel[u, v] = Float(u * 8 + v) * 0.125 + Float(index) } }
        var food = Mask(width: w, height: h, repeating: false)
        var bowl = Mask(width: w, height: h, repeating: false)
        for v in 1..<4 { for u in 2..<6 { food[u, v] = true } }
        for v in 0..<h { for u in 1..<7 { bowl[u, v] = true } }
        let tw = simd_double4x4(rowMajor: [0.0, -1.0, 0.0, 0.25 + Double(index),
                                           1.0, 0.0, 0.0, -0.125,
                                           0.0, 0.0, 1.0, 0.5,
                                           0.0, 0.0, 0.0, 1.0])
        let k = Intrinsics(fx: 200.5, fy: 199.5, cx: 4.25, cy: 3.125)
        let fps: [SIMD3<Double>] = [SIMD3(0.5, -0.25, 1.0), SIMD3(-1.5, 2.0, 0.125), SIMD3(0.0, 0.0, -3.0)]
        return SweepFrame(t: 0.25 * Double(index), T_wc: tw, K: k, relDepth: rel, depthM: nil, conf: nil,
                          masks: ["rice": food, "bowl": bowl], labels: ["rice": "rice_white", "bowl": "container_bowl"],
                          featurePointsW: fps)
    }

    private func makeMeta() -> DumpMeta {
        var meta = DumpMeta()
        meta.deviceModel = "iPhone-test"
        meta.iosVersion = "17.4"
        meta.imageSize = [1920, 1440]
        meta.depthModel = "unit-test"
        meta.plane = Plane(n: SIMD3(0.0, 1.0, 0.0), offset: -0.75, inlierRMSEm: 0.002)
        meta.planeSource = "arkit"
        meta.containers["rice"] = ContainerSpec(shape: .bowl, itemKey: "bowl",
                                                rimDiameterCm: Triple(min: 11.0, typ: 12.0, max: 13.0),
                                                depthCm: Triple(scalar: 4.5))
        return meta
    }

    func testRoundTrip() throws {
        let dir = try tempDir("dumpio_rt")
        defer { try? FileManager.default.removeItem(at: dir) }
        let frames = [makeFrame(index: 0), makeFrame(index: 1), makeFrame(index: 2)]
        let meta = makeMeta()
        try DumpIO.write(frames: frames, meta: meta, rgbJPEGs: nil, to: dir)

        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("manifest.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("rel_02.f16").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("mask_01_rice.png").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("fp_00.f32").path))

        let back = try DumpIO.read(dir)
        XCTAssertEqual(back.frames.count, frames.count)
        for (a, b) in zip(frames, back.frames) {
            XCTAssertEqual(a.t, b.t, accuracy: 1e-12)
            XCTAssertEqual(b.width, a.width)
            XCTAssertEqual(b.height, a.height)
            for i in 0..<a.relDepth.data.count {
                XCTAssertEqual(a.relDepth.data[i], b.relDepth.data[i], accuracy: 1e-6)
            }
            XCTAssertEqual(Set(a.masks.keys), Set(b.masks.keys))
            for (key, m) in a.masks {
                XCTAssertEqual(m.data, b.masks[key]?.data, "mask \(key)")
            }
            XCTAssertEqual(a.labels, b.labels)
            XCTAssertEqual(a.K, b.K)
            let ra = a.T_wc.rowMajor
            let rb = b.T_wc.rowMajor
            for i in 0..<16 { XCTAssertEqual(ra[i], rb[i], accuracy: 1e-12) }
            XCTAssertEqual(a.featurePointsW.count, b.featurePointsW.count)
            for (p, q) in zip(a.featurePointsW, b.featurePointsW) {
                XCTAssertEqual(p.x, q.x, accuracy: 1e-6)
                XCTAssertEqual(p.y, q.y, accuracy: 1e-6)
                XCTAssertEqual(p.z, q.z, accuracy: 1e-6)
            }
        }

        let m2 = back.meta
        XCTAssertEqual(m2.deviceModel, "iPhone-test")
        XCTAssertEqual(m2.iosVersion, "17.4")
        XCTAssertEqual(m2.imageSize ?? [], [1920, 1440])
        XCTAssertEqual(m2.depthModel, "unit-test")
        XCTAssertEqual(m2.planeSource, "arkit")
        let p = try XCTUnwrap(m2.plane)
        XCTAssertEqual(p.n.y, 1.0, accuracy: 1e-12)
        XCTAssertEqual(p.offset, -0.75, accuracy: 1e-12)
        XCTAssertEqual(p.inlierRMSEm, 0.0)
        let c = try XCTUnwrap(m2.containers["rice"])
        XCTAssertEqual(c, meta.containers["rice"])
    }

    func testRejectsBadItemKey() throws {
        let dir = try tempDir("dumpio_badkey")
        defer { try? FileManager.default.removeItem(at: dir) }
        var f = makeFrame(index: 0)
        f.masks["bad key/../x"] = Mask(width: 8, height: 6, repeating: true)
        XCTAssertThrowsError(try DumpIO.write(frames: [f], meta: DumpMeta(), rgbJPEGs: nil, to: dir))
    }

    func testNoPlaneAndEmptyFeaturePoints() throws {
        let dir = try tempDir("dumpio_noplane")
        defer { try? FileManager.default.removeItem(at: dir) }
        var f = makeFrame(index: 0)
        f.featurePointsW = []
        try DumpIO.write(frames: [f], meta: DumpMeta(), rgbJPEGs: [nil], to: dir)
        let back = try DumpIO.read(dir)
        XCTAssertNil(back.meta.plane)
        XCTAssertEqual(back.frames[0].featurePointsW.count, 0)
        XCTAssertTrue(back.meta.containers.isEmpty)
    }

    func testZipProducesFile() throws {
        let dir = try tempDir("dumpio_zip")
        defer { try? FileManager.default.removeItem(at: dir) }
        try DumpIO.write(frames: [makeFrame(index: 0)], meta: makeMeta(), rgbJPEGs: nil, to: dir)
        let zipURL: URL
        do {
            zipURL = try DumpIO.zip(dir)
        } catch {
            // NSFileCoordinator .forUploading behaviour on a bare macOS test host is the only unknown here.
            throw XCTSkip("zip not available in this environment: \(error)")
        }
        defer { try? FileManager.default.removeItem(at: zipURL) }
        let data = try Data(contentsOf: zipURL)
        XCTAssertGreaterThan(data.count, 100)
        XCTAssertEqual(Array(data.prefix(2)), [0x50, 0x4B], "zip magic 'PK'")
        XCTAssertEqual(zipURL.pathExtension, "zip")
    }

    /// Reads a Python-written dump (only when CL_FIXTURES is set).
    func testReadsPythonDump() throws {
        guard let fx = ProcessInfo.processInfo.environment["CL_FIXTURES"], !fx.isEmpty else {
            throw XCTSkip("CL_FIXTURES not set")
        }
        let dir = URL(fileURLWithPath: fx, isDirectory: true).appendingPathComponent("bowl_dome", isDirectory: true)
        let back = try DumpIO.read(dir)
        XCTAssertEqual(back.frames.count, 12)
        for f in back.frames {
            XCTAssertEqual(f.width, 256)
            XCTAssertEqual(f.height, 192)
            XCTAssertEqual(f.K.fx, 200.0, accuracy: 1e-9)
            XCTAssertEqual(f.K.cx, 128.0, accuracy: 1e-9)
            XCTAssertEqual(f.K.cy, 96.0, accuracy: 1e-9)
            // rotation part orthonormal, last row (0 0 0 1)
            let r = f.T_wc.rowMajor
            XCTAssertEqual(r[12], 0.0, accuracy: 1e-12)
            XCTAssertEqual(r[15], 1.0, accuracy: 1e-12)
            let c0 = SIMD3<Double>(r[0], r[4], r[8])
            XCTAssertEqual(simd_length(c0), 1.0, accuracy: 1e-6)
            XCTAssertEqual(Set(f.masks.keys), Set(["rice", "bowl"]))
            XCTAssertEqual(f.labels["bowl"], "container_bowl")
            XCTAssertTrue(f.masks["rice"]?.data.contains(true) ?? false)
            XCTAssertFalse(f.featurePointsW.isEmpty)
            XCTAssertTrue(f.relDepth.data.allSatisfy { $0.isFinite })
        }
        XCTAssertNotNil(back.meta.plane)
        let c = try XCTUnwrap(back.meta.containers["rice"])
        XCTAssertEqual(c.shape, .bowl)
        XCTAssertEqual(c.itemKey, "bowl")
        let rim = try XCTUnwrap(c.rimDiameterCm)
        let dep = try XCTUnwrap(c.depthCm)
        XCTAssertEqual(rim.typ, 12.0, accuracy: 1e-9)
        XCTAssertEqual(dep.typ, 4.5, accuracy: 1e-9)
    }
}
