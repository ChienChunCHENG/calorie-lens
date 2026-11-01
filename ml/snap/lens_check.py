"""Live-lens check: runs the real site with a FAKE CAMERA (a .y4m clip) in headless Chromium and records what the lens
shows over time via window.__clLens. Used to verify speed, label stability and that empty containers show nothing.

    cd web && npm run build -- --base=/calorie-lens/ && npx vite preview --base=/calorie-lens/ --port 4176
    python ml/snap/lens_check.py clip1.y4m clip2.y4m ... [--secs 15] [--shots DIR]
"""
import argparse, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright


def run(clip: str, url: str, secs: float, shots: Path | None):
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                    f"--use-file-for-fake-video-capture={clip}"])
        ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, permissions=["camera"])
        pg = ctx.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.get_by_role("button", name="Turn on camera").click()
        tl = time.time()  # wait for the models (first analysis), then time the live behaviour
        pg.wait_for_function("window.__clLens && window.__clLens.frames >= 1", timeout=180_000)
        load_s = round(time.time() - tl, 1)
        t0 = time.time(); samples = []
        while time.time() - t0 < secs:
            time.sleep(1)
            st = pg.evaluate("window.__clLens ? JSON.parse(JSON.stringify(window.__clLens)) : null")
            if st: samples.append({"t": round(time.time() - t0, 1), "tracks": [(x["name"], x["p50"]) for x in st.get("tracks", [])],
                                    "frames": st.get("frames"), "lastMs": st.get("lastMs"), "first": st.get("firstLabelMs"),
                                    "backend": st.get("backend")})
            if shots and abs((time.time() - t0) - secs / 2) < 0.6:
                pg.screenshot(path=str(shots / (Path(clip).stem + "_mid.png")))
        b.close()
    last = samples[-1] if samples else {}
    names = sorted({n for s in samples for n, _ in s["tracks"]})
    shown = [s for s in samples if s["tracks"]]
    fr = [s["frames"] for s in samples if s["frames"] is not None]
    rate = round((fr[-1] - fr[0]) / max(1, samples[-1]["t"] - samples[0]["t"]), 2) if len(fr) > 1 else None
    return {"clip": Path(clip).name, "load_s": load_s, "analyses_per_s": rate, "first_label_ms": last.get("first"), "frames": last.get("frames"), "last_ms": last.get("lastMs"),
            "secs_with_labels": len(shown), "names": names, "final": last.get("tracks"), "errors": errs[:3]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+"); ap.add_argument("--url", default="http://localhost:4176/calorie-lens/")
    ap.add_argument("--secs", type=float, default=15); ap.add_argument("--shots")
    a = ap.parse_args()
    shots = Path(a.shots) if a.shots else None
    if shots: shots.mkdir(parents=True, exist_ok=True)
    for c in a.clips:
        print(json.dumps(run(c, a.url, a.secs, shots), ensure_ascii=False), flush=True)
