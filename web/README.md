# Calorie Lens: Web Lens (Vite + TypeScript)

Rear camera + on-device YOLO11 detection (onnxruntime-web, WebGPU with WASM fallback) + IoU tracking + smoothed box overlay + HUD.

```
npm install
npm run dev -- --host      # http://localhost:5173  (predev copies ORT wasm files to public/ort)
npm run build              # tsc --noEmit + vite build -> dist/
npm run check:model        # runs one dummy inference on public/models/detector.onnx (Node, wasm)
```

URL params: `?model=<url>` (override model), `?imgsz=320` (square input, multiple of 32; must match export), `?hz=8` (target inference rate, max 10).
No `/models/detector.onnx` means camera-only mode. Class names come from `/models/detector.labels.json` (array), else COCO-80.

The HUD shows the backend in use, median inference ms (last 20), effective Hz, JS heap (Chrome only; n/a on Safari), and elapsed time for the 5-minute soak test.

## Testing on an iPhone from Windows

`getUserMedia` requires HTTPS (localhost is exempt, the phone is not).

### Option A: cloudflared quick tunnel (recommended)

cloudflared is not installed by default. Install it:

```
winget install --id Cloudflare.cloudflared
```

Open a new terminal (so PATH refreshes), then:

```
npm run dev -- --host
# in a second terminal:
cloudflared tunnel --url http://localhost:5173
```

Open the printed `https://<random>.trycloudflare.com` URL in Safari on the iPhone and tap "Start camera". Vite is already configured with `allowedHosts: true`.

### Option B: local HTTPS with @vitejs/plugin-basic-ssl

`@vitejs/plugin-basic-ssl` is already a devDependency. Add to `vite.config.ts`:

```ts
import basicSsl from '@vitejs/plugin-basic-ssl';
// plugins: [basicSsl()],
```

Run `npm run dev -- --host`, find your PC's LAN IP (`ipconfig`), and open `https://<LAN-IP>:5173` on the iPhone (same Wi-Fi; allow Node through Windows Firewall). Safari warns about the self-signed certificate: tap "Show Details" > "visit this website". Camera permission works after that. Cloudflared is less fiddly.

## Deploy to Cloudflare Pages

`public/_headers` sets COOP `same-origin` and COEP `require-corp` (cross-origin isolation, needed later for multi-threaded WASM). All assets, including ORT wasm, are same-origin, so COEP is satisfied.

- Dashboard: Pages > Create project > Connect to Git. Root directory `web`, build command `npm run build`, output directory `dist`, Node 22.
- Or CLI: `npm run build && npx wrangler pages deploy dist --project-name calorie-lens`.

Note: Cloudflare Pages rejects single files over 25 MB; `detector.onnx` (about 10 MB) is fine.

## Model

`public/models/detector.onnx` is COCO YOLO11n exported at imgsz 320 (output `[1, 84, 2100]`). To regenerate (e.g. in Colab, or any Python with torch):

```
pip install ultralytics onnx onnxslim
yolo export model=yolo11n.pt format=onnx imgsz=320 opset=17 simplify=True
```

Copy `yolo11n.onnx` to `web/public/models/detector.onnx`. For a custom model, also supply `detector.labels.json` (JSON array of class names, in class-index order). Non-square or dynamic exports are not supported by the letterbox code; export with a fixed square `imgsz`.

## Notes

- Vite `server.fs.allow` includes the repo root so later code can import `../core/ts/rangeEngine.ts`.
- Dev: ORT wasm files are copied from `node_modules/onnxruntime-web/dist` to `public/ort/` by `scripts/copy-ort.mjs` (gitignored, runs on `predev`). Production build: the ORT wasm files (15-29 MB) would break Cloudflare Pages 25 MiB per-file limit, so prod loads the pinned version from jsDelivr (`?ortbase=https://.../` overrides; self-host smaller files there if you need offline).
- Wasm is single-threaded unless `crossOriginIsolated` (Vite dev and Pages both send COOP/COEP).
- iOS 26 WebGPU has had memory-leak reports; watch the HUD heap/time during the 5-minute soak and switch to `wasm` if it degrades.
