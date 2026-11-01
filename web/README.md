# Calorie Lens: Snap log (Vite + TypeScript PWA)

Take or choose one meal photo; everything runs on the phone (onnxruntime-web in a module Web Worker, WebGPU else WASM).
Pipeline and file formats: `../spec/snap_contract.md`. Models are read at runtime from `public/models/` (written by `ml/snap/`).

```
npm ci
npm run dev        # predev copies ORT wasm to public/ort; http://localhost:5173
npm test           # pure modules (node --experimental-strip-types)
npm run build -- --base=/calorie-lens/
npm run icons      # regenerate public/icons (needs Pillow)
```

- `src/worker.ts` glue (ORT, cache) -> `src/pipeline.ts` (injected runners, unit-tested) -> `yolo.ts`, `imageops.ts`, `classify.ts`, `mixture.ts`.
- Storage: IndexedDB `calorie-lens/meals` (`src/db.ts`). Each item stores `{p10,p50,p90}` and `q`: 64 quantiles at (i+0.5)/64 of its 4000 Monte Carlo samples
  (whole kcal). Meal/day totals resample each item by inverse CDF from `q` (seeded, n=4000) and sum, treating items as independent.
- Models are cached in Cache Storage (`calorie-lens-models-v1`); `public/sw.js` caches the app shell network-first and the pinned ORT wasm from jsDelivr.
- `?ortbase=<url/>` overrides where ORT wasm is loaded from.
