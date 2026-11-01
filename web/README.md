# Calorie Lens: live food lens (Vite + TypeScript PWA)

One button ("Turn on camera"), then a Google-Lens-style live view: corner brackets and a "<food> · ~<kcal> kcal" chip on every recognised item,
plus an "On screen · ~820 kcal" pill. Everything runs on the phone (onnxruntime-web, wasm, in a module Web Worker). No logging, no history, no storage.
Recognition contract: `../spec/snap_contract.md` (locator -> CLIP crop classifier -> kcal mixture). Models are read at runtime from `public/models/`.

```
npm ci
npm run dev        # predev copies ORT wasm to public/ort; http://localhost:5173
npm test           # pure modules (node --experimental-strip-types)
npm run build -- --base=/calorie-lens/
npm run icons      # regenerate public/icons (needs Pillow)
```

- `src/main.ts` UI + camera + rAF drawing; `src/worker.ts` ORT glue; `src/pipeline.ts` `LensEngine` (injected runners, unit-tested);
  pure helpers `yolo.ts`, `imageops.ts`, `classify.ts`, `mixture.ts`, `format.ts`, `md.ts`, `assets.ts` (Cache Storage for models).
- Loop: latest frame wins. The main thread grabs one frame (`createImageBitmap(video)`, long edge 960), transfers it to the worker and sends the next
  only after the result comes back. The worker runs the locator on every frame, tracks boxes (class-agnostic greedy IoU, match >= 0.25, dropped after 4 misses),
  classifies a track when it is new and then at most every 2 s (max 2 crops per frame), and accumulates votes (sum of candidate probabilities per vocab id,
  decayed x0.9 per classification). The label is the vote winner and only changes when a challenger has > 1.3x the current label's votes.
  A track is shown after >= 2 sightings and while food votes outweigh gate rejections (negative / low-food classifications).
  kcal p10/p50/p90 come from the mixture over the voted candidates (same Monte Carlo as the contract). Frames with no box show nothing (no whole-image fallback).
- The main thread smooths boxes at rAF (time constant 90 ms), lays chips out without overlap inside the safe area, and sums visible items with `sumRange`.
- Test hook: `window.__clLens = {frames, lastMs, fps, tracks:[{id,name,p50,p10,p90,box,since}], firstLabelMs, tapToLabelMs, changes, prepMs, locMs, clsMs}`.
- `?threads=N` forces the wasm thread count (default: min(4, cores) when cross-origin isolated, else 1). `?ortbase=<url/>` overrides the ORT wasm location.
- Models are cached in Cache Storage (`calorie-lens-models-v1`); `public/sw.js` caches the app shell network-first and the pinned ORT wasm from jsDelivr.
- The "i" sheet renders `public/models/ATTRIBUTION.md` through `md.ts` (HTML-escaped first) with the "Estimates only, not medical advice." line.
