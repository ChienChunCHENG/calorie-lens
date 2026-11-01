import { defineConfig, type Plugin } from 'vite';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

// onnxruntime-web references its wasm via new URL(); Vite would emit a ~29 MB copy, which Cloudflare Pages
// (25 MiB per file) rejects. We load wasm via ort.env.wasm.wasmPaths instead, so drop the emitted copies.
const dropOrtWasm: Plugin = {
  name: 'drop-ort-wasm',
  generateBundle(_o, bundle) {
    for (const k of Object.keys(bundle)) if (k.endsWith('.wasm')) delete bundle[k];
  },
};

export default defineConfig({
  plugins: [dropOrtWasm],
  server: {
    host: true,
    port: 5173,
    allowedHosts: true, // cloudflared tunnel hostnames
    fs: { allow: [repoRoot] }, // lets web import ../core/ts/rangeEngine.ts
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
  worker: { format: 'es', plugins: () => [dropOrtWasm] },
  build: { target: 'es2022' },
});
