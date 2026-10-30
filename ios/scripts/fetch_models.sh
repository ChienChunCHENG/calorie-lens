#!/usr/bin/env bash
# Fetch Core ML models into ios/Sources/CalorieLens/Resources/Models/ (git-ignored).
#   - Apple Depth Anything V2 Small (F16) from Hugging Face (always)
#   - Detector: only if DETECTOR_URL points to a .zip containing a *.mlpackage
# Idempotent: skips anything already present and complete. Uses only BSD-safe tools.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/Sources/CalorieLens/Resources/Models"
mkdir -p "$DEST"

HF_REPO="apple/coreml-depth-anything-v2-small"
HF_REV="main"
PKG="DepthAnythingV2SmallF16.mlpackage"
# Paths and sizes verified via the HF tree API.
FILES=(
  "Manifest.json"
  "Data/com.apple.CoreML/model.mlmodel"
  "Data/com.apple.CoreML/weights/weight.bin"
)
WEIGHT_SIZE=49419072

size_of() { wc -c < "$1" | tr -d ' '; }

depth_ok() {
  local w="$DEST/$PKG/Data/com.apple.CoreML/weights/weight.bin"
  [ -f "$DEST/$PKG/Manifest.json" ] && [ -f "$DEST/$PKG/Data/com.apple.CoreML/model.mlmodel" ] \
    && [ -f "$w" ] && [ "$(size_of "$w")" = "$WEIGHT_SIZE" ]
}

if depth_ok; then
  echo "[models] $PKG already present, skipping"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  for f in "${FILES[@]}"; do
    mkdir -p "$TMP/$PKG/$(dirname "$f")"
    echo "[models] downloading $f"
    curl -fL -sS --retry 4 --retry-delay 3 \
      -o "$TMP/$PKG/$f" \
      "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$PKG/$f"
  done
  if [ "$(size_of "$TMP/$PKG/Data/com.apple.CoreML/weights/weight.bin")" != "$WEIGHT_SIZE" ]; then
    echo "[models] weight.bin size mismatch" >&2
    exit 1
  fi
  rm -rf "${DEST:?}/$PKG"
  mv "$TMP/$PKG" "$DEST/$PKG"
  echo "[models] installed $PKG"
fi

# Optional detector.
if [ -n "${DETECTOR_URL:-}" ]; then
  if [ -f "$DEST/CalorieLensDetector.mlpackage/Manifest.json" ]; then
    echo "[models] CalorieLensDetector.mlpackage already present, skipping"
  else
    TMP2="$(mktemp -d)"
    echo "[models] downloading detector"
    curl -fL -sS --retry 4 --retry-delay 3 -o "$TMP2/d.zip" "$DETECTOR_URL"
    mkdir "$TMP2/x"
    unzip -q "$TMP2/d.zip" -d "$TMP2/x"
    SRC="$(find "$TMP2/x" -type d -name '*.mlpackage' | head -n 1)"
    if [ -z "$SRC" ]; then
      echo "[models] no .mlpackage in detector zip" >&2
      rm -rf "$TMP2"
      exit 1
    fi
    rm -rf "${DEST:?}/CalorieLensDetector.mlpackage"
    mv "$SRC" "$DEST/CalorieLensDetector.mlpackage"
    LBL="$(find "$TMP2/x" -type f -name '*labels*.json' | head -n 1)"
    if [ -n "$LBL" ]; then cp "$LBL" "$DEST/labels.json"; fi
    rm -rf "$TMP2"
    echo "[models] installed CalorieLensDetector.mlpackage"
  fi
else
  echo "[models] DETECTOR_URL not set, skipping detector"
fi

ls -l "$DEST"
