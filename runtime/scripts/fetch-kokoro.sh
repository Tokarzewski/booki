#!/usr/bin/env bash
# Fetches the sherpa-onnx Kokoro INT8 bundle and extracts model.onnx into
# runtime/cache/. Used by CI and for local bench/eval runs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/cache"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2"
TARBALL="$CACHE/kokoro-int8-multi-lang-v1_1.tar.bz2"

mkdir -p "$CACHE"

if [ -f "$CACHE/model.onnx" ]; then
    echo "Already have $CACHE/model.onnx"
    exit 0
fi

if [ ! -f "$TARBALL" ]; then
    echo "Downloading Kokoro int8 (~140 MB)…"
    curl -L --fail -o "$TARBALL" "$URL"
fi

echo "Extracting model.onnx…"
# The int8 bundle ships the model as `model.int8.onnx`; the fp32 bundle as
# `model.onnx`. Extract both names and normalize to `model.onnx`.
tar -xjf "$TARBALL" -C "$CACHE" --strip-components=1 --wildcards \
    '*/model.onnx' '*/model.int8.onnx' 2>/dev/null || true
if [ ! -f "$CACHE/model.onnx" ] && [ -f "$CACHE/model.int8.onnx" ]; then
    mv "$CACHE/model.int8.onnx" "$CACHE/model.onnx"
fi
[ -f "$CACHE/model.onnx" ] || { echo "No model.onnx in archive" >&2; exit 1; }

echo "Done. Model at $CACHE/model.onnx"
