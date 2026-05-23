# Booki runtime

The custom ML runtime that will eventually replace sherpa-onnx on the
synthesis path. See [issue #13](https://github.com/Tokarzewski/booki/issues/13)
for the long-horizon plan.

This directory ships a **baseline** today — an ONNX-Runtime-backed
implementation of the `booki_runtime.h` surface. The baseline exists so the
CI scaffolding has something real to bench and gate against. When the
hand-rolled runtime starts in #13, the same header stays put and the
implementation `.c` file is what gets swapped.

## Build

```bash
cd runtime
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

ONNX Runtime is fetched automatically by CMake (Linux x86_64, Linux ARM64,
macOS universal supported). First configure adds ~50 MB to the build dir.

## Run the bench locally

```bash
./scripts/fetch-kokoro.sh                          # downloads model to runtime/cache/
./build/booki_bench --model cache/model.onnx --iters 5 --tokens 128 --json
```

## Record / check the eval golden

```bash
# Regenerate the fixture (only after a deliberate runtime change):
./build/booki_eval --model cache/model.onnx --golden golden/kokoro-int8.bin --record

# Validate against the committed fixture (CI runs this on every PR):
./build/booki_eval --model cache/model.onnx --golden golden/kokoro-int8.bin --tolerance 1e-3
```

## Layout

```
runtime/
  CMakeLists.txt
  booki_baseline.h      public C surface (stable across baseline → custom)
  booki_baseline.c      ONNX Runtime implementation (replaced in #13)
  tools/
    booki_bench.c       micro-benchmark, emits JSON for the CI gate
    booki_eval.c        golden-output diff, emits report JSON
  tests/
    test_smoke.c        ctest smoke check
  scripts/
    fetch-kokoro.sh     downloads the model used by bench + eval
  golden/
    kokoro-int8.bin     first 1024 samples of the canonical inference
  cache/                model files (gitignored)
  build/                cmake output (gitignored)
```
