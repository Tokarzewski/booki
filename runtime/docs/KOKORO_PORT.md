# Porting Kokoro to the Booki runtime

This document captures the findings from inspecting the official Kokoro
ONNX model (`sherpa-onnx/kokoro-int8-multi-lang-v1_1`) with
`runtime/tools/inspect_onnx.py`. It's the planning artifact for the
multi-session work tracked in
[issue #13](https://github.com/Tokarzewski/booki/issues/13).

## TL;DR

A direct one-session port is **not feasible**. The model needs ~15
additional op implementations plus a full int8 quantization layer that
the runtime doesn't have yet. The work breaks into ~6 focused milestones
listed below.

## What Kokoro actually is

Kokoro is a full TTS pipeline, not just a transformer. From the op
histogram, the model has three distinct sections stacked together:

1. **Text encoder + style modulation** — transformer-ish with
   `InstanceNormalization` (not RMSNorm), `MatMul`, `Add`, `Mul`, plus
   a handful of `Sin` / `Cos` (rotary or sinusoidal embeddings).
2. **Duration predictor** — uses `DynamicQuantizeLSTM` (LSTM with
   per-call int8 quantization). Predicts how long each phoneme should
   sound.
3. **Vocoder** (HiFi-GAN-family) — `Conv`, `ConvTranspose` (upsampling),
   `LeakyRelu`, `Tanh`, `Resize`. Decodes mel-style features into a
   24 kHz waveform.

## Op-coverage scan

Running `~/.booki-venv/bin/python runtime/tools/inspect_onnx.py model.onnx`
on the int8 v1.1 bundle yields the following:

```
Initializers: 908, total bytes: 107.4 MB
Graph nodes: 5895
```

### Already covered by the Booki runtime
| Op (ONNX name)        | How Booki handles it                              |
|-----------------------|---------------------------------------------------|
| `MatMul`              | `booki_graph_matmul` (SME backend, ~141 GFLOPS)   |
| `Add`, `Mul`          | `booki_graph_add` / `booki_graph_mul` (NEON)      |
| `Softmax`             | `booki_graph_softmax`                              |
| `Gather`              | `booki_graph_embedding` (when `axis=0`)            |
| `Conv` (1-D)          | `booki_graph_conv1d`                               |
| `Sigmoid` (composed with `Mul`) | `booki_graph_silu`                        |
| `Tanh`                | NEON poly inside `gelu_ps_neon`                    |
| `ReduceMean`, `Sqrt`, `Pow`, `Div`, `Reshape`, `Transpose`, `Concat`, `Slice`, `Cast`, `Squeeze`, `Unsqueeze`, `Shape`, `Range`, `Where`, `Less`, `Greater`, `Equal`, `Not`, `Identity`, `Expand`, `Constant` | fold in converter (no runtime kernel needed) |

### Missing — implement before Kokoro runs

Grouped by what they actually do, with rough effort estimates:

**Quantization stack** (only needed if we run the int8 variant)
- `DynamicQuantizeLinear` (98) — float → int8 with per-tensor scale & zero-point
- `MatMulInteger` (83) — int8 × int8 → int32 matmul
- `ConvInteger` (90) — int8 conv with zero-point
- `DequantizeLinear` (4) — int32 → float

The cleaner path is to **target the FP32 variant** (`kokoro-multi-lang-v1_1`,
348 MB) and skip this stack entirely. The fp32 model still has
`InstanceNorm`, `LSTM`, and `ConvTranspose` to deal with but drops ~275
of the missing op invocations.

**Norm + activations (broad-impact ops)** — needed by both variants
- `InstanceNormalization` (70) — per-channel mean/var normalization. ~80 LOC + NEON
- `LeakyRelu` (28) — trivial, ~10 LOC + NEON
- `Sin` (51) / `Cos` (1) — NEON polynomial, similar to the exp we have
- `Sub` (36), `Reciprocal` (48), `ConstantOfShape` (45), `Exp` (1) — small, scalar-fallback OK

**LSTM** (for the duration predictor)
- `DynamicQuantizeLSTM` (6) — a single op, but LSTM is a chunky implementation.
  In the fp32 variant this becomes plain `LSTM`. ~250 LOC. Likely a session of its own.

**Vocoder decoder ops**
- `ConvTranspose` (7) — 1-D transposed convolution for upsampling. ~150 LOC; same
  shape as `Conv` but with output-side padding/stride. We need this for any
  HiFi-GAN-family decoder.
- `Resize` (6) — bilinear / nearest 1-D resampling.
- `ScatterND` (7) / `ScatterElements` (4) — index-write ops, used inside the
  vocoder's variable-length output assembly.
- `TopK` (4), `ReduceProd` (4), `ReduceMax` (1) — scalar utility.

**Control flow & dynamic shapes (the painful part)**
- `If` (2), `Loop` (1) — graph dispatch. We'd extend `booki_graph` with
  conditional subgraph support. Workable but invasive.
- `SplitToSequence` (2), `SequenceEmpty` (1), `ConcatFromSequence` (1) —
  variable-length sequences. Probably folded into the converter rather
  than implemented as runtime ops.
- `RandomUniformLike`, `RandomNormalLike` — used in stochastic style
  sampling. Trivial (PRNG + fill) but introduces nondeterminism.

## Milestone breakdown

1. **Download + inspect the FP32 model.** Confirms it really does drop the
   quantization stack. ~5 minutes.
2. **Add 4 small ops:** `InstanceNorm`, `LeakyReLU`, `Sin`, `Sub`. Each is
   <100 LOC. ~1 session.
3. **Add `ConvTranspose` + `Resize`.** ~1 session.
4. **Add `LSTM`** (fp32, plain — not the dynamic-quantized variant). ~1 session.
5. **Build the ONNX → topology converter** for the actual Kokoro graph,
   handling the foldable ops in Python. ~1–2 sessions.
6. **Phonemizer integration.** Reuse sherpa-onnx's espeak-ng-data path; bridge
   to it via a thin JNI module on the Android side or invoke its C API
   from a small wrapper. ~1 session.

After all six: an end-to-end Kokoro forward pass running on the native
runtime. Performance optimization (NEON for the new ops, blocked matmul
tiling, multi-thread) is the next arc beyond.

## Open questions

- **InstanceNorm or LayerNorm?** Kokoro's text-encoder layers use
  InstanceNorm in this ONNX, but the upstream paper (StyleTTS2) uses
  AdaIN (adaptive instance norm). The converter needs to recognise both.
- **Style input handling.** The model takes a `[1, 256]` style vector;
  inside, this gets broadcast and modulates intermediate activations
  (AdaIN-style). Need to confirm the modulation is plain `Mul`/`Add`
  with broadcast, which we already handle.
- **LSTM direction.** Kokoro's duration predictor LSTM might be
  bidirectional — would need both forward and backward kernels.

## Why this is worth doing despite the scope

- Once we have the missing ops, we cover every TTS architecture in this
  family (Matcha-TTS, VITS, F5-TTS) with the same ~10 ops. Each becomes
  ~1 session of converter work + zero runtime work.
- The optimization wins from SME matmul (6.5× over NEON) already imply
  the runtime can compete with MLAS for the matmul-heavy parts of the
  text encoder. Once `ConvTranspose` and `Conv` get NEON paths the
  vocoder is in reach too.
- The APK shrinks significantly if/when we drop sherpa-onnx (currently
  ~28 MB of binary) in favour of the native runtime.
