# Booki (Android)

Android port of [audiblez](../audiblez): EPUB → `.m4a` audiobook on-device using
Kokoro-82M via ONNX Runtime.

## Build

```bash
cd android
./gradlew :app:assembleDebug
# APK at app/build/outputs/apk/debug/app-debug.apk
```

A Gradle wrapper is not committed — generate one with `gradle wrapper --gradle-version 8.7`
if you don't already have Gradle 8.7+ installed.

## Provisioning the Kokoro model

The app expects three files in app-private storage at `Context.filesDir/kokoro/`:

| File | Source |
|------|--------|
| `kokoro-v1.0.onnx` | <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX> |
| `voices.bin` | packed style table — see format in `tts/VoiceStore.kt` (convert from `voices-v1.0.bin`) |
| `vocab.json` | `{ "<bos>": 0, "<eos>": 1, "<unk>": 2, "a": 3, ... }` token map |

You can `adb push` them for development:

```bash
adb shell mkdir -p /data/local/tmp/kokoro
adb push kokoro-v1.0.onnx voices.bin vocab.json /data/local/tmp/kokoro/
adb shell run-as dev.booki sh -c \
  'mkdir -p files/kokoro && cp /data/local/tmp/kokoro/* files/kokoro/'
```

For release, add an in-app downloader on first launch.

## Architecture

```
ui/MainActivity        Compose UI: pick EPUB, voice, speed; launch service
epub/EpubReader        epublib + jsoup → list of (title, text) chapters
tts/KokoroEngine       ONNX Runtime session over kokoro-v1.0.onnx
tts/KokoroTokenizer    text → token ids (placeholder; swap in espeak-ng-jni)
tts/VoiceStore         per-voice style embedding tables
tts/SynthesisService   foreground service running the pipeline
audio/M4aMuxer         MediaCodec AAC encoder + MP4 muxer
audio/WavWriter        per-chapter PCM dump (optional)
```

## Known gaps before shipping

- **Phonemization.** The bundled tokenizer is a per-character fallback. Real
  Kokoro quality needs espeak-ng phonemes — integrate
  [espeak-ng-jni](https://github.com/rhasspy/espeak-ng-jni) or
  [piper-phonemize](https://github.com/rhasspy/piper-phonemize) via NDK.
- **Chapter markers.** Output is `.m4a`; for true `.m4b` with chapter offsets,
  post-process the MP4 with a `chpl` atom or use mp4parser.
- **GPU.** `KokoroEngine` uses the CPU EP. Add NNAPI/XNNPACK execution providers
  for ~3× speedup on recent SoCs.
- **Streaming playback.** Generation is offline-only today; pipe samples to an
  `AudioTrack` for live preview.
