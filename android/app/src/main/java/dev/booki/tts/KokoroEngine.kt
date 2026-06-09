package dev.booki.tts

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import dev.booki.data.Voices
import dev.booki.runtime.NativeBootstrap
import java.io.File

/**
 * sherpa-onnx-backed Kokoro engine. Comes in two variants:
 *   - FP32 multilingual v1.1 (~650 MB unpacked; reference quality)
 *   - INT8 multilingual v1.1 (~310 MB unpacked; ~3× faster, slight quality drop)
 *
 * Both variants share the same on-disk layout under `filesDir/kokoro/<variant>/`,
 * the same voice catalog, and the same sample rate — they're interchangeable.
 */
class KokoroEngine private constructor(
    override val name: String,
    private val tts: OfflineTts,
) : SpeechEngine {

    override val sampleRate: Int = tts.sampleRate()

    override fun synthesize(text: String, voiceId: String, speed: Float): FloatArray {
        val sid = Voices.sidOf(voiceId)
        return tts.generate(text = text, sid = sid, speed = speed).samples
    }

    override fun close() {
        tts.release()
    }

    object Fp32Factory : SpeechEngine.Factory {
        override val id = SpeechEngine.Quality.KOKORO_FP32
        override val displayName = "Kokoro (high quality)"
        override val downloadSizeMb = 348
        override val ramMb = 600
        override fun isProvisioned(context: Context) = bundleDir(context, "fp32").let(::isComplete)
        override fun load(context: Context) = build(context, "fp32", "Kokoro v1.1 fp32")
    }

    object Int8Factory : SpeechEngine.Factory {
        override val id = SpeechEngine.Quality.KOKORO_INT8
        override val displayName = "Kokoro (fast, INT8)"
        override val downloadSizeMb = 140
        override val ramMb = 300
        override fun isProvisioned(context: Context) = bundleDir(context, "int8").let(::isComplete)
        override fun load(context: Context) = build(context, "int8", "Kokoro v1.1 int8")
    }

    companion object {
        /** Subdirectory under filesDir/kokoro for the given variant. */
        fun bundleDir(context: Context, variant: String): File =
            File(context.filesDir, "kokoro/$variant")

        private fun isComplete(dir: File): Boolean =
            File(dir, "model.onnx").exists() &&
                File(dir, "voices.bin").exists() &&
                File(dir, "tokens.txt").exists() &&
                File(dir, "espeak-ng-data").isDirectory

        private fun build(context: Context, variant: String, name: String): KokoroEngine {
            // Issue #7: in dynamic builds the sherpa-onnx .so files live in
            // filesDir and must be loaded before any sherpa class initializes.
            NativeBootstrap.ensureLoaded(context)
            val dir = bundleDir(context, variant)
            check(isComplete(dir)) { "Kokoro/$variant bundle incomplete. Re-run setup." }

            val lexicons = listOf(
                "lexicon-us-en.txt", "lexicon-en-us.txt", "lexicon-gb-en.txt",
                "lexicon-zh.txt", "lexicon-ja.txt",
            ).map { File(dir, it) }.filter { it.exists() }
                .joinToString(",") { it.absolutePath }

            val cfg = OfflineTtsConfig().apply {
                model = OfflineTtsModelConfig().apply {
                    kokoro = OfflineTtsKokoroModelConfig(
                        model = File(dir, "model.onnx").absolutePath,
                        voices = File(dir, "voices.bin").absolutePath,
                        tokens = File(dir, "tokens.txt").absolutePath,
                        dataDir = File(dir, "espeak-ng-data").absolutePath,
                        lexicon = lexicons,
                    )
                    numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
                    provider = "cpu"
                    debug = false
                }
            }
            return KokoroEngine(name = name, tts = OfflineTts(assetManager = null, config = cfg))
        }
    }
}
