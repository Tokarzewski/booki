package dev.booki.tts

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsMatchaModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import dev.booki.data.Voices
import java.io.File

/**
 * sherpa-onnx-backed Matcha-TTS engine. Two variants:
 *   - `en_US` single neutral English voice (~73 MB download)
 *   - `zh_en` bilingual English/Chinese voice (~75 MB download)
 *
 * Both share the same on-disk layout under `filesDir/matcha/<variant>/` and
 * the same sample rate. Matcha uses a flow-matching architecture rather than
 * Kokoro's diffusion approach — different sound profile, smaller footprint.
 */
class MatchaEngine private constructor(
    override val name: String,
    private val tts: OfflineTts,
) : SpeechEngine {

    override val sampleRate: Int = tts.sampleRate()

    override fun synthesize(text: String, voiceId: String, speed: Float): FloatArray {
        // Matcha-TTS has a single voice per model — sid=0 is the only valid index.
        val sid = 0
        return tts.generate(text = text, sid = sid, speed = speed).samples
    }

    override fun close() {
        tts.release()
    }

    object EnFactory : SpeechEngine.Factory {
        override val id = SpeechEngine.Quality.MATCHA_EN
        override val displayName = "Matcha-TTS (English)"
        override val downloadSizeMb = 73
        override val ramMb = 200
        override fun isProvisioned(context: Context) = bundleDir(context, "en_us").let(::isComplete)
        override fun load(context: Context) = build(context, "en_us", "Matcha-TTS en_US")
    }

    object ZhEnFactory : SpeechEngine.Factory {
        override val id = SpeechEngine.Quality.MATCHA_ZH_EN
        override val displayName = "Matcha-TTS (Chinese-English)"
        override val downloadSizeMb = 75
        override val ramMb = 200
        override fun isProvisioned(context: Context) = bundleDir(context, "zh_en").let(::isComplete)
        override fun load(context: Context) = build(context, "zh_en", "Matcha-TTS zh_en")
    }

    companion object {
        fun bundleDir(context: Context, variant: String): File =
            File(context.filesDir, "matcha/$variant")

        private fun isComplete(dir: File): Boolean =
            File(dir, "model.onnx").exists() &&
                File(dir, "vocoder.onnx").exists() &&
                File(dir, "tokens.txt").exists()

        private fun build(context: Context, variant: String, name: String): MatchaEngine {
            val dir = bundleDir(context, variant)
            check(isComplete(dir)) { "Matcha/$variant bundle incomplete. Re-run setup." }

            val cfg = OfflineTtsConfig().apply {
                model = OfflineTtsModelConfig().apply {
                    matcha = OfflineTtsMatchaModelConfig(
                        acousticModel = File(dir, "model.onnx").absolutePath,
                        vocoder = File(dir, "vocoder.onnx").absolutePath,
                        lexicon = "",
                        tokens = File(dir, "tokens.txt").absolutePath,
                        dataDir = "",
                        dictDir = "",
                        noiseScale = 0.667f,
                        lengthScale = 1f,
                    )
                    numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
                    provider = "cpu"
                    debug = false
                }
            }
            return MatchaEngine(name = name, tts = OfflineTts(assetManager = null, config = cfg))
        }
    }
}
