package dev.booki.tts

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import dev.booki.data.Voices
import java.io.File

/**
 * Thin wrapper around sherpa-onnx's [OfflineTts] configured for Kokoro.
 *
 * Sherpa-onnx already handles phonemization via the bundled espeak-ng (and the
 * Chinese/Japanese G2P lexicons) — so we just pass raw text in and get audio
 * out. The Kokoro bundle layout under `filesDir/kokoro/` is:
 *
 *   model.onnx
 *   voices.bin
 *   tokens.txt
 *   lexicon-us-en.txt, lexicon-zh.txt, lexicon-ja.txt, ...
 *   espeak-ng-data/
 */
class KokoroEngine private constructor(private val tts: OfflineTts) : AutoCloseable {

    val sampleRate: Int = tts.sampleRate()

    fun synthesize(text: String, voiceId: String, speed: Float = 1f): FloatArray {
        val sid = Voices.sidOf(voiceId)
        return tts.generate(text = text, sid = sid, speed = speed).samples
    }

    override fun close() {
        tts.release()
    }

    companion object {
        fun load(context: Context): KokoroEngine {
            val dir = File(context.filesDir, "kokoro")
            check(File(dir, "model.onnx").exists()) {
                "Kokoro bundle missing under $dir. Re-run first-launch setup."
            }

            val lexicons = listOf(
                "lexicon-us-en.txt", "lexicon-en-us.txt", "lexicon-gb-en.txt",
                "lexicon-zh.txt", "lexicon-ja.txt",
            ).map { File(dir, it) }.filter { it.exists() }.joinToString(",") { it.absolutePath }

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
            return KokoroEngine(OfflineTts(assetManager = null, config = cfg))
        }
    }
}
