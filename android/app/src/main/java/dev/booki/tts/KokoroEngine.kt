package dev.booki.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Wraps the Kokoro-82M ONNX model. The model and tokenizer/voice embeddings must be
 * provisioned by the user — see assets/README.md.
 *
 * Expected model inputs (kokoro-onnx convention):
 *   tokens : int64[1, T]      phoneme/token ids
 *   style  : float32[1, 256]  speaker style embedding (per voice)
 *   speed  : float32[1]       1.0 = normal
 * Output:
 *   audio  : float32[N]       waveform @ 24 kHz
 */
class KokoroEngine private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val tokenizer: KokoroTokenizer,
    private val voices: VoiceStore,
) : AutoCloseable {

    val sampleRate: Int = 24_000

    fun synthesize(text: String, voiceId: String, speed: Float = 1f): FloatArray {
        val tokens = tokenizer.tokenize(text)
        val style = voices.styleFor(voiceId, tokens.size)

        val tokensTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong()))
        val styleTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(style), longArrayOf(1, style.size.toLong()))
        val speedTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1))

        return tokensTensor.use { tt ->
            styleTensor.use { st ->
                speedTensor.use { sp ->
                    session.run(mapOf("tokens" to tt, "style" to st, "speed" to sp)).use { out ->
                        @Suppress("UNCHECKED_CAST")
                        (out[0].value as FloatArray)
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        fun load(context: Context): KokoroEngine {
            val modelFile = File(context.filesDir, "kokoro/kokoro-v1.0.onnx")
            check(modelFile.exists()) {
                "Kokoro model missing. Re-run first-launch setup to download it."
            }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            val session = env.createSession(modelFile.absolutePath, opts)
            val tokenizer = KokoroTokenizer.load(context)
            val voices = VoiceStore.load(context)
            return KokoroEngine(env, session, tokenizer, voices)
        }
    }
}
