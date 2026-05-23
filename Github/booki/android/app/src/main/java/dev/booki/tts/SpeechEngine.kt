package dev.booki.tts

import android.content.Context
import dev.booki.util.DeviceCapabilities

/**
 * Pluggable TTS backend. Each implementation owns its native runtime, its
 * model files, and its voice catalog. The [SynthesisService] depends only on
 * this interface so the engine can be swapped at runtime from settings.
 */
interface SpeechEngine : AutoCloseable {
    val name: String
    val sampleRate: Int

    /** Synthesize one chunk of plain text into PCM float samples in [-1, 1]. */
    fun synthesize(text: String, voiceId: String, speed: Float = 1f): FloatArray

    /** Whether this engine ships voices grouped by language code (a/b/e/f/...). */
    val supportsMultilingual: Boolean get() = true

    interface Factory {
        val id: Quality
        val displayName: String
        val downloadSizeMb: Int
        val ramMb: Int
        val isExperimental: Boolean get() = false
        fun isProvisioned(context: Context): Boolean
        fun load(context: Context): SpeechEngine

        /** True if the current device has enough RAM headroom to run this engine. */
        fun isSupportedOn(context: Context): Boolean {
            val caps = DeviceCapabilities.of(context)
            return caps.isArm64 && caps.ramMb >= ramMb * 1.5
        }
    }

    enum class Quality { KOKORO_INT8, KOKORO_FP32, MATCHA_EN, MATCHA_ZH_EN, CLOUD_ELEVENLABS, CLOUD_FISH, ORPHEUS_INT4 }
}

object Engines {
    val factories: List<SpeechEngine.Factory> by lazy {
        listOf(
            KokoroEngine.Int8Factory,
            KokoroEngine.Fp32Factory,
            MatchaEngine.EnFactory,
            MatchaEngine.ZhEnFactory,
            CloudTtsEngine.ElevenLabsFactory,
            CloudTtsEngine.FishFactory,
            OrpheusEngine.Factory,
        )
    }

    fun factoryFor(quality: SpeechEngine.Quality): SpeechEngine.Factory? =
        factories.firstOrNull { it.id == quality }

    fun supported(context: Context): List<SpeechEngine.Factory> =
        factories.filter { it.isSupportedOn(context) }
}
