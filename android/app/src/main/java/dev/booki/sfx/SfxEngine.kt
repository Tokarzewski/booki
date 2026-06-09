package dev.booki.sfx

import android.content.Context
import dev.booki.util.DeviceCapabilities

/**
 * Pluggable generative SFX backend (issue #8), mirroring
 * [dev.booki.tts.SpeechEngine]'s structure: each implementation owns its
 * runtime + model files, and the synthesis pipeline depends only on this
 * interface.
 */
interface SfxEngine : AutoCloseable {
    val name: String
    val sampleRate: Int

    /** Generate a loopable ambient bed for one chapter. PCM floats in [-1, 1]. */
    fun generateAmbient(mood: Mood, durationSeconds: Int): FloatArray

    /** Generate a short (2–4 s) section-transition sting. */
    fun generateSting(mood: Mood): FloatArray

    interface Factory {
        val displayName: String
        val downloadSizeMb: Int
        val ramMb: Int
        val isExperimental: Boolean get() = false
        fun isProvisioned(context: Context): Boolean
        fun load(context: Context): SfxEngine

        /**
         * Device gate from the issue spec: generative SFX is only offered on
         * arm64 devices with ≥ 8 GB RAM.
         */
        fun isSupportedOn(context: Context): Boolean {
            val caps = DeviceCapabilities.of(context)
            return caps.isArm64 && caps.ramMb >= MIN_DEVICE_RAM_MB
        }
    }

    companion object {
        const val MIN_DEVICE_RAM_MB = 8_000L
    }
}

/**
 * Placeholder for the MOSS-TTS generative SFX engine. Tracked in issue #8.
 *
 * Blocked on the shared llama.cpp mobile runtime (issues #6/#7) — MOSS-TTS
 * reuses the same native dependency Orpheus adds. The mood classifier
 * ([MoodClassifier]) and mixer ([dev.booki.audio.SfxMixer]) in front of and
 * behind this engine are real and tested; only generation is stubbed.
 */
object MossSfxEngine {
    object Factory : SfxEngine.Factory {
        override val displayName = "MOSS-TTS generative SFX"
        override val downloadSizeMb = 800
        override val ramMb = 3_000
        override val isExperimental = true

        override fun isProvisioned(context: Context): Boolean = false
        override fun load(context: Context): SfxEngine =
            error("MOSS-TTS SFX is not yet implemented. Tracking: github.com/Tokarzewski/booki/issues/8")
    }
}
