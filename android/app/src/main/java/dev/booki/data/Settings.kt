package dev.booki.data

import android.content.Context
import dev.booki.audio.SfxMixer
import dev.booki.tts.SpeechEngine

/** Lightweight key-value settings backed by SharedPreferences. */
object Settings {
    private const val PREFS = "booki.settings"
    private const val KEY_QUALITY = "quality"
    private const val KEY_DEFAULT_VOICE = "default_voice"
    private const val KEY_DEFAULT_SPEED = "default_speed"
    private const val KEY_SFX_ENABLED = "sfx_enabled"
    private const val KEY_SFX_INTENSITY_DB = "sfx_intensity_db"

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    var Context.quality: SpeechEngine.Quality
        get() = prefs(this).getString(KEY_QUALITY, null)?.let { runCatching { SpeechEngine.Quality.valueOf(it) }.getOrNull() }
            ?: SpeechEngine.Quality.KOKORO_FP32
        set(value) { prefs(this).edit().putString(KEY_QUALITY, value.name).apply() }

    var Context.defaultVoice: String
        get() = prefs(this).getString(KEY_DEFAULT_VOICE, null) ?: Voices.DEFAULT
        set(value) { prefs(this).edit().putString(KEY_DEFAULT_VOICE, value).apply() }

    var Context.defaultSpeed: Float
        get() = prefs(this).getFloat(KEY_DEFAULT_SPEED, 1f)
        set(value) { prefs(this).edit().putFloat(KEY_DEFAULT_SPEED, value).apply() }

    // Generative SFX (issue #8) — off by default; honored once an SfxEngine ships.
    var Context.sfxEnabled: Boolean
        get() = prefs(this).getBoolean(KEY_SFX_ENABLED, false)
        set(value) { prefs(this).edit().putBoolean(KEY_SFX_ENABLED, value).apply() }

    /** Ambient bed level under narration, in dB (issue #8 spec: -30..-10). */
    var Context.sfxIntensityDb: Float
        get() = prefs(this).getFloat(KEY_SFX_INTENSITY_DB, SfxMixer.DEFAULT_AMBIENT_GAIN_DB)
            .coerceIn(SfxMixer.MIN_AMBIENT_GAIN_DB, SfxMixer.MAX_AMBIENT_GAIN_DB)
        set(value) {
            prefs(this).edit().putFloat(
                KEY_SFX_INTENSITY_DB,
                value.coerceIn(SfxMixer.MIN_AMBIENT_GAIN_DB, SfxMixer.MAX_AMBIENT_GAIN_DB),
            ).apply()
        }
}
