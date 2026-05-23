package dev.booki.data

import android.content.Context
import dev.booki.tts.SpeechEngine

/** Lightweight key-value settings backed by SharedPreferences. */
object Settings {
    private const val PREFS = "booki.settings"
    private const val KEY_QUALITY = "quality"
    private const val KEY_DEFAULT_VOICE = "default_voice"
    private const val KEY_DEFAULT_SPEED = "default_speed"

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
}
