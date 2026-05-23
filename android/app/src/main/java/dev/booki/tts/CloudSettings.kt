package dev.booki.tts

import android.content.Context

/**
 * Cloud TTS settings — API keys and voice preferences for the cloud backends.
 * Stored in plain SharedPreferences (user-supplied keys, not sensitive enough
 * for EncryptedSharedPreferences; swap to encrypted if the threat model changes).
 */
object CloudSettings {
    private const val PREFS = "booki.cloud"
    private const val KEY_PROVIDER = "cloud_provider"
    private const val KEY_API_KEY = "cloud_api_key"
    private const val KEY_VOICE_ID = "cloud_voice_id"

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** Which cloud provider the user selected. */
    enum class Provider { ELEVENLABS, FISH }

    var Context.cloudProvider: Provider
        get() = prefs(this).getString(KEY_PROVIDER, null)?.let {
            runCatching { Provider.valueOf(it) }.getOrNull()
        } ?: Provider.ELEVENLABS
        set(value) { prefs(this).edit().putString(KEY_PROVIDER, value.name).apply() }

    /** User-supplied API key for the selected provider. */
    var Context.cloudApiKey: String
        get() = prefs(this).getString(KEY_API_KEY, "").orEmpty()
        set(value) { prefs(this).edit().putString(KEY_API_KEY, value).apply() }

    /** Selected cloud voice ID (provider-specific). */
    var Context.cloudVoiceId: String
        get() = prefs(this).getString(KEY_VOICE_ID, null) ?: defaultVoiceFor(cloudProvider)
        set(value) { prefs(this).edit().putString(KEY_VOICE_ID, value).apply() }

    fun hasKey(context: Context): Boolean =
        context.cloudApiKey.isNotBlank()

    private fun defaultVoiceFor(provider: Provider): String = when (provider) {
        Provider.ELEVENLABS -> "21m00Tcm4TlvDq8ikWAM"  // Rachel
        Provider.FISH -> "fish-angel"
    }

    /** Rough cost estimates per 1M characters. */
    fun estimatedCostPerChar(provider: Provider): Double = when (provider) {
        Provider.ELEVENLABS -> 30.0 / 1_000_000   // $30/M chars (Starter)
        Provider.FISH -> 4.0 / 1_000_000           // $4/M chars
    }

    fun estimateCost(provider: Provider, charCount: Int): String {
        val cost = charCount * estimatedCostPerChar(provider)
        return if (cost < 0.01) "<$0.01" else "$%.2f".format(cost)
    }
}
