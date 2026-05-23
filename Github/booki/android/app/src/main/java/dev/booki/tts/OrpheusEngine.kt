package dev.booki.tts

import android.content.Context

/**
 * Placeholder for the future Orpheus 3B engine. Tracked in issue #6.
 *
 * Real implementation is blocked on llama.cpp merging Orpheus / SNAC support
 * (https://github.com/ggml-org/llama.cpp/pull/12487). This stub keeps the
 * [Engines] registry stable so the Settings UI can list Orpheus and the
 * service path falls back to Kokoro cleanly when the user has it selected.
 */
object OrpheusEngine {
    object Factory : SpeechEngine.Factory {
        override val id = SpeechEngine.Quality.ORPHEUS_INT4
        override val displayName = "Orpheus 3B (studio)"
        override val downloadSizeMb = 2_100
        override val ramMb = 3_500
        override val isExperimental = true

        override fun isProvisioned(context: Context): Boolean = false
        override fun load(context: Context): SpeechEngine =
            error("Orpheus is not yet implemented. Tracking: github.com/Tokarzewski/booki/issues/6")
    }
}
