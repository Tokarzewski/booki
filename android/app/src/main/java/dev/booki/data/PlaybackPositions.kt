package dev.booki.data

import android.content.Context

/**
 * Last-known playback position per audiobook (keyed by file path). Backed by
 * SharedPreferences so it survives process death and uninstall-resistant for
 * the lifetime of the app data.
 *
 * Positions newer than [MIN_RESUME_MS] are considered "resumable" — anything
 * less than ~20 s in is treated as "from start" since the user almost
 * certainly didn't intend to abandon there.
 */
object PlaybackPositions {
    private const val PREFS = "booki.positions"
    private const val MIN_RESUME_MS = 20_000L
    private const val MIN_PROGRESS_MS = 5_000L

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** Returns the saved position in ms, or 0 if there's nothing worth resuming. */
    fun positionMs(context: Context, audiobookPath: String): Long {
        val ms = prefs(context).getLong(audiobookPath, 0L)
        return if (ms >= MIN_RESUME_MS) ms else 0L
    }

    /** Returns true iff [audiobookPath] has a resumable position. */
    fun hasResumable(context: Context, audiobookPath: String): Boolean =
        positionMs(context, audiobookPath) > 0

    fun save(context: Context, audiobookPath: String, positionMs: Long) {
        if (positionMs < MIN_PROGRESS_MS) return
        prefs(context).edit().putLong(audiobookPath, positionMs).apply()
    }

    fun clear(context: Context, audiobookPath: String) {
        prefs(context).edit().remove(audiobookPath).apply()
    }

    fun clearAll(context: Context) {
        prefs(context).edit().clear().apply()
    }
}
