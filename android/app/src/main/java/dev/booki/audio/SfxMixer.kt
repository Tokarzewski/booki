package dev.booki.audio

import kotlin.math.min

/**
 * Mixes a generated ambient bed under narration (issue #8).
 *
 * The ambient clip is looped to the narration's length, attenuated by
 * [ambientGainDb] (issue spec: -18 dB default, user-adjustable -30..-10), given
 * a fade-in/fade-out envelope at the chapter boundaries, then summed with the
 * narration and hard-clipped to [-1, 1].
 *
 * Pure function of its inputs — no Android dependencies — so the gain math and
 * envelope are unit-testable on the JVM.
 */
object SfxMixer {

    const val DEFAULT_AMBIENT_GAIN_DB = -18f
    const val MIN_AMBIENT_GAIN_DB = -30f
    const val MAX_AMBIENT_GAIN_DB = -10f
    const val DEFAULT_FADE_SECONDS = 2f

    /** dB → linear amplitude factor. */
    fun dbToLinear(db: Float): Float = Math.pow(10.0, db / 20.0).toFloat()

    /**
     * @param narration  PCM float samples in [-1, 1]
     * @param ambient    ambient clip; looped if shorter than the narration
     * @param sampleRate sample rate shared by both buffers
     * @param ambientGainDb attenuation applied to the ambient bed, clamped to
     *                   [[MIN_AMBIENT_GAIN_DB], [MAX_AMBIENT_GAIN_DB]]
     * @param fadeSeconds fade-in/out duration at the start/end of the chapter
     * @return a new buffer of the narration's length
     */
    fun mix(
        narration: FloatArray,
        ambient: FloatArray,
        sampleRate: Int,
        ambientGainDb: Float = DEFAULT_AMBIENT_GAIN_DB,
        fadeSeconds: Float = DEFAULT_FADE_SECONDS,
    ): FloatArray {
        require(sampleRate > 0) { "sampleRate must be positive" }
        if (narration.isEmpty() || ambient.isEmpty()) return narration.copyOf()

        val gain = dbToLinear(
            ambientGainDb.coerceIn(MIN_AMBIENT_GAIN_DB, MAX_AMBIENT_GAIN_DB))
        // Fades may not overlap: cap each at half the narration.
        val fadeSamples = min(
            (fadeSeconds * sampleRate).toInt().coerceAtLeast(0),
            narration.size / 2,
        )

        val out = FloatArray(narration.size)
        for (i in narration.indices) {
            val envelope = when {
                fadeSamples > 0 && i < fadeSamples -> i / fadeSamples.toFloat()
                fadeSamples > 0 && i >= narration.size - fadeSamples ->
                    (narration.size - 1 - i) / fadeSamples.toFloat()
                else -> 1f
            }
            val bed = ambient[i % ambient.size] * gain * envelope
            out[i] = (narration[i] + bed).coerceIn(-1f, 1f)
        }
        return out
    }
}
