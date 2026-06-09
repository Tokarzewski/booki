package dev.booki.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SfxMixerTest {

    private val sr = 1000 // 1 kHz keeps sample math readable

    @Test fun dbToLinearReferencePoints() {
        assertEquals(1f, SfxMixer.dbToLinear(0f), 1e-6f)
        assertEquals(0.5f, SfxMixer.dbToLinear(-6.0206f), 1e-4f)
        // The issue-spec default: -18 dB ≈ 0.1259
        assertEquals(0.12589f, SfxMixer.dbToLinear(-18f), 1e-4f)
    }

    @Test fun ambientLevelInSteadyState() {
        // 10 s of silent narration, constant full-scale ambient.
        val narration = FloatArray(10 * sr)
        val ambient = FloatArray(sr) { 1f }
        val out = SfxMixer.mix(narration, ambient, sr, ambientGainDb = -18f)
        // Middle of the buffer is past the 2 s fades → pure gain.
        assertEquals(SfxMixer.dbToLinear(-18f), out[5 * sr], 1e-5f)
    }

    @Test fun fadeInStartsAtZeroAndRises() {
        val narration = FloatArray(10 * sr)
        val ambient = FloatArray(sr) { 1f }
        val out = SfxMixer.mix(narration, ambient, sr, ambientGainDb = -18f, fadeSeconds = 2f)
        assertEquals(0f, out[0], 1e-6f)
        // Half-way through the 2 s fade-in → half the steady-state level.
        assertEquals(SfxMixer.dbToLinear(-18f) * 0.5f, out[sr], 1e-4f)
        // Last sample fades back to (near) zero.
        assertEquals(0f, out[out.size - 1], 1e-5f)
    }

    @Test fun ambientLoopsWhenShorterThanNarration() {
        val narration = FloatArray(6 * sr)
        // 1 s ramp as ambient — looping means sample k and k + sr match (steady state).
        val ambient = FloatArray(sr) { it / sr.toFloat() }
        val out = SfxMixer.mix(narration, ambient, sr, fadeSeconds = 0f)
        assertEquals(out[2 * sr + 123], out[3 * sr + 123], 1e-6f)
    }

    @Test fun narrationPreservedAndClamped() {
        val narration = FloatArray(8 * sr) { 0.9f }
        val ambient = FloatArray(sr) { 1f }
        val out = SfxMixer.mix(narration, ambient, sr, ambientGainDb = -10f)
        // Steady state: 0.9 + 0.316… < 1 → above narration but clamped ≤ 1.
        assertTrue(out[4 * sr] > 0.9f)
        out.forEach { assertTrue(it in -1f..1f) }
    }

    @Test fun gainIsClampedToSpecRange() {
        val narration = FloatArray(8 * sr)
        val ambient = FloatArray(sr) { 1f }
        // Requesting 0 dB must clamp to -10 dB (MAX_AMBIENT_GAIN_DB).
        val out = SfxMixer.mix(narration, ambient, sr, ambientGainDb = 0f)
        assertEquals(SfxMixer.dbToLinear(-10f), out[4 * sr], 1e-5f)
    }

    @Test fun shortNarrationCapsFadeAtHalfLength() {
        // 1 s narration with 2 s fades → fades cap at half a second each side.
        val narration = FloatArray(sr)
        val ambient = FloatArray(sr) { 1f }
        val out = SfxMixer.mix(narration, ambient, sr)
        assertEquals(narration.size, out.size)
        assertEquals(0f, out[0], 1e-6f)
        out.forEach { assertTrue(it in -1f..1f) }
    }

    @Test fun emptyAmbientReturnsNarrationUnchanged() {
        val narration = FloatArray(100) { 0.5f }
        val out = SfxMixer.mix(narration, FloatArray(0), sr)
        assertEquals(narration.toList(), out.toList())
    }

    @Test fun emptyNarrationReturnsEmpty() {
        assertEquals(0, SfxMixer.mix(FloatArray(0), FloatArray(10) { 1f }, sr).size)
    }
}
