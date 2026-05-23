package dev.booki.tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechEngineTest {

    @Test fun factoryCount() {
        // Int8, Fp32, Matcha EN, Matcha ZH, ElevenLabs, Fish, Orpheus
        assertTrue("Expected at least 7 factories, got ${Engines.factories.size}",
            Engines.factories.size >= 7)
    }

    @Test fun factoryForKnownQualities() {
        assertEquals(
            SpeechEngine.Quality.KOKORO_FP32,
            Engines.factoryFor(SpeechEngine.Quality.KOKORO_FP32)?.id)
        assertEquals(
            SpeechEngine.Quality.KOKORO_INT8,
            Engines.factoryFor(SpeechEngine.Quality.KOKORO_INT8)?.id)
        assertEquals(
            SpeechEngine.Quality.MATCHA_EN,
            Engines.factoryFor(SpeechEngine.Quality.MATCHA_EN)?.id)
        assertEquals(
            SpeechEngine.Quality.CLOUD_ELEVENLABS,
            Engines.factoryFor(SpeechEngine.Quality.CLOUD_ELEVENLABS)?.id)
    }

    @Test fun factoryForUnknown() {
        // No factory maps to ORPHEUS_INT4 yet (placeholder only)
        val f = Engines.factoryFor(SpeechEngine.Quality.ORPHEUS_INT4)
        assertEquals(SpeechEngine.Quality.ORPHEUS_INT4, f?.id)
    }

    @Test fun qualityEnumCompleteness() {
        val names = SpeechEngine.Quality.values().map { it.name }.toSet()
        assertTrue(names.contains("KOKORO_FP32"))
        assertTrue(names.contains("KOKORO_INT8"))
        assertTrue(names.contains("MATCHA_EN"))
        assertTrue(names.contains("MATCHA_ZH_EN"))
        assertTrue(names.contains("CLOUD_ELEVENLABS"))
        assertTrue(names.contains("CLOUD_FISH"))
        assertTrue(names.contains("ORPHEUS_INT4"))
    }
}
