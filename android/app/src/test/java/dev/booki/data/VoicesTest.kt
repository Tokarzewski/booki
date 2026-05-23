package dev.booki.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VoicesTest {

    @Test fun allVoicesAreUnique() {
        val ids = Voices.all.map { it.id }
        assertEquals(ids.size, ids.toSet().size)
    }

    @Test fun sidsAreSequential() {
        val sids = Voices.all.map { it.sid }
        assertEquals((0 until Voices.all.size).toList(), sids)
    }

    @Test fun defaultVoiceExists() {
        val default = Voices.all.firstOrNull { it.id == Voices.DEFAULT }
        assertTrue("Default voice '${Voices.DEFAULT}' not found", default != null)
    }

    @Test fun sidOfReturnsCorrectIndex() {
        assertEquals(0, Voices.sidOf("af_alloy"))
        assertEquals(10, Voices.sidOf("af_sky"))
        // Unknown voice falls back to af_sky (sid 10).
        assertEquals(10, Voices.sidOf("unknown_voice"))
    }

    @Test fun languageMapping() {
        val am = Voices.all.find { it.id == "am_adam" }
        assertEquals("American English", am?.language)

        val bf = Voices.all.find { it.id == "bf_alice" }
        assertEquals("British English", bf?.language)

        val zf = Voices.all.find { it.id == "zf_xiaobei" }
        assertEquals("Mandarin Chinese", zf?.language)

        val jf = Voices.all.find { it.id == "jf_alpha" }
        assertEquals("Japanese", jf?.language)
    }

    @Test fun genderFlags() {
        val af = Voices.all.find { it.id == "af_heart" }
        assertEquals(true, af?.female)

        val am = Voices.all.find { it.id == "am_adam" }
        assertEquals(false, am?.female)
    }

    @Test fun voiceCount() {
        // 11 af + 10 am + 4 bf + 4 bm + 3 ef + 1 ff + 4 hf + 2 if
        // + 5 jf + 3 pf + 8 zf = 55 voices
        assertTrue("Expected at least 50 voices, got ${Voices.all.size}",
            Voices.all.size >= 50)
    }
}
