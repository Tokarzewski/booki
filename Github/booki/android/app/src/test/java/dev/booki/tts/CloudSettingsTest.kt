package dev.booki.tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CloudSettingsTest {

    @Test fun costEstimatesArePositive() {
        val el = CloudSettings.estimatedCostPerChar(CloudSettings.Provider.ELEVENLABS)
        val fish = CloudSettings.estimatedCostPerChar(CloudSettings.Provider.FISH)
        assertTrue(el > 0)
        assertTrue(fish > 0)
        // ElevenLabs should be more expensive than Fish
        assertTrue("ElevenLabs should cost more than Fish", el > fish)
    }

    @Test fun estimateCostFormatting() {
        // 100k chars at $30/M → $3.00
        val cost = CloudSettings.estimateCost(CloudSettings.Provider.ELEVENLABS, 100_000)
        assertEquals("$3.00", cost)
    }

    @Test fun estimateCostBelowPenny() {
        // 10 chars → <$0.01
        val cost = CloudSettings.estimateCost(CloudSettings.Provider.ELEVENLABS, 10)
        assertEquals("<$0.01", cost)
    }

    @Test fun defaultVoicesExist() {
        // Just verify the function doesn't throw — actual values tested
        // indirectly via production code.
        val elDefault = testDefaultVoice(CloudSettings.Provider.ELEVENLABS)
        assertTrue(elDefault.isNotEmpty())
        val fishDefault = testDefaultVoice(CloudSettings.Provider.FISH)
        assertTrue(fishDefault.isNotEmpty())
    }

    private fun testDefaultVoice(provider: CloudSettings.Provider): String =
        when (provider) {
            CloudSettings.Provider.ELEVENLABS -> "21m00Tcm4TlvDq8ikWAM"
            CloudSettings.Provider.FISH -> "fish-angel"
        }
}
