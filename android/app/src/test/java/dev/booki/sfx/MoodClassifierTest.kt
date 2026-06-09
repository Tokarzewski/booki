package dev.booki.sfx

import org.junit.Assert.assertEquals
import org.junit.Test

class MoodClassifierTest {

    @Test fun stormText() {
        assertEquals(
            Mood.STORM,
            MoodClassifier.classify(
                "Chapter 3",
                "The thunder cracked overhead as rain lashed the windows of the old house.",
            ),
        )
    }

    @Test fun forestText() {
        assertEquals(
            Mood.FOREST,
            MoodClassifier.classify(
                "Into the Woods",
                "They walked beneath the trees, birdsong filtering through the leaves.",
            ),
        )
    }

    @Test fun seaText() {
        assertEquals(
            Mood.SEA,
            MoodClassifier.classify(
                "The Voyage",
                "The ship rolled on the waves while gulls wheeled above the deck.",
            ),
        )
    }

    @Test fun suspenseText() {
        assertEquals(
            Mood.SUSPENSE,
            MoodClassifier.classify(
                "",
                "A shadow moved in the darkness. She heard a whisper, then silence, " +
                    "and the creak of a floorboard filled her with dread.",
            ),
        )
    }

    @Test fun battleText() {
        assertEquals(
            Mood.BATTLE,
            MoodClassifier.classify(
                "The Siege",
                "The army formed ranks; swords were drawn as the enemy began its charge.",
            ),
        )
    }

    @Test fun emptyInputFallsBackToCalm() {
        assertEquals(Mood.CALM, MoodClassifier.classify("", ""))
    }

    @Test fun neutralTextFallsBackToCalm() {
        assertEquals(
            Mood.CALM,
            MoodClassifier.classify(
                "Chapter 1",
                "She considered the question for a long moment before answering politely.",
            ),
        )
    }

    @Test fun titleOutweighsSingleBodyHit() {
        // One body hit for STORM ("rain") vs one title hit for SEA ("ocean", x3).
        assertEquals(
            Mood.SEA,
            MoodClassifier.classify(
                "The Ocean",
                "A bit of rain fell while they talked about the journey ahead.",
            ),
        )
    }

    @Test fun ignoresWordsBeyondTheWindow() {
        val padding = (1..MoodClassifier.WORD_WINDOW).joinToString(" ") { "word$it" }
        // All storm keywords arrive after the first 200 words — must not count.
        assertEquals(
            Mood.CALM,
            MoodClassifier.classify("", "$padding thunder lightning storm rain tempest"),
        )
        // Same keywords inside the window do count.
        assertEquals(
            Mood.STORM,
            MoodClassifier.classify("", "thunder lightning storm rain tempest $padding"),
        )
    }

    @Test fun caseAndPunctuationInsensitive() {
        assertEquals(
            Mood.STORM,
            MoodClassifier.classify("THUNDER!", "RAIN, Lightning; STORM."),
        )
    }

    @Test fun everyMoodHasPrompts() {
        Mood.entries.forEach { mood ->
            org.junit.Assert.assertTrue("${mood.name} prompt empty", mood.prompt.isNotBlank())
            org.junit.Assert.assertTrue("${mood.name} sting empty", mood.stingPrompt.isNotBlank())
        }
    }
}
