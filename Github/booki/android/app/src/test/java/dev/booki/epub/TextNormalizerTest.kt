package dev.booki.epub

import org.junit.Assert.assertEquals
import org.junit.Test

class TextNormalizerTest {

    @Test fun expandsAbbreviations() {
        assertEquals("Mister Smith arrived at Saint Petersburg.",
            TextNormalizer.normalize("Mr. Smith arrived at St. Petersburg."))
        assertEquals("Doctor Watson, Professor Moriarty, and Misses Hudson.",
            TextNormalizer.normalize("Dr. Watson, Prof. Moriarty, and Mrs. Hudson."))
    }

    @Test fun expandsDottedAcronyms() {
        assertEquals("The U S A and the U K.",
            TextNormalizer.normalize("The U.S.A. and the U.K."))
    }

    @Test fun spellsOutYearsInLiteraryRange() {
        assertEquals("It was nineteen twenty-three.",
            TextNormalizer.normalize("It was 1923."))
        assertEquals("In two thousand five he left.",
            TextNormalizer.normalize("In 2005 he left."))
        assertEquals("twenty twenty-six",
            TextNormalizer.normalize("2026"))
    }

    @Test fun spellsOutCardinals() {
        assertEquals("Chapter forty-two.", TextNormalizer.normalize("Chapter 42."))
        assertEquals("one hundred", TextNormalizer.normalize("100"))
        assertEquals("one thousand two hundred thirty-four",
            TextNormalizer.normalize("1234"))
    }

    @Test fun stripsFootnoteMarkers() {
        assertEquals("She entered the room.",
            TextNormalizer.normalize("She entered the room.[12]"))
        assertEquals("She entered the room.",
            TextNormalizer.normalize("She entered the room.²"))
        assertEquals("She entered the room ( see ).",
            TextNormalizer.normalize("She entered the room (p. 42 see fn. 3)."))
    }

    @Test fun normalizesSmartPunctuation() {
        assertEquals("\"Hello,\" she said - he replied.",
            TextNormalizer.normalize("“Hello,” she said — he replied."))
    }

    @Test fun collapsesWhitespace() {
        assertEquals("Hello world.",
            TextNormalizer.normalize("Hello    world.   "))
    }

    @Test fun handlesNegativeAndDecimal() {
        assertEquals("minus seven point one four",
            TextNormalizer.normalize("-7.14"))
    }
}
