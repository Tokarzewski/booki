package dev.booki.epub

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TextSplitterTest {

    @Test fun basicSentenceSplit() {
        val result = TextSplitter.split("Hello world. How are you? Fine!")
        assertTrue(result.isNotEmpty())
        // All sentences should fit in one chunk at default 500 chars.
        assertEquals(1, result.size)
    }

    @Test fun longTextIsChunked() {
        val long = (1..30).joinToString(". ") { "Sentence number $it" } + "."
        val result = TextSplitter.split(long, maxLen = 80)
        assertTrue(result.size > 1)
        result.forEach { chunk ->
            assertTrue("Chunk too long: ${chunk.length}", chunk.length <= 80)
        }
    }

    @Test fun singleLongSentenceIsHardChunked() {
        val s = "a".repeat(200)
        val result = TextSplitter.split(s, maxLen = 50)
        assertTrue(result.size >= 4)
        result.forEach { assertTrue(it.length <= 50) }
    }

    @Test fun emptyAndWhitespace() {
        assertTrue(TextSplitter.split("").isEmpty())
        assertTrue(TextSplitter.split("   \n  ").isEmpty())
    }

    @Test fun preservesSentenceBoundaries() {
        // Sentences should not be split across chunk boundaries when under maxLen.
        val text = "First sentence. Second sentence. Third sentence."
        val result = TextSplitter.split(text, maxLen = 500)
        assertEquals(1, result.size)
        assertEquals("First sentence. Second sentence. Third sentence.", result[0])
    }

    @Test fun cjkPunctuation() {
        val text = "你好！早上好。再见？"
        val result = TextSplitter.split(text, maxLen = 100)
        assertTrue(result.isNotEmpty())
    }

    @Test fun exactMaxLen() {
        val text = "A".repeat(500)
        val result = TextSplitter.split(text, maxLen = 500)
        // No sentence breaks, so it falls back to hard chunking.
        result.forEach { assertTrue(it.length <= 500) }
    }
}
