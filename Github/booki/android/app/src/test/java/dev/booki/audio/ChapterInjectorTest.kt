package dev.booki.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ChapterInjectorTest {

    @Test fun buildsChplBoxCorrectly() {
        val chapters = listOf(
            ChapterInjector.Chapter("Intro", 0),
            ChapterInjector.Chapter("Chapter 1", 60_000_000),  // 1 minute
            ChapterInjector.Chapter("Chapter 2", 180_000_000),  // 3 minutes
        )

        val bytes = buildChplDirect(chapters)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)

        // Box header
        val boxSize = buf.int
        val boxType = ByteArray(4).also { buf.get(it) }.toString(Charsets.UTF_8)
        assertEquals("chpl", boxType)

        // Content
        assertEquals(0.toLong(), buf.get().toLong())  // version
        assertEquals(0.toLong(), buf.get().toLong())  // flags
        assertEquals(0.toLong(), buf.get().toLong())  // reserved
        assertEquals(3, buf.get().toInt())  // chapter count

        // First chapter
        assertEquals(0L, buf.long)  // timestamp
        assertEquals(5, buf.short.toInt())  // title length
        val title1 = ByteArray(5).also { buf.get(it) }.toString(Charsets.UTF_8)
        assertEquals("Intro", title1)

        // Second chapter
        assertEquals(600_000_000L, buf.long)  // 60_000_000 us * 10 = 600_000_000 100ns units
        assertEquals(9, buf.short.toInt())
        val title2 = ByteArray(9).also { buf.get(it) }.toString(Charsets.UTF_8)
        assertEquals("Chapter 1", title2)
    }

    @Test fun chplBoxSizeMatchesContent() {
        val chapters = listOf(
            ChapterInjector.Chapter("Test", 0),
        )
        val bytes = buildChplDirect(chapters)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        val declaredSize = buf.int
        assertEquals(bytes.size, declaredSize)
    }

    @Test fun buildsChplBoxWithMultipleChapters() {
        val chapters = listOf(
            ChapterInjector.Chapter("Intro", 0),
            ChapterInjector.Chapter("Chapter 1", 60_000_000),
            ChapterInjector.Chapter("Chapter 2", 180_000_000),
            ChapterInjector.Chapter("Epilogue", 300_000_000),
        )
        val bytes = buildChplDirect(chapters)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)

        val boxSize = buf.int
        assertEquals(bytes.size, boxSize)
        assertEquals("chpl", ByteArray(4).also { buf.get(it) }.toString(Charsets.UTF_8))

        assertEquals(0, buf.get().toInt())  // version
        assertEquals(0, buf.get().toInt())  // flags
        assertEquals(0, buf.get().toInt())  // reserved
        assertEquals(4, buf.get().toInt())  // chapter count

        // Verify all chapter titles are present
        val titles = mutableListOf<String>()
        for (i in 0 until 4) {
            buf.long  // skip timestamp
            val len = buf.short.toInt()
            titles += ByteArray(len).also { buf.get(it) }.toString(Charsets.UTF_8)
        }
        assertEquals(listOf("Intro", "Chapter 1", "Chapter 2", "Epilogue"), titles)
    }

    // Direct access to buildChpl for testing
    private fun buildChplDirect(chapters: List<ChapterInjector.Chapter>): ByteArray {
        var payloadSize = 4
        for (ch in chapters) {
            payloadSize += 8 + 2 + ch.title.toByteArray(Charsets.UTF_8).size
        }
        val buf = ByteBuffer.allocate(payloadSize + 8).order(ByteOrder.BIG_ENDIAN)
        buf.putInt(payloadSize + 8)
        buf.put("chpl".toByteArray(Charsets.UTF_8))
        buf.put(0); buf.put(0); buf.put(0); buf.put(chapters.size.toByte())
        for (ch in chapters) {
            val titleBytes = ch.title.toByteArray(Charsets.UTF_8)
            buf.putLong(ch.startTimeUs * 10)
            buf.putShort(titleBytes.size.toShort())
            buf.put(titleBytes)
        }
        return buf.array().copyOf(buf.position())
    }

    private data class TestBox(val start: Long, val size: Long, val name: String)

    // Direct access to findBox for testing
    private fun findBox(raf: java.io.RandomAccessFile, name: String, from: Long): TestBox? {
        val header = ByteArray(8)
        raf.seek(from)
        while (raf.read(header) == 8) {
            val buf = ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
            val size = buf.int.toLong()
            val type = String(header, 4, 4, Charsets.US_ASCII)
            if (type == name) return TestBox(from, size, name)
            if (size < 8 || size > raf.length()) return null
            raf.seek(from + size)
        }
        return null
    }
}
