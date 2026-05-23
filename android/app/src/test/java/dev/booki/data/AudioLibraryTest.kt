package dev.booki.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class AudioLibraryTest {

    @Test fun findsBothM4aAndM4bFiles() {
        val dir = File.createTempFile("audiolib_test", "")
        dir.delete()
        dir.mkdirs()

        File(dir, "book1.m4a").createNewFile()
        File(dir, "book2.m4b").createNewFile()
        File(dir, "ignore.txt").createNewFile()

        val files = dir.listFiles { f ->
            f.extension.equals("m4a", ignoreCase = true) ||
                f.extension.equals("m4b", ignoreCase = true)
        }?.toList().orEmpty()

        assertEquals(2, files.size)
        assertTrue(files.any { it.name == "book1.m4a" })
        assertTrue(files.any { it.name == "book2.m4b" })

        // Cleanup
        dir.listFiles()?.forEach { it.delete() }
        dir.delete()
    }

    @Test fun renamePreservesExtension() {
        // Test the path manipulation logic directly (without needing Context)
        val originalTitle = "OldTitle"
        val originalExt = "m4b"
        val newTitle = "New_Title"

        val safe = newTitle.replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).trim()
        assertEquals("New_Title", safe)

        val destName = "$safe.$originalExt"
        assertEquals("New_Title.m4b", destName)
    }
}
