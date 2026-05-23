package dev.booki.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class LibraryTest {

    @Test fun sanitizeRejectsBlankTitle() {
        val file = File.createTempFile("library_test", ".epub")
        val book = TestBook(title = "Test_Book", author = "Author", file = file)
        assertNull(renameEpub(book, ""))
        assertNull(renameEpub(book, "   "))
        // Regex special chars get replaced with _ then trimmed — "[^]" → "__" which is not blank.
        // Only truly blank/whitespace-only titles should be rejected.
        val specialResult = renameEpub(book, "[^]")
        assertNotNull("Regex special chars should produce a valid sanitized title", specialResult)
    }

    @Test fun sanitizeStripsSpecialChars() {
        val file = File.createTempFile("library_test", ".epub")
        val book = TestBook(title = "Test_Book", author = "Author", file = file)
        val result = renameEpub(book, "Hello! @#\$%^&* World")
        assertNotNull(result)
        assertFalse("Renamed title should not contain special chars",
            result!!.title.contains(Regex("[^A-Za-z0-9 _.-]")))
    }

    @Test fun titleTruncatedTo80Chars() {
        val file = File.createTempFile("library_test", ".epub")
        val book = TestBook(title = "Test_Book", author = "Author", file = file)
        val result = renameEpub(book, "A".repeat(200))
        assertNotNull(result)
        assertTrue(result!!.title.length <= 80)
    }

    @Test fun renameRejectsExistingDestination() {
        val file = File.createTempFile("library_test", ".epub")
        val book = TestBook(title = "Test_Book", author = "Author", file = file)
        val dest = File(file.parentFile, "Existing.m4a")
        dest.createNewFile()
        assertNull(renameM4a(TestAudioBook(title = "Old", file = file), "Existing"))
        dest.delete()
    }

    private data class TestBook(val title: String, val author: String, val file: File)
    private data class TestAudioBook(val title: String, val file: File)

    private fun renameEpub(book: TestBook, newTitle: String): TestBook? {
        val safe = newTitle.replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).trim()
        if (safe.isBlank()) return null
        val dest = File(book.file.parentFile, "$safe.epub")
        if (dest.exists()) return null
        return TestBook(title = safe, author = book.author, file = dest)
    }

    private fun renameM4a(item: TestAudioBook, newTitle: String): TestAudioBook? {
        val safe = newTitle.replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).trim()
        if (safe.isBlank()) return null
        val dest = File(item.file.parentFile, "$safe.m4a")
        if (dest.exists()) return null
        return TestAudioBook(title = safe, file = dest)
    }
}
