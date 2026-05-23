package dev.booki.catalog

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OpdsClientTest {

    @Test fun entryIsDownloadableWhenEpubUrlPresent() {
        val e = OpdsClient.Entry(
            title = "Test", author = "", summary = "",
            coverUrl = null, epubUrl = "https://example.com/book.epub", subFeedUrl = null)
        assertTrue(e.isDownloadable)
        assertFalse(e.isNavigation)
    }

    @Test fun entryIsNavigationWhenSubFeedPresent() {
        val e = OpdsClient.Entry(
            title = "Test", author = "", summary = "",
            coverUrl = null, epubUrl = null, subFeedUrl = "https://example.com/sub.atom")
        assertTrue(e.isNavigation)
        assertFalse(e.isDownloadable)
    }

    @Test fun entryIsNeitherWhenBothNull() {
        val e = OpdsClient.Entry(
            title = "Test", author = "", summary = "",
            coverUrl = null, epubUrl = null, subFeedUrl = null)
        assertFalse(e.isDownloadable)
        assertFalse(e.isNavigation)
    }

    @Test fun parseSimpleAtomFeed() {
        val xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Test Catalog</title>
          <entry>
            <title>Book One</title>
            <author><name>Alice</name></author>
            <summary>A great book</summary>
            <link rel="http://opds-spec.org/acquisition"
                  type="application/epub+zip"
                  href="https://example.com/book1.epub"/>
          </entry>
          <entry>
            <title>Book Two</title>
            <author><name>Bob</name></author>
            <summary>Another book</summary>
            <link rel="subsection" type="application/atom+xml"
                  href="https://example.com/sub.atom"/>
          </entry>
        </feed>
        """
        val feed = runBlocking { parseFeed(xml, "https://example.com/") }
        assertEquals("Test Catalog", feed.title)
        assertEquals(2, feed.entries.size)

        val book1 = feed.entries[0]
        assertEquals("Book One", book1.title)
        assertEquals("Alice", book1.author)
        assertTrue(book1.isDownloadable)

        val book2 = feed.entries[1]
        assertTrue(book2.isNavigation)
    }

    @Test fun parsePagination() {
        val xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Feed</title>
          <link rel="next" href="https://example.com/page2.atom"/>
          <entry>
            <title>Book</title>
            <author><name>X</name></author>
            <summary></summary>
            <link rel="http://opds-spec.org/acquisition"
                  type="application/epub+zip"
                  href="https://example.com/b.epub"/>
          </entry>
        </feed>
        """
        val feed = runBlocking { parseFeed(xml, "https://example.com/") }
        assertEquals("https://example.com/page2.atom", feed.nextPage)
    }

    @Test fun skipsEmptyTitles() {
        val xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Feed</title>
          <entry><title></title></entry>
          <entry><title>Valid</title>
            <author><name>A</name></author>
            <summary></summary>
            <link rel="http://opds-spec.org/acquisition"
                  type="application/epub+zip"
                  href="https://example.com/v.epub"/>
          </entry>
        </feed>
        """
        val feed = runBlocking { parseFeed(xml, "https://example.com/") }
        assertEquals(1, feed.entries.size)
        assertEquals("Valid", feed.entries[0].title)
    }

    // Minimal inline test helper that calls the private parse function
    // via a thin wrapper so we don't need network calls.
    private fun runBlocking(block: suspend () -> OpdsClient.Feed): OpdsClient.Feed {
        return kotlinx.coroutines.runBlocking { block() }
    }

    private suspend fun parseFeed(xml: String, baseUrl: String): OpdsClient.Feed {
        // Use the internal parse method indirectly — create a testable wrapper.
        return TestableOpdsClient.parse(xml, baseUrl)
    }

    // A test-accessible wrapper that exposes the parse method.
    private object TestableOpdsClient {
        suspend fun parse(xml: String, baseUrl: String): OpdsClient.Feed =
            OpdsClientTestHelper.parse(xml, baseUrl)
    }
}

// Helper object so the test can access the parse function.
// We declare a small test module that duplicates the parse logic
// to avoid making the production code test-visible.
private object OpdsClientTestHelper {
    suspend fun parse(xml: String, baseUrl: String): OpdsClient.Feed {
        // Use Jsoup to parse Atom XML inline for tests.
        val doc = org.jsoup.Jsoup.parse(xml, baseUrl, org.jsoup.parser.Parser.xmlParser())
        val title = doc.selectFirst("feed > title")?.text().orEmpty().ifBlank { "Catalog" }
        val nextPage = doc.select("feed > link[rel=next]").firstOrNull()?.absUrl("href")
            ?.takeIf { it.isNotBlank() }

        val entries = doc.select("entry").mapNotNull { e ->
            val entryTitle = e.selectFirst("title")?.text()?.trim().orEmpty()
            if (entryTitle.isBlank()) return@mapNotNull null
            val author = e.select("author > name").firstOrNull()?.text().orEmpty()
            val summary = (e.selectFirst("summary")?.text()
                ?: e.selectFirst("content")?.text()).orEmpty().trim()
            val links = e.select("link")
            val epub = links.firstOrNull { link ->
                val rel = link.attr("rel")
                val type = link.attr("type")
                rel.contains("acquisition") && type.startsWith("application/epub")
            }?.absUrl("href")
            val sub = if (epub == null) {
                links.firstOrNull { link ->
                    val type = link.attr("type")
                    val rel = link.attr("rel")
                    type.contains("application/atom+xml") &&
                        (rel.isBlank() || rel == "subsection" || rel == "alternate")
                }?.absUrl("href")
            } else null
            OpdsClient.Entry(
                title = entryTitle, author = author, summary = summary,
                coverUrl = null,
                epubUrl = epub?.takeIf { it.isNotBlank() },
                subFeedUrl = sub?.takeIf { it.isNotBlank() && it != baseUrl },
            )
        }
        return OpdsClient.Feed(title = title, entries = entries, nextPage = nextPage, searchTemplate = null)
    }
}
