package dev.booki.catalog

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.jsoup.Jsoup
import org.jsoup.parser.Parser
import java.net.HttpURLConnection
import java.net.URI
import java.net.URLEncoder

/**
 * Minimal OPDS 1.x (Atom) client. Parses navigation + acquisition feeds and
 * surfaces a flat list of books with a direct EPUB download URL.
 *
 * OPDS spec: https://specs.opds.io/opds-1.2
 */
object OpdsClient {

    data class Entry(
        val title: String,
        val author: String,
        val summary: String,
        val coverUrl: String?,
        val epubUrl: String?,
        val subFeedUrl: String?,
    ) {
        val isDownloadable: Boolean get() = epubUrl != null
        val isNavigation: Boolean get() = subFeedUrl != null && epubUrl == null
    }

    data class Feed(
        val title: String,
        val entries: List<Entry>,
        val nextPage: String?,
        /** OpenSearch / atom search template with literal `{searchTerms}` placeholder. */
        val searchTemplate: String?,
    )

    suspend fun fetch(url: String): Feed = withContext(Dispatchers.IO) {
        val xml = httpGetText(url)
        parse(xml, baseUrl = url)
    }

    /** Resolve `{searchTerms}` against [query] and fetch the result. */
    suspend fun search(feed: Feed, query: String): Feed {
        val tpl = feed.searchTemplate ?: error("This feed does not advertise a search endpoint")
        val q = URLEncoder.encode(query, "UTF-8")
        return fetch(tpl.replace("{searchTerms}", q))
    }

    private suspend fun parse(xml: String, baseUrl: String): Feed = withContext(Dispatchers.IO) {
        val doc = Jsoup.parse(xml, baseUrl, Parser.xmlParser())
        val title = doc.selectFirst("feed > title")?.text().orEmpty().ifBlank { "Catalog" }
        val nextPage = doc.select("feed > link[rel=next]").firstOrNull()?.absUrl("href")
            ?.takeIf { it.isNotBlank() }
        val searchTemplate = resolveSearchTemplate(doc)

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

            val cover = links.firstOrNull { link ->
                val rel = link.attr("rel")
                rel.endsWith("/image") || rel.endsWith("/thumbnail") ||
                    rel == "http://opds-spec.org/cover"
            }?.absUrl("href")

            val sub = if (epub == null) {
                links.firstOrNull { link ->
                    val type = link.attr("type")
                    val rel = link.attr("rel")
                    type.contains("application/atom+xml") &&
                        (rel.isBlank() || rel == "subsection" || rel == "alternate")
                }?.absUrl("href")
            } else null

            Entry(
                title = entryTitle,
                author = author,
                summary = summary,
                coverUrl = cover?.takeIf { it.isNotBlank() },
                epubUrl = epub?.takeIf { it.isNotBlank() },
                subFeedUrl = sub?.takeIf { it.isNotBlank() && it != baseUrl },
            )
        }

        Feed(title = title, entries = entries, nextPage = nextPage, searchTemplate = searchTemplate)
    }

    private fun resolveSearchTemplate(doc: org.jsoup.nodes.Document): String? {
        val direct = doc.select("feed > link[rel=search]").firstOrNull { link ->
            link.attr("type").contains("application/atom+xml")
        }?.absUrl("href")
        if (direct != null && direct.contains("{searchTerms}")) return direct

        val osd = doc.select("feed > link[rel=search]").firstOrNull { link ->
            link.attr("type").contains("opensearchdescription")
        }?.absUrl("href") ?: return null

        return runCatching {
            val osdXml = httpGetText(osd)
            val osdDoc = Jsoup.parse(osdXml, osd, Parser.xmlParser())
            osdDoc.select("Url").firstOrNull { it.attr("type").contains("application/atom+xml") }
                ?.absUrl("template")
                ?.takeIf { it.contains("{searchTerms}") }
        }.getOrNull()
    }

    private fun httpGetText(url: String): String {
        val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Booki/0.1 (+https://github.com/Tokarzewski/booki)")
            setRequestProperty("Accept", "application/atom+xml, application/xml;q=0.9, */*;q=0.5")
        }
        conn.inputStream.use { return it.readBytes().toString(Charsets.UTF_8) }
    }
}
