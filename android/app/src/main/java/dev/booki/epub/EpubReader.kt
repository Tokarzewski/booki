package dev.booki.epub

import android.content.Context
import android.net.Uri
import io.documentnode.epub4j.epub.EpubReader as EpubLibReader
import org.jsoup.Jsoup

data class Chapter(val index: Int, val title: String, val text: String) {
    val charCount: Int get() = text.length
}

data class Book(val title: String, val author: String, val chapters: List<Chapter>)

object EpubReader {
    fun read(context: Context, uri: Uri): Book {
        val book = context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open EPUB stream" }
            EpubLibReader().readEpub(input)
        }

        val title = book.title.orEmpty().ifBlank { "Untitled" }
        val author = book.metadata.authors.firstOrNull()?.let { "${it.firstname} ${it.lastname}".trim() } ?: ""

        val chapters = book.spine.spineReferences.mapIndexedNotNull { idx, ref ->
            val resource = ref.resource ?: return@mapIndexedNotNull null
            val html = resource.reader.readText()
            val doc = Jsoup.parse(html)
            val text = doc.body().text().trim()
            if (text.isBlank()) return@mapIndexedNotNull null
            val chapterTitle = doc.selectFirst("h1, h2, h3")?.text()?.takeIf { it.isNotBlank() }
                ?: resource.title?.takeIf { it.isNotBlank() }
                ?: "Chapter ${idx + 1}"
            Chapter(index = idx, title = chapterTitle, text = text)
        }

        return Book(title = title, author = author, chapters = chapters)
    }
}
