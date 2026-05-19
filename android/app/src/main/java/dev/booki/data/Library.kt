package dev.booki.data

import android.content.Context
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URI

/**
 * Local EPUB library stored under `filesDir/library/`. Downloads are dropped
 * here by the catalog browser and surfaced back as file:// URIs that can be
 * fed to [dev.booki.epub.EpubReader].
 */
object Library {
    private val _books = MutableStateFlow<List<Book>>(emptyList())
    val books: StateFlow<List<Book>> = _books

    data class Book(val title: String, val author: String, val file: File) {
        val sizeKb: Long get() = file.length() / 1024
    }

    fun refresh(context: Context) {
        val dir = libraryDir(context)
        val list = dir.listFiles { f -> f.extension.equals("epub", ignoreCase = true) }
            ?.map { Book(title = it.nameWithoutExtension, author = "", file = it) }
            ?.sortedBy { it.title }
            .orEmpty()
        _books.value = list
    }

    fun delete(context: Context, book: Book) {
        if (book.file.delete()) refresh(context)
    }

    fun uriFor(context: Context, book: Book) =
        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", book.file)

    suspend fun download(
        context: Context,
        title: String,
        author: String,
        url: String,
        onProgress: (done: Long, total: Long) -> Unit = { _, _ -> },
    ): Book = withContext(Dispatchers.IO) {
        val dir = libraryDir(context).apply { mkdirs() }
        val safe = "${title.sanitize()} - ${author.sanitize()}".trim('-', ' ').ifBlank { "book" }
        val out = File(dir, "$safe.epub")
        val tmp = File(dir, "$safe.epub.part")

        val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Booki/0.1")
        }
        conn.inputStream.use { input ->
            tmp.outputStream().use { output ->
                val total = conn.contentLengthLong
                val buf = ByteArray(64 * 1024)
                var done = 0L
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    output.write(buf, 0, n)
                    done += n
                    onProgress(done, total)
                }
            }
        }
        check(tmp.renameTo(out)) { "Failed to finalize $out" }
        refresh(context)
        Book(title = title, author = author, file = out)
    }

    private fun libraryDir(context: Context): File = File(context.filesDir, "library")

    private fun String.sanitize(): String =
        replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).trim()
}
