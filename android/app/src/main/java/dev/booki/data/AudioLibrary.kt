package dev.booki.data

import android.content.Context
import androidx.core.content.FileProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/**
 * Catalog of generated audiobooks under `filesDir/out/`. Mirrors [Library] but
 * for the `.m4a` outputs of [dev.booki.tts.SynthesisService].
 */
object AudioLibrary {
    private val _items = MutableStateFlow<List<Audiobook>>(emptyList())
    val items: StateFlow<List<Audiobook>> = _items

    data class Audiobook(val title: String, val file: File) {
        val sizeMb: Long get() = file.length() / 1_048_576
        val modified: Long get() = file.lastModified()
    }

    fun refresh(context: Context) {
        val dir = outDir(context)
        val list = dir.listFiles { f -> f.extension.equals("m4a", ignoreCase = true) }
            ?.map { Audiobook(title = it.nameWithoutExtension, file = it) }
            ?.sortedByDescending { it.modified }
            .orEmpty()
        _items.value = list
    }

    fun delete(context: Context, item: Audiobook) {
        if (item.file.delete()) refresh(context)
    }

    fun rename(context: Context, item: Audiobook, newTitle: String): Audiobook? {
        val safe = newTitle.replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).trim()
        if (safe.isBlank()) return null
        val dest = File(item.file.parentFile, "$safe.m4a")
        if (dest.exists()) return null
        return if (item.file.renameTo(dest)) {
            refresh(context)
            Audiobook(title = safe, file = dest)
        } else null
    }

    fun uriFor(context: Context, item: Audiobook) =
        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", item.file)

    private fun outDir(context: Context): File = File(context.filesDir, "out").apply { mkdirs() }
}
