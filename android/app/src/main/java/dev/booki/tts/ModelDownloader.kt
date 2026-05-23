package dev.booki.tts

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URI

/**
 * Downloads sherpa-onnx TTS bundles to `filesDir/{kokoro,matcha}/{variant}/`.
 *
 * Each [Variant] is independent — a user can keep multiple installed and switch
 * between them in settings without re-downloading. Bundles are `.tar.bz2`
 * archives hosted on the sherpa-onnx GitHub release `tts-models` tag.
 */
object ModelDownloader {

    enum class Variant(val subdir: String, val tarName: String, val sizeMb: Int) {
        // Kokoro models
        FP32("fp32", "kokoro-multi-lang-v1_1", 348),
        INT8("int8", "kokoro-int8-multi-lang-v1_1", 140),
        // Matcha-TTS models
        MATCHA_EN("matcha/en_us", "matcha-icefall-en_US-ljspeech", 73),
        MATCHA_ZH_EN("matcha/zh_en", "matcha-icefall-zh_en", 75),
    }

    private const val BASE =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"

    fun isProvisioned(context: Context, variant: Variant): Boolean {
        val dir = File(context.filesDir, "kokoro/${variant.subdir}")
        return File(dir, "model.onnx").exists() &&
            File(dir, "voices.bin").exists() &&
            File(dir, "tokens.txt").exists() &&
            File(dir, "espeak-ng-data").isDirectory
    }

    /** True if either variant is fully provisioned — caller can pick one to load. */
    fun anyProvisioned(context: Context): Boolean = Variant.entries.any { isProvisioned(context, it) }

    suspend fun download(
        context: Context,
        variant: Variant,
        onProgress: (stage: String, done: Long, total: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val outDir = File(context.filesDir, variant.subdir).apply { mkdirs() }
        val cacheDir = context.cacheDir
        val url = "$BASE/${variant.tarName}.tar.bz2"
        val tarball = File(cacheDir, "${variant.tarName}.tar.bz2")
        val tmp = File(cacheDir, "${variant.tarName}.tar.bz2.part")

        if (!tarball.exists()) {
            val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Booki/0.4 (+https://github.com/Tokarzewski/booki)")
            }
            check(conn.responseCode in 200..299) {
                "Download failed: HTTP ${conn.responseCode} for $url"
            }
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val total = conn.contentLengthLong
                    val buf = ByteArray(256 * 1024)
                    var read: Int
                    var done = 0L
                    while (input.read(buf).also { read = it } != -1) {
                        output.write(buf, 0, read)
                        done += read
                        onProgress("Downloading ${variant.subdir}", done, total)
                    }
                }
            }
            check(tmp.renameTo(tarball)) { "Failed to finalize $tarball" }
        }

        onProgress("Extracting ${variant.subdir}", 0, tarball.length())
        tarball.inputStream().buffered().use { raw ->
            BZip2CompressorInputStream(raw).use { bz2 ->
                TarArchiveInputStream(bz2).use { tar ->
                    extract(tar, outDir) { bytes -> onProgress("Extracting ${variant.subdir}", bytes, -1L) }
                }
            }
        }
        tarball.delete()
    }

    /** Removes a downloaded variant from disk. */
    fun remove(context: Context, variant: Variant) {
        val dir = File(context.filesDir, "kokoro/${variant.subdir}")
        if (dir.exists()) dir.deleteRecursively()
    }

    private fun extract(
        tar: TarArchiveInputStream,
        outDir: File,
        onChunk: (Long) -> Unit,
    ) {
        var bytes = 0L
        while (true) {
            val entry = tar.nextEntry ?: break
            // Strip the top-level "kokoro-xxx/" prefix so files land directly under outDir.
            val rel = entry.name.substringAfter('/', missingDelimiterValue = entry.name)
            if (rel.isBlank()) continue
            val target = File(outDir, rel)
            require(target.canonicalPath.startsWith(outDir.canonicalPath)) {
                "Refusing to extract outside $outDir: ${entry.name}"
            }
            if (entry.isDirectory) {
                target.mkdirs()
                continue
            }
            target.parentFile?.mkdirs()
            target.outputStream().use { copy(tar, it) { n -> bytes += n; onChunk(bytes) } }
        }
    }

    private inline fun copy(input: InputStream, output: java.io.OutputStream, onChunk: (Int) -> Unit) {
        val buf = ByteArray(64 * 1024)
        var n: Int
        while (input.read(buf).also { n = it } != -1) {
            output.write(buf, 0, n)
            onChunk(n)
        }
    }
}
