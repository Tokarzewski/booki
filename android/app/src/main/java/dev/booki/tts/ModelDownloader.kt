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
 * Fetches and unpacks the sherpa-onnx Kokoro bundle (a `.tar.bz2` containing
 * the model, voices, tokens, lexicons and espeak-ng-data) into
 * `filesDir/kokoro/` on first launch.
 *
 * Default is the v1.1 multilingual fp32 build (~348 MB compressed, ~640 MB
 * unpacked). Pass [Quality.INT8] for a ~140 MB / ~310 MB unpacked variant
 * that runs ~3× faster on weak SoCs with a small quality penalty.
 */
object ModelDownloader {

    enum class Quality(val tarName: String) {
        FP32("kokoro-multi-lang-v1_1"),
        INT8("kokoro-int8-multi-lang-v1_1"),
    }

    private const val BASE =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"

    /** Files we expect to find inside the extracted bundle. */
    private val REQUIRED_FILES = listOf("model.onnx", "voices.bin", "tokens.txt")

    fun isProvisioned(context: Context): Boolean {
        val dir = File(context.filesDir, "kokoro")
        return REQUIRED_FILES.all { File(dir, it).exists() } &&
            File(dir, "espeak-ng-data").isDirectory
    }

    suspend fun download(
        context: Context,
        quality: Quality = Quality.FP32,
        onProgress: (stage: String, done: Long, total: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val dir = File(context.filesDir, "kokoro").apply { mkdirs() }
        val url = "$BASE/${quality.tarName}.tar.bz2"
        val tmp = File(context.cacheDir, "${quality.tarName}.tar.bz2.part")
        val tarball = File(context.cacheDir, "${quality.tarName}.tar.bz2")

        if (!tarball.exists()) {
            val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Booki/0.3 (+https://github.com/Tokarzewski/booki)")
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
                        onProgress("Downloading", done, total)
                    }
                }
            }
            check(tmp.renameTo(tarball)) { "Failed to finalize $tarball" }
        }

        onProgress("Extracting", 0, tarball.length())
        tarball.inputStream().buffered().use { raw ->
            BZip2CompressorInputStream(raw).use { bz2 ->
                TarArchiveInputStream(bz2).use { tar ->
                    extract(tar, dir, onProgress)
                }
            }
        }
        tarball.delete()
    }

    private fun extract(
        tar: TarArchiveInputStream,
        outDir: File,
        onProgress: (stage: String, done: Long, total: Long) -> Unit,
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
            target.outputStream().use { copy(tar, it) { n -> bytes += n; onProgress("Extracting $rel", bytes, -1L) } }
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
