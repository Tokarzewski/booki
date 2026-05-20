package dev.booki.tts

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URI

/**
 * Downloads the Kokoro model bundle to `filesDir/kokoro/` on first launch.
 *
 * We use the kokoro-onnx GitHub release artifacts because they package the
 * model in a single file plus a single NPZ of voice style embeddings —
 * exactly what [KokoroEngine] expects.
 *
 * vocab is shipped as an app asset, not downloaded.
 */
object ModelDownloader {

    data class Asset(val name: String, val url: String)

    private const val BASE =
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

    private val ASSETS = listOf(
        Asset("kokoro-v1.0.onnx", "$BASE/kokoro-v1.0.onnx"),
        Asset("voices-v1.0.bin",  "$BASE/voices-v1.0.bin"),
    )

    fun isProvisioned(context: Context): Boolean {
        val dir = File(context.filesDir, "kokoro")
        return ASSETS.all { File(dir, it.name).exists() }
    }

    suspend fun download(
        context: Context,
        onProgress: (asset: String, downloaded: Long, total: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val dir = File(context.filesDir, "kokoro").apply { mkdirs() }
        for (asset in ASSETS) {
            val out = File(dir, asset.name)
            if (out.exists()) continue
            val tmp = File(dir, "${asset.name}.part")
            val conn = (URI(asset.url).toURL().openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Booki/0.2 (+https://github.com/Tokarzewski/booki)")
            }
            check(conn.responseCode in 200..299) {
                "Download failed for ${asset.name}: HTTP ${conn.responseCode}"
            }
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val total = conn.contentLengthLong
                    val buf = ByteArray(128 * 1024)
                    var read: Int
                    var done = 0L
                    while (input.read(buf).also { read = it } != -1) {
                        output.write(buf, 0, read)
                        done += read
                        onProgress(asset.name, done, total)
                    }
                }
            }
            check(tmp.renameTo(out)) { "Failed to finalize ${asset.name}" }
        }
    }
}
