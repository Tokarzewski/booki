package dev.booki.tts

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads the Kokoro model bundle to `filesDir/kokoro/` on first launch.
 * URLs are placeholders — point at a mirror you control. The expected layout
 * is documented in [KokoroEngine].
 */
object ModelDownloader {

    data class Asset(val name: String, val url: String, val sha256: String? = null)

    private val ASSETS = listOf(
        Asset("kokoro-v1.0.onnx",
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"),
        Asset("voices.bin",
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices-v1.0.bin"),
        Asset("vocab.json",
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/raw/main/tokenizer.json"),
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
            val conn = (URL(asset.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000; readTimeout = 60_000
                instanceFollowRedirects = true
            }
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val total = conn.contentLengthLong
                    val buf = ByteArray(64 * 1024)
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
