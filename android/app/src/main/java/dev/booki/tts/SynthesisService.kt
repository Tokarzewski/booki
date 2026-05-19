package dev.booki.tts

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import dev.booki.audio.M4aMuxer
import dev.booki.epub.EpubReader
import kotlinx.coroutines.*
import java.io.File

/**
 * Foreground service that converts an EPUB to .m4a in app-private storage.
 * Progress is exposed via [Progress.flow] for the UI.
 */
class SynthesisService : Service() {

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val uri = intent?.getParcelableExtra<Uri>(EXTRA_URI) ?: run { stopSelf(); return START_NOT_STICKY }
        val voice = intent.getStringExtra(EXTRA_VOICE) ?: "af_sky"
        val speed = intent.getFloatExtra(EXTRA_SPEED, 1f)
        val pickedChapters = intent.getIntArrayExtra(EXTRA_CHAPTERS)?.toSet()

        startForeground(NOTIF_ID, buildNotification("Preparing…"))

        scope.launch {
            runCatching { synthesize(uri, voice, speed, pickedChapters) }
                .onFailure { Progress.error(it.message ?: "Synthesis failed") }
                .onSuccess { Progress.done(it) }
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private suspend fun synthesize(
        uri: Uri,
        voice: String,
        speed: Float,
        pickedChapters: Set<Int>?,
    ): File = withContext(Dispatchers.Default) {
        val book = EpubReader.read(this@SynthesisService, uri)
        val chapters = book.chapters.filter { pickedChapters == null || it.index in pickedChapters }
        val totalChars = chapters.sumOf { it.charCount }.coerceAtLeast(1)
        Progress.start(book.title, totalChars)

        val outDir = File(filesDir, "out").apply { mkdirs() }
        val outFile = File(outDir, "${book.title.sanitize()}.m4a")

        KokoroEngine.load(this@SynthesisService).use { engine ->
            M4aMuxer(outFile, sampleRate = engine.sampleRate).use { muxer ->
                var processed = 0
                for (chapter in chapters) {
                    Progress.chapter(chapter.title)
                    for (chunk in chapter.text.splitForTts()) {
                        ensureActive()
                        val audio = engine.synthesize(chunk, voice, speed)
                        muxer.writeSamples(audio)
                        processed += chunk.length
                        Progress.tick(processed, totalChars)
                        updateNotification("${book.title} — ${100 * processed / totalChars}%")
                    }
                }
            }
        }
        outFile
    }

    private fun buildNotification(text: String): Notification {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            mgr.createNotificationChannel(
                NotificationChannel(CHANNEL, "Audiobook synthesis", NotificationManager.IMPORTANCE_LOW))
        }
        return NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle("Booki")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        mgr.notify(NOTIF_ID, buildNotification(text))
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_URI = "uri"
        const val EXTRA_VOICE = "voice"
        const val EXTRA_SPEED = "speed"
        const val EXTRA_CHAPTERS = "chapters"
        private const val CHANNEL = "booki.synth"
        private const val NOTIF_ID = 42
    }
}

private fun String.sanitize(): String = replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).ifBlank { "audiobook" }

/** Split into ~500-char chunks at sentence boundaries to keep tensors small. */
private fun String.splitForTts(maxLen: Int = 500): List<String> {
    val sentences = split(Regex("(?<=[.!?])\\s+"))
    val out = ArrayList<String>()
    val buf = StringBuilder()
    for (s in sentences) {
        if (buf.length + s.length > maxLen && buf.isNotEmpty()) {
            out += buf.toString(); buf.setLength(0)
        }
        if (s.length > maxLen) {
            if (buf.isNotEmpty()) { out += buf.toString(); buf.setLength(0) }
            s.chunked(maxLen).forEach { out += it }
        } else {
            if (buf.isNotEmpty()) buf.append(' ')
            buf.append(s)
        }
    }
    if (buf.isNotEmpty()) out += buf.toString()
    return out
}
