package dev.booki.tts

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.content.IntentCompat
import dev.booki.audio.AudioPreview
import dev.booki.audio.ChapterInjector
import dev.booki.audio.M4aMuxer
import dev.booki.data.Settings
import dev.booki.data.Settings.quality
import dev.booki.epub.EpubReader
import dev.booki.epub.TextNormalizer
import dev.booki.epub.TextSplitter
import kotlinx.coroutines.*
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service that converts an EPUB to .m4a in app-private storage and,
 * when [EXTRA_STREAM_LIVE] is true, simultaneously streams the synthesized
 * audio to the speaker. Declared as [foregroundServiceType=mediaPlayback] so
 * playback continues with the screen off and the app backgrounded.
 *
 * Lifecycle:
 *   - START / STOP via [Intent] action
 *   - Stop is wired to the notification's stop action
 *   - A partial wake lock prevents the CPU from sleeping mid-synth
 */
class SynthesisService : Service() {

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val stopping = AtomicBoolean(false)
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopping.set(true)
            scope.cancel()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val uri = intent?.let { IntentCompat.getParcelableExtra(it, EXTRA_URI, Uri::class.java) }
            ?: run { stopSelf(); return START_NOT_STICKY }
        val voice = intent.getStringExtra(EXTRA_VOICE) ?: "af_sky"
        val speed = intent.getFloatExtra(EXTRA_SPEED, 1f)
        val streamLive = intent.getBooleanExtra(EXTRA_STREAM_LIVE, false)
        val pickedChapters = intent.getIntArrayExtra(EXTRA_CHAPTERS)?.toSet()

        startForeground(NOTIF_ID, buildNotification("Preparing…", streamLive))
        acquireWakeLock()

        scope.launch {
            runCatching { synthesize(uri, voice, speed, streamLive, pickedChapters) }
                .onFailure {
                    if (!stopping.get()) Progress.error(it.message ?: "Synthesis failed")
                }
                .onSuccess { Progress.done(it) }
            releaseWakeLock()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private suspend fun synthesize(
        uri: Uri,
        voice: String,
        speed: Float,
        streamLive: Boolean,
        pickedChapters: Set<Int>?,
    ): File = withContext(Dispatchers.Default) {
        val book = EpubReader.read(this@SynthesisService, uri)
        val chapters = book.chapters.filter { pickedChapters == null || it.index in pickedChapters }
        val totalChars = chapters.sumOf { it.charCount }.coerceAtLeast(1)
        Progress.start(book.title, totalChars)

        val outDir = File(filesDir, "out").apply { mkdirs() }
        val outFile = File(outDir, "${book.title.sanitize()}.m4a")

        // Pick the user's preferred engine if provisioned, else fall back to the
        // first available provisioned engine (any order). Last resort: error.
        val factory = Engines.factoryFor(quality)?.takeIf { it.isProvisioned(this@SynthesisService) }
            ?: Engines.factories.firstOrNull { it.isProvisioned(this@SynthesisService) }
            ?: error("No TTS engine provisioned. Run setup first.")
        factory.load(this@SynthesisService).use { engine ->
            val preview = if (streamLive) AudioPreview(engine.sampleRate) else null
            // Track chapter start times for .m4b chapter markers.
            val chapterTimes = mutableListOf<ChapterInjector.Chapter>()
            var chapterStartUs = 0L
            try {
                M4aMuxer(outFile, sampleRate = engine.sampleRate).use { muxer ->
                    var processed = 0
                    for (chapter in chapters) {
                        chapterTimes += ChapterInjector.Chapter(chapter.title, chapterStartUs)
                        for (chunk in TextSplitter.split(TextNormalizer.normalize(chapter.text))) {
                            ensureActive()
                            val audio = engine.synthesize(chunk, voice, speed)
                            preview?.write(audio)            // blocks until played → natural backpressure
                            muxer.writeSamples(audio)
                            processed += chunk.length
                            // Track chapter duration from samples written
                            chapterStartUs += audio.size * 1_000_000L / engine.sampleRate
                            Progress.tick(processed, totalChars)
                            updateNotification(
                                "${book.title} — ${100 * processed / totalChars}%",
                                streamLive,
                            )
                        }
                    }
                }
            } finally {
                preview?.close()
            }

            // Inject chapter markers and rename to .m4b
            val m4bFile = ChapterInjector.inject(outFile, chapterTimes)
            m4bFile ?: outFile  // Fall back to .m4a if injection failed
        }
    }

    // ---------------------------------------------------------------------
    // Wake lock — partial lock keeps the CPU running while the screen is off.
    // mediaPlayback foreground type keeps AudioTrack alive across Doze.
    // ---------------------------------------------------------------------
    private fun acquireWakeLock() {
        if (wakeLock != null) return
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "booki:synth").apply {
            setReferenceCounted(false)
            acquire(2 * 60 * 60 * 1000L)  // 2-hour safety cap
        }
    }
    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    // ---------------------------------------------------------------------
    // Notification
    // ---------------------------------------------------------------------
    private fun buildNotification(text: String, streamLive: Boolean): Notification {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            mgr.createNotificationChannel(
                NotificationChannel(CHANNEL, "Audiobook synthesis", NotificationManager.IMPORTANCE_LOW))
        }
        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, SynthesisService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle(if (streamLive) "Booki — playing" else "Booki — generating")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopIntent)
            .build()
    }

    private fun updateNotification(text: String, streamLive: Boolean) {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        mgr.notify(NOTIF_ID, buildNotification(text, streamLive))
    }

    override fun onDestroy() {
        releaseWakeLock()
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_URI = "uri"
        const val EXTRA_VOICE = "voice"
        const val EXTRA_SPEED = "speed"
        const val EXTRA_CHAPTERS = "chapters"
        const val EXTRA_STREAM_LIVE = "stream_live"
        const val ACTION_STOP = "dev.booki.action.STOP"
        private const val CHANNEL = "booki.synth"
        private const val NOTIF_ID = 42
    }
}

private fun String.sanitize(): String =
    replace(Regex("[^A-Za-z0-9 _.-]"), "_").take(80).ifBlank { "audiobook" }
