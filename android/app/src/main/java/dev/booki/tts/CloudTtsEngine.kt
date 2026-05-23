package dev.booki.tts

import android.annotation.SuppressLint
import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import dev.booki.tts.CloudSettings.cloudApiKey
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI

/**
 * Cloud-based TTS via user-supplied API keys. Two providers supported:
 *   - ElevenLabs v2 Turbo — AAA broadcast quality
 *   - Fish S2 Pro — competitive quality, lower cost
 *
 * The engine streams text to the cloud API and decodes the returned audio
 * into float PCM samples. No on-device model needed.
 */
class CloudTtsEngine(
    override val name: String,
    private val provider: CloudSettings.Provider,
    private val apiKey: String,
) : SpeechEngine {

    override val sampleRate: Int = 24_000  // target for both providers

    /**
     * Synthesize text via cloud API.
     * Runs on the calling thread — the caller (SynthesisService) already
     * dispatches to Dispatchers.Default.
     */
    override fun synthesize(text: String, voiceId: String, speed: Float): FloatArray {
        val audioBytes = when (provider) {
            CloudSettings.Provider.ELEVENLABS -> elevenLabsTts(text, voiceId)
            CloudSettings.Provider.FISH -> fishTts(text, voiceId)
        }
        return decodeAudio(audioBytes)
    }

    override fun close() { /* no resources to release */ }

    // -----------------------------------------------------------------------
    // ElevenLabs API
    // -----------------------------------------------------------------------

    private fun elevenLabsTts(text: String, voiceId: String): ByteArray {
        val url = "https://api.elevenlabs.io/v1/text-to-speech/$voiceId?output_format=pcm_24000"
        val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 30_000
            readTimeout = 120_000
            setRequestProperty("xi-api-key", apiKey)
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("User-Agent", "Booki/0.6 (+https://github.com/Tokarzewski/booki)")
            doOutput = true
        }
        val body = JSONObject().apply {
            put("text", text)
            put("model_id", "eleven_multilingual_v2")
        }
        conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }

        check(conn.responseCode in 200..299) {
            val err = conn.errorStream?.readBytes()?.toString(Charsets.UTF_8).orEmpty()
            "ElevenLabs HTTP ${conn.responseCode}: $err"
        }
        return conn.inputStream.use { it.readBytes() }
    }

    // -----------------------------------------------------------------------
    // Fish.audio API
    // -----------------------------------------------------------------------

    private fun fishTts(text: String, voiceId: String): ByteArray {
        val url = "https://api.fish.audio/v1/tts"
        val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 30_000
            readTimeout = 120_000
            setRequestProperty("Authorization", "Bearer $apiKey")
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("User-Agent", "Booki/0.6 (+https://github.com/Tokarzewski/booki)")
            doOutput = true
        }
        val body = JSONObject().apply {
            put("text", text)
            put("voice_id", voiceId)
            put("model", "fish-speech-1.5")
        }
        conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }

        check(conn.responseCode in 200..299) {
            val err = conn.errorStream?.readBytes()?.toString(Charsets.UTF_8).orEmpty()
            "Fish.audio HTTP ${conn.responseCode}: $err"
        }
        return conn.inputStream.use { it.readBytes() }
    }

    // -----------------------------------------------------------------------
    // Audio decoding
    // -----------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    private fun decodeAudio(raw: ByteArray): FloatArray {
        // ElevenLabs PCM output is already 24 kHz 16-bit PCM LE.
        if (provider == CloudSettings.Provider.ELEVENLABS) {
            return pcm16ToF32(raw, sampleRate)
        }

        // Fish returns MP3 — decode via MediaCodec.
        return decodeMp3(raw)
    }

    private fun pcm16ToF32(bytes: ByteArray, sampleRate: Int): FloatArray {
        val samples = ByteArrayBufferInputStream(bytes)
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val lo = samples.read().toUInt()
            val hi = samples.read().toUInt()
            val s = ((hi.toInt() shl 8) or lo.toInt()).toShort().toInt()
            out[i] = s / 32768f
        }
        return out
    }

    private fun decodeMp3(mp3: ByteArray): FloatArray {
        val extractor = MediaExtractor()
        val tempFile = java.io.File.createTempFile("booki-cloud", ".mp3")
        tempFile.writeBytes(mp3)

        return try {
            extractor.setDataSource(tempFile.absolutePath)
            val trackIdx = (0 until extractor.trackCount).find { i ->
                val fmt = extractor.getTrackFormat(i)
                fmt.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("No audio track in MP3")

            extractor.selectTrack(trackIdx)
            val fmt = extractor.getTrackFormat(trackIdx)
            val mime = fmt.getString(MediaFormat.KEY_MIME)!!
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(fmt, null, null, 0)
            codec.start()

            val bufferInfo = MediaCodec.BufferInfo()
            val outSamples = mutableListOf<Float>()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                // Feed input
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        buf.clear()
                        val toRead = minOf(buf.remaining(), 65536)
                        @Suppress("NewApi") val available = extractor.sampleSize.toInt().coerceAtMost(toRead)
                        buf.put(mp3, 0, available)
                        val flags = if (available == 0) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0
                        codec.queueInputBuffer(inIdx, 0, available, 0L, flags)
                        if (flags != 0) inputDone = true
                    }
                }

                // Drain output
                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                if (outIdx >= 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.position(bufferInfo.offset)
                    outBuf.limit(bufferInfo.offset + bufferInfo.size)
                    val pcm = ByteArray(bufferInfo.size)
                    @Suppress("NewApi") outBuf.get(pcm)
                    outSamples.addAll(pcm16ToF32List(pcm))
                    codec.releaseOutputBuffer(outIdx, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
            }

            codec.stop()
            codec.release()
            outSamples.toFloatArray()
        } finally {
            extractor.release()
            tempFile.delete()
        }
    }

    private fun pcm16ToF32List(bytes: ByteArray): List<Float> {
        val out = mutableListOf<Float>()
        var i = 0
        while (i + 1 < bytes.size) {
            val lo = bytes[i].toInt() and 0xFF
            val hi = bytes[i + 1].toInt() and 0xFF
            val s = ((hi shl 8) or lo).toShort().toInt()
            out += s / 32768f
            i += 2
        }
        return out
    }

    /** Simple InputStream wrapper over a ByteArray. */
    private class ByteArrayBufferInputStream(private val bytes: ByteArray) {
        private var pos = 0
        fun read(): Int = if (pos < bytes.size) bytes[pos++].toInt() and 0xFF else -1
    }

    // -----------------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------------

    abstract class CloudFactory(
        override val id: SpeechEngine.Quality,
        override val displayName: String,
        private val provider: CloudSettings.Provider,
    ) : SpeechEngine.Factory {
        override val downloadSizeMb = 0
        override val ramMb = 50  // minimal — just HTTP + codec
        override val isExperimental = true

        override fun isProvisioned(context: Context) = CloudSettings.hasKey(context)
        override fun isSupportedOn(context: Context) = true  // cloud = no RAM gate

        override fun load(context: Context): SpeechEngine {
            val key = context.cloudApiKey
            check(key.isNotBlank()) { "No API key configured for $displayName" }
            return CloudTtsEngine(displayName, provider, key)
        }
    }

    object ElevenLabsFactory : CloudFactory(
        id = SpeechEngine.Quality.CLOUD_ELEVENLABS,
        displayName = "ElevenLabs (cloud)",
        provider = CloudSettings.Provider.ELEVENLABS,
    )

    object FishFactory : CloudFactory(
        id = SpeechEngine.Quality.CLOUD_FISH,
        displayName = "Fish.audio (cloud)",
        provider = CloudSettings.Provider.FISH,
    )
}
