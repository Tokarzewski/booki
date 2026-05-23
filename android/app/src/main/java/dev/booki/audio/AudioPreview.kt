package dev.booki.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack

/** Streams float PCM samples through an [AudioTrack] for live preview. */
class AudioPreview(sampleRate: Int = 24_000) : AutoCloseable {
    private val track: AudioTrack

    init {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(maxOf(minBuf, sampleRate * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track.play()
    }

    fun write(samples: FloatArray) {
        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
    }

    override fun close() {
        runCatching { track.stop() }
        track.release()
    }
}
