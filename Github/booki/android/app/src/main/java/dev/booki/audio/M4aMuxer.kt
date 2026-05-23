package dev.booki.audio

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * Encodes a stream of float PCM samples to AAC inside an MP4/M4A container using
 * MediaCodec + MediaMuxer (no ffmpeg dependency). Chapter markers aren't written
 * here — extend with an MP4 `chpl`/`chap` post-processor if you want true .m4b.
 */
class M4aMuxer(
    output: File,
    private val sampleRate: Int = 24_000,
    bitrate: Int = 64_000,
) : AutoCloseable {

    private val codec: MediaCodec
    private val muxer: MediaMuxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private var trackIndex: Int = -1
    private var muxerStarted = false
    private var presentationUs: Long = 0
    private val bufferInfo = MediaCodec.BufferInfo()

    init {
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16384)
        }
        codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            start()
        }
    }

    fun writeSamples(samples: FloatArray) {
        val pcm = ByteBuffer.allocate(samples.size * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (s in samples) {
            val v = (s.coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
            pcm.putShort(v)
        }
        pcm.flip()
        feed(pcm, endOfStream = false)
    }

    private fun feed(input: ByteBuffer, endOfStream: Boolean) {
        while (input.hasRemaining() || endOfStream) {
            val inIdx = codec.dequeueInputBuffer(10_000)
            if (inIdx >= 0) {
                val buf = codec.getInputBuffer(inIdx)!!
                buf.clear()
                val toCopy = minOf(buf.capacity(), input.remaining())
                if (toCopy > 0) {
                    val slice = input.duplicate()
                    slice.limit(slice.position() + toCopy)
                    buf.put(slice)
                    input.position(input.position() + toCopy)
                }
                val flags = if (endOfStream && !input.hasRemaining()) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0
                codec.queueInputBuffer(inIdx, 0, toCopy, presentationUs, flags)
                presentationUs += (toCopy / 2L) * 1_000_000L / sampleRate
                if (flags != 0) break
            }
            drain(endOfStream)
            if (!input.hasRemaining() && !endOfStream) return
        }
    }

    private fun drain(endOfStream: Boolean) {
        while (true) {
            val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return else continue
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    check(!muxerStarted) { "format changed twice" }
                    trackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outIdx >= 0 -> {
                    val out = codec.getOutputBuffer(outIdx)!!
                    if (bufferInfo.size > 0 && muxerStarted &&
                        bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        out.position(bufferInfo.offset)
                        out.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, out, bufferInfo)
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    override fun close() {
        try {
            feed(ByteBuffer.allocate(0), endOfStream = true)
        } catch (_: Exception) {
            // Codec may already be stopped — drain best-effort only.
        }
        runCatching { codec.stop() }
        codec.release()
        if (muxerStarted) { muxer.stop() }
        muxer.release()
    }
}
