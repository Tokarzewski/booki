package dev.booki.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WavWriter {
    /** Writes a 16-bit PCM mono WAV from [-1, 1] float samples. */
    fun write(file: File, samples: FloatArray, sampleRate: Int) {
        val byteRate = sampleRate * 2
        val dataSize = samples.size * 2
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray())
            header.putInt(36 + dataSize)
            header.put("WAVE".toByteArray())
            header.put("fmt ".toByteArray())
            header.putInt(16)             // PCM chunk size
            header.putShort(1)            // format = PCM
            header.putShort(1)            // mono
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(2)            // block align
            header.putShort(16)           // bits per sample
            header.put("data".toByteArray())
            header.putInt(dataSize)
            raf.write(header.array())

            val pcm = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
            for (s in samples) {
                val clamped = (s.coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
                pcm.putShort(clamped)
            }
            raf.write(pcm.array())
        }
    }
}
