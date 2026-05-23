package dev.booki.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Appends a Nero `chpl` chapter atom to an MP4/M4A file by inserting it
 * at the end of the `moov` atom and shifting the trailing `mdat` atom.
 *
 * Output is renamed from `.m4a` to `.m4b` so audiobook players recognise it.
 */
object ChapterInjector {

    data class Chapter(val title: String, val startTimeUs: Long)

    /**
     * Inject chapters into an M4A file and rename it to .m4b.
     *
     * @param m4aFile  the source .m4a file from M4aMuxer
     * @param chapters list of (title, start_time_in_microseconds)
     * @return the resulting .m4b file, or null on failure
     */
    fun inject(m4aFile: File, chapters: List<Chapter>): File? {
        if (chapters.isEmpty()) {
            // No chapters — just rename to .m4b if desired
            return null
        }

        return try {
            val m4bFile = File(m4aFile.parent, m4aFile.nameWithoutExtension + ".m4b")

            RandomAccessFile(m4aFile, "r").use { src ->
                val fileSize = src.length()

                // Build the udta/chpl payload
                val chplBytes = buildChpl(chapters)
                val udtaSize = chplBytes.size + 8L  // udta box header

                // Find moov atom
                val moov = findBox(src, "moov", 0) ?: return@use null
                val mdat = findBox(src, "mdat", 0) ?: return@use null

                // We'll write a new file: everything up to mdat,
                // then expanded moov (with udta/chpl appended),
                // then mdat with updated size.
                RandomAccessFile(m4bFile, "rw").use { dst ->
                    // Copy ftyp + everything before mdat
                    dst.channel.transferFrom(src.channel, 0, mdat.start)

                    // Write expanded moov (original + udta/chpl)
                    val newMoovSize = moov.size + udtaSize
                    val moovSizeBuf = ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN)
                    moovSizeBuf.putInt(newMoovSize.toInt())
                    moovSizeBuf.flip()
                    dst.channel.write(moovSizeBuf)

                    // Copy original moov body (skip size header we already wrote)
                    src.seek(moov.start + 4)
                    dst.channel.transferFrom(src.channel, dst.channel.position(), moov.size - 4)

                    // Write udta/chpl box
                    val udtaBuf = ByteBuffer.allocate(udtaSize.toInt()).order(ByteOrder.BIG_ENDIAN)
                    udtaBuf.putInt(udtaSize.toInt())
                    udtaBuf.put("udta".toByteArray(Charsets.UTF_8))
                    udtaBuf.put(chplBytes)
                    udtaBuf.flip()
                    dst.channel.write(udtaBuf)

                    // Write mdat with updated size
                    val newMdatSize = (fileSize - mdat.start) + udtaSize
                    val mdatSizeBuf = ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN)
                    mdatSizeBuf.putInt(newMdatSize.toInt())
                    mdatSizeBuf.flip()
                    dst.channel.write(mdatSizeBuf)

                    // Copy mdat body
                    src.seek(mdat.start + 4)
                    dst.channel.transferFrom(src.channel, dst.channel.position(), fileSize - mdat.start - 4)
                }
            }

            m4aFile.delete()
            m4bFile
        } catch (e: Exception) {
            android.util.Log.w("ChapterInjector", "Failed to inject chapters", e)
            m4aFile.delete()  // Clean up partial output
            null
        }
    }

    private fun buildChpl(chapters: List<Chapter>): ByteArray {
        var payloadSize = 4  // version + flags + reserved + count
        for (ch in chapters) {
            payloadSize += 8 + 2 + ch.title.toByteArray(Charsets.UTF_8).size
        }

        val buf = ByteBuffer.allocate(payloadSize + 8).order(ByteOrder.BIG_ENDIAN)
        buf.putInt(payloadSize + 8)  // chpl box size
        buf.put("chpl".toByteArray(Charsets.UTF_8))
        buf.put(0)  // version
        buf.put(0)  // flags
        buf.put(0)  // reserved
        buf.put(chapters.size.toByte())

        for (ch in chapters) {
            val titleBytes = ch.title.toByteArray(Charsets.UTF_8)
            buf.putLong(ch.startTimeUs * 10)  // 100-nanosecond units
            buf.putShort(titleBytes.size.toShort())
            buf.put(titleBytes)
        }

        return buf.array().copyOf(buf.position())
    }

    private data class Box(val start: Long, val size: Long, val name: String)

    private fun findBox(raf: RandomAccessFile, name: String, from: Long): Box? {
        val header = ByteArray(8)
        raf.seek(from)
        while (raf.read(header) == 8) {
            val buf = ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
            val size = buf.int.toLong()
            val type = String(header, 4, 4, Charsets.US_ASCII)
            if (type == name) return Box(from, size, name)
            if (size < 8 || size > raf.length()) return null
            raf.seek(from + size)
        }
        return null
    }
}
