package dev.booki.tts

import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/**
 * Minimal NumPy `.npz` reader for the small subset Kokoro voices use:
 *   - the archive contains one `<voice>.npy` file per voice
 *   - each array is float32, little-endian, 3-D shape (N, 1, 256)
 *
 * NPY format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 */
object NpzReader {

    data class Array3D(val shape: IntArray, val data: FloatArray)

    fun read(file: File): Map<String, Array3D> {
        val out = LinkedHashMap<String, Array3D>()
        ZipFile(file).use { zip ->
            val entries = zip.entries().toList().filter { it.name.endsWith(".npy") }
            for (entry in entries) {
                val name = entry.name.removeSuffix(".npy")
                zip.getInputStream(entry).use { input ->
                    out[name] = readNpy(DataInputStream(input.buffered()))
                }
            }
        }
        return out
    }

    private fun readNpy(input: DataInputStream): Array3D {
        // Magic: \x93NUMPY
        val magic = ByteArray(6).also { input.readFully(it) }
        check(magic.toString(Charsets.ISO_8859_1) == "NUMPY") { "Bad NPY magic" }
        val major = input.readByte().toInt() and 0xFF
        val minor = input.readByte().toInt() and 0xFF
        val headerLen = if (major >= 2) {
            java.lang.Integer.reverseBytes(input.readInt())
        } else {
            java.lang.Short.reverseBytes(input.readShort()).toInt() and 0xFFFF
        }
        val header = ByteArray(headerLen).also { input.readFully(it) }.toString(Charsets.US_ASCII)
        // Parse the small dict literal: {'descr': '<f4', 'fortran_order': False, 'shape': (N, 1, 256), }
        val shape = SHAPE_RE.find(header)?.groupValues?.get(1)
            ?.split(',')?.mapNotNull { it.trim().toIntOrNull() }
            ?: error("NPY shape not found in: $header")
        check(header.contains("'descr': '<f4'")) { "Only float32 little-endian supported. Got: $header" }
        check(header.contains("'fortran_order': False")) { "Only C-order arrays supported" }
        val total = shape.fold(1, Int::times)
        val bytes = ByteArray(total * 4).also { input.readFully(it) }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(total) { buf.float }
        return Array3D(shape.toIntArray(), floats)
    }

    private val SHAPE_RE = Regex("""'shape':\s*\(([^)]*)\)""")
    private val Major: Int = 1
}
