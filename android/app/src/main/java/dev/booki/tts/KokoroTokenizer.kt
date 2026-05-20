package dev.booki.tts

import android.content.Context

/**
 * Builds a phoneme/character → token id map from the bundled `vocab.txt` asset.
 * Each character in the file is one symbol; its index becomes its token id.
 * Special tokens `<bos>` / `<eos>` / `<pad>` are reserved beyond the alphabet.
 *
 * NOTE: For high-quality Kokoro speech, text must be phonemized via espeak-ng
 * before tokenization. This per-character fallback exists so the pipeline runs
 * end-to-end without the NDK build. Quality is poor — see
 * [README](android/README.md) for the phonemizer integration path.
 */
class KokoroTokenizer(private val vocab: Map<String, Int>) {

    private val unk = vocab["$"] ?: 0
    private val pad = vocab["<pad>"] ?: (vocab.values.max() + 1)
    private val bos = vocab["<bos>"] ?: (pad + 1)
    private val eos = vocab["<eos>"] ?: (pad + 2)

    fun tokenize(text: String): LongArray {
        val out = ArrayList<Long>(text.length + 2)
        out += bos.toLong()
        for (ch in text) out += (vocab[ch.toString()] ?: unk).toLong()
        out += eos.toLong()
        return out.toLongArray()
    }

    companion object {
        fun load(context: Context): KokoroTokenizer {
            val bytes = context.assets.open("kokoro/vocab.txt").use { it.readBytes() }
            val text = bytes.toString(Charsets.UTF_8).trim('\n', '\r', ' ')
            val map = HashMap<String, Int>(text.length)
            text.forEachIndexed { idx, ch -> map.putIfAbsent(ch.toString(), idx) }
            return KokoroTokenizer(map)
        }
    }
}
