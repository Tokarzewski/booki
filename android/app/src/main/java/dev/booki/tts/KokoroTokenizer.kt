package dev.booki.tts

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * Minimal phoneme tokenizer for Kokoro. The reference pipeline uses espeak-ng to
 * convert graphemes -> IPA phonemes, then maps phonemes -> ids via a vocab file.
 *
 * espeak-ng isn't bundled here. Two viable paths:
 *  - ship a precomputed `espeakng-android` NDK build and call it via JNI
 *  - or replace this with an on-device G2P (e.g. piper-phonemize, espeak-ng-jni)
 *
 * For now we fall back to a per-character mapping against the vocab so the
 * pipeline is end-to-end runnable for ASCII text. Swap in real phonemization
 * before shipping.
 */
class KokoroTokenizer(private val vocab: Map<String, Long>) {

    fun tokenize(text: String): LongArray {
        val unk = vocab["<unk>"] ?: 0L
        val bos = vocab["<bos>"]
        val eos = vocab["<eos>"]
        val out = ArrayList<Long>(text.length + 2)
        bos?.let { out += it }
        for (ch in text) {
            out += vocab[ch.toString()] ?: unk
        }
        eos?.let { out += it }
        return out.toLongArray()
    }

    companion object {
        fun load(context: Context): KokoroTokenizer {
            val file = File(context.filesDir, "kokoro/vocab.json")
            check(file.exists()) { "vocab.json missing under app files/kokoro/." }
            val json = JSONObject(file.readText())
            val map = HashMap<String, Long>(json.length())
            for (k in json.keys()) map[k] = json.getLong(k)
            return KokoroTokenizer(map)
        }
    }
}
