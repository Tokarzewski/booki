package dev.booki.epub

/**
 * Greedy sentence-aware splitter. Keeps chunks under [maxLen] characters and
 * never splits mid-sentence unless a single sentence already exceeds the cap,
 * in which case it falls back to hard chunking.
 */
object TextSplitter {
    private val SENT = Regex("(?<=[.!?。！？])\\s+")

    fun split(text: String, maxLen: Int = 500): List<String> {
        val out = ArrayList<String>()
        val buf = StringBuilder()
        for (sentence in text.split(SENT)) {
            val s = sentence.trim()
            if (s.isEmpty()) continue
            if (s.length > maxLen) {
                if (buf.isNotEmpty()) { out += buf.toString(); buf.setLength(0) }
                s.chunked(maxLen).forEach { out += it }
                continue
            }
            if (buf.length + s.length + 1 > maxLen) {
                out += buf.toString(); buf.setLength(0)
            }
            if (buf.isNotEmpty()) buf.append(' ')
            buf.append(s)
        }
        if (buf.isNotEmpty()) out += buf.toString()
        return out
    }
}
