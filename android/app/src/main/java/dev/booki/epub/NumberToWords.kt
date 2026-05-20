package dev.booki.epub

/**
 * Cardinal numbers and years → English words. Covers the range that shows up
 * in books (chapter numbers, years, prices). Capped at 9_999_999.
 */
internal object NumberToWords {

    private val ONES = arrayOf(
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    )
    private val TENS = arrayOf(
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    )

    fun cardinal(n: Long): String {
        if (n == 0L) return "zero"
        val neg = n < 0
        val abs = if (neg) -n else n
        val s = chunk(abs)
        return if (neg) "minus $s" else s
    }

    /** Years 1100–2099 spoken in pairs ("nineteen twenty-three"). */
    fun year(y: Int): String {
        if (y in 2000..2009) return "two thousand" + if (y == 2000) "" else " " + cardinal(y % 10L)
        if (y in 2010..2099) return "twenty ${chunkUnder100(y % 100)}"
        if (y in 1100..1999) {
            val hi = y / 100
            val lo = y % 100
            return if (lo == 0) "${chunkUnder100(hi)} hundred"
            else "${chunkUnder100(hi)} ${if (lo < 10) "oh " + ONES[lo] else chunkUnder100(lo)}"
        }
        return cardinal(y.toLong())
    }

    private fun chunk(n: Long): String {
        if (n < 100) return chunkUnder100(n.toInt())
        if (n < 1000) {
            val h = (n / 100).toInt()
            val r = (n % 100).toInt()
            return ONES[h] + " hundred" + if (r == 0) "" else " " + chunkUnder100(r)
        }
        if (n < 1_000_000) {
            val thousands = (n / 1000).toInt()
            val r = (n % 1000).toInt()
            val s = chunk(thousands.toLong()) + " thousand"
            return if (r == 0) s else "$s ${chunk(r.toLong())}"
        }
        val millions = (n / 1_000_000).toInt()
        val r = (n % 1_000_000).toInt()
        val s = chunk(millions.toLong()) + " million"
        return if (r == 0) s else "$s ${chunk(r.toLong())}"
    }

    private fun chunkUnder100(n: Int): String {
        if (n < 20) return ONES[n]
        val tens = n / 10
        val ones = n % 10
        return if (ones == 0) TENS[tens] else "${TENS[tens]}-${ONES[ones]}"
    }
}
