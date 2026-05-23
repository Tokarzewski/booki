package dev.booki.epub

/**
 * Cleans EPUB text before TTS sees it. Goals:
 *   - Expand common abbreviations and acronyms so they're pronounced as words.
 *   - Spell out numbers (cardinals + years).
 *   - Strip footnote / page markers and license boilerplate.
 *   - Normalize unicode punctuation that throws off espeak-ng pacing.
 *
 * Engine-agnostic — every backend benefits. Pure functions with no allocation
 * surprises, so safe to call per chunk in the synth loop.
 */
object TextNormalizer {

    fun normalize(text: String): String {
        var t = text
        t = stripBoilerplate(t)
        t = normalizePunctuation(t)
        t = stripFootnoteMarkers(t)
        t = expandAcronyms(t)
        t = expandAbbreviations(t)
        t = expandNumbers(t)
        t = collapseWhitespace(t)
        return t
    }

    // ---------------------------------------------------------------------
    // Boilerplate (license headers etc.)
    // ---------------------------------------------------------------------
    private val BOILERPLATE_PATTERNS = listOf(
        Regex("""\*{3}\s*START OF (?:THIS |THE )?PROJECT GUTENBERG[^*]*\*{3}""", RegexOption.IGNORE_CASE),
        Regex("""\*{3}\s*END OF (?:THIS |THE )?PROJECT GUTENBERG[^*]*\*{3}""", RegexOption.IGNORE_CASE),
        Regex("""This eBook is for the use of anyone anywhere.*?(?=\n\n)""",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)),
    )

    private fun stripBoilerplate(text: String): String {
        var t = text
        for (re in BOILERPLATE_PATTERNS) t = re.replace(t, "")
        return t
    }

    // ---------------------------------------------------------------------
    // Punctuation normalization
    // ---------------------------------------------------------------------
    private val PUNCT_MAP = mapOf(
        '‘' to '\'', '’' to '\'', // smart single quotes → '
        '“' to '"',  '”' to '"',  // smart double quotes → "
        '–' to '-',  '—' to '-',  // en/em dash → -
        '…' to ' ', // ellipsis → space
        ' ' to ' ', // nbsp → space
    )

    private fun normalizePunctuation(text: String): String =
        text.map { PUNCT_MAP[it] ?: it }.joinToString("")

    // ---------------------------------------------------------------------
    // Footnote / page markers
    // ---------------------------------------------------------------------
    private val FOOTNOTE_RE = Regex("""\[\s*\d+\s*\]""")             // [1], [12]
    private val FN_PREFIX_RE = Regex("""\bfn\.\s*\d+\b""", RegexOption.IGNORE_CASE)
    private val PAGE_REF_RE = Regex("""\bp\.\s*\d+\b""", RegexOption.IGNORE_CASE)
    // Superscript digits (²³⁴ etc.) used as inline footnote markers
    private val SUP_DIGITS_RE = Regex("""[²³¹⁰-⁹]+""")

    private fun stripFootnoteMarkers(text: String): String = text
        .replace(FOOTNOTE_RE, "")
        .replace(FN_PREFIX_RE, "")
        .replace(PAGE_REF_RE, "")
        .replace(SUP_DIGITS_RE, "")

    // ---------------------------------------------------------------------
    // Acronyms — dotted (U.K., U.S.A.) → spelled out, preserving sentence end
    // ---------------------------------------------------------------------
    // Matches "U.S.A." / "U.K." etc. and captures whether the final dot is
    // followed by a space + capital (acronym-internal) vs end-of-sentence (keep
    // the period). We always strip the inner dots and only the final dot when
    // it's clearly internal to the acronym.
    private val DOTTED_ACRONYM_RE = Regex("""\b(?:[A-Z]\.){2,}""")

    private fun expandAcronyms(text: String): String =
        DOTTED_ACRONYM_RE.replace(text) { match ->
            val spelled = match.value.split('.').filter { it.isNotEmpty() }.joinToString(" ")
            // Look ahead past whitespace: if the next visible character is a
            // lowercase letter the trailing dot was an acronym terminator
            // mid-sentence, so drop it. Otherwise (end of text, uppercase
            // letter, or punctuation) it was also a sentence terminator —
            // preserve the period.
            var i = match.range.last + 1
            while (i < text.length && (text[i] == ' ' || text[i] == '\t')) i++
            val nextChar = text.getOrNull(i)
            val midSentence = nextChar != null && nextChar.isLowerCase()
            if (midSentence) spelled else "$spelled."
        }

    // ---------------------------------------------------------------------
    // Abbreviations
    // ---------------------------------------------------------------------
    private val ABBREVIATIONS = mapOf(
        "Mr." to "Mister", "Mrs." to "Misses", "Ms." to "Miz", "Dr." to "Doctor",
        "Prof." to "Professor", "St." to "Saint", "Mt." to "Mount",
        "Jr." to "Junior", "Sr." to "Senior",
        "vs." to "versus", "etc." to "et cetera", "i.e." to "that is",
        "e.g." to "for example", "cf." to "compare", "viz." to "namely",
        "Inc." to "Incorporated", "Ltd." to "Limited", "Co." to "Company",
        "No." to "Number", "Ave." to "Avenue", "Blvd." to "Boulevard",
        "Rd." to "Road", "Hwy." to "Highway",
    )

    private val ABBREV_RE = Regex(
        ABBREVIATIONS.keys.joinToString("|") { Regex.escape(it) } + """(?=\s|$)"""
    )

    private fun expandAbbreviations(text: String): String =
        ABBREV_RE.replace(text) { ABBREVIATIONS.getValue(it.value) }

    // ---------------------------------------------------------------------
    // Numbers
    //
    // Years are only spoken with year-pair pronunciation when the context
    // makes it unambiguous ("in 1923", "circa 2026") — bare digits like
    // "1234" stay as cardinals because they're usually not years.
    // ---------------------------------------------------------------------
    private val YEAR_CONTEXT_RE = Regex(
        """(?i)(?<=\b(?:in|of|from|around|circa|year|by|since|until|before|after|was)\s)\d{4}\b"""
    )
    // 4-digit numbers starting with 2 are almost always modern years in
    // literary text — pronounce them as years even without leading context.
    private val MODERN_YEAR_RE = Regex("""\b20\d{2}\b""")
    private val NUMBER_RE = Regex("""(?<![A-Za-z])-?\d{1,6}(?:,\d{3})*(?:\.\d+)?(?!\w)""")

    private fun expandNumbers(text: String): String {
        var t = YEAR_CONTEXT_RE.replace(text) { match ->
            val y = match.value.toIntOrNull() ?: return@replace match.value
            if (y in 1100..2099) NumberToWords.year(y) else match.value
        }
        t = MODERN_YEAR_RE.replace(t) { match ->
            val y = match.value.toIntOrNull() ?: return@replace match.value
            if (y in 2000..2099) NumberToWords.year(y) else match.value
        }
        t = NUMBER_RE.replace(t) { match ->
            val raw = match.value.replace(",", "")
            val negative = raw.startsWith('-')
            val abs = if (negative) raw.substring(1) else raw
            val n = abs.toDoubleOrNull() ?: return@replace match.value
            val spoken = if (n.rem(1.0) != 0.0) {
                val parts = abs.split('.')
                val whole = NumberToWords.cardinal(parts[0].toLong())
                val frac = parts[1].toCharArray()
                    .joinToString(" ") { NumberToWords.cardinal(Character.digit(it, 10).toLong()) }
                "$whole point $frac"
            } else NumberToWords.cardinal(n.toLong())
            if (negative) "minus $spoken" else spoken
        }
        return t
    }

    // ---------------------------------------------------------------------
    // Whitespace cleanup — final pass
    // ---------------------------------------------------------------------
    private val MULTI_SPACE_RE = Regex("""[ \t]+""")
    private val MULTI_NL_RE = Regex("""\n{3,}""")

    private fun collapseWhitespace(text: String): String = text
        .replace(MULTI_SPACE_RE, " ")
        .replace(MULTI_NL_RE, "\n\n")
        .trim()
}
