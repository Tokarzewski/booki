package dev.booki.sfx

/**
 * Chapter mood for generative ambient SFX (issue #8).
 *
 * [prompt] is the MOSS-TTS text prompt used to generate the ambient bed for
 * the chapter; [stingPrompt] generates the short section-transition sting.
 */
enum class Mood(val prompt: String, val stingPrompt: String) {
    STORM(
        "Distant rolling thunder, heavy rain on a window, occasional wind gusts",
        "A single low thunder rumble fading into rain",
    ),
    FOREST(
        "Quiet forest ambience, birdsong, leaves rustling in a light breeze",
        "A short swell of birdsong and rustling leaves",
    ),
    SEA(
        "Waves breaking on a shore, distant gulls, creaking timber",
        "One wave washing onto sand with receding foam",
    ),
    INDOOR(
        "Warm room tone, a crackling fireplace, faint clock ticking",
        "A fire crackle swell with a soft room echo",
    ),
    URBAN(
        "Distant city traffic, occasional footsteps on pavement, muffled voices",
        "A short passing-car whoosh with city murmur",
    ),
    SUSPENSE(
        "Low ominous drone, sparse creaks, barely audible heartbeat",
        "A rising dissonant drone cut off abruptly",
    ),
    BATTLE(
        "Distant clash of steel, shouting crowds, war drums far away",
        "A sword ring and a single war-drum hit",
    ),
    CALM(
        "Soft neutral room tone with very gentle air movement",
        "A gentle warm swell fading to silence",
    ),
}

/**
 * LLM-free mood classifier: a weighted keyword bag over the chapter title and
 * the first [WORD_WINDOW] words of the chapter body. Title hits count [TITLE_WEIGHT]×
 * because titles are short and dense ("The Tempest" should beat one stray
 * "rain" in the body). Falls back to [Mood.CALM] when nothing scores.
 */
object MoodClassifier {

    const val WORD_WINDOW = 200
    const val TITLE_WEIGHT = 3

    private val keywords: Map<Mood, Set<String>> = mapOf(
        Mood.STORM to setOf(
            "storm", "storms", "thunder", "lightning", "rain", "raining", "downpour",
            "tempest", "gale", "hail", "drizzle", "squall", "hurricane", "monsoon",
        ),
        Mood.FOREST to setOf(
            "forest", "forests", "wood", "woods", "trees", "grove", "glade", "jungle",
            "birds", "birdsong", "leaves", "moss", "thicket", "wilderness", "meadow",
        ),
        Mood.SEA to setOf(
            "sea", "ocean", "waves", "shore", "ship", "ships", "sail", "sails", "sailor",
            "harbour", "harbor", "tide", "gulls", "deck", "voyage", "island",
        ),
        Mood.INDOOR to setOf(
            "room", "hall", "fireplace", "hearth", "candle", "candles", "kitchen",
            "library", "parlour", "parlor", "chamber", "cellar", "attic", "inn", "tavern",
        ),
        Mood.URBAN to setOf(
            "city", "street", "streets", "traffic", "crowd", "crowds", "market",
            "alley", "station", "train", "cars", "pavement", "sidewalk", "subway",
        ),
        Mood.SUSPENSE to setOf(
            "dark", "darkness", "shadow", "shadows", "whisper", "whispers", "fear",
            "afraid", "silence", "silent", "creak", "creaking", "dread", "ghost",
            "haunted", "murder", "scream", "screamed",
        ),
        Mood.BATTLE to setOf(
            "battle", "war", "sword", "swords", "army", "armies", "fight", "fighting",
            "soldiers", "enemy", "charge", "blade", "shield", "cannon", "siege",
        ),
    )

    private val wordSplit = Regex("[^\\p{L}\\p{Nd}]+")

    /**
     * Classify a chapter. Considers [chapterTitle] (weighted) plus the first
     * [WORD_WINDOW] words of [text]; everything after the window is ignored so
     * a 10k-word chapter classifies in O(window).
     */
    fun classify(chapterTitle: String, text: String): Mood {
        val titleWords = tokenize(chapterTitle)
        val bodyWords = tokenize(text).take(WORD_WINDOW)

        val scores = HashMap<Mood, Int>()
        for (word in titleWords) {
            for ((mood, bag) in keywords) {
                if (word in bag) scores.merge(mood, TITLE_WEIGHT, Int::plus)
            }
        }
        for (word in bodyWords) {
            for ((mood, bag) in keywords) {
                if (word in bag) scores.merge(mood, 1, Int::plus)
            }
        }
        return scores.maxByOrNull { it.value }?.key ?: Mood.CALM
    }

    private fun tokenize(s: String): List<String> =
        s.lowercase().split(wordSplit).filter { it.isNotBlank() }
}
