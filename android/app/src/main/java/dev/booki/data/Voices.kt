package dev.booki.data

/**
 * Kokoro-82M voice catalog. First char = language, second = gender (f/m).
 * The `sid` is the integer speaker id sherpa-onnx expects when calling
 * `OfflineTts.generate(text, sid, speed)` — taken from the order in
 * `voices.bin` shipped with sherpa-onnx's `kokoro-multi-lang-v1_*` bundle.
 */
object Voices {
    data class Voice(val id: String, val sid: Int, val language: String, val female: Boolean)

    private val ORDER: List<String> = listOf(
        // American English
        "af_alloy","af_aoede","af_bella","af_heart","af_jessica","af_kore",
        "af_nicole","af_nova","af_river","af_sarah","af_sky",
        "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael",
        "am_onyx","am_puck","am_santa",
        // British English
        "bf_alice","bf_emma","bf_isabella","bf_lily",
        "bm_daniel","bm_fable","bm_george","bm_lewis",
        // Spanish
        "ef_dora","em_alex","em_santa",
        // French
        "ff_siwis",
        // Hindi
        "hf_alpha","hf_beta","hm_omega","hm_psi",
        // Italian
        "if_sara","im_nicola",
        // Japanese
        "jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro","jm_kumo",
        // Brazilian Portuguese
        "pf_dora","pm_alex","pm_santa",
        // Mandarin Chinese
        "zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi",
        "zm_yunjian","zm_yunxi","zm_yunxia","zm_yunyang",
    )

    val all: List<Voice> = ORDER.mapIndexed { sid, id ->
        Voice(id = id, sid = sid, language = languageName(id[0]), female = id[1] == 'f')
    }

    private val bySid: Map<String, Int> = all.associate { it.id to it.sid }

    fun sidOf(voiceId: String): Int = bySid[voiceId] ?: 10  // default to af_sky

    private fun languageName(c: Char): String = when (c) {
        'a' -> "American English"
        'b' -> "British English"
        'e' -> "Spanish"
        'f' -> "French"
        'h' -> "Hindi"
        'i' -> "Italian"
        'j' -> "Japanese"
        'p' -> "Brazilian Portuguese"
        'z' -> "Mandarin Chinese"
        else -> "Unknown"
    }

    const val DEFAULT: String = "af_sky"
}
