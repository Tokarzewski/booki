package dev.booki.catalog

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Catalog sources = built-in defaults + user-added feeds stored in SharedPrefs.
 */
object CatalogSources {
    data class Source(
        val name: String,
        val description: String,
        val url: String,
        val isCustom: Boolean = false,
    )

    private const val PREFS = "booki.catalog"
    private const val KEY_CUSTOM = "custom_feeds"

    val defaults: List<Source> = listOf(
        Source(
            name = "Project Gutenberg",
            description = "~75k public-domain books",
            url = "https://m.gutenberg.org/ebooks.opds/",
        ),
        Source(
            name = "Standard Ebooks",
            description = "Public-domain books with polished typography",
            url = "https://standardebooks.org/opds",
        ),
        Source(
            name = "Feedbooks — Public Domain",
            description = "Curated public-domain catalog",
            url = "https://catalog.feedbooks.com/catalog/public_domain.atom",
        ),
        Source(
            name = "Internet Archive",
            description = "Mixed catalog from archive.org",
            url = "https://bookserver.archive.org/",
        ),
    )

    fun load(context: Context): List<Source> = defaults + custom(context)

    fun custom(context: Context): List<Source> {
        val raw = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_CUSTOM, null) ?: return emptyList()
        return runCatching {
            val arr = JSONArray(raw)
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                Source(
                    name = o.getString("name"),
                    description = o.optString("description", o.getString("url")),
                    url = o.getString("url"),
                    isCustom = true,
                )
            }
        }.getOrDefault(emptyList())
    }

    fun add(context: Context, name: String, url: String) {
        val list = custom(context).toMutableList()
        list += Source(name = name, description = url, url = url, isCustom = true)
        save(context, list)
    }

    fun remove(context: Context, source: Source) {
        if (!source.isCustom) return
        save(context, custom(context).filterNot { it.url == source.url })
    }

    private fun save(context: Context, list: List<Source>) {
        val arr = JSONArray()
        list.forEach { s ->
            arr.put(JSONObject().apply {
                put("name", s.name)
                put("description", s.description)
                put("url", s.url)
            })
        }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_CUSTOM, arr.toString()).apply()
    }
}
