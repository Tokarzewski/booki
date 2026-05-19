package dev.booki.catalog

/** Pre-seeded OPDS feeds. Users can add more later via settings (TODO). */
object CatalogSources {
    data class Source(val name: String, val description: String, val url: String)

    val all: List<Source> = listOf(
        Source(
            name = "Project Gutenberg",
            description = "~75k public-domain books",
            url = "https://m.gutenberg.org/ebooks.opds/",
        ),
        Source(
            name = "Standard Ebooks",
            description = "Public domain books with polished typography",
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
}
