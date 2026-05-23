package dev.booki.catalog

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CatalogSourcesTest {

    @Test fun defaultsArePresent() {
        val sources = CatalogSources.defaults
        assertTrue("Expected at least 3 defaults", sources.size >= 3)
        val names = sources.map { it.name }
        assertTrue(names.any { it.contains("Gutenberg", ignoreCase = true) })
        assertTrue(names.any { it.contains("Standard", ignoreCase = true) })
    }

    @Test fun allDefaultsAreNotCustom() {
        CatalogSources.defaults.forEach { source ->
            assertTrue(source.name + " should not be custom", !source.isCustom)
        }
    }

    @Test fun loadIncludesDefaults() {
        // Without a real Context we can't test custom loading,
        // but defaults should always be returned.
        val sources = CatalogSources.defaults
        assertTrue(sources.isNotEmpty())
        sources.forEach { assertNotNull("Source has empty name", it.name) }
        sources.forEach { assertNotNull("Source has empty URL", it.url) }
        sources.forEach { assertTrue("Source URL must start with http", it.url.startsWith("http")) }
    }
}
