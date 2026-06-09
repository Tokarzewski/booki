package dev.booki.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class NativeBootstrapTest {

    @get:Rule val tmp = TemporaryFolder()

    // -- sha256 ------------------------------------------------------------

    @Test fun sha256OfKnownVector() {
        val f = tmp.newFile().apply { writeText("abc") }
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            sha256Of(f),
        )
    }

    @Test fun sha256OfEmptyFile() {
        val f = tmp.newFile()
        assertEquals(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            sha256Of(f),
        )
    }

    // -- built-in manifest sanity -------------------------------------------

    @Test fun builtInManifestShape() {
        val m = NativeLibManifest.CURRENT
        assertEquals("sherpa-onnx", m.runtime)
        assertEquals("arm64-v8a", m.abi)
        assertTrue(m.archiveUrl.startsWith("https://"))
        assertTrue(m.archiveUrl.contains(m.version))
        // Load order: the JNI lib links against onnxruntime, so onnxruntime first.
        assertEquals(listOf("libonnxruntime.so", "libsherpa-onnx-jni.so"), m.libs.map { it.name })
        m.libs.forEach { lib ->
            assertTrue("${lib.name} hash must be 64 hex chars",
                lib.sha256.matches(Regex("[0-9a-f]{64}")))
            assertTrue("${lib.name} must live under jni/<abi>/",
                lib.zipPath == "jni/${m.abi}/${lib.name}")
            assertTrue(lib.sizeBytes > 0)
        }
    }

    @Test fun manifestJsonRoundTrip() {
        val m = NativeLibManifest.CURRENT
        assertEquals(m, NativeLibManifest.fromJson(m.toJson()))
    }

    // -- isSatisfiedBy / verify ---------------------------------------------

    @Test fun satisfiedAndVerifiedAfterExtraction() {
        val (archive, manifest) = fakeBundle("first lib" to "second lib")
        val dest = tmp.newFolder()
        extractLibs(archive, manifest, dest)
        assertTrue(manifest.isSatisfiedBy(dest))
        assertTrue(manifest.verify(dest))
    }

    @Test fun notSatisfiedWhenMissingOrTruncated() {
        val (archive, manifest) = fakeBundle("first lib" to "second lib")
        val dest = tmp.newFolder()
        assertFalse(manifest.isSatisfiedBy(dest)) // nothing extracted yet

        extractLibs(archive, manifest, dest)
        File(dest, manifest.libs[0].name).writeText("x") // truncate
        assertFalse(manifest.isSatisfiedBy(dest))
        assertFalse(manifest.verify(dest))
    }

    @Test fun verifyCatchesSameSizeCorruption() {
        val (archive, manifest) = fakeBundle("first lib" to "second lib")
        val dest = tmp.newFolder()
        extractLibs(archive, manifest, dest)
        // Same length, flipped content — the cheap size check can't see this...
        File(dest, manifest.libs[0].name).writeText("FIRST LIB")
        assertTrue(manifest.isSatisfiedBy(dest))
        // ...but the hash check must.
        assertFalse(manifest.verify(dest))
    }

    // -- extractLibs ----------------------------------------------------------

    @Test fun extractRefusesHashMismatch() {
        val (archive, good) = fakeBundle("first lib" to "second lib")
        val tampered = good.copy(
            libs = good.libs.mapIndexed { i, lib ->
                if (i == 0) lib.copy(sha256 = "0".repeat(64)) else lib
            },
        )
        val dest = tmp.newFolder()
        val err = runCatching { extractLibs(archive, tampered, dest) }.exceptionOrNull()
        assertTrue("expected IOException, got $err", err is IOException)
        assertTrue(err!!.message!!.contains("sha256 mismatch"))
        // No partial or final file may be left behind for the bad lib.
        assertFalse(File(dest, tampered.libs[0].name).exists())
        assertFalse(File(dest, "${tampered.libs[0].name}.part").exists())
    }

    @Test fun extractRefusesMissingEntry() {
        val (archive, good) = fakeBundle("first lib" to "second lib")
        val manifest = good.copy(
            libs = good.libs + NativeLib(
                name = "libghost.so",
                zipPath = "jni/arm64-v8a/libghost.so",
                sha256 = "0".repeat(64),
                sizeBytes = 1,
            ),
        )
        val dest = tmp.newFolder()
        val err = runCatching { extractLibs(archive, manifest, dest) }.exceptionOrNull()
        assertTrue("expected IOException, got $err", err is IOException)
        assertTrue(err!!.message!!.contains("libghost.so"))
    }

    @Test fun extractIsIdempotent() {
        val (archive, manifest) = fakeBundle("first lib" to "second lib")
        val dest = tmp.newFolder()
        extractLibs(archive, manifest, dest)
        extractLibs(archive, manifest, dest) // overwrite in place
        assertTrue(manifest.verify(dest))
    }

    // -- helpers --------------------------------------------------------------

    /**
     * Builds a zip laid out like the sherpa-onnx AAR (`jni/<abi>/lib*.so`)
     * with the given contents, and a manifest whose hashes match it.
     */
    private fun fakeBundle(contents: Pair<String, String>): Pair<File, NativeLibManifest> {
        val names = listOf("libonnxruntime.so", "libsherpa-onnx-jni.so")
        val bodies = listOf(contents.first, contents.second)
        val archive = tmp.newFile("bundle.zip")
        ZipOutputStream(archive.outputStream()).use { zip ->
            names.forEachIndexed { i, name ->
                zip.putNextEntry(ZipEntry("jni/arm64-v8a/$name"))
                zip.write(bodies[i].toByteArray())
                zip.closeEntry()
            }
        }
        val libs = names.mapIndexed { i, name ->
            val scratch = tmp.newFile().apply { writeText(bodies[i]) }
            NativeLib(
                name = name,
                zipPath = "jni/arm64-v8a/$name",
                sha256 = sha256Of(scratch),
                sizeBytes = bodies[i].toByteArray().size.toLong(),
            )
        }
        val manifest = NativeLibManifest(
            runtime = "sherpa-onnx",
            version = "test",
            abi = "arm64-v8a",
            archiveUrl = "https://example.invalid/bundle.zip",
            archiveSizeBytes = archive.length(),
            libs = libs,
        )
        return archive to manifest
    }
}
