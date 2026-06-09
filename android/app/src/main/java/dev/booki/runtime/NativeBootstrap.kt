package dev.booki.runtime

import android.annotation.SuppressLint
import android.content.Context
import dev.booki.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.security.MessageDigest
import java.util.zip.ZipFile

/**
 * Issue #7: dynamic native library loading.
 *
 * When the APK is built with `-Pbooki.dynamicNativeLibs=true`, the sherpa-onnx
 * / ONNX Runtime `.so` files are stripped from the APK (~25 MB smaller) and
 * downloaded on first launch instead:
 *
 *  1. [install] downloads the official sherpa-onnx release archive, extracts
 *     the libs for the current ABI to `filesDir/native/<abi>/`, and verifies
 *     each against the hard-coded sha256 in [NativeLibManifest.CURRENT].
 *  2. [ensureLoaded] re-verifies the hashes, registers the directory with the
 *     app ClassLoader (so sherpa-onnx's own `System.loadLibrary` resolves) and
 *     `System.load`s each lib in dependency order — called by the engines
 *     before any sherpa-onnx class is touched.
 *  3. A corrupted install never loads: hash mismatch throws with a hint to use
 *     the "Repair runtime" button in Settings, which wipes and re-downloads.
 *
 * In default builds (`DYNAMIC_NATIVE_LIBS = false`) every entry point is a
 * no-op and the libs load from the APK as usual.
 */
object NativeBootstrap {

    val isDynamic: Boolean get() = BuildConfig.DYNAMIC_NATIVE_LIBS

    val manifest: NativeLibManifest get() = NativeLibManifest.CURRENT

    @Volatile private var loaded = false

    fun installDir(context: Context): File =
        File(context.filesDir, "native/${manifest.abi}")

    /** True when nothing needs downloading — always true for bundled builds. */
    fun isInstalled(context: Context): Boolean =
        !isDynamic || manifest.isSatisfiedBy(installDir(context))

    /** Bytes on disk used by the downloaded runtime (0 when not installed). */
    fun installedSizeBytes(context: Context): Long =
        installDir(context).listFiles()?.sumOf { it.length() } ?: 0L

    /**
     * Download + extract + verify the native runtime. Safe to call when
     * already installed (no-op). Progress mirrors [dev.booki.tts.ModelDownloader]'s
     * `(stage, done, total)` convention so setup UI can reuse its rendering.
     */
    suspend fun install(
        context: Context,
        onProgress: (stage: String, done: Long, total: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        if (isInstalled(context)) return@withContext
        val dir = installDir(context).apply { mkdirs() }
        val archive = File(context.cacheDir, manifest.archiveUrl.substringAfterLast('/'))

        if (!archive.exists() || archive.length() != manifest.archiveSizeBytes) {
            download(manifest.archiveUrl, archive) { done, total ->
                onProgress("Downloading native runtime", done, total)
            }
        }
        onProgress("Verifying native runtime", 0, -1)
        extractLibs(archive, manifest, dir)
        archive.delete()
        onProgress("Native runtime ready", 1, 1)
    }

    /**
     * Make sure the sherpa-onnx JNI libs are loadable, then load them.
     * Must be called before the first `com.k2fsa.sherpa.onnx.*` class is
     * initialized. Idempotent; no-op for bundled builds.
     */
    // System.load from filesDir is the point of this feature; every lib is
    // sha256-verified against a hash compiled into the APK immediately before.
    @SuppressLint("UnsafeDynamicallyLoadedCode")
    @Synchronized
    fun ensureLoaded(context: Context) {
        if (!isDynamic || loaded) return
        val dir = installDir(context)
        check(manifest.verify(dir)) {
            "Native runtime is missing or corrupted. " +
                "Use Settings → Native runtime → Repair to re-download."
        }
        // Register the directory so sherpa-onnx's own System.loadLibrary()
        // calls resolve against it.
        NativeDirInjector.inject(context.classLoader, dir)
        // Pre-load in dependency order so DT_NEEDED entries resolve.
        manifest.libs.forEach { System.load(File(dir, it.name).absolutePath) }
        loaded = true
    }

    /** Wipes the installed runtime; caller should re-run [install] afterwards. */
    fun repair(context: Context) {
        installDir(context).deleteRecursively()
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    private fun download(url: String, dest: File, onProgress: (Long, Long) -> Unit) {
        val tmp = File(dest.parentFile, "${dest.name}.part")
        val conn = (URI(url).toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Booki (+https://github.com/Tokarzewski/booki)")
        }
        check(conn.responseCode in 200..299) {
            "Download failed: HTTP ${conn.responseCode} for $url"
        }
        conn.inputStream.use { input ->
            tmp.outputStream().use { output ->
                val total = conn.contentLengthLong
                val buf = ByteArray(256 * 1024)
                var read: Int
                var done = 0L
                while (input.read(buf).also { read = it } != -1) {
                    output.write(buf, 0, read)
                    done += read
                    onProgress(done, total)
                }
            }
        }
        check(tmp.renameTo(dest)) { "Failed to finalize $dest" }
    }
}

/**
 * Extracts every lib in [manifest] from [archive] (a zip — the release AAR)
 * into [destDir], verifying sha256 before the file becomes visible under its
 * final name. Throws [IOException] on a missing entry or digest mismatch and
 * leaves no partial files behind.
 */
internal fun extractLibs(archive: File, manifest: NativeLibManifest, destDir: File) {
    destDir.mkdirs()
    ZipFile(archive).use { zip ->
        manifest.libs.forEach { lib ->
            val entry = zip.getEntry(lib.zipPath)
                ?: throw IOException("${lib.zipPath} not found in ${archive.name}")
            val tmp = File(destDir, "${lib.name}.part")
            try {
                zip.getInputStream(entry).use { input ->
                    tmp.outputStream().use { input.copyTo(it) }
                }
                val digest = sha256Of(tmp)
                if (!digest.equals(lib.sha256, ignoreCase = true)) {
                    throw IOException(
                        "sha256 mismatch for ${lib.name}: expected ${lib.sha256}, got $digest")
                }
                val dest = File(destDir, lib.name)
                dest.delete()
                if (!tmp.renameTo(dest)) throw IOException("Failed to finalize $dest")
            } finally {
                tmp.delete()
            }
        }
    }
}

/** Hex sha256 of a file's contents. */
internal fun sha256Of(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buf = ByteArray(256 * 1024)
        var read: Int
        while (input.read(buf).also { read = it } != -1) md.update(buf, 0, read)
    }
    return md.digest().joinToString("") { "%02x".format(it) }
}
