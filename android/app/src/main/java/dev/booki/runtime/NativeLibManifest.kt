package dev.booki.runtime

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * One native library that must be present before sherpa-onnx classes are
 * touched. [zipPath] is the entry inside the runtime archive; [sha256] is the
 * hex digest of the *extracted* file and is verified on install and on load.
 */
data class NativeLib(
    val name: String,
    val zipPath: String,
    val sha256: String,
    val sizeBytes: Long,
)

/**
 * Manifest describing the native runtime bundle for one ABI (issue #7).
 *
 * The built-in [CURRENT] manifest carries hard-coded hashes so a tampered or
 * truncated download can never be `System.load`ed. The JSON round-trip exists
 * so a future remote manifest (e.g. attached to a GitHub release) can replace
 * the built-in one without a code change to the parsing path.
 *
 * [libs] is ordered by load order — dependencies first (`libonnxruntime.so`
 * before `libsherpa-onnx-jni.so`, which links against it).
 */
data class NativeLibManifest(
    val runtime: String,
    val version: String,
    val abi: String,
    val archiveUrl: String,
    val archiveSizeBytes: Long,
    val libs: List<NativeLib>,
) {
    /** Cheap check (existence + size) used to decide whether setup must run. */
    fun isSatisfiedBy(dir: File): Boolean =
        libs.all { File(dir, it.name).length() == it.sizeBytes }

    /** Full integrity check — sha256 of every lib on disk. */
    fun verify(dir: File): Boolean =
        libs.all { lib ->
            val f = File(dir, lib.name)
            f.exists() && sha256Of(f).equals(lib.sha256, ignoreCase = true)
        }

    fun toJson(): String = JSONObject().apply {
        put("runtime", runtime)
        put("version", version)
        put("abi", abi)
        put("archiveUrl", archiveUrl)
        put("archiveSizeBytes", archiveSizeBytes)
        put("libs", JSONArray().apply {
            libs.forEach { lib ->
                put(JSONObject().apply {
                    put("name", lib.name)
                    put("zipPath", lib.zipPath)
                    put("sha256", lib.sha256)
                    put("sizeBytes", lib.sizeBytes)
                })
            }
        })
    }.toString()

    companion object {
        const val RUNTIME_SHERPA = "sherpa-onnx"
        const val SHERPA_VERSION = "1.13.2"
        const val ABI_ARM64 = "arm64-v8a"

        /**
         * arm64-v8a libs from the official sherpa-onnx release AAR — the same
         * artifact the Gradle build compiles against, so the Kotlin binding
         * and the downloaded .so files always match versions.
         */
        val CURRENT = NativeLibManifest(
            runtime = RUNTIME_SHERPA,
            version = SHERPA_VERSION,
            abi = ABI_ARM64,
            archiveUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
                "v$SHERPA_VERSION/sherpa-onnx-$SHERPA_VERSION.aar",
            archiveSizeBytes = 56_655_608,
            libs = listOf(
                NativeLib(
                    name = "libonnxruntime.so",
                    zipPath = "jni/$ABI_ARM64/libonnxruntime.so",
                    sha256 = "4d2318b3849abb8862133d3068fc7e807ed8b2671cc6d83657fff2fcb9e1caad",
                    sizeBytes = 25_831_632,
                ),
                NativeLib(
                    name = "libsherpa-onnx-jni.so",
                    zipPath = "jni/$ABI_ARM64/libsherpa-onnx-jni.so",
                    sha256 = "fc072f201dc1923ee98b594eb61c796b538ef087f7f18d08dcfdf0565167a8bd",
                    sizeBytes = 4_623_192,
                ),
            ),
        )

        fun fromJson(json: String): NativeLibManifest {
            val obj = JSONObject(json)
            val arr = obj.getJSONArray("libs")
            return NativeLibManifest(
                runtime = obj.getString("runtime"),
                version = obj.getString("version"),
                abi = obj.getString("abi"),
                archiveUrl = obj.getString("archiveUrl"),
                archiveSizeBytes = obj.getLong("archiveSizeBytes"),
                libs = (0 until arr.length()).map { i ->
                    val lib = arr.getJSONObject(i)
                    NativeLib(
                        name = lib.getString("name"),
                        zipPath = lib.getString("zipPath"),
                        sha256 = lib.getString("sha256"),
                        sizeBytes = lib.getLong("sizeBytes"),
                    )
                },
            )
        }
    }
}
