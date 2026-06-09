package dev.booki.runtime

import java.io.File
import java.lang.reflect.Field

/**
 * Registers an app-private directory as a native library search path on the
 * app's ClassLoader, so third-party classes whose static initializers call
 * `System.loadLibrary("...")` (sherpa-onnx does) resolve libraries that were
 * downloaded at runtime instead of packaged in the APK.
 *
 * This is the long-standing Tinker/SoLoader technique: prepend the directory
 * to `BaseDexClassLoader.pathList.nativeLibraryDirectories` and rebuild
 * `nativeLibraryPathElements` via `DexPathList.makePathElements`. The fields
 * are on Android's unsupported-but-allowed reflection list and stable for
 * API 26+ (our minSdk).
 *
 * Only used when `BuildConfig.DYNAMIC_NATIVE_LIBS` is true (issue #7).
 */
internal object NativeDirInjector {

    /**
     * @throws IllegalStateException if the classloader internals don't match —
     *         callers should surface this as "use the bundled APK instead".
     */
    fun inject(classLoader: ClassLoader, dir: File) {
        try {
            val pathList = findField(classLoader.javaClass, "pathList").get(classLoader)
                ?: error("pathList is null")

            @Suppress("UNCHECKED_CAST")
            val libDirs = findField(pathList.javaClass, "nativeLibraryDirectories")
                .get(pathList) as MutableList<File>
            if (dir !in libDirs) libDirs.add(0, dir)

            @Suppress("UNCHECKED_CAST")
            val systemDirs = findField(pathList.javaClass, "systemNativeLibraryDirectories")
                .get(pathList) as List<File>

            val allDirs = ArrayList<File>(libDirs.size + systemDirs.size).apply {
                addAll(libDirs)
                addAll(systemDirs)
            }
            val makePathElements = pathList.javaClass
                .getDeclaredMethod("makePathElements", List::class.java)
                .apply { isAccessible = true }
            val elements = makePathElements.invoke(null, allDirs)

            findField(pathList.javaClass, "nativeLibraryPathElements").set(pathList, elements)
        } catch (t: Throwable) {
            throw IllegalStateException(
                "Failed to register native library directory $dir on $classLoader. " +
                    "This Android version may not support dynamic native libs — " +
                    "install the standard (bundled) APK instead.",
                t,
            )
        }
    }

    private fun findField(start: Class<*>, name: String): Field {
        var cls: Class<*>? = start
        while (cls != null) {
            try {
                return cls.getDeclaredField(name).apply { isAccessible = true }
            } catch (_: NoSuchFieldException) {
                cls = cls.superclass
            }
        }
        throw NoSuchFieldException("$name not found on $start or its superclasses")
    }
}
