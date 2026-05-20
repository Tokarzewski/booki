# sherpa-onnx JNI bridge — reflection-heavy, must be preserved end-to-end.
-keep class com.k2fsa.sherpa.onnx.** { *; }
-dontwarn com.k2fsa.sherpa.onnx.**

# epub4j relies on reflection for resource loading and zip entry classes.
-keep class io.documentnode.epub4j.** { *; }
-dontwarn io.documentnode.epub4j.**

# media3 / ExoPlayer — keep IL2-style internal classes used via reflection.
-keep class androidx.media3.** { *; }
-dontwarn androidx.media3.**

# commons-compress optional dependencies we don't use.
-dontwarn org.brotli.dec.**
-dontwarn org.tukaani.xz.**
-dontwarn com.github.luben.zstd.**

# Coroutines internal sentinels.
-keepclassmembers class kotlinx.coroutines.** { volatile <fields>; }
