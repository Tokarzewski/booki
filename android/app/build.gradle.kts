import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val sherpaOnnxVersion = "1.13.2"

val downloadSherpaOnnx by tasks.registering {
    val aar = layout.projectDirectory.file("libs/sherpa-onnx-$sherpaOnnxVersion.aar").asFile
    outputs.file(aar)
    doLast {
        if (aar.exists()) return@doLast
        aar.parentFile.mkdirs()
        val url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
            "v$sherpaOnnxVersion/sherpa-onnx-$sherpaOnnxVersion.aar"
        logger.lifecycle("Downloading $url")
        URI(url).toURL().openStream().use { input ->
            aar.outputStream().use { input.copyTo(it) }
        }
        logger.lifecycle("Saved ${aar.length() / 1024 / 1024} MB to $aar")
    }
}

androidComponents.onVariants { variant ->
    afterEvaluate {
        tasks.matching { it.name.startsWith("pre") && it.name.endsWith("Build") }
            .configureEach { dependsOn(downloadSherpaOnnx) }
        tasks.matching { it.name == "collect${variant.name.replaceFirstChar { it.uppercase() }}Dependencies" }
            .configureEach { dependsOn(downloadSherpaOnnx) }
    }
}

android {
    namespace = "dev.booki"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.booki"
        minSdk = 26
        targetSdk = 36
        versionCode = 5
        versionName = "0.5.0"
        // Modern phones are arm64-v8a; dropping armeabi-v7a saves ~22 MB.
        ndk { abiFilters += setOf("arm64-v8a") }
        // Only ship English resources (Compose/AndroidX bundle ~70 locales).
        resourceConfigurations += setOf("en")
    }

    signingConfigs {
        create("release") {
            val keystoreFile = System.getenv("KEYSTORE_FILE")
                ?: project.findProperty("KEYSTORE_FILE") as String?
            if (!keystoreFile.isNullOrBlank()) {
                storeFile = file(keystoreFile)
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                    ?: project.findProperty("KEYSTORE_PASSWORD") as String?
                keyAlias = System.getenv("KEY_ALIAS")
                    ?: project.findProperty("KEY_ALIAS") as String?
                keyPassword = System.getenv("KEY_PASSWORD")
                    ?: project.findProperty("KEY_PASSWORD") as String?
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            val ks = System.getenv("KEYSTORE_FILE") ?: project.findProperty("KEYSTORE_FILE") as String?
            signingConfig = if (!ks.isNullOrBlank()) signingConfigs.getByName("release") else signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    packaging {
        resources.excludes += setOf(
            "META-INF/AL2.0",
            "META-INF/LGPL2.1",
            "META-INF/DEPENDENCIES",
        )
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.09.02")
    implementation(composeBom)
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.documentfile:documentfile:1.0.1")

    // EPUB parsing (epub4j is the maintained fork of epublib on Maven Central).
    // Excludes: org.slf4j (no logger needed); xmlpull + net.sf.kxml (collide
    // with the XmlPullParser interface that Android already ships).
    implementation("io.documentnode:epub4j-core:4.2.2") {
        exclude(group = "org.slf4j")
        exclude(group = "xmlpull")
        exclude(group = "net.sf.kxml")
    }
    implementation("org.jsoup:jsoup:1.18.1")

    // sherpa-onnx: wraps Kokoro + espeak-ng phonemization + ONNX Runtime.
    // The AAR is downloaded by the `:app:downloadSherpaOnnx` task into app/libs/.
    implementation(files("libs/sherpa-onnx-$sherpaOnnxVersion.aar"))

    // tar.bz2 extraction for the Kokoro model bundle.
    implementation("org.apache.commons:commons-compress:1.27.1")
    implementation("org.tukaani:xz:1.10")

    // Coroutines / WorkManager
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.work:work-runtime-ktx:2.9.1")

    // ExoPlayer + MediaSession for lock-screen / Bluetooth playback
    implementation("androidx.media3:media3-exoplayer:1.4.1")
    implementation("androidx.media3:media3-session:1.4.1")
    implementation("androidx.media3:media3-common:1.4.1")

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
}
