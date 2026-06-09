package dev.booki.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.data.Settings
import dev.booki.data.Settings.defaultSpeed
import dev.booki.data.Settings.defaultVoice
import dev.booki.data.Settings.quality
import dev.booki.data.Voices
import dev.booki.runtime.NativeBootstrap
import dev.booki.tts.CloudSettings
import dev.booki.tts.CloudSettings.Provider
import dev.booki.tts.CloudSettings.cloudApiKey
import dev.booki.tts.CloudSettings.cloudProvider
import dev.booki.tts.Engines
import dev.booki.tts.ModelDownloader
import dev.booki.tts.SpeechEngine
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()
    var currentQuality by remember { mutableStateOf(context.quality) }
    var defaultVoiceId by remember { mutableStateOf(context.defaultVoice) }
    var defaultSpeedValue by remember { mutableFloatStateOf(context.defaultSpeed) }
    var voiceMenu by remember { mutableStateOf(false) }
    var voiceFilter by remember { mutableStateOf("") }
    var downloadStage by remember { mutableStateOf<String?>(null) }
    var downloadFrac by remember { mutableFloatStateOf(0f) }
    val snackbar = remember { SnackbarHostState() }

    val filteredVoices = remember(voiceFilter) {
        if (voiceFilter.isBlank()) Voices.all
        else Voices.all.filter {
            it.id.contains(voiceFilter, ignoreCase = true) ||
                it.language.contains(voiceFilter, ignoreCase = true)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        Column(
            modifier = Modifier.padding(pad).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Text("Voice model", style = MaterialTheme.typography.titleMedium)
            ModelVariantSection(
                onAction = { variant, isInstalled ->
                    if (isInstalled) {
                        ModelDownloader.remove(context, variant)
                        scope.launch { snackbar.showSnackbar("Removed ${variant.subdir}") }
                    } else {
                        scope.launch {
                            downloadStage = "Downloading ${variant.subdir}"; downloadFrac = 0f
                            runCatching {
                                ModelDownloader.download(context, variant) { stage, done, total ->
                                    downloadStage = stage
                                    downloadFrac = if (total > 0) done.toFloat() / total else 0f
                                }
                            }.onSuccess { snackbar.showSnackbar("Installed ${variant.subdir}") }
                                .onFailure { snackbar.showSnackbar("Failed: ${it.message}") }
                            downloadStage = null
                        }
                    }
                },
                downloadStage = downloadStage,
                downloadFrac = downloadFrac,
            )

            HorizontalDivider()

            Text("Active quality", style = MaterialTheme.typography.titleMedium)
            Engines.factories.forEach { factory ->
                val installed = factory.isProvisioned(context)
                val supported = factory.isSupportedOn(context)
                val selectable = installed && supported && !factory.isExperimental
                ListItem(
                    leadingContent = {
                        RadioButton(
                            selected = currentQuality == factory.id,
                            enabled = selectable,
                            onClick = {
                                with(Settings) { context.quality = factory.id }
                                currentQuality = factory.id
                            },
                        )
                    },
                    headlineContent = { Text(factory.displayName) },
                    supportingContent = {
                        Text(
                            when {
                                factory.isExperimental -> "Coming soon — tracked in issue #6"
                                !supported -> "Not supported on this device (needs ~${factory.ramMb * 3 / 2} MB RAM)"
                                installed -> "Installed · ~${factory.ramMb} MB RAM"
                                else -> "Not installed · ${factory.downloadSizeMb} MB download"
                            },
                        )
                    },
                )
            }

            if (NativeBootstrap.isDynamic) {
                HorizontalDivider()

                // Dynamic native runtime status + repair (Issue #7)
                Text("Native runtime", style = MaterialTheme.typography.titleMedium)
                NativeRuntimeSection(snackbar)
            }

            HorizontalDivider()

            // Cloud TTS configuration (Issue #9)
            Text("Cloud TTS", style = MaterialTheme.typography.titleMedium)
            CloudTtsConfigSection()

            HorizontalDivider()

            Text("Default voice", style = MaterialTheme.typography.titleMedium)
            ExposedDropdownMenuBox(expanded = voiceMenu, onExpandedChange = { voiceMenu = !voiceMenu }) {
                OutlinedTextField(
                    value = defaultVoiceId,
                    onValueChange = {},
                    readOnly = true,
                    modifier = Modifier
                        .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                        .fillMaxWidth(),
                )
                ExposedDropdownMenu(expanded = voiceMenu, onDismissRequest = { voiceMenu = false }) {
                    OutlinedTextField(
                        value = voiceFilter,
                        onValueChange = { voiceFilter = it },
                        singleLine = true,
                        placeholder = { Text("Filter voices…") },
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                    filteredVoices.forEach { v ->
                        DropdownMenuItem(
                            text = { Text("${v.id} — ${v.language}") },
                            onClick = {
                                defaultVoiceId = v.id
                                with(Settings) { context.defaultVoice = v.id }
                                voiceMenu = false; voiceFilter = ""
                            },
                        )
                    }
                }
            }

            Column {
                Text("Default speed: ${"%.2f".format(defaultSpeedValue)}×",
                    style = MaterialTheme.typography.titleMedium)
                Slider(
                    value = defaultSpeedValue,
                    onValueChange = {
                        defaultSpeedValue = it
                        with(Settings) { context.defaultSpeed = it }
                    },
                    valueRange = 0.5f..2f,
                )
            }
        }
    }
}

/**
 * Issue #7: shown only in dynamic-native-lib builds. Reports the active
 * runtime version + on-disk size and offers "Repair runtime" — the recovery
 * path for a corrupted .so (which would otherwise crash synthesis).
 */
@Composable
private fun NativeRuntimeSection(snackbar: SnackbarHostState) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()
    var installed by remember { mutableStateOf(NativeBootstrap.isInstalled(context)) }
    var busy by remember { mutableStateOf(false) }
    var stage by remember { mutableStateOf<String?>(null) }
    var frac by remember { mutableFloatStateOf(0f) }
    val manifest = NativeBootstrap.manifest

    ListItem(
        headlineContent = { Text("${manifest.runtime} ${manifest.version} (${manifest.abi})") },
        supportingContent = {
            Text(
                if (installed) {
                    "Installed · ${NativeBootstrap.installedSizeBytes(context) / 1_048_576} MB on disk"
                } else {
                    "Not installed — synthesis unavailable until repaired"
                },
            )
        },
        trailingContent = {
            TextButton(
                enabled = !busy,
                onClick = {
                    busy = true
                    scope.launch {
                        NativeBootstrap.repair(context)
                        installed = false
                        runCatching {
                            NativeBootstrap.install(context) { s, done, total ->
                                stage = s
                                frac = if (total > 0) done.toFloat() / total else 0f
                            }
                        }.onSuccess {
                            installed = NativeBootstrap.isInstalled(context)
                            snackbar.showSnackbar("Native runtime reinstalled")
                        }.onFailure {
                            snackbar.showSnackbar("Repair failed: ${it.message}")
                        }
                        stage = null
                        busy = false
                    }
                },
            ) { Text("Repair runtime") }
        },
    )
    stage?.let { s ->
        Column {
            Text(s, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(4.dp))
            LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun CloudTtsConfigSection() {
    val context = LocalContextSafe.current
    var provider by remember { mutableStateOf(context.cloudProvider) }
    var apiKey by remember { mutableStateOf(context.cloudApiKey) }
    var keyVisible by remember { mutableStateOf(false) }

    // Provider toggle
    Row(
        verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.padding(vertical = 4.dp),
    ) {
        Text("ElevenLabs", style = MaterialTheme.typography.bodyMedium)
        Switch(
            checked = provider == CloudSettings.Provider.FISH,
            onCheckedChange = {
                provider = if (it) CloudSettings.Provider.FISH else CloudSettings.Provider.ELEVENLABS
                with(CloudSettings) { context.cloudProvider = provider }
            },
        )
        Text("Fish.audio", style = MaterialTheme.typography.bodyMedium)
    }

    // API key input
    OutlinedTextField(
        value = apiKey,
        onValueChange = {
            apiKey = it
            with(CloudSettings) { context.cloudApiKey = it }
        },
        label = { Text("API key") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
        visualTransformation = if (keyVisible) androidx.compose.ui.text.input.VisualTransformation.None
            else androidx.compose.ui.text.input.PasswordVisualTransformation(),
        trailingIcon = {
            IconButton(onClick = { keyVisible = !keyVisible }) {
                Text(if (keyVisible) "Hide" else "Show")
            }
        },
    )

    // Cost warning
    if (apiKey.isNotBlank()) {
        val sampleCost = CloudSettings.estimateCost(provider, 100_000)
        Text(
            "≈ $sampleCost per 100k chars · ~\$${CloudSettings.estimatedCostPerChar(provider) * 1_000_000}/M chars",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ModelVariantSection(
    onAction: (ModelDownloader.Variant, isInstalled: Boolean) -> Unit,
    downloadStage: String?,
    downloadFrac: Float,
) {
    val context = LocalContextSafe.current
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ModelDownloader.Variant.entries.forEach { variant ->
            val installed = ModelDownloader.isProvisioned(context, variant)
            ListItem(
                headlineContent = {
                    Text(when (variant) {
                        ModelDownloader.Variant.FP32 -> "Kokoro v1.1 (FP32)"
                        ModelDownloader.Variant.INT8 -> "Kokoro v1.1 (INT8)"
                        ModelDownloader.Variant.MATCHA_EN -> "Matcha-TTS (English)"
                        ModelDownloader.Variant.MATCHA_ZH_EN -> "Matcha-TTS (Chinese-English)"
                    })
                },
                supportingContent = {
                    Text(if (installed) "Installed" else "${variant.sizeMb} MB")
                },
                trailingContent = {
                    IconButton(onClick = { onAction(variant, installed) }) {
                        Icon(
                            if (installed) Icons.Default.Delete else Icons.Default.Download,
                            contentDescription = if (installed) "Remove" else "Download",
                        )
                    }
                },
            )
        }
        if (downloadStage != null) {
            Surface(tonalElevation = 2.dp, modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text(downloadStage, style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(progress = { downloadFrac }, modifier = Modifier.fillMaxWidth())
                }
            }
        }
    }
}
