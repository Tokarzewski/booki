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
                ListItem(
                    leadingContent = {
                        RadioButton(
                            selected = currentQuality == factory.id,
                            enabled = installed,
                            onClick = {
                                with(Settings) { context.quality = factory.id }
                                currentQuality = factory.id
                            },
                        )
                    },
                    headlineContent = { Text(factory.displayName) },
                    supportingContent = {
                        Text(
                            if (installed) "Installed · ~${factory.ramMb} MB RAM"
                            else "Not installed · ${factory.downloadSizeMb} MB download",
                        )
                    },
                )
            }

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
                    Text(if (variant == ModelDownloader.Variant.FP32) "Kokoro v1.1 (FP32)" else "Kokoro v1.1 (INT8)")
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
