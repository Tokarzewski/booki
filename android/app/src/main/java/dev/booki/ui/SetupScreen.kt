package dev.booki.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.data.Settings
import dev.booki.data.Settings.quality
import dev.booki.runtime.NativeBootstrap
import dev.booki.tts.Engines
import dev.booki.tts.ModelDownloader
import dev.booki.tts.SpeechEngine
import kotlinx.coroutines.launch

private sealed interface DownloadState {
    data object Idle : DownloadState
    data class Running(val stage: String, val done: Long, val total: Long) : DownloadState {
        val fraction: Float? get() = if (total > 0) done.toFloat() / total else null
        val isIndeterminate: Boolean get() = total <= 0
    }
    data class Failed(val message: String) : DownloadState
}

@Composable
fun SetupScreen(onProvisioned: () -> Unit) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<DownloadState>(DownloadState.Idle) }

    // Pre-select the highest-quality engine the device can actually run.
    // Issue #11: gate by RAM so low-end phones don't crash on FP32.
    val bestVariant = remember {
        val supported = Engines.supported(context)
        when {
            supported.any { it.id == SpeechEngine.Quality.KOKORO_FP32 } -> ModelDownloader.Variant.FP32
            supported.any { it.id == SpeechEngine.Quality.KOKORO_INT8 } -> ModelDownloader.Variant.INT8
            supported.any { it.id == SpeechEngine.Quality.MATCHA_EN } -> ModelDownloader.Variant.MATCHA_EN
            supported.any { it.id == SpeechEngine.Quality.MATCHA_ZH_EN } -> ModelDownloader.Variant.MATCHA_ZH_EN
            else -> ModelDownloader.Variant.MATCHA_EN // safe fallback (lowest RAM)
        }
    }
    var pickedVariant by remember { mutableStateOf(bestVariant) }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Booki", style = MaterialTheme.typography.headlineMedium)
            Text(
                "Pick a voice model to download. You can switch later in settings.",
                style = MaterialTheme.typography.bodyMedium,
            )

            VariantChoice(
                selected = pickedVariant,
                onSelect = { pickedVariant = it },
                enabled = state !is DownloadState.Running,
            )

            when (val s = state) {
                DownloadState.Idle -> Button(onClick = {
                    state = DownloadState.Running("preparing…", 0, 0)
                    scope.launch {
                        runCatching {
                            // Issue #7: dynamic builds fetch the native runtime
                            // before the model. No-op for bundled builds.
                            NativeBootstrap.install(context) { stage, done, total ->
                                state = DownloadState.Running(stage, done, total)
                            }
                            ModelDownloader.download(context, pickedVariant) { stage, done, total ->
                                state = DownloadState.Running(stage, done, total)
                            }
                        }.onSuccess {
                            with(Settings) {
                                context.quality = when (pickedVariant) {
                                    ModelDownloader.Variant.FP32 -> SpeechEngine.Quality.KOKORO_FP32
                                    ModelDownloader.Variant.INT8 -> SpeechEngine.Quality.KOKORO_INT8
                                    ModelDownloader.Variant.MATCHA_EN -> SpeechEngine.Quality.MATCHA_EN
                                    ModelDownloader.Variant.MATCHA_ZH_EN -> SpeechEngine.Quality.MATCHA_ZH_EN
                                }
                            }
                            onProvisioned()
                        }.onFailure { state = DownloadState.Failed(it.message ?: "Download failed") }
                    }
                }) { Text("Download ${pickedVariant.sizeMb} MB") }

                is DownloadState.Running -> Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(s.stage, style = MaterialTheme.typography.bodySmall)
                    val frac = s.fraction
                    if (frac != null && !s.isIndeterminate) {
                        LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
                        Text("${(frac * 100).toInt()}%  (${s.done / 1_048_576} / ${s.total / 1_048_576} MB)")
                    } else {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text("${s.done / 1_048_576} MB extracted")
                    }
                }

                is DownloadState.Failed -> Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Error: ${s.message}", color = MaterialTheme.colorScheme.error)
                    Button(onClick = { state = DownloadState.Idle }) { Text("Retry") }
                }
            }
        }
    }
}

@Composable
private fun VariantChoice(
    selected: ModelDownloader.Variant,
    onSelect: (ModelDownloader.Variant) -> Unit,
    enabled: Boolean,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        VariantCard(
            title = "Kokoro v1.1 — high quality",
            subtitle = "348 MB download · ~650 MB on disk · reference fp32 weights",
            selected = selected == ModelDownloader.Variant.FP32,
            enabled = enabled,
            onClick = { onSelect(ModelDownloader.Variant.FP32) },
        )
        VariantCard(
            title = "Kokoro v1.1 — fast (INT8)",
            subtitle = "140 MB download · ~310 MB on disk · ~3× faster, slight quality drop",
            selected = selected == ModelDownloader.Variant.INT8,
            enabled = enabled,
            onClick = { onSelect(ModelDownloader.Variant.INT8) },
        )
        VariantCard(
            title = "Matcha-TTS — English",
            subtitle = "73 MB download · ~200 MB on disk · smaller footprint, neutral voice",
            selected = selected == ModelDownloader.Variant.MATCHA_EN,
            enabled = enabled,
            onClick = { onSelect(ModelDownloader.Variant.MATCHA_EN) },
        )
        VariantCard(
            title = "Matcha-TTS — Chinese-English",
            subtitle = "75 MB download · ~200 MB on disk · bilingual zh/en support",
            selected = selected == ModelDownloader.Variant.MATCHA_ZH_EN,
            enabled = enabled,
            onClick = { onSelect(ModelDownloader.Variant.MATCHA_ZH_EN) },
        )
    }
}

@Composable
private fun VariantCard(
    title: String,
    subtitle: String,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    OutlinedCard(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.fillMaxWidth(),
        border = androidx.compose.foundation.BorderStroke(
            width = if (selected) 2.dp else 1.dp,
            color = if (selected) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.outlineVariant,
        ),
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            RadioButton(selected = selected, onClick = onClick, enabled = enabled)
            Column {
                Text(title, style = MaterialTheme.typography.titleSmall)
                Text(subtitle, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
