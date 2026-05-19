package dev.booki.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.tts.ModelDownloader
import kotlinx.coroutines.launch

private sealed interface DownloadState {
    data object Idle : DownloadState
    data class Running(val asset: String, val done: Long, val total: Long) : DownloadState {
        val fraction: Float? get() = if (total > 0) done.toFloat() / total else null
    }
    data class Failed(val message: String) : DownloadState
}

@Composable
fun SetupScreen(onProvisioned: () -> Unit) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<DownloadState>(DownloadState.Idle) }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Booki", style = MaterialTheme.typography.headlineMedium)
            Text(
                "First-time setup downloads the Kokoro-82M voice model (~330 MB) " +
                    "into private app storage. Wi-Fi recommended.",
                style = MaterialTheme.typography.bodyMedium,
            )

            when (val s = state) {
                DownloadState.Idle -> Button(onClick = {
                    state = DownloadState.Running("preparing…", 0, 0)
                    scope.launch {
                        runCatching {
                            ModelDownloader.download(context) { name, done, total ->
                                state = DownloadState.Running(name, done, total)
                            }
                        }.onSuccess { onProvisioned() }
                            .onFailure { state = DownloadState.Failed(it.message ?: "Download failed") }
                    }
                }) { Text("Download voice model") }

                is DownloadState.Running -> Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(s.asset)
                    val frac = s.fraction
                    if (frac != null) {
                        LinearProgressIndicator(
                            progress = { frac },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text("${(frac * 100).toInt()}%  (${s.done / 1_048_576} / ${s.total / 1_048_576} MB)")
                    } else {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
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
