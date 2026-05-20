package dev.booki.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.FastForward
import androidx.compose.material.icons.filled.FastRewind
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.booki.player.PlayerController
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlayerScreen(onBack: () -> Unit) {
    val context = LocalContextSafe.current
    val state by PlayerController.state.collectAsState()

    LaunchedEffect(Unit) { PlayerController.connect(context) }
    LaunchedEffect(state.isPlaying) {
        var ticks = 0
        while (state.isPlaying) {
            PlayerController.refresh()
            ticks++
            // Persist position every ~5 seconds.
            if (ticks % 10 == 0) PlayerController.savePosition(context)
            delay(500)
        }
        // Save once when playback pauses or ends.
        PlayerController.savePosition(context)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(state.title.ifBlank { "Player" }, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { pad ->
        Column(
            modifier = Modifier.padding(pad).padding(24.dp).fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(formatTime(state.positionMs) + " / " + formatTime(state.durationMs))
            Slider(
                value = state.positionMs.toFloat(),
                onValueChange = { PlayerController.seekTo(it.toLong()) },
                valueRange = 0f..(state.durationMs.coerceAtLeast(1L)).toFloat(),
                modifier = Modifier.fillMaxWidth(),
            )

            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FilledIconButton(onClick = { PlayerController.skip(-30_000) }) {
                    Icon(Icons.Default.FastRewind, contentDescription = "Back 30s")
                }
                FilledIconButton(
                    onClick = {
                        if (state.isPlaying) PlayerController.pauseAndSave(context)
                        else PlayerController.resume()
                    },
                    modifier = Modifier.size(72.dp),
                ) {
                    Icon(
                        if (state.isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                        contentDescription = if (state.isPlaying) "Pause" else "Play",
                        modifier = Modifier.size(36.dp),
                    )
                }
                FilledIconButton(onClick = { PlayerController.skip(30_000) }) {
                    Icon(Icons.Default.FastForward, contentDescription = "Forward 30s")
                }
            }

            Spacer(Modifier.height(16.dp))
            Text("Speed: ${"%.2f".format(state.speed)}×")
            Slider(
                value = state.speed,
                onValueChange = { PlayerController.setSpeed(it) },
                valueRange = 0.5f..3f,
                steps = 9,  // 0.5, 0.75, 1.0, ..., 3.0
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

private fun formatTime(ms: Long): String {
    if (ms <= 0) return "0:00"
    val total = ms / 1000
    val h = total / 3600
    val m = (total % 3600) / 60
    val s = total % 60
    return if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%d:%02d".format(m, s)
}
