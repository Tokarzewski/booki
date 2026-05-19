package dev.booki.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.data.Voices
import dev.booki.tts.Progress
import dev.booki.tts.SynthState
import dev.booki.tts.SynthesisService

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { BookiApp() } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BookiApp() {
    val context = LocalContextSafe.current
    var epubUri by remember { mutableStateOf<Uri?>(null) }
    var voice by remember { mutableStateOf(Voices.DEFAULT) }
    var speed by remember { mutableFloatStateOf(1f) }
    var voiceMenu by remember { mutableStateOf(false) }
    val state by Progress.state.collectAsState()

    val pickEpub = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            epubUri = uri
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("Booki") }) }) { pad ->
        Column(
            modifier = Modifier.padding(pad).padding(16.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Button(onClick = { pickEpub.launch(arrayOf("application/epub+zip")) }) {
                Text(if (epubUri == null) "Choose EPUB…" else "Change EPUB")
            }
            epubUri?.let { Text("Selected: ${it.lastPathSegment ?: it.toString()}") }

            ExposedDropdownMenuBox(expanded = voiceMenu, onExpandedChange = { voiceMenu = !voiceMenu }) {
                OutlinedTextField(
                    value = voice,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Voice") },
                    modifier = Modifier.menuAnchor().fillMaxWidth(),
                )
                ExposedDropdownMenu(expanded = voiceMenu, onDismissRequest = { voiceMenu = false }) {
                    Voices.all.forEach { v ->
                        DropdownMenuItem(
                            text = { Text("${v.id} — ${v.language}") },
                            onClick = { voice = v.id; voiceMenu = false },
                        )
                    }
                }
            }

            Column {
                Text("Speed: ${"%.2f".format(speed)}×")
                Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.5f..2f)
            }

            Button(
                enabled = epubUri != null && state !is SynthState.Running,
                onClick = {
                    val uri = epubUri ?: return@Button
                    val intent = Intent(context, SynthesisService::class.java).apply {
                        putExtra(SynthesisService.EXTRA_URI, uri)
                        putExtra(SynthesisService.EXTRA_VOICE, voice)
                        putExtra(SynthesisService.EXTRA_SPEED, speed)
                    }
                    context.startForegroundService(intent)
                },
            ) { Text("Generate audiobook") }

            when (val s = state) {
                SynthState.Idle -> {}
                is SynthState.Running -> Column {
                    Text(s.bookTitle); Text(s.chapter)
                    LinearProgressIndicator(progress = { s.fraction }, modifier = Modifier.fillMaxWidth())
                    Text("${(s.fraction * 100).toInt()}%")
                }
                is SynthState.Done -> Text("Saved: ${s.output.name}")
                is SynthState.Error -> Text("Error: ${s.message}", color = MaterialTheme.colorScheme.error)
            }
        }
    }
}
