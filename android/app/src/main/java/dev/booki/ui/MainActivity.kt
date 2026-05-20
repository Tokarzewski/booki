package dev.booki.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.LibraryBooks
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Headphones
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.data.AudioLibrary
import dev.booki.data.Library
import dev.booki.data.Settings
import dev.booki.data.Settings.defaultSpeed
import dev.booki.data.Settings.defaultVoice
import dev.booki.data.Voices
import dev.booki.player.PlayerController
import dev.booki.tts.ModelDownloader
import dev.booki.tts.Progress
import dev.booki.tts.SynthState
import dev.booki.tts.SynthesisService

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Root() } }
    }
}

private sealed interface Screen {
    data object Home : Screen
    data object Catalog : Screen
    data object Player : Screen
    data object Settings : Screen
    data class Picker(val uri: Uri, val voice: String, val speed: Float, val streamLive: Boolean) : Screen
}

@Composable
private fun Root() {
    val context = LocalContextSafe.current
    var provisioned by remember { mutableStateOf(ModelDownloader.anyProvisioned(context)) }
    var screen: Screen by remember { mutableStateOf(Screen.Home) }

    LaunchedEffect(Unit) {
        Library.refresh(context)
        AudioLibrary.refresh(context)
    }

    val synthState by Progress.state.collectAsState()
    LaunchedEffect(synthState) {
        if (synthState is SynthState.Done) AudioLibrary.refresh(context)
    }

    if (!provisioned) {
        SetupScreen(onProvisioned = { provisioned = true })
        return
    }

    when (val s = screen) {
        Screen.Home -> BookiApp(
            onOpenCatalog = { screen = Screen.Catalog },
            onOpenPlayer = { screen = Screen.Player },
            onOpenSettings = { screen = Screen.Settings },
            onOpenPicker = { uri, voice, speed, streamLive ->
                screen = Screen.Picker(uri, voice, speed, streamLive)
            },
        )
        Screen.Catalog -> CatalogScreen(onBack = {
            Library.refresh(context); screen = Screen.Home
        })
        Screen.Player -> PlayerScreen(onBack = { screen = Screen.Home })
        Screen.Settings -> SettingsScreen(onBack = { screen = Screen.Home })
        is Screen.Picker -> ChapterPickerScreen(
            epubUri = s.uri,
            onCancel = { screen = Screen.Home },
            onConfirm = { indices ->
                val intent = Intent(context, SynthesisService::class.java).apply {
                    putExtra(SynthesisService.EXTRA_URI, s.uri)
                    putExtra(SynthesisService.EXTRA_VOICE, s.voice)
                    putExtra(SynthesisService.EXTRA_SPEED, s.speed)
                    putExtra(SynthesisService.EXTRA_STREAM_LIVE, s.streamLive)
                    putExtra(SynthesisService.EXTRA_CHAPTERS, indices)
                }
                context.startForegroundService(intent)
                screen = Screen.Home
            },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BookiApp(
    onOpenCatalog: () -> Unit,
    onOpenPlayer: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenPicker: (Uri, voice: String, speed: Float, streamLive: Boolean) -> Unit,
) {
    val context = LocalContextSafe.current
    var epubUri by remember { mutableStateOf<Uri?>(null) }
    var epubLabel by remember { mutableStateOf<String?>(null) }
    var voice by remember { mutableStateOf(with(Settings) { context.defaultVoice }) }
    var voiceFilter by remember { mutableStateOf("") }
    var speed by remember { mutableFloatStateOf(with(Settings) { context.defaultSpeed }) }
    var streamLive by remember { mutableStateOf(true) }
    var voiceMenu by remember { mutableStateOf(false) }
    var renameTarget by remember { mutableStateOf<Library.Book?>(null) }
    var deleteTarget by remember { mutableStateOf<Library.Book?>(null) }
    var renameAudio by remember { mutableStateOf<AudioLibrary.Audiobook?>(null) }
    var deleteAudio by remember { mutableStateOf<AudioLibrary.Audiobook?>(null) }
    val state by Progress.state.collectAsState()
    val library by Library.books.collectAsState()
    val audiobooks by AudioLibrary.items.collectAsState()

    LaunchedEffect(Unit) { PlayerController.connect(context) }

    val pickEpub = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            epubUri = uri
            epubLabel = uri.lastPathSegment
        }
    }

    val filteredVoices = remember(voiceFilter) {
        if (voiceFilter.isBlank()) Voices.all
        else Voices.all.filter { v ->
            v.id.contains(voiceFilter, ignoreCase = true) ||
                v.language.contains(voiceFilter, ignoreCase = true)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Booki") },
                actions = {
                    IconButton(onClick = onOpenPlayer) {
                        Icon(Icons.Default.Headphones, contentDescription = "Player")
                    }
                    IconButton(onClick = onOpenCatalog) {
                        Icon(Icons.Default.Public, contentDescription = "Catalog")
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
    ) { pad ->
        Column(
            modifier = Modifier.padding(pad).padding(16.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { pickEpub.launch(arrayOf("application/epub+zip")) }) {
                    Text("Pick EPUB…")
                }
                OutlinedButton(onClick = onOpenCatalog) {
                    Icon(Icons.Default.Public, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Browse catalog")
                }
            }
            epubLabel?.let { Text("Selected: $it") }

            if (audiobooks.isNotEmpty()) {
                Text("Audiobooks", style = MaterialTheme.typography.titleMedium)
                Surface(tonalElevation = 1.dp, modifier = Modifier.fillMaxWidth()) {
                    LazyColumn(modifier = Modifier.heightIn(max = 240.dp)) {
                        items(audiobooks, key = { it.file.path }) { item ->
                            var menuOpen by remember(item.file.path) { mutableStateOf(false) }
                            ListItem(
                                leadingContent = {
                                    IconButton(onClick = {
                                        PlayerController.play(context, item)
                                        onOpenPlayer()
                                    }) {
                                        Icon(Icons.Default.PlayArrow, contentDescription = "Play")
                                    }
                                },
                                headlineContent = { Text(item.title) },
                                supportingContent = { Text("${item.sizeMb} MB") },
                                trailingContent = {
                                    Box {
                                        IconButton(onClick = { menuOpen = true }) {
                                            Icon(Icons.Default.MoreVert, contentDescription = "More")
                                        }
                                        DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                                            DropdownMenuItem(
                                                text = { Text("Rename") },
                                                leadingIcon = { Icon(Icons.Default.Edit, null) },
                                                onClick = { menuOpen = false; renameAudio = item },
                                            )
                                            DropdownMenuItem(
                                                text = { Text("Delete") },
                                                leadingIcon = { Icon(Icons.Default.Delete, null) },
                                                onClick = { menuOpen = false; deleteAudio = item },
                                            )
                                        }
                                    }
                                },
                                modifier = Modifier.clickable {
                                    PlayerController.play(context, item)
                                    onOpenPlayer()
                                },
                            )
                            HorizontalDivider()
                        }
                    }
                }
            }

            if (library.isNotEmpty()) {
                Text("EPUB library", style = MaterialTheme.typography.titleMedium)
                Surface(tonalElevation = 1.dp, modifier = Modifier.fillMaxWidth()) {
                    LazyColumn(modifier = Modifier.heightIn(max = 240.dp)) {
                        items(library, key = { it.file.path }) { book ->
                            var menuOpen by remember(book.file.path) { mutableStateOf(false) }
                            ListItem(
                                leadingContent = {
                                    Icon(Icons.AutoMirrored.Filled.LibraryBooks, contentDescription = null)
                                },
                                headlineContent = { Text(book.title) },
                                supportingContent = { Text("${book.sizeKb} KB") },
                                trailingContent = {
                                    Box {
                                        IconButton(onClick = { menuOpen = true }) {
                                            Icon(Icons.Default.MoreVert, contentDescription = "More")
                                        }
                                        DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                                            DropdownMenuItem(
                                                text = { Text("Rename") },
                                                leadingIcon = { Icon(Icons.Default.Edit, null) },
                                                onClick = { menuOpen = false; renameTarget = book },
                                            )
                                            DropdownMenuItem(
                                                text = { Text("Delete") },
                                                leadingIcon = { Icon(Icons.Default.Delete, null) },
                                                onClick = { menuOpen = false; deleteTarget = book },
                                            )
                                        }
                                    }
                                },
                                modifier = Modifier.clickable {
                                    epubUri = Library.uriFor(context, book)
                                    epubLabel = book.title
                                },
                            )
                            HorizontalDivider()
                        }
                    }
                }
            }

            ExposedDropdownMenuBox(expanded = voiceMenu, onExpandedChange = { voiceMenu = !voiceMenu }) {
                OutlinedTextField(
                    value = voice,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Voice") },
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
                            onClick = { voice = v.id; voiceMenu = false; voiceFilter = "" },
                        )
                    }
                }
            }

            Column {
                Text("Speed: ${"%.2f".format(speed)}×")
                Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.5f..2f)
            }

            Row(
                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Switch(checked = streamLive, onCheckedChange = { streamLive = it })
                Column {
                    Text("Play while generating", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "Streams audio to the speaker as it's synthesized. Continues with screen off.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    enabled = epubUri != null && state !is SynthState.Running,
                    onClick = {
                        val uri = epubUri ?: return@Button
                        val intent = Intent(context, SynthesisService::class.java).apply {
                            putExtra(SynthesisService.EXTRA_URI, uri)
                            putExtra(SynthesisService.EXTRA_VOICE, voice)
                            putExtra(SynthesisService.EXTRA_SPEED, speed)
                            putExtra(SynthesisService.EXTRA_STREAM_LIVE, streamLive)
                        }
                        context.startForegroundService(intent)
                    },
                ) { Text(if (streamLive) "Generate + play" else "Generate") }

                OutlinedButton(
                    enabled = epubUri != null && state !is SynthState.Running,
                    onClick = { epubUri?.let { onOpenPicker(it, voice, speed, streamLive) } },
                ) { Text("Pick chapters…") }
            }

            if (state is SynthState.Running) {
                OutlinedButton(onClick = {
                    val intent = Intent(context, SynthesisService::class.java)
                        .setAction(SynthesisService.ACTION_STOP)
                    context.startService(intent)
                }) { Text("Stop") }
            }

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

    renameTarget?.let { book ->
        var newName by remember(book.file.path) { mutableStateOf(book.title) }
        AlertDialog(
            onDismissRequest = { renameTarget = null },
            title = { Text("Rename EPUB") },
            text = {
                OutlinedTextField(
                    value = newName, onValueChange = { newName = it },
                    singleLine = true, modifier = Modifier.fillMaxWidth(),
                )
            },
            confirmButton = {
                TextButton(
                    enabled = newName.isNotBlank() && newName != book.title,
                    onClick = { Library.rename(context, book, newName); renameTarget = null },
                ) { Text("Rename") }
            },
            dismissButton = { TextButton(onClick = { renameTarget = null }) { Text("Cancel") } },
        )
    }

    deleteTarget?.let { book ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Delete \"${book.title}\"?") },
            text = { Text("This removes the local EPUB. The synthesized audiobook is not affected.") },
            confirmButton = {
                TextButton(onClick = { Library.delete(context, book); deleteTarget = null }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("Cancel") } },
        )
    }

    renameAudio?.let { item ->
        var newName by remember(item.file.path) { mutableStateOf(item.title) }
        AlertDialog(
            onDismissRequest = { renameAudio = null },
            title = { Text("Rename audiobook") },
            text = {
                OutlinedTextField(
                    value = newName, onValueChange = { newName = it },
                    singleLine = true, modifier = Modifier.fillMaxWidth(),
                )
            },
            confirmButton = {
                TextButton(
                    enabled = newName.isNotBlank() && newName != item.title,
                    onClick = { AudioLibrary.rename(context, item, newName); renameAudio = null },
                ) { Text("Rename") }
            },
            dismissButton = { TextButton(onClick = { renameAudio = null }) { Text("Cancel") } },
        )
    }

    deleteAudio?.let { item ->
        AlertDialog(
            onDismissRequest = { deleteAudio = null },
            title = { Text("Delete \"${item.title}\"?") },
            text = { Text("Removes the audiobook .m4a file.") },
            confirmButton = {
                TextButton(onClick = { AudioLibrary.delete(context, item); deleteAudio = null }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleteAudio = null }) { Text("Cancel") } },
        )
    }
}
