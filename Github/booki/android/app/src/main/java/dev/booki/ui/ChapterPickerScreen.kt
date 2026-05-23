package dev.booki.ui

import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.epub.Chapter
import dev.booki.epub.EpubReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChapterPickerScreen(
    epubUri: Uri,
    onCancel: () -> Unit,
    onConfirm: (selectedIndices: IntArray) -> Unit,
) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()

    var loading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }
    var bookTitle by remember { mutableStateOf("Loading…") }
    var chapters by remember { mutableStateOf<List<Chapter>>(emptyList()) }
    val selected = remember { mutableStateMapOf<Int, Boolean>() }

    LaunchedEffect(epubUri) {
        scope.launch {
            runCatching { withContext(Dispatchers.IO) { EpubReader.read(context, epubUri) } }
                .onSuccess { book ->
                    bookTitle = book.title
                    chapters = book.chapters
                    // Heuristic: default-skip very short chapters (TOC, copyright, dedication)
                    book.chapters.forEach { selected[it.index] = it.charCount >= 800 }
                }
                .onFailure { error = it.message ?: "Failed to read EPUB" }
            loading = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(bookTitle, maxLines = 1) },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Cancel")
                    }
                },
                actions = {
                    TextButton(onClick = { chapters.forEach { selected[it.index] = true } }) {
                        Text("All")
                    }
                    TextButton(onClick = { chapters.forEach { selected[it.index] = false } }) {
                        Text("None")
                    }
                },
            )
        },
        bottomBar = {
            val pickedCount = selected.count { it.value }
            val pickedChars = chapters.filter { selected[it.index] == true }.sumOf { it.charCount }
            BottomAppBar {
                Column(modifier = Modifier.padding(horizontal = 16.dp).weight(1f)) {
                    Text("$pickedCount of ${chapters.size} chapters")
                    Text("≈ ${pickedChars / 1000}k characters")
                }
                Button(
                    enabled = pickedCount > 0 && !loading,
                    onClick = {
                        val indices = chapters
                            .filter { selected[it.index] == true }
                            .map { it.index }
                            .toIntArray()
                        onConfirm(indices)
                    },
                ) { Text("Generate") }
                Spacer(Modifier.width(8.dp))
            }
        },
    ) { pad ->
        Box(modifier = Modifier.padding(pad).fillMaxSize()) {
            when {
                loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                error != null -> Column(
                    modifier = Modifier.fillMaxSize().padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text("Error: $error", color = MaterialTheme.colorScheme.error)
                    Button(onClick = onCancel) { Text("Back") }
                }
                else -> LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(chapters, key = { it.index }) { chapter ->
                        val isSelected = selected[chapter.index] == true
                        ListItem(
                            leadingContent = {
                                Checkbox(
                                    checked = isSelected,
                                    onCheckedChange = { selected[chapter.index] = it },
                                )
                            },
                            headlineContent = { Text(chapter.title) },
                            supportingContent = { Text("${chapter.charCount} characters") },
                            modifier = Modifier.clickable {
                                selected[chapter.index] = !isSelected
                            },
                        )
                        HorizontalDivider()
                    }
                }
            }
        }
    }
}
