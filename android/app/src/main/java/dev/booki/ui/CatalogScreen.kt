package dev.booki.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Download
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.booki.catalog.CatalogSources
import dev.booki.catalog.OpdsClient
import dev.booki.data.Library
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CatalogScreen(onBack: () -> Unit) {
    val context = LocalContextSafe.current
    val scope = rememberCoroutineScope()
    val stack = remember { mutableStateListOf<String>() }  // URL stack for back-navigation
    var feed by remember { mutableStateOf<OpdsClient.Feed?>(null) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var downloadingTitle by remember { mutableStateOf<String?>(null) }
    var downloadFrac by remember { mutableFloatStateOf(0f) }
    val snackbar = remember { SnackbarHostState() }

    fun loadUrl(url: String) {
        stack.add(url)
        loading = true; error = null; feed = null
        scope.launch {
            runCatching { OpdsClient.fetch(url) }
                .onSuccess { feed = it }
                .onFailure { error = it.message ?: "Network error" }
            loading = false
        }
    }

    fun popBack() {
        if (stack.size >= 2) {
            stack.removeAt(stack.lastIndex)
            val prev = stack.removeAt(stack.lastIndex)
            loadUrl(prev)
        } else {
            stack.clear(); feed = null; error = null
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(feed?.title ?: "Library catalog") },
                navigationIcon = {
                    IconButton(onClick = {
                        if (stack.isEmpty()) onBack() else popBack()
                    }) { Icon(Icons.Default.ArrowBack, contentDescription = "Back") }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        Box(modifier = Modifier.padding(pad).fillMaxSize()) {
            when {
                stack.isEmpty() -> SourceList(onSource = { loadUrl(it.url) })
                loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                error != null -> Column(
                    Modifier.fillMaxSize().padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text("Error: $error", color = MaterialTheme.colorScheme.error)
                    Button(onClick = { stack.lastOrNull()?.let { url ->
                        stack.removeAt(stack.lastIndex); loadUrl(url)
                    } }) { Text("Retry") }
                }
                feed != null -> EntryList(
                    entries = feed!!.entries,
                    onOpenSub = { loadUrl(it) },
                    onDownload = { entry ->
                        val url = entry.epubUrl ?: return@EntryList
                        scope.launch {
                            downloadingTitle = entry.title
                            downloadFrac = 0f
                            runCatching {
                                Library.download(context, entry.title, entry.author, url) { d, t ->
                                    downloadFrac = if (t > 0) d.toFloat() / t else 0f
                                }
                            }.onSuccess { snackbar.showSnackbar("Downloaded: ${entry.title}") }
                                .onFailure { snackbar.showSnackbar("Failed: ${it.message}") }
                            downloadingTitle = null
                        }
                    },
                )
            }

            downloadingTitle?.let { title ->
                Surface(
                    modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                    tonalElevation = 4.dp,
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Downloading: $title", style = MaterialTheme.typography.bodyMedium)
                        Spacer(Modifier.height(8.dp))
                        LinearProgressIndicator(
                            progress = { downloadFrac },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SourceList(onSource: (CatalogSources.Source) -> Unit) {
    LazyColumn(modifier = Modifier.fillMaxSize().padding(8.dp)) {
        items(CatalogSources.all) { source ->
            ListItem(
                headlineContent = { Text(source.name) },
                supportingContent = { Text(source.description) },
                modifier = Modifier.clickable { onSource(source) },
            )
            HorizontalDivider()
        }
    }
}

@Composable
private fun EntryList(
    entries: List<OpdsClient.Entry>,
    onOpenSub: (String) -> Unit,
    onDownload: (OpdsClient.Entry) -> Unit,
) {
    if (entries.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("No entries", style = MaterialTheme.typography.bodyMedium)
        }
        return
    }
    LazyColumn(modifier = Modifier.fillMaxSize()) {
        items(entries) { entry ->
            ListItem(
                headlineContent = { Text(entry.title) },
                supportingContent = {
                    Text(
                        listOfNotNull(
                            entry.author.takeIf { it.isNotBlank() },
                            entry.summary.takeIf { it.isNotBlank() }?.take(140),
                        ).joinToString(" — "),
                    )
                },
                trailingContent = {
                    when {
                        entry.isDownloadable -> IconButton(onClick = { onDownload(entry) }) {
                            Icon(Icons.Default.Download, contentDescription = "Download")
                        }
                        else -> {}
                    }
                },
                modifier = Modifier.clickable {
                    when {
                        entry.isNavigation -> entry.subFeedUrl?.let(onOpenSub)
                        entry.isDownloadable -> onDownload(entry)
                    }
                },
            )
            HorizontalDivider()
        }
    }
}
