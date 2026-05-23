package dev.booki.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
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

    val stack = remember { mutableStateListOf<String>() }
    var feed by remember { mutableStateOf<OpdsClient.Feed?>(null) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var downloadingTitle by remember { mutableStateOf<String?>(null) }
    var downloadFrac by remember { mutableFloatStateOf(0f) }
    var addDialog by remember { mutableStateOf(false) }
    var sources by remember { mutableStateOf(CatalogSources.load(context)) }
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
                    IconButton(onClick = { if (stack.isEmpty()) onBack() else popBack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (stack.isEmpty()) {
                        IconButton(onClick = { addDialog = true }) {
                            Icon(Icons.Default.Add, contentDescription = "Add feed")
                        }
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        Box(modifier = Modifier.padding(pad).fillMaxSize()) {
            when {
                stack.isEmpty() -> SourceList(
                    sources = sources,
                    onSource = { loadUrl(it.url) },
                    onDelete = { src ->
                        CatalogSources.remove(context, src)
                        sources = CatalogSources.load(context)
                    },
                )
                loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                error != null -> Column(
                    Modifier.fillMaxSize().padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text("Error: $error", color = MaterialTheme.colorScheme.error)
                    Button(onClick = {
                        stack.lastOrNull()?.let { url ->
                            stack.removeAt(stack.lastIndex); loadUrl(url)
                        }
                    }) { Text("Retry") }
                }
                feed != null -> FeedView(
                    feed = feed!!,
                    onOpenSub = { loadUrl(it) },
                    onSearch = { query ->
                        loading = true; error = null
                        scope.launch {
                            runCatching { OpdsClient.search(feed!!, query) }
                                .onSuccess { result ->
                                    stack.add("search:$query")
                                    feed = result
                                }
                                .onFailure { error = it.message ?: "Search failed" }
                            loading = false
                        }
                    },
                    onDownload = { entry ->
                        val url = entry.epubUrl ?: return@FeedView
                        scope.launch {
                            downloadingTitle = entry.title; downloadFrac = 0f
                            runCatching {
                                Library.download(context, entry.title, entry.author, url) { d, t ->
                                    downloadFrac = if (t > 0) d.toFloat() / t else 0f
                                }
                            }.onSuccess { snackbar.showSnackbar("Downloaded: ${entry.title}") }
                                .onFailure { snackbar.showSnackbar("Failed: ${it.message}") }
                            downloadingTitle = null
                        }
                    },
                    onNextPage = { feed!!.nextPage?.let { loadUrl(it) } },
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

    if (addDialog) {
        AddFeedDialog(
            onDismiss = { addDialog = false },
            onAdd = { name, url ->
                CatalogSources.add(context, name, url)
                sources = CatalogSources.load(context)
                addDialog = false
            },
        )
    }
}

@Composable
private fun SourceList(
    sources: List<CatalogSources.Source>,
    onSource: (CatalogSources.Source) -> Unit,
    onDelete: (CatalogSources.Source) -> Unit,
) {
    LazyColumn(modifier = Modifier.fillMaxSize().padding(vertical = 8.dp)) {
        items(sources) { source ->
            ListItem(
                headlineContent = { Text(source.name) },
                supportingContent = { Text(source.description) },
                trailingContent = if (source.isCustom) {
                    {
                        IconButton(onClick = { onDelete(source) }) {
                            Icon(Icons.Default.Delete, contentDescription = "Remove")
                        }
                    }
                } else null,
                modifier = Modifier.clickable { onSource(source) },
            )
            HorizontalDivider()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FeedView(
    feed: OpdsClient.Feed,
    onOpenSub: (String) -> Unit,
    onSearch: (String) -> Unit,
    onDownload: (OpdsClient.Entry) -> Unit,
    onNextPage: () -> Unit,
) {
    var query by remember { mutableStateOf("") }
    Column(Modifier.fillMaxSize()) {
        if (feed.searchTemplate != null) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("Search this catalog") },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                trailingIcon = if (query.isNotEmpty()) {
                    { IconButton(onClick = { query = "" }) { Icon(Icons.Default.Close, contentDescription = "Clear") } }
                } else null,
                singleLine = true,
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = ImeAction.Search),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                    onSearch = { if (query.isNotBlank()) onSearch(query.trim()) }),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
            )
        }
        if (feed.entries.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No entries", style = MaterialTheme.typography.bodyMedium)
            }
            return@Column
        }
        LazyColumn(modifier = Modifier.fillMaxSize()) {
            items(feed.entries) { entry ->
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
                        if (entry.isDownloadable) {
                            IconButton(onClick = { onDownload(entry) }) {
                                Icon(Icons.Default.Download, contentDescription = "Download")
                            }
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
            if (feed.nextPage != null) {
                item {
                    TextButton(onClick = onNextPage, modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                        Text("Load more →")
                    }
                }
            }
        }
    }
}

@Composable
private fun AddFeedDialog(onDismiss: () -> Unit, onAdd: (name: String, url: String) -> Unit) {
    var name by remember { mutableStateOf("") }
    var url by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add OPDS feed") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = name, onValueChange = { name = it },
                    label = { Text("Name") }, singleLine = true,
                )
                OutlinedTextField(
                    value = url, onValueChange = { url = it },
                    label = { Text("OPDS URL") }, singleLine = true,
                    placeholder = { Text("https://…/opds") },
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank() && url.startsWith("http"),
                onClick = { onAdd(name.trim(), url.trim()) },
            ) { Text("Add") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
