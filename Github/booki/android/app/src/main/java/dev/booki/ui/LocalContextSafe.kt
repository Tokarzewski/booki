package dev.booki.ui

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.platform.LocalContext

/** Re-export of [LocalContext] so call-sites don't need to import androidx.compose.ui.platform. */
val LocalContextSafe = LocalContext
