package dev.booki.tts

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

sealed interface SynthState {
    data object Idle : SynthState
    data class Running(
        val bookTitle: String,
        val chapter: String,
        val processed: Int,
        val total: Int,
    ) : SynthState {
        val fraction: Float get() = if (total == 0) 0f else processed.toFloat() / total
    }
    data class Done(val output: File) : SynthState
    data class Error(val message: String) : SynthState
}

object Progress {
    private val _state = MutableStateFlow<SynthState>(SynthState.Idle)
    val state: StateFlow<SynthState> = _state

    private var bookTitle: String = ""
    private var chapter: String = ""
    private var total: Int = 0

    fun start(title: String, totalChars: Int) {
        bookTitle = title; total = totalChars; chapter = ""
        _state.value = SynthState.Running(title, "", 0, totalChars)
    }
    fun chapter(name: String) {
        chapter = name
        val cur = _state.value as? SynthState.Running ?: return
        _state.value = cur.copy(chapter = name)
    }
    fun tick(processed: Int, totalChars: Int) {
        total = totalChars
        _state.value = SynthState.Running(bookTitle, chapter, processed, totalChars)
    }
    fun done(file: File) { _state.value = SynthState.Done(file) }
    fun error(msg: String) { _state.value = SynthState.Error(msg) }
    fun reset() { _state.value = SynthState.Idle }
}
