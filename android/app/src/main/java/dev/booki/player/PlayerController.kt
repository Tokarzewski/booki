package dev.booki.player

import android.content.ComponentName
import android.content.Context
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import com.google.common.util.concurrent.MoreExecutors
import dev.booki.data.AudioLibrary
import dev.booki.data.PlaybackPositions
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * Singleton that connects to [PlayerService] and exposes Compose-friendly
 * state. Survives Activity recreation.
 */
@UnstableApi
object PlayerController {
    data class State(
        val title: String = "",
        val isPlaying: Boolean = false,
        val positionMs: Long = 0L,
        val durationMs: Long = 0L,
        val speed: Float = 1f,
    ) {
        val fraction: Float
            get() = if (durationMs <= 0) 0f else (positionMs.toFloat() / durationMs).coerceIn(0f, 1f)
    }

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state

    private var controller: MediaController? = null

    suspend fun connect(context: Context) {
        if (controller != null) return
        val token = SessionToken(context, ComponentName(context, PlayerService::class.java))
        controller = suspendCancellableCoroutine { cont ->
            val future = MediaController.Builder(context, token).buildAsync()
            future.addListener({ cont.resume(future.get()) }, MoreExecutors.directExecutor())
            cont.invokeOnCancellation { future.cancel(true) }
        }
        controller?.addListener(object : Player.Listener {
            override fun onIsPlayingChanged(isPlaying: Boolean) = refresh()
            override fun onPositionDiscontinuity(o: Player.PositionInfo, n: Player.PositionInfo, r: Int) = refresh()
            override fun onMediaMetadataChanged(metadata: MediaMetadata) = refresh()
            override fun onPlaybackStateChanged(state: Int) = refresh()
            override fun onPlaybackParametersChanged(params: androidx.media3.common.PlaybackParameters) = refresh()
        })
        refresh()
    }

    fun release() {
        controller?.release()
        controller = null
    }

    /** Polled from the UI to advance the position bar without callback churn. */
    fun refresh() {
        val c = controller ?: return
        _state.value = State(
            title = c.mediaMetadata.title?.toString().orEmpty(),
            isPlaying = c.isPlaying,
            positionMs = c.currentPosition.coerceAtLeast(0L),
            durationMs = c.duration.coerceAtLeast(0L),
            speed = c.playbackParameters.speed,
        )
    }

    private var currentPath: String? = null

    fun play(context: Context, item: AudioLibrary.Audiobook, fromStart: Boolean = false) {
        val c = controller ?: return
        val uri = AudioLibrary.uriFor(context, item)
        val mediaItem = MediaItem.Builder()
            .setUri(uri)
            .setMediaMetadata(MediaMetadata.Builder().setTitle(item.title).build())
            .build()
        c.setMediaItem(mediaItem)
        c.prepare()
        currentPath = item.file.path
        val startMs = if (fromStart) 0L else PlaybackPositions.positionMs(context, item.file.path)
        if (startMs > 0L) c.seekTo(startMs)
        c.playWhenReady = true
        refresh()
    }

    /** Persists the current playhead. Call when pausing or every few seconds while playing. */
    fun savePosition(context: Context) {
        val c = controller ?: return
        val path = currentPath ?: return
        if (c.duration > 0 && c.currentPosition < c.duration - 5_000) {
            PlaybackPositions.save(context, path, c.currentPosition)
        } else if (c.duration > 0) {
            // Reached the end — drop the bookmark.
            PlaybackPositions.clear(context, path)
        }
    }

    fun pause() { controller?.pause(); refresh() }
    fun pauseAndSave(context: Context) { pause(); savePosition(context) }
    fun resume() { controller?.play(); refresh() }
    fun seekTo(ms: Long) { controller?.seekTo(ms); refresh() }
    fun setSpeed(factor: Float) {
        controller?.setPlaybackSpeed(factor.coerceIn(0.5f, 3f))
        refresh()
    }
    fun skip(deltaMs: Long) {
        val c = controller ?: return
        c.seekTo((c.currentPosition + deltaMs).coerceIn(0L, c.duration.coerceAtLeast(0L)))
        refresh()
    }
}
