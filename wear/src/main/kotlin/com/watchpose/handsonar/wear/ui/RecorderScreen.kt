package com.watchpose.handsonar.wear.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Build
import android.os.SystemClock
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material3.AppScaffold
import androidx.wear.compose.material3.Button
import androidx.wear.compose.material3.FilledTonalButton
import androidx.wear.compose.material3.ListHeader
import androidx.wear.compose.material3.MaterialTheme
import androidx.wear.compose.material3.ScreenScaffold
import androidx.wear.compose.material3.Text
import com.watchpose.handsonar.shared.SonarSpec
import com.watchpose.handsonar.shared.StorageBudget
import com.watchpose.handsonar.shared.SweepWindow
import com.watchpose.handsonar.wear.HandSonarApp
import com.watchpose.handsonar.wear.record.RecorderService
import com.watchpose.handsonar.wear.record.RecorderState
import com.watchpose.handsonar.wear.record.RecorderStatus
import com.watchpose.handsonar.wear.record.SonarSettings
import com.watchpose.handsonar.wear.transfer.TransferState
import com.watchpose.handsonar.wear.transfer.TransferStatus
import kotlinx.coroutines.delay
import java.util.Locale
import kotlin.math.roundToInt

/** 녹음하지 않을 때 남은 녹음·볼륨을 다시 읽는 주기. */
private const val IDLE_REFRESH_MS = 5_000L

/** 녹음하지 않을 때 보이는 값. 녹음 중에는 서비스가 [RecorderStatus]로 준다. */
private data class IdleInfo(val minutesLeft: Double, val volumeIndex: Int, val volumeMax: Int)

@Composable
fun RecorderApp(app: HandSonarApp) {
    MaterialTheme {
        AppScaffold {
            val context = LocalContext.current
            val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                if (granted(context, Manifest.permission.RECORD_AUDIO)) {
                    RecorderService.start(context)
                } else {
                    RecorderState.update { s -> s.copy(error = "마이크 권한이 있어야 녹음할 수 있습니다") }
                }
            }
            val onStart: () -> Unit = {
                val missing = buildList {
                    add(Manifest.permission.RECORD_AUDIO)
                    if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
                }.filterNot { granted(context, it) }
                if (missing.isEmpty()) RecorderService.start(context) else launcher.launch(missing.toTypedArray())
            }
            RecorderScreen(app.settings, onStart = onStart, onStop = { RecorderService.stop(context) })
        }
    }
}

private fun granted(context: Context, permission: String): Boolean =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

/**
 * 워치의 유일한 화면(설계 §5.1): 시작·정지, 녹음 시간, 남은 녹음, 볼륨, 재생 끊김, 처프 창·세기, 자동 정지(녹음 중에는 남은 시간).
 * 창·세기·자동 정지는 녹음하지 않을 때만 바꾼다. 녹음하지 않을 때는 폰으로 보내기 상태 한 줄이 있다(전송 설계 §2).
 */
@Composable
fun RecorderScreen(settings: SonarSettings, onStart: () -> Unit, onStop: () -> Unit) {
    val status by RecorderState.status.collectAsStateWithLifecycle()
    val transfer by TransferState.status.collectAsStateWithLifecycle()
    val values by settings.state.collectAsStateWithLifecycle()
    val idle = rememberIdleInfo()
    val now by produceState(SystemClock.elapsedRealtime()) {
        while (true) {
            value = SystemClock.elapsedRealtime()
            delay(1_000)
        }
    }
    val listState = rememberScalingLazyListState()
    ScreenScaffold(scrollState = listState) { contentPadding ->
        ScalingLazyColumn(state = listState, contentPadding = contentPadding, modifier = Modifier.fillMaxSize()) {
            item { ListHeader { Text("손 소나") } }
            item {
                Button(
                    onClick = if (status.running) onStop else onStart,
                    modifier = Modifier.fillMaxWidth(),
                    secondaryLabel = { Text(if (status.running) runningLabel(now, status) else "18–21 kHz 처프 + 녹음") },
                    label = { Text(if (status.running) "정지" else "녹음 시작") },
                )
            }
            item { Details(status, idle, transfer) }
            if (!status.running) {
                item {
                    FilledTonalButton(
                        onClick = { settings.setWindow(if (values.window == SweepWindow.NONE) SweepWindow.HANN else SweepWindow.NONE) },
                        modifier = Modifier.fillMaxWidth(),
                        secondaryLabel = { Text(if (values.window == SweepWindow.NONE) "창 없음" else "한 창") },
                        label = { Text("처프 창") },
                    )
                }
                item {
                    FilledTonalButton(
                        onClick = settings::cycleAmplitude,
                        modifier = Modifier.fillMaxWidth(),
                        secondaryLabel = { Text("누르면 10%씩, 100% 다음은 10%") },
                        label = { Text("세기 ${(values.amplitude * 100).roundToInt()}%") },
                    )
                }
                item {
                    FilledTonalButton(
                        onClick = settings::cycleAutoStop,
                        modifier = Modifier.fillMaxWidth(),
                        secondaryLabel = { Text(if (values.autoStopMinutes == 0) "끔" else "${values.autoStopMinutes}분 뒤 정지") },
                        label = { Text("자동 정지") },
                    )
                }
            }
        }
    }
}

@Composable
private fun rememberIdleInfo(): IdleInfo {
    val context = LocalContext.current
    val info by produceState(IdleInfo(Double.NaN, -1, -1)) {
        val am = context.getSystemService(AudioManager::class.java)
        while (true) {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            value = IdleInfo(
                StorageBudget.minutesLeft(dir.usableSpace),
                am.getStreamVolume(AudioManager.STREAM_MUSIC),
                am.getStreamMaxVolume(AudioManager.STREAM_MUSIC),
            )
            delay(IDLE_REFRESH_MS)
        }
    }
    return info
}

@Composable
private fun Details(s: RecorderStatus, idle: IdleInfo, transfer: TransferStatus) {
    val dim = MaterialTheme.colorScheme.onSurfaceVariant
    val small = MaterialTheme.typography.bodySmall
    val minutesLeft = if (s.running) s.minutesLeft else idle.minutesLeft
    val warn = if (s.running) s.storageWarning else minutesLeft < StorageBudget.WARN_MINUTES
    val volumeIndex = if (s.running) s.volumeIndex else idle.volumeIndex
    val volumeMax = if (s.running) s.volumeMax else idle.volumeMax
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        if (s.running) {
            Text("녹음 ${minutes(s.recordedFrames)} · 재생 끊김 ${s.underruns}번", style = small, textAlign = TextAlign.Center)
        }
        if (!minutesLeft.isNaN()) {
            Text(
                "남은 녹음 ${minutesLeft.roundToInt()}분",
                style = small,
                color = if (warn) MaterialTheme.colorScheme.error else dim,
                textAlign = TextAlign.Center,
            )
        }
        if (volumeMax > 0) {
            val silent = volumeIndex == 0 // 볼륨 0이면 처프가 안 나가 그 녹음을 못 쓴다: 녹음 전에도 중에도 오류 색으로 알린다
            Text(
                if (silent) "미디어 볼륨 0/$volumeMax · 처프가 안 나감" else "미디어 볼륨 $volumeIndex/$volumeMax",
                style = small,
                color = if (silent) MaterialTheme.colorScheme.error else dim,
                textAlign = TextAlign.Center,
            )
        }
        if (s.folder.isNotEmpty()) Text("폴더 ${s.folder}", style = small, color = dim, textAlign = TextAlign.Center)
        if (!s.running) transferLine(transfer)?.let { Text(it, style = small, color = dim, textAlign = TextAlign.Center) }
        s.error?.let { Text(it, style = small, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center) }
    }
}

/** 녹음하지 않을 때의 폰으로 보내기 한 줄(전송 설계 §2). 할 말이 없으면 null. */
private fun transferLine(t: TransferStatus): String? = when {
    t.active && t.file.isNotEmpty() -> {
        val percent = if (t.total > 0) (t.bytes * 100 / t.total).coerceIn(0L, 100L) else 0L
        "폰으로 보내는 중 ${t.file} $percent% · 남은 녹음 ${t.pending}개"
    }
    t.active -> "폰으로 보내는 중 · 남은 녹음 ${t.pending}개"
    t.message != null -> t.message
    t.pending > 0 -> "폰으로 보낼 녹음 ${t.pending}개"
    else -> null
}

private fun clock(ms: Long): String {
    val total = (ms / 1000).coerceAtLeast(0)
    return String.format(Locale.ROOT, "%02d:%02d", total / 60, total % 60)
}

/**
 * 녹음 중 정지 버튼의 보조 글. 자동 정지가 없으면 지난 시간만, 있으면 "지난 시간 · 남은 m:ss"다.
 * 남은 시간은 올림해서 지난 시간(내림)과 더하면 정한 시간이 되고, 시각이 지나도 멈추기 전까지는 음수 대신 0:00을 보인다.
 */
private fun runningLabel(now: Long, s: RecorderStatus): String {
    val elapsed = clock(now - s.startedElapsedMs)
    if (s.autoStopAtElapsedMs <= 0L) return elapsed
    val left = ((s.autoStopAtElapsedMs - now).coerceAtLeast(0) + 999) / 1000
    return String.format(Locale.ROOT, "%s · 남은 %d:%02d", elapsed, left / 60, left % 60)
}

private fun minutes(frames: Long): String = String.format(Locale.ROOT, "%.1f분", frames.toDouble() / SonarSpec.SAMPLE_RATE / 60)
