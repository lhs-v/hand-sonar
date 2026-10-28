package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.mobile.export.ExportStatus
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.session.SessionSummary
import com.watchpose.handsonar.mobile.session.UiState
import com.watchpose.handsonar.mobile.transfer.InboxStatus
import com.watchpose.handsonar.shared.SessionMeta
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/**
 * 기록 화면(설계 §6.3, 이번 계획 범위): 보관·휴지통 목록, 세션마다 시각·길이·손 인식·워치 처프·표지음·의심, 휴지통으로·되살리기·완전 삭제
 * (확인 창 뒤, 휴지통에서만), 오늘과 전체 수집 시간(보관한 세션), 워치에서 받은 녹음 줄([inbox], 전송 설계 §3).
 * "내보내기"(전송 설계 §5)는 확인 창에서 "전부"·"새 것만"을 고르면 [onExport]를 부르고, 진행률과 결과([export])를 보인다.
 * 영상 재생·타임라인·메모는 다음 폰 계획.
 */
@Composable
fun RecordsScreen(ui: UiState, c: SessionController, inbox: InboxStatus, export: ExportStatus, onExport: (onlyNew: Boolean) -> Unit) {
    var deleteTarget by remember { mutableStateOf<SessionSummary?>(null) }
    var askExport by remember { mutableStateOf(false) }
    val kept = ui.records.filter { it.status != SessionMeta.STATUS_TRASH }
    val trash = ui.records.filter { it.status == SessionMeta.STATUS_TRASH }
    val todayStart = remember { startOfToday() }
    val todayMin = kept.filter { it.startWallMs >= todayStart }.sumOf { it.durationSec } / 60
    val totalMin = kept.sumOf { it.durationSec } / 60
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            TextButton(onClick = c::backToReady) { Text("← 돌아가기", style = MaterialTheme.typography.titleLarge) }
            Text("기록", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.weight(1f))
            Text("오늘 %.1f분 · 전체 %.1f분 (보관한 세션)".format(todayMin, totalMin), style = MaterialTheme.typography.titleLarge)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            FilterChip(selected = !ui.showTrash, onClick = { c.setShowTrash(false) }, label = { Text("보관 ${kept.size}") })
            FilterChip(selected = ui.showTrash, onClick = { c.setShowTrash(true) }, label = { Text("휴지통 ${trash.size}") })
            Spacer(Modifier.weight(1f))
            OutlinedButton(onClick = { askExport = true }, enabled = !export.busy) {
                Text(if (export.busy) "내보내는 중 ${(export.progress * 100).roundToInt()}%" else "내보내기")
            }
        }
        WatchInboxLine(inbox)
        export.message?.let { if (export.error) WarningText(it) else Text(it, style = MaterialTheme.typography.titleMedium) }
        ui.error?.let { WarningText(it) }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(if (ui.showTrash) trash else kept, key = { it.name }) { s ->
                RecordRow(s, onTrash = { c.trash(s.dir) }, onRestore = { c.restore(s.dir) }, onDelete = { deleteTarget = s })
            }
        }
    }
    deleteTarget?.let { s ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            confirmButton = {
                TextButton(onClick = {
                    deleteTarget = null
                    c.deleteForever(s.dir)
                }) { Text("완전 삭제", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("그만두기") } },
            title = { Text("완전히 지울까요?") },
            text = { Text("${s.name} 세션의 파일을 모두 지웁니다. 되돌릴 수 없습니다.") },
        )
    }
    if (askExport) {
        AlertDialog(
            onDismissRequest = { askExport = false },
            confirmButton = {
                Row {
                    TextButton(onClick = {
                        askExport = false
                        onExport(false)
                    }) { Text("전부") }
                    TextButton(onClick = {
                        askExport = false
                        onExport(true)
                    }) { Text("새 것만") }
                }
            },
            dismissButton = { TextButton(onClick = { askExport = false }) { Text("그만두기") } },
            title = { Text("내보내기") },
            text = {
                Text(
                    "보관한 세션과 받은 워치 녹음을 Download/HandSonar/ 아래 zip 하나로 저장합니다. PC의 hand-sonar-data에 그대로 풀면 됩니다. " +
                        "'새 것만'은 지난번 내보낸 뒤 파일 수·크기·시각이 그대로인 폴더를 뺍니다.",
                )
            },
        )
    }
}

@Composable
private fun RecordRow(s: SessionSummary, onTrash: () -> Unit, onRestore: () -> Unit, onDelete: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Row(
            Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Column(Modifier.weight(1f)) {
                Text(dateTime(s.startWallMs), style = MaterialTheme.typography.titleLarge)
                Text(
                    "${mmss(s.durationSec.toInt())} · 손 ${percent(s.handRatio)} · 워치 ${percent(s.watchRatio)} · " +
                        "표지음 ${mark(s.startMarker)}${mark(s.endMarker)}",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            if (s.cancelled) Text("취소함", color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (s.suspect) Text("의심", color = StatusColors.warn, style = MaterialTheme.typography.titleMedium)
            if (s.status == SessionMeta.STATUS_TRASH) {
                OutlinedButton(onClick = onRestore) { Text("되살리기") }
                TextButton(onClick = onDelete) { Text("완전 삭제", color = MaterialTheme.colorScheme.error) }
            } else {
                OutlinedButton(onClick = onTrash) { Text("휴지통으로") }
            }
        }
    }
}

private fun mark(found: Boolean?): String = when (found) {
    true -> "✓"
    false -> "✗"
    null -> "—"
}

private fun dateTime(ms: Long): String = SimpleDateFormat("M월 d일 HH:mm:ss", Locale.KOREA).format(Date(ms))

private fun startOfToday(): Long = Calendar.getInstance().apply {
    set(Calendar.HOUR_OF_DAY, 0)
    set(Calendar.MINUTE, 0)
    set(Calendar.SECOND, 0)
    set(Calendar.MILLISECOND, 0)
}.timeInMillis
