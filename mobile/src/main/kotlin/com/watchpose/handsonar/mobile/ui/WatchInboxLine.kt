package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import com.watchpose.handsonar.mobile.transfer.InboxStatus
import com.watchpose.handsonar.mobile.transfer.Receiving
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/** 워치에서 받은 녹음 줄(전송 설계 §3): "워치 녹음 N개 받음 (마지막 HH:mm)"과, 받는 중이면 녹음·파일·진행률·남은 파일 수. */
@Composable
fun WatchInboxLine(s: InboxStatus) {
    Column {
        Text(receivedText(s), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        s.receiving?.let { Text(receivingText(it), style = MaterialTheme.typography.titleMedium, color = StatusColors.ok) }
    }
}

private fun receivedText(s: InboxStatus): String = when {
    s.received == 0 -> "워치 녹음 받은 것 없음"
    s.lastMs > 0 -> "워치 녹음 ${s.received}개 받음 (마지막 ${shortTime(s.lastMs)})"
    else -> "워치 녹음 ${s.received}개 받음"
}

private fun receivingText(r: Receiving): String {
    val percent = if (r.total > 0) (r.bytes * 100 / r.total).coerceIn(0L, 100L) else 0L
    return "워치에서 받는 중 · ${r.recording} ${r.file} $percent% · 남은 파일 ${r.filesLeft}개"
}

/** 오늘이면 "HH:mm", 아니면 "M월 d일 HH:mm". */
private fun shortTime(ms: Long): String {
    val today = Calendar.getInstance().apply {
        set(Calendar.HOUR_OF_DAY, 0)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
    }.timeInMillis
    return SimpleDateFormat(if (ms >= today) "HH:mm" else "M월 d일 HH:mm", Locale.KOREA).format(Date(ms))
}
