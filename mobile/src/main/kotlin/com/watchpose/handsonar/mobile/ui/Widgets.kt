package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.shared.ChirpCheck
import com.watchpose.handsonar.shared.ChirpHeard

enum class Level { OK, WARN, BAD, IDLE }

fun Level.color(): Color = when (this) {
    Level.OK -> StatusColors.ok
    Level.WARN -> StatusColors.warn
    Level.BAD -> StatusColors.bad
    Level.IDLE -> StatusColors.idle
}

/** 상태 한 줄: 색 점, 이름, 값. 50 cm 떨어져 읽히게 크게 쓴다. */
@Composable
fun StatusRow(label: String, value: String, level: Level, modifier: Modifier = Modifier) {
    Row(modifier, verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(18.dp).background(level.color(), CircleShape))
        Spacer(Modifier.width(12.dp))
        Text(label, style = MaterialTheme.typography.titleLarge, modifier = Modifier.width(140.dp))
        Text(value, style = MaterialTheme.typography.titleLarge)
    }
}

/** 미리보기 위에 겹치는 작은 상태 표. */
@Composable
fun StatusChip(label: String, level: Level) {
    Row(
        Modifier.background(Color(0x99000000), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(12.dp).background(level.color(), CircleShape))
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.titleMedium, color = Color.White)
    }
}

enum class ButtonKind { PRIMARY, TONAL, OUTLINED, DANGER }

/** 높이 72 dp 이상의 큰 버튼. */
@Composable
fun BigButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    kind: ButtonKind = ButtonKind.PRIMARY,
) {
    val m = modifier.heightIn(min = 72.dp)
    val content: @Composable RowScope.() -> Unit = { Text(text, style = MaterialTheme.typography.headlineSmall) }
    when (kind) {
        ButtonKind.PRIMARY -> Button(onClick = onClick, modifier = m, enabled = enabled, content = content)
        ButtonKind.TONAL -> FilledTonalButton(onClick = onClick, modifier = m, enabled = enabled, content = content)
        ButtonKind.OUTLINED -> OutlinedButton(onClick = onClick, modifier = m, enabled = enabled, content = content)
        ButtonKind.DANGER -> Button(
            onClick = onClick,
            modifier = m,
            enabled = enabled,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.error,
                contentColor = MaterialTheme.colorScheme.onError,
            ),
            content = content,
        )
    }
}

@Composable
fun WarningText(text: String) {
    Text("⚠ $text", style = MaterialTheme.typography.titleMedium, color = StatusColors.warn)
}

/** 워치 처프 상태 글과 수준. 괄호 안은 봉우리/잡음 비. */
fun chirpStatus(c: ChirpCheck?): Pair<String, Level> {
    if (c == null) return "확인 중…" to Level.IDLE
    return when (c.heard) {
        ChirpHeard.HEARD -> "들림 (%.0f)".format(c.ratio) to Level.OK
        ChirpHeard.WEAK -> "약함 (%.0f)".format(c.ratio) to Level.WARN
        ChirpHeard.NONE -> "안 들림 — 워치 녹음을 켜세요" to Level.BAD
    }
}

fun mmss(sec: Int): String = "%d:%02d".format(sec / 60, sec % 60)

fun percent(x: Double?): String = if (x == null || x.isNaN()) "—" else "%.0f%%".format(x * 100)
