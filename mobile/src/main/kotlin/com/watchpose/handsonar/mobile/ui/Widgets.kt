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

/**
 * 워치가 녹음 중인지(준비 화면 한 줄과 수집 화면 칩). 기울기([UiState.watchTilt] — 녹음 중인 워치만 보냄)를 받으면 확실하다. 못 받으면 처프가
 * 들리는지로 본다. 폰 마이크로는 처프가 잘 안 들려(2026-10-31 사용자) 처프가 안 들리는 것만으로는 빨강이 아니라 노랑이다.
 * [short]면 칩에 맞게 짧게 쓴다.
 */
fun watchStatus(tilt: Boolean, chirp: ChirpCheck?, short: Boolean = false): Pair<String, Level> = when {
    tilt -> (if (short) "워치 녹음 중" else "녹음 중(기울기 받는 중)") to Level.OK
    chirp?.heard == ChirpHeard.HEARD || chirp?.heard == ChirpHeard.WEAK ->
        (if (short) "워치 처프만" else "처프는 들림 — 기울기 없음(기준 자세는 손만 봄)") to Level.WARN
    else -> (if (short) "워치 확인 안 됨" else "확인 안 됨 — 워치 녹음을 먼저 켜세요") to Level.WARN
}

fun mmss(sec: Int): String = "%d:%02d".format(sec / 60, sec % 60)

fun percent(x: Double?): String = if (x == null || x.isNaN()) "—" else "%.0f%%".format(x * 100)
