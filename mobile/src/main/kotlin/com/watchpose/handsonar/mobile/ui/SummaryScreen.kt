package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.session.UiState
import com.watchpose.handsonar.shared.SessionChecks

/**
 * 세션 직후 화면(설계 §6.3): 누르지 않아도 이미 저장됐다. 결과 요약과 "방금 세션 되돌리기"(다음 세션 시작 전까지).
 * 맨 아래 줄은 그 세션의 meta.json에 적은 세션 정보와 "워치 다시 참"이다(계획 6). "기준 자세"는 그 단계를 거친 세션에만 나오며(2026-10-31부터
 * 끔, SessionController.CALIBRATE) 통과·건너뜀과 옆면·정면 유지 구간의 손 각도(옆면 = 손등 쪽, 정면 = 옆 꺾임)·팔뚝 각도 중앙값이다(워치
 * 기울기 없이 손만 보고 넘어갔으면 노랑).
 */
private fun degrees(v: Double?): String = v?.let { "%.0f°".format(it) } ?: "—"

@Composable
fun SummaryScreen(ui: UiState, c: SessionController) {
    val s = ui.summary ?: return
    Row(Modifier.fillMaxSize().padding(24.dp), horizontalArrangement = Arrangement.spacedBy(32.dp)) {
        Column(Modifier.weight(1.2f), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(if (s.flags.isEmpty()) "세션 저장됨" else "세션 저장됨 · 확인 필요", style = MaterialTheme.typography.displaySmall)
            Text(s.dir.name, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            StatusRow("길이", mmss(s.durationSec.toInt()), Level.OK)
            StatusRow("손 인식", percent(s.handRatio), if (s.handRatio >= SessionChecks.MIN_HAND_RATIO) Level.OK else Level.WARN)
            StatusRow(
                "초당 프레임",
                s.fps?.let { "%.1f".format(it) + if (s.delegate.isNotEmpty()) " · ${s.delegate}" else "" } ?: "—",
                if ((s.fps ?: 0.0) >= SessionChecks.MIN_FPS) Level.OK else Level.WARN,
            )
            // 워치 녹음: 기울기를 받고 있던 시간 비율(폰 마이크로는 처프가 잘 안 들려 처프 비율은 meta.json에만 둔다)
            val link = s.watchLinkRatio
            StatusRow("워치 연결", percent(link), if (link != null && link >= SessionChecks.MIN_LINK_RATIO) Level.OK else Level.WARN)
            StatusRow("시작 표지음", if (s.startMarker) "찾음" else "못 찾음", if (s.startMarker) Level.OK else Level.BAD)
            StatusRow("끝 표지음", if (s.endMarker) "찾음" else "못 찾음", if (s.endMarker) Level.OK else Level.BAD)
            StatusRow("영상·소리", if (s.videoOk) "저장됨" else "오류", if (s.videoOk) Level.OK else Level.BAD)
            s.calibration?.let { cal ->
                StatusRow(
                    "기준 자세",
                    if (cal.result == "skip") {
                        "건너뜀"
                    } else {
                        "통과 · 옆면 %s · 정면 %s · 팔뚝 %s".format(
                            degrees(cal.side?.handDeg),
                            degrees(cal.front?.handDeg),
                            if (cal.checks == "hand") "워치 기울기 없음" else degrees(cal.front?.forearmDeg),
                        )
                    },
                    if (cal.result == "skip" || cal.checks == "hand") Level.WARN else Level.OK,
                )
            }
            StatusRow("세션 정보", s.context.line(s.rewear), Level.IDLE) // meta.json에 적은 세션 정보와 "워치 다시 참"(계획 6)
            if (s.undone) WarningText("휴지통으로 보냈습니다. 기록 화면에서 되살릴 수 있습니다")
            ui.error?.let { WarningText(it) }
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            BigButton("다음 세션 준비", c::nextSession, Modifier.fillMaxWidth())
            BigButton(
                "방금 세션 되돌리기",
                c::undoLast,
                Modifier.fillMaxWidth(),
                enabled = ui.lastDir != null && !s.undone,
                kind = ButtonKind.OUTLINED,
            )
            BigButton("기록 보기", c::showRecords, Modifier.fillMaxWidth(), kind = ButtonKind.OUTLINED)
        }
    }
}
