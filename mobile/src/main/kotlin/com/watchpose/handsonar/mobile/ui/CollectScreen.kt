package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.mobile.camera.TrackedHand
import com.watchpose.handsonar.mobile.session.CalibrationUi
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.session.UiState
import com.watchpose.handsonar.shared.Labels
import com.watchpose.handsonar.shared.RunnerPhase

private const val HAND_LOST_WARN_MS = 1_000L

/**
 * 수집 화면(설계 §6.1-2): 왼쪽에 거울 미리보기·관절 뼈대·상태 표, 오른쪽에 지금 동작(이름·설명·손 모양), 남은 시간 막대, 다음 동작,
 * 세션 진행 막대, 일시정지·틀림·취소. 손이 1 s 넘게 안 보이거나, 이 세션에서 받던 워치 기울기가 10 s 넘게 끊기면 미리보기 아래에 경고한다
 * (설계 §6.5·§6.6). 폰 마이크로는 워치 처프가 잘 안 들려 처프로는 경고하지 않는다(2026-10-31 사용자).
 * 기준 자세 단계(세션 처음)에는 팔뚝·손·손가락 판정 세 줄과 유지 막대를 보이고, "틀림" 자리에 "건너뛰기"를 둔다.
 */
@Composable
fun CollectScreen(ui: UiState, c: SessionController) {
    var confirmCancel by remember { mutableStateOf(false) }
    Row(Modifier.fillMaxSize().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
        Box(Modifier.weight(1.25f).fillMaxHeight()) {
            CameraView(c.frame, Modifier.fillMaxSize().clip(RoundedCornerShape(16.dp)))
            Column(Modifier.align(Alignment.TopStart).padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                StatusChip(if (ui.handVisible) "손 인식" else "손 없음", if (ui.handVisible) Level.OK else Level.BAD)
                val (watchText, watchLevel) = watchStatus(ui.watchTilt, ui.chirp, short = true)
                StatusChip(watchText, watchLevel)
                StatusChip(
                    when (ui.startMarkerFound) {
                        null -> "표지음 확인 중"
                        true -> "시작 표지음 확인"
                        false -> "시작 표지음 못 찾음"
                    },
                    when (ui.startMarkerFound) {
                        null -> Level.IDLE
                        true -> Level.OK
                        false -> Level.BAD
                    },
                )
            }
            val warning = when {
(ui.phase == RunnerPhase.PROMPT || ui.phase == RunnerPhase.CALIB) && ui.handLostMs > HAND_LOST_WARN_MS -> "손이 안 보입니다 — 카메라 앞으로"
                ui.watchLinkLost -> "워치 기울기가 끊겼습니다 — 워치 녹음을 확인하세요"
                else -> null
            }
            if (warning != null) {
                Text(
                    warning,
                    Modifier.align(Alignment.BottomCenter).padding(16.dp)
                        .background(Color(0xCC000000), RoundedCornerShape(8.dp))
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    color = StatusColors.warn,
                    style = MaterialTheme.typography.headlineSmall,
                )
            }
        }
        Column(Modifier.weight(1f).fillMaxHeight(), verticalArrangement = Arrangement.SpaceBetween) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(segmentTitle(ui), style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                // 손 그림은 지금 동작 이름 바로 옆에만 둔다(사용자가 "다음" 옆 그림을 다음 동작으로 읽어 헷갈림, 2026-10-27 실기기).
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                    val shape = when {
                        ui.phase == RunnerPhase.CALIB -> if (Labels.CALIBRATION_IDS[ui.calibStep] == Labels.NEUTRAL_ID) Labels.NEUTRAL.shape else null
                        ui.phase == RunnerPhase.PROMPT || ui.phase == RunnerPhase.PAUSED -> ui.prompt?.shape
                        else -> null
                    }
                    if (shape != null) {
                        HandShapeView(shape, Modifier.size(width = 150.dp, height = 185.dp), right = ui.trackedHand == TrackedHand.RIGHT)
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(mainText(ui), style = MaterialTheme.typography.displayLarge, fontWeight = FontWeight.Bold)
                        Text(
                            if (ui.phase == RunnerPhase.PROMPT) ui.prompt?.hint.orEmpty() else phaseHint(ui),
                            style = MaterialTheme.typography.headlineSmall,
                        )
                    }
                }
                val cal = ui.calibration
                if (ui.phase == RunnerPhase.CALIB) CalibrationLines(cal, ui.watchTilt)
                val bar = if (ui.phase == RunnerPhase.CALIB && cal != null) (cal.heldMs.toFloat() / cal.holdMs).coerceIn(0f, 1f) else ui.promptProgress
                LinearProgressIndicator(progress = { bar }, modifier = Modifier.fillMaxWidth().height(16.dp))
                Text(
                    "다음: ${ui.nextPrompt?.name ?: "—"}",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("세션 ${mmss(ui.elapsedSec)} / ${mmss(ui.plannedSec)}", style = MaterialTheme.typography.titleLarge)
                LinearProgressIndicator(progress = { ui.progress }, modifier = Modifier.fillMaxWidth().height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    BigButton(if (ui.phase == RunnerPhase.PAUSED) "계속" else "일시정지", c::togglePause, Modifier.weight(1f), enabled = !ui.busy)
                    if (ui.phase == RunnerPhase.CALIB) {
                        BigButton("건너뛰기", c::skipCalibration, Modifier.weight(1f), enabled = !ui.busy, kind = ButtonKind.TONAL)
                    } else {
                        BigButton("틀림", c::markWrong, Modifier.weight(1f), enabled = ui.phase == RunnerPhase.PROMPT && !ui.busy, kind = ButtonKind.TONAL)
                    }
                    BigButton("취소", { confirmCancel = true }, Modifier.weight(1f), enabled = !ui.busy, kind = ButtonKind.DANGER)
                }
            }
        }
    }
    if (confirmCancel) {
        AlertDialog(
            onDismissRequest = { confirmCancel = false },
            confirmButton = {
                TextButton(onClick = {
                    confirmCancel = false
                    c.cancelSession()
                }) { Text("세션 취소") }
            },
            dismissButton = { TextButton(onClick = { confirmCancel = false }) { Text("계속 수집") } },
            title = { Text("세션을 취소할까요?") },
            text = { Text("지금 세션을 멈추고 휴지통으로 보냅니다. 기록 화면에서 되살릴 수 있습니다.") },
        )
    }
}

/** 기준 자세 판정 세 줄: 팔뚝(워치 기울기), 손(그림 평면의 각), 손가락 펴짐. 워치 기울기가 없으면 팔뚝 줄이 그 까닭을 말한다. */
@Composable
private fun CalibrationLines(cal: CalibrationUi?, watchTilt: Boolean) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        val forearm = cal?.forearmDeg
        when {
            !watchTilt || cal?.forearmOk == null -> StatusRow("팔뚝", "워치 기울기 없음 — 손만 봅니다", Level.WARN)
            cal.forearmOk && forearm != null -> StatusRow("팔뚝", "%.0f° ✓".format(forearm), Level.OK)
            else -> StatusRow("팔뚝", "%.0f° — 팔뚝을 똑바로 세우세요".format(forearm ?: 0.0), Level.BAD)
        }
        val hand = cal?.handDeg
        val label = if (cal?.side == true) "손등 쪽" else "옆 꺾임" // 옆면은 손등·손바닥 쪽, 정면은 새끼·엄지 쪽 꺾임이 화면에 보인다
        when {
            hand == null -> StatusRow(label, "손이 안 보임", Level.BAD)
            cal.handOk -> StatusRow(label, "%.0f° ✓".format(hand), Level.OK)
            else -> StatusRow(label, "%.0f° — 손을 팔과 일자로".format(hand), Level.BAD)
        }
        if (cal?.side != true) {
            if (cal?.fingersOk == true) StatusRow("손가락", "폄 ✓", Level.OK) else StatusRow("손가락", "모두 펴세요", Level.BAD)
        }
    }
}

private fun segmentTitle(ui: UiState): String {
    if (ui.phase == RunnerPhase.CALIB) return "기준 자세 ${ui.calibStep + 1}/${Labels.CALIBRATION_IDS.size}"
    if (ui.phase == RunnerPhase.TAIL || ui.phase == RunnerPhase.CLOSING) return "마무리"
    if (ui.segment == 0) return "세션 준비"
    val kind = when (ui.segment) {
        1, 2 -> "손가락 굽히기"
        3, 4 -> "ASL 숫자"
        5, 6 -> "손목"
        else -> "자유 동작"
    }
    return "구간 ${ui.segment}/${ui.segmentCount} · $kind"
}

private fun mainText(ui: UiState): String = when (ui.phase) {
    RunnerPhase.IDLE, RunnerPhase.LEAD -> "준비"
    RunnerPhase.CALIB -> Labels.byId(Labels.CALIBRATION_IDS[ui.calibStep]).name
    RunnerPhase.PROMPT -> ui.prompt?.name.orEmpty()
    RunnerPhase.REST -> "쉬기"
    RunnerPhase.TAIL, RunnerPhase.CLOSING -> "마무리"
    RunnerPhase.PAUSED -> "일시정지됨"
    RunnerPhase.FINISHED, RunnerPhase.CANCELLED -> "저장 중…"
}

private fun phaseHint(ui: UiState): String = when (ui.phase) {
    RunnerPhase.IDLE, RunnerPhase.LEAD -> "표지음이 울리면 시작합니다"
    RunnerPhase.CALIB -> Labels.byId(Labels.CALIBRATION_IDS[ui.calibStep]).hint
    RunnerPhase.REST -> "손을 편하게 두세요"
    RunnerPhase.TAIL, RunnerPhase.CLOSING -> "손을 그대로 두세요"
    RunnerPhase.PAUSED -> "계속을 누르면 이어서 합니다"
    else -> ""
}
