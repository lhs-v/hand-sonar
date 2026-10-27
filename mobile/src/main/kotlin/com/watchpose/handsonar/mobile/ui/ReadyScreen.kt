package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.session.UiState
import com.watchpose.handsonar.shared.ChirpHeard

/**
 * 준비 화면(설계 §6.1-1): 추적하는 손(워치를 찬 손, "추적" 버튼으로 왼손·오른손을 고름)이 보이는지, 워치 처프가 들리는지,
 * 표지음 자기 확인이 됐는지. 손·표지음이 녹색이고 장치에 오류가 없으면 시작 버튼이 켜진다.
 * 워치 처프가 안 들리면 확인 창을 거쳐 시작할 수 있다.
 */
@Composable
fun ReadyScreen(ui: UiState, c: SessionController, onRequestPermission: () -> Unit) {
    var confirmNoWatch by remember { mutableStateOf(false) }
    // 카메라 오류는 한 번 나면 지워지지 않으므로 시작 조건에 넣지 않는다: 추적하는 손이 보이면 카메라는 돌고 있다.
    // 표지음 시험 중에 시작하면 두 표지음이 겹치므로 막는다.
    val canStart = ui.devicesOpen && ui.micError == null && ui.handVisible &&
        ui.selfCheck?.found == true && !ui.selfCheckBusy && !ui.busy
    Row(Modifier.fillMaxSize().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
        CameraView(
            c.frame,
            Modifier.weight(1.25f).fillMaxHeight().clip(RoundedCornerShape(16.dp)),
            message = when {
                !ui.permissionsGranted -> "카메라·마이크 권한이 필요합니다"
                ui.cameraError != null -> ui.cameraError
                else -> null
            },
        )
        Column(
            Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text("손 소나 수집", style = MaterialTheme.typography.displaySmall)
            Text(ui.cameraLabel, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            StatusRow(
                ui.trackedHand.label,
                if (ui.handVisible) "보임" else "안 보임 — ${ui.trackedHand.label}을 카메라 앞에",
                if (ui.handVisible) Level.OK else Level.BAD,
            )
            val (chirpText, chirpLevel) = chirpStatus(ui.chirp)
            StatusRow("워치 처프", chirpText, chirpLevel)
            val self = ui.selfCheck
            StatusRow(
                "표지음",
                when {
                    ui.selfCheckBusy -> "확인 중…"
                    self == null -> "아직 — 표지음 시험을 누르세요"
                    self.found -> "확인됨 (%.0f)".format(self.ratio ?: 0.0)
                    else -> "못 들음 — 미디어 볼륨을 올리세요"
                },
                when {
                    self?.found == true -> Level.OK
                    self == null || ui.selfCheckBusy -> Level.IDLE
                    else -> Level.BAD
                },
            )
            if (ui.devicesOpen && ui.mediaVolume == 0) WarningText("미디어 볼륨이 0이라 표지음이 나가지 않습니다")
            if (ui.devicesOpen && ui.freeGb < 1.0) WarningText("남은 저장 공간이 1 GB 아래입니다(%.1f GB)".format(ui.freeGb))
            ui.micError?.let { WarningText(it) }
            ui.error?.let { WarningText(it) }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                BigButton(
                    "표지음 시험",
                    c::testMarker,
                    Modifier.weight(1f),
                    enabled = ui.devicesOpen && !ui.selfCheckBusy,
                    kind = ButtonKind.TONAL,
                )
                BigButton(
                    "시작",
                    { if (ui.chirp == null || ui.chirp.heard == ChirpHeard.NONE) confirmNoWatch = true else c.startSession() },
                    Modifier.weight(1f),
                    enabled = canStart,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = c::toggleHand, enabled = !ui.busy) { Text("추적: ${ui.trackedHand.label}") }
                TextButton(onClick = c::cycleRotation) { Text("회전 보정 ${ui.extraRotation}°") }
                if (!ui.permissionsGranted) Button(onClick = onRequestPermission) { Text("권한 요청") }
            }
            // 세션 버튼은 줄을 따로 둔다: 되돌리기까지 한 줄이면 오른쪽 열(약 459 dp)을 넘어 끝 버튼이 접힌다.
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                if (ui.lastDir != null) OutlinedButton(onClick = c::undoLast) { Text("방금 세션 되돌리기") }
                OutlinedButton(onClick = c::showRecords) { Text("기록") }
            }
            Text(
                "미디어 볼륨 ${ui.mediaVolume}/${ui.mediaMax} · 저장 공간 %.1f GB".format(ui.freeGb),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    if (confirmNoWatch) {
        AlertDialog(
            onDismissRequest = { confirmNoWatch = false },
            confirmButton = {
                TextButton(onClick = {
                    confirmNoWatch = false
                    c.startSession()
                }) { Text("그래도 시작") }
            },
            dismissButton = { TextButton(onClick = { confirmNoWatch = false }) { Text("그만두기") } },
            title = { Text("워치 처프가 안 들립니다") },
            text = { Text("워치 녹음이 켜져 있는지 확인하세요. 처프 없이 모은 세션은 학습에 쓸 수 없습니다.") },
        )
    }
}
