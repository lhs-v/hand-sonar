package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.session.UiState
import com.watchpose.handsonar.mobile.transfer.InboxStatus
import com.watchpose.handsonar.shared.ArmSupport
import com.watchpose.handsonar.shared.ChirpHeard
import com.watchpose.handsonar.shared.Posture
import com.watchpose.handsonar.shared.SessionContext
import com.watchpose.handsonar.shared.Strap

/**
 * 준비 화면(설계 §6.1-1): 추적하는 손(워치를 찬 손, "추적" 버튼으로 왼손·오른손을 고름)이 보이는지, 워치 처프가 들리는지,
 * 표지음 자기 확인이 됐는지. 손·표지음이 녹색이고 장치에 오류가 없으면 시작 버튼이 켜진다.
 * 워치 처프가 안 들리면 확인 창을 거쳐 시작할 수 있다. 이어폰·블루투스 오디오가 연결돼 있으면 경고만 띄운다(표지음이 스피커로 안 나가면
 * 표지음 자기 확인이 녹색이 안 돼 시작이 막힌다). 맨 아래에 워치에서 받은 녹음 줄([inbox], 전송 설계 §3)이 있다.
 * 세션 정보(전송 설계 §8, 계획 6): 장소(누르면 쓰는 창), 자세·팔 받침·줄 조임(칩 — 고른 칩을 다시 누르면 지움), "워치 다시 참"(워치를 벗었다
 * 다시 찼으면 켬 — 다음 세션이 보관되면 꺼진다). 고른 값은 다음 실행에도 남고 세션을 시작할 때 meta.json에 들어간다. 50 cm에서 읽히게 칩 글자를
 * 크게(titleMedium, 높이 48 dp) 하고, 오른쪽 열이 좁으면 묶음째 다음 줄로 넘긴다(FlowRow). 되돌리기·기록 버튼 아래에 두어 그 버튼이 밀리지 않게
 * 한다(검토 M8).
 */
@Composable
fun ReadyScreen(ui: UiState, c: SessionController, inbox: InboxStatus, onRequestPermission: () -> Unit) {
    var confirmNoWatch by remember { mutableStateOf(false) }
    var editPlace by remember { mutableStateOf(false) }
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
                    else -> "못 들음 — 이어폰·블루투스 오디오를 빼고 다시 시험하세요"
                },
                when {
                    self?.found == true -> Level.OK
                    self == null || ui.selfCheckBusy -> Level.IDLE
                    else -> Level.BAD
                },
            )
            if (ui.externalOutputs.isNotEmpty()) {
                WarningText(
                    "이어폰·블루투스 오디오가 연결돼 있습니다(${ui.externalOutputs.joinToString(", ")}) — " +
                        "표지음은 폰 스피커로 냅니다. 표지음 시험이 실패하면 빼 주세요",
                )
            }
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
            // 세션 정보(전송 설계 §8): 처리에는 쓰지 않고, 나중에 환경 변수를 가려 보려고 meta.json에 남긴다
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                itemVerticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedButton(onClick = { editPlace = true }, modifier = Modifier.heightIn(min = 48.dp), enabled = !ui.busy) {
                    Text(
                        "장소: ${ui.context.place.ifBlank { "안 정함" }}",
                        style = MaterialTheme.typography.titleMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                BigChip("워치 다시 참", ui.rewear, !ui.busy, c::toggleRewear)
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Choice("자세", Posture.entries, ui.context.posture, { it.label }, !ui.busy) { p -> c.updateContext { it.copy(posture = p) } }
                Choice("팔", ArmSupport.entries, ui.context.armSupport, { it.label }, !ui.busy) { a -> c.updateContext { it.copy(armSupport = a) } }
                Choice("줄 조임", Strap.entries, ui.context.strap, { it.label }, !ui.busy) { s -> c.updateContext { it.copy(strap = s) } }
            }
            if (ui.rewear) {
                Text(
                    "다음 세션을 워치를 다시 찬 세션으로 적습니다(그 세션이 보관되면 꺼짐)",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                "미디어 볼륨 ${ui.mediaVolume}/${ui.mediaMax} (표지음 때만 잠깐 올림) · 저장 공간 %.1f GB".format(ui.freeGb),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            WatchInboxLine(inbox)
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
    if (editPlace) {
        PlaceDialog(ui.context.place, onDismiss = { editPlace = false }) { place ->
            editPlace = false
            c.updateContext { it.copy(place = place) }
        }
    }
}

/** 이름 하나와 고르는 칩들(세션 정보, 계획 6). 한 묶음은 함께 줄을 넘긴다. 고른 칩을 다시 누르면 null(고르지 않음)로 돌린다. */
@Composable
private fun <T> Choice(label: String, options: List<T>, selected: T?, name: (T) -> String, enabled: Boolean, onSelect: (T?) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.titleMedium)
        for (option in options) {
            BigChip(name(option), option == selected, enabled) { onSelect(if (option == selected) null else option) }
        }
    }
}

/** 50 cm에서 읽히는 칩: 글자 titleMedium, 높이 48 dp 이상(기본 칩은 14 sp·32 dp, 검토 M8). */
@Composable
private fun BigChip(text: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        modifier = Modifier.heightIn(min = 48.dp),
        enabled = enabled,
        label = { Text(text, style = MaterialTheme.typography.titleMedium) },
    )
}

/** 장소를 쓰는 창(계획 6): 한 줄, [SessionContext.MAX_PLACE]자(코드 포인트)까지. 비워 저장하면 장소를 지운다. 공백 정리는 조정자가 한다. */
@Composable
private fun PlaceDialog(initial: String, onDismiss: () -> Unit, onSave: (String) -> Unit) {
    var text by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = { onSave(text) }) { Text("저장") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("취소") } },
        title = { Text("장소") },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = SessionContext.limit(it) }, // 코드 포인트로 잘라 이모지를 반으로 자르지 않는다
                singleLine = true,
                placeholder = { Text("예: 집 책상") },
            )
        },
    )
}
