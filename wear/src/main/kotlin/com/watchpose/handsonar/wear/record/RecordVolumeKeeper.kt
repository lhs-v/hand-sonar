package com.watchpose.handsonar.wear.record

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import com.watchpose.handsonar.shared.RecordVolumeRules
import com.watchpose.handsonar.shared.VolumeControl
import com.watchpose.handsonar.shared.VolumePending
import com.watchpose.handsonar.shared.VolumeRestore
import com.watchpose.handsonar.shared.VolumeStore

/**
 * 녹음 볼륨(전송 설계 §8, 계획 6)의 워치 쪽: AudioManager(지금 출력 기기의 미디어 볼륨, 외부 출력 목록)와 앱 설정(`record-volume`)으로
 * [RecordVolumeRules]에 붙는다. 규칙(외부 출력이면 바꾸지·되돌리지 않음, 앱이 죽은 뒤에는 그사이 바뀌지 않았을 때만 되돌림)은 shared에 있다.
 * 기록은 `commit`으로 바로 쓴다: 바꾼 직후 앱이 죽어도 원래 값이 남게. 볼륨 창은 띄우지 않는다(flags 0). 주 스레드에서 부른다.
 */
class RecordVolumeKeeper(context: Context) : VolumeControl, VolumeStore {
    private val audio: AudioManager = context.getSystemService(AudioManager::class.java)
    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    override fun volume(): Int = audio.getStreamVolume(AudioManager.STREAM_MUSIC)

    override fun setVolume(index: Int) {
        runCatching { audio.setStreamVolume(AudioManager.STREAM_MUSIC, index, 0) }
    }

    override fun externalOutputs(): List<String> = WatchOutputs.external(audio)

    /** 맞춘 값이 없는 기록은 −1로 읽는다(그사이 바뀐 것으로 보아 덮지 않는다). */
    override fun load(): VolumePending? {
        val original = prefs.getInt(KEY_ORIGINAL, -1)
        return if (original < 0) null else VolumePending(original, prefs.getInt(KEY_APPLIED, -1))
    }

    override fun save(pending: VolumePending?) {
        val edit = prefs.edit()
        if (pending == null) {
            edit.remove(KEY_ORIGINAL).remove(KEY_APPLIED)
        } else {
            edit.putInt(KEY_ORIGINAL, pending.original).putInt(KEY_APPLIED, pending.applied)
        }
        edit.commit() // apply가 아니다: 바로 다음 줄에서 볼륨을 바꾸고, 그 뒤 앱이 죽어도 기록이 남아야 한다
    }

    private companion object {
        const val PREFS = "record-volume"
        const val KEY_ORIGINAL = "original"
        const val KEY_APPLIED = "applied"
    }
}

/**
 * 워치에 이어진 외부 오디오 출력(계획 6 고침 I1). 이런 출력이 있으면 안드로이드가 미디어 볼륨을 그 기기 값으로 다뤄, 녹음 볼륨을 바꾸거나
 * 되돌리면 처프가 나가는 내장 스피커가 아니라 그 기기의 볼륨이 바뀐다. 폰 `mobile.audio.AudioOutputs`와 같은 형을 외부로 치되, 블루투스 통화(SCO)는
 * 미디어가 가지 않고 워치가 폰의 통화를 받는 기기라 늘 보일 수 있어 뺀다. 이름은 rec.json에 적는 형 이름이다.
 */
object WatchOutputs {
    private val KINDS: Map<Int, String> = buildMap {
        put(AudioDeviceInfo.TYPE_BLUETOOTH_A2DP, "bluetooth_a2dp")
        put(AudioDeviceInfo.TYPE_HEARING_AID, "hearing_aid")
        put(AudioDeviceInfo.TYPE_WIRED_HEADSET, "wired_headset")
        put(AudioDeviceInfo.TYPE_WIRED_HEADPHONES, "wired_headphones")
        put(AudioDeviceInfo.TYPE_USB_HEADSET, "usb_headset")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            put(AudioDeviceInfo.TYPE_BLE_HEADSET, "ble_headset")
            put(AudioDeviceInfo.TYPE_BLE_SPEAKER, "ble_speaker")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) put(AudioDeviceInfo.TYPE_BLE_BROADCAST, "ble_broadcast")
    }

    /** 지금 이어진 외부 출력의 형 이름(같은 것은 하나로, 이름 순). 없으면 빈 목록. */
    fun external(audio: AudioManager?): List<String> =
        audio?.getDevices(AudioManager.GET_DEVICES_OUTPUTS).orEmpty().mapNotNull { KINDS[it.type] }.distinct().sorted()
}

/** 미뤄진 되돌리기([RecordVolumeRules.restoreLeftover])의 화면 한 줄. 다룰 기록이 없었으면 null. */
fun leftoverNote(r: VolumeRestore): String? {
    val p = r.pending ?: return null
    return when (r.skipped) {
        null -> "앞 녹음의 녹음 볼륨을 원래대로 되돌림(지금 ${r.restored})"
        RecordVolumeRules.SKIP_CHANGED -> "앞 녹음 뒤 볼륨이 바뀌어 그대로 둠(원래 ${p.original})"
        else -> "외부 출력이 이어져 있어 앞 녹음의 볼륨을 아직 못 되돌림(원래 ${p.original}) — 스피커로 돌아오면 앱을 열 때 되돌림"
    }
}
