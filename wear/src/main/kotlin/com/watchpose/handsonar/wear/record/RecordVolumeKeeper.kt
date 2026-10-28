package com.watchpose.handsonar.wear.record

import android.content.Context
import android.media.AudioManager

/**
 * 녹음 볼륨(전송 설계 §8, 계획 6): 녹음을 시작할 때 미디어 볼륨을 고른 단계로 맞추고([apply]), 멈출 때 원래대로 되돌린다([restore]).
 * 바꾸기 **전에** 원래 값을 앱 설정(`record-volume`)에 `commit`으로 먼저 적는다: 녹음 중 앱이 죽어 되돌리지 못했으면 다음에 워치 화면이
 * 보이거나 다음 녹음을 시작할 때 [restore]가 그 값으로 되돌린다. 볼륨 창은 띄우지 않는다(flags 0). 주 스레드에서 부른다.
 * 미디어 볼륨은 출력 기기마다 따로라, 워치에 블루투스 이어폰이 이어져 있으면 그 기기의 볼륨이 바뀐다(처프는 내장 스피커로 낸다) —
 * 그래서 맞춘 뒤 다시 읽은 값을 rec.json에 적는다.
 */
class RecordVolumeKeeper(context: Context) {
    private val audio: AudioManager = context.getSystemService(AudioManager::class.java)
    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** 되돌릴 원래 볼륨(바꿔 둔 것이 없으면 −1). 녹음 화면이 "끝나면 N으로"를 보인다. */
    val pending: Int get() = prefs.getInt(KEY_ORIGINAL, -1)

    /**
     * 미디어 볼륨을 [target]으로 맞추고, 맞춘 뒤 다시 읽은 실제 값을 돌려준다. 지금 값([current])과 같으면 바꾸지 않는다.
     * 바꾸기 전에 [current]를 적어 둔다(앱이 죽어도 되돌릴 수 있게 — 그래서 apply가 아니라 commit).
     */
    fun apply(target: Int, current: Int): Int {
        if (target != current) {
            prefs.edit().putInt(KEY_ORIGINAL, current).commit()
            runCatching { audio.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0) }
        }
        return audio.getStreamVolume(AudioManager.STREAM_MUSIC)
    }

    /** 바꿔 둔 볼륨이 있으면 원래 값으로 되돌리고 되돌린 뒤 다시 읽은 값을, 없으면 null을 돌려준다. */
    fun restore(): Int? {
        val original = prefs.getInt(KEY_ORIGINAL, -1)
        if (original < 0) return null
        runCatching { audio.setStreamVolume(AudioManager.STREAM_MUSIC, original, 0) }
        prefs.edit().remove(KEY_ORIGINAL).apply()
        return audio.getStreamVolume(AudioManager.STREAM_MUSIC)
    }

    private companion object {
        const val PREFS = "record-volume"
        const val KEY_ORIGINAL = "original"
    }
}
