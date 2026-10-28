package com.watchpose.handsonar.mobile.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.watchpose.handsonar.shared.MarkerSpec

/**
 * 표지음(설계 §6.5)을 폰 내장 스피커로 낸다. AudioTrack 정적 모드(미디어 용도)에 한 번 써 두고, 낼 때마다 처음부터 다시 튼다.
 *
 * 소리 크기는 폰 미디어 볼륨을 따르는데, 평소 볼륨을 0으로 두어도 표지음이 나가도록 낼 때만 올렸다 되돌린다.
 * [play] 직전에 미디어 볼륨이 목표([targetVolume])보다 낮으면 목표로 올리고(볼륨 막대는 띄우지 않음) 원래 값을 기억해 두었다가,
 * 낸 뒤 [RESTORE_DELAY_MS]가 지나면 되돌린다. 그사이 또 [play]하면 되돌림을 미루고, 되돌릴 때 볼륨이 우리가 올린 값이 아니면
 * (사용자가 그새 바꿈) 건드리지 않으며, [close]는 예약된 되돌림을 바로 한다. 볼륨을 못 바꿔도(예외) 표지음은 낸다.
 * 안 들리면 준비 화면의 표지음 자기 확인이 알려 준다.
 *
 * 안드로이드는 미디어 볼륨을 출력 기기마다 따로 둔다. 이어폰·블루투스 오디오가 연결돼 있으면 올리는 것은 그쪽 값이고
 * 폰 스피커 값은 바꿀 수 없다(표지음은 스피커로 고정해 두므로 소리 크기는 바뀌지 않은 스피커 값을 따른다). 메인 스레드에서 부른다.
 */
class MarkerPlayer(context: Context) : AutoCloseable {
    private val pcm = MarkerSpec.pcm16()
    private val audioManager: AudioManager? = context.getSystemService(AudioManager::class.java)
    private val handler = Handler(Looper.getMainLooper())
    private val restoreRunnable = Runnable { restoreVolume() }

    /** 올리기 전의 미디어 볼륨. [raisedVolume]이 [NOT_RAISED]가 아닐 때만 뜻이 있다. */
    private var savedVolume = 0

    /** 우리가 올려서 시스템이 가진 미디어 볼륨. 올려 둔 것이 없으면 [NOT_RAISED]. */
    private var raisedVolume = NOT_RAISED

    private val track: AudioTrack = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build(),
        )
        .setAudioFormat(
            AudioFormat.Builder()
                .setSampleRate(MarkerSpec.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
        )
        .setTransferMode(AudioTrack.MODE_STATIC)
        .setBufferSizeInBytes(pcm.size * 2)
        .build()

    init {
        check(track.write(pcm, 0, pcm.size) == pcm.size) { "표지음을 재생 버퍼에 못 씀" }
        audioManager
            ?.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
            ?.firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
            ?.let { track.setPreferredDevice(it) }
    }

    /**
     * 표지음을 낸다. 미디어 볼륨이 목표보다 낮으면 먼저 올린다(되돌림은 [RESTORE_DELAY_MS] 뒤).
     * play()를 부른 직후의 elapsedNs를 돌려준다(실제로 소리가 난 시각은 폰 마이크 녹음에서 찾는다).
     */
    fun play(): Long {
        raiseVolume()
        try {
            runCatching { track.stop() }
            track.reloadStaticData()
            track.play()
            return SystemClock.elapsedRealtimeNanos()
        } finally {
            scheduleRestore() // 재생이 예외로 끝나도 올려 둔 볼륨은 되돌린다
        }
    }

    override fun close() {
        handler.removeCallbacks(restoreRunnable)
        restoreVolume()
        track.release()
    }

    /** 미디어 볼륨이 목표보다 낮으면 목표로 올리고 원래 값을 기억한다. 못 올려도 예외는 내지 않는다. */
    private fun raiseVolume() {
        val am = audioManager ?: return
        try {
            val now = am.getStreamVolume(AudioManager.STREAM_MUSIC)
            if (raisedVolume != NOT_RAISED) {
                if (now == raisedVolume) return // 앞 표지음 때 올린 채 그대로: 원래 값은 그대로 기억한다
                raisedVolume = NOT_RAISED // 그새 사용자가 바꿨다: 지금 값을 새 원래 값으로 본다
            }
            val target = targetVolume(am.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
            if (now >= target) return
            am.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0) // 0: 볼륨 막대 같은 화면 표시 없음
            savedVolume = now
            raisedVolume = target
            // 기기가 한도를 두면 목표보다 낮게 받아들일 수 있다: 되돌릴 때 비교할 값은 시스템이 실제로 가진 값
            val after = am.getStreamVolume(AudioManager.STREAM_MUSIC)
            if (after != now) raisedVolume = after
        } catch (e: Exception) {
            Log.w(TAG, "미디어 볼륨을 못 올림(표지음은 그대로 냄)", e)
        }
    }

    /** 올려 둔 것이 있으면 [RESTORE_DELAY_MS] 뒤에 되돌리도록 (다시) 예약한다. 앞 예약은 취소한다. */
    private fun scheduleRestore() {
        handler.removeCallbacks(restoreRunnable)
        if (raisedVolume != NOT_RAISED) handler.postDelayed(restoreRunnable, RESTORE_DELAY_MS)
    }

    /** 올려 둔 볼륨을 원래 값으로 되돌린다. 지금 값이 우리가 올린 값이 아니면(사용자가 바꿈) 건드리지 않는다. */
    private fun restoreVolume() {
        val raised = raisedVolume
        if (raised == NOT_RAISED) return
        raisedVolume = NOT_RAISED
        val am = audioManager ?: return
        try {
            if (am.getStreamVolume(AudioManager.STREAM_MUSIC) == raised) {
                am.setStreamVolume(AudioManager.STREAM_MUSIC, savedVolume, 0)
            }
        } catch (e: Exception) {
            Log.w(TAG, "미디어 볼륨을 못 되돌림", e)
        }
    }

    private companion object {
        const val TAG = "MarkerPlayer"
        const val NOT_RAISED = -1

        /** 목표 볼륨은 최대의 11/15(올림): 15단계 기기에서 11. 2026-10-27 실기기에서 워치가 표지음을 비 100–230으로 들은 값. */
        const val TARGET_NUM = 11
        const val TARGET_DEN = 15

        /** 표지음을 낸 뒤 볼륨을 되돌리기까지. 40 ms 소리와 재생 지연이 끝나고도 남는 시간. */
        const val RESTORE_DELAY_MS = 600L

        /** 최대 볼륨이 [max]단계인 기기에서 표지음을 낼 때 맞추는 미디어 볼륨. */
        fun targetVolume(max: Int): Int = (max * TARGET_NUM + TARGET_DEN - 1) / TARGET_DEN
    }
}
