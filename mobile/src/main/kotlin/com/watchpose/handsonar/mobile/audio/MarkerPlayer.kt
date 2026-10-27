package com.watchpose.handsonar.mobile.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock
import com.watchpose.handsonar.shared.MarkerSpec

/**
 * 표지음(설계 §6.5)을 폰 내장 스피커로 낸다. AudioTrack 정적 모드(미디어 용도)에 한 번 써 두고, 낼 때마다 처음부터 다시 튼다.
 * 소리 크기는 폰 미디어 볼륨을 따른다. 메인 스레드에서 부른다.
 */
class MarkerPlayer(context: Context) : AutoCloseable {
    private val pcm = MarkerSpec.pcm16()
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
        context.getSystemService(AudioManager::class.java)
            ?.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
            ?.firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
            ?.let { track.setPreferredDevice(it) }
    }

    /** 표지음을 낸다. play()를 부른 직후의 elapsedNs를 돌려준다(실제로 소리가 난 시각은 폰 마이크 녹음에서 찾는다). */
    fun play(): Long {
        runCatching { track.stop() }
        track.reloadStaticData()
        track.play()
        return SystemClock.elapsedRealtimeNanos()
    }

    override fun close() = track.release()
}
