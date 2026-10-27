package com.watchpose.handsonar.wear

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/** 녹음 시작·정지·오류 진동(설계 §5.1·§5.4). 시작은 짧게 한 번, 정지는 톡톡, 오류는 길게 세 번이라 손목만으로 구별된다. */
class Haptics(context: Context) {
    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= 31) {
        context.getSystemService(VibratorManager::class.java).defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Vibrator::class.java)
    }

    fun start() = play(VibrationEffect.createOneShot(60, VibrationEffect.DEFAULT_AMPLITUDE))

    fun stop() = play(VibrationEffect.createWaveform(longArrayOf(0, 40, 80, 40), -1))

    fun error() = play(VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 200, 100, 200), -1))

    private fun play(effect: VibrationEffect) {
        runCatching { vibrator.vibrate(effect) }
    }
}
