package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * 송신 처프 한 주기(설계 §3·§5.2). 워치는 이 주기를 이어 붙여 쉬지 않고 낸다.
 * 위상 φ(t) = 2π(f0·t + ½·k·t²), k = 폭/길이라 순간 주파수가 18 kHz에서 21 kHz로 곧게 오른다.
 */
object Sweep {
    /** 진폭 1의 한 주기([SonarSpec.SWEEP_SAMPLES]개). HANN이면 대칭 한 창을 곱해 양 끝을 0으로 만든다. */
    fun wave(window: SweepWindow): FloatArray {
        val n = SonarSpec.SWEEP_SAMPLES
        val rate = SonarSpec.SAMPLE_RATE.toDouble()
        val slope = SonarSpec.BANDWIDTH_HZ / (n / rate)
        return FloatArray(n) { i ->
            val t = i / rate
            val phase = 2 * PI * (SonarSpec.START_HZ * t + 0.5 * slope * t * t)
            val w = when (window) {
                SweepWindow.NONE -> 1.0
                SweepWindow.HANN -> 0.5 - 0.5 * cos(2 * PI * i / (n - 1))
            }
            (sin(phase) * w).toFloat()
        }
    }

    /** 16비트 PCM 한 주기. [amplitude]는 0..1로 자른다. */
    fun pcm16(window: SweepWindow, amplitude: Float): ShortArray {
        val a = amplitude.coerceIn(0f, 1f)
        val w = wave(window)
        return ShortArray(w.size) { i -> (w[i] * a * 32767f).roundToInt().coerceIn(-32768, 32767).toShort() }
    }

    /** i번째 샘플의 순간 주파수(Hz). */
    fun instantFrequency(i: Int): Double = SonarSpec.START_HZ + SonarSpec.BANDWIDTH_HZ * i / SonarSpec.SWEEP_SAMPLES
}
