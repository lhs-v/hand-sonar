package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * 송신 처프 한 주기(설계 §3·§5.2). 워치는 이 주기를 이어 붙여 쉬지 않고 낸다.
 * 톱니: 위상 φ(t) = 2π(f0·t + ½·k·t²), k = 폭/길이라 순간 주파수가 18 kHz에서 21 kHz로 곧게 오른다.
 * 삼각: 앞 반은 18 kHz → [SonarSpec.TRIANGLE_TOP_HZ]로, 뒤 반은 다시 18 kHz로 곧게 내린다(위상은 반 사이에서 이어진다).
 */
object Sweep {
    /**
     * 진폭 1의 한 주기([SonarSpec.SWEEP_SAMPLES]개). HANN이면 대칭 한 창을 곱해 양 끝을 0으로 만든다. TUKEY면 양 끝 [SonarSpec.TUKEY_RAMP_SAMPLES]만
     * 반 코사인 경사(0.5 − 0.5·cos(π·j/m), j = 끝에서 센 자리)를 곱하고 가운데는 그대로 둔다. 창은 모양과 따로 고른다(삼각은 창이 없어도 이음새가 매끄럽다).
     */
    fun wave(window: SweepWindow, shape: SweepShape = SweepShape.SAW): FloatArray {
        val n = SonarSpec.SWEEP_SAMPLES
        val m = SonarSpec.TUKEY_RAMP_SAMPLES
        return FloatArray(n) { i ->
            val w = when (window) {
                SweepWindow.NONE -> 1.0
                SweepWindow.HANN -> 0.5 - 0.5 * cos(2 * PI * i / (n - 1))
                SweepWindow.TUKEY -> minOf(i, n - 1 - i).let { j -> if (j >= m) 1.0 else 0.5 - 0.5 * cos(PI * j / m) }
            }
            (sin(phase(i, shape)) * w).toFloat()
        }
    }

    /** i번째 샘플의 위상(라디안). i = [SonarSpec.SWEEP_SAMPLES]이면 한 주기를 다 돈 위상이다(삼각은 2π의 정수배). */
    fun phase(i: Int, shape: SweepShape = SweepShape.SAW): Double {
        val rate = SonarSpec.SAMPLE_RATE.toDouble()
        val f0 = SonarSpec.START_HZ
        val t = i / rate
        return when (shape) {
            SweepShape.SAW -> {
                val slope = SonarSpec.BANDWIDTH_HZ / (SonarSpec.SWEEP_SAMPLES / rate)
                2 * PI * (f0 * t + 0.5 * slope * t * t)
            }
            SweepShape.TRIANGLE -> {
                val half = SonarSpec.SWEEP_SAMPLES / 2 / rate
                val top = SonarSpec.TRIANGLE_TOP_HZ
                val slope = (top - f0) / half
                if (t <= half) {
                    2 * PI * (f0 * t + 0.5 * slope * t * t)
                } else {
                    val u = t - half
                    2 * PI * (f0 * half + 0.5 * slope * half * half + top * u - 0.5 * slope * u * u)
                }
            }
        }
    }

    /** 16비트 PCM 한 주기. [amplitude]는 0..1로 자른다. */
    fun pcm16(window: SweepWindow, amplitude: Float, shape: SweepShape = SweepShape.SAW): ShortArray {
        val a = amplitude.coerceIn(0f, 1f)
        val w = wave(window, shape)
        return ShortArray(w.size) { i -> (w[i] * a * 32767f).roundToInt().coerceIn(-32768, 32767).toShort() }
    }

    /** i번째 샘플의 순간 주파수(Hz). */
    fun instantFrequency(i: Int, shape: SweepShape = SweepShape.SAW): Double = when (shape) {
        SweepShape.SAW -> SonarSpec.START_HZ + SonarSpec.BANDWIDTH_HZ * i / SonarSpec.SWEEP_SAMPLES
        SweepShape.TRIANGLE -> {
            val half = SonarSpec.SWEEP_SAMPLES / 2
            val span = SonarSpec.TRIANGLE_TOP_HZ - SonarSpec.START_HZ
            if (i <= half) SonarSpec.START_HZ + span * i / half else SonarSpec.TRIANGLE_TOP_HZ - span * (i - half) / half
        }
    }

    /** 이 모양의 가장 높은 주파수(Hz): rec.json `tx.topHz`. */
    fun topHz(shape: SweepShape): Double = when (shape) {
        SweepShape.SAW -> SonarSpec.START_HZ + SonarSpec.BANDWIDTH_HZ
        SweepShape.TRIANGLE -> SonarSpec.TRIANGLE_TOP_HZ
    }
}
