package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * 폰이 세션 시작과 끝에 한 번씩 내는 표지음(설계 §6.5): 21.5 kHz에서 23 kHz로 곧게 오르는 처프, 40 ms, 한 창.
 * 워치 녹음과 폰 녹음 양쪽에서 이 소리를 찾아 두 기기의 시각을 맞춘다(설계 §4).
 * 워치가 이 대역을 못 들으면(설계 §11-3) [START_HZ]·[END_HZ]만 15 kHz·17.5 kHz로 바꾼다.
 */
object MarkerSpec {
    const val SAMPLE_RATE = 48_000
    const val START_HZ = 21_500.0
    const val END_HZ = 23_000.0

    /** 40 ms. */
    const val SAMPLES = 1_920

    /** 16비트로 바꿀 때의 진폭(0..1). 소리 크기는 폰 미디어 볼륨이 정한다. */
    const val AMPLITUDE = 0.8f

    /** 진폭 1의 표지음. 위상 φ(t) = 2π(f0·t + ½·k·t²), k = (끝 − 시작)/길이. 대칭 한 창이라 양 끝이 0이다. */
    fun wave(): FloatArray {
        val rate = SAMPLE_RATE.toDouble()
        val slope = (END_HZ - START_HZ) / (SAMPLES / rate)
        return FloatArray(SAMPLES) { i ->
            val t = i / rate
            val window = 0.5 - 0.5 * cos(2 * PI * i / (SAMPLES - 1))
            (sin(2 * PI * (START_HZ * t + 0.5 * slope * t * t)) * window).toFloat()
        }
    }

    /** 16비트 PCM([AMPLITUDE]를 곱함). */
    fun pcm16(): ShortArray {
        val w = wave()
        return ShortArray(w.size) { i -> (w[i] * AMPLITUDE * 32_767f).roundToInt().toShort() }
    }
}
