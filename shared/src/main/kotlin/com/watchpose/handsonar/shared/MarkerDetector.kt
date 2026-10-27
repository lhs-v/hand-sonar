package com.watchpose.handsonar.shared

import kotlin.math.abs
import kotlin.math.sqrt

/** 찾은 표지음. [sample]은 녹음 안에서 표지음이 시작하는 샘플(소수), [ratio]는 상관 봉우리 / 봉우리 둘레를 뺀 상관 RMS. */
data class MarkerHit(val sample: Double, val ratio: Double)

/**
 * 녹음에서 표지음을 찾는다(설계 §6.5 자기 확인, PC의 markers.py와 같은 방식): 표지음 파형과의 상관(정합 필터) 절댓값이
 * 가장 큰 칸을 찾고 이웃 세 점으로 포물선 보간한다. 봉우리 ±[GUARD]칸을 뺀 상관 RMS의 [MIN_RATIO]배보다 작으면 없음으로 본다.
 * 잡음만 있으면 수만 칸에서 봉우리/RMS가 약 4(대개 3.5–5.5)다. 22 kHz 반송파 탓에 위치는 ±7칸(0.15 ms)쯤 흔들린다.
 * 폰의 확인·카메라 맞춤(33 ms)에는 충분하고, PC의 정밀 맞춤은 해석 신호 포락선 위에서 따로 찾는다.
 */
object MarkerDetector {
    const val MIN_RATIO = 8.0
    const val GUARD = 64

    fun find(x: FloatArray, template: FloatArray = MarkerSpec.wave()): MarkerHit? {
        if (x.size < template.size + 2 * GUARD) return null
        val c = Fft.correlateValid(x, template)
        var best = 0
        for (k in c.indices) if (abs(c[k]) > abs(c[best])) best = k
        var sumSq = 0.0
        var count = 0
        for (k in c.indices) {
            if (abs(k - best) <= GUARD) continue
            sumSq += c[k] * c[k]
            count++
        }
        if (count == 0) return null
        val rms = sqrt(sumSq / count)
        val peak = abs(c[best])
        if (rms <= 0.0 || peak < MIN_RATIO * rms) return null
        var offset = 0.0
        if (best in 1 until c.size - 1) {
            val a = abs(c[best - 1])
            val d = abs(c[best + 1])
            val den = a - 2 * peak + d
            if (den != 0.0) offset = (0.5 * (a - d) / den).coerceIn(-0.5, 0.5)
        }
        return MarkerHit(best + offset, peak / rms)
    }
}
