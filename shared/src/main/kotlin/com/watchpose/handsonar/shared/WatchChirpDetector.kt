package com.watchpose.handsonar.shared

import kotlin.math.abs
import kotlin.math.min

/** 폰 마이크로 들은 워치 처프 상태(설계 §6.5). [id]는 기록에 적는 이름. */
enum class ChirpHeard(val id: String) { HEARD("heard"), WEAK("weak"), NONE("none") }

data class ChirpCheck(val ratio: Double, val heard: ChirpHeard)

/**
 * 0.5 s 녹음에 워치 처프(18–21 kHz, 600샘플 주기)가 있는지 본다(설계 §6.5). 워치는 처프를 쉬지 않고 반복하므로 녹음을
 * 600샘플 주기로 접어 더하면(동기 평균) 처프는 쌓이고 잡음은 덜 쌓인다. 접은 600샘플을 송신 처프와 원형 상관해,
 * 봉우리 / 바닥(봉우리 ±[GUARD]칸 밖 |상관|의 중앙값 ÷ 0.6745)을 잰다. 잡음만 있으면 약 3이다. 바닥을 중앙값으로 잡아
 * 반사 경로가 있어도 비가 갇히지 않는다. 두 기기 시계 차이가 30 ppm 안이면 거의 그대로(−1.2 dB) 쌓이고, 50 ppm이면 −3.5 dB,
 * 100 ppm 가까이면 쌓이지 않는다(이 기기 쌍 실측 1.8–3.7 ppm). 기준값은 실기기에서 워치를 켜고 끈 값으로 다시 정한다(계획 작업 10).
 */
object WatchChirpDetector {
    const val BLOCK_SAMPLES = 24_000
    const val HEARD_RATIO = 8.0
    const val WEAK_RATIO = 5.0
    const val GUARD = 32
    private const val MAX_RATIO = 999.0

    private val template: FloatArray by lazy { Sweep.wave(SweepWindow.HANN) }

    fun check(block: ShortArray, count: Int = block.size): ChirpCheck {
        val n = SonarSpec.SWEEP_SAMPLES
        val fold = DoubleArray(n)
        for (i in 0 until count) fold[i % n] += block[i].toDouble()
        val corr = DoubleArray(n)
        var best = 0
        for (k in 0 until n) {
            var c = 0.0
            var idx = k
            for (j in 0 until n) {
                c += fold[idx] * template[j]
                idx++
                if (idx == n) idx = 0
            }
            corr[k] = abs(c)
            if (corr[k] > corr[best]) best = k
        }
        // 바닥: 봉우리 ±GUARD 밖 |상관|의 중앙값 ÷ 0.6745(가우스 σ 환산). RMS를 쓰면 반사 경로의 봉우리가 바닥에 들어가
        // 다중 경로에서 비가 약 5에 갇힌다(2026-10-22 같은 기기 쌍 실측 경로로 모의, 계획 2 작업 3 검토).
        val outside = DoubleArray(n)
        var used = 0
        for (k in 0 until n) {
            val d = abs(k - best)
            if (min(d, n - d) <= GUARD) continue
            outside[used++] = corr[k]
        }
        outside.sort(0, used)
        val median = if (used % 2 == 1) outside[used / 2] else 0.5 * (outside[used / 2 - 1] + outside[used / 2])
        val floor = median / 0.6745
        val peak = corr[best]
        val ratio = when {
            peak <= 0.0 -> 0.0
            floor <= 0.0 -> MAX_RATIO
            else -> min(peak / floor, MAX_RATIO)
        }
        val heard = when {
            ratio >= HEARD_RATIO -> ChirpHeard.HEARD
            ratio >= WEAK_RATIO -> ChirpHeard.WEAK
            else -> ChirpHeard.NONE
        }
        return ChirpCheck(ratio, heard)
    }
}
