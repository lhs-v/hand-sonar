package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SweepTest {
    private fun crossings(s: FloatArray, range: IntRange): Int = range.drop(1).count { (s[it - 1] < 0f) != (s[it] < 0f) }

    @Test
    fun oneSweepIs600SamplesWithinUnitAmplitude() {
        for (w in SweepWindow.entries) {
            for (shape in SweepShape.entries) {
                val s = Sweep.wave(w, shape)
                assertEquals(600, s.size)
                assertTrue(s.all { it in -1f..1f })
            }
        }
    }

    @Test
    fun zeroCrossingsMatchTheAverageFrequency() {
        // 18→21 kHz의 평균 19.5 kHz × 12.5 ms = 243.75주기 → 부호가 약 487번 바뀐다
        val s = Sweep.wave(SweepWindow.NONE)
        val n = crossings(s, s.indices)
        assertTrue(n in 484..490, "부호 바뀜 $n")
    }

    @Test
    fun frequencyRisesWithinTheSweep() {
        val s = Sweep.wave(SweepWindow.NONE)
        assertTrue(crossings(s, 0 until 300) < crossings(s, 300 until 600))
    }

    @Test
    fun hannEndsAtZeroAndNoneDoesNot() {
        assertEquals(0f, Sweep.wave(SweepWindow.HANN).last(), 1e-6f)
        assertTrue(abs(Sweep.wave(SweepWindow.NONE).last()) > 0.01f)
    }

    @Test
    fun tukeyTapersOnlyTheEnds() {
        // 양 끝 60샘플만 0.5 − 0.5·cos(π·j/60)을 곱하고(j = 끝에서 센 자리, 0이면 0) 가운데는 창 없음과 같다
        val tukey = Sweep.wave(SweepWindow.TUKEY)
        val none = Sweep.wave(SweepWindow.NONE)
        assertEquals(0f, tukey.first(), 1e-6f)
        assertEquals(0f, tukey.last(), 1e-6f)
        for (i in 60 until 540) assertEquals(none[i], tukey[i], 1e-6f)
        for (j in listOf(1, 30, 59)) {
            val w = 0.5 - 0.5 * kotlin.math.cos(kotlin.math.PI * j / 60)
            assertEquals((none[j] * w).toFloat(), tukey[j], 1e-6f)
            assertEquals((none[599 - j] * w).toFloat(), tukey[599 - j], 1e-6f)
        }
    }

    @Test
    fun pcm16ScalesAndClampsAmplitude() {
        val peak = Sweep.pcm16(SweepWindow.NONE, 0.5f).maxOf { abs(it.toInt()) }
        assertTrue(peak in 16_000..16_384, "최대 $peak")
        assertTrue(Sweep.pcm16(SweepWindow.NONE, 1.5f).maxOf { abs(it.toInt()) } <= 32_767)
    }

    @Test
    fun triangleRisesToTheTopAndComesBack() {
        assertEquals(18_000.0, Sweep.instantFrequency(0, SweepShape.TRIANGLE))
        assertEquals(SonarSpec.TRIANGLE_TOP_HZ, Sweep.instantFrequency(300, SweepShape.TRIANGLE))
        assertEquals(18_000.0, Sweep.instantFrequency(600, SweepShape.TRIANGLE))
        // 꼭대기(가운데) 근처가 양 끝보다 부호가 자주 바뀐다
        val s = Sweep.wave(SweepWindow.NONE, SweepShape.TRIANGLE)
        assertTrue(crossings(s, 0 until 100) < crossings(s, 250 until 350))
        assertTrue(crossings(s, 500 until 600) < crossings(s, 250 until 350))
    }

    @Test
    fun triangleLoopsWithoutAPhaseJumpButSawDoesNot() {
        // 한 주기를 다 돈 위상: 삼각은 244바퀴라 다음 주기 첫 샘플(위상 0)과 이어지고, 톱니는 243.75바퀴라 4분의 1 바퀴 어긋난다
        assertEquals(244.0, Sweep.phase(600, SweepShape.TRIANGLE) / (2 * PI), 1e-9)
        assertEquals(243.75, Sweep.phase(600, SweepShape.SAW) / (2 * PI), 1e-9)
        // 꼭대기에서도 위상 증가가 이어진다(앞 반 식과 뒤 반 식이 맞물림)
        val before = Sweep.phase(300, SweepShape.TRIANGLE) - Sweep.phase(299, SweepShape.TRIANGLE)
        val after = Sweep.phase(301, SweepShape.TRIANGLE) - Sweep.phase(300, SweepShape.TRIANGLE)
        assertEquals(before, after, 1e-6)
    }

    @Test
    fun instantFrequencyRunsFrom18To21kHz() {
        assertEquals(18_000.0, Sweep.instantFrequency(0))
        assertEquals(21_000.0, Sweep.instantFrequency(600))
    }
}
