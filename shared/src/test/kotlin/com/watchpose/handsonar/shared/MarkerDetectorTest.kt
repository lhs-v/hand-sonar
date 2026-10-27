package com.watchpose.handsonar.shared

import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MarkerDetectorTest {
    private fun noise(n: Int, sigma: Double, seed: Long): FloatArray {
        val r = Random(seed)
        return FloatArray(n) { (r.nextGaussian() * sigma).toFloat() }
    }

    @Test
    fun findsInsertedMarker() {
        val x = noise(48_000, 0.05, 1)
        val m = MarkerSpec.wave()
        val at = 12_345
        for (i in m.indices) x[at + i] += 0.5f * m[i]
        val hit = assertNotNull(MarkerDetector.find(x))
        assertEquals(at.toDouble(), hit.sample, 1.0)
        assertTrue(hit.ratio > 50, "ratio ${hit.ratio}")
    }

    @Test
    fun findsWeakMarkerInLouderNoise() {
        val x = noise(72_000, 0.3, 3) // 표지음 0.25: 봉우리 약 90, 잡음 상관 RMS 약 5.7 → 비 약 16
        val m = MarkerSpec.wave()
        for (i in m.indices) x[40_000 + i] += 0.25f * m[i]
        assertEquals(40_000.0, assertNotNull(MarkerDetector.find(x)).sample, 1.0)
    }

    @Test
    fun noiseOnlyGivesNothing() {
        assertNull(MarkerDetector.find(noise(48_000, 0.05, 2)))
    }

    @Test
    fun tooShortGivesNothing() {
        assertNull(MarkerDetector.find(FloatArray(100)))
    }
}
