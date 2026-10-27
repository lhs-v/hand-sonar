package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MarkerSpecTest {
    private fun dftMagnitude(x: FloatArray, hz: Double): Double {
        var re = 0.0
        var im = 0.0
        for (i in x.indices) {
            val a = 2 * PI * hz * i / MarkerSpec.SAMPLE_RATE
            re += x[i] * cos(a)
            im -= x[i] * sin(a)
        }
        return hypot(re, im)
    }

    @Test
    fun fortyMillisecondsAt48k() {
        assertEquals(1_920, MarkerSpec.SAMPLES)
        assertEquals(1_920, MarkerSpec.wave().size)
        assertEquals(40.0, MarkerSpec.SAMPLES * 1_000.0 / MarkerSpec.SAMPLE_RATE)
    }

    @Test
    fun hannEndsAreZero() {
        val w = MarkerSpec.wave()
        assertEquals(0f, w.first(), 1e-6f)
        assertEquals(0f, w.last(), 1e-6f)
    }

    @Test
    fun energyIsInMarkerBand() {
        val w = MarkerSpec.wave()
        val inBand = dftMagnitude(w, 22_250.0)
        assertTrue(inBand > 10 * dftMagnitude(w, 20_000.0), "20 kHz")
        assertTrue(inBand > 10 * dftMagnitude(w, 15_000.0), "15 kHz")
    }

    @Test
    fun pcmPeakIsAmplitude() {
        val peak = MarkerSpec.pcm16().maxOf { abs(it.toInt()) }
        assertTrue(peak <= (0.8 * 32_767).toInt() + 1, "peak $peak")
        assertTrue(peak >= (0.75 * 32_767).toInt(), "peak $peak")
    }
}
