package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class SonarSpecTest {
    @Test
    fun sweepMatchesPublicDataset() {
        assertEquals(48_000, SonarSpec.SAMPLE_RATE)
        assertEquals(18_000.0, SonarSpec.START_HZ)
        assertEquals(3_000.0, SonarSpec.BANDWIDTH_HZ)
        assertEquals(600, SonarSpec.SWEEP_SAMPLES)
    }

    @Test
    fun partIsOneMinuteOf16BitMono() {
        assertEquals(2_880_000, SonarSpec.PART_FRAMES)
        assertEquals(96_000, SonarSpec.BYTES_PER_SECOND)
    }

    @Test
    fun windowIdsAreRecJsonNames() {
        assertEquals(listOf("none", "hann"), SweepWindow.entries.map { it.id })
    }

    @Test
    fun defaultsAreHannAndHalfAmplitude() {
        assertEquals(SweepWindow.HANN, SonarSpec.DEFAULT_WINDOW)
        assertEquals(0.5f, SonarSpec.DEFAULT_AMPLITUDE)
    }

    @Test
    fun amplitudeStepsByTenPercentAndWraps() {
        assertEquals(0.6f, SonarSpec.nextAmplitude(0.5f))
        assertEquals(0.2f, SonarSpec.nextAmplitude(0.1f))
        assertEquals(0.1f, SonarSpec.nextAmplitude(1.0f))
    }

    @Test
    fun autoStopCyclesOffFiveTenThirty() {
        assertEquals(5, SonarSpec.nextAutoStop(0))
        assertEquals(10, SonarSpec.nextAutoStop(5))
        assertEquals(30, SonarSpec.nextAutoStop(10))
        assertEquals(0, SonarSpec.nextAutoStop(30))
        assertEquals(0, SonarSpec.nextAutoStop(7))
    }
}
