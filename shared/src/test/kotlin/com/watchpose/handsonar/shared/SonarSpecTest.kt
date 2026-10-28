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

    @Test
    fun recordVolumeCyclesKeepElevenThirteenFifteen() {
        assertEquals(listOf(0, 11, 13, 15), SonarSpec.RECORD_VOLUME_CHOICES)
        assertEquals(11, SonarSpec.nextRecordVolume(0))
        assertEquals(13, SonarSpec.nextRecordVolume(11))
        assertEquals(15, SonarSpec.nextRecordVolume(13))
        assertEquals(0, SonarSpec.nextRecordVolume(15))
        assertEquals(0, SonarSpec.nextRecordVolume(9)) // 선택지에 없는 값은 그대로(0)로
    }

    @Test
    fun recordVolumeTargetKeepsCurrentOrSetsTheChoiceWithinMax() {
        assertEquals(7, SonarSpec.recordVolumeTarget(choice = 0, current = 7, max = 15)) // 그대로
        assertEquals(13, SonarSpec.recordVolumeTarget(choice = 13, current = 7, max = 15)) // 올림
        assertEquals(11, SonarSpec.recordVolumeTarget(choice = 11, current = 15, max = 15)) // 고른 값이면 낮추기도 한다
        assertEquals(10, SonarSpec.recordVolumeTarget(choice = 15, current = 3, max = 10)) // 최대 단계 안으로
    }
}
