package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class RecordClockRuleTest {
    @Test
    fun timestampLinesPassThrough() {
        val r = RecordClockRule()
        assertEquals(ClockLine(100, 5_000, 120), r.onSecond(true, 100, 5_000, 120, 9_999))
        assertEquals("timestamp", r.source)
    }

    @Test
    fun repeatedTimestampIsSkipped() {
        val r = RecordClockRule()
        r.onSecond(true, 100, 5_000, 120, 0)
        assertNull(r.onSecond(true, 100, 5_000, 48_120, 0))
        assertEquals(ClockLine(48_100, 1_000_005_000, 96_120), r.onSecond(true, 48_100, 1_000_005_000, 96_120, 0))
    }

    @Test
    fun threeFailuresBeforeAnySuccessSwitchToReadTime() {
        val r = RecordClockRule()
        assertNull(r.onSecond(false, 0, 0, 960, 1))
        assertNull(r.onSecond(false, 0, 0, 48_960, 2))
        assertEquals(ClockLine(96_960, 3, 96_960), r.onSecond(false, 0, 0, 96_960, 3))
        assertEquals("read", r.source)
        assertEquals(ClockLine(144_960, 4, 144_960), r.onSecond(true, 7, 7, 144_960, 4))
    }

    @Test
    fun failureAfterSuccessOnlySkips() {
        val r = RecordClockRule()
        r.onSecond(true, 1, 1, 1, 1)
        repeat(5) { assertNull(r.onSecond(false, 0, 0, 0, 0)) }
        assertEquals("timestamp", r.source)
    }
}
