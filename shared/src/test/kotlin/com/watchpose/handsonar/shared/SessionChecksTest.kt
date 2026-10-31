package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class SessionChecksTest {
    @Test
    fun cleanSessionHasNoFlags() {
        assertEquals(emptyList<String>(), SessionChecks.flags(0.95, 0.99, startMarkerFound = true, endMarkerFound = true))
    }

    @Test
    fun eachProblemAddsItsFlag() {
        assertEquals(
            listOf("suspect-hand", "suspect-watch", "marker-missing-start", "marker-missing-end"),
            SessionChecks.flags(0.89, 0.0, startMarkerFound = false, endMarkerFound = false),
        )
        assertEquals(listOf("suspect-watch"), SessionChecks.flags(0.9, 0.009, startMarkerFound = true, endMarkerFound = true))
        assertEquals(emptyList<String>(), SessionChecks.flags(0.9, 0.05, startMarkerFound = true, endMarkerFound = true)) // 손 든 세션의 실측 5%
    }

    @Test
    fun watchLinkClearsTheWatchSuspicionWhenTheChirpIsNotHeard() {
        assertEquals(emptyList<String>(), SessionChecks.flags(0.95, 0.0, true, true, watchLinkRatio = 0.9)) // 폰에 처프가 안 들려도 기울기를 받았다
        assertEquals(listOf("suspect-watch"), SessionChecks.flags(0.95, 0.0, true, true, watchLinkRatio = 0.2)) // 기울기도 거의 못 받음
        assertEquals(listOf("suspect-watch"), SessionChecks.flags(0.95, 0.0, true, true, watchLinkRatio = null)) // 기울기 칸 전 세션은 처프만 본다
    }

    @Test
    fun measuredFpsIsFrameGapsOverTheFirstToLastSpan() {
        // 301프레임 = 간격 300개가 10 s에 걸림 → 초당 30
        assertEquals(30.0, assertNotNull(SessionChecks.measuredFps(301, 0, 10_000_000_000)), 1e-9)
    }

    @Test
    fun measuredFpsIsNullWithoutASpan() {
        assertNull(SessionChecks.measuredFps(1, 5, 5)) // 프레임 하나
        assertNull(SessionChecks.measuredFps(5, 100, 100)) // 간격 0
        assertNull(SessionChecks.measuredFps(5, 200, 100)) // 마지막이 첫 프레임보다 앞
    }
}
