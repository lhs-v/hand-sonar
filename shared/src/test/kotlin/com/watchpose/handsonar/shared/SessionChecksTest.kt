package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

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
}
