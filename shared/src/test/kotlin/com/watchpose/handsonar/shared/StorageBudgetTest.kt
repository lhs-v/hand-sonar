package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class StorageBudgetTest {
    @Test
    fun minutesFromFreeBytes() {
        assertEquals(10.0, StorageBudget.minutesLeft(96_000L * 600), 1e-9)
    }

    @Test
    fun warnsBelowTenMinutesAndStopsBelowOne() {
        assertEquals(StorageBudget.Level.OK, StorageBudget.level(96_000L * 60 * 10))
        assertEquals(StorageBudget.Level.WARN, StorageBudget.level(96_000L * 60 * 10 - 1))
        assertEquals(StorageBudget.Level.WARN, StorageBudget.level(96_000L * 60))
        assertEquals(StorageBudget.Level.STOP, StorageBudget.level(96_000L * 60 - 1))
    }
}
