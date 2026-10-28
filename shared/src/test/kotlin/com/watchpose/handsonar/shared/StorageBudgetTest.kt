package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class StorageBudgetTest {
    @Test
    fun oneSecondIsSoundPlusSensors() {
        assertEquals(96_000 + 32_000, StorageBudget.BYTES_PER_SECOND)
    }

    @Test
    fun minutesFromFreeBytes() {
        assertEquals(10.0, StorageBudget.minutesLeft(128_000L * 600), 1e-9)
    }

    @Test
    fun warnsBelowTenMinutesAndStopsBelowOne() {
        val minute = StorageBudget.BYTES_PER_SECOND * 60L
        assertEquals(StorageBudget.Level.OK, StorageBudget.level(minute * 10))
        assertEquals(StorageBudget.Level.WARN, StorageBudget.level(minute * 10 - 1))
        assertEquals(StorageBudget.Level.WARN, StorageBudget.level(minute))
        assertEquals(StorageBudget.Level.STOP, StorageBudget.level(minute - 1))
    }
}
