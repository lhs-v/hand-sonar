package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LandmarkCsvTest {
    @Test
    fun headerHas130Columns() {
        val cols = LandmarkCsv.HEADER.split(',')
        assertEquals(130, cols.size)
        assertEquals(listOf("frameNs", "hands", "handedness", "score", "x0", "y0", "z0"), cols.take(7))
        assertEquals("wx0", cols[67])
        assertEquals("wz20", cols.last())
    }

    @Test
    fun rowWithHand() {
        val img = FloatArray(63) { it / 100f }
        val world = FloatArray(63) { -it / 1_000f }
        val f = LandmarkCsv.row(123L, 2, "Left", 0.9f, img, world).split(',')
        assertEquals(130, f.size)
        assertEquals(listOf("123", "2", "Left", "0.9"), f.take(4))
        assertEquals("0.01", f[5])
        assertEquals((-0.062f).toString(), f[129])
    }

    @Test
    fun rowWithoutHandKeepsColumnCount() {
        val f = LandmarkCsv.row(5L, 1, null, null, null, null).split(',')
        assertEquals(130, f.size)
        assertEquals(listOf("5", "1", "", ""), f.take(4))
        assertTrue(f.drop(4).all { it.isEmpty() })
    }
}
