package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class TiltMessageTest {
    @Test
    fun roundTripsTwentyBytes() {
        val t = Tilt(123_456_789_012L, 6.9f, -2.7f, 6.3f)
        val bytes = TiltMessage.encode(t)
        assertEquals(20, bytes.size)
        assertEquals(t, TiltMessage.decode(bytes))
    }

    @Test
    fun rejectsWrongSizeAndNonFiniteValues() {
        assertNull(TiltMessage.decode(ByteArray(19)))
        assertNull(TiltMessage.decode(TiltMessage.encode(Tilt(1L, Float.NaN, 0f, 9.8f))))
    }
}
