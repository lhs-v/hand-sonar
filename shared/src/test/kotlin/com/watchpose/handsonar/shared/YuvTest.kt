package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class YuvTest {
    private fun u8(b: Byte) = b.toInt() and 0xFF
    private fun solid(c: Int) = Yuv.argbToI420(IntArray(16) { c }, 4, 4)

    @Test
    fun whiteAndBlack() {
        val w = solid(0xFFFFFFFF.toInt())
        assertEquals(235, u8(w.y[0]))
        assertEquals(128, u8(w.u[0]))
        assertEquals(128, u8(w.v[0]))
        val k = solid(0xFF000000.toInt())
        assertEquals(16, u8(k.y[5]))
        assertEquals(128, u8(k.u[3]))
    }

    @Test
    fun pureRedFollowsBt601() {
        val r = solid(0xFFFF0000.toInt())
        assertEquals(82, u8(r.y[0]))
        assertEquals(90, u8(r.u[0]))
        assertEquals(240, u8(r.v[0]))
    }

    @Test
    fun planeSizes() {
        val o = I420(6, 4)
        assertEquals(24, o.y.size)
        assertEquals(6, o.u.size)
        assertEquals(6, o.v.size)
    }

    @Test
    fun chromaAveragesTwoByTwo() {
        val px = IntArray(16) { 0xFF000000.toInt() }
        px[0] = 0xFFFF0000.toInt() // 왼쪽 위 2×2 중 한 칸만 빨강 → 평균 r = 64
        val o = Yuv.argbToI420(px, 4, 4)
        assertEquals(156, u8(o.v[0])) // ((112·64 + 128) >> 8) + 128
        assertEquals(128, u8(o.v[1]))
    }
}
