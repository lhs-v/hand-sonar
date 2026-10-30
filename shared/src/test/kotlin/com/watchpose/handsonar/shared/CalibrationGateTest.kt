package com.watchpose.handsonar.shared

import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CalibrationGateTest {
    private val ms = 1_000_000L

    private fun input(tMs: Long, hand: Double? = 3.0, fingers: Boolean = true, forearm: Double? = 4.0) =
        CalibrationInput(tMs * ms, hand, fingers, forearm)

    @Test
    fun passesAfterTwoSecondsOfHoldingAndReportsMedians() {
        val g = CalibrationGate()
        var last: CalibrationState? = null
        for (t in 0L..2_000L step 50) last = g.update(input(t, hand = if (t < 1_000) 2.0 else 6.0))
        val s = assertNotNull(last)
        assertTrue(s.done)
        assertEquals(0L, s.holdStartNs)
        assertEquals(2_000L, s.heldMs)
        assertEquals(6.0, s.handMedianDeg) // 41프레임 중 21개가 6°
        assertEquals(4.0, s.forearmMedianDeg)
    }

    @Test
    fun forearmTiltedBlocksAndBriefDropoutsAreForgiven() {
        val g = CalibrationGate()
        assertFalse(g.update(input(0, forearm = 45.0)).done) // 평소 수집 자세: 팔뚝 45° 기울기
        assertNull(g.update(input(50, forearm = 45.0)).holdStartNs)
        g.update(input(100))
        g.update(input(300, fingers = false)) // 한 프레임 손가락 판정 흔들림(0.2 s < 0.3 s)
        val kept = g.update(input(400))
        assertEquals(100 * ms, kept.holdStartNs)
        g.update(input(450, hand = 20.0))
        val lost = g.update(input(800, hand = 20.0)) // 0.35 s 동안 손이 기울어짐 → 처음부터
        assertNull(lost.holdStartNs)
        assertFalse(lost.handOk)
    }

    @Test
    fun withoutWatchTiltOnlyTheHandIsChecked() {
        val g = CalibrationGate()
        var s: CalibrationState? = null
        for (t in 0L..2_000L step 100) s = g.update(input(t, forearm = null))
        assertTrue(assertNotNull(s).done)
        assertNull(s.forearmOk)
        assertNull(s.forearmMedianDeg)
    }

    @Test
    fun projectedUpFollowsDisplayRotation() {
        // 폰을 세워 두고(기기 y가 위) 뒤로 15° 눕힘: 화면 회전 0이면 그림 위쪽(y −1)
        val upright = floatArrayOf(0f, (9.8 * cos(Math.toRadians(15.0))).toFloat(), (9.8 * sin(Math.toRadians(15.0))).toFloat())
        val up0 = assertNotNull(CalibrationGeometry.projectedUp(upright, 0))
        assertEquals(0.0, up0[0], 1e-9)
        assertEquals(-1.0, up0[1], 1e-9)
        // 기기를 시계 반대로 90° 돌림(ROTATION_90): 기기 +x가 위
        val up1 = assertNotNull(CalibrationGeometry.projectedUp(floatArrayOf(9.8f, 0f, 0f), 1))
        assertEquals(0.0, up1[0], 1e-9)
        assertEquals(-1.0, up1[1], 1e-9)
        // 폰이 오른쪽으로 10° 기울면 위쪽이 그림에서 왼쪽으로 기운다
        val tilted = assertNotNull(CalibrationGeometry.projectedUp(floatArrayOf((-9.8 * sin(Math.toRadians(10.0))).toFloat(), (9.8 * cos(Math.toRadians(10.0))).toFloat(), 0f), 0))
        assertEquals(-sin(Math.toRadians(10.0)), tilted[0], 1e-6)
        assertNull(CalibrationGeometry.projectedUp(floatArrayOf(0f, 0f, 9.8f), 0)) // 폰을 눕힘
    }

    @Test
    fun handAngleUsesPixelsNotNormalizedUnits() {
        val image = FloatArray(63)
        // 손목 (0.5, 0.8), 중지 MCP (0.5 + 30/480, 0.8 − 90/640): 픽셀로 (30, −90) → 수직에서 18.43°
        image[0] = 0.5f; image[1] = 0.8f
        image[27] = 0.5f + 30f / 480f; image[28] = 0.8f - 90f / 640f
        val deg = assertNotNull(CalibrationGeometry.handInPlaneDeg(image, 480, 640, doubleArrayOf(0.0, -1.0)))
        assertEquals(Math.toDegrees(kotlin.math.atan(30.0 / 90.0)), deg, 1e-4)
        assertNull(CalibrationGeometry.handInPlaneDeg(FloatArray(63), 480, 640, doubleArrayOf(0.0, -1.0)))
    }

    @Test
    fun fingersStraightNeedsAllFourAndIgnoresWholeHandScale() {
        fun hand(scale: Float, bentPinky: Boolean): FloatArray {
            val p = FloatArray(63)
            fun put(i: Int, x: Float, y: Float) { p[3 * i] = 0.5f + x * scale; p[3 * i + 1] = 0.8f + y * scale }
            put(0, 0f, 0f)
            for ((k, mcp) in listOf(5, 9, 13, 17).withIndex()) {
                val x = (k - 1.5f) * 0.03f
                put(mcp, x, -0.2f)
                put(mcp + 3, x, if (mcp == 17 && bentPinky) -0.18f else -0.38f)
            }
            return p
        }
        assertTrue(CalibrationGeometry.fingersStraight(hand(1f, false), 480, 640))
        assertTrue(CalibrationGeometry.fingersStraight(hand(0.5f, false), 480, 640)) // 손이 멀어져 작아져도 비는 같다
        assertFalse(CalibrationGeometry.fingersStraight(hand(1f, true), 480, 640))
    }

    @Test
    fun forearmAngleFromWatchGravity() {
        assertEquals(0.0, assertNotNull(CalibrationGeometry.forearmDeg(floatArrayOf(9.8f, 0f, 0f))), 1e-6)
        assertEquals(0.0, assertNotNull(CalibrationGeometry.forearmDeg(floatArrayOf(-9.8f, 0f, 0f))), 1e-6) // 버튼을 반대로 참
        assertEquals(45.0, assertNotNull(CalibrationGeometry.forearmDeg(floatArrayOf(6.93f, 0f, 6.93f))), 1e-3)
        assertNull(CalibrationGeometry.forearmDeg(floatArrayOf(0f, 0f, 0f)))
    }
}
