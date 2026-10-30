package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class LabelsTest {
    @Test
    fun eighteenGesturesInWatchHandOrder() {
        assertEquals((0..17).toList(), Labels.ALL.map { it.id })
        assertEquals(18, Labels.ALL.map { it.key }.toSet().size)
        assertEquals((0..17).toList(), (Labels.FINGERS + Labels.ASL + Labels.WRIST).toList())
        assertEquals(listOf("thumb", "index", "middle", "ring", "pinky"), Labels.FINGERS.map { Labels.byId(it).key })
        assertEquals("asl_0", Labels.byId(5).key)
        assertEquals("asl_9", Labels.byId(14).key)
        assertEquals(listOf("wrist_up", "wrist_down", "wrist_left"), Labels.WRIST.map { Labels.byId(it).key })
    }

    @Test
    fun specialIdsDoNotCollide() {
        assertEquals(listOf("free", "pause", "neutral"), listOf(Labels.FREE_ID, Labels.PAUSE_ID, Labels.NEUTRAL_ID).map { Labels.byId(it).key })
        assertEquals(3, setOf(Labels.FREE_ID, Labels.PAUSE_ID, Labels.NEUTRAL_ID).size)
        assertEquals(WristPose.NEUTRAL, Labels.NEUTRAL.shape.wrist)
    }

    @Test
    fun fingerFlexBendsOnlyThatFinger() {
        for (id in Labels.FINGERS) {
            val f = Labels.byId(id).shape.fingers
            assertEquals(FingerState.BENT, f[id])
            assertEquals(4, f.count { it == FingerState.EXTENDED })
        }
    }

    @Test
    fun aslShapes() {
        val e = FingerState.EXTENDED
        assertEquals(List(5) { e }, Labels.byId(10).shape.fingers) // 숫자 5
        assertEquals(listOf(FingerState.BENT, e, FingerState.BENT, FingerState.BENT, FingerState.BENT), Labels.byId(6).shape.fingers) // 숫자 1
        assertEquals(FingerState.TOUCH, Labels.byId(11).shape.fingers[4]) // 숫자 6: 엄지에 소지
    }

    @Test
    fun wristShapes() {
        assertEquals(listOf(WristPose.UP, WristPose.DOWN, WristPose.LEFT), Labels.WRIST.map { Labels.byId(it).shape.wrist })
        assertEquals(WristPose.NEUTRAL, Labels.byId(0).shape.wrist)
    }

    @Test
    fun specialIds() {
        assertEquals("자유 동작", Labels.byId(Labels.FREE_ID).name)
        assertEquals("일시정지", Labels.byId(Labels.PAUSE_ID).name)
        assertFailsWith<IllegalArgumentException> { Labels.byId(18) }
    }
}
