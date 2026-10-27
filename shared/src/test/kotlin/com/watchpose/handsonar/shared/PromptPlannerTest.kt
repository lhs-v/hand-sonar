package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class PromptPlannerTest {
    @Test
    fun segmentSizesFollowDesignTable() {
        val p = PromptPlanner.plan(seed = 1L)
        assertEquals(listOf(10, 10, 20, 20, 8, 4, 1), p.segments.map { it.size })
        assertEquals((1..7).toList(), p.segments.map { it.first().segment })
        assertEquals(Labels.VERSION, p.version)
    }

    @Test
    fun eachFingerAndDigitTwicePerSegment() {
        val p = PromptPlanner.plan(seed = 7L)
        for (s in listOf(0, 1)) assertEquals((0..4).flatMap { listOf(it, it) }, p.segments[s].map { it.classId }.sorted())
        for (s in listOf(2, 3)) assertEquals((5..14).flatMap { listOf(it, it) }, p.segments[s].map { it.classId }.sorted())
        assertEquals(listOf(15, 15, 15, 16, 16, 16, 17, 17), p.segments[4].map { it.classId }.sorted())
        assertEquals(listOf(15, 15, 16, 17), p.segments[5].map { it.classId }.sorted())
        assertEquals(setOf(2_000L), p.segments.take(6).flatten().map { it.durationMs }.toSet())
    }

    @Test
    fun sameSeedSameOrderAndDifferentSeedDifferentOrder() {
        assertEquals(PromptPlanner.plan(42L), PromptPlanner.plan(42L))
        assertNotEquals(PromptPlanner.plan(42L).prompts, PromptPlanner.plan(43L).prompts)
    }

    @Test
    fun plannedLengthIsAboutThreePointSixMinutes() {
        // 1 + (10+10+20+20+8+4)×2 + 60 + 6×2 + 1 = 218 s
        assertEquals(218_000L, PromptPlanner.plan(1L).plannedMs)
        // 자유 동작 0 s면 구간 6개, 사이 5번: 1 + 144 + 10 + 1 = 156 s
        val noFree = PromptPlanner.plan(1L, freeMotionSec = 0)
        assertEquals(6, noFree.segments.size)
        assertEquals(156_000L, noFree.plannedMs)
    }

    @Test
    fun freeMotionIsOnePromptWithFreeId() {
        assertEquals(PlannedPrompt(7, Labels.FREE_ID, 90_000L), PromptPlanner.plan(3L, freeMotionSec = 90).prompts.last())
    }
}
