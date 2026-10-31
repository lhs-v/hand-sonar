package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SessionRunnerTest {
    private val s = 1_000_000_000L

    /** 자유 동작 없이 동작 72개, 구간 6개: 끝 표지음 156 s, 끝 156.8 s. */
    private val plan = PromptPlanner.plan(seed = 5L, freeMotionSec = 0)

    @Test
    fun fullRunTimeline() {
        val r = SessionRunner(plan)
        r.start(0)
        assertEquals(RunnerPhase.LEAD, r.phase)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(0)), r.tick(1 * s))
        val events = r.tick(1_000 * s)
        assertEquals(RunnerEvent.Finished, events.last())
        assertEquals(RunnerEvent.EndMarkerDue, events[events.size - 2])
        assertEquals(5, events.count { it == RunnerEvent.RestStarted })
        assertEquals(71, events.count { it is RunnerEvent.PromptStarted })
        assertEquals(RunnerPhase.FINISHED, r.phase)
        assertEquals(72, r.records.size)
        assertEquals(156_800_000_000L, r.endNs)
        val first = r.records.first()
        assertEquals(1 * s, first.startNs)
        assertEquals(3 * s, first.endNs)
        assertEquals("", first.flag)
        assertEquals(23 * s, r.records.first { it.segment == 2 }.startNs) // 1 + 20 + 2
        assertEquals((0 until 72).toList(), r.records.map { it.index })
        assertEquals(Labels.byId(first.classId).name, first.name)
        assertEquals(1f, r.progress(1_000 * s))
    }

    @Test
    fun calibrationWaitsForTheGateAndRecordsTheHold() {
        val r = SessionRunner(plan, calibrate = true)
        r.start(0)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.CalibrationStarted), r.tick(1 * s))
        assertEquals(RunnerPhase.CALIB, r.phase)
        assertTrue(r.calibrating)
        assertTrue(r.tick(60 * s).isEmpty()) // 시간으로 끝나지 않는다
        assertEquals(RunnerPhase.CALIB, r.phase)
        assertEquals(0.0f, r.progress(60 * s), 0.01f) // 기준 자세 시간은 진행률에 넣지 않는다(시작 1 s만)
        assertEquals(Labels.NEUTRAL_SIDE_ID, r.calibrationId) // 첫 차례는 옆면
        assertEquals(listOf<RunnerEvent>(RunnerEvent.CalibrationStarted), r.calibrated(holdStartNs = 0, nowNs = 61 * s)) // 옆면 → 정면 차례
        assertEquals(Pair(RunnerPhase.CALIB, Labels.NEUTRAL_ID), Pair(r.phase, r.calibrationId))
        assertEquals(0.0f, r.progress(62 * s), 0.01f) // 두 차례 모두 진행률에 넣지 않는다
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(0)), r.calibrated(holdStartNs = 61_500_000_000L, nowNs = 63 * s))
        assertEquals(RunnerPhase.PROMPT, r.phase)
        val (side, front) = r.records
        assertEquals(Labels.NEUTRAL_SIDE_ID, side.classId)
        assertEquals(Triple(1 * s, 61 * s, ""), Triple(side.startNs, side.endNs, side.flag)) // 유지 시작은 차례 시작보다 앞일 수 없다
        assertEquals(Labels.NEUTRAL_ID, front.classId)
        assertEquals(Pair(61_500_000_000L, 63 * s), Pair(front.startNs, front.endNs))
        val events = r.tick(1_000 * s)
        assertEquals(RunnerEvent.Finished, events.last())
        assertEquals(74, r.records.size)
        assertEquals((0 until 74).toList(), r.records.map { it.index })
        assertEquals(Pair(63 * s, 65 * s), Pair(r.records[2].startNs, r.records[2].endNs)) // 첫 동작은 기준 자세가 끝난 63 s부터 2 s
        assertEquals(156_800_000_000L + 62 * s, r.endNs) // 기준 자세 62 s만큼 늦게 끝난다
    }

    @Test
    fun skippedCalibrationIsRecordedWholeAndPauseRestartsIt() {
        val r = SessionRunner(plan, calibrate = true)
        r.start(0)
        r.tick(1 * s)
        r.pause(4 * s)
        assertTrue(r.calibrating)
        assertEquals(-1, r.resumePrompt)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.CalibrationStarted), r.resume(10 * s))
        assertEquals(RunnerPhase.CALIB, r.phase)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(0)), r.skipCalibration(15 * s))
        val pause = r.records[0]
        val skip = r.records[1]
        assertEquals(Triple(Labels.PAUSE_ID, 4 * s, 10 * s), Triple(pause.classId, pause.startNs, pause.endNs))
        assertEquals(Triple(Labels.NEUTRAL_SIDE_ID, 10 * s, 15 * s), Triple(skip.classId, skip.startNs, skip.endNs)) // 다시 시작한 차례 전체
        assertEquals(RunnerPhase.PROMPT, r.phase) // 남은 정면 차례도 건너뛴다
        assertEquals(2, r.records.size)
        assertEquals(SessionRunner.FLAG_SKIP, skip.flag)
        assertTrue(r.skipCalibration(16 * s).isEmpty()) // CALIB가 아니면 아무것도 하지 않는다
        assertTrue(r.calibrated(0, 16 * s).isEmpty())
    }

    @Test
    fun holdStartBeforeARestartedCalibrationIsClipped() {
        val r = SessionRunner(plan, calibrate = true)
        r.start(0)
        r.tick(1 * s)
        r.pause(2 * s)
        r.resume(5 * s)
        r.calibrated(holdStartNs = 3 * s, nowNs = 8 * s)
        val calib = r.records.last { it.classId == Labels.NEUTRAL_SIDE_ID }
        assertEquals(Pair(5 * s, 8 * s), Pair(calib.startNs, calib.endNs))
        r.pause(9 * s) // 정면 차례에서 일시정지하면 다시 시작할 때 정면부터
        r.resume(12 * s)
        assertEquals(Pair(RunnerPhase.CALIB, Labels.NEUTRAL_ID), Pair(r.phase, r.calibrationId))
    }

    @Test
    fun withoutCalibrationLeadGoesStraightToTheFirstPrompt() {
        val r = SessionRunner(plan)
        r.start(0)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(0)), r.tick(1 * s))
        assertFalse(r.calibrating)
    }

    @Test
    fun tickBetweenBoundariesGivesNoEvents() {
        val r = SessionRunner(plan)
        r.start(0)
        assertTrue(r.tick(s / 2).isEmpty())
        r.tick(1 * s)
        assertTrue(r.tick(2 * s).isEmpty())
        assertEquals(0.5f, r.promptProgress(2 * s))
    }

    @Test
    fun pauseCutsPromptAndResumeRepeatsIt() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(1 * s)
        r.pause(1_500_000_000L)
        assertEquals(RunnerPhase.PAUSED, r.phase)
        assertTrue(r.tick(10 * s).isEmpty())
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(0)), r.resume(5 * s))
        r.tick(7 * s)
        val cut = r.records[0]
        val pause = r.records[1]
        val redo = r.records[2]
        assertEquals(Triple(1 * s, 1_500_000_000L, "pause"), Triple(cut.startNs, cut.endNs, cut.flag))
        assertEquals(Labels.PAUSE_ID, pause.classId)
        assertEquals(Triple(1_500_000_000L, 5 * s, "pause"), Triple(pause.startNs, pause.endNs, pause.flag))
        assertEquals(cut.classId, redo.classId)
        assertEquals(Triple(5 * s, 7 * s, ""), Triple(redo.startNs, redo.endNs, redo.flag))
    }

    @Test
    fun pauseDuringRestResumesWithNextPrompt() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(21 * s) // 구간 1이 21 s에 끝나고 쉼
        assertEquals(RunnerPhase.REST, r.phase)
        r.pause(21_500_000_000L)
        assertEquals(listOf<RunnerEvent>(RunnerEvent.PromptStarted(10)), r.resume(30 * s))
        assertEquals(2, r.plan.prompts[10].segment)
    }

    @Test
    fun wrongFlagsOnlyCurrentPrompt() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(1 * s)
        r.markWrong()
        r.tick(5 * s)
        assertEquals("wrong", r.records[0].flag)
        assertEquals("", r.records[1].flag)
    }

    @Test
    fun cancelStopsEverything() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(1 * s)
        r.cancel(2 * s)
        assertEquals(RunnerPhase.CANCELLED, r.phase)
        assertEquals(1, r.records.size)
        assertEquals(2 * s, r.records[0].endNs)
        assertTrue(r.tick(100 * s).isEmpty())
    }

    @Test
    fun progressExcludesPause() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(10 * s)
        r.pause(10 * s)
        val p = r.progress(10 * s)
        assertEquals(p, r.progress(50 * s))
        assertEquals(10.0 / 156.0, p.toDouble(), 1e-6)
    }

    @Test
    fun cancelWhilePausedExcludesPause() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(10 * s)
        r.pause(10 * s)
        r.cancel(20 * s)
        assertEquals(10.0 / 156.0, r.progress(20 * s).toDouble(), 1e-6)
        assertEquals(Labels.PAUSE_ID, r.records.last().classId)
        assertEquals(20 * s, r.records.last().endNs)
    }

    @Test
    fun resumePromptTellsWhatComesBack() {
        val r = SessionRunner(plan)
        r.start(0)
        r.pause(s / 2) // LEAD 도중
        assertEquals(0, r.resumePrompt)
        r.resume(1 * s)
        r.tick(3 * s + s / 2) // 동작 0 [1, 3] 끝, 동작 1 도중
        r.pause(4 * s)
        assertEquals(1, r.resumePrompt)
    }

    @Test
    fun pauseDuringTailResumesTail() {
        val r = SessionRunner(plan)
        r.start(0)
        r.tick(155_500_000_000L) // 마지막 동작이 155 s에 끝나고 TAIL(155–156 s)
        assertEquals(RunnerPhase.TAIL, r.phase)
        r.pause(155_500_000_000L)
        assertEquals(-1, r.resumePrompt)
        assertTrue(r.resume(200 * s).isEmpty())
        assertTrue(r.tick(201 * s - 1).isEmpty())
        assertEquals(listOf<RunnerEvent>(RunnerEvent.EndMarkerDue), r.tick(201 * s))
    }

    @Test
    fun activeTimeExcludesPauseAndStartsAtZero() {
        val r = SessionRunner(plan)
        assertEquals(0L, r.activeNs(5 * s))
        r.start(0)
        r.tick(10 * s)
        r.pause(10 * s)
        assertEquals(10 * s, r.activeNs(50 * s))
        r.resume(50 * s)
        assertEquals(15 * s, r.activeNs(55 * s))
    }

    @Test
    fun cancelBeforeStartHasNoProgress() {
        val r = SessionRunner(plan)
        r.cancel(5 * s)
        assertEquals(0f, r.progress(5 * s))
        assertEquals(0L, r.activeNs(5 * s))
    }
}
