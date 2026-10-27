package com.watchpose.handsonar.shared

/** prompts.csv 한 줄(설계 §7.2). [flag]는 "", "wrong"(틀림 표시), "pause"(일시정지로 끊긴 동작 또는 일시정지 구간). */
data class PromptRecord(
    val index: Int,
    val segment: Int,
    val classId: Int,
    val name: String,
    val startNs: Long,
    val endNs: Long,
    val flag: String,
)

/** [SessionRunner]가 단계를 바꿀 때 내는 사건. */
sealed interface RunnerEvent {
    /** SessionPlan.prompts의 [promptIndex]번 동작을 보이기 시작했다. */
    data class PromptStarted(val promptIndex: Int) : RunnerEvent

    data object RestStarted : RunnerEvent

    /** 마지막 동작 뒤 1 s가 지났다: 지금 끝 표지음을 낸다. */
    data object EndMarkerDue : RunnerEvent

    /** 끝 표지음이 녹음될 시간([SessionRunner.CLOSING_MS])까지 지났다: 녹음을 멈춘다. */
    data object Finished : RunnerEvent
}

enum class RunnerPhase { IDLE, LEAD, PROMPT, REST, TAIL, CLOSING, PAUSED, FINISHED, CANCELLED }

/**
 * 세션 진행 상태 기계(설계 §6.2·§6.3). 시각은 모두 elapsedNs로 받고 스스로 시계를 읽지 않는다(시험할 수 있게).
 * 화면 쪽이 50 ms쯤마다 [tick]을 부른다. 단계 경계는 예정 시각으로 넘기므로 tick이 늦어도 동작 길이가 2 s로 유지된다.
 *
 * 흐름: [start] → LEAD 1 s(시작 표지음은 부르는 쪽이 start와 함께 낸다) → 동작(2 s씩, 자유 동작은 정한 길이) →
 * 구간이 바뀌면 REST 2 s → … → 마지막 동작 뒤 TAIL 1 s → EndMarkerDue → CLOSING 0.8 s → Finished.
 *
 * - 일시정지([pause]): 안내만 멈춘다(녹음은 부르는 쪽이 계속한다). 동작 도중이면 그 동작을 지금까지로 끊어 "pause"로 적고,
 *   다시 시작([resume])하면 일시정지 구간을 "pause"로 적은 뒤 끊긴 동작을 처음부터 다시 보인다.
 *   LEAD나 REST 도중이면 다음 동작부터, TAIL 도중이면 TAIL부터 다시 한다.
 * - 틀림([markWrong]): 지금 동작에 "wrong" 표시만 남기고 계속한다(학습은 카메라 관절만 쓴다).
 * - 취소([cancel]): 바로 CANCELLED가 된다. 동작 도중이면 그 동작을 지금까지로 적는다.
 * [pause]·[cancel]·[markWrong]는 부르기 직전에 [tick]으로 시간을 반영해 둔 상태라고 가정한다(조정자가 50 ms마다 tick을 부르므로
 * 경계와의 어긋남은 50 ms 안이다).
 */
class SessionRunner(val plan: SessionPlan) {
    var phase: RunnerPhase = RunnerPhase.IDLE
        private set

    /** 마지막으로 보인(끝났거나 일시정지로 끊긴) 동작의 plan.prompts 번호. 아직 없으면 -1. PAUSED에서 다시 보일 동작은 [resumePrompt]다. */
    var currentPrompt: Int = -1
        private set

    /** PAUSED일 때 [resume]하면 다시 보일 동작의 plan.prompts 번호. 끝(TAIL)부터 다시 하면 -1. PAUSED가 아니면 의미 없다. */
    val resumePrompt: Int
        get() = when (pausedFrom) {
            RunnerPhase.PROMPT -> currentPrompt
            RunnerPhase.TAIL -> -1
            else -> nextPrompt
        }

    /** 시작부터 [nowNs]까지 일시정지를 뺀 시간(ns). 시작하지 않았으면 0. 끝난 뒤에는 끝난 시각까지. */
    fun activeNs(nowNs: Long): Long {
        if (!started) return 0
        val until = if (phase == RunnerPhase.FINISHED || phase == RunnerPhase.CANCELLED) endNs else nowNs
        val paused = pausedTotalNs + if (phase == RunnerPhase.PAUSED) nowNs - pauseStartNs else 0L
        return (until - startNs - paused).coerceAtLeast(0)
    }

    /** 지금 단계가 시작된 시각과 끝나는 시각(ns). PAUSED와 끝난 뒤에는 의미 없다. */
    var phaseStartNs: Long = 0
        private set
    var phaseEndsNs: Long = 0
        private set

    var startNs: Long = 0
        private set

    /** FINISHED·CANCELLED가 된 시각. */
    var endNs: Long = 0
        private set

    private val out = mutableListOf<PromptRecord>()
    val records: List<PromptRecord> get() = out

    private var nextPrompt = 0
    private var wrong = false
    private var started = false
    private var pausedFrom = RunnerPhase.IDLE
    private var pauseStartNs = 0L
    private var pausedTotalNs = 0L

    fun start(nowNs: Long) {
        check(phase == RunnerPhase.IDLE) { "이미 시작함" }
        started = true
        startNs = nowNs
        nextPrompt = 0
        enter(RunnerPhase.LEAD, nowNs, PromptPlanner.LEAD_MS)
    }

    /** 시간이 [nowNs]까지 흐른 것을 반영하고, 그동안 생긴 사건을 순서대로 돌려준다. */
    fun tick(nowNs: Long): List<RunnerEvent> {
        val events = mutableListOf<RunnerEvent>()
        while (phase in TIMED && nowNs >= phaseEndsNs) advance(phaseEndsNs, events)
        return events
    }

    fun pause(nowNs: Long) {
        if (phase !in PAUSABLE) return
        if (phase == RunnerPhase.PROMPT) recordPrompt(currentPrompt, phaseStartNs, nowNs, FLAG_PAUSE)
        pausedFrom = phase
        pauseStartNs = nowNs
        phase = RunnerPhase.PAUSED
    }

    fun resume(nowNs: Long): List<RunnerEvent> {
        if (phase != RunnerPhase.PAUSED) return emptyList()
        recordPause(nowNs)
        pausedTotalNs += nowNs - pauseStartNs
        val events = mutableListOf<RunnerEvent>()
        when (pausedFrom) {
            RunnerPhase.PROMPT -> beginPrompt(currentPrompt, nowNs, events)
            RunnerPhase.TAIL -> enter(RunnerPhase.TAIL, nowNs, PromptPlanner.TAIL_MS)
            else -> beginPrompt(nextPrompt, nowNs, events) // LEAD·REST
        }
        return events
    }

    fun markWrong() {
        if (phase == RunnerPhase.PROMPT) wrong = true
    }

    fun cancel(nowNs: Long) {
        if (phase == RunnerPhase.FINISHED || phase == RunnerPhase.CANCELLED) return
        when (phase) {
            RunnerPhase.PROMPT -> recordPrompt(currentPrompt, phaseStartNs, nowNs, if (wrong) FLAG_WRONG else "")
            RunnerPhase.PAUSED -> {
                recordPause(nowNs)
                pausedTotalNs += nowNs - pauseStartNs
            }
            else -> Unit
        }
        endNs = nowNs
        phase = RunnerPhase.CANCELLED
    }

    /** 일시정지 시간을 뺀 진행률 0..1(세션 진행 막대). */
    fun progress(nowNs: Long): Float {
        if (phase == RunnerPhase.FINISHED) return 1f
        return (activeNs(nowNs).toDouble() / (plan.plannedMs * 1_000_000.0)).coerceIn(0.0, 1.0).toFloat()
    }

    /** 지금 동작의 진행률 0..1(남은 시간 막대). 동작 중이 아니면 0. */
    fun promptProgress(nowNs: Long): Float {
        if (phase != RunnerPhase.PROMPT) return 0f
        val length = (phaseEndsNs - phaseStartNs).coerceAtLeast(1)
        return ((nowNs - phaseStartNs).toDouble() / length).coerceIn(0.0, 1.0).toFloat()
    }

    private fun advance(t: Long, events: MutableList<RunnerEvent>) {
        when (phase) {
            RunnerPhase.LEAD -> beginPrompt(0, t, events)
            RunnerPhase.PROMPT -> {
                recordPrompt(currentPrompt, phaseStartNs, t, if (wrong) FLAG_WRONG else "")
                val next = currentPrompt + 1
                when {
                    next >= plan.prompts.size -> enter(RunnerPhase.TAIL, t, PromptPlanner.TAIL_MS)
                    plan.prompts[next].segment != plan.prompts[currentPrompt].segment -> {
                        nextPrompt = next
                        enter(RunnerPhase.REST, t, PromptPlanner.REST_MS)
                        events += RunnerEvent.RestStarted
                    }
                    else -> beginPrompt(next, t, events)
                }
            }
            RunnerPhase.REST -> beginPrompt(nextPrompt, t, events)
            RunnerPhase.TAIL -> {
                enter(RunnerPhase.CLOSING, t, CLOSING_MS)
                events += RunnerEvent.EndMarkerDue
            }
            RunnerPhase.CLOSING -> {
                endNs = t
                phase = RunnerPhase.FINISHED
                events += RunnerEvent.Finished
            }
            else -> error("시간으로 넘어가는 단계가 아님: $phase")
        }
    }

    private fun beginPrompt(i: Int, t: Long, events: MutableList<RunnerEvent>) {
        currentPrompt = i
        wrong = false
        enter(RunnerPhase.PROMPT, t, plan.prompts[i].durationMs)
        events += RunnerEvent.PromptStarted(i)
    }

    private fun enter(p: RunnerPhase, t: Long, lengthMs: Long) {
        phase = p
        phaseStartNs = t
        phaseEndsNs = t + lengthMs * 1_000_000
    }

    private fun recordPrompt(i: Int, from: Long, to: Long, flag: String) {
        val p = plan.prompts[i]
        out += PromptRecord(out.size, p.segment, p.classId, Labels.byId(p.classId).name, from, to, flag)
    }

    private fun recordPause(to: Long) {
        out += PromptRecord(out.size, 0, Labels.PAUSE_ID, Labels.PAUSE.name, pauseStartNs, to, FLAG_PAUSE)
    }

    companion object {
        /** 끝 표지음을 낸 뒤 녹음을 멈추기까지 기다리는 시간. 폰 출력 지연과 워치까지 가는 시간을 넉넉히 덮는다. */
        const val CLOSING_MS = 800L
        const val FLAG_WRONG = "wrong"
        const val FLAG_PAUSE = "pause"
        private val TIMED = setOf(RunnerPhase.LEAD, RunnerPhase.PROMPT, RunnerPhase.REST, RunnerPhase.TAIL, RunnerPhase.CLOSING)
        private val PAUSABLE = setOf(RunnerPhase.LEAD, RunnerPhase.PROMPT, RunnerPhase.REST, RunnerPhase.TAIL)
    }
}
