package com.watchpose.handsonar.shared

import java.util.Collections
import java.util.Random

/** 계획된 동작 하나. [segment]는 1–7(설계 §6.2 표의 구간). */
data class PlannedPrompt(val segment: Int, val classId: Int, val durationMs: Long)

/** 한 세션의 동작 순서. [segments]는 구간 순서대로, 구간 안의 동작은 씨앗값으로 섞은 순서다. 자유 동작이 0 s면 구간 7이 없다. */
data class SessionPlan(val version: String, val seed: Long, val freeMotionSec: Int, val segments: List<List<PlannedPrompt>>) {
    val prompts: List<PlannedPrompt> = segments.flatten()

    /** 끝까지 갔을 때의 길이(ms): 시작 1 s + 모든 동작 + 구간 사이 2 s씩 + 끝 1 s. 일시정지와 마무리(표지음 녹음 대기)는 뺀다. */
    val plannedMs: Long = PromptPlanner.LEAD_MS + prompts.sumOf { it.durationMs } +
        PromptPlanner.REST_MS * (segments.size - 1).coerceAtLeast(0) + PromptPlanner.TAIL_MS
}

/** 설계 §6.2의 세션 동작 순서를 만든다. */
object PromptPlanner {
    const val PROMPT_MS = 2_000L
    const val LEAD_MS = 1_000L
    const val REST_MS = 2_000L
    const val TAIL_MS = 1_000L
    const val DEFAULT_FREE_SEC = 60
    const val MAX_FREE_SEC = 180

    /**
     * 구간 1·2: 손가락 0–4를 두 번씩(10개), 3·4: ASL 숫자 0–9(번호 5–14)를 두 번씩(20개), 5: 손목 15–17로 8개(15·16·17·15·16·17·15·16),
     * 6: 손목 4개(15·16·17·15), 7: 자유 동작 하나([freeMotionSec] 초, 0이면 없음). 구간마다 같은 java.util.Random(seed)로 차례로 섞는다.
     */
    fun plan(seed: Long, freeMotionSec: Int = DEFAULT_FREE_SEC): SessionPlan {
        require(freeMotionSec in 0..MAX_FREE_SEC) { "자유 동작 길이는 0–$MAX_FREE_SEC s" }
        val random = Random(seed)
        fun segment(index: Int, ids: List<Int>): List<PlannedPrompt> =
            ids.toMutableList().also { Collections.shuffle(it, random) }.map { PlannedPrompt(index, it, PROMPT_MS) }
        val fingers = Labels.FINGERS.toList()
        val asl = Labels.ASL.toList()
        val segments = mutableListOf(
            segment(1, fingers + fingers),
            segment(2, fingers + fingers),
            segment(3, asl + asl),
            segment(4, asl + asl),
            segment(5, listOf(15, 16, 17, 15, 16, 17, 15, 16)),
            segment(6, listOf(15, 16, 17, 15)),
        )
        if (freeMotionSec > 0) segments.add(listOf(PlannedPrompt(7, Labels.FREE_ID, freeMotionSec * 1_000L)))
        return SessionPlan(Labels.VERSION, seed, freeMotionSec, segments)
    }
}
