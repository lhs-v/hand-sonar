package com.watchpose.handsonar.shared

/** recordClock 한 줄(설계 §7.1·§7.2): [녹음 위치 프레임, elapsedNs, 그때까지 읽은 프레임 수]. */
data class ClockLine(val position: Long, val elapsedNs: Long, val framesRead: Long)

/**
 * 1초마다 녹음 시각을 무엇으로 적을지 정한다(워치 SonarRecorder와 같은 규칙에 "같은 값 거르기"를 더함).
 * 1. 타임스탬프가 되면 (위치, 시각, 읽은 수)를 적는다. 단 직전에 적은 타임스탬프와 위치·시각이 모두 같으면 이번 초를 건너뛴다
 *    (기기가 지난 값을 다시 준 것, 2026-10-27 워치8 녹음에서 봄).
 * 2. 안 되면 이번 초를 건너뛴다. 다만 한 번도 성공하지 못한 채 [SWITCH_AFTER_FAILURES]번 연속 실패하면 그때부터 끝까지
 *    "읽은 직후 시각"(읽은 수, 지금, 읽은 수)으로 적고 [source]를 "read"로 바꾼다. 한 번 바꾸면 되돌리지 않는다.
 *    그래서 한 파일에는 한 출처만 남는다.
 */
class RecordClockRule {
    var source: String = SOURCE_TIMESTAMP
        private set

    private var worked = false
    private var failures = 0
    private var lastPosition = Long.MIN_VALUE
    private var lastNs = Long.MIN_VALUE

    fun onSecond(timestampOk: Boolean, position: Long, nanoTime: Long, framesRead: Long, nowNs: Long): ClockLine? {
        if (source == SOURCE_READ) return ClockLine(framesRead, nowNs, framesRead)
        if (timestampOk) {
            if (position == lastPosition && nanoTime == lastNs) return null
            worked = true
            lastPosition = position
            lastNs = nanoTime
            return ClockLine(position, nanoTime, framesRead)
        }
        if (worked) return null
        failures++
        if (failures < SWITCH_AFTER_FAILURES) return null
        source = SOURCE_READ
        return ClockLine(framesRead, nowNs, framesRead)
    }

    companion object {
        const val SOURCE_TIMESTAMP = "timestamp"
        const val SOURCE_READ = "read"
        const val SWITCH_AFTER_FAILURES = 3
    }
}
