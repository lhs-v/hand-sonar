package com.watchpose.handsonar.shared

/**
 * 남은 저장 공간으로 녹음 가능 시간을 재고 경고·자동 정지를 정한다(설계 §5.4). 녹음 1초는 소리 [SonarSpec.BYTES_PER_SECOND]바이트와
 * 센서 기록 약 [SensorPlan.WATCH_BYTES_PER_SECOND]바이트다(계획 6 — 넣지 않으면 남은 시간이 약 25% 넉넉하게 나온다).
 */
object StorageBudget {
    const val WARN_MINUTES = 10.0
    const val STOP_MINUTES = 1.0

    /** 녹음 1초에 쓰는 바이트(소리 + 센서). */
    const val BYTES_PER_SECOND = SonarSpec.BYTES_PER_SECOND + SensorPlan.WATCH_BYTES_PER_SECOND

    enum class Level { OK, WARN, STOP }

    fun minutesLeft(freeBytes: Long): Double = freeBytes.toDouble() / BYTES_PER_SECOND / 60.0

    fun level(freeBytes: Long): Level {
        val m = minutesLeft(freeBytes)
        return when {
            m < STOP_MINUTES -> Level.STOP
            m < WARN_MINUTES -> Level.WARN
            else -> Level.OK
        }
    }
}
