package com.watchpose.handsonar.shared

/** 남은 저장 공간으로 녹음 가능 시간을 재고 경고·자동 정지를 정한다(설계 §5.4). 녹음은 1초에 [SonarSpec.BYTES_PER_SECOND]바이트다. */
object StorageBudget {
    const val WARN_MINUTES = 10.0
    const val STOP_MINUTES = 1.0

    enum class Level { OK, WARN, STOP }

    fun minutesLeft(freeBytes: Long): Double = freeBytes.toDouble() / SonarSpec.BYTES_PER_SECOND / 60.0

    fun level(freeBytes: Long): Level {
        val m = minutesLeft(freeBytes)
        return when {
            m < STOP_MINUTES -> Level.STOP
            m < WARN_MINUTES -> Level.WARN
            else -> Level.OK
        }
    }
}
