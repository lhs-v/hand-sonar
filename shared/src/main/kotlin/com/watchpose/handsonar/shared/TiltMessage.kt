package com.watchpose.handsonar.shared

import java.nio.ByteBuffer

/** 워치 기울기 하나: 워치 중력 센서 값(m/s², 위쪽을 가리킴)과 그 이벤트의 워치 elapsedNs. */
data class Tilt(val elapsedNs: Long, val x: Float, val y: Float, val z: Float) {
    fun values(): FloatArray = floatArrayOf(x, y, z)
}

/**
 * 워치 → 폰 기울기 메시지(MessageClient.sendMessage). 워치는 녹음하는 동안 [SEND_MS]마다 최근 중력 값을 보내고, 폰은 기준 자세 판정에서
 * 팔뚝이 섰는지 본다(받은 지 [FRESH_MS]가 넘은 값은 쓰지 않는다). 몸통은 elapsedNs(Long) + x·y·z(Float) 20바이트, 큰 끝 순서다.
 * 기록에는 쓰지 않는다(워치 센서 CSV가 따로 있다).
 */
object TiltMessage {
    const val PATH = "/hand-sonar/tilt"
    const val SEND_MS = 200L
    const val FRESH_MS = 1_000L
    private const val SIZE = 20

    fun encode(t: Tilt): ByteArray = ByteBuffer.allocate(SIZE).putLong(t.elapsedNs).putFloat(t.x).putFloat(t.y).putFloat(t.z).array()

    /** 길이가 다르거나 값이 유한하지 않으면 null. */
    fun decode(bytes: ByteArray): Tilt? {
        if (bytes.size != SIZE) return null
        val b = ByteBuffer.wrap(bytes)
        val t = Tilt(b.long, b.float, b.float, b.float)
        return if (t.x.isFinite() && t.y.isFinite() && t.z.isFinite()) t else null
    }
}
