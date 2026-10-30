package com.watchpose.handsonar.shared

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * 기준 자세 판정 한 프레임의 재료. 각도는 도이고 null은 모른다는 뜻이다.
 * [handDeg]: 그림 평면에서 손(손목 → 중지 MCP)과 그림 속 수직선 사이 각([CalibrationGeometry.handInPlaneDeg]). 손이 없거나 폰 중력을 모르면 null.
 * [forearmDeg]: 팔뚝(워치 x축)과 수직 사이 각([CalibrationGeometry.forearmDeg]). 워치 기울기를 못 받았으면 null.
 */
data class CalibrationInput(val nowNs: Long, val handDeg: Double?, val fingersStraight: Boolean, val forearmDeg: Double?)

/**
 * 판정 상태(화면과 meta.json `calibration`). [forearmOk]는 워치 기울기를 모르면 null이다(그때는 손만 보고 넘긴다).
 * [holdStartNs]는 조건이 이어진 첫 시각(아니면 null), [heldMs]는 그때부터 지금까지, [done]이면 [holdMs]만큼 유지됐다.
 * [handMedianDeg]·[forearmMedianDeg]는 유지 구간 프레임들의 중앙값(그 값이 하나도 없으면 null).
 */
data class CalibrationState(
    val handOk: Boolean,
    val fingersOk: Boolean,
    val forearmOk: Boolean?,
    val holdStartNs: Long?,
    val heldMs: Long,
    val done: Boolean,
    val handMedianDeg: Double? = null,
    val forearmMedianDeg: Double? = null,
)

/**
 * 기준 자세(팔뚝과 손을 함께 세운 평행 자세)를 판정한다. 팔뚝은 워치 중력으로, 손은 카메라 그림 평면에서 본다.
 * MediaPipe의 3D 좌표는 그림 투영에 맞춘 것이라 깊이(카메라 쪽 기울기) 성분은 믿기 어렵다(실데이터: 손목 굽히기에서 깊이 기울기의 부호가
 * 세션마다 갈림). 그래서 손은 그림 평면 안의 각도와 손가락 펴짐만 보고 깊이는 쓰지 않는다 — 팔뚝을 세웠으면(워치) 손이 그림 평면 안에서
 * 서 있는지가 평행의 한 축(옆 꺾임)을 확인하고, 다른 축(손등·손바닥 쪽 꺾임)은 깊이라 안내 글에 맡기고 PC가 나중에 따로 잰다.
 *
 * 조건(손 [handMaxDeg]° 안 · 손가락 폄 · 팔뚝 [forearmMaxDeg]° 안(아는 경우))이 [holdMs] 동안 이어지면 [CalibrationState.done].
 * 한 프레임씩 조건이 깨지는 것(MediaPipe 흔들림)은 [gapMs]까지 봐준다. [update]는 프레임 시각이 늘어나는 차례로 부른다.
 */
class CalibrationGate(
    val holdMs: Long = HOLD_MS,
    val handMaxDeg: Double = MAX_DEG,
    val forearmMaxDeg: Double = MAX_DEG,
    val gapMs: Long = GAP_MS,
) {
    private var holdStart: Long? = null
    private var lastOkNs = 0L
    private val hand = ArrayList<Double>()
    private val forearm = ArrayList<Double>()

    fun update(input: CalibrationInput): CalibrationState {
        val handOk = input.handDeg != null && input.handDeg <= handMaxDeg
        val forearmOk = input.forearmDeg?.let { it <= forearmMaxDeg }
        val ok = handOk && input.fingersStraight && forearmOk != false
        if (ok) {
            if (holdStart == null) {
                holdStart = input.nowNs
                hand.clear()
                forearm.clear()
            }
            lastOkNs = input.nowNs
            input.handDeg?.let(hand::add)
            input.forearmDeg?.let(forearm::add)
        } else if (holdStart != null && input.nowNs - lastOkNs > gapMs * 1_000_000) {
            holdStart = null
        }
        val start = holdStart
        val held = if (start == null) 0L else (input.nowNs - start) / 1_000_000
        return CalibrationState(
            handOk = handOk,
            fingersOk = input.fingersStraight,
            forearmOk = forearmOk,
            holdStartNs = start,
            heldMs = held,
            done = ok && start != null && held >= holdMs,
            handMedianDeg = if (start == null) null else median(hand),
            forearmMedianDeg = if (start == null) null else median(forearm),
        )
    }

    /** 다시 처음부터(기준 자세 단계를 새로 시작할 때). */
    fun reset() {
        holdStart = null
        hand.clear()
        forearm.clear()
    }

    private fun median(v: List<Double>): Double? {
        if (v.isEmpty()) return null
        val s = v.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }

    companion object {
        const val HOLD_MS = 2_000L
        const val MAX_DEG = 10.0
        const val GAP_MS = 300L
    }
}

/**
 * 기준 자세 판정에 쓰는 기하(폰·워치 센서 값과 MediaPipe 그림 좌표). 좌표 약속:
 * - 폰 기기 축(안드로이드): x 화면 오른쪽, y 화면 위, z 화면 밖(정면 카메라가 보는 쪽). 가속도는 가만히 있으면 위쪽을 가리킨다(+9.8).
 * - 그림: 똑바로 세우고 좌우를 뒤집은 거울 그림(폰 앱 `CameraPipeline`). x는 화면 오른쪽, y는 화면 아래(0..1, 너비·높이로 나눔).
 *   그림의 위는 화면의 위다(사용자가 미리보기를 똑바로 맞춘다 — 회전 보정 포함). 화면의 위·오른쪽은 기기 회전(Display.getRotation)으로 정한다.
 * - 워치 기기 축: 왼손목에 버튼이 손 쪽이면 x가 손 쪽(팔뚝 방향)이다(2026-10-28 실데이터: 손을 든 동안 중력 x 성분 +0.7).
 */
object CalibrationGeometry {
    /** 엄지를 뺀 네 손가락의 (MCP, 끝) 점 번호. */
    private val FINGERS = listOf(5 to 8, 9 to 12, 13 to 16, 17 to 20)

    /** 손가락 펴짐 문턱: 그림에서 |손목→끝| ÷ |손목→MCP|. 실데이터 편 손 1.6–2.0, 굽힌 손가락 0.7–1.1(2026-10-28 세션 9개). */
    const val STRAIGHT_RATIO = 1.5

    /**
     * 폰 가속도([accel], 기기 축)에서 세계 위쪽이 그림(거울, x 오른쪽 · y 아래)에서 가리키는 2D 단위 방향. [displayRotation]은 0–3
     * (Surface.ROTATION_0…270). 가속도가 0이거나 위쪽이 그림 평면에 거의 수직이면(폰을 눕힘) null. 원근은 무시한다(손이 그림 가운데 근처).
     */
    fun projectedUp(accel: FloatArray, displayRotation: Int): DoubleArray? {
        if (accel.size < 3) return null
        val (right, up) = screenAxes(displayRotation)
        val ax = accel[0].toDouble()
        val ay = accel[1].toDouble()
        val az = accel[2].toDouble()
        val norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm < 1e-3) return null
        val x = (ax * right[0] + ay * right[1]) / norm // 위쪽의 화면 오른쪽 성분
        val y = -(ax * up[0] + ay * up[1]) / norm // 그림 y는 아래가 +
        val len = hypot(x, y)
        if (len < 0.3) return null
        return doubleArrayOf(x / len, y / len)
    }

    /** 기기 회전마다 화면 오른쪽·위가 기기 x·y 축에서 어느 쪽인지((rx, ry), (ux, uy)). ROTATION_90은 기기를 시계 반대로 90° 돌린 것이다. */
    private fun screenAxes(displayRotation: Int): Pair<IntArray, IntArray> = when (((displayRotation % 4) + 4) % 4) {
        1 -> intArrayOf(0, -1) to intArrayOf(1, 0)
        2 -> intArrayOf(-1, 0) to intArrayOf(0, -1)
        3 -> intArrayOf(0, 1) to intArrayOf(-1, 0)
        else -> intArrayOf(1, 0) to intArrayOf(0, 1)
    }

    /**
     * 그림 평면에서 손(손목 0 → 중지 MCP 9)과 [up] 사이 각(도, 0–180). [image]는 21점 x,y,z(63개, 0..1), [width]·[height]는 그림 픽셀 크기
     * (x·y의 단위를 맞추려고). 손이 너무 작으면(두 점이 같음) null.
     */
    fun handInPlaneDeg(image: FloatArray, width: Int, height: Int, up: DoubleArray): Double? {
        val dx = (image[27] - image[0]).toDouble() * width
        val dy = (image[28] - image[1]).toDouble() * height
        val len = hypot(dx, dy)
        if (len < 1e-6) return null
        val cos = ((dx * up[0] + dy * up[1]) / len).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    /** 네 손가락이 모두 [STRAIGHT_RATIO] 이상으로 펴졌는가(그림 평면 길이의 비라 손 전체가 기울어도 비는 그대로다). */
    fun fingersStraight(image: FloatArray, width: Int, height: Int): Boolean = FINGERS.all { (mcp, tip) ->
        val m = dist(image, 0, mcp, width, height)
        m > 1e-6 && dist(image, 0, tip, width, height) / m >= STRAIGHT_RATIO
    }

    /** 팔뚝(워치 x축)과 수직 사이 각(도, 0–90). [gravity]는 워치 중력 센서 값(위쪽을 가리킴). 버튼을 반대로 차도(x가 팔꿈치 쪽) 같은 값이다. */
    fun forearmDeg(gravity: FloatArray): Double? {
        if (gravity.size < 3) return null
        val x = gravity[0].toDouble()
        val n = sqrt(x * x + gravity[1].toDouble() * gravity[1] + gravity[2].toDouble() * gravity[2])
        if (n < 1e-3) return null
        return Math.toDegrees(acos((abs(x) / n).coerceIn(0.0, 1.0)))
    }

    private fun dist(image: FloatArray, a: Int, b: Int, width: Int, height: Int): Double =
        hypot((image[3 * b] - image[3 * a]).toDouble() * width, (image[3 * b + 1] - image[3 * a + 1]).toDouble() * height)
}
