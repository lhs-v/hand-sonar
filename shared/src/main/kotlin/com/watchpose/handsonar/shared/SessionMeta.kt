package com.watchpose.handsonar.shared

/**
 * meta.json의 camera. [display]는 "inner" | "cover", [timeBase]는 "realtime" | "arrival"(프레임 도착 시각으로 대신함, 근사).
 * [fps]는 카메라에 건 목표값(30)이고, [measuredFps]는 세션 중 실제 평균((landmarks.csv 줄 수 − 1) ÷ 첫·마지막 frameNs 간격,
 * 시작 때 쓴 meta와 프레임 2개 미만이면 null)이다. [delegate]는 손 추적이 쓰던 계산 장치("GPU" | "CPU", 모르면 null)로,
 * 끝에서 다시 쓴 meta에는 세션이 끝날 때의 값이 들어간다.
 */
data class CameraMeta(
    val id: String,
    val display: String,
    val width: Int,
    val height: Int,
    val fps: Int,
    val timeBase: String,
    val mirrored: Boolean,
    val rotation: Int,
    val delegate: String? = null,
    val measuredFps: Double? = null,
    /** 화면 회전(Display.getRotation, 0–3). 그림의 위가 기기의 어느 쪽인지 정한다(PC가 폰 중력을 그림 축으로 옮길 때). 이 칸 전 세션은 null(0으로 본다). */
    val displayRotation: Int? = null,
)

/** 기준 자세 한 차례의 결과: 유지 구간 프레임들의 손(그림 평면)·팔뚝 각 중앙값(도, 모르면 null)과 유지 시간. */
data class CalibrationStep(val handDeg: Double?, val forearmDeg: Double?, val holdMs: Long)

/**
 * meta.json `calibration`: 기준 자세 단계의 결과(설계: 손목 상태 학습). [result]는 "ok"(두 차례 모두 통과) | "skip"(건너뜀)이다.
 * [checks]는 "hand+forearm"(워치 기울기도 봄) | "hand"(워치 기울기를 못 받아 손만 봄). [side]·[front]는 옆면·정면 차례(통과하지 않았으면 null).
 * 유지 구간 시각은 prompts.csv의 classId −4(옆면)·−3(정면) 줄에 있다.
 */
data class CalibrationMeta(val result: String, val checks: String, val side: CalibrationStep?, val front: CalibrationStep?)

/** 표지음 하나의 기록. [micFrame]은 phone_audio.wav 안에서 찾은 시작 샘플, [elapsedNs]는 그 샘플의 폰 시각. 못 찾으면 found=false. */
data class MarkerMeta(val found: Boolean, val playNs: Long, val micFrame: Double?, val elapsedNs: Long?, val ratio: Double?)

data class VideoMeta(val file: String, val width: Int, val height: Int, val frames: Long, val ok: Boolean)

data class SessionStats(
    val frames: Long,
    val handRatio: Double,
    val watchHeardRatio: Double,
    /** 1 s 넘게 워치 처프가 "안 들림"이었던 구간 [시작 ns, 끝 ns]. */
    val watchGaps: List<LongArray>,
    val startMarker: MarkerMeta?,
    val endMarker: MarkerMeta?,
    val durationSec: Double,
    /** 세션 동안 워치 기울기(녹음 중인 워치만 보냄, [TiltMessage])를 1 s 안에 받고 있던 시간 비율. 이 칸 전 세션은 null. */
    val watchLinkRatio: Double? = null,
)

/**
 * meta.json(설계 §7.2). 세션 시작 때 end·stats 없이 한 번 쓰고, 끝날 때 모두 채워 다시 쓴다.
 * [hand]는 추적한 손(워치를 찬 손)으로 "left" | "right"다.
 * [rewear]는 이 세션 전에 워치를 벗었다 다시 찼다는 표시다(준비 화면 "워치 다시 참", 계획 6 — 다시 찬 뒤 처음 보관된 세션에 참).
 * [context]는 준비 화면에서 고른 세션 정보, [sensors]는 폰 센서 기록(`phone_sensors-NNN.csv`)의 보고로 끝에서 다시 쓴 meta에만 있다.
 * 두 칸은 `rewear` 바로 뒤에 적는다. [calibration]은 기준 자세 단계의 결과로 `sensors` 뒤에 적는다(그 단계 전에 끝났거나 이 기능 전 세션은 null).
 */
data class SessionMeta(
    val app: String,
    val device: String,
    val android: String,
    val camera: CameraMeta?,
    val hand: String,
    val startWallMs: Long,
    val startNs: Long,
    val endWallMs: Long?,
    val endNs: Long?,
    val plan: SessionPlan,
    val status: String,
    val flags: List<String>,
    val memo: String,
    val stats: SessionStats?,
    val rewear: Boolean,
    val video: VideoMeta?,
    val audioFile: String,
    val audioSource: String?,
    val cancelled: Boolean,
    val context: SessionContext = SessionContext(),
    val sensors: SensorReport? = null,
    val calibration: CalibrationMeta? = null,
) {
    fun toJson(): String = JsonOut.write(
        linkedMapOf<String, Any?>(
            "format" to 1,
            "app" to app,
            "device" to device,
            "android" to android,
            "camera" to camera?.let {
                linkedMapOf<String, Any?>(
                    "id" to it.id, "display" to it.display, "width" to it.width, "height" to it.height,
                    "fps" to it.fps, "timeBase" to it.timeBase, "mirrored" to it.mirrored, "rotation" to it.rotation,
                    "delegate" to it.delegate, "measuredFps" to it.measuredFps, "displayRotation" to it.displayRotation,
                )
            },
            "hand" to hand,
            "start" to linkedMapOf<String, Any?>("wallMs" to startWallMs, "elapsedNs" to startNs),
            "end" to if (endWallMs != null && endNs != null) linkedMapOf<String, Any?>("wallMs" to endWallMs, "elapsedNs" to endNs) else null,
            "prompts" to linkedMapOf<String, Any?>(
                "version" to plan.version,
                "seed" to plan.seed,
                "freeMotionSec" to plan.freeMotionSec,
                "promptMs" to PromptPlanner.PROMPT_MS,
                "leadMs" to PromptPlanner.LEAD_MS,
                "restMs" to PromptPlanner.REST_MS,
                "tailMs" to PromptPlanner.TAIL_MS,
                "segments" to plan.segments.map { seg -> seg.map { it.classId } },
            ),
            "status" to status,
            "flags" to flags,
            "memo" to memo,
            "stats" to stats?.let { s ->
                linkedMapOf<String, Any?>(
                    "frames" to s.frames,
                    "handRatio" to s.handRatio,
                    "watchHeardRatio" to s.watchHeardRatio,
                    "watchGaps" to s.watchGaps.map { it.toList() },
                    "markers" to linkedMapOf<String, Any?>(
                        "start" to s.startMarker?.let { markerMap(it) },
                        "end" to s.endMarker?.let { markerMap(it) },
                    ),
                    "durationSec" to s.durationSec,
                    "watchLinkRatio" to s.watchLinkRatio,
                )
            },
            "rewear" to rewear,
            "context" to context.toMap(),
            "sensors" to sensors?.toMap(),
            "calibration" to calibration?.let {
                linkedMapOf<String, Any?>("result" to it.result, "checks" to it.checks, "side" to it.side?.let(::stepMap), "front" to it.front?.let(::stepMap))
            },
            "video" to video?.let {
                linkedMapOf<String, Any?>("file" to it.file, "width" to it.width, "height" to it.height, "frames" to it.frames, "ok" to it.ok)
            },
            "audio" to linkedMapOf<String, Any?>("file" to audioFile, "source" to audioSource),
            "cancelled" to cancelled,
        ),
    )

    private fun stepMap(s: CalibrationStep) = linkedMapOf<String, Any?>("handDeg" to s.handDeg, "forearmDeg" to s.forearmDeg, "holdMs" to s.holdMs)

    private fun markerMap(m: MarkerMeta) = linkedMapOf<String, Any?>(
        "found" to m.found, "playNs" to m.playNs, "micFrame" to m.micFrame, "elapsedNs" to m.elapsedNs, "ratio" to m.ratio,
    )

    companion object {
        const val STATUS_KEPT = "kept"
        const val STATUS_TRASH = "trash"
    }
}

/** phone_audio.json(설계 §7.2): 녹음 소스와 recordClock. 위치와 읽은 수는 phone_audio.wav의 첫 샘플을 0으로 센다. */
object PhoneAudioJson {
    fun write(source: String, clockSource: String, clock: List<ClockLine>, frames: Long): String = JsonOut.write(
        linkedMapOf<String, Any?>(
            "format" to 1,
            "file" to "phone_audio.wav",
            "rate" to 48_000,
            "bits" to 16,
            "channels" to 1,
            "source" to source,
            "frames" to frames,
            "recordClockSource" to clockSource,
            "recordClock" to clock.map { listOf(it.position, it.elapsedNs, it.framesRead) },
        ),
    )
}
