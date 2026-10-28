package com.watchpose.handsonar.shared

/** 기기 시각 한 쌍(설계 §7): [wallMs] = System.currentTimeMillis, [elapsedNs] = SystemClock.elapsedRealtimeNanos. */
data class ClockPoint(val wallMs: Long, val elapsedNs: Long)

/** 녹음이 끝난 이유(rec.json의 stop.reason). */
enum class StopReason(val id: String) {
    USER("user"),
    /** 정한 시간이 되어 스스로 멈춤(자동 정지, 설계 §5.1). 오류가 아니다. */
    TIMER("timer"),
    STORAGE("storage"),
    ERROR("error"),
}

/**
 * 녹음 기록 rec.json(설계 §7.1). 녹음 중에 채우고 [toJson]으로 쓴다(서비스가 10초마다, 끝날 때 한 번 더).
 * 녹음 스레드(녹음 시각)와 서비스 틱(재생 시각·끊김·조각·볼륨·센서)이 함께 덧붙이므로 모든 공개 멤버는 [lock]으로 잠근다.
 * recordClock 한 줄은 [녹음 위치, elapsedNs, 그때까지 읽은 프레임 수]다. 녹음이 넘쳐 샘플이 빠지면 첫째와 셋째의 차이가 튄다.
 *
 * `volume`(계획 6): [volumeIndex]는 녹음 동안의 실제 미디어 볼륨(녹음 볼륨으로 맞춘 뒤 다시 읽은 값), [volumeChoice]는 고른 단계
 * (null = 그대로), [volumeOriginal]은 녹음 전 볼륨이다. 멈춘 뒤 되돌린 값과 되돌리지 않은 까닭([setVolumeRestore]), 녹음 중 볼륨이 바뀐 때
 * ([addVolumeChange])를 함께 적는다. 볼륨 값은 그때 미디어가 나가는 기기의 것이라(안드로이드는 출력 기기마다 볼륨을 둔다) 외부 출력도 적는다
 * (계획 6 고침 I1): [volumeOutputs]는 시작 때 이어진 외부 출력, [volumeTarget]은 맞추려던 값(그대로이거나 걸지 않았으면 null),
 * [volumeSkipped]는 녹음 볼륨을 걸지 않은 까닭, 녹음 중 외부 출력 목록이 바뀐 때([addOutputChange]), [volumeLeftover]는 이 녹음을 시작하기
 * 전에 앞 녹음이 남긴 볼륨 기록을 다룬 결과다. `sensors`(계획 6)는 맨 끝이고 센서 기록이 없으면 null이다.
 */
class RecordingLog(
    private val app: String,
    private val device: String,
    private val android: String,
    private val start: ClockPoint,
    private val source: String,
    private val window: SweepWindow,
    private val amplitude: Float,
    private val volumeStream: String,
    private val volumeIndex: Int,
    private val volumeMax: Int,
    private val volumeChoice: Int? = null,
    private val volumeOriginal: Int = volumeIndex,
    private val volumeTarget: Int? = null,
    private val volumeOutputs: List<String> = emptyList(),
    private val volumeSkipped: String? = null,
    private val volumeLeftover: VolumeRestore? = null,
) {
    private val lock = Any()
    private val recordClock = mutableListOf<LongArray>()
    private val playClock = mutableListOf<LongArray>()
    private val underruns = mutableListOf<LongArray>()
    private val volumeChanges = mutableListOf<LongArray>()
    private val outputChanges = mutableListOf<List<Any>>()
    private var volumeRestored: Int? = null
    private var volumeRestoreSkipped: String? = null
    private var recordClockSource = "timestamp"
    private var parts: List<PartInfo> = emptyList()
    private var sensors: SensorReport? = null
    private var stop: ClockPoint? = null
    private var stopReason: StopReason? = null

    fun addRecordClock(frame: Long, elapsedNs: Long, framesRead: Long) {
        synchronized(lock) { recordClock += longArrayOf(frame, elapsedNs, framesRead) }
    }

    fun addPlayClock(frame: Long, elapsedNs: Long) {
        synchronized(lock) { playClock += longArrayOf(frame, elapsedNs) }
    }

    fun addUnderrun(elapsedNs: Long, total: Int) {
        synchronized(lock) { underruns += longArrayOf(elapsedNs, total.toLong()) }
    }

    /** 녹음 중 미디어 볼륨이 [index]로 바뀌었다(서비스 틱이 1초마다 읽어 견준다). */
    fun addVolumeChange(elapsedNs: Long, index: Int) {
        synchronized(lock) { volumeChanges += longArrayOf(elapsedNs, index.toLong()) }
    }

    /** 녹음 중 이어진 외부 출력 목록이 [outputs]로 바뀌었다(빈 목록 = 스피커로 돌아옴). */
    fun addOutputChange(elapsedNs: Long, outputs: List<String>) {
        synchronized(lock) { outputChanges += listOf(elapsedNs, outputs.toList()) }
    }

    /**
     * 멈춘 뒤: [restored]는 원래 볼륨으로 되돌리고 다시 읽은 값(되돌리지 않았으면 null), [skipped]는 되돌리지 않은 까닭
     * (외부 출력이 이어져 있어 미룸 — [RecordVolumeRules.SKIP_EXTERNAL]) 또는 null.
     */
    fun setVolumeRestore(restored: Int?, skipped: String?) {
        synchronized(lock) {
            volumeRestored = restored
            volumeRestoreSkipped = skipped
        }
    }

    /** 센서 기록의 지금 보고(계획 6). */
    fun setSensors(report: SensorReport?) {
        synchronized(lock) { sensors = report }
    }

    /** 기기가 녹음 타임스탬프를 주지 않아 읽은 직후 시각으로 대신했음을 적는다. */
    fun useReadTimeClock() {
        synchronized(lock) { recordClockSource = "read" }
    }

    fun setParts(parts: List<PartInfo>) {
        synchronized(lock) { this.parts = parts }
    }

    fun finish(at: ClockPoint, reason: StopReason, parts: List<PartInfo>) {
        synchronized(lock) {
            stop = at
            stopReason = reason
            this.parts = parts
        }
    }

    fun toJson(): String = synchronized(lock) {
        JsonOut.write(
            linkedMapOf<String, Any?>(
                "format" to FORMAT,
                "app" to app,
                "device" to device,
                "android" to android,
                "start" to clock(start),
                "stop" to stop?.let { clock(it) + ("reason" to stopReason?.id) },
                "audio" to linkedMapOf<String, Any?>("rate" to SonarSpec.SAMPLE_RATE, "bits" to 16, "channels" to 1, "source" to source),
                "tx" to linkedMapOf<String, Any?>(
                    "f0Hz" to SonarSpec.START_HZ,
                    "bandwidthHz" to SonarSpec.BANDWIDTH_HZ,
                    "samples" to SonarSpec.SWEEP_SAMPLES,
                    "window" to window.id,
                    "amplitude" to amplitude,
                ),
                "volume" to linkedMapOf<String, Any?>(
                    "stream" to volumeStream,
                    "index" to volumeIndex,
                    "max" to volumeMax,
                    "choice" to volumeChoice,
                    "original" to volumeOriginal,
                    "restored" to volumeRestored,
                    "changes" to volumeChanges.toList(),
                    "target" to volumeTarget,
                    "outputs" to volumeOutputs,
                    "outputChanges" to outputChanges.toList(),
                    "skipped" to volumeSkipped,
                    "restoreSkipped" to volumeRestoreSkipped,
                    "leftover" to volumeLeftover?.pending?.let { p ->
                        linkedMapOf<String, Any?>(
                            "original" to p.original,
                            "applied" to p.applied,
                            "restored" to volumeLeftover.restored,
                            "skipped" to volumeLeftover.skipped,
                        )
                    },
                ),
                "recordClockSource" to recordClockSource,
                "recordClock" to recordClock.toList(),
                "playClock" to playClock.toList(),
                "underruns" to underruns.toList(),
                "parts" to parts.map { linkedMapOf<String, Any?>("file" to it.file, "firstFrame" to it.firstFrame, "frames" to it.frames) },
                "sensors" to sensors?.toMap(),
            ),
        )
    }

    private fun clock(c: ClockPoint) = linkedMapOf<String, Any?>("wallMs" to c.wallMs, "elapsedNs" to c.elapsedNs)

    companion object {
        const val FORMAT = 1
    }
}
