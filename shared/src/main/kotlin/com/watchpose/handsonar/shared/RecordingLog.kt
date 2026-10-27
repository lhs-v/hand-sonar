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
 * 녹음 스레드(녹음 시각)와 서비스 틱(재생 시각·끊김·조각)이 함께 덧붙이므로 모든 공개 멤버는 [lock]으로 잠근다.
 * recordClock 한 줄은 [녹음 위치, elapsedNs, 그때까지 읽은 프레임 수]다. 녹음이 넘쳐 샘플이 빠지면 첫째와 셋째의 차이가 튄다.
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
) {
    private val lock = Any()
    private val recordClock = mutableListOf<LongArray>()
    private val playClock = mutableListOf<LongArray>()
    private val underruns = mutableListOf<LongArray>()
    private var recordClockSource = "timestamp"
    private var parts: List<PartInfo> = emptyList()
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
                "volume" to linkedMapOf<String, Any?>("stream" to volumeStream, "index" to volumeIndex, "max" to volumeMax),
                "recordClockSource" to recordClockSource,
                "recordClock" to recordClock.toList(),
                "playClock" to playClock.toList(),
                "underruns" to underruns.toList(),
                "parts" to parts.map { linkedMapOf<String, Any?>("file" to it.file, "firstFrame" to it.firstFrame, "frames" to it.frames) },
            ),
        )
    }

    private fun clock(c: ClockPoint) = linkedMapOf<String, Any?>("wallMs" to c.wallMs, "elapsedNs" to c.elapsedNs)

    companion object {
        const val FORMAT = 1
    }
}
