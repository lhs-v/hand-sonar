package com.watchpose.handsonar.mobile.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.MediaRecorder
import android.os.Process
import android.os.SystemClock
import com.watchpose.handsonar.shared.ChirpCheck
import com.watchpose.handsonar.shared.ClockLine
import com.watchpose.handsonar.shared.MarkerDetector
import com.watchpose.handsonar.shared.RecordClockRule
import com.watchpose.handsonar.shared.WatchChirpDetector
import com.watchpose.handsonar.shared.WavWriter
import java.io.File

/** 세션 하나의 폰 녹음 결과. [clock]의 위치·읽은 수는 phone_audio.wav 첫 샘플을 0으로 센다. [startTotalFrame]은 그 첫 샘플의 마이크 프레임 번호. */
class SessionAudio(
    val source: String,
    val clockSource: String,
    val clock: List<ClockLine>,
    val frames: Long,
    val startTotalFrame: Long,
    val error: String?,
)

/** 고리 버퍼에서 찾은 표지음. [totalFrame]은 마이크를 연 뒤부터 센 프레임 번호(소수). */
class MarkerFound(val totalFrame: Double, val ratio: Double)

/**
 * 폰 마이크(설계 §6.5). 준비·수집 화면 동안 48 kHz 16비트 1채널로 계속 녹음한다.
 * - 0.5 s마다 워치 처프 들림을 판정해 [onChirp]로 낸다(녹음 스레드에서).
 * - 최근 [RING_SECONDS]초를 고리 버퍼에 두어 표지음을 찾는다([findMarker]).
 * - 세션 동안([beginSession]~[endSession]) phone_audio.wav에 쓰고, 1 s마다 녹음 시각을 [RecordClockRule]로 적는다.
 * 녹음 소스는 UNPROCESSED를 지원하면 그것, 아니면 VOICE_RECOGNITION이다.
 */
class PhoneMic(
    private val context: Context,
    private val onChirp: (ChirpCheck) -> Unit,
    private val onError: (String) -> Unit,
) {
    private class Session(val wav: WavWriter, val startTotal: Long) {
        val rule = RecordClockRule()
        val clock = mutableListOf<ClockLine>()
        var frames = 0L
        var nextClockAt = 0L
        var error: String? = null
    }

    @Volatile var running = false
        private set
    private var thread: Thread? = null
    private var record: AudioRecord? = null

    private val ring = ShortArray(RING_SECONDS * RATE)
    private var ringTotal = 0L // 고리 버퍼에 쓴 전체 프레임 수 = 마이크를 연 뒤 읽은 수
    private val ringLock = Any()

    private val sessionLock = Any()
    private var pendingWav: File? = null
    private var session: Session? = null

    @Volatile private var lastTsPosition = -1L
    @Volatile private var lastTsNs = 0L

    private val source: Int by lazy {
        val am = context.getSystemService(AudioManager::class.java)
        if (am?.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true") {
            MediaRecorder.AudioSource.UNPROCESSED
        } else {
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        }
    }

    /** phone_audio.json의 source 값. */
    val sourceName: String
        get() = if (source == MediaRecorder.AudioSource.UNPROCESSED) "UNPROCESSED" else "VOICE_RECOGNITION"

    /** 마이크를 열고 녹음 스레드를 시작한다. 되면(또는 이미 돌고 있으면) null, 안 되면 사람이 읽을 이유. */
    @SuppressLint("MissingPermission") // 화면이 RECORD_AUDIO를 받은 뒤에만 부른다
    fun start(): String? {
        if (running) return null
        stop() // 마이크 오류로 멈춘 뒤라면 남은 녹음기를 먼저 푼다(처음이면 할 일 없음)
        val minBuffer = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val r = try {
            AudioRecord(source, RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, RATE * 2))
        } catch (e: Exception) {
            return "마이크를 못 엶: ${e.message ?: e.javaClass.simpleName}"
        }
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            r.release()
            return "마이크 초기화 실패"
        }
        val started = runCatching { r.startRecording() }.isSuccess && r.recordingState == AudioRecord.RECORDSTATE_RECORDING
        if (!started) {
            r.release()
            return "마이크 시작 실패(다른 앱이 마이크를 쓰는 중일 수 있음)"
        }
        // 새 AudioRecord의 타임스탬프 위치는 0부터 센다. 다시 열 때 고리 버퍼 번호와 최근 타임스탬프도 처음부터 센다.
        synchronized(ringLock) { ringTotal = 0L }
        lastTsPosition = -1L
        lastTsNs = 0L
        record = r
        running = true
        thread = Thread({ loop(r) }, "handsonar-mic").also { it.start() }
        return null
    }

    fun stop() {
        running = false
        thread?.join(STOP_JOIN_MS)
        thread = null
        record?.let {
            runCatching { it.stop() }
            it.release()
        }
        record = null
        synchronized(sessionLock) {
            session?.let { runCatching { it.wav.close() } }
            session = null
            pendingWav = null
        }
    }

    /** 다음 읽기부터 [wav]에 쓴다. */
    fun beginSession(wav: File) {
        synchronized(sessionLock) { pendingWav = wav }
    }

    /** 세션 녹음을 닫고 결과를 돌려준다. 세션이 없었으면 null. */
    fun endSession(): SessionAudio? {
        val s = synchronized(sessionLock) {
            pendingWav = null
            session.also { session = null }
        } ?: return null
        val closeError = runCatching { s.wav.close() }.exceptionOrNull()?.let { "phone_audio.wav 닫기 실패: ${it.message}" }
        return SessionAudio(sourceName, s.rule.source, s.clock.toList(), s.frames, s.startTotal, s.error ?: closeError)
    }

    /** 고리 버퍼의 최근 [lookbackSec]초에서 표지음을 찾는다. 없거나 아직 그만큼 녹음하지 않았으면 null. */
    fun findMarker(lookbackSec: Double = 1.5): MarkerFound? {
        val n = (lookbackSec * RATE).toInt().coerceAtMost(ring.size)
        val x = FloatArray(n)
        val first: Long
        synchronized(ringLock) {
            if (ringTotal < n) return null
            first = ringTotal - n
            for (i in 0 until n) x[i] = ring[((first + i) % ring.size).toInt()].toFloat()
        }
        val hit = MarkerDetector.find(x) ?: return null
        return MarkerFound(first + hit.sample, hit.ratio)
    }

    /** 마이크 프레임 번호의 elapsedNs(0.5 s마다 갱신하는 최근 타임스탬프 기준). 타임스탬프를 아직 못 받았으면 null. */
    fun elapsedNsOf(totalFrame: Double): Long? {
        val position = lastTsPosition
        if (position < 0) return null
        return lastTsNs + ((totalFrame - position) * 1e9 / RATE).toLong()
    }

    private fun loop(r: AudioRecord) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val buffer = ShortArray(READ_FRAMES)
        val block = ShortArray(WatchChirpDetector.BLOCK_SAMPLES)
        var blockFill = 0
        val ts = AudioTimestamp()
        try {
            while (running) {
                val n = r.read(buffer, 0, buffer.size)
                if (!running) break
                if (n <= 0) {
                    running = false
                    onError(if (n < 0) "마이크 읽기 오류 ($n)" else "마이크가 멈춤")
                    return
                }
                val totalBefore: Long
                synchronized(ringLock) {
                    totalBefore = ringTotal
                    for (i in 0 until n) ring[((ringTotal + i) % ring.size).toInt()] = buffer[i]
                    ringTotal += n
                }
                writeSession(r, buffer, n, totalBefore, ts)
                var i = 0
                while (i < n) {
                    val k = minOf(n - i, block.size - blockFill)
                    System.arraycopy(buffer, i, block, blockFill, k)
                    blockFill += k
                    i += k
                    if (blockFill == block.size) {
                        onChirp(WatchChirpDetector.check(block))
                        blockFill = 0
                        if (r.getTimestamp(ts, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS) {
                            lastTsNs = ts.nanoTime
                            lastTsPosition = ts.framePosition
                        }
                    }
                }
            }
        } catch (e: Exception) {
            if (running) {
                running = false
                onError("마이크 오류: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    private fun writeSession(r: AudioRecord, buffer: ShortArray, n: Int, totalBefore: Long, ts: AudioTimestamp) {
        synchronized(sessionLock) {
            pendingWav?.let { f ->
                pendingWav = null
                session?.let { runCatching { it.wav.close() } }
                session = try {
                    Session(WavWriter(f, RATE), totalBefore)
                } catch (e: Exception) {
                    onError("phone_audio.wav를 못 엶: ${e.message ?: e.javaClass.simpleName}")
                    null
                }
            }
            val s = session ?: return
            if (s.error != null) return
            try {
                s.wav.write(buffer, n)
                s.frames += n
                if (s.frames >= s.nextClockAt) {
                    val ok = r.getTimestamp(ts, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS
                    s.rule.onSecond(ok, ts.framePosition - s.startTotal, ts.nanoTime, s.frames, SystemClock.elapsedRealtimeNanos())
                        ?.let { s.clock += it }
                    s.nextClockAt += RATE
                }
            } catch (e: Exception) {
                val message = "phone_audio.wav 쓰기 오류: ${e.message ?: e.javaClass.simpleName}"
                s.error = message
                onError(message)
            }
        }
    }

    companion object {
        const val RATE = 48_000
        const val RING_SECONDS = 3
        private const val READ_FRAMES = 960
        private const val STOP_JOIN_MS = 500L
    }
}
