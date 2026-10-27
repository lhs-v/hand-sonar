package com.watchpose.handsonar.wear.record

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.MediaRecorder
import android.os.Process
import android.os.SystemClock
import com.watchpose.handsonar.shared.PartWriter
import com.watchpose.handsonar.shared.SonarSpec

/**
 * 48 kHz 16비트 1채널로 녹음해 [writer]에 그대로 쓴다(설계 §5.2). 녹음 소스는 UNPROCESSED를 지원하면 그것,
 * 아니면 VOICE_RECOGNITION이다(워치8은 VOICE_RECOGNITION). 녹음 버퍼를 1초로 크게 잡아 파일 쓰기가 잠깐 막혀도 넘치지 않게 한다.
 * 1초마다 [onClock]으로 (녹음 위치, 부팅 후 시각, 읽은 프레임 수, 타임스탬프에서 왔는지)를 낸다.
 * 녹음이 끊기거나 예외가 나면 [onError]를 한 번 부르고 멈춘다. 다시 열지 않는다: 이어 붙이면 시간이 어긋난다.
 *
 * 녹음 시각은 한 녹음 안에서 출처 하나로만 낸다. rec.json의 recordClockSource에는 값 하나만 적을 수 있고 PC는 그 값으로
 * 기록을 해석하므로, 타임스탬프 기록과 읽은 직후 기록이 한 파일에 섞이면 PC가 둘을 가를 수 없기 때문이다. 규칙:
 * 1. 1초마다 AudioRecord.getTimestamp(BOOTTIME)로 잰다. 성공하면 그 값을 fromTimestamp = true로 낸다.
 * 2. 실패한 초는 기록을 건너뛴다. 녹음 시작 직후에는 타임스탬프가 아직 준비되지 않아 처음 한두 번 실패할 수 있다.
 * 3. 다만 시작 뒤 한 번도 성공하지 못한 채 3번(SWITCH_AFTER_FAILURES) 연속 실패하면 이 기기는 타임스탬프를 주지 않는 것으로 보고,
 *    그 순간부터 녹음이 끝날 때까지 "읽은 직후 시각"으로 바꾼다: 그 초부터 매초 (읽은 프레임 수, SystemClock.elapsedRealtimeNanos())를
 *    fromTimestamp = false로 낸다. 한 번 바꾸면 타임스탬프로 되돌리지 않는다. 바꾸기 전에는 성공한 적이 없어 앞선 기록이 없으므로
 *    파일에는 한 출처만 남는다.
 * 4. 한 번이라도 성공한 뒤의 실패는 그 초만 건너뛴다. 방식을 바꾸지 않는다(일시적인 실패이고, 바꾸면 두 출처가 섞인다).
 */
class SonarRecorder(
    private val context: Context,
    private val writer: PartWriter,
    private val onClock: (frame: Long, elapsedNs: Long, framesRead: Long, fromTimestamp: Boolean) -> Unit,
    private val onError: (String) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null
    private var record: AudioRecord? = null

    /** 지금까지 읽어 쓴 프레임 수. */
    @Volatile
    var frames: Long = 0
        private set

    /** rec.json의 audio.source 값. */
    val sourceName: String
        get() = if (source == MediaRecorder.AudioSource.UNPROCESSED) "UNPROCESSED" else "VOICE_RECOGNITION"

    private val source: Int by lazy {
        val am = context.getSystemService(AudioManager::class.java)
        if (am?.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true") {
            MediaRecorder.AudioSource.UNPROCESSED
        } else {
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        }
    }

    /** 녹음을 열고 녹음 스레드를 시작한다. 되면 null, 안 되면 사람이 읽을 이유. */
    @SuppressLint("MissingPermission") // 화면이 RECORD_AUDIO를 받은 뒤에만 서비스를 시작한다
    fun start(): String? {
        val rate = SonarSpec.SAMPLE_RATE
        val minBuffer = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val r = try {
            AudioRecord(source, rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, SonarSpec.BYTES_PER_SECOND))
        } catch (e: Exception) {
            return "녹음을 못 엶: ${e.message ?: e.javaClass.simpleName}"
        }
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            r.release()
            return "녹음 초기화 실패"
        }
        val started = runCatching { r.startRecording() }.isSuccess && r.recordingState == AudioRecord.RECORDSTATE_RECORDING
        if (!started) {
            r.release()
            return "녹음 시작 실패(다른 앱이 마이크를 쓰는 중일 수 있음)"
        }
        record = r
        running = true
        thread = Thread({ loop(r) }, "handsonar-record").also { it.start() }
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
    }

    private fun loop(r: AudioRecord) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val buffer = ShortArray(READ_FRAMES)
        val ts = AudioTimestamp()
        var nextClockAt = 0L
        var timestampWorked = false // 타임스탬프가 한 번이라도 성공했다(규칙 4)
        var failures = 0 // 성공하기 전까지의 연속 실패 횟수(규칙 3). 성공한 적이 없는 동안의 실패는 모두 연속이다
        var readTimeMode = false // 규칙 3으로 바뀐 뒤 true. 한 번 true가 되면 녹음이 끝날 때까지 되돌리지 않는다
        try {
            while (running) {
                val n = r.read(buffer, 0, buffer.size)
                if (!running) break
                if (n <= 0) {
                    running = false
                    onError(if (n < 0) "녹음 읽기 오류 ($n)" else "녹음이 멈춤")
                    return
                }
                writer.write(buffer, n)
                frames += n
                if (frames >= nextClockAt) {
                    if (!readTimeMode) {
                        if (r.getTimestamp(ts, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS) {
                            timestampWorked = true
                            onClock(ts.framePosition, ts.nanoTime, frames, true)
                        } else if (!timestampWorked) {
                            failures++
                            if (failures >= SWITCH_AFTER_FAILURES) readTimeMode = true
                        }
                    }
                    // 방금 읽은 덩어리의 끝을 지금 시각으로 본다(근사). 바꾼 그 초에도 낸다.
                    if (readTimeMode) onClock(frames, SystemClock.elapsedRealtimeNanos(), frames, false)
                    nextClockAt += SonarSpec.SAMPLE_RATE
                }
            }
        } catch (e: Exception) {
            if (running) {
                running = false
                onError("녹음 오류: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    private companion object {
        /** 한 번에 읽는 양(20 ms). */
        const val READ_FRAMES = 960

        /**
         * 타임스탬프가 한 번도 성공하지 못한 채 이만큼 연속 실패하면 "읽은 직후 시각"으로 바꾼다(KDoc 규칙 3).
         * 첫 시도가 첫 읽기(시작 약 20 ms 뒤)이고 이후 1초마다 재므로, 약 2초 동안 준비되지 않은 기기를 가려낸다.
         */
        const val SWITCH_AFTER_FAILURES = 3
        const val STOP_JOIN_MS = 500L
    }
}
