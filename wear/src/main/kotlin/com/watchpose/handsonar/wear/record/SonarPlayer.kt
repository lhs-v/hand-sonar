package com.watchpose.handsonar.wear.record

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.os.Process
import android.os.SystemClock
import com.watchpose.handsonar.shared.SonarSpec

/**
 * 처프 한 주기([sweep])를 쉬지 않고 이어 재생한다(설계 §5.2). 내장 스피커로 고정하고 미디어 볼륨을 따른다.
 * 한 번에 처프 8개(100 ms)를 블로킹으로 써서 재생 속도로 돈다. 덩어리가 처프 길이의 배수라 이음새가 처프 경계와 맞는다.
 * 쓰기가 덜 되거나 예외가 나면 [onError]를 한 번 부르고 스스로 멈춘다. 한 번 시작·정지한 객체는 다시 쓰지 않는다.
 */
class SonarPlayer(private val context: Context, private val sweep: ShortArray, private val onError: (String) -> Unit) {
    @Volatile private var track: AudioTrack? = null
    @Volatile private var running = false
    private var thread: Thread? = null

    /** 트랙에 넘긴 프레임 수. */
    @Volatile
    var framesWritten: Long = 0
        private set

    /** AudioTrack이 센 재생 끊김(underrun) 횟수. */
    val underruns: Int get() = track?.let { runCatching { it.underrunCount }.getOrDefault(0) } ?: 0

    /** 트랙을 열고 재생 스레드를 시작한다. 되면 null, 안 되면 사람이 읽을 이유. */
    fun start(): String? {
        val rate = SonarSpec.SAMPLE_RATE
        val minBuffer = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val t = try {
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build(),
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(rate)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build(),
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(maxOf(minBuffer, CHUNK_FRAMES * 2 * 2))
                .build()
        } catch (e: Exception) {
            return "재생 트랙을 못 엶: ${e.message ?: e.javaClass.simpleName}"
        }
        if (t.state != AudioTrack.STATE_INITIALIZED) {
            t.release()
            return "재생 트랙 초기화 실패"
        }
        val am = context.getSystemService(AudioManager::class.java)
        am.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }?.let { t.setPreferredDevice(it) }
        if (runCatching { t.play() }.isFailure) {
            t.release()
            return "재생 시작 실패"
        }
        track = t
        running = true
        thread = Thread({ loop(t) }, "handsonar-play").also { it.start() }
        return null
    }

    fun stop() {
        running = false
        thread?.join(STOP_JOIN_MS)
        thread = null
        track?.let {
            runCatching { it.pause() }
            runCatching { it.flush() }
            it.release()
        }
        track = null
    }

    /**
     * 재생 위치와 그 시각 [프레임, elapsedNs]. 트랙이 아직 시각을 못 주면 null.
     * AudioTrack 시각은 단조 시계(System.nanoTime) 기준이라, 지금 두 시계의 차이를 더해 부팅 후 시각으로 옮긴다(wakelock 중에는 그 차이가 변하지 않는다).
     */
    fun clock(): LongArray? {
        val t = track ?: return null
        val ts = AudioTimestamp()
        if (!runCatching { t.getTimestamp(ts) }.getOrDefault(false)) return null
        val shift = SystemClock.elapsedRealtimeNanos() - System.nanoTime()
        return longArrayOf(ts.framePosition, ts.nanoTime + shift)
    }

    private fun loop(t: AudioTrack) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val chunk = ShortArray(CHUNK_FRAMES) { sweep[it % sweep.size] }
        try {
            while (running) {
                val n = t.write(chunk, 0, chunk.size) // 블로킹: 버퍼가 차면 재생 속도로 기다린다
                if (!running) break
                if (n != chunk.size) {
                    running = false
                    onError("재생 쓰기 실패 ($n/${chunk.size})")
                    return
                }
                framesWritten += n
            }
        } catch (e: Exception) {
            if (running) {
                running = false
                onError("재생 오류: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    private companion object {
        /** 한 번에 쓰는 양: 처프 8개 = 4,800프레임 = 100 ms. 처프 길이의 배수여야 한다. */
        const val CHUNK_FRAMES = SonarSpec.SWEEP_SAMPLES * 8
        const val STOP_JOIN_MS = 300L
    }
}
