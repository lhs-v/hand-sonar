package com.watchpose.handsonar.shared

import java.io.File
import java.util.Locale

/** 녹음 조각 하나의 기록(rec.json의 parts, 설계 §7.1). [firstFrame]은 녹음 전체에서 이 조각 첫 프레임의 번호다. */
data class PartInfo(val file: String, val firstFrame: Long, val frames: Long)

/**
 * 녹음을 [framesPerPart]마다 새 WAV 조각(part-000.wav, part-001.wav, …)으로 나눠 쓴다(설계 §5.3).
 * 조각 경계에 걸친 덩어리는 둘로 나눠 앞 조각을 꽉 채운 뒤 다음 조각을 연다. 데이터가 없으면 조각을 만들지 않는다.
 * 녹음 스레드가 쓰고 서비스의 1초 틱이 [parts]를 읽으므로 공개 멤버는 모두 잠근다.
 * [close] 뒤의 [write]는 예외 없이 무시한다(샘플은 버려지고 [totalFrames]에도 들어가지 않는다). 정지 때 녹음 스레드가 늦게 끝나도
 * rec.json의 parts에 없는 조각을 새로 만들지 않게 하고, 정지와 겹친 쓰기가 예외로 녹음 스레드를 죽이지 않게 하려는 것이다.
 */
class PartWriter(
    private val dir: File,
    private val sampleRate: Int = SonarSpec.SAMPLE_RATE,
    private val framesPerPart: Int = SonarSpec.PART_FRAMES,
) : AutoCloseable {
    private val done = mutableListOf<PartInfo>()
    private var current: WavWriter? = null
    private var currentName = ""
    private var currentFirst = 0L
    private var closed = false

    /** 지금까지 쓴 전체 프레임 수. */
    @get:Synchronized
    var totalFrames: Long = 0
        private set

    /** 닫힌 조각과, 쓰는 중인 조각(지금까지 쓴 만큼). */
    val parts: List<PartInfo>
        @Synchronized get() = done + listOfNotNull(current?.let { PartInfo(currentName, currentFirst, it.frames) })

    /** samples[0 until count]를 조각에 나눠 쓴다. [close] 뒤에 부르면 아무것도 하지 않는다. */
    @Synchronized
    fun write(samples: ShortArray, count: Int = samples.size) {
        if (closed) return
        var offset = 0
        while (offset < count) {
            val w = current ?: open()
            val room = framesPerPart - w.frames
            val n = minOf(room, (count - offset).toLong()).toInt()
            if (offset == 0 && n == count) w.write(samples, count) else w.write(samples.copyOfRange(offset, offset + n), n)
            offset += n
            totalFrames += n
            if (w.frames >= framesPerPart) closeCurrent()
        }
    }

    @Synchronized
    override fun close() {
        closed = true
        closeCurrent()
    }

    private fun open(): WavWriter {
        currentName = String.format(Locale.ROOT, "part-%03d.wav", done.size)
        currentFirst = totalFrames
        return WavWriter(File(dir, currentName), sampleRate).also { current = it }
    }

    private fun closeCurrent() {
        val w = current ?: return
        w.close()
        done += PartInfo(currentName, currentFirst, w.frames)
        current = null
    }
}
