package com.watchpose.handsonar.shared

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 16비트 1채널 PCM WAV를 이어 쓴다. 머리말의 길이 칸은 [close] 때 고친다.
 * 닫기 전에 앱이 죽으면 data 길이가 0으로 남는다(RIFF 길이는 36). PC 점검 도구는 그때 파일 크기로 길이를 다시 잰다.
 */
class WavWriter(file: File, private val sampleRate: Int) : AutoCloseable {
    private val out = RandomAccessFile(file, "rw").apply { setLength(0) }
    private val buffer = ByteArray(BUFFER_SAMPLES * 2)

    /** 지금까지 쓴 프레임(샘플) 수. */
    var frames: Long = 0
        private set

    init {
        out.write(header(0))
    }

    /** samples[0 until count]를 덧붙인다. */
    fun write(samples: ShortArray, count: Int = samples.size) {
        var i = 0
        while (i < count) {
            val n = minOf(BUFFER_SAMPLES, count - i)
            for (k in 0 until n) {
                val v = samples[i + k].toInt()
                buffer[2 * k] = (v and 0xFF).toByte()
                buffer[2 * k + 1] = ((v shr 8) and 0xFF).toByte()
            }
            out.write(buffer, 0, n * 2)
            i += n
        }
        frames += count
    }

    override fun close() {
        out.seek(0)
        out.write(header(frames * 2))
        out.close()
    }

    private fun header(dataBytes: Long): ByteArray {
        val b = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray(Charsets.US_ASCII))
        b.putInt((36 + dataBytes).toInt())
        b.put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
        b.putInt(16)
        b.putShort(1) // PCM
        b.putShort(1) // 1채널
        b.putInt(sampleRate)
        b.putInt(sampleRate * 2)
        b.putShort(2)
        b.putShort(16)
        b.put("data".toByteArray(Charsets.US_ASCII))
        b.putInt(dataBytes.toInt())
        return b.array()
    }

    companion object {
        const val HEADER_BYTES = 44
        private const val BUFFER_SAMPLES = 4_096
    }
}
