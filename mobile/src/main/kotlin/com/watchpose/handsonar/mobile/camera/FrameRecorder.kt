package com.watchpose.handsonar.mobile.camera

import android.graphics.Bitmap
import com.watchpose.handsonar.shared.LandmarkCsv
import java.io.File

/**
 * 세션 동안 프레임을 파일로 쓴다(설계 §6.4·§7.2): landmarks.csv 한 줄과 같은 프레임의 작은 영상 한 장.
 * [write]는 카메라 분석 스레드에서, [close]는 다른 스레드에서 부르므로 잠금으로 막는다. 닫은 뒤의 [write]는 무시한다.
 * 영상 시각은 첫 프레임을 0으로 한 카메라 시각(µs)이다.
 */
class FrameRecorder(dir: File) {
    private val lock = Any()
    private val csv = File(dir, LANDMARKS_FILE).bufferedWriter(Charsets.UTF_8, 1 shl 16)
    private val videoFile = File(dir, VIDEO_FILE)
    private var encoder: VideoEncoder? = null
    private var firstNs = -1L
    private var closed = false

    @Volatile var frames: Long = 0
        private set

    @Volatile var handFrames: Long = 0
        private set

    @Volatile var videoWidth: Int = 0
        private set

    @Volatile var videoHeight: Int = 0
        private set

    /** mp4에 실제로 쓴 프레임 수(정상이면 [frames]와 같다). */
    val videoFrames: Long get() = encoder?.written ?: 0

    init {
        csv.write(LandmarkCsv.HEADER)
        csv.newLine()
    }

    fun write(frame: CameraFrame) {
        synchronized(lock) {
            if (closed) return
            val bitmap = frame.bitmap
            val hand = frame.hand
            val enc = encoder ?: run {
                val (w, h) = VideoEncoder.sizeFor(bitmap.width, bitmap.height)
                videoWidth = w
                videoHeight = h
                VideoEncoder(videoFile, w, h).also {
                    it.start()
                    encoder = it
                }
            }
            val scaled = Bitmap.createScaledBitmap(bitmap, videoWidth, videoHeight, true)
            val pixels = IntArray(videoWidth * videoHeight)
            scaled.getPixels(pixels, 0, videoWidth, 0, 0, videoWidth, videoHeight)
            if (scaled !== bitmap) scaled.recycle()
            csv.write(LandmarkCsv.row(hand.frameNs, hand.hands, hand.handedness, hand.score, hand.image, hand.world))
            csv.newLine()
            frames++
            if (hand.image != null) handFrames++
            if (firstNs < 0) firstNs = hand.frameNs
            enc.submit(pixels, (hand.frameNs - firstNs) / 1_000)
        }
    }

    /** CSV를 닫고 영상을 마무리한다. 영상이 성하면(또는 프레임이 하나도 없었으면) true. */
    fun close(): Boolean {
        val enc: VideoEncoder?
        val csvOk: Boolean
        synchronized(lock) {
            if (closed) return encoder?.error == null
            closed = true
            enc = encoder
            csvOk = runCatching { csv.close() }.isSuccess
        }
        val videoOk = enc?.finish() ?: true
        return csvOk && videoOk
    }

    companion object {
        const val VIDEO_FILE = "video.mp4"
        const val LANDMARKS_FILE = "landmarks.csv"
    }
}
