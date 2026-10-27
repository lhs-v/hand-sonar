package com.watchpose.handsonar.mobile.camera

import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
import com.watchpose.handsonar.shared.I420
import com.watchpose.handsonar.shared.Yuv
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

/**
 * 작은 영상(설계 §6.4): H.264, 약 1 Mbps, 초당 30, 긴 변 480 px. [submit]으로 받은 ARGB 프레임을 자기 스레드에서 차례로 인코딩해
 * MP4로 쓴다. 프레임을 버리지 않는다(영상 n번째 프레임 = landmarks.csv n번째 줄): 대기 줄(8장)이 차면 [submit]이 기다린다.
 * 오류가 나면 [error]에 이유를 남기고, 그 뒤로 받는 프레임은 버린다(분석 스레드가 막히지 않게).
 */
class VideoEncoder(private val file: File, val width: Int, val height: Int) {
    private class Item(val argb: IntArray?, val ptsUs: Long)

    private class State {
        var track = -1
        var muxerStarted = false
    }

    private val queue = ArrayBlockingQueue<Item>(QUEUE)
    private val thread = Thread({ run() }, "handsonar-video")

    @Volatile var error: String? = null
        private set

    /** 인코더에 넣은 프레임 수. */
    @Volatile var frames: Long = 0
        private set

    /** mp4에 실제로 쓴 샘플 수. 정상이면 [frames]와 같다. */
    @Volatile var written: Long = 0
        private set

    fun start() = thread.start()

    /** [argb]는 width×height ARGB 픽셀, [ptsUs]는 늘어나는 영상 시각(µs). 오류 뒤에는 버린다. 대기줄이 2 s 넘게 막히면 오류로 보고 버린다. */
    fun submit(argb: IntArray, ptsUs: Long) {
        if (error != null) return
        if (!queue.offer(Item(argb, ptsUs), SUBMIT_WAIT_S, TimeUnit.SECONDS)) error = "영상 대기줄이 ${SUBMIT_WAIT_S} s 넘게 막힘"
    }

    /** 남은 프레임을 다 넣고 파일을 닫는다. 영상이 성하면 true. */
    fun finish(timeoutMs: Long = 10_000): Boolean {
        queue.offer(Item(null, 0), 2, TimeUnit.SECONDS)
        thread.join(timeoutMs)
        if (thread.isAlive && error == null) error = "영상 저장이 ${timeoutMs / 1_000} s 안에 끝나지 않음"
        if (error == null && written != frames) error = "영상에 쓴 프레임 ${written}개가 넣은 ${frames}개와 다름"
        return error == null
    }

    private fun run() {
        var codec: MediaCodec? = null
        var muxer: MediaMuxer? = null
        val state = State()
        try {
            val format = MediaFormat.createVideoFormat(MIME, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE)
                setInteger(MediaFormat.KEY_FRAME_RATE, 30)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            val c = MediaCodec.createEncoderByType(MIME)
            codec = c
            c.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            c.start()
            val m = MediaMuxer(file.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = m
            val yuv = I420(width, height)
            var lastPts = -1L
            while (true) {
                val item = queue.take()
                val pixels = item.argb ?: break
                Yuv.argbToI420(pixels, width, height, yuv)
                val index = dequeueInput(c, m, state)
                val image = c.getInputImage(index) ?: throw IllegalStateException("인코더 입력 그림 없음")
                copyPlane(yuv.y, width, height, image.planes[0])
                copyPlane(yuv.u, width / 2, height / 2, image.planes[1])
                copyPlane(yuv.v, width / 2, height / 2, image.planes[2])
                val pts = if (item.ptsUs > lastPts) item.ptsUs else lastPts + 1
                lastPts = pts
                c.queueInputBuffer(index, 0, width * height * 3 / 2, pts, 0)
                frames++
                drain(c, m, state, endOfStream = false)
            }
            val index = dequeueInput(c, m, state)
            c.queueInputBuffer(index, 0, 0, lastPts + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            if (!drain(c, m, state, endOfStream = true)) error = "영상 끝 표시를 받지 못함(꼬리가 잘렸을 수 있음)"
        } catch (e: Throwable) {
            error = "영상 저장 오류: ${e.message ?: e.javaClass.simpleName}"
            while (true) { // 끝 표시가 올 때까지 남은 프레임을 버린다
                val item = queue.poll(30, TimeUnit.SECONDS) ?: break
                if (item.argb == null) break
            }
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            if (state.muxerStarted) {
                try {
                    muxer?.stop()
                } catch (e: Exception) {
                    if (error == null) error = "영상 파일 마무리 실패: ${e.message ?: e.javaClass.simpleName}"
                }
            }
            runCatching { muxer?.release() }
        }
    }

    private fun dequeueInput(c: MediaCodec, m: MediaMuxer, state: State): Int {
        repeat(INPUT_TRIES) {
            val index = c.dequeueInputBuffer(10_000)
            if (index >= 0) return index
            drain(c, m, state, endOfStream = false)
        }
        throw IllegalStateException("인코더 입력 버퍼가 비지 않음")
    }

    private fun drain(c: MediaCodec, m: MediaMuxer, state: State, endOfStream: Boolean): Boolean {
        val info = MediaCodec.BufferInfo()
        var waits = 0
        while (true) {
            val index = c.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream || ++waits > OUTPUT_WAITS) return false
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    state.track = m.addTrack(c.outputFormat)
                    m.start()
                    state.muxerStarted = true
                }
                index >= 0 -> {
                    val buffer = c.getOutputBuffer(index)
                    val isConfig = (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0
                    if (buffer != null && info.size > 0 && !isConfig && state.muxerStarted) {
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        m.writeSampleData(state.track, buffer, info)
                        written++
                    }
                    c.releaseOutputBuffer(index, false)
                    if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) return true
                }
            }
        }
    }

    /** 평면 하나를 인코더 입력 그림에 옮긴다. 행 간격(rowStride)과 화소 간격(pixelStride)을 따른다(NV12처럼 섞인 U·V도 된다). */
    private fun copyPlane(src: ByteArray, w: Int, h: Int, plane: Image.Plane) {
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (row in 0 until h) {
            val base = row * rowStride
            if (pixelStride == 1) {
                buffer.position(base)
                buffer.put(src, row * w, w)
            } else {
                var o = base
                var s = row * w
                repeat(w) {
                    buffer.put(o, src[s])
                    o += pixelStride
                    s++
                }
            }
        }
    }

    companion object {
        private const val MIME = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val BIT_RATE = 1_000_000
        private const val QUEUE = 8
        private const val INPUT_TRIES = 300
        private const val OUTPUT_WAITS = 300
        private const val SUBMIT_WAIT_S = 2L
        const val LONG_SIDE = 480

        /** 원본 크기를 긴 변 480 px로 줄인 영상 크기(짝수). 인코더가 그 크기를 못 받으면 16의 배수로 내린다. */
        fun sizeFor(srcW: Int, srcH: Int): Pair<Int, Int> {
            val scale = LONG_SIDE.toDouble() / maxOf(srcW, srcH)
            var w = ((srcW * scale).roundToInt() / 2) * 2
            var h = ((srcH * scale).roundToInt() / 2) * 2
            val caps = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
                .firstOrNull { it.isEncoder && it.supportedTypes.any { t -> t.equals(MIME, ignoreCase = true) } }
                ?.getCapabilitiesForType(MIME)?.videoCapabilities
            if (caps != null && !caps.isSizeSupported(w, h)) {
                w = w / 16 * 16
                h = h / 16 * 16
            }
            return w to h
        }
    }
}
