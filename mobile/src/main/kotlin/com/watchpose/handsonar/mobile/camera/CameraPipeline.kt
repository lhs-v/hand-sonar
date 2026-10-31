package com.watchpose.handsonar.mobile.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CaptureRequest
import android.os.SystemClock
import android.util.Range
import android.util.Size
import androidx.annotation.MainThread
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** 분석 스레드에서 나오는 프레임 하나: 똑바로 세우고 좌우를 뒤집은(거울) 그림과 그 프레임의 손 추적 결과. */
class CameraFrame(val bitmap: Bitmap, val hand: HandFrame)

/**
 * 정면 카메라를 CameraX ImageAnalysis 하나로 연다(설계 §6.4): RGBA, 640×480 근처, 초당 30 고정. 프레임마다 똑바로 세우고
 * (ImageProxy 회전 + [extraRotation]) 좌우를 뒤집어(거울) [HandTracker]에 넣고, 그 결과를 [onFrame]으로 낸다.
 * 미리보기도 이 거울 그림을 그대로 그리므로 관절 좌표와 화면이 같은 좌표계다. [onFrame]·[onError]는 분석 스레드에서 불린다.
 * 프레임 시각은 카메라가 REALTIME(= elapsedNs)이면 센서 시각, 아니면 프레임이 도착한 elapsedNs다([timeBase]).
 */
class CameraPipeline(
    private val context: Context,
    private val onFrame: (CameraFrame) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "handsonar-camera") }
    private var provider: ProcessCameraProvider? = null
    private var tracker: HandTracker? = null // 분석 스레드에서만 만들고 쓴다
    @Volatile private var reported: String? = null

    @Volatile var chosen: ChosenCamera? = null
        private set

    @Volatile var timeBase: String = CameraChooser.ARRIVAL
        private set

    @Volatile var frameWidth: Int = 0
        private set

    @Volatile var frameHeight: Int = 0
        private set

    /** 마지막 프레임에 적용한 회전(°). */
    @Volatile var rotation: Int = 0

    /** 최근 초당 프레임(지수 이동 평균). 준비 화면 카메라 표시줄에 보인다. */
    @Volatile var fps: Float = 0f
        private set

    /** 손 추적이 쓰는 계산 장치("GPU" | "CPU"), 아직 모르면 "". */
    @Volatile var delegate: String = ""
        private set

    private var lastFrameNs = 0L
        private set

    /** 기기에서 그림이 눕혀 보이면 준비 화면의 "회전 보정"으로 90°씩 더한다(0·90·180·270). */
    @Volatile var extraRotation: Int = 0

    /** 추적하는 손(워치를 찬 손). 준비 화면에서 고르고, 분석 스레드가 프레임마다 [HandTracker]에 넘긴다. */
    @Volatile var trackedHand: TrackedHand = TrackedHand.LEFT

    /** 참이면 손이 하나만 보일 때 이름과 상관없이 그 손을 추적한다([HandTracker.anyHand], 기준 자세 동안). */
    @Volatile var anyHand: Boolean = false

    @MainThread
    fun start(owner: LifecycleOwner, smallestWidthDp: Int, displayRotation: Int) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val p = try {
                future.get()
            } catch (e: Exception) {
                report("카메라를 못 엶: ${e.message ?: e.javaClass.simpleName}")
                return@addListener
            }
            provider = p
            val pick = CameraChooser.choose(p, smallestWidthDp)
            if (pick == null) {
                report("정면 카메라가 없습니다")
                return@addListener
            }
            chosen = pick
            timeBase = CameraChooser.timeBase(p, pick.id)
            val analysis = buildAnalysis(displayRotation)
            analysis.setAnalyzer(executor) { proxy -> analyze(proxy) }
            try {
                p.unbindAll()
                p.bindToLifecycle(owner, CameraChooser.selector(pick.id), analysis)
            } catch (e: Exception) {
                report("카메라 연결 실패: ${e.message ?: e.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    @MainThread
    fun stop() {
        provider?.unbindAll()
    }

    /** 앱이 끝날 때 한 번. */
    @MainThread
    fun close() {
        stop()
        executor.execute {
            tracker?.close()
            tracker = null
        }
        executor.shutdown()
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun buildAnalysis(displayRotation: Int): ImageAnalysis {
        val builder = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(displayRotation)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                    )
                    .build(),
            )
        Camera2Interop.Extender(builder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(FPS, FPS))
        return builder.build()
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val t = tracker ?: try {
                HandTracker(context).also { tracker = it }
            } catch (e: Exception) {
                report("손 추적 모델을 못 엶: ${e.message ?: e.javaClass.simpleName}")
                return
            }
            val frameNs = if (timeBase == CameraChooser.REALTIME) proxy.imageInfo.timestamp else SystemClock.elapsedRealtimeNanos()
            val degrees = (proxy.imageInfo.rotationDegrees + extraRotation) % 360
            val raw = proxy.toBitmap()
            val matrix = Matrix().apply {
                postRotate(degrees.toFloat())
                postScale(-1f, 1f)
            }
            val mirrored = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
            if (mirrored !== raw) raw.recycle()
            rotation = degrees
            frameWidth = mirrored.width
            frameHeight = mirrored.height
            t.hand = trackedHand
            t.anyHand = anyHand
            onFrame(CameraFrame(mirrored, t.detect(mirrored, frameNs)))
            delegate = t.delegate
            if (lastFrameNs > 0 && frameNs > lastFrameNs) {
                val now = 1e9f / (frameNs - lastFrameNs)
                fps = if (fps == 0f) now else 0.9f * fps + 0.1f * now
            }
            lastFrameNs = frameNs
        } catch (e: Exception) {
            report("카메라 프레임 처리 오류: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            proxy.close()
        }
    }

    private fun report(message: String) {
        if (reported == message) return
        reported = message
        onError(message)
    }

    companion object {
        const val ANALYSIS_WIDTH = 640
        const val ANALYSIS_HEIGHT = 480
        const val FPS = 30
    }
}
