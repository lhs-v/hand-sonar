package com.watchpose.handsonar.mobile.camera

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/**
 * 추적하는 손(워치를 찬 손). [id]는 landmarks.csv의 handedness와 meta.json의 hand("left" | "right")에 적는 이름이고,
 * [mediapipe]는 거울 그림을 넣었을 때 MediaPipe가 그 손에 붙이는 이름이다(2026-10-27 폴드8 실기기: 실제 왼손이 "Right",
 * 오른손이 "Left", 설계 §11-5 확인). 미리보기·CSV 좌표·영상은 거울 그림 그대로 둔다. [label]은 화면에 보이는 이름이다.
 */
enum class TrackedHand(val id: String, val label: String, val mediapipe: String) {
    LEFT("Left", "왼손", "Right"),
    RIGHT("Right", "오른손", "Left"),
}

/**
 * 프레임 하나의 손 추적 결과(설계 §6.4). [hands]는 찾은 손 개수(추적하는 손이 아니어도 센다). [image]·[world]는 고른 손
 * ([TrackedHand])의 21점 x,y,z(63개): image는 거울 프레임 기준 0..1 좌표, world는 손 중심 기준 미터.
 * 추적하는 손이 없으면 handedness부터 null이다.
 */
class HandFrame(
    val frameNs: Long,
    val hands: Int,
    val handedness: String?,
    val score: Float?,
    val image: FloatArray?,
    val world: FloatArray?,
)

/**
 * MediaPipe Hand Landmarker를 VIDEO 모드(동기)로 돌린다. 손은 2개까지 찾고, 추적하는 손([hand], MediaPipe 이름은
 * [TrackedHand.mediapipe]) 중 점수가 가장 높은 것을 고른다. 넣는 프레임은 똑바로 세우고 좌우를 뒤집은 거울 그림이다.
 * 카메라 분석 스레드 하나에서만 부른다.
 */
class HandTracker(private val context: Context) : AutoCloseable {
    /** 추적하는 손. [CameraPipeline]이 프레임마다 [CameraPipeline.trackedHand]로 맞춘다. */
    @Volatile var hand: TrackedHand = TrackedHand.LEFT

    /**
     * 참이면 추적하는 손 이름의 손이 없어도 화면에 손이 하나뿐일 때 그 손을 쓴다. 기준 자세 옆면(엄지가 카메라 쪽)에서는 MediaPipe가
     * 왼손·오른손을 헷갈릴 수 있어서 그 단계 동안만 켠다([CameraPipeline.anyHand]).
     */
    @Volatile var anyHand: Boolean = false

    /**
     * 쓰는 계산 장치("GPU" | "CPU"). CPU만으로는 폴드8에서 프레임당 33 ms를 넘어 초당 약 15–18이었다(2026-10-27 실기기).
     * GPU를 먼저 쓰고, 만들거나 돌리다 실패하면 CPU로 바꿔 계속한다.
     */
    @Volatile var delegate: String = DELEGATE_GPU
        private set

    private var landmarker: HandLandmarker = try {
        create(Delegate.GPU)
    } catch (e: Exception) {
        delegate = DELEGATE_CPU
        create(Delegate.CPU)
    }
    private var lastMs = Long.MIN_VALUE

    private fun create(d: Delegate): HandLandmarker = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL).setDelegate(d).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build(),
    )

    fun detect(bitmap: Bitmap, frameNs: Long): HandFrame {
        val want = hand // 프레임 도중에 바뀌어도 이 프레임은 한 손으로만 고른다
        val ms = maxOf(frameNs / 1_000_000, lastMs + 1) // VIDEO 모드는 시각이 늘 늘어나야 한다
        lastMs = ms
        // MPImage를 닫으면 감싼 Bitmap까지 재활용될 수 있어 닫지 않는다(미리보기·영상이 같은 Bitmap을 쓴다).
        val result = try {
            landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), ms)
        } catch (e: Exception) {
            if (delegate != DELEGATE_GPU) throw e
            landmarker.close() // GPU가 돌다 실패하면 CPU로 바꿔 이 프레임부터 다시 한다
            landmarker = create(Delegate.CPU)
            delegate = DELEGATE_CPU
            landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), ms)
        }
        val count = result.landmarks().size
        var best = -1
        var bestScore = -1f
        for (i in 0 until count) {
            val category = result.handedness().getOrNull(i)?.firstOrNull() ?: continue
            if (category.categoryName() == want.mediapipe && category.score() > bestScore) {
                best = i
                bestScore = category.score()
            }
        }
        if (best < 0 && anyHand && count == 1) { // 손이 하나뿐이면 이름이 달라도 그 손(기준 자세 옆면)
            best = 0
            bestScore = result.handedness().getOrNull(0)?.firstOrNull()?.score() ?: 0f
        }
        val points = result.landmarks().getOrNull(best)
        val worldPoints = result.worldLandmarks().getOrNull(best)
        if (best < 0 || points == null || worldPoints == null) return HandFrame(frameNs, count, null, null, null, null)
        val image = FloatArray(63)
        points.forEachIndexed { k, p ->
            image[3 * k] = p.x()
            image[3 * k + 1] = p.y()
            image[3 * k + 2] = p.z()
        }
        val world = FloatArray(63)
        worldPoints.forEachIndexed { k, p ->
            world[3 * k] = p.x()
            world[3 * k + 1] = p.y()
            world[3 * k + 2] = p.z()
        }
        return HandFrame(frameNs, count, want.id, bestScore, image, world)
    }

    override fun close() = landmarker.close()

    companion object {
        const val MODEL = "hand_landmarker.task"

        const val DELEGATE_GPU = "GPU"
        const val DELEGATE_CPU = "CPU"
    }
}
