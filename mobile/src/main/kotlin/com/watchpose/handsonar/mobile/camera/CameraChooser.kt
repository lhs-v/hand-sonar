package com.watchpose.handsonar.mobile.camera

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider

/** 고른 정면 카메라. [display]는 "inner"(펼친 안쪽 화면) | "cover"(접은 덮개 화면). */
data class ChosenCamera(val id: String, val display: String)

/**
 * 지금 쓰는 화면 쪽 정면 카메라를 고른다(설계 §6.4). 폴드8의 앱용 정면 카메라는 둘이다: "1"(덮개 화면, 3648×2736)과
 * "3"(안쪽 화면, 2784×2080). 화면의 가장 짧은 변이 600 dp 이상이면 안쪽 화면으로 보고 화소가 적은 쪽을, 아니면 많은 쪽을 고른다.
 * 화소가 같으면 번호가 작은 쪽. 정면 카메라가 하나뿐인 폰이면 그것을 쓴다.
 */
object CameraChooser {
    const val INNER = "inner"
    const val COVER = "cover"
    const val REALTIME = "realtime"
    const val ARRIVAL = "arrival"

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    fun choose(provider: ProcessCameraProvider, smallestWidthDp: Int): ChosenCamera? {
        val fronts = provider.availableCameraInfos.filter { it.lensFacing == CameraSelector.LENS_FACING_FRONT }
        if (fronts.isEmpty()) return null
        val display = if (smallestWidthDp >= 600) INNER else COVER
        val candidates = fronts.map { info ->
            val c = Camera2CameraInfo.from(info)
            val size = c.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            Triple(c.cameraId, size?.let { it.width.toLong() * it.height } ?: 0L, c.cameraId.toIntOrNull() ?: Int.MAX_VALUE)
        }
        val pick = if (display == INNER) {
            candidates.sortedWith(compareBy({ it.second }, { it.third })).first()
        } else {
            candidates.sortedWith(compareBy({ -it.second }, { it.third })).first()
        }
        return ChosenCamera(pick.first, display)
    }

    /** 이 카메라의 프레임 시각이 elapsedNs(REALTIME)이면 "realtime", 아니면 "arrival"(프레임 도착 시각으로 대신함, 근사). */
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    fun timeBase(provider: ProcessCameraProvider, id: String): String {
        val info = provider.availableCameraInfos.firstOrNull { Camera2CameraInfo.from(it).cameraId == id } ?: return ARRIVAL
        val source = Camera2CameraInfo.from(info).getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        return if (source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) REALTIME else ARRIVAL
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    fun selector(id: String): CameraSelector = CameraSelector.Builder()
        .addCameraFilter { infos -> infos.filter { Camera2CameraInfo.from(it).cameraId == id } }
        .build()
}
