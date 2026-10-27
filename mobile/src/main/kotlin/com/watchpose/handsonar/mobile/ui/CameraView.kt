package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.watchpose.handsonar.mobile.camera.CameraFrame
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.min
import kotlin.math.roundToInt

/** MediaPipe 손 연결 21선(점 번호 쌍). */
private val EDGES = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,
    0 to 5, 5 to 6, 6 to 7, 7 to 8,
    5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16,
    13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
)

/**
 * 거울 프레임과 왼손 관절 뼈대. 그림 비율을 지키며 칸 안에 맞춘다. [frames]를 여기서 모아 프레임이 바뀔 때 이 부분만 다시 그린다.
 * [message]가 있으면 가운데에 크게 쓴다(권한·오류 안내).
 */
@Composable
fun CameraView(frames: StateFlow<CameraFrame?>, modifier: Modifier = Modifier, message: String? = null) {
    val frame by frames.collectAsStateWithLifecycle()
    Box(modifier.background(Color.Black)) {
        Canvas(Modifier.fillMaxSize()) {
            val f = frame ?: return@Canvas
            val image = f.bitmap.asImageBitmap()
            val scale = min(size.width / image.width, size.height / image.height)
            val w = image.width * scale
            val h = image.height * scale
            val left = (size.width - w) / 2f
            val top = (size.height - h) / 2f
            drawImage(
                image,
                dstOffset = IntOffset(left.roundToInt(), top.roundToInt()),
                dstSize = IntSize(w.roundToInt(), h.roundToInt()),
            )
            val points = f.hand.image ?: return@Canvas
            fun at(i: Int) = Offset(left + points[3 * i] * w, top + points[3 * i + 1] * h)
            for ((a, b) in EDGES) drawLine(SkeletonColors.bone, at(a), at(b), strokeWidth = 6f, cap = StrokeCap.Round)
            for (i in 0 until 21) drawCircle(SkeletonColors.joint, radius = 7f, center = at(i))
        }
        if (message != null) {
            Text(
                message,
                Modifier.align(Alignment.Center).padding(24.dp),
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall,
            )
        }
    }
}
