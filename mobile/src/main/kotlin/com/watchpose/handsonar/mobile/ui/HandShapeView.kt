package com.watchpose.handsonar.mobile.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.scale
import com.watchpose.handsonar.shared.FingerState
import com.watchpose.handsonar.shared.HandShape
import com.watchpose.handsonar.shared.WristPose

private val SKIN = Color(0xFFE0B48C)

/**
 * 동작 손 모양 그림(설계 §6.1). 거울 미리보기와 같은 방향의 왼손이다: 엄지가 오른쪽, 새끼손가락이 왼쪽.
 * [right]가 참이면 팔뚝까지 그림 전체의 좌우를 뒤집어 오른손으로 그린다(엄지가 왼쪽, 새끼손가락이 오른쪽).
 * 편 손가락은 길게 살색, 굽힌 손가락은 짧게 강조색, 엄지에 닿는 손가락(TOUCH)은 짧게 그리고 엄지 끝까지 선을 잇는다.
 * 새끼 쪽 꺾기(LEFT)는 손을 새끼손가락 쪽(왼손은 왼쪽, 오른손은 오른쪽)으로 기울여 보이고, 위·아래 꺾기는 손을 조금 줄여 보인다
 * (방향은 아래 안내 글이 말한다).
 */
@Composable
fun HandShapeView(shape: HandShape, modifier: Modifier = Modifier, right: Boolean = false) {
    val accent = StatusColors.warn
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val stroke = w * 0.11f
        scale(scaleX = if (right) -1f else 1f, scaleY = 1f, pivot = center) {
            drawRoundRect(
                SKIN.copy(alpha = 0.6f),
                topLeft = Offset(w * 0.30f, h * 0.80f),
                size = Size(w * 0.34f, h * 0.20f),
                cornerRadius = CornerRadius(w * 0.04f),
            )
            val wrist = Offset(w * 0.47f, h * 0.82f)
            val tilt = if (shape.wrist == WristPose.LEFT) -25f else 0f
            val squash = if (shape.wrist == WristPose.UP || shape.wrist == WristPose.DOWN) 0.8f else 1f
            rotate(tilt, pivot = wrist) {
                scale(scaleX = 1f, scaleY = squash, pivot = wrist) {
                    drawRoundRect(
                        SKIN,
                        topLeft = Offset(w * 0.22f, h * 0.45f),
                        size = Size(w * 0.50f, h * 0.38f),
                        cornerRadius = CornerRadius(w * 0.08f),
                    )
                    val thumbExtended = shape.fingers[0] == FingerState.EXTENDED
                    val thumbBase = Offset(w * 0.70f, h * 0.70f)
                    val thumbTip = if (thumbExtended) Offset(w * 0.92f, h * 0.46f) else Offset(w * 0.56f, h * 0.56f)
                    drawLine(if (thumbExtended) SKIN else accent, thumbBase, thumbTip, strokeWidth = stroke, cap = StrokeCap.Round)
                    for (k in 1..4) { // 검지(엄지 옆) → 소지(왼쪽)
                        val x = w * (0.64f - (k - 1) * 0.125f)
                        val state = shape.fingers[k]
                        val tipY = when (state) {
                            FingerState.EXTENDED -> h * 0.06f
                            FingerState.BENT -> h * 0.36f
                            FingerState.TOUCH -> h * 0.33f
                        }
                        val tip = Offset(x, tipY)
                        drawLine(if (state == FingerState.EXTENDED) SKIN else accent, Offset(x, h * 0.47f), tip, strokeWidth = stroke, cap = StrokeCap.Round)
                        if (state == FingerState.TOUCH) {
                            drawLine(accent, tip, thumbTip, strokeWidth = stroke * 0.3f, cap = StrokeCap.Round)
                            drawCircle(accent, radius = stroke * 0.35f, center = thumbTip)
                        }
                    }
                }
            }
        }
    }
}
