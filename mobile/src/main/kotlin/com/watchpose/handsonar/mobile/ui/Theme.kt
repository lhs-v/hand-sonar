package com.watchpose.handsonar.mobile.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** 상태 색. 글자 옆 점과 경고 글에만 쓴다. */
object StatusColors {
    val ok = Color(0xFF43A047)
    val warn = Color(0xFFFFB300)
    val bad = Color(0xFFE53935)
    val idle = Color(0xFF9E9E9E)
}

/** 관절 뼈대 색. */
object SkeletonColors {
    val bone = Color(0xFF7FD1FF)
    val joint = Color(0xFFFFFFFF)
}

/** 카메라 화면이 중심이라 어두운 테마 하나만 쓴다. */
@Composable
fun HandSonarTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = darkColorScheme(), content = content)
}
