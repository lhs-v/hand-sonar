package com.watchpose.handsonar.wear.record

import android.content.Context
import com.watchpose.handsonar.shared.SonarSpec
import com.watchpose.handsonar.shared.SweepWindow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 처프 창·진폭·자동 정지 분(설계 §5.1·§5.2). 앱 설정에 남겨 다음 녹음에도 쓴다. 녹음 중에는 바꾸지 않는다(화면이 감춘다).
 * 자동 정지 분은 [SonarSpec.AUTO_STOP_CHOICES] 안의 값이고(0 = 끔), 저장된 값이 거기에 없으면 끔으로 읽는다.
 */
class SonarSettings(context: Context) {
    data class Values(val window: SweepWindow, val amplitude: Float, val autoStopMinutes: Int)

    private val prefs = context.applicationContext.getSharedPreferences("sonar", Context.MODE_PRIVATE)
    private val mutable = MutableStateFlow(read())
    val state: StateFlow<Values> = mutable.asStateFlow()

    fun setWindow(window: SweepWindow) = save(mutable.value.copy(window = window))

    fun cycleAmplitude() = save(mutable.value.copy(amplitude = SonarSpec.nextAmplitude(mutable.value.amplitude)))

    fun cycleAutoStop() = save(mutable.value.copy(autoStopMinutes = SonarSpec.nextAutoStop(mutable.value.autoStopMinutes)))

    private fun read(): Values {
        val window = SweepWindow.entries.firstOrNull { it.id == prefs.getString(KEY_WINDOW, null) } ?: SonarSpec.DEFAULT_WINDOW
        val amplitude = prefs.getFloat(KEY_AMPLITUDE, SonarSpec.DEFAULT_AMPLITUDE).coerceIn(0.1f, 1f)
        val autoStop = prefs.getInt(KEY_AUTO_STOP, 0).takeIf { it in SonarSpec.AUTO_STOP_CHOICES } ?: 0
        return Values(window, amplitude, autoStop)
    }

    private fun save(v: Values) {
        prefs.edit()
            .putString(KEY_WINDOW, v.window.id)
            .putFloat(KEY_AMPLITUDE, v.amplitude)
            .putInt(KEY_AUTO_STOP, v.autoStopMinutes)
            .apply()
        mutable.value = v
    }

    private companion object {
        const val KEY_WINDOW = "window"
        const val KEY_AMPLITUDE = "amplitude"
        const val KEY_AUTO_STOP = "autoStopMinutes"
    }
}
