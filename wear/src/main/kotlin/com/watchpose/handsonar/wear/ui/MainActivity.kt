package com.watchpose.handsonar.wear.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.content.ContextCompat
import com.watchpose.handsonar.wear.HandSonarApp
import com.watchpose.handsonar.wear.record.RecorderService
import com.watchpose.handsonar.wear.record.RecorderState

/**
 * 워치의 유일한 화면을 연다. 컨트롤러(PC)가 adb로 녹음을 원격 시작할 수 있다(설계 §5.1):
 * `adb shell am start -n com.watchpose.handsonar/com.watchpose.handsonar.wear.ui.MainActivity --ez start true --ei autoStopMinutes 10`
 * 서비스만 adb로 띄우면 앱이 백그라운드라서 포그라운드 서비스 시작 제한과 마이크 사용 제한에 걸린다.
 * 이 액티비티가 앞에 떠 있을 때 서비스를 시작하므로 그 제한에 걸리지 않는다.
 * 대신 원격 시작은 권한 창을 띄우지 않으므로(사람이 눌러야 한다) 마이크 권한이 이미 허용돼 있어야 한다.
 * 허용돼 있지 않으면 시작하지 않고 화면에 이유를 보인다.
 * `autoStopMinutes`는 이번 녹음만 설정의 자동 정지를 대신한다(0 = 끔). 생략하면 설정을 따른다.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = application as HandSonarApp
        setContent { RecorderApp(app) }
        // 화면이 다시 만들어질 때(회전·프로세스 복원)는 처음 인텐트가 남아 있어도 다시 시작하지 않는다
        if (savedInstanceState == null) remoteStart(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        remoteStart(intent)
    }

    private fun remoteStart(intent: Intent) {
        if (!intent.getBooleanExtra(EXTRA_START, false) || RecorderState.status.value.running) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            RecorderService.start(this, intent.getIntExtra(RecorderService.EXTRA_AUTO_STOP_MINUTES, -1).takeIf { it >= 0 })
        } else {
            RecorderState.update { it.copy(error = "마이크 권한이 없어 원격 시작을 못 했습니다. 워치에서 녹음을 한 번 시작해 권한을 허용하세요") }
        }
    }

    private companion object {
        const val EXTRA_START = "start"
    }
}
