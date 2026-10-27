package com.watchpose.handsonar.mobile

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.ui.App
import com.watchpose.handsonar.mobile.ui.HandSonarTheme

/** 화면 하나짜리 앱. 수집 중 화면이 꺼지지 않게 하고, 앞/뒤 전환을 조정자에 알린다(벗어나면 세션 일시정지, 설계 §6.6). */
class MainActivity : ComponentActivity() {
    private val controller: SessionController get() = (application as HandSonarPhoneApp).controller

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { HandSonarTheme { App(controller, this) } }
    }

    override fun onStart() {
        super.onStart()
        controller.onAppVisible(true)
    }

    override fun onStop() {
        controller.onAppVisible(false)
        super.onStop()
    }
}
