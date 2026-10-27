package com.watchpose.handsonar.wear

import android.app.Application
import com.watchpose.handsonar.wear.record.SonarSettings

/** 앱에 하나뿐인 설정을 들고 있다(화면과 서비스가 같은 값을 본다). */
class HandSonarApp : Application() {
    val settings: SonarSettings by lazy { SonarSettings(this) }
}
