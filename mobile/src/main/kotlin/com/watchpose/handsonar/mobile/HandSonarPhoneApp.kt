package com.watchpose.handsonar.mobile

import android.app.Application
import com.watchpose.handsonar.mobile.session.SessionController

/** 앱 하나에 하나. 화면이 다시 만들어져도 세션이 이어지도록 조정자를 여기에 둔다. */
class HandSonarPhoneApp : Application() {
    val controller: SessionController by lazy { SessionController(this) }
}
