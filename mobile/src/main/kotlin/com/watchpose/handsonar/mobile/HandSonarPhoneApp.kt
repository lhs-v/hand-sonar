package com.watchpose.handsonar.mobile

import android.app.Application
import com.watchpose.handsonar.mobile.session.SessionController
import com.watchpose.handsonar.mobile.transfer.WatchInbox

/** 앱 하나에 하나. 화면이 다시 만들어져도 세션이 이어지도록 조정자를 여기에 둔다. */
class HandSonarPhoneApp : Application() {
    val controller: SessionController by lazy { SessionController(this) }

    /** 워치에서 받은 녹음(전송 설계 §3). 데이터 레이어 서비스(앱 화면이 없을 때도)와 화면이 같은 것을 본다. */
    val inbox: WatchInbox by lazy { WatchInbox(this) }
}
