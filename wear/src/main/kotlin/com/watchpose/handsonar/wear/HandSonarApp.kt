package com.watchpose.handsonar.wear

import android.app.Application
import com.watchpose.handsonar.shared.NameLog
import com.watchpose.handsonar.shared.RecordVolumeRules
import com.watchpose.handsonar.wear.record.RecordVolumeKeeper
import com.watchpose.handsonar.wear.record.SonarSettings
import java.io.File

/** 앱에 하나뿐인 설정을 들고 있다(화면과 서비스가 같은 값을 본다). */
class HandSonarApp : Application() {
    val settings: SonarSettings by lazy { SonarSettings(this) }

    /**
     * 녹음 볼륨 맞추기·되돌리기(계획 6). 녹음 서비스와 화면(앱이 죽었거나 미룬 되돌리기)이 같은 것을 쓴다: 이 녹음이 볼륨을 바꿨는지를
     * 프로세스 안에서 기억한다.
     */
    val volumeRules: RecordVolumeRules by lazy { RecordVolumeKeeper(this).let { RecordVolumeRules(it, it) } }

    /** 녹음 폴더들의 부모(앱 외부 저장소 `files/recordings`, 수집 설계 §5.3). 녹음 서비스가 쓰고 보내기 서비스가 읽는다. */
    val recordingsRoot: File get() = File(getExternalFilesDir(null) ?: filesDir, "recordings")

    /** 폰이 다 받았다고 답한 녹음(전송 설계 §2). 녹음 폴더 밖(앱 내부 저장소)에 두어 PC로 받는 폴더 모양을 바꾸지 않는다. */
    val sentLog: NameLog by lazy { NameLog(File(filesDir, "sent-recordings.txt")) }
}
