package com.watchpose.handsonar.mobile.audio

import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build

/**
 * 연결된 외부 오디오 출력(이어폰·블루투스 오디오 등)을 찾는다. 이런 출력이 있으면 안드로이드가 미디어 볼륨을 그쪽 값으로 다뤄
 * [MarkerPlayer]가 올리는 볼륨이 폰 스피커에 닿지 않을 수 있으므로, 준비 화면이 미리 알린다.
 */
object AudioOutputs {
    /** 외부 출력으로 치는 장치 형과, 장치가 이름을 주지 않을 때 대신 보일 이름. 블루투스 LE 형은 API 31·33부터 있다. */
    private val KINDS: Map<Int, String> = buildMap {
        put(AudioDeviceInfo.TYPE_WIRED_HEADSET, "유선 헤드셋")
        put(AudioDeviceInfo.TYPE_WIRED_HEADPHONES, "유선 헤드폰")
        put(AudioDeviceInfo.TYPE_BLUETOOTH_A2DP, "블루투스 오디오")
        put(AudioDeviceInfo.TYPE_BLUETOOTH_SCO, "블루투스 통화")
        put(AudioDeviceInfo.TYPE_USB_HEADSET, "USB 헤드셋")
        put(AudioDeviceInfo.TYPE_HEARING_AID, "보청기")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            put(AudioDeviceInfo.TYPE_BLE_HEADSET, "BLE 헤드셋")
            put(AudioDeviceInfo.TYPE_BLE_SPEAKER, "BLE 스피커")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            put(AudioDeviceInfo.TYPE_BLE_BROADCAST, "BLE 브로드캐스트")
        }
    }

    /** 지금 연결된 외부 출력의 이름(같은 이름은 하나로). 없으면 빈 목록. 이름은 장치가 알리는 이름, 비면 형 이름이다. */
    fun externalNames(audioManager: AudioManager?): List<String> =
        audioManager?.getDevices(AudioManager.GET_DEVICES_OUTPUTS).orEmpty().mapNotNull { device ->
            val kindName = KINDS[device.type] ?: return@mapNotNull null
            val product = device.productName.toString().trim()
            // 이름이 없는 장치에는 시스템이 폰 모델명을 대신 채워 주는 일이 있어, 그것도 이름으로 치지 않는다
            if (product.isEmpty() || product == Build.MODEL) kindName else product
        }.distinct()
}
