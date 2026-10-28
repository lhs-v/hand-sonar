package com.watchpose.handsonar.wear.record

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** 화면에 보이는 녹음 상태(설계 §5.1). 서비스가 시작·정지 때와 1초마다 고친다. */
data class RecorderStatus(
    val running: Boolean = false,
    /** 녹음을 시작한 부팅 후 시각(ms). 화면이 지난 시간을 그린다. */
    val startedElapsedMs: Long = 0L,
    /** 자동 정지 시각(부팅 후 ms, SystemClock.elapsedRealtime 기준). 0이면 자동 정지가 없다. 화면이 남은 시간을 그린다. */
    val autoStopAtElapsedMs: Long = 0L,
    val recordedFrames: Long = 0L,
    val underruns: Int = 0,
    val minutesLeft: Double = Double.NaN,
    val storageWarning: Boolean = false,
    val volumeIndex: Int = -1,
    val volumeMax: Int = -1,
    /** 녹음 볼륨으로 바꿨으면 멈출 때 되돌릴 원래 볼륨(계획 6). 바꾸지 않았으면 −1. */
    val volumeRestoreTo: Int = -1,
    /** 녹음 폴더 이름(yyyyMMdd-HHmmss). PC로 받을 때 찾는다. */
    val folder: String = "",
    /** 센서 기록의 문제(못 엶·쓰기 실패·기기에 없는 센서, 계획 6). 없으면 null. 센서 기록이 실패해도 녹음은 이어 간다. */
    val sensorNote: String? = null,
    val error: String? = null,
)

object RecorderState {
    private val mutable = MutableStateFlow(RecorderStatus())
    val status: StateFlow<RecorderStatus> = mutable.asStateFlow()

    fun update(change: (RecorderStatus) -> RecorderStatus) = mutable.update(change)
}
