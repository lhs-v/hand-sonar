package com.watchpose.handsonar.wear.transfer

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** 워치 화면의 보내기 상태(전송 설계 §2). 보내기 서비스가 고치고, 녹음하지 않을 때 화면이 한 줄로 그린다. */
data class TransferStatus(
    /** 보내기 서비스가 도는 중. */
    val active: Boolean = false,
    /** 아직 폰이 확인하지 않은 녹음 수. */
    val pending: Int = 0,
    /** 보내는 중인 파일(없으면 빈 글). */
    val file: String = "",
    val bytes: Long = 0L,
    val total: Long = 0L,
    /** 마지막 결과나 멈춘 까닭(없으면 null). */
    val message: String? = null,
)

object TransferState {
    private val mutable = MutableStateFlow(TransferStatus())
    val status: StateFlow<TransferStatus> = mutable.asStateFlow()

    fun update(change: (TransferStatus) -> TransferStatus) = mutable.update(change)
}
