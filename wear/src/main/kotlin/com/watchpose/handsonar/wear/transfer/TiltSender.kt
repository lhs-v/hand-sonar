package com.watchpose.handsonar.wear.transfer

import android.content.Context
import android.os.SystemClock
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.CapabilityClient
import com.google.android.gms.wearable.Wearable
import com.watchpose.handsonar.shared.Tilt
import com.watchpose.handsonar.shared.TiltMessage
import com.watchpose.handsonar.shared.TransferProtocol
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.util.concurrent.TimeUnit

/**
 * 녹음하는 동안 워치 중력 값을 [TiltMessage.SEND_MS]마다 폰으로 보낸다(shared [TiltMessage]). 폰은 받은 값으로 기준 자세에서 팔뚝이 섰는지 본다.
 * 워치는 재고 보내기만 하고 판정은 폰이 한다. 폰을 못 찾거나 보내기가 실패해도 녹음에는 영향이 없다: 그 값은 버리고, 폰 노드는
 * [FIND_EVERY_MS]마다 다시 찾는다. [run]은 녹음 서비스가 Dispatchers.Default에서 부르고(노드 찾기가 기다리므로 주 스레드가 아니어야 한다),
 * 녹음이 끝나면 그 코루틴을 취소한다.
 */
class TiltSender(context: Context) {
    private val capabilities = Wearable.getCapabilityClient(context)
    private val messages = Wearable.getMessageClient(context)
    @Volatile private var phone: String? = null // 보내기 실패 콜백(주 스레드)이 지운다
    private var lastFindMs = Long.MIN_VALUE / 2

    suspend fun run(latest: () -> Tilt?) {
        while (currentCoroutineContext().isActive) {
            latest()?.let(::send)
            delay(TiltMessage.SEND_MS)
        }
    }

    private fun send(t: Tilt) {
        val now = SystemClock.elapsedRealtime()
        if (phone == null && now - lastFindMs >= FIND_EVERY_MS) {
            lastFindMs = now
            phone = findPhone()
        }
        val node = phone ?: return
        runCatching { messages.sendMessage(node, TiltMessage.PATH, TiltMessage.encode(t)).addOnFailureListener { phone = null } }
    }

    /** 폰 앱이 선언한 capability를 가진, 지금 닿는 노드(가까운 노드 먼저). 없거나 실패하면 null. */
    private fun findPhone(): String? = runCatching {
        val task = capabilities.getCapability(TransferProtocol.CAPABILITY, CapabilityClient.FILTER_REACHABLE)
        Tasks.await(task, FIND_TIMEOUT_MS, TimeUnit.MILLISECONDS).nodes.sortedByDescending { it.isNearby }.firstOrNull()?.id
    }.getOrNull()

    private companion object {
        const val FIND_EVERY_MS = 5_000L
        const val FIND_TIMEOUT_MS = 2_000L
    }
}
