package com.watchpose.handsonar.wear.transfer

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.CapabilityClient
import com.google.android.gms.wearable.ChannelClient
import com.google.android.gms.wearable.Node
import com.google.android.gms.wearable.Wearable
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.TransferProtocol
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** 파일 하나를 보낸 결과. */
sealed interface FileResult {
    /** 폰이 확인했다(OK) 또는 이미 같은 크기로 가지고 있었다(HAVE). */
    data object Done : FileResult

    /** 녹음이 시작돼 끊었다. */
    data object Held : FileResult

    data class Failed(val reason: String) : FileResult
}

/**
 * 워치 쪽 데이터 레이어 호출(전송 설계 §2). 모든 메서드는 보내기 서비스의 작업 스레드에서 부른다: `Tasks.await`로 기다리므로
 * 주 스레드에서 부르면 IllegalStateException이 난다([abort]만 주 스레드에서 부른다).
 * 파일 결과는 폰이 채널을 닫을 때 넘긴 코드다. 채널 닫힘은 앱 전체 콜백 하나로 받아 채널 객체(토큰·노드·경로로 견준다)마다 자리에 넣는다.
 * 보내는 쪽과 콜백이 모두 `computeIfAbsent`로 같은 자리를 쓰므로, 폰이 우리가 기다리기 전에 닫아도(이미 있음) 놓치지 않는다.
 */
class PhoneSender(context: Context) : AutoCloseable {
    /** 채널이 닫힌 까닭. [remote]면 폰이 닫았고 [code]가 그 결과 코드다. */
    private data class End(val remote: Boolean, val code: Int)

    private val capabilities = Wearable.getCapabilityClient(context)
    private val messages = Wearable.getMessageClient(context)
    private val channels = Wearable.getChannelClient(context)
    private val main = Handler(Looper.getMainLooper())
    private val ends = ConcurrentHashMap<ChannelClient.Channel, CompletableFuture<End>>()
    @Volatile private var current: ChannelClient.Channel? = null
    @Volatile private var registered = false
    private val callback = object : ChannelClient.ChannelCallback() {
        override fun onChannelClosed(channel: ChannelClient.Channel, closeReason: Int, appSpecificErrorCode: Int) {
            endOf(channel).complete(End(closeReason == ChannelClient.ChannelCallback.CLOSE_REASON_REMOTE_CLOSE, appSpecificErrorCode))
        }
    }

    /** 채널 닫힘 콜백을 단다. 못 달면 false(파일 결과를 받을 수 없으므로 보내지 않는다). */
    fun open(): Boolean {
        registered = runCatching { Tasks.await(channels.registerChannelCallback(callback), TIMEOUT_S, TimeUnit.SECONDS) }.isSuccess
        return registered
    }

    /** 폰 앱이 선언한 capability를 가진, 지금 닿는 노드(가까운 노드 먼저). 없거나 실패하면 null. */
    fun findPhone(): Node? = runCatching {
        Tasks.await(capabilities.getCapability(TransferProtocol.CAPABILITY, CapabilityClient.FILTER_REACHABLE), TIMEOUT_S, TimeUnit.SECONDS)
            .nodes
            .sortedByDescending { it.isNearby }
            .firstOrNull()
    }.getOrNull()

    /** 목록 요청을 보내고 폰에 없는 파일 이름을 받는다. 답이 없거나 읽지 못하면 null. */
    fun offer(phone: String, offer: Offer): List<String>? = runCatching {
        val reply = Tasks.await(messages.sendRequest(phone, TransferProtocol.OFFER_PATH, TransferProtocol.encodeOffer(offer)), TIMEOUT_S, TimeUnit.SECONDS)
        TransferProtocol.decodeNeed(reply)
    }.getOrNull()

    /**
     * [file]을 채널 하나로 처음부터 끝까지 보내고 폰의 결과 코드를 기다린다. [held]가 참이 되면(녹음 시작) 채널을 CANCELLED로 닫고 Held.
     * [STALL_MS] 동안 한 바이트도 못 쓰면 STALLED로 닫는다(블루투스가 끊겼는데 쓰기가 막힌 채일 때).
     * 다 쓴 뒤에는 폰의 결과를 [closeWaitMs]까지 기다린다. [onProgress]는 덩어리마다 지금까지 쓴 바이트다.
     */
    fun sendFile(phone: String, recording: String, file: File, size: Long, held: () -> Boolean, onProgress: (Long) -> Unit): FileResult {
        val path = TransferProtocol.channelPath(recording, file.name, size)
        ends.clear() // 앞 파일들의 늦은 닫힘이 쌓이지 않게. 이 파일의 채널은 아직 없다
        val channel = runCatching { Tasks.await(channels.openChannel(phone, path), TIMEOUT_S, TimeUnit.SECONDS) }
            .getOrElse { return FileResult.Failed("채널을 못 엶(${reason(it)})") }
        current = channel
        val lastWrite = AtomicLong(SystemClock.elapsedRealtime())
        val stalled = AtomicBoolean(false)
        val watchdog = object : Runnable {
            override fun run() {
                if (SystemClock.elapsedRealtime() - lastWrite.get() >= STALL_MS) {
                    stalled.set(true)
                    channels.close(channel, TransferProtocol.CODE_STALLED)
                } else {
                    main.postDelayed(this, WATCHDOG_MS)
                }
            }
        }
        try {
            main.postDelayed(watchdog, WATCHDOG_MS)
            val cancelled = try {
                write(channel, file, held, lastWrite, onProgress)
            } finally {
                main.removeCallbacks(watchdog)
            }
            if (cancelled) {
                channels.close(channel, TransferProtocol.CODE_CANCELLED)
                return FileResult.Held
            }
            val end = runCatching { endOf(channel).get(closeWaitMs(size), TimeUnit.MILLISECONDS) }.getOrNull()
            return when {
                end != null && end.remote && (end.code == TransferProtocol.CODE_OK || end.code == TransferProtocol.CODE_HAVE) -> FileResult.Done
                end != null && end.remote -> FileResult.Failed("폰: ${TransferProtocol.describe(end.code)}")
                stalled.get() -> FileResult.Failed(TransferProtocol.describe(TransferProtocol.CODE_STALLED))
                end != null -> FileResult.Failed("연결이 끊김")
                else -> FileResult.Failed("폰이 확인하지 않음")
            }
        } finally {
            current = null
            ends.remove(channel)
            channels.close(channel) // 이미 닫혔으면 아무 일도 없다(그 Task의 실패는 버린다)
        }
    }

    /** 보내던 채널을 끊는다(서비스가 멈출 때, 주 스레드). 막힌 쓰기가 풀린다. */
    fun abort() {
        current?.let { channels.close(it, TransferProtocol.CODE_CANCELLED) }
    }

    override fun close() {
        if (registered) channels.unregisterChannelCallback(callback)
        registered = false
    }

    /**
     * 파일을 출력 스트림에 16 KB씩 쓰고 닫는다. [held]가 참이 되면 그만두고 true. 폰이 먼저 닫았거나(이미 있음·오류) 연결이 끊겨
     * 쓰기가 실패해도 false다: 닫힌 까닭은 콜백이 알려 준다.
     */
    private fun write(channel: ChannelClient.Channel, file: File, held: () -> Boolean, lastWrite: AtomicLong, onProgress: (Long) -> Unit): Boolean =
        runCatching {
            Tasks.await(channels.getOutputStream(channel), TIMEOUT_S, TimeUnit.SECONDS).use { out ->
                FileInputStream(file).use { input ->
                    val buffer = ByteArray(CHUNK_BYTES)
                    var sent = 0L
                    while (true) {
                        if (held()) return@runCatching true
                        val n = input.read(buffer)
                        if (n < 0) break
                        out.write(buffer, 0, n)
                        sent += n
                        lastWrite.set(SystemClock.elapsedRealtime())
                        onProgress(sent)
                    }
                }
            }
            false
        }.getOrDefault(false)

    private fun endOf(channel: ChannelClient.Channel): CompletableFuture<End> = ends.computeIfAbsent(channel) { CompletableFuture() }

    private fun reason(e: Throwable): String = (e.cause ?: e).let { it.message ?: it.javaClass.simpleName }

    private companion object {
        const val TIMEOUT_S = 30L
        const val CHUNK_BYTES = 16 * 1024
        const val STALL_MS = 90_000L
        const val WATCHDOG_MS = 10_000L

        /** 다 쓴 뒤 폰의 결과를 기다리는 시간: 120 s + 크기 ÷ 20 KB/s(Play 서비스가 쌓아 둔 것을 천천히 보낼 수 있어서). */
        fun closeWaitMs(size: Long): Long = 120_000L + size / 20
    }
}
