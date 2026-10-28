package com.watchpose.handsonar.wear.transfer

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.CapabilityClient
import com.google.android.gms.wearable.ChannelClient
import com.google.android.gms.wearable.Node
import com.google.android.gms.wearable.Wearable
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.TransferProtocol
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** 파일 하나를 보낸 결과. */
sealed interface FileResult {
    /** 폰이 확인했다(OK) 또는 이미 같은 크기로 가지고 있었다(HAVE). */
    data object Done : FileResult

    /** 녹음이 시작돼(또는 서비스가 멈춰) 끊었다. */
    data object Held : FileResult

    /** [permanent]면 다시 보내도 같은 답이 나올 거절이다(폰이 머리말 모양·이름을 거절함, [TransferProtocol.isPermanent]). */
    data class Failed(val reason: String, val permanent: Boolean = false) : FileResult
}

/**
 * 워치 쪽 데이터 레이어 호출(전송 설계 §2). 모든 메서드는 보내기 서비스의 작업 스레드에서 부른다([abort]만 주 스레드).
 * 기다림은 모두 [SLICE_MS]씩 `Tasks.await`로 나눠 하고 그 사이마다 [stop]을 본다: 녹음이 시작되면(또는 서비스가 멈추면) 쓰기 중이든
 * 폰의 답·결과를 기다리는 중이든 0.5 s 안에 그만두고 보내던 채널을 CANCELLED로 닫는다(녹음 중에는 보내지 않는다).
 * 파일 결과는 폰이 채널을 닫을 때 넘긴 코드다. 채널 닫힘은 앱 전체 콜백 하나로 받아 채널 객체(토큰·노드·경로로 견준다)마다 자리에 넣는다.
 * 보내는 쪽과 콜백이 모두 `computeIfAbsent`로 같은 자리를 쓰므로, 폰이 우리가 기다리기 전에 닫아도(이미 있음) 놓치지 않는다.
 */
class PhoneSender(context: Context, private val stop: () -> Boolean) : AutoCloseable {
    /** 채널이 닫힌 까닭. [remote]면 폰이 닫았고 [code]가 그 결과 코드다. */
    private data class End(val remote: Boolean, val code: Int)

    /** 쓰기의 끝. */
    private sealed interface Written {
        /** [stop]이 참이 되어 그만뒀다. */
        data object Stopped : Written

        /** 파일 끝까지 쓰고 스트림을 닫았다. */
        data object Done : Written

        /** 도중에 끊겼다: 이 워치에서 실패했거나(출력을 못 엶·파일을 못 읽음) 폰이 먼저 닫았다(이미 있음·오류). */
        data class Broken(val bytes: Long, val reason: String) : Written
    }

    private val capabilities = Wearable.getCapabilityClient(context)
    private val messages = Wearable.getMessageClient(context)
    private val channels = Wearable.getChannelClient(context)
    private val main = Handler(Looper.getMainLooper())
    private val ends = ConcurrentHashMap<ChannelClient.Channel, CompletableFuture<End>>()
    @Volatile private var current: ChannelClient.Channel? = null
    private val callback = object : ChannelClient.ChannelCallback() {
        override fun onChannelClosed(channel: ChannelClient.Channel, closeReason: Int, appSpecificErrorCode: Int) {
            endOf(channel).complete(End(closeReason == ChannelClient.ChannelCallback.CLOSE_REASON_REMOTE_CLOSE, appSpecificErrorCode))
        }
    }

    /** 채널 닫힘 콜백을 단다. 못 달면 false(파일 결과를 받을 수 없으므로 보내지 않는다). */
    fun open(): Boolean = succeeded(channels.registerChannelCallback(callback), TIMEOUT_MS)

    /** 폰 앱이 선언한 capability를 가진, 지금 닿는 노드(가까운 노드 먼저). 없거나 실패하거나 그만두면 null. */
    fun findPhone(): Node? {
        val task = capabilities.getCapability(TransferProtocol.CAPABILITY, CapabilityClient.FILTER_REACHABLE)
        if (!succeeded(task, TIMEOUT_MS)) return null
        return task.result.nodes.sortedByDescending { it.isNearby }.firstOrNull()
    }

    /** 목록 요청을 보내고 폰에 없는 파일 이름을 받는다. 답이 없거나 읽지 못하거나 그만두면 null. */
    fun offer(phone: String, offer: Offer): List<String>? {
        val task = messages.sendRequest(phone, TransferProtocol.OFFER_PATH, TransferProtocol.encodeOffer(offer))
        if (!succeeded(task, TIMEOUT_MS)) return null
        return TransferProtocol.decodeNeed(task.result)
    }

    /**
     * [file]을 채널 하나로 처음부터 끝까지 보내고 폰의 결과 코드를 기다린다. 그사이 [stop]이 참이 되면 채널을 CANCELLED로 닫고 Held.
     * [STALL_MS] 동안 한 바이트도 못 쓰면 STALLED로 닫는다(블루투스가 끊겼는데 쓰기가 막힌 채일 때).
     * 다 쓴 뒤에는 폰의 결과를 [closeWaitMs]까지 기다린다. 도중에 끊겼으면 폰이 먼저 닫은 까닭(이미 있음 등)만 [BROKEN_WAIT_MS] 기다린다:
     * 이 워치에서 난 실패를 폰의 긴 결과 기다림으로 끌지 않는다. [onProgress]는 덩어리마다 지금까지 쓴 바이트다.
     */
    fun sendFile(phone: String, recording: String, file: File, size: Long, onProgress: (Long) -> Unit): FileResult {
        if (stop()) return FileResult.Held
        val path = TransferProtocol.channelPath(recording, file.name, size)
        ends.clear() // 앞 파일들의 늦은 닫힘이 쌓이지 않게. 이 파일의 채널은 아직 없다
        val opening = channels.openChannel(phone, path)
        if (!succeeded(opening, TIMEOUT_MS)) {
            opening.addOnSuccessListener { channels.close(it, TransferProtocol.CODE_CANCELLED) } // 늦게 열려도 바로 닫는다
            return if (stop()) FileResult.Held else FileResult.Failed("채널을 못 엶(${reason(opening)})")
        }
        val channel = opening.result
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
            val written = try {
                write(channel, file, size, lastWrite, onProgress)
            } finally {
                main.removeCallbacks(watchdog)
            }
            if (written == Written.Stopped) return cancel(channel)
            val end = endWithin(channel, if (written == Written.Done) closeWaitMs(size) else BROKEN_WAIT_MS)
            if (end == null && stop()) return cancel(channel)
            return when {
                end != null && end.remote && (end.code == TransferProtocol.CODE_OK || end.code == TransferProtocol.CODE_HAVE) -> FileResult.Done
                end != null && end.remote -> FileResult.Failed("폰: ${TransferProtocol.describe(end.code)}", TransferProtocol.isPermanent(end.code))
                stalled.get() -> FileResult.Failed(TransferProtocol.describe(TransferProtocol.CODE_STALLED))
                written is Written.Broken -> FileResult.Failed("보내다 끊김(${written.reason})")
                end != null -> FileResult.Failed("연결이 끊김")
                else -> FileResult.Failed("폰이 확인하지 않음")
            }
        } finally {
            current = null
            ends.remove(channel)
            channels.close(channel) // 이미 닫혔으면 아무 일도 없다(그 Task의 실패는 버린다)
        }
    }

    /** 보내던 채널을 끊는다(녹음이 시작될 때·서비스가 멈출 때, 주 스레드). 막힌 쓰기가 곧 풀린다. */
    fun abort() {
        current?.let { channels.close(it, TransferProtocol.CODE_CANCELLED) }
    }

    /** 콜백을 뗀다. 달리지 않았거나 이미 뗐어도 해가 없다(그만둔 사이에 늦게 달렸을 수도 있어 늘 부른다). */
    override fun close() {
        channels.unregisterChannelCallback(callback)
    }

    /** 파일을 출력 스트림에 16 KB씩 쓰고 닫는다. 끝까지 썼는지, 그만뒀는지, 몇 바이트에서 왜 끊겼는지를 돌려준다. */
    private fun write(channel: ChannelClient.Channel, file: File, size: Long, lastWrite: AtomicLong, onProgress: (Long) -> Unit): Written {
        val opening = channels.getOutputStream(channel)
        if (!succeeded(opening, TIMEOUT_MS)) return if (stop()) Written.Stopped else Written.Broken(0L, "출력을 못 엶(${reason(opening)})")
        var sent = 0L
        return try {
            opening.result.use { out ->
                FileInputStream(file).use { input ->
                    val buffer = ByteArray(CHUNK_BYTES)
                    while (true) {
                        if (stop()) return Written.Stopped
                        val n = input.read(buffer)
                        if (n < 0) break
                        out.write(buffer, 0, n)
                        sent += n
                        lastWrite.set(SystemClock.elapsedRealtime())
                        onProgress(sent)
                    }
                }
            }
            if (sent == size) Written.Done else Written.Broken(sent, "파일 크기가 바뀜")
        } catch (e: IOException) {
            Written.Broken(sent, e.message ?: e.javaClass.simpleName)
        }
    }

    /** 보내던 채널을 CANCELLED로 닫는다(녹음 시작). */
    private fun cancel(channel: ChannelClient.Channel): FileResult {
        channels.close(channel, TransferProtocol.CODE_CANCELLED)
        return FileResult.Held
    }

    /**
     * [task]가 [timeoutMs] 안에 성공했는가. `Tasks.await`로 [SLICE_MS]씩 기다리며 그 사이마다 [stop]을 본다: 참이 되면 바로 false
     * (기다리던 일은 버린다). 실패·시간 초과도 false다.
     */
    private fun succeeded(task: Task<*>, timeoutMs: Long): Boolean {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (!stop()) {
            val left = deadline - SystemClock.elapsedRealtime()
            if (left <= 0) return false
            if (runCatching { Tasks.await(task, minOf(left, SLICE_MS), TimeUnit.MILLISECONDS) }.isSuccess) return true
            if (task.isComplete) return false
        }
        return false
    }

    /** 폰이 그 채널을 닫은 까닭을 [timeoutMs]까지 [SLICE_MS]씩 기다린다. 시간이 지나거나 [stop]이 참이 되면 null. */
    private fun endWithin(channel: ChannelClient.Channel, timeoutMs: Long): End? {
        val future = endOf(channel)
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (!stop()) {
            val left = deadline - SystemClock.elapsedRealtime()
            if (left <= 0) return null
            runCatching { future.get(minOf(left, SLICE_MS), TimeUnit.MILLISECONDS) }.getOrNull()?.let { return it }
        }
        return null
    }

    private fun endOf(channel: ChannelClient.Channel): CompletableFuture<End> = ends.computeIfAbsent(channel) { CompletableFuture() }

    private fun reason(task: Task<*>): String =
        task.exception?.let { (it.cause ?: it).message ?: it.javaClass.simpleName } ?: "${TIMEOUT_MS / 1_000}초 넘게 답이 없음"

    private companion object {
        const val TIMEOUT_MS = 30_000L
        const val SLICE_MS = 500L
        const val CHUNK_BYTES = 16 * 1024
        const val STALL_MS = 90_000L
        const val WATCHDOG_MS = 10_000L

        /** 쓰기가 도중에 끊긴 뒤 폰이 닫은 까닭을 기다리는 시간. */
        const val BROKEN_WAIT_MS = 10_000L

        /** 다 쓴 뒤 폰의 결과를 기다리는 시간: 120 s + 크기 ÷ 20 KB/s(Play 서비스가 쌓아 둔 것을 천천히 보낼 수 있어서). */
        fun closeWaitMs(size: Long): Long = 120_000L + size / 20
    }
}
