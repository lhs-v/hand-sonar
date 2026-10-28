package com.watchpose.handsonar.wear.transfer

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.RecFile
import com.watchpose.handsonar.shared.TransferRules
import com.watchpose.handsonar.wear.HandSonarApp
import com.watchpose.handsonar.wear.R
import com.watchpose.handsonar.wear.record.RecorderState
import com.watchpose.handsonar.wear.ui.MainActivity
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 녹음을 폰으로 보내는 포그라운드 서비스(전송 설계 §2, 종류 dataSync). [start]가 띄우고, 할 일이 끝나면 스스로 멈춘다.
 * 작업 스레드 하나가 안 보낸 녹음(오래된 것부터)마다: 목록 요청 → 폰에 없는 파일마다 채널 하나 → 목록 요청 한 번 더(빈 답 = 폰의 확인)
 * → "보냄" 기록([HandSonarApp.sentLog]). 녹음 폴더의 파일은 지우지 않는다.
 * 녹음 중에는 보내지 않는다: 녹음 서비스가 시작할 때 [holdForRecording]을 부르면 지금 채널을 바로 끊고, 쓰기·폰의 답·결과를 기다리는
 * 모든 곳이 0.5 s마다 그 표시를 보므로 곧 멈춘다(녹음마다, "보냄"으로 적기 직전에도 본다).
 * 실패(폰을 못 찾음·답 없음·파일 실패)가 이어지면 [RETRY_MS] 쉬고 [MAX_FAILED_ROUNDS]번까지 다시 한다. 그 뒤는 다음 녹음 정지나 화면을 열 때다.
 * 화면이 꺼져도 끝까지 가도록 부분 깨움을 잡는다. 안드로이드 15+의 dataSync 하루 6시간이 다 차면 [onTimeout]에서 바로 멈춘다.
 * 시작 요청이 겹치면 지금 도는 차례가 끝난 뒤 한 번 더 돈다(주 스레드의 busy·again으로만 정한다).
 */
class TransferService : Service() {
    private val main = Handler(Looper.getMainLooper())
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var stopping = false

    // 아래 셋은 주 스레드에서만 읽고 쓴다
    private var busy = false
    private var again = false
    private var lastStartId = 0

    private sealed interface Sent {
        data object Complete : Sent

        data object Held : Sent

        data class Failed(val reason: String) : Sent
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        lastStartId = startId
        try {
            startInForeground() // startForegroundService 뒤 몇 초 안에 해야 하므로 맨 먼저 한다
        } catch (e: Exception) {
            TransferState.update { it.copy(active = false, message = "보내기 시작 실패(${e.javaClass.simpleName})") }
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (busy) again = true else launch()
        return START_NOT_STICKY
    }

    /** 안드로이드 15+: dataSync의 하루 몫(6시간)을 다 썼다. 몇 초 안에 멈춰야 한다. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        stopping = true
        TransferState.update { it.copy(message = "오늘 보내기 시간을 다 써서 멈춤 — 다음에 다시 보냄") }
        stopSelf()
    }

    override fun onDestroy() {
        stopping = true
        activeSender?.abort()
        executor.shutdownNow()
        TransferState.update { it.copy(active = false, file = "", bytes = 0L, total = 0L) }
        super.onDestroy()
    }

    private fun launch() {
        if (stopping) return
        busy = true
        executor.execute {
            runQueue()
            main.post(::afterRun)
        }
    }

    private fun afterRun() {
        if (again && !stopping) {
            again = false
            launch()
        } else {
            busy = false
            stopSelf(lastStartId)
        }
    }

    /** 작업 스레드: 깨움을 잡고 [loop]를 돌린 뒤 화면 상태를 마무리한다. */
    private fun runQueue() {
        val app = application as HandSonarApp
        val wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG).apply {
            setReferenceCounted(false)
            acquire(WAKE_LOCK_TIMEOUT_MS)
        }
        val link = PhoneSender(this, ::stop)
        activeSender = link
        try {
            finishRun(app, if (link.open()) loop(app, link) else "데이터 레이어를 열지 못함 — 다음에 다시 보냄")
        } catch (e: Exception) {
            finishRun(app, "보내기 오류: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            activeSender = null
            link.close()
            if (wake.isHeld) wake.release()
        }
    }

    /** 보낼 녹음이 없거나, 녹음이 시작되거나, 실패가 [MAX_FAILED_ROUNDS]번 이어질 때까지 돈다. 화면에 남길 글(없으면 null)을 돌려준다. */
    private fun loop(app: HandSonarApp, link: PhoneSender): String? {
        var failedRounds = 0
        var sent = 0
        while (!stopping) {
            if (held()) return HELD_MESSAGE
            val pending = TransferRules.pending(folderNames(app), app.sentLog.names(), liveRecording())
            if (pending.isEmpty()) return if (sent > 0) "폰으로 다 보냄(녹음 ${sent}개)" else null
            TransferState.update { it.copy(active = true, pending = pending.size, message = null) }
            val phone = link.findPhone()
            if (held()) return HELD_MESSAGE
            var failure: String? = if (phone == null) "폰을 찾지 못함(블루투스 연결과 폰 앱을 확인)" else null
            if (phone != null) {
                for (name in pending) {
                    when (val r = sendRecording(link, phone.id, File(app.recordingsRoot, name))) {
                        Sent.Complete -> {
                            if (held()) return HELD_MESSAGE // 적기 직전에도 본다: 그사이 시작된 녹음 폴더를 "보냄"으로 적지 않게
                            app.sentLog.add(name)
                            sent++
                            failedRounds = 0
                            TransferState.update { it.copy(pending = (it.pending - 1).coerceAtLeast(0), file = "", bytes = 0L, total = 0L) }
                        }
                        Sent.Held -> return HELD_MESSAGE
                        is Sent.Failed -> {
                            failure = "$name ${r.reason}"
                            break
                        }
                    }
                }
            }
            if (failure == null) continue
            failedRounds++
            if (failedRounds >= MAX_FAILED_ROUNDS) return "$failure — 다음 녹음 뒤나 앱을 열 때 다시 보냄"
            TransferState.update { it.copy(file = "", bytes = 0L, total = 0L, message = "$failure — ${RETRY_MS / 1_000}초 뒤 다시") }
            if (!rest(RETRY_MS)) return if (held()) HELD_MESSAGE else null
        }
        return null
    }

    /** 녹음 하나: 목록 요청 → 폰에 없는 파일만 보냄 → 목록 요청 한 번 더. 두 번째 답이 비어야 Complete(폰의 확인)다. */
    private fun sendRecording(link: PhoneSender, phone: String, dir: File): Sent {
        if (held()) return Sent.Held
        val files = TransferRules.transferable((dir.listFiles() ?: emptyArray()).filter { it.isFile }.map { RecFile(it.name, it.length()) })
        if (files.isEmpty()) return Sent.Complete // 보낼 파일이 없는 폴더(시작하다 만 녹음): 폰에 묻지 않고 끝낸다
        val offer = Offer(dir.name, files)
        val need = link.offer(phone, offer) ?: return if (held()) Sent.Held else Sent.Failed("폰이 답하지 않음")
        if (need.isEmpty()) return Sent.Complete
        for (f in files) {
            if (f.name !in need) continue
            if (held()) return Sent.Held
            TransferState.update { it.copy(file = f.name, bytes = 0L, total = f.size) }
            when (val r = link.sendFile(phone, dir.name, File(dir, f.name), f.size) { b -> TransferState.update { it.copy(bytes = b) } }) {
                FileResult.Done -> Unit
                FileResult.Held -> return Sent.Held
                is FileResult.Failed -> return Sent.Failed("${f.name}: ${r.reason}")
            }
        }
        val left = link.offer(phone, offer) ?: return if (held()) Sent.Held else Sent.Failed("폰이 답하지 않음")
        return if (left.isEmpty()) Sent.Complete else Sent.Failed("폰에 아직 없는 파일 ${left.size}개")
    }

    /** [ms] 동안 1초씩 쉰다. 그사이 녹음이 시작되거나 서비스가 멈추면 false. */
    private fun rest(ms: Long): Boolean {
        var left = ms
        while (left > 0) {
            if (stopping || held()) return false
            if (runCatching { Thread.sleep(minOf(left, 1_000L)) }.isFailure) return false
            left -= 1_000L
        }
        return !stopping && !held()
    }

    private fun held(): Boolean = hold.get() || RecorderState.status.value.running

    /** 데이터 레이어 기다림을 그만둘 때: 녹음 중이거나 서비스가 멈추는 중. */
    private fun stop(): Boolean = held() || stopping

    private fun liveRecording(): String? = RecorderState.status.value.let { if (it.running) it.folder else null }

    /** 한 차례를 마치며 화면 상태를 정리한다. 서비스가 멈추는 중이면(onTimeout 등) 그쪽이 남긴 글을 지우지 않는다. */
    private fun finishRun(app: HandSonarApp, message: String?) {
        val pending = TransferRules.pending(folderNames(app), app.sentLog.names(), liveRecording()).size
        TransferState.update {
            it.copy(active = false, pending = pending, file = "", bytes = 0L, total = 0L, message = if (stopping) it.message else message)
        }
    }

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "폰으로 보내기", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_sonar)
            .setContentTitle("손 소나")
            .setContentText("녹음을 폰으로 보내는 중")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setContentIntent(open)
            .build()
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
    }

    companion object {
        private const val CHANNEL_ID = "transfer"
        private const val NOTIFICATION_ID = 2 // 녹음 서비스는 1
        private const val WAKE_TAG = "handsonar:transfer"
        private const val WAKE_LOCK_TIMEOUT_MS = 2 * 60 * 60 * 1000L
        private const val RETRY_MS = 60_000L
        private const val MAX_FAILED_ROUNDS = 5
        private const val HELD_MESSAGE = "녹음 중이라 보내기를 멈춤 — 녹음이 끝나면 다시 보냄"
        private val hold = AtomicBoolean(false)

        /** 지금 도는 보내기 작업의 데이터 레이어 호출. [holdForRecording]이 그 채널을 바로 끊는다. */
        @Volatile private var activeSender: PhoneSender? = null

        /**
         * 녹음 서비스가 시작할 때 맨 먼저 부른다(주 스레드): 보내던 채널을 바로 끊고 서비스를 멈추게 한다(녹음 중에는 보내지 않는다).
         * 막힌 쓰기도 채널이 닫히면 풀린다. 다음 [start]가 표시를 푼다.
         */
        fun holdForRecording() {
            hold.set(true)
            activeSender?.abort()
        }

        /**
         * 아직 폰이 확인하지 않은 녹음이 있으면 서비스를 띄운다(녹음이 멈출 때, 녹음 시작이 실패할 때, 워치 화면이 보일 때). 띄웠으면 true.
         * 녹음 서비스는 `running`을 내리기 전에 부르므로 여기서는 지금 녹음 중인 폴더를 빼지 않는다(작업 스레드가 뺀다).
         * 앱이 보이지 않고 포그라운드 서비스도 없으면 안드로이드 12+가 막는다(ForegroundServiceStartNotAllowedException):
         * 화면에 알리고 다음 기회에 한다. 녹음 서비스는 포그라운드를 내려놓기 전에 부르므로 막히지 않는다.
         */
        fun start(context: Context): Boolean {
            hold.set(false)
            val app = context.applicationContext as HandSonarApp
            val pending = TransferRules.pending(folderNames(app), app.sentLog.names(), live = null).size
            TransferState.update { it.copy(pending = pending) }
            if (pending == 0) return false
            return try {
                ContextCompat.startForegroundService(context, Intent(context, TransferService::class.java))
                true
            } catch (e: Exception) {
                TransferState.update { it.copy(message = "보내기를 시작하지 못함(${e.javaClass.simpleName}) — 워치 앱을 열면 다시 보냄") }
                false
            }
        }

        private fun folderNames(app: HandSonarApp): List<String> =
            app.recordingsRoot.listFiles()?.filter { it.isDirectory }?.map { it.name }.orEmpty()
    }
}
