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
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.RecFile
import com.watchpose.handsonar.shared.RetryBook
import com.watchpose.handsonar.shared.TransferRules
import com.watchpose.handsonar.wear.HandSonarApp
import com.watchpose.handsonar.wear.R
import com.watchpose.handsonar.wear.record.RecorderState
import com.watchpose.handsonar.wear.ui.MainActivity
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 녹음을 폰으로 보내는 포그라운드 서비스(전송 설계 §2, 종류 dataSync). [start]가 띄우고, 할 일이 끝나면 스스로 멈춘다.
 * 작업 스레드 하나가 안 보낸 녹음(새것부터)마다: 목록 요청 → 폰에 없는 파일마다 채널 하나 → 목록 요청 한 번 더(빈 답 = 폰의 확인)
 * → "보냄" 기록([HandSonarApp.sentLog]). 녹음 폴더의 파일은 지우지 않는다. 보낼 파일이 없는 폴더는 건너뛴다(적지 않는다).
 * 녹음 중에는 보내지 않는다: 녹음 서비스가 시작할 때 [holdForRecording]을 부르면 지금 채널을 바로 끊고, 쓰기·폰의 답·결과를 기다리는
 * 모든 곳이 0.5 s마다 그 표시를 보므로 곧 멈춘다(녹음마다, "보냄"으로 적기 직전에도 본다).
 * 실패는 녹음마다 센다([RetryBook]): 실패한 녹음은 그 차례에서 건너뛰고 다음 녹음으로 간 뒤, 60 s·120 s·240 s·480 s 뒤에 다시 하고
 * [MAX_TRIES]번이면 이번 실행에서 뺀다(다음 녹음 정지나 화면을 열 때 다시). 다시 보내도 같은 답이 나올 거절(폰이 머리말 모양·이름을 거절, 보낼 수 없는 이름)은
 * 그 녹음만 막는다(앱 프로세스가 사는 동안). 실패 뒤에는 폰이 아직 닿는지 다시 보고, 사라졌으면 그 차례를 그만두고 쉰다.
 * 폰을 못 찾는 차례가 [MAX_PHONE_MISSES]번 이어지면 멈춘다. 막힌 녹음은 화면 한 줄에 이름과 까닭을 남긴다.
 * 부분 깨움은 일하는 동안만 잡는다: 폰 찾기·목록 요청 앞에서 [WAKE_STEP_MS], 파일마다 [WAKE_FILE_MS]로 다시 잡고, 쉬는 동안은 놓는다.
 * 안드로이드 15+의 dataSync 하루 6시간이 다 차면 [onTimeout]에서 바로 멈춘다.
 * 시작 요청이 겹치면 지금 도는 차례가 끝난 뒤 한 번 더 돈다(주 스레드의 busy·again으로만 정한다).
 */
class TransferService : Service() {
    private val main = Handler(Looper.getMainLooper())
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var stopping = false

    /** 작업 스레드만 쓴다. */
    private var wake: PowerManager.WakeLock? = null

    // 아래 셋은 주 스레드에서만 읽고 쓴다
    private var busy = false
    private var again = false
    private var lastStartId = 0

    private sealed interface Sent {
        data object Complete : Sent

        data object Held : Sent

        /** [permanent]면 다시 보내도 같은 결과다(그 녹음을 막는다). */
        data class Failed(val reason: String, val permanent: Boolean = false) : Sent
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

    /** 작업 스레드: [loop]를 돌린 뒤 화면 상태를 마무리하고 깨움을 놓는다. */
    private fun runQueue() {
        val app = application as HandSonarApp
        wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG).apply {
            setReferenceCounted(false)
        }
        awake(WAKE_STEP_MS)
        val link = PhoneSender(this, ::stop)
        activeSender = link
        try {
            finishRun(app, if (link.open()) loop(app, link) else "데이터 레이어를 열지 못함 — 다음에 다시 보냄")
        } catch (e: Exception) {
            finishRun(app, "보내기 오류: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            activeSender = null
            link.close()
            asleep()
            wake = null
        }
    }

    /** 보낼 녹음이 없거나, 녹음이 시작되거나, 남은 녹음이 모두 막히거나, 폰을 계속 못 찾을 때까지 돈다. 화면에 남길 글(없으면 null)을 돌려준다. */
    private fun loop(app: HandSonarApp, link: PhoneSender): String? {
        val book = RetryBook(MAX_TRIES, RETRY_MS, MAX_BACKOFF_MS)
        var phoneMisses = 0
        var sent = 0
        while (!stopping) {
            if (held()) return HELD_MESSAGE
            val pending = pendingOf(app, liveRecording())
            if (book.tryable(pending).isEmpty()) return endMessage(sent, book)
            val now = SystemClock.elapsedRealtime()
            val due = book.due(pending, now)
            if (due.isEmpty()) { // 남은 녹음이 모두 쉬는 중: 가장 이른 때까지 깨움 없이 쉰다
                val next = book.nextAt(pending) ?: now
                TransferState.update { it.copy(file = "", bytes = 0L, total = 0L, message = "다시 보낼 녹음 기다림 — ${(next - now + 999) / 1_000}초 뒤") }
                asleep()
                if (!rest(next - now)) return if (held()) HELD_MESSAGE else null
                continue
            }
            TransferState.update { it.copy(active = true, pending = pending.size, message = null) }
            awake(WAKE_STEP_MS)
            val phone = link.findPhone()
            if (held()) return HELD_MESSAGE
            if (phone == null) {
                phoneMisses++
                val why = "폰을 찾지 못함(블루투스 연결과 폰 앱을 확인)"
                if (phoneMisses >= MAX_PHONE_MISSES) return "$why — 다음 녹음 뒤나 앱을 열 때 다시 보냄"
                TransferState.update { it.copy(file = "", bytes = 0L, total = 0L, message = "$why — ${RETRY_MS / 1_000}초 뒤 다시") }
                asleep()
                if (!rest(RETRY_MS)) return if (held()) HELD_MESSAGE else null
                continue
            }
            phoneMisses = 0
            var phoneId = phone.id
            var checkPhone = false
            for (name in due) {
                if (held()) return HELD_MESSAGE
                if (checkPhone) { // 앞 녹음이 실패했다: 폰이 사라졌으면 이 차례를 그만두고 쉰다(남은 녹음마다 30 s씩 묻지 않게)
                    awake(WAKE_STEP_MS)
                    phoneId = link.findPhone()?.id ?: break
                    if (held()) return HELD_MESSAGE
                    checkPhone = false
                }
                awake(WAKE_STEP_MS)
                when (val r = sendRecording(link, phoneId, File(app.recordingsRoot, name))) {
                    Sent.Complete -> {
                        if (held()) return HELD_MESSAGE // 적기 직전에도 본다: 그사이 시작된 녹음 폴더를 "보냄"으로 적지 않게
                        app.sentLog.add(name)
                        sent++
                        book.succeeded(name)
                        TransferState.update { it.copy(pending = (it.pending - 1).coerceAtLeast(0), file = "", bytes = 0L, total = 0L) }
                    }
                    Sent.Held -> return HELD_MESSAGE
                    is Sent.Failed -> {
                        if (r.permanent) {
                            blocked[name] = r.reason
                        } else {
                            book.failed(name, r.reason, SystemClock.elapsedRealtime())
                            checkPhone = true
                        }
                        TransferState.update { it.copy(file = "", bytes = 0L, total = 0L, message = "$name ${r.reason}") }
                    }
                }
            }
        }
        return null
    }

    /** 녹음 하나: 목록 요청 → 폰에 없는 파일만 보냄 → 목록 요청 한 번 더. 두 번째 답이 비어야 Complete(폰의 확인)다. */
    private fun sendRecording(link: PhoneSender, phone: String, dir: File): Sent {
        if (held()) return Sent.Held
        val all = (dir.listFiles() ?: emptyArray()).filter { it.isFile }.map { RecFile(it.name, it.length()) }
        val unsafe = TransferRules.unsendable(all)
        if (unsafe.isNotEmpty()) return Sent.Failed("보낼 수 없는 파일 이름 ${unsafe.joinToString(", ")}", permanent = true)
        val files = TransferRules.transferable(all)
        if (files.isEmpty()) return Sent.Failed("보낼 파일이 없음") // 고를 때는 있었다(드묾): 다음 차례에 다시 본다
        val offer = Offer(dir.name, files)
        val need = link.offer(phone, offer) ?: return if (held()) Sent.Held else Sent.Failed("폰이 답하지 않음")
        if (need.isEmpty()) return Sent.Complete
        for (f in files) {
            if (f.name !in need) continue
            if (held()) return Sent.Held
            awake(WAKE_FILE_MS)
            TransferState.update { it.copy(file = f.name, bytes = 0L, total = f.size) }
            when (val r = link.sendFile(phone, dir.name, File(dir, f.name), f.size) { b -> TransferState.update { it.copy(bytes = b) } }) {
                FileResult.Done -> Unit
                FileResult.Held -> return Sent.Held
                is FileResult.Failed -> return Sent.Failed("${f.name}: ${r.reason}", r.permanent)
            }
        }
        awake(WAKE_STEP_MS)
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

    /** 이번 실행의 끝 글: 막힌 녹음(다시 해도 같은 거절, 또는 이번 실행의 시도를 다 씀)이 있으면 그 이름과 까닭, 아니면 보낸 수. */
    private fun endMessage(sent: Int, book: RetryBook): String? {
        val stuck = book.exhausted().map { (name, reason) -> "$name $reason" } +
            blocked.map { (name, reason) -> "$name $reason(다시 보내지 않음)" }
        return when {
            stuck.isNotEmpty() -> "막힌 녹음 ${stuck.size}개: ${stuck.first()}" + (if (stuck.size > 1) " 외" else "") +
                (if (sent > 0) " · 보낸 녹음 ${sent}개" else "")
            sent > 0 -> "폰으로 다 보냄(녹음 ${sent}개)"
            else -> null
        }
    }

    private fun held(): Boolean = hold.get() || RecorderState.status.value.running

    /** 데이터 레이어 기다림을 그만둘 때: 녹음 중이거나 서비스가 멈추는 중. */
    private fun stop(): Boolean = held() || stopping

    private fun liveRecording(): String? = RecorderState.status.value.let { if (it.running) it.folder else null }

    /** 부분 깨움을 [ms] 동안 잡는다(잡혀 있으면 그때부터 다시 센다). */
    private fun awake(ms: Long) {
        wake?.acquire(ms)
    }

    private fun asleep() {
        wake?.let { if (it.isHeld) it.release() }
    }

    /** 한 차례를 마치며 화면 상태를 정리한다. 서비스가 멈추는 중이면(onTimeout 등) 그쪽이 남긴 글을 지우지 않는다. */
    private fun finishRun(app: HandSonarApp, message: String?) {
        val pending = pendingOf(app, liveRecording()).size
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

        /** 폰 찾기·목록 요청 한 번(30 s씩 기다림)을 넉넉히 덮는 깨움. */
        private const val WAKE_STEP_MS = 2 * 60 * 1000L

        /** 파일 하나(1분 조각 5.76 MB를 10 KB/s로 쓰고 폰의 결과를 기다리는 시간)를 덮는 깨움. 파일마다 다시 잡는다. */
        private const val WAKE_FILE_MS = 30 * 60 * 1000L
        private const val RETRY_MS = 60_000L
        private const val MAX_BACKOFF_MS = 8 * 60_000L
        private const val MAX_TRIES = 5
        private const val MAX_PHONE_MISSES = 5
        private const val HELD_MESSAGE = "녹음 중이라 보내기를 멈춤 — 녹음이 끝나면 다시 보냄"
        private val hold = AtomicBoolean(false)

        /** 다시 보내도 같은 거절이 난 녹음과 그 까닭. 앱 프로세스가 사는 동안 보내지 않는다(다시 뜨면 한 번 더 해 본다). */
        private val blocked = ConcurrentHashMap<String, String>()

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
            val pending = pendingOf(app, live = null).size
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

        /** 보낼 녹음(새것부터): 보낼 파일이 하나라도 있는 녹음 폴더 중 아직 "보냄"이 아니고, 녹음 중이 아니고, 막히지 않은 것. */
        private fun pendingOf(app: HandSonarApp, live: String?): List<String> =
            TransferRules.pending(folderNames(app), app.sentLog.names(), live).filter { !blocked.containsKey(it) }

        /** 일반 파일이 하나라도 있는 녹음 폴더(빈 폴더 — 막 만들어 아직 쓰기 전인 녹음 — 는 보낼 것이 없다). */
        private fun folderNames(app: HandSonarApp): List<String> =
            app.recordingsRoot.listFiles()
                ?.filter { dir -> dir.isDirectory && dir.listFiles()?.any { it.isFile && !TransferRules.isLeftover(it.name) } == true }
                ?.map { it.name }
                .orEmpty()

    }
}
