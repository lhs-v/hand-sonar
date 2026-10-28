package com.watchpose.handsonar.wear.record

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import androidx.wear.ongoing.OngoingActivity
import androidx.wear.ongoing.Status
import com.watchpose.handsonar.shared.ClockPoint
import com.watchpose.handsonar.shared.PartWriter
import com.watchpose.handsonar.shared.RecordingLog
import com.watchpose.handsonar.shared.SensorPlan
import com.watchpose.handsonar.shared.SonarSpec
import com.watchpose.handsonar.shared.StopReason
import com.watchpose.handsonar.shared.StorageBudget
import com.watchpose.handsonar.shared.Sweep
import com.watchpose.handsonar.wear.BuildConfig
import com.watchpose.handsonar.wear.HandSonarApp
import com.watchpose.handsonar.wear.Haptics
import com.watchpose.handsonar.wear.R
import com.watchpose.handsonar.wear.transfer.TransferService
import com.watchpose.handsonar.wear.ui.MainActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * 녹음 하나의 수명을 맡는 포그라운드 서비스(설계 §5). 시작하면 녹음을 먼저 열고(처프의 시작부터 담기게) 재생을 시작한다.
 * 1초마다 재생 시각·재생 끊김·저장 공간·화면 상태를 고치고, 10초마다 rec.json을 다시 쓴다(앱이 죽어도 기록이 남게).
 * 정지·오류·저장 공간 부족이면 녹음과 재생을 멈추고 마지막 조각과 rec.json을 닫는다. 오디오 스레드의 오류는 주 스레드로 넘겨 끝낸다.
 * 자동 정지(설계 §5.1): 시작 인텐트의 분(없으면 설정의 분)이 지나면 틱이 `timer`로 같은 정지 경로를 한 번 타고, 오류가 아니라서 정지 진동이 난다(0이면 하지 않는다).
 * 미디어 볼륨이 0이면 시작하지 않는다(설계 §5.4): 처프가 나가지 않아 녹음 전체가 쓸모없고, 워치 미디어 볼륨은 adb로 바꿀 수 없어 다시 일어나기 쉽다.
 * 녹음 볼륨(전송 설계 §8, 계획 6): 설정에서 단계를 골랐으면 폴더를 만들기 전에 미디어 볼륨을 그 단계로 맞추고([RecordVolumeKeeper], 원래 값을
 * 먼저 적어 둠) 볼륨 0 막기는 맞춘 뒤의 값으로 한다. 멈출 때(재생이 멈춘 뒤)와 시작 실패 때 되돌린다. 고른 값·실제 값·원래 값·되돌린 값과
 * 녹음 중 바뀐 때를 rec.json `volume`에 적는다.
 * 폰으로 보내기(전송 설계 §2): 시작할 때 맨 먼저 보내기를 멈추게 하고(녹음 중에는 보내지 않는다), 멈출 때(이유와 상관없이)와
 * 시작하다 실패할 때 포그라운드를 내려놓기 전에 [TransferService]를 띄운다.
 * 센서 기록(전송 설계 §8, 계획 6): 녹음·재생을 다 연 뒤에 [SensorRecorder]를 시작하고(소리 녹음을 방해하지 않게), 멈출 때 정지 시각을 잰 뒤
 * 마지막 rec.json과 보내기 전에 닫는다(보내기는 다 닫힌 파일만 옮긴다). 센서 기록이 실패해도 녹음은 이어 간다.
 * 원격 시작에 `sensors` false를 주면 그 녹음만 센서를 듣지 않는다(센서 기록이 소리에 주는 영향을 견줄 때).
 */
class RecorderService : LifecycleService() {
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var recorder: SonarRecorder? = null
    @Volatile private var player: SonarPlayer? = null
    @Volatile private var writer: PartWriter? = null
    @Volatile private var sensors: SensorRecorder? = null
    @Volatile private var log: RecordingLog? = null
    @Volatile private var folder: File? = null
    @Volatile private var finished = false

    /** 자동 정지 시각(부팅 후 ms, [SystemClock.elapsedRealtime] 기준). 0이면 없다. 시작 때 한 번 정하고 틱이 읽는다. */
    @Volatile private var autoStopAtMs = 0L

    private var ticker: Job? = null
    private val haptics by lazy { Haptics(this) }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == ACTION_STOP) {
            finish(StopReason.USER, null)
        } else if (!RecorderState.status.value.running) {
            begin(intent?.getIntExtra(EXTRA_AUTO_STOP_MINUTES, -1)?.takeIf { it >= 0 }, intent?.getBooleanExtra(EXTRA_SENSORS, true) ?: true)
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (!finished && recorder != null) finish(StopReason.ERROR, "서비스가 끝남")
        super.onDestroy()
    }

    /**
     * [autoStopOverride]는 이번 녹음만 설정 대신 쓰는 자동 정지 분이다(원격 시작이 넘긴다). null이면 설정을 따른다.
     * [withSensors]가 false면 이번 녹음은 센서를 듣지 않는다(원격 시작 `sensors` false).
     */
    private fun begin(autoStopOverride: Int?, withSensors: Boolean) {
        TransferService.holdForRecording() // 녹음 중에는 보내지 않는다(전송 설계 §2): 보내던 파일을 바로 끊게 한다
        finished = false
        sensors = null // 멈춘 직후 다시 시작하면 서비스 객체가 그대로 쓰인다: 앞 녹음의 센서 보고가 새 rec.json에 들지 않게
        try {
            startInForeground() // startForegroundService 뒤 몇 초 안에 해야 하므로 맨 먼저 한다
        } catch (e: Exception) {
            RecorderState.update { it.copy(running = false, error = "포그라운드 시작 실패: ${e.message ?: e.javaClass.simpleName}") }
            // 녹음은 열리지 않았다: 맨 앞에서 멈춘 보내기를 다시 잇는다. 백그라운드라 서비스를 못 띄워도 멈춤 표시는 풀린다
            TransferService.start(this)
            stopSelf()
            return
        }
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "handsonar:record")
            .apply {
                setReferenceCounted(false)
                acquire(WAKE_LOCK_TIMEOUT_MS)
            }
        val root = (application as HandSonarApp).recordingsRoot.apply { mkdirs() }
        val free = root.usableSpace
        if (StorageBudget.level(free) == StorageBudget.Level.STOP) {
            failToStart("저장 공간이 부족합니다(남은 녹음 ${String.format(Locale.ROOT, "%.1f", StorageBudget.minutesLeft(free))}분)")
            return
        }
        val settings = (application as HandSonarApp).settings.state.value
        // 녹음 볼륨(전송 설계 §8): 앞 녹음에서 앱이 죽어 못 되돌린 볼륨을 먼저 되돌린 뒤 지금 볼륨을 원래 값으로 읽고, 고른 단계로 맞춘다.
        val am = getSystemService(AudioManager::class.java)
        val keeper = (application as HandSonarApp).volumeKeeper
        keeper.restore()
        val originalVolume = am.getStreamVolume(AudioManager.STREAM_MUSIC)
        val volumeMax = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val targetVolume = SonarSpec.recordVolumeTarget(settings.recordVolume, originalVolume, volumeMax)
        val volume = keeper.apply(targetVolume, originalVolume)
        // 미디어 볼륨 0이면 처프가 녹음되지 않아 그 녹음 전체를 못 쓴다. 폴더를 만들기 전에 막는다(원격 시작도 서비스에서 막히므로 같은 길을 탄다).
        if (volume == 0) {
            failToStart(
                if (settings.recordVolume == 0) {
                    "미디어 볼륨이 0이라 처프가 나가지 않습니다. 워치에서 미디어 볼륨을 올린 뒤 다시 시작하세요"
                } else {
                    "미디어 볼륨을 고른 단계 ${targetVolume}에 맞추지 못했습니다(지금 0). 워치에서 미디어 볼륨을 올린 뒤 다시 시작하세요"
                },
            )
            return
        }
        val dir = File(root, LocalDateTime.now().format(STAMP)).apply { mkdirs() }
        folder = dir
        val parts = PartWriter(dir)
        writer = parts
        val rec = SonarRecorder(this, parts, onClock = ::onRecordClock, onError = ::onAudioError)
        val startedAt = ClockPoint(System.currentTimeMillis(), SystemClock.elapsedRealtimeNanos())
        log = RecordingLog(
            app = BuildConfig.VERSION_NAME,
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            android = "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
            start = startedAt,
            source = rec.sourceName,
            window = settings.window,
            amplitude = settings.amplitude,
            volumeStream = "music",
            volumeIndex = volume,
            volumeMax = volumeMax,
            volumeChoice = settings.recordVolume.takeIf { it > 0 },
            volumeOriginal = originalVolume,
        )
        recorder = rec
        rec.start()?.let {
            failToStart(it)
            return
        }
        val play = SonarPlayer(this, Sweep.pcm16(settings.window, settings.amplitude), onError = ::onAudioError)
        player = play
        play.start()?.let {
            finish(StopReason.ERROR, it)
            return
        }
        // 센서는 소리가 다 열린 뒤에 듣는다(소리 녹음을 방해하지 않게). 못 열어도 녹음은 이어 가고 rec.json·화면에 남긴다.
        if (withSensors) sensors = SensorRecorder(this, dir, SensorPlan.WATCH_BASE, SensorPlan.WATCH).also { it.start() }
        writeLog()
        val autoStopMinutes = autoStopOverride ?: settings.autoStopMinutes
        autoStopAtMs = if (autoStopMinutes > 0) startedAt.elapsedNs / 1_000_000 + autoStopMinutes * 60_000L else 0L
        RecorderState.update {
            RecorderStatus(
                running = true,
                startedElapsedMs = startedAt.elapsedNs / 1_000_000,
                autoStopAtElapsedMs = autoStopAtMs,
                minutesLeft = StorageBudget.minutesLeft(free),
                storageWarning = StorageBudget.level(free) == StorageBudget.Level.WARN,
                volumeIndex = volume,
                volumeMax = volumeMax,
                volumeRestoreTo = keeper.pending,
                folder = dir.name,
                sensorNote = sensors?.note(),
            )
        }
        haptics.start()
        ticker = lifecycleScope.launch(Dispatchers.Default) { tickLoop(volume) }
    }

    /** [startVolume]은 녹음을 시작할 때의 미디어 볼륨이다. 그 뒤 바뀌면 rec.json `volume.changes`에 적는다. */
    private suspend fun tickLoop(startVolume: Int) {
        var lastUnderruns = 0
        var lastVolume = startVolume
        var ticks = 0
        while (currentCoroutineContext().isActive && !finished) {
            delay(TICK_MS)
            val stopAt = autoStopAtMs
            if (stopAt > 0 && SystemClock.elapsedRealtime() >= stopAt) {
                withContext(Dispatchers.Main) { finish(StopReason.TIMER, null) }
                return
            }
            val rec = recorder ?: return
            val play = player ?: return
            val l = log ?: return
            val dir = folder ?: return
            play.clock()?.let { l.addPlayClock(it[0], it[1]) }
            val underruns = play.underruns
            if (underruns != lastUnderruns) {
                l.addUnderrun(SystemClock.elapsedRealtimeNanos(), underruns)
                lastUnderruns = underruns
            }
            val free = dir.usableSpace
            val level = StorageBudget.level(free)
            if (level == StorageBudget.Level.STOP) {
                withContext(Dispatchers.Main) { finish(StopReason.STORAGE, "저장 공간이 부족해 멈췄습니다") }
                return
            }
            val am = getSystemService(AudioManager::class.java)
            val volume = am.getStreamVolume(AudioManager.STREAM_MUSIC)
            if (volume != lastVolume) { // 녹음 중 사람이 볼륨을 바꿨다: 볼륨별로 견줄 때 가려낼 수 있게 남긴다
                l.addVolumeChange(SystemClock.elapsedRealtimeNanos(), volume)
                lastVolume = volume
            }
            RecorderState.update {
                it.copy(
                    recordedFrames = rec.frames,
                    underruns = underruns,
                    minutesLeft = StorageBudget.minutesLeft(free),
                    storageWarning = level == StorageBudget.Level.WARN,
                    volumeIndex = volume,
                    volumeMax = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC),
                    sensorNote = sensors?.note(),
                )
            }
            ticks++
            if (ticks % LOG_EVERY_TICKS == 0) writeLog()
        }
    }

    private fun onRecordClock(frame: Long, elapsedNs: Long, framesRead: Long, fromTimestamp: Boolean) {
        val l = log ?: return
        l.addRecordClock(frame, elapsedNs, framesRead)
        if (!fromTimestamp) l.useReadTimeClock()
    }

    /** 오디오 스레드에서 불린다. 정리는 주 스레드에서 한다. */
    private fun onAudioError(message: String) {
        lifecycleScope.launch(Dispatchers.Main) { finish(StopReason.ERROR, message) }
    }

    /** rec.json을 임시 파일에 쓰고 이름을 바꿔 덮는다(쓰는 도중에 죽어도 앞 판이 남게). */
    @Synchronized
    private fun writeLog() {
        val dir = folder ?: return
        val l = log ?: return
        writer?.let { l.setParts(it.parts) }
        sensors?.let { l.setSensors(it.report()) }
        val json = l.toJson()
        val tmp = File(dir, "rec.json.tmp")
        tmp.writeText(json)
        if (!tmp.renameTo(File(dir, "rec.json"))) {
            File(dir, "rec.json").writeText(json)
            tmp.delete()
        }
    }

    private fun finish(reason: StopReason, error: String?) {
        if (finished) return
        finished = true
        ticker?.cancel()
        ticker = null
        recorder?.stop()
        player?.stop()
        writer?.close()
        val stoppedAt = ClockPoint(System.currentTimeMillis(), SystemClock.elapsedRealtimeNanos()) // 소리가 멈춘 때(센서를 닫기 전)
        sensors?.stop() // 마지막 rec.json과 보내기 전에 CSV를 닫는다
        val restored = (application as HandSonarApp).volumeKeeper.restore() // 재생이 멈춘 뒤 녹음 볼륨을 원래대로
        log?.let {
            it.finish(stoppedAt, reason, writer?.parts.orEmpty())
            it.setVolumeRestored(restored)
        }
        writeLog()
        // 진동은 포그라운드를 내려놓기 전에 낸다: 내려놓은 뒤의 백그라운드 진동은 무시될 수 있다(화면이 꺼진 자동 정지 끝).
        if (error == null) haptics.stop() else haptics.error()
        // 멈춘 녹음을 폰으로 보낸다(전송 설계 §2). 포그라운드 서비스가 도는 동안이라 백그라운드 시작 제한에 걸리지 않는다.
        TransferService.start(this)
        releaseAndStop(error)
    }

    /** 녹음을 열지 못했을 때: 남길 것이 없으므로 빈 폴더를 지우고 멈춘다. */
    private fun failToStart(message: String) {
        finished = true
        recorder?.stop()
        writer?.close()
        sensors?.stop() // 센서는 재생까지 연 뒤에 시작하므로 보통 없다
        (application as HandSonarApp).volumeKeeper.restore() // 녹음 볼륨으로 바꿨으면 되돌린다(바꾸기 전 실패면 할 일 없음)
        folder?.let { dir -> if (dir.listFiles().isNullOrEmpty()) dir.delete() }
        haptics.error() // finish()와 같은 이유로 포그라운드를 내려놓기 전에 낸다
        TransferService.start(this) // 녹음 시작이 끊은 보내기를 다시 잇는다(같은 까닭으로 내려놓기 전에)
        releaseAndStop(message)
    }

    private fun releaseAndStop(error: String?) {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        RecorderState.update { it.copy(running = false, error = error) }
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "소나 녹음", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_sonar)
            .setContentTitle("손 소나")
            .setContentText("처프 재생·녹음 중")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_WORKOUT)
            .setContentIntent(open)
        OngoingActivity.Builder(applicationContext, NOTIFICATION_ID, builder)
            .setStaticIcon(R.drawable.ic_stat_sonar)
            .setTouchIntent(open)
            .setStatus(Status.Builder().addTemplate("녹음 중").build())
            .build()
            .apply(applicationContext)
        ServiceCompat.startForeground(this, NOTIFICATION_ID, builder.build(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
    }

    companion object {
        const val ACTION_STOP = "com.watchpose.handsonar.action.STOP"

        /** 시작 인텐트에 넣는 자동 정지 분(Int, 0 = 끔). 있으면 이번 녹음만 설정 대신 쓴다. 원격 시작(adb)이 넘긴다. */
        const val EXTRA_AUTO_STOP_MINUTES = "autoStopMinutes"

        /** 시작 인텐트의 센서 기록 여부(Boolean, 없으면 켬). false면 이번 녹음만 센서를 듣지 않는다. 원격 시작(adb)이 넘긴다. */
        const val EXTRA_SENSORS = "sensors"
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 1
        private const val TICK_MS = 1_000L
        private const val LOG_EVERY_TICKS = 10
        private const val WAKE_LOCK_TIMEOUT_MS = 4 * 60 * 60 * 1000L
        private val STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")

        /**
         * [autoStopMinutes]가 null이면 설정의 자동 정지를 따르고, 값이 있으면(0 = 끔) 이번 녹음만 그 값을 쓴다.
         * [sensors]가 false면 이번 녹음만 센서를 듣지 않는다.
         */
        fun start(context: Context, autoStopMinutes: Int? = null, sensors: Boolean = true): Boolean = try {
            val intent = Intent(context, RecorderService::class.java)
            if (autoStopMinutes != null) intent.putExtra(EXTRA_AUTO_STOP_MINUTES, autoStopMinutes)
            if (!sensors) intent.putExtra(EXTRA_SENSORS, false)
            ContextCompat.startForegroundService(context, intent)
            true
        } catch (e: Exception) {
            RecorderState.update { it.copy(error = "시작 실패: 워치 앱을 열고 다시 시도하세요 (${e.javaClass.simpleName})") }
            false
        }

        fun stop(context: Context) {
            runCatching { context.startService(Intent(context, RecorderService::class.java).setAction(ACTION_STOP)) }
        }
    }
}
