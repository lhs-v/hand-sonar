package com.watchpose.handsonar.mobile.session

import android.app.Application
import android.content.Context
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.annotation.MainThread
import androidx.lifecycle.LifecycleOwner
import com.watchpose.handsonar.mobile.BuildConfig
import com.watchpose.handsonar.mobile.audio.AudioOutputs
import com.watchpose.handsonar.mobile.audio.MarkerFound
import com.watchpose.handsonar.mobile.audio.MarkerPlayer
import com.watchpose.handsonar.mobile.audio.PhoneMic
import com.watchpose.handsonar.mobile.camera.CameraChooser
import com.watchpose.handsonar.mobile.camera.CameraFrame
import com.watchpose.handsonar.mobile.camera.CameraPipeline
import com.watchpose.handsonar.mobile.camera.FrameRecorder
import com.watchpose.handsonar.mobile.camera.TrackedHand
import com.watchpose.handsonar.shared.ArmSupport
import com.watchpose.handsonar.shared.CameraMeta
import com.watchpose.handsonar.shared.ChirpCheck
import com.watchpose.handsonar.shared.ChirpHeard
import com.watchpose.handsonar.shared.Gesture
import com.watchpose.handsonar.shared.Labels
import com.watchpose.handsonar.shared.MarkerMeta
import com.watchpose.handsonar.shared.PhoneAudioJson
import com.watchpose.handsonar.shared.Posture
import com.watchpose.handsonar.shared.PromptPlanner
import com.watchpose.handsonar.shared.RunnerEvent
import com.watchpose.handsonar.shared.RunnerPhase
import com.watchpose.handsonar.shared.SensorPlan
import com.watchpose.handsonar.shared.SessionChecks
import com.watchpose.handsonar.shared.SessionContext
import com.watchpose.handsonar.shared.SessionMeta
import com.watchpose.handsonar.shared.SessionRunner
import com.watchpose.handsonar.shared.SessionStats
import com.watchpose.handsonar.shared.Strap
import com.watchpose.handsonar.shared.VideoMeta
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** 지금 화면. */
enum class Screen { READY, COLLECT, SUMMARY, RECORDS }

/** 표지음 자기 확인 결과(준비 화면). */
data class SelfCheck(val found: Boolean, val ratio: Double?)

/** 방금 끝난 세션 요약(수집 직후 화면). */
data class SummaryInfo(
    val dir: File,
    val durationSec: Double,
    val handRatio: Double,
    val watchRatio: Double,
    val startMarker: Boolean,
    val endMarker: Boolean,
    val flags: List<String>,
    val videoOk: Boolean,
    val fps: Double?,
    val delegate: String,
    val undone: Boolean,
    /** 그 세션의 meta.json에 적은 세션 정보와 "워치 다시 참"(요약 화면 한 줄). */
    val context: SessionContext,
    val rewear: Boolean,
)

/** 화면이 그리는 상태 전부. 바뀔 때마다 새 값으로 바꾼다. */
data class UiState(
    val screen: Screen = Screen.READY,
    val permissionsGranted: Boolean = false,
    val devicesOpen: Boolean = false,
    val cameraLabel: String = "",
    val cameraError: String? = null,
    val micError: String? = null,
    val error: String? = null,
    val handVisible: Boolean = false,
    val handLostMs: Long = 0,
    val chirp: ChirpCheck? = null,
    val selfCheck: SelfCheck? = null,
    val selfCheckBusy: Boolean = false,
    val mediaVolume: Int = 0,
    val mediaMax: Int = 15,
    /** 연결된 이어폰·블루투스 오디오 등 외부 출력의 이름(없으면 빈 목록). 표지음이 폰 스피커로 안 나갈 수 있어 준비 화면이 경고한다. */
    val externalOutputs: List<String> = emptyList(),
    val freeGb: Double = 0.0,
    val extraRotation: Int = 0,
    /** 추적하는 손(워치를 찬 손). 준비 화면에서 고르고 다음 실행에도 남는다. */
    val trackedHand: TrackedHand = TrackedHand.LEFT,
    /** 세션 정보(전송 설계 §8, 계획 6): 준비 화면에서 고르고 다음 실행에도 남는다. 세션을 시작할 때 meta.json `context`에 들어간다. */
    val context: SessionContext = SessionContext(),
    /** "워치 다시 참": 켜져 있으면 다음 세션의 meta.json `rewear`가 참이다. 그 세션이 보관되면 꺼지고, 그 세션을 되돌리면 다시 켜진다. */
    val rewear: Boolean = false,
    // 수집 중
    val phase: RunnerPhase = RunnerPhase.IDLE,
    val segment: Int = 0,
    val segmentCount: Int = 0,
    val prompt: Gesture? = null,
    val nextPrompt: Gesture? = null,
    val promptProgress: Float = 0f,
    val progress: Float = 0f,
    val elapsedSec: Int = 0,
    val plannedSec: Int = 0,
    val startMarkerFound: Boolean? = null,
    val watchGapWarning: Boolean = false,
    val busy: Boolean = false,
    // 수집 뒤
    val summary: SummaryInfo? = null,
    val lastDir: File? = null,
    // 기록
    val records: List<SessionSummary> = emptyList(),
    val showTrash: Boolean = false,
)

/** 세션 중 찾은 표지음. [elapsedNs]는 찾은 그때의 최근 마이크 타임스탬프로 바꾼 값. */
private class FoundMarker(val found: MarkerFound, val elapsedNs: Long?)

/**
 * 장치(카메라·마이크·표지음)와 세션 진행·저장, 화면 상태를 묶는다(설계 §6). 앱 하나에 하나([com.watchpose.handsonar.mobile.HandSonarPhoneApp])라
 * 화면이 다시 만들어져도 세션이 이어진다. 화면은 [ui]·[frame]을 그리고 메서드를 부른다. 메서드는 메인 스레드에서 부른다.
 *
 * 세션 흐름: [startSession] → 폴더·meta.json(시작) → 폰 녹음·프레임 기록·폰 센서 기록 시작 → 0.3 s 뒤 시작 표지음 + [SessionRunner] 시작 →
 * 50 ms마다 tick → EndMarkerDue에 끝 표지음 → Finished에 [finishSession](파일 닫기, 끝 표지음 찾기, 통계, meta.json 마무리) → 요약 화면.
 * 취소하면 같은 길로 닫고 status를 trash로 적는다. 앱이 화면에서 벗어나면 세션을 일시정지한다(설계 §6.6).
 * 폰 센서(전송 설계 §8, 계획 6): 세션 동안 [SensorRecorder]가 가속도·자이로·주변 밝기를 `phone_sensors-NNN.csv`에 적고, 세션을 닫을 때
 * meta.json(끝)보다 먼저 닫아 그 보고를 meta.json `sensors`에 넣는다. 센서를 못 열어도 세션은 이어 간다.
 * 세션 정보·"워치 다시 참"(전송 설계 §8, 계획 6): 준비 화면에서 고른 값은 앱 설정 `phone`에 남고, 세션을 시작할 때 meta.json `context`·`rewear`에
 * 들어간다. "다시 참"은 그 세션이 보관되면 꺼지고(취소면 켜 둠), 방금 세션을 되돌리면 다시 켜진다.
 */
class SessionController(private val app: Application) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val store = SessionStore(app)
    private val audioManager: AudioManager? = app.getSystemService(AudioManager::class.java)
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val state = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = state.asStateFlow()

    private val frames = MutableStateFlow<CameraFrame?>(null)

    /** 가장 최근 카메라 프레임(거울 그림 + 관절). 미리보기가 그린다. */
    val frame: StateFlow<CameraFrame?> = frames.asStateFlow()

    private val camera = CameraPipeline(app, ::onFrame) { message -> state.update { it.copy(cameraError = message) } }
    private val mic = PhoneMic(app, ::onChirp) { message -> state.update { it.copy(micError = message) } }
    private var marker: MarkerPlayer? = null

    /** 이어폰·블루투스 오디오가 꽂히고 빠질 때마다 [UiState.externalOutputs]를 다시 읽는다. */
    private val outputCallback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>?) = refreshOutputs()

        override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>?) = refreshOutputs()
    }

    init {
        // 추적하는 손은 다음 실행에도 남는다. 저장된 값을 모르면 왼손.
        val saved = prefs.getString(PREF_HAND, TrackedHand.LEFT.id)
        val hand = TrackedHand.entries.firstOrNull { it.id == saved } ?: TrackedHand.LEFT
        camera.trackedHand = hand
        // 세션 정보와 "워치 다시 참"도 다음 실행에 남는다(계획 6)
        state.update { it.copy(trackedHand = hand, context = readContext(), rewear = prefs.getBoolean(PREF_REWEAR, false)) }
        // 외부 출력 감시는 앱 수명 동안 이어 가므로(조정자가 앱 하나에 하나) 풀지 않는다. 시작 때 현재 목록을 한 번 채운다.
        audioManager?.registerAudioDeviceCallback(outputCallback, Handler(Looper.getMainLooper()))
        refreshOutputs()
    }

    @Volatile private var lastHandNs = 0L
    @Volatile private var recorder: FrameRecorder? = null
    private var sensors: SensorRecorder? = null // 주 스레드만 바꾼다
    @Volatile private var logChirps = false
    private val chirpLock = Any()
    private val chirpLog = ArrayList<Pair<Long, ChirpHeard>>() // 세션 동안 (판정한 elapsedNs, 상태)

    private var runner: SessionRunner? = null
    private var sessionDir: File? = null
    private var baseMeta: SessionMeta? = null
    private var startMarker: FoundMarker? = null
    private var startMarkerPlayNs = 0L
    private var endMarkerPlayNs = 0L
    private var loopJob: Job? = null
    private var appVisible = true

    /** 방금 보관한 세션([UiState.lastDir])이 "워치 다시 참"으로 적혔다: 되돌리면 표시를 다시 켠다. */
    private var lastRewear = false

    /** 권한을 받은 뒤(화면이 다시 만들어질 때마다) 부른다: 카메라를 이 화면에 잇고, 마이크·표지음을 연다. */
    @MainThread
    fun openDevices(owner: LifecycleOwner, smallestWidthDp: Int, displayRotation: Int) {
        state.update { it.copy(permissionsGranted = true, error = null) }
        camera.start(owner, smallestWidthDp, displayRotation)
        val micError = mic.start()
        if (marker == null) {
            marker = try {
                MarkerPlayer(app)
            } catch (e: Exception) {
                state.update { it.copy(error = "표지음 재생기를 못 엶: ${e.message ?: e.javaClass.simpleName}") }
                null
            }
        }
        state.update { it.copy(devicesOpen = true, micError = micError) }
        refreshEnvironment()
        startLoop()
    }

    @MainThread
    fun onPermissionsDenied() {
        state.update { it.copy(permissionsGranted = false, error = "카메라와 마이크 권한이 있어야 수집할 수 있습니다") }
    }

    /** 앱이 화면에 보이거나 사라질 때. 사라지면 세션은 일시정지하고, 세션이 없으면 마이크를 멈춘다. */
    @MainThread
    fun onAppVisible(visible: Boolean) {
        appVisible = visible
        if (!state.value.devicesOpen) return
        if (visible) {
            if (!mic.running) {
                val micError = mic.start()
                state.update { it.copy(micError = micError) }
            }
            refreshEnvironment()
        } else {
            val r = runner
            if (r != null) {
                if (r.phase != RunnerPhase.PAUSED) r.pause(SystemClock.elapsedRealtimeNanos())
            } else if (!state.value.busy) {
                mic.stop()
            }
        }
    }

    @MainThread
    fun testMarker() {
        val m = marker ?: return
        if (state.value.selfCheckBusy || runner != null) return
        state.update { it.copy(selfCheckBusy = true) }
        runCatching { m.play() }
        scope.launch {
            delay(MARKER_WAIT_MS)
            val found = withContext(Dispatchers.Default) { mic.findMarker() }
            state.update { it.copy(selfCheckBusy = false, selfCheck = SelfCheck(found != null, found?.ratio)) }
        }
    }

    @MainThread
    fun cycleRotation() {
        val next = (camera.extraRotation + 90) % 360
        camera.extraRotation = next
        state.update { it.copy(extraRotation = next) }
    }

    /** 추적하는 손(워치를 찬 손)을 반대쪽으로 바꾸고 기억해 둔다. 세션 중이거나 저장 중이면 바꾸지 않는다. */
    @MainThread
    fun toggleHand() {
        if (runner != null || state.value.busy) return
        val next = if (state.value.trackedHand == TrackedHand.LEFT) TrackedHand.RIGHT else TrackedHand.LEFT
        camera.trackedHand = next
        prefs.edit().putString(PREF_HAND, next.id).apply()
        lastHandNs = 0L // 앞 손의 인식은 버린다: 바뀐 손이 보일 때까지 "안 보임"
        state.update { it.copy(trackedHand = next) }
    }

    /** 세션 정보를 바꾸고 기억해 둔다(전송 설계 §8, 계획 6). 장소는 다듬어 둔다. 세션 중이거나 저장 중이면 바꾸지 않는다. */
    @MainThread
    fun updateContext(change: (SessionContext) -> SessionContext) {
        if (runner != null || state.value.busy) return
        val next = change(state.value.context).let { it.copy(place = SessionContext.cleanPlace(it.place)) }
        prefs.edit()
            .putString(PREF_PLACE, next.place)
            .putString(PREF_POSTURE, next.posture?.id) // null이면 지운다
            .putString(PREF_ARM, next.armSupport?.id)
            .putString(PREF_STRAP, next.strap?.id)
            .apply()
        state.update { it.copy(context = next) }
    }

    /** "워치 다시 참"을 켜고 끈다(계획 6). 세션 중이거나 저장 중이면 바꾸지 않는다. */
    @MainThread
    fun toggleRewear() {
        if (runner != null || state.value.busy) return
        setRewear(!state.value.rewear)
    }

    private fun setRewear(on: Boolean) {
        prefs.edit().putBoolean(PREF_REWEAR, on).apply()
        state.update { it.copy(rewear = on) }
    }

    private fun readContext(): SessionContext {
        val posture = prefs.getString(PREF_POSTURE, null)
        val arm = prefs.getString(PREF_ARM, null)
        val strap = prefs.getString(PREF_STRAP, null)
        return SessionContext(
            place = prefs.getString(PREF_PLACE, null).orEmpty(),
            posture = Posture.entries.firstOrNull { it.id == posture },
            armSupport = ArmSupport.entries.firstOrNull { it.id == arm },
            strap = Strap.entries.firstOrNull { it.id == strap },
        )
    }

    @MainThread
    fun startSession() {
        if (runner != null || state.value.busy) return
        val m = marker ?: return
        val wallMs = System.currentTimeMillis()
        val dir = try {
            store.newFolder(wallMs)
        } catch (e: Exception) {
            state.update { it.copy(error = "세션 폴더를 못 만듦: ${e.message}") }
            return
        }
        val plan = PromptPlanner.plan(seed = wallMs, freeMotionSec = PromptPlanner.DEFAULT_FREE_SEC)
        val meta = SessionMeta(
            app = BuildConfig.VERSION_NAME,
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            android = "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
            camera = cameraMeta(),
            hand = camera.trackedHand.id.lowercase(),
            startWallMs = wallMs,
            startNs = SystemClock.elapsedRealtimeNanos(),
            endWallMs = null,
            endNs = null,
            plan = plan,
            status = SessionMeta.STATUS_KEPT,
            flags = emptyList(),
            memo = "",
            stats = null,
            rewear = state.value.rewear, // 준비 화면 "워치 다시 참"(계획 6)
            video = null,
            audioFile = SessionStore.AUDIO,
            audioSource = mic.sourceName,
            cancelled = false,
            context = state.value.context,
        )
        try {
            store.writeMeta(dir, meta.toJson())
        } catch (e: Exception) {
            state.update { it.copy(error = "meta.json을 못 씀: ${e.message}") }
            return
        }
        sessionDir = dir
        baseMeta = meta
        lastRewear = false // 앞 세션은 이제 되돌릴 수 없다(lastDir도 지운다)
        startMarker = null
        startMarkerPlayNs = 0L
        endMarkerPlayNs = 0L
        synchronized(chirpLock) { chirpLog.clear() }
        logChirps = true
        mic.beginSession(File(dir, SessionStore.AUDIO))
        recorder = FrameRecorder(dir)
        // 폰 센서(전송 설계 §8): 폰이 그대로였는지(가속도·자이로)와 카메라 조건(주변 밝기). 못 열어도 세션은 이어 가고 meta.json에 남긴다.
        sensors = SensorRecorder(app, dir, SensorPlan.PHONE_BASE, SensorPlan.PHONE).also { it.start() }
        val r = SessionRunner(plan)
        runner = r
        state.update {
            it.copy(
                screen = Screen.COLLECT,
                summary = null,
                lastDir = null,
                startMarkerFound = null,
                error = null,
                plannedSec = (plan.plannedMs / 1_000).toInt(),
                segmentCount = plan.segments.size,
                phase = RunnerPhase.IDLE,
            )
        }
        scope.launch {
            delay(START_DELAY_MS) // 폰 녹음이 시작될 틈
            if (runner !== r) return@launch // 그사이 취소됨
            startMarkerPlayNs = runCatching { m.play() }.getOrDefault(0L)
            val startedAt = SystemClock.elapsedRealtimeNanos()
            r.start(startedAt)
            if (!appVisible) r.pause(startedAt) // 시작 지연 사이에 앱이 화면에서 벗어남(설계 §6.6)
            delay(MARKER_WAIT_MS)
            val found = withContext(Dispatchers.Default) { mic.findMarker() }
            if (runner !== r) return@launch
            startMarker = found?.let { FoundMarker(it, mic.elapsedNsOf(it.totalFrame)) }
            state.update { it.copy(startMarkerFound = found != null) }
        }
    }

    @MainThread
    fun togglePause() {
        val r = runner ?: return
        val now = SystemClock.elapsedRealtimeNanos()
        if (r.phase == RunnerPhase.PAUSED) r.resume(now) else r.pause(now)
    }

    @MainThread
    fun markWrong() {
        runner?.markWrong()
    }

    @MainThread
    fun cancelSession() {
        val r = runner ?: return
        r.cancel(SystemClock.elapsedRealtimeNanos())
        finishSession(r, cancelled = true)
    }

    /**
     * 방금 세션을 휴지통으로(설계 §6.3). 다음 세션을 시작하기 전까지 할 수 있다. 그 세션이 "워치 다시 참"으로 적혔으면 표시를 다시 켠다
     * (다시 찬 뒤 첫 보관 세션은 다음 세션이 된다, 계획 6).
     */
    @MainThread
    fun undoLast() {
        val dir = state.value.lastDir ?: return
        scope.launch {
            val ok = withContext(Dispatchers.IO) { runCatching { store.setStatus(dir, SessionMeta.STATUS_TRASH) }.isSuccess }
            if (ok && lastRewear) {
                lastRewear = false
                setRewear(true)
            }
            state.update {
                if (ok) it.copy(lastDir = null, summary = it.summary?.copy(undone = true)) else it.copy(error = "되돌리기 실패: meta.json을 못 바꿈")
            }
        }
    }

    @MainThread
    fun nextSession() {
        state.update { it.copy(screen = Screen.READY, summary = null) }
        refreshEnvironment()
    }

    @MainThread
    fun showRecords() {
        state.update { it.copy(screen = Screen.RECORDS) }
        reloadRecords()
    }

    @MainThread
    fun setShowTrash(trash: Boolean) {
        state.update { it.copy(showTrash = trash) }
    }

    @MainThread
    fun backToReady() {
        state.update { it.copy(screen = Screen.READY) }
        refreshEnvironment()
    }

    @MainThread
    fun trash(dir: File) = changeStatus(dir, SessionMeta.STATUS_TRASH)

    @MainThread
    fun restore(dir: File) = changeStatus(dir, SessionMeta.STATUS_KEPT)

    /** 휴지통 세션을 지운다(되돌릴 수 없음). 기록 화면에서 사용자가 확인한 뒤에만 부른다. */
    @MainThread
    fun deleteForever(dir: File) {
        scope.launch {
            val ok = withContext(Dispatchers.IO) { store.deleteForever(dir) }
            state.update {
                it.copy(
                    error = if (ok) it.error else "지우지 못함: ${dir.name}",
                    lastDir = if (it.lastDir == dir) null else it.lastDir,
                )
            }
            reloadRecords()
        }
    }

    private fun changeStatus(dir: File, status: String) {
        scope.launch {
            val ok = withContext(Dispatchers.IO) { runCatching { store.setStatus(dir, status) }.isSuccess }
            if (!ok) state.update { it.copy(error = "상태를 못 바꿈: ${dir.name}") }
            reloadRecords()
        }
    }

    private fun reloadRecords() {
        scope.launch {
            val list = withContext(Dispatchers.IO) { store.list() }
            state.update { it.copy(records = list) }
        }
    }

    private fun startLoop() {
        if (loopJob?.isActive == true) return
        loopJob = scope.launch {
            while (isActive) {
                val now = SystemClock.elapsedRealtimeNanos()
                runner?.let { tick(it, now) }
                val seen = lastHandNs
                val lostMs = if (seen == 0L) Long.MAX_VALUE else (now - seen) / 1_000_000
                state.update {
                    it.copy(handVisible = lostMs < HAND_VISIBLE_MS, handLostMs = if (seen == 0L) 0 else lostMs, cameraLabel = cameraLabel())
                }
                delay(if (runner != null) TICK_MS else IDLE_TICK_MS)
            }
        }
    }

    private fun tick(r: SessionRunner, now: Long) {
        for (event in r.tick(now)) {
            when (event) {
                RunnerEvent.EndMarkerDue -> endMarkerPlayNs = runCatching { marker?.play() ?: 0L }.getOrDefault(0L)
                RunnerEvent.Finished -> {
                    finishSession(r, cancelled = false)
                    return
                }
                else -> Unit
            }
        }
        // 보일 동작: 동작 중이면 지금 동작, 일시정지면 다시 시작할 동작(끝부터 다시면 없음)
        val shownIndex = when (r.phase) {
            RunnerPhase.PROMPT -> r.currentPrompt
            RunnerPhase.PAUSED -> r.resumePrompt
            else -> -1
        }
        val nextIndex = when (r.phase) {
            RunnerPhase.IDLE, RunnerPhase.LEAD -> 0
            RunnerPhase.PROMPT, RunnerPhase.REST -> r.currentPrompt + 1
            RunnerPhase.PAUSED -> if (r.resumePrompt >= 0) r.resumePrompt + 1 else -1
            else -> -1
        }
        val shown = r.plan.prompts.getOrNull(shownIndex)
        val next = r.plan.prompts.getOrNull(nextIndex)
        state.update {
            it.copy(
                phase = r.phase,
                segment = (shown ?: next)?.segment ?: 0,
                prompt = shown?.let { p -> Labels.byId(p.classId) },
                nextPrompt = next?.let { p -> Labels.byId(p.classId) },
                promptProgress = r.promptProgress(now),
                progress = r.progress(now),
                elapsedSec = (r.activeNs(now) / 1_000_000_000L).toInt(),
                watchGapWarning = watchGapNow(),
            )
        }
    }

    /** 세션을 닫는다: 프레임·소리 기록을 멈추고, 끝 표지음을 찾고, 통계·prompts.csv·phone_audio.json·meta.json을 쓴다. */
    @MainThread
    private fun finishSession(r: SessionRunner, cancelled: Boolean) {
        if (runner !== r) return
        val dir = sessionDir ?: return
        val meta0 = baseMeta ?: return
        runner = null
        logChirps = false
        val rec = recorder
        recorder = null
        val sens = sensors
        sensors = null
        val endNs = if (r.endNs > 0) r.endNs else SystemClock.elapsedRealtimeNanos()
        val endWallMs = System.currentTimeMillis()
        val started = r.startNs > 0
        val startFound = startMarker
        val startPlayNs = startMarkerPlayNs
        val endPlayNs = endMarkerPlayNs
        val chirps = synchronized(chirpLock) { chirpLog.toList() }
        state.update { it.copy(busy = true, phase = r.phase) }
        scope.launch {
            // 끝 표지음은 낸 지 0.8 s가 지난 뒤 고리 버퍼에서 찾는다. 보통 Finished가 그 시각이지만, tick이 늦어 EndMarkerDue와
            // Finished가 한꺼번에 오면 표지음을 막 낸 참이므로 남은 시간을 기다린다. 취소면 끝 표지음이 없다.
            val endFound = if (!cancelled && endPlayNs > 0) {
                val waitMs = (endPlayNs + SessionRunner.CLOSING_MS * 1_000_000 - SystemClock.elapsedRealtimeNanos()) / 1_000_000
                if (waitMs > 0) delay(waitMs)
                withContext(Dispatchers.Default) { mic.findMarker()?.let { FoundMarker(it, mic.elapsedNsOf(it.totalFrame)) } }
            } else {
                null
            }
            val outcome = withContext(Dispatchers.IO) {
                runCatching {
                    // 소리를 먼저 닫는다: 영상 마무리는 길게 막힐 수 있어(최대 약 12 s) 그동안 WAV가 열린 채 남지 않게.
                    val audio = mic.endSession()
                    // 센서 CSV도 영상보다 먼저 닫는다(0.5 s 안). end가 적힌 meta.json의 세션은 CSV가 다 닫혀 있다.
                    val sensorReport = sens?.let {
                        it.stop()
                        it.report()
                    }
                    val videoOk = rec?.close() ?: false
                    val fps = rec?.measuredFps
                    val frameCount = rec?.frames ?: 0L
                    val handRatio = if (frameCount > 0) (rec?.handFrames ?: 0L).toDouble() / frameCount else 0.0
                    val heard = chirps.count { it.second != ChirpHeard.NONE }
                    val watchRatio = if (chirps.isNotEmpty()) heard.toDouble() / chirps.size else 0.0
                    fun markerMeta(f: FoundMarker?, playNs: Long) = MarkerMeta(
                        found = f != null,
                        playNs = playNs,
                        micFrame = if (audio == null) null else f?.let { it.found.totalFrame - audio.startTotalFrame },
                        elapsedNs = f?.elapsedNs,
                        ratio = f?.found?.ratio,
                    )
                    val startMeta = markerMeta(startFound, startPlayNs)
                    val endMeta = markerMeta(endFound, endPlayNs)
                    val stats = SessionStats(
                        frames = frameCount,
                        handRatio = handRatio,
                        watchHeardRatio = watchRatio,
                        watchGaps = watchGaps(chirps),
                        startMarker = startMeta,
                        endMarker = endMeta,
                        durationSec = if (started) (endNs - r.startNs) / 1e9 else 0.0,
                    )
                    val flags = SessionChecks.flags(handRatio, watchRatio, startMeta.found, endMeta.found)
                    store.writePrompts(dir, r.records)
                    if (audio != null) {
                        store.writeText(dir, SessionStore.AUDIO_JSON, PhoneAudioJson.write(audio.source, audio.clockSource, audio.clock, audio.frames))
                    }
                    val meta = meta0.copy(
                        camera = cameraMeta(fps) ?: meta0.camera,
                        endWallMs = endWallMs,
                        endNs = endNs,
                        status = if (cancelled) SessionMeta.STATUS_TRASH else SessionMeta.STATUS_KEPT,
                        flags = flags,
                        stats = stats,
                        // 프레임이 하나도 없으면 video.mp4도 없다: video는 null
                        video = rec?.takeIf { it.frames > 0 }?.let { VideoMeta(FrameRecorder.VIDEO_FILE, it.videoWidth, it.videoHeight, it.videoFrames, videoOk) },
                        cancelled = cancelled,
                        sensors = sensorReport,
                    )
                    store.writeMeta(dir, meta.toJson())
                    SummaryInfo(
                        dir = dir,
                        durationSec = stats.durationSec,
                        handRatio = handRatio,
                        watchRatio = watchRatio,
                        startMarker = startMeta.found,
                        endMarker = endMeta.found,
                        flags = flags,
                        // 폰 녹음이 없거나(마이크 오류 뒤 다시 열림) 오류면 "오류"로 보인다
                        videoOk = videoOk && frameCount > 0 && audio != null && audio.error == null,
                        fps = fps,
                        delegate = camera.delegate,
                        undone = false,
                        context = meta0.context,
                        rewear = meta0.rewear,
                    )
                }
            }
            outcome.onSuccess { summary ->
                // "워치 다시 참"은 다시 찬 뒤 첫 보관 세션에 적고 끈다. 취소한 세션(휴지통)이면 켜 둔다: 첫 보관 세션은 아직 오지 않았다.
                lastRewear = !cancelled && meta0.rewear
                if (lastRewear) setRewear(false)
                state.update {
                    it.copy(
                        busy = false,
                        phase = RunnerPhase.IDLE,
                        prompt = null,
                        nextPrompt = null,
                        screen = if (cancelled) Screen.READY else Screen.SUMMARY,
                        summary = if (cancelled) null else summary,
                        lastDir = if (cancelled) null else dir,
                    )
                }
            }.onFailure { e ->
                state.update {
                    it.copy(
                        busy = false,
                        phase = RunnerPhase.IDLE,
                        prompt = null,
                        nextPrompt = null,
                        screen = Screen.READY,
                        error = "세션 저장 오류: ${e.message ?: e.javaClass.simpleName}",
                    )
                }
            }
            refreshEnvironment()
            if (!appVisible) mic.stop()
        }
    }

    private fun onFrame(f: CameraFrame) { // 카메라 분석 스레드
        frames.value = f
        if (f.hand.image != null) lastHandNs = f.hand.frameNs
        recorder?.write(f)
    }

    private fun onChirp(c: ChirpCheck) { // 마이크 스레드
        state.update { it.copy(chirp = c) }
        if (logChirps) synchronized(chirpLock) { chirpLog += SystemClock.elapsedRealtimeNanos() to c.heard }
    }

    /**
     * 마지막 판정 [WARN_BLOCKS]개(30 s)가 내리 "안 들림"이면 참(설계 §6.5 경고). 손을 들면 워치 스피커가 폰 반대쪽을 향해
     * 폰이 처프를 잘 못 듣는 것이 정상이라(2026-10-27 실기기 세션 중 5%), 녹음이 꺼졌다고 볼 만큼 오래 안 들릴 때만 경고한다.
     */
    private fun watchGapNow(): Boolean = synchronized(chirpLock) {
        chirpLog.size >= WARN_BLOCKS && chirpLog.takeLast(WARN_BLOCKS).all { it.second == ChirpHeard.NONE }
    }

    /** "안 들림" 판정이 [GAP_BLOCKS]개 이상 이어진 구간을 [시작 ns, 끝 ns]로. 판정 하나는 그 시각까지의 0.5 s다. */
    private fun watchGaps(chirps: List<Pair<Long, ChirpHeard>>): List<LongArray> {
        val gaps = mutableListOf<LongArray>()
        var runStart = -1
        for (i in 0..chirps.size) {
            val none = i < chirps.size && chirps[i].second == ChirpHeard.NONE
            if (none && runStart < 0) runStart = i
            if (!none && runStart >= 0) {
                if (i - runStart >= GAP_BLOCKS) gaps += longArrayOf(chirps[runStart].first - BLOCK_NS, chirps[i - 1].first)
                runStart = -1
            }
        }
        return gaps
    }

    private fun cameraMeta(measuredFps: Double? = null): CameraMeta? {
        val c = camera.chosen ?: return null
        return CameraMeta(
            c.id, c.display, camera.frameWidth, camera.frameHeight, CameraPipeline.FPS, camera.timeBase,
            mirrored = true, rotation = camera.rotation, delegate = camera.delegate.ifEmpty { null }, measuredFps = measuredFps,
        )
    }

    private fun cameraLabel(): String {
        val c = camera.chosen ?: return "카메라 고르는 중"
        val side = if (c.display == CameraChooser.INNER) "안쪽" else "덮개"
        val size = if (camera.frameWidth > 0) " · ${camera.frameWidth}×${camera.frameHeight}" else ""
        val time = if (camera.timeBase == CameraChooser.REALTIME) "" else " · 프레임 시각 근사"
        val rate = if (camera.fps > 0f) " · 초당 ${camera.fps.toInt()}" else ""
        val device = if (camera.delegate.isNotEmpty()) " · ${camera.delegate}" else ""
        return "$side 카메라 ${c.id}$size$rate$device$time"
    }

    private fun refreshEnvironment() {
        val am = audioManager
        state.update {
            it.copy(
                mediaVolume = am?.getStreamVolume(AudioManager.STREAM_MUSIC) ?: 0,
                mediaMax = am?.getStreamMaxVolume(AudioManager.STREAM_MUSIC) ?: 15,
                freeGb = store.freeBytes() / 1e9,
            )
        }
    }

    private fun refreshOutputs() {
        val names = AudioOutputs.externalNames(audioManager)
        state.update { it.copy(externalOutputs = names) }
    }

    private companion object {
        const val PREFS = "phone"
        const val PREF_HAND = "hand"
        const val PREF_PLACE = "ctx.place"
        const val PREF_POSTURE = "ctx.posture"
        const val PREF_ARM = "ctx.arm"
        const val PREF_STRAP = "ctx.strap"
        const val PREF_REWEAR = "rewear"
        const val START_DELAY_MS = 300L
        const val MARKER_WAIT_MS = 800L
        const val TICK_MS = 50L
        const val IDLE_TICK_MS = 250L
        const val HAND_VISIBLE_MS = 500L
        const val GAP_BLOCKS = 2
        const val WARN_BLOCKS = 60
        const val BLOCK_NS = 500_000_000L
    }
}
