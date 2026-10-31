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
import android.util.Log
import androidx.annotation.MainThread
import androidx.lifecycle.LifecycleOwner
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.Wearable
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
import com.watchpose.handsonar.shared.CalibrationGate
import com.watchpose.handsonar.shared.CalibrationGeometry
import com.watchpose.handsonar.shared.CalibrationInput
import com.watchpose.handsonar.shared.CalibrationMeta
import com.watchpose.handsonar.shared.CalibrationStep
import com.watchpose.handsonar.shared.CameraMeta
import com.watchpose.handsonar.shared.ChirpCheck
import com.watchpose.handsonar.shared.ChirpHeard
import com.watchpose.handsonar.shared.Gesture
import com.watchpose.handsonar.shared.Labels
import com.watchpose.handsonar.shared.MarkerMeta
import com.watchpose.handsonar.shared.PhoneAudioJson
import com.watchpose.handsonar.shared.Posture
import com.watchpose.handsonar.shared.PromptPlanner
import com.watchpose.handsonar.shared.RewearMark
import com.watchpose.handsonar.shared.RunnerEvent
import com.watchpose.handsonar.shared.RunnerPhase
import com.watchpose.handsonar.shared.SensorPlan
import com.watchpose.handsonar.shared.SessionChecks
import com.watchpose.handsonar.shared.SessionContext
import com.watchpose.handsonar.shared.SessionMeta
import com.watchpose.handsonar.shared.SessionRunner
import com.watchpose.handsonar.shared.SessionStats
import com.watchpose.handsonar.shared.Strap
import com.watchpose.handsonar.shared.Tilt
import com.watchpose.handsonar.shared.TiltMessage
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
    /** 기준 자세 결과(meta.json `calibration`). 그 단계 전에 끝났으면 null. */
    val calibration: CalibrationMeta? = null,
    /** 세션 동안 워치 기울기를 받고 있던 시간 비율(meta.json `stats.watchLinkRatio`). */
    val watchLinkRatio: Double? = null,
)

/**
 * 기준 자세 판정의 지금 모습(수집 화면). 각도는 도이고 모르면 null. [forearmOk]가 null이면 워치 기울기를 못 받아 손만 본다.
 * [heldMs]는 조건이 이어진 시간, [holdMs]는 넘어가는 데 필요한 시간이다.
 */
data class CalibrationUi(
    /** 옆면 차례(엄지가 카메라 쪽, 손 각도가 손등·손바닥 쪽 꺾임)인가. 아니면 정면(손 각도가 새끼·엄지 쪽 꺾임). */
    val side: Boolean,
    val handDeg: Double?,
    val forearmDeg: Double?,
    val handOk: Boolean,
    val fingersOk: Boolean,
    val forearmOk: Boolean?,
    val heldMs: Long,
    val holdMs: Long,
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
    /**
     * "워치 다시 참": 켜져 있으면 다음 세션의 meta.json `rewear`가 참이다. 그 세션이 보관되면 꺼지고, 다음 세션 전에 그 세션을 휴지통으로 보내면
     * (되돌리기·기록 화면) 다시 켜지며, 그 뒤 되살리면 다시 꺼진다(shared `RewearMark`).
     */
    val rewear: Boolean = false,
    /**
     * 워치 기울기(녹음 중인 워치가 0.2 s마다 보냄)를 최근 1 s 안에 받았다. 기준 자세에서 팔뚝이 섰는지 보고, "워치 녹음 중" 표시로도 쓴다
     * (폰 마이크로는 워치 처프가 잘 안 들려 처프보다 확실하다).
     */
    val watchTilt: Boolean = false,
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
    /** 이 세션에서 받던 워치 기울기가 [LINK_LOST_MS] 넘게 끊겼다(워치 녹음이 멈췄을 수 있다). 처음부터 못 받았으면 거짓(시작 때 이미 물었다). */
    val watchLinkLost: Boolean = false,
    /** 기준 자세 단계(또는 그 단계에서 일시정지)면 판정 모습, 아니면 null. */
    val calibration: CalibrationUi? = null,
    /** 기준 자세의 지금 차례(shared Labels.CALIBRATION_IDS의 번호, 0 = 옆면). */
    val calibStep: Int = 0,
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
 * 기준 자세(CALIB: 팔뚝과 손을 함께 세운 평행 자세가 2 s 이어지면 넘어감, 건너뛸 수 있음) → 50 ms마다 tick → EndMarkerDue에 끝 표지음 →
 * Finished에 [finishSession](파일 닫기, 끝 표지음 찾기, 통계, meta.json 마무리) → 요약 화면.
 * 기준 자세 판정([CalibrationGate]): 새 카메라 프레임마다 손이 그림 평면에서 그림 속 수직선(폰 가속도로 정함)과 이루는 각, 손가락 펴짐, 워치가
 * 보낸 기울기([TiltMessage], 받은 지 1 s 안)로 본 팔뚝 각을 넣는다. MediaPipe 깊이는 쓰지 않는다(그림 투영에 맞춘 값이라). 결과는 prompts.csv
 * classId −3 줄(유지 구간)과 meta.json `calibration`에 남는다.
 * 취소하면 같은 길로 닫고 status를 trash로 적는다. 앱이 화면에서 벗어나면 세션을 일시정지한다(설계 §6.6).
 * 폰 센서(전송 설계 §8, 계획 6): 세션 동안 [SensorRecorder]가 가속도·자이로·주변 밝기를 `phone_sensors-NNN.csv`에 적고, 세션을 닫을 때
 * meta.json(끝)보다 먼저 닫아 그 보고를 meta.json `sensors`에 넣는다. 센서를 못 열어도 세션은 이어 간다.
 * 세션 정보·"워치 다시 참"(전송 설계 §8, 계획 6): 준비 화면에서 고른 값은 앱 설정 `phone`에 남고, 세션을 시작할 때 meta.json `context`·`rewear`에
 * 들어간다. "다시 참"의 켬·끔은 shared `RewearMark`가 정하고(그 세션이 보관되면 끔, 취소면 켜 둠, 다음 세션 전에 그 세션을 휴지통으로 보내면
 * 다시 켬, 되살리면 다시 끔 — 한 번 다시 찬 것에 표시가 두 세션에 남지 않게, 검토 M1), 바뀔 때마다 `commit`으로 바로 남긴다(검토 M11).
 */
class SessionController(private val app: Application) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val store = SessionStore(app)
    private val audioManager: AudioManager? = app.getSystemService(AudioManager::class.java)
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** "워치 다시 참" 표시(주 스레드에서만). 앱 설정에서 읽어 시작한다. */
    private val rewearMark = RewearMark<File>(prefs.getBoolean(PREF_REWEAR, false))

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
        state.update { it.copy(trackedHand = hand, context = readContext(), rewear = rewearMark.on) }
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

    /** 화면 회전(Display.getRotation). 그림의 위가 기기의 어느 쪽인지 정한다(기준 자세 판정, meta.json camera.displayRotation). */
    private var displayRotation = 0

    // 기준 자세(주 스레드만). 워치 기울기는 메시지 콜백이 바꾼다.
    private val gate = CalibrationGate()
    private var gateFrameNs = 0L
    private var gatePhaseNs = -1L // 판정기가 따르는 CALIB 단계의 시작 시각(바뀌면 — 처음이거나 일시정지 뒤 다시 — 판정기를 비운다)
    private var calibration: CalibrationMeta? = null
    private var sideStep: CalibrationStep? = null // 옆면 차례 결과(정면까지 마치면 meta.json에 함께 적는다)
    @Volatile private var tilt: Tilt? = null
    @Volatile private var tiltAtNs = 0L
    private val tiltListener = MessageClient.OnMessageReceivedListener { event ->
        if (event.path == TiltMessage.PATH) TiltMessage.decode(event.data)?.let {
            tilt = it
            tiltAtNs = SystemClock.elapsedRealtimeNanos()
        }
    }

    // 세션 동안 기울기 연결(주 스레드만): 50 ms tick 수와 그중 기울기가 1 s 안이던 수, 이 세션에서 한 번이라도 받았는가
    private var sessionTicks = 0
    private var linkTicks = 0
    private var linkSeen = false

    init {
        // 워치 기울기도 앱 수명 동안 듣는다(준비 화면에 받는지 보이고, 기준 자세에서 팔뚝을 본다). 못 달면 손만 보고 판정한다.
        // tiltListener 선언 뒤에 둔다: 클래스 앞쪽 init 블록에서는 아직 null이라 등록이 조용히 실패했다(2026-10-31 실기기 "워치 기울기 없음").
        Wearable.getMessageClient(app).addListener(tiltListener).addOnFailureListener { Log.w(TAG, "워치 기울기 수신기를 못 담", it) }
    }

    /** 권한을 받은 뒤(화면이 다시 만들어질 때마다) 부른다: 카메라를 이 화면에 잇고, 마이크·표지음을 연다. */
    @MainThread
    fun openDevices(owner: LifecycleOwner, smallestWidthDp: Int, displayRotation: Int) {
        this.displayRotation = displayRotation
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
        rewearMark.toggle()
        saveRewear()
    }

    /** 표시를 바로 남기고(commit — 보관 직후 앱이 죽어도 꺼진 것이 남게, 검토 M11) 화면에 알린다. */
    private fun saveRewear() {
        prefs.edit().putBoolean(PREF_REWEAR, rewearMark.on).commit()
        state.update { it.copy(rewear = rewearMark.on) }
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
            rewear = rewearMark.on, // 준비 화면 "워치 다시 참"(계획 6)
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
        rewearMark.started() // 앞 세션은 이제 "다시 참" 규칙에서 빠진다(그 뒤 세션은 같은 착용)
        startMarker = null
        startMarkerPlayNs = 0L
        endMarkerPlayNs = 0L
        synchronized(chirpLock) { chirpLog.clear() }
        logChirps = true
        mic.beginSession(File(dir, SessionStore.AUDIO))
        recorder = FrameRecorder(dir)
        // 폰 센서(전송 설계 §8): 폰이 그대로였는지(가속도·자이로)와 카메라 조건(주변 밝기). 못 열어도 세션은 이어 가고 meta.json에 남긴다.
        sensors = SensorRecorder(app, dir, SensorPlan.PHONE_BASE, SensorPlan.PHONE).also { it.start() }
        val r = SessionRunner(plan, calibrate = true)
        runner = r
        gate.reset()
        gateFrameNs = 0L
        gatePhaseNs = -1L
        calibration = null
        sideStep = null
        sessionTicks = 0
        linkTicks = 0
        linkSeen = false
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

    /** 기준 자세를 건너뛴다(그 단계일 때만). prompts.csv에 "skip"으로 남고 meta.json `calibration.result`가 "skip"이다. */
    @MainThread
    fun skipCalibration() {
        val r = runner ?: return
        if (r.phase != RunnerPhase.CALIB) return
        calibration = CalibrationMeta(SKIP, checksNow(SystemClock.elapsedRealtimeNanos()), sideStep, null)
        r.skipCalibration(SystemClock.elapsedRealtimeNanos())
        state.update { it.copy(calibration = null) }
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
            if (ok) {
                rewearMark.trashed(dir)
                saveRewear()
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

    /** 기록 화면의 휴지통. 방금 보관한 "다시 참" 세션이면 되돌리기와 같이 표시를 다시 켜고, 더는 되돌릴 대상이 아니다(검토 M1). */
    @MainThread
    fun trash(dir: File) = changeStatus(dir, SessionMeta.STATUS_TRASH) {
        rewearMark.trashed(dir)
        saveRewear()
        if (state.value.lastDir == dir) state.update { it.copy(lastDir = null) }
    }

    /** 기록 화면의 되살리기. 휴지통으로 보내 "다시 참"을 다시 켰던 그 세션이면 표시를 다시 끈다(두 세션에 표시가 남지 않게). */
    @MainThread
    fun restore(dir: File) = changeStatus(dir, SessionMeta.STATUS_KEPT) {
        rewearMark.restored(dir)
        saveRewear()
    }

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

    /** meta.json status를 바꾸고, 되면 [then](주 스레드)을 부른다. */
    private fun changeStatus(dir: File, status: String, then: () -> Unit) {
        scope.launch {
            val ok = withContext(Dispatchers.IO) { runCatching { store.setStatus(dir, status) }.isSuccess }
            if (ok) then() else state.update { it.copy(error = "상태를 못 바꿈: ${dir.name}") }
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
                    it.copy(
                        handVisible = lostMs < HAND_VISIBLE_MS,
                        handLostMs = if (seen == 0L) 0 else lostMs,
                        cameraLabel = cameraLabel(),
                        watchTilt = freshTilt(now) != null,
                    )
                }
                delay(if (runner != null) TICK_MS else IDLE_TICK_MS)
            }
        }
    }

    private fun tick(r: SessionRunner, now: Long) {
        sessionTicks++
        if (freshTilt(now) != null) {
            linkTicks++
            linkSeen = true
        }
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
        camera.anyHand = r.calibrating // 옆면에서 손 이름이 바뀌어도 하나뿐인 손을 놓치지 않게(기준 자세 동안만)
        if (r.phase == RunnerPhase.CALIB) calibrate(r, now)
        // 보일 동작: 동작 중이면 지금 동작, 일시정지면 다시 시작할 동작(끝부터 다시면 없음)
        val shownIndex = when (r.phase) {
            RunnerPhase.PROMPT -> r.currentPrompt
            RunnerPhase.PAUSED -> r.resumePrompt
            else -> -1
        }
        val nextIndex = when {
            r.calibrating -> 0
            r.phase == RunnerPhase.IDLE || r.phase == RunnerPhase.LEAD -> 0
            r.phase == RunnerPhase.PROMPT || r.phase == RunnerPhase.REST -> r.currentPrompt + 1
            r.phase == RunnerPhase.PAUSED -> if (r.resumePrompt >= 0) r.resumePrompt + 1 else -1
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
                watchLinkLost = linkSeen && now - tiltAtNs > LINK_LOST_MS * 1_000_000,
                calibration = if (r.calibrating) it.calibration else null,
                calibStep = r.calibStep,
            )
        }
    }

    /**
     * 기준 자세 단계에서 새 카메라 프레임이 왔으면 판정에 넣는다. 손: 그림 평면의 각(손목 → 중지 MCP, 그림 속 수직선은 폰 가속도로)과 손가락
     * 펴짐(정면만 — 옆면은 손가락이 겹쳐 그림 길이 비가 흔들린다). 팔뚝: 워치 기울기(받은 지 1 s 안일 때만). 2 s 이어지면 그 유지 구간을
     * 넘긴다: 옆면이면 정면 차례로, 정면이면 첫 동작으로 간다. 옆면에서는 그림 평면 각이 손등·손바닥 쪽 꺾임이고, 정면에서는 새끼·엄지 쪽 꺾임이다.
     * CALIB 단계가 새로 시작됐으면(처음이거나 일시정지 뒤) 판정기를 비우고, 그 단계 시작 전에 찍힌 프레임은 넣지 않는다(멈춘 시간이 유지로 세지 않게).
     */
    private fun calibrate(r: SessionRunner, now: Long) {
        if (r.phaseStartNs != gatePhaseNs) {
            gate.reset()
            gatePhaseNs = r.phaseStartNs
        }
        val f = frames.value ?: return
        if (f.hand.frameNs <= gateFrameNs || f.hand.frameNs < r.phaseStartNs) return
        gateFrameNs = f.hand.frameNs
        val image = f.hand.image
        val w = f.bitmap.width
        val h = f.bitmap.height
        val up = sensors?.latestAccel?.let { CalibrationGeometry.projectedUp(it, displayRotation) }
        val handDeg = if (image != null && up != null) CalibrationGeometry.handInPlaneDeg(image, w, h, up) else null
        val side = r.calibrationId == Labels.NEUTRAL_SIDE_ID
        val fingers = side || (image != null && CalibrationGeometry.fingersStraight(image, w, h))
        val forearmDeg = freshTilt(now)?.let { CalibrationGeometry.forearmDeg(it.values()) }
        val st = gate.update(CalibrationInput(f.hand.frameNs, handDeg, fingers, forearmDeg))
        state.update {
            it.copy(calibration = CalibrationUi(side, handDeg, forearmDeg, st.handOk, st.fingersOk, st.forearmOk, st.heldMs, gate.holdMs))
        }
        val start = st.holdStartNs
        if (st.done && start != null) {
            val step = CalibrationStep(st.handMedianDeg, st.forearmMedianDeg, st.heldMs)
            if (side) sideStep = step else calibration = CalibrationMeta(OK, checksNow(now), sideStep, step)
            r.calibrated(start, now) // 옆면이면 정면 차례가 새로 시작하고(판정기는 차례 시작 시각이 바뀌어 비워진다), 정면이면 첫 동작
            state.update { it.copy(calibration = null) }
        }
    }

    /** 받은 지 [TiltMessage.FRESH_MS] 안의 워치 기울기, 아니면 null. */
    private fun freshTilt(now: Long): Tilt? = tilt?.takeIf { now - tiltAtNs <= TiltMessage.FRESH_MS * 1_000_000 }

    /** meta.json `calibration.checks`: 워치 기울기를 받고 있으면 "hand+forearm", 아니면 "hand". */
    private fun checksNow(now: Long): String = if (freshTilt(now) != null) CHECKS_BOTH else CHECKS_HAND

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
        val calib = calibration // 다음 세션이 지우기 전에 잡아 둔다
        camera.anyHand = false
        val linkRatio = if (sessionTicks > 0) linkTicks.toDouble() / sessionTicks else null
        state.update { it.copy(busy = true, phase = r.phase, calibration = null) }
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
                        watchLinkRatio = linkRatio,
                    )
                    val flags = SessionChecks.flags(handRatio, watchRatio, startMeta.found, endMeta.found, linkRatio)
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
                        calibration = calib,
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
                        calibration = calib,
                        watchLinkRatio = linkRatio,
                    )
                }
            }
            outcome.onSuccess { summary ->
                // "워치 다시 참"은 다시 찬 뒤 첫 보관 세션에 적고 끈다. 취소한 세션(휴지통)이면 켜 둔다: 첫 보관 세션은 아직 오지 않았다.
                rewearMark.closed(dir, marked = meta0.rewear, kept = !cancelled)
                saveRewear()
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
            displayRotation = displayRotation,
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
        const val BLOCK_NS = 500_000_000L
        const val OK = "ok"
        const val SKIP = "skip"
        const val CHECKS_BOTH = "hand+forearm"
        const val CHECKS_HAND = "hand"
        const val LINK_LOST_MS = 10_000L
        const val TAG = "HandSonar"
    }
}
