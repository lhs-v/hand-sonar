package com.watchpose.handsonar.shared

/**
 * 소나 송수신 규격(설계 §3). 공개 데이터(WatchHand)의 송신 파일 `fmcw18000_b3000_l600_s48k.wav`와 같은 값이다:
 * 18 kHz에서 3 kHz 폭, 600샘플(48 kHz에서 12.5 ms) 선형 처프를 쉬지 않고 반복한다.
 */
object SonarSpec {
    const val SAMPLE_RATE = 48_000
    const val START_HZ = 18_000.0
    const val BANDWIDTH_HZ = 3_000.0
    const val SWEEP_SAMPLES = 600

    /** WAV 조각 하나의 프레임 수(1분, 설계 §5.3). 앱이 중간에 죽어도 앞 조각은 남는다. */
    const val PART_FRAMES = SAMPLE_RATE * 60

    /** 녹음 1초의 바이트 수(16비트 1채널). 남은 녹음 시간 계산에 쓴다. */
    const val BYTES_PER_SECOND = SAMPLE_RATE * 2

    /** 처프 진폭(0..1)의 처음 값. 첫 실기기 시험에서 0.5·미디어 볼륨 11/15이면 직접 경로가 포화 없이 또렷했다. 공개 데이터 세기에 맞추지 않는다(설계 §5.2). */
    const val DEFAULT_AMPLITUDE = 0.5f

    /**
     * 처프 창의 처음 값은 Tukey 창이다(2026-10-28). 창 없음은 이음새 딸깍(80 Hz)과 10–11 kHz 울림이 들렸고(2026-10-27 실기기), 그래서 쓰던
     * 한 창은 대역폭을 깎아 직접 경로 봉우리를 두 배로 넓혔다(반높이 폭 34–85칸, 창 없음·공개 WatchHand 프로파일은 20–21칸). 넓은 봉우리가
     * 손목 가까운 손가락 에코를 덮는다. Tukey 창은 양 끝 [TUKEY_RAMP_SAMPLES]만 줄여 봉우리 폭을 창 없음 가까이(시뮬레이션 24칸 대 22칸) 두고
     * 들리는 대역(15 kHz 아래) 누설을 창 없음보다 약 49 dB 낮춘다. 워치 설정에 저장된 창이 없을 때(새로 설치하거나 앱 데이터를 지운 뒤)만 쓰인다.
     */
    val DEFAULT_WINDOW: SweepWindow = SweepWindow.TUKEY

    /** Tukey 창의 양 끝 경사 길이(샘플): 600샘플 처프의 10%씩(1.25 ms). PC `tools/profiles.py`의 송신 처프가 같은 값을 쓴다. */
    const val TUKEY_RAMP_SAMPLES = 60

    /** 워치 화면 "처프 창" 버튼이 도는 차례. */
    val WINDOW_CHOICES: List<SweepWindow> = listOf(SweepWindow.TUKEY, SweepWindow.HANN, SweepWindow.NONE)

    /** "처프 창" 버튼의 다음 값. */
    fun nextWindow(current: SweepWindow): SweepWindow = WINDOW_CHOICES[(WINDOW_CHOICES.indexOf(current) + 1) % WINDOW_CHOICES.size]

    /**
     * 처프 모양의 처음 값은 톱니(18 → 21 kHz를 되풀이)다. 공개 WatchHand 데이터와 같은 모양이라 견주거나 미리 학습할 수 있다.
     * 삼각(18 → 21 → 18 kHz, 2026-10-30)은 이음새에서 주파수와 위상이 끊기지 않아 창 없이도 딸깍이 없고, 한 줄 전체로 상관하면 움직이는
     * 반사체의 거리 어긋남(도플러)이 오름·내림 반에서 서로 상쇄된다(시뮬레이션, 10 cm·0.3 m/s: 톱니 2.6 cm, 삼각 0.2 cm). 대역폭이 같아
     * 봉우리 폭은 그대로다.
     */
    val DEFAULT_SHAPE: SweepShape = SweepShape.SAW

    /**
     * 삼각 처프의 꼭대기 주파수. 한 주기(600샘플)의 위상이 정수 바퀴(평균 19.52 kHz × 12.5 ms = 244바퀴)가 되게 21 kHz에서 40 Hz 올렸다.
     * 그래서 주기를 이어 붙인 자리에서 위상도 이어진다. PC `tools/profiles.py`의 송신 처프가 같은 값을 쓴다.
     */
    const val TRIANGLE_TOP_HZ = 21_040.0

    /** 워치 화면 "처프 모양" 버튼이 도는 차례. */
    val SHAPE_CHOICES: List<SweepShape> = listOf(SweepShape.SAW, SweepShape.TRIANGLE)

    /** "처프 모양" 버튼의 다음 값. */
    fun nextShape(current: SweepShape): SweepShape = SHAPE_CHOICES[(SHAPE_CHOICES.indexOf(current) + 1) % SHAPE_CHOICES.size]

    /**
     * 녹음 음원의 처음 값은 UNPROCESSED다(2026-10-28). WatchHand는 워치에서 UNPROCESSED로 녹음했다(§3.2). 워치8은 지원을 선언하지 않아 자동이면
     * VOICE_RECOGNITION이 되고, 그 녹음은 공개 데이터보다 손 움직임 신호가 약했다. 열리지 않으면 워치가 VOICE_RECOGNITION으로 바꿔 녹음한다.
     */
    val DEFAULT_RECORD_SOURCE: RecordSource = RecordSource.UNPROCESSED

    /** 워치 화면 "녹음 음원" 버튼이 도는 차례. */
    val RECORD_SOURCE_CHOICES: List<RecordSource> = listOf(RecordSource.UNPROCESSED, RecordSource.VOICE_RECOGNITION, RecordSource.AUTO)

    /** "녹음 음원" 버튼의 다음 값. */
    fun nextRecordSource(current: RecordSource): RecordSource =
        RECORD_SOURCE_CHOICES[(RECORD_SOURCE_CHOICES.indexOf(current) + 1) % RECORD_SOURCE_CHOICES.size]

    /** 워치 화면 "세기" 버튼의 다음 값: 10%씩 올리고 100% 다음은 10%로 돌아간다. */
    fun nextAmplitude(current: Float): Float {
        val step = Math.round(current * 10).coerceIn(1, 10)
        return if (step >= 10) 0.1f else (step + 1) / 10f
    }

    /** 자동 정지 선택지(분, 0 = 끔, 설계 §5.1). 워치 화면의 "자동 정지" 버튼이 이 순서로 돈다. */
    val AUTO_STOP_CHOICES: List<Int> = listOf(0, 5, 10, 30)

    /** "자동 정지" 버튼의 다음 값. 선택지에 없는 값이면 끔(0)으로 돌아간다. */
    fun nextAutoStop(current: Int): Int {
        val i = AUTO_STOP_CHOICES.indexOf(current)
        return if (i < 0) 0 else AUTO_STOP_CHOICES[(i + 1) % AUTO_STOP_CHOICES.size]
    }

    /**
     * 워치 "녹음 볼륨" 선택지(전송 설계 §8, 계획 6): 0 = 그대로, 아니면 녹음하는 동안 미디어 볼륨을 그 단계로 맞추고 끝나면 되돌린다.
     * 나중에 볼륨별로 직접 경로 세기·차분 잡음·학습 성능을 견주려는 것이다. 볼륨 11 세션의 평상시 신호가 약 −22 dBFS라 15(약 +8 dB)에서도
     * 14 dB쯤 여유가 있다(설계 §8). 워치 화면의 "녹음 볼륨" 버튼이 이 순서로 돈다.
     */
    val RECORD_VOLUME_CHOICES: List<Int> = listOf(0, 11, 13, 15)

    /** "녹음 볼륨" 버튼의 다음 값. 선택지에 없는 값이면 그대로(0)로 돌아간다. */
    fun nextRecordVolume(current: Int): Int {
        val i = RECORD_VOLUME_CHOICES.indexOf(current)
        return if (i < 0) 0 else RECORD_VOLUME_CHOICES[(i + 1) % RECORD_VOLUME_CHOICES.size]
    }

    /** 녹음 동안 맞출 미디어 볼륨: 그대로(0 이하)면 [current], 아니면 [choice]를 [max] 안으로(지금보다 낮아도 고른 값으로 맞춘다). */
    fun recordVolumeTarget(choice: Int, current: Int, max: Int): Int = if (choice <= 0) current else choice.coerceAtMost(max)
}

/** 처프 창 함수(설계 §5.2). [id]는 rec.json의 `tx.window`에 적는 이름이다. 기본은 [SonarSpec.DEFAULT_WINDOW]. */
enum class SweepWindow(val id: String) {
    NONE("none"),
    HANN("hann"),

    /** 양 끝 [SonarSpec.TUKEY_RAMP_SAMPLES]만 반 코사인으로 올리고 내린다. */
    TUKEY("tukey"),
}

/** 처프 모양(2026-10-30). [id]는 rec.json의 `tx.shape`에 적는 이름이다. `tx.shape`가 없는 옛 녹음은 톱니다. */
enum class SweepShape(val id: String) {
    /** 600샘플 동안 18 → 21 kHz로 오르고 다음 주기에 18 kHz로 돌아간다(공개 데이터와 같음). */
    SAW("saw"),

    /** 300샘플 동안 18 kHz → [SonarSpec.TRIANGLE_TOP_HZ]로 오르고 300샘플 동안 18 kHz로 내린다. */
    TRIANGLE("triangle"),
    ;

    companion object {
        /** rec.json·설정의 이름에서. 모르는 이름이면 null. */
        fun of(id: String?): SweepShape? = entries.firstOrNull { it.id == id }
    }
}

/**
 * 워치 녹음 음원 선택(2026-10-28). [id]는 rec.json의 `audio.sourceRequested`에 적는 이름이다. 실제로 연 음원은 `audio.source`에
 * ("UNPROCESSED" | "VOICE_RECOGNITION") 적는다. [AUTO]는 기기가 UNPROCESSED 지원을 선언하면 그것, 아니면 VOICE_RECOGNITION이다(전부터의 방식).
 */
enum class RecordSource(val id: String) {
    AUTO("auto"),
    UNPROCESSED("unprocessed"),
    VOICE_RECOGNITION("voice_recognition"),
    ;

    companion object {
        /** rec.json·원격 시작의 이름에서. 모르는 이름이면 null. */
        fun of(id: String?): RecordSource? = entries.firstOrNull { it.id == id }
    }
}
