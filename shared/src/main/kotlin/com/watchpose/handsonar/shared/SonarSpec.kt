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
     * 처프 창의 처음 값은 한 창이다(설계 §5.2). 창 없음은 이음새 딸깍(80 Hz)과 10–11 kHz 울림이 들렸다(2026-10-27 실기기).
     * 워치 설정에 저장된 창이 없을 때(새로 설치하거나 앱 데이터를 지운 뒤)만 쓰인다.
     */
    val DEFAULT_WINDOW: SweepWindow = SweepWindow.HANN

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
}
