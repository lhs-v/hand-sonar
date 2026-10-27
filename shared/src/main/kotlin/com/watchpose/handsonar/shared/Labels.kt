package com.watchpose.handsonar.shared

/** 손가락 하나의 모양(손 모양 그림용). TOUCH는 끝이 엄지 끝에 닿은 모양(ASL 0·6–9). */
enum class FingerState { EXTENDED, BENT, TOUCH }

/** 손목 꺾음(손 모양 그림용). */
enum class WristPose { NEUTRAL, UP, DOWN, LEFT }

/** 그림으로 보일 손 모양. [fingers]는 엄지·검지·중지·약지·소지 순서로 5개. */
data class HandShape(val fingers: List<FingerState>, val wrist: WristPose = WristPose.NEUTRAL) {
    init {
        require(fingers.size == 5) { "손가락은 5개" }
    }
}

/**
 * 동작 하나(설계 §6.2). [id]는 WatchHand 번호(0–17), [key]는 WatchHand 이름, [name]·[hint]는 화면에 보일 한국어다.
 * 손가락 번호 순서와 손목 세 가지의 방향은 공개 데이터 라벨로 아직 확인하지 않았다(설계 §11-8). 학습은 카메라 관절만 쓰므로
 * 이 대응이 틀려도 데이터는 그대로 쓸 수 있고, [Labels.VERSION]을 올려 다시 매기면 된다.
 */
data class Gesture(val id: Int, val key: String, val name: String, val hint: String, val shape: HandShape)

object Labels {
    /** prompts.csv·meta.json에 적는 라벨 판. 대응을 바꾸면 올린다. */
    const val VERSION = "watchhand-1"

    /** 자유 동작(구간 7)의 번호. */
    const val FREE_ID = -1

    /** 일시정지 기록의 번호. */
    const val PAUSE_ID = -2

    val FINGERS: IntRange = 0..4
    val ASL: IntRange = 5..14
    val WRIST: IntRange = 15..17

    private val E = FingerState.EXTENDED
    private val B = FingerState.BENT
    private val T = FingerState.TOUCH

    /** 번호 0–17 순서. */
    val ALL: List<Gesture> = listOf(
        Gesture(0, "thumb", "엄지 굽히기", "엄지만 손바닥 쪽으로 접기", HandShape(listOf(B, E, E, E, E))),
        Gesture(1, "index", "검지 굽히기", "검지만 접기", HandShape(listOf(E, B, E, E, E))),
        Gesture(2, "middle", "중지 굽히기", "중지만 접기", HandShape(listOf(E, E, B, E, E))),
        Gesture(3, "ring", "약지 굽히기", "약지만 접기", HandShape(listOf(E, E, E, B, E))),
        Gesture(4, "pinky", "소지 굽히기", "새끼손가락만 접기", HandShape(listOf(E, E, E, E, B))),
        Gesture(5, "asl_0", "숫자 0", "네 손가락 끝을 엄지에 모아 O", HandShape(listOf(B, T, T, T, T))),
        Gesture(6, "asl_1", "숫자 1", "검지만 펴기", HandShape(listOf(B, E, B, B, B))),
        Gesture(7, "asl_2", "숫자 2", "검지·중지 펴기(V)", HandShape(listOf(B, E, E, B, B))),
        Gesture(8, "asl_3", "숫자 3", "엄지·검지·중지 펴기", HandShape(listOf(E, E, E, B, B))),
        Gesture(9, "asl_4", "숫자 4", "엄지만 접고 네 손가락 펴기", HandShape(listOf(B, E, E, E, E))),
        Gesture(10, "asl_5", "숫자 5", "다섯 손가락 모두 펴기", HandShape(listOf(E, E, E, E, E))),
        Gesture(11, "asl_6", "숫자 6", "엄지 끝에 새끼손가락을 대고 나머지 펴기", HandShape(listOf(B, E, E, E, T))),
        Gesture(12, "asl_7", "숫자 7", "엄지 끝에 약지를 대고 나머지 펴기", HandShape(listOf(B, E, E, T, E))),
        Gesture(13, "asl_8", "숫자 8", "엄지 끝에 중지를 대고 나머지 펴기", HandShape(listOf(B, E, T, E, E))),
        Gesture(14, "asl_9", "숫자 9", "엄지 끝에 검지를 대고 나머지 펴기", HandShape(listOf(B, T, E, E, E))),
        Gesture(15, "wrist_up", "손목 젖히기", "손등 쪽으로 꺾기", HandShape(listOf(E, E, E, E, E), WristPose.UP)),
        Gesture(16, "wrist_down", "손목 굽히기", "손바닥 쪽으로 꺾기", HandShape(listOf(E, E, E, E, E), WristPose.DOWN)),
        Gesture(17, "wrist_left", "새끼 쪽으로 꺾기", "손을 새끼손가락 쪽으로 꺾기", HandShape(listOf(E, E, E, E, E), WristPose.LEFT)),
    )

    val FREE = Gesture(FREE_ID, "free", "자유 동작", "손가락과 손목을 자유롭게 움직이기", HandShape(listOf(E, E, E, E, E)))
    val PAUSE = Gesture(PAUSE_ID, "pause", "일시정지", "", HandShape(listOf(E, E, E, E, E)))

    /** 번호로 동작을 찾는다(0–17, [FREE_ID], [PAUSE_ID]). */
    fun byId(id: Int): Gesture = when (id) {
        FREE_ID -> FREE
        PAUSE_ID -> PAUSE
        in 0..17 -> ALL[id]
        else -> throw IllegalArgumentException("없는 동작 번호: $id")
    }
}
