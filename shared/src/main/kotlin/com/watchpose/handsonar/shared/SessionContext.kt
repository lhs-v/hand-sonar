package com.watchpose.handsonar.shared

/** 세션 자세(전송 설계 §8, 계획 6). [id]는 meta.json 값, [label]은 화면 글이다. */
enum class Posture(val id: String, val label: String) {
    SITTING("sitting", "앉음"),
    STANDING("standing", "섬"),
}

/** 팔 받침: 팔뚝을 책상에 댔는지, 공중에 들었는지. */
enum class ArmSupport(val id: String, val label: String) {
    DESK("desk", "책상"),
    AIR("air", "공중"),
}

/** 워치 줄 조임. */
enum class Strap(val id: String, val label: String) {
    LOOSE("loose", "헐렁"),
    NORMAL("normal", "보통"),
    TIGHT("tight", "꽉"),
}

/**
 * 세션 정보(전송 설계 §8, 계획 6): 폰 준비 화면에서 고르고 다음 실행에도 남는다. 고르지 않은 칸은 null(장소는 빈 글)이고 meta.json에
 * null로 적는다. 나중에 환경 변수를 가려 보려는 기록이라 처리에는 쓰지 않는다.
 */
data class SessionContext(
    val place: String = "",
    val posture: Posture? = null,
    val armSupport: ArmSupport? = null,
    val strap: Strap? = null,
) {
    /** meta.json `context`. */
    fun toMap(): Map<String, Any?> = linkedMapOf<String, Any?>(
        "place" to place.ifBlank { null },
        "posture" to posture?.id,
        "armSupport" to armSupport?.id,
        "strap" to strap?.id,
    )

    /** 요약 화면 한 줄: "앉음 · 팔 책상 · 줄 보통 · 집 책상 · 워치 다시 참". 고르지 않은 것은 빼고, 모두 비면 "세션 정보 없음". */
    fun line(rewear: Boolean): String = listOfNotNull(
        posture?.label,
        armSupport?.let { "팔 ${it.label}" },
        strap?.let { "줄 ${it.label}" },
        place.ifBlank { null },
        if (rewear) "워치 다시 참" else null,
    ).joinToString(" · ").ifEmpty { "세션 정보 없음" }

    companion object {
        /** 장소 글의 최대 길이(글자 = 코드 포인트, 이모지도 한 글자). */
        const val MAX_PLACE = 40

        /** 손으로 쓴 장소 글을 다듬는다: 줄바꿈·이어진 공백(전각 공백 포함)을 한 칸으로, 앞뒤 공백을 빼고 [MAX_PLACE]자까지([limit]). */
        fun cleanPlace(text: String): String = limit(text.replace(WHITESPACE, " ").trim())

        /** [MAX_PLACE]자까지 자른다. 코드 포인트로 세어 이모지(UTF-16 두 칸)를 반으로 자르지 않는다(검토 M11). 입력 창도 이것으로 막는다. */
        fun limit(text: String): String =
            if (text.codePointCount(0, text.length) <= MAX_PLACE) text else text.substring(0, text.offsetByCodePoints(0, MAX_PLACE))

        /**
         * 유니코드 공백(ASCII 공백 + `\p{Z}` — 전각 공백 U+3000도 든다). `(?U)`는 PC 자바에서만 되고 안드로이드 ICU 정규식이
         * 몰라서 클래스 초기화 때 PatternSyntaxException으로 앱이 시작하자마자 죽었다(2026-10-28 폴드8).
         */
        private val WHITESPACE = Regex("""[\s\p{Z}]+""")
    }
}
