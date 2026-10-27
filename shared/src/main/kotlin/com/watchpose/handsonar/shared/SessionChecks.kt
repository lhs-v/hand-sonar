package com.watchpose.handsonar.shared

/** 세션 의심 표시(설계 §6.3 기록 화면). meta.json의 flags에 적는다. */
object SessionChecks {
    const val MIN_HAND_RATIO = 0.9
    /**
     * 세션 중 폰이 워치 처프를 들은 비율의 의심 기준. 손을 들면 워치 스피커가 폰 반대쪽을 향해 잘 안 들리는 것이 정상이라
     * (2026-10-27 실기기 5%) 거의 안 들릴 때만 의심한다. 워치 녹음 여부는 준비 화면(손 내림)과 PC(워치 녹음 안 폰 표지음)로 본다.
     */
    const val MIN_WATCH_RATIO = 0.01
    /** 세션 중 실제 초당 프레임(카메라 정답 시간 밀도) 기준(설계 §9-2). 의심 표시(flags)에는 넣지 않고 요약 화면 색에만 쓴다. */
    const val MIN_FPS = 25.0
    const val SUSPECT_HAND = "suspect-hand"
    const val SUSPECT_WATCH = "suspect-watch"
    const val MARKER_MISSING_START = "marker-missing-start"
    const val MARKER_MISSING_END = "marker-missing-end"

    fun flags(handRatio: Double, watchRatio: Double, startMarkerFound: Boolean, endMarkerFound: Boolean): List<String> = buildList {
        if (handRatio < MIN_HAND_RATIO) add(SUSPECT_HAND)
        if (watchRatio < MIN_WATCH_RATIO) add(SUSPECT_WATCH)
        if (!startMarkerFound) add(MARKER_MISSING_START)
        if (!endMarkerFound) add(MARKER_MISSING_END)
    }

    /** 프레임 [frames]개가 첫 [firstNs]부터 마지막 [lastNs]까지 걸렸을 때의 평균 초당 프레임. 2개 미만이거나 간격이 0 이하면 null. */
    fun measuredFps(frames: Long, firstNs: Long, lastNs: Long): Double? =
        if (frames < 2 || lastNs <= firstNs) null else (frames - 1) * 1e9 / (lastNs - firstNs)
}
