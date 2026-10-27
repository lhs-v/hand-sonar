package com.watchpose.handsonar.shared

/**
 * landmarks.csv(설계 §7.2): 카메라 프레임 하나에 한 줄. meta.json의 video.ok가 참이고 video.frames가 stats.frames와 같으면
 * 영상의 n번째 프레임이 이 파일의 n번째 데이터 줄이다.
 * 열: frameNs, hands(찾은 손 개수), handedness(추적하는 손, "Left" | "Right" — meta.json의 hand와 같은 손), score,
 * x0,y0,z0 … x20,y20,z20, wx0,wy0,wz0 … wx20,wy20,wz20. 추적하는 손을 못 찾으면 handedness부터 빈 칸이다.
 * - x·y: 회전·거울 뒤 프레임(meta.json camera.width×height, 폴드8 480×640) 기준 0..1 정규화 좌표. 손이 화면 끝에 걸리면
 *   0..1을 조금 넘을 수 있다.
 * - z: 손목 기준 상대 깊이(x와 비슷한 크기, 음수 가능).
 * - wx·wy·wz: 손 중심이 원점인 미터 좌표. 거울 그림에서 나왔으므로 좌우가 뒤집혀 있다(실제 손 모양은 x → 1 − x, wx → −wx).
 */
object LandmarkCsv {
    const val POINTS = 21
    const val VALUES = POINTS * 3

    val HEADER: String = buildString {
        append("frameNs,hands,handedness,score")
        for (i in 0 until POINTS) append(",x$i,y$i,z$i")
        for (i in 0 until POINTS) append(",wx$i,wy$i,wz$i")
    }

    /** [image]·[world]는 x,y,z 순서로 21점씩 63개. 추적하는 손이 없으면 handedness·score·image·world를 모두 null로 준다. */
    fun row(frameNs: Long, hands: Int, handedness: String?, score: Float?, image: FloatArray?, world: FloatArray?): String {
        val sb = StringBuilder(1_600)
        sb.append(frameNs).append(',').append(hands).append(',')
        if (handedness != null && score != null && image != null && world != null) {
            require(image.size == VALUES && world.size == VALUES) { "점 값 개수가 63이 아님" }
            sb.append(handedness).append(',').append(score)
            for (v in image) sb.append(',').append(v)
            for (v in world) sb.append(',').append(v)
        } else {
            sb.append(',') // handedness와 score 빈 칸
            repeat(2 * VALUES) { sb.append(',') }
        }
        return sb.toString()
    }
}
