package com.watchpose.handsonar.shared

/** I420 평면 세 개(Y 전체, U·V는 가로세로 절반). */
class I420(val width: Int, val height: Int) {
    init {
        require(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0) { "가로세로는 짝수" }
    }

    val y = ByteArray(width * height)
    val u = ByteArray(width * height / 4)
    val v = ByteArray(width * height / 4)
}

/** ARGB_8888 픽셀(Bitmap.getPixels 결과, 행 우선)을 BT.601 제한 범위 I420으로 바꾼다(작은 영상 인코딩용). 색차는 2×2 평균이다. */
object Yuv {
    fun argbToI420(argb: IntArray, width: Int, height: Int, out: I420 = I420(width, height)): I420 {
        require(out.width == width && out.height == height && argb.size >= width * height) { "크기가 맞지 않음" }
        for (p in 0 until width * height) {
            val c = argb[p]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            out.y[p] = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).toByte()
        }
        val cw = width / 2
        for (row in 0 until height / 2) {
            for (col in 0 until cw) {
                val a = 2 * row * width + 2 * col
                val c0 = argb[a]
                val c1 = argb[a + 1]
                val c2 = argb[a + width]
                val c3 = argb[a + width + 1]
                val r = ((((c0 shr 16) and 0xFF) + ((c1 shr 16) and 0xFF) + ((c2 shr 16) and 0xFF) + ((c3 shr 16) and 0xFF)) + 2) / 4
                val g = ((((c0 shr 8) and 0xFF) + ((c1 shr 8) and 0xFF) + ((c2 shr 8) and 0xFF) + ((c3 shr 8) and 0xFF)) + 2) / 4
                val b = (((c0 and 0xFF) + (c1 and 0xFF) + (c2 and 0xFF) + (c3 and 0xFF)) + 2) / 4
                out.u[row * cw + col] = (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).toByte()
                out.v[row * cw + col] = (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).toByte()
            }
        }
        return out
    }
}
