package com.watchpose.handsonar.shared

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** 작은 FFT(반복 radix-2, 제자리)와 그것으로 하는 선형 상호상관. 표지음 찾기에 쓴다. */
object Fft {
    /** n 이상인 가장 작은 2의 거듭제곱. */
    fun nextPow2(n: Int): Int {
        require(n in 1..(1 shl 30)) { "길이 범위 밖: $n" }
        var p = 1
        while (p < n) p = p shl 1
        return p
    }

    /** [re]·[im](길이가 같은 2의 거듭제곱)을 제자리에서 변환한다. [inverse]면 역변환하고 1/n을 곱한다. */
    fun transform(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        val n = re.size
        require(im.size == n && n > 0 && (n and (n - 1)) == 0) { "길이는 같은 2의 거듭제곱" }
        var j = 0
        for (i in 1 until n) { // 비트 뒤집기 순서로 바꾸기
            var bit = n shr 1
            while ((j and bit) != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                var t = re[i]
                re[i] = re[j]
                re[j] = t
                t = im[i]
                im[i] = im[j]
                im[j] = t
            }
        }
        val sign = if (inverse) 1.0 else -1.0
        var len = 2
        while (len <= n) {
            val angle = sign * 2 * PI / len
            val wr = cos(angle)
            val wi = sin(angle)
            val half = len / 2
            var start = 0
            while (start < n) {
                var cr = 1.0
                var ci = 0.0
                for (k in 0 until half) {
                    val a = start + k
                    val b = a + half
                    val xr = re[b] * cr - im[b] * ci
                    val xi = re[b] * ci + im[b] * cr
                    re[b] = re[a] - xr
                    im[b] = im[a] - xi
                    re[a] += xr
                    im[a] += xi
                    val nr = cr * wr - ci * wi
                    ci = cr * wi + ci * wr
                    cr = nr
                }
                start += len
            }
            len = len shl 1
        }
        if (inverse) {
            for (k in 0 until n) {
                re[k] /= n
                im[k] /= n
            }
        }
    }

    /** 유효 구간 선형 상호상관: c[k] = Σ_j x[k+j]·t[j], k = 0..x.size−t.size. 0을 채워 원형 겹침이 없게 한다. */
    fun correlateValid(x: FloatArray, t: FloatArray): DoubleArray {
        require(t.isNotEmpty() && x.size >= t.size) { "x가 t보다 짧음" }
        val n = nextPow2(x.size + t.size)
        val xr = DoubleArray(n)
        val xi = DoubleArray(n)
        val tr = DoubleArray(n)
        val ti = DoubleArray(n)
        for (i in x.indices) xr[i] = x[i].toDouble()
        for (i in t.indices) tr[i] = t[i].toDouble()
        transform(xr, xi, inverse = false)
        transform(tr, ti, inverse = false)
        for (k in 0 until n) { // X · conj(T)
            val r = xr[k] * tr[k] + xi[k] * ti[k]
            val i = xi[k] * tr[k] - xr[k] * ti[k]
            xr[k] = r
            xi[k] = i
        }
        transform(xr, xi, inverse = true)
        return DoubleArray(x.size - t.size + 1) { xr[it] }
    }
}
