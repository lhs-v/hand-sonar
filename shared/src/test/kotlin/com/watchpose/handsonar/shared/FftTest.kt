package com.watchpose.handsonar.shared

import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class FftTest {
    @Test
    fun nextPow2() {
        assertEquals(1, Fft.nextPow2(1))
        assertEquals(8, Fft.nextPow2(5))
        assertEquals(8, Fft.nextPow2(8))
        assertEquals(131_072, Fft.nextPow2(73_921))
    }

    @Test
    fun forwardThenInverseGivesBack() {
        val rnd = Random(1)
        val re = DoubleArray(64) { rnd.nextGaussian() }
        val im = DoubleArray(64)
        val orig = re.copyOf()
        Fft.transform(re, im, inverse = false)
        Fft.transform(re, im, inverse = true)
        for (i in 0 until 64) {
            assertEquals(orig[i], re[i], 1e-9)
            assertEquals(0.0, im[i], 1e-9)
        }
    }

    @Test
    fun impulseHasFlatSpectrum() {
        val re = DoubleArray(16)
        val im = DoubleArray(16)
        re[0] = 1.0
        Fft.transform(re, im, inverse = false)
        for (i in 0 until 16) {
            assertEquals(1.0, re[i], 1e-12)
            assertEquals(0.0, im[i], 1e-12)
        }
    }

    @Test
    fun correlationMatchesDirectSum() {
        val rnd = Random(2)
        val x = FloatArray(300) { rnd.nextGaussian().toFloat() }
        val t = FloatArray(37) { rnd.nextGaussian().toFloat() }
        val c = Fft.correlateValid(x, t)
        assertEquals(264, c.size)
        for (k in c.indices) {
            var d = 0.0
            for (j in t.indices) d += x[k + j].toDouble() * t[j]
            assertEquals(d, c[k], 1e-6)
        }
    }
}
