package com.watchpose.handsonar.shared

import java.util.Random
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class WatchChirpDetectorTest {
    private fun signal(chirpAmp: Double, noiseSigma: Double, seed: Long, offset: Int = 137): ShortArray {
        val sweep = Sweep.wave(SweepWindow.HANN)
        val r = Random(seed)
        return ShortArray(WatchChirpDetector.BLOCK_SAMPLES) { i ->
            val v = chirpAmp * sweep[(i + offset) % 600] + noiseSigma * r.nextGaussian()
            v.roundToInt().coerceIn(-32_768, 32_767).toShort()
        }
    }

    @Test
    fun repeatedChirpIsHeard() {
        assertEquals(ChirpHeard.HEARD, WatchChirpDetector.check(signal(300.0, 1_000.0, 1)).heard)
    }

    @Test
    fun noiseIsNotHeard() {
        assertEquals(ChirpHeard.NONE, WatchChirpDetector.check(signal(0.0, 1_000.0, 2)).heard)
    }

    @Test
    fun strongerChirpGivesHigherRatio() {
        val weak = WatchChirpDetector.check(signal(100.0, 1_000.0, 3)).ratio
        val strong = WatchChirpDetector.check(signal(400.0, 1_000.0, 3)).ratio
        assertTrue(weak < strong, "$weak < $strong")
    }

    @Test
    fun silenceIsNone() {
        val c = WatchChirpDetector.check(ShortArray(WatchChirpDetector.BLOCK_SAMPLES))
        assertEquals(ChirpHeard.NONE, c.heard)
        assertEquals(0.0, c.ratio)
    }

    @Test
    fun ratioIsCappedForCleanChirp() {
        val c = WatchChirpDetector.check(signal(1_000.0, 0.0, 4))
        assertEquals(ChirpHeard.HEARD, c.heard)
        assertTrue(c.ratio <= 999.0)
    }

    @Test
    fun multipathChirpIsStillHeard() {
        // 2026-10-22 같은 기기 쌍 실측 경로: (지연 칸, 상대 진폭) = (0, .65), (60, 1.0), (111, .77), (169, .52)
        val sweep = Sweep.wave(SweepWindow.HANN)
        val paths = listOf(0 to 0.65, 60 to 1.0, 111 to 0.77, 169 to 0.52)
        val r = Random(5)
        val block = ShortArray(WatchChirpDetector.BLOCK_SAMPLES) { i ->
            var v = 1_000.0 * r.nextGaussian()
            for ((delay, gain) in paths) v += 800.0 * gain * sweep[((i - delay) % 600 + 600) % 600]
            v.roundToInt().coerceIn(-32_768, 32_767).toShort()
        }
        assertEquals(ChirpHeard.HEARD, WatchChirpDetector.check(block).heard)
    }
}
