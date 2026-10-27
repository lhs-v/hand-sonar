package com.watchpose.handsonar.shared

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals

class WavWriterTest {
    private fun le16(b: ByteArray, at: Int) = (b[at].toInt() and 0xFF) or ((b[at + 1].toInt() and 0xFF) shl 8)
    private fun le32(b: ByteArray, at: Int) = le16(b, at) or (le16(b, at + 2) shl 16)

    @Test
    fun headerAndSamplesAreLittleEndianPcm16Mono() {
        val dir = createTempDirectory("wav").toFile()
        try {
            val f = File(dir, "a.wav")
            WavWriter(f, 48_000).use { w ->
                w.write(shortArrayOf(1, -2, 300), 3)
                w.write(ShortArray(997) { 7 })
                assertEquals(1_000L, w.frames)
            }
            val b = f.readBytes()
            assertEquals(44 + 2_000, b.size)
            assertEquals("RIFF", String(b, 0, 4, Charsets.US_ASCII))
            assertEquals(36 + 2_000, le32(b, 4))
            assertEquals("WAVEfmt ", String(b, 8, 8, Charsets.US_ASCII))
            assertEquals(16, le32(b, 16))
            assertEquals(1, le16(b, 20)) // PCM
            assertEquals(1, le16(b, 22)) // 1채널
            assertEquals(48_000, le32(b, 24))
            assertEquals(96_000, le32(b, 28))
            assertEquals(2, le16(b, 32))
            assertEquals(16, le16(b, 34))
            assertEquals("data", String(b, 36, 4, Charsets.US_ASCII))
            assertEquals(2_000, le32(b, 40))
            assertEquals(1, le16(b, 44))
            assertEquals(0xFFFE, le16(b, 46)) // -2
            assertEquals(300, le16(b, 48))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun unclosedFileHasAllSamplesAndEmptyDataLength() {
        val dir = createTempDirectory("wav").toFile()
        try {
            val f = File(dir, "b.wav")
            val w = WavWriter(f, 48_000)
            w.write(ShortArray(10) { 1 })
            val b = f.readBytes()
            assertEquals(44 + 20, b.size)
            assertEquals(36, le32(b, 4)) // 닫기 전 RIFF 길이 = 36 + 0
            assertEquals(0, le32(b, 40))
            w.close()
        } finally {
            dir.deleteRecursively()
        }
    }
}
