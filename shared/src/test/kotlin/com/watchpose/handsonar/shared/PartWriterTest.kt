package com.watchpose.handsonar.shared

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals

class PartWriterTest {
    @Test
    fun splitsAtPartBoundariesAndKeepsFrameNumbers() {
        val dir = createTempDirectory("parts").toFile()
        try {
            val chunk = ShortArray(300) { it.toShort() }
            val w = PartWriter(dir, sampleRate = 48_000, framesPerPart = 1_000)
            repeat(7) { w.write(chunk) }
            assertEquals(2_100L, w.totalFrames)
            val expected = listOf(
                PartInfo("part-000.wav", 0, 1_000),
                PartInfo("part-001.wav", 1_000, 1_000),
                PartInfo("part-002.wav", 2_000, 100),
            )
            assertEquals(expected, w.parts)
            w.close()
            assertEquals(expected, w.parts)
            assertEquals(listOf(44L + 2_000, 44L + 2_000, 44L + 200), expected.map { File(dir, it.file).length() })
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun samplesContinueAcrossTheBoundary() {
        val dir = createTempDirectory("parts").toFile()
        try {
            PartWriter(dir, framesPerPart = 1_000).use { w -> w.write(ShortArray(1_500) { it.toShort() }) }
            val second = File(dir, "part-001.wav").readBytes()
            assertEquals(1_000, (second[44].toInt() and 0xFF) or ((second[45].toInt() and 0xFF) shl 8))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun closingWithoutDataLeavesNoParts() {
        val dir = createTempDirectory("parts").toFile()
        try {
            val w = PartWriter(dir)
            w.close()
            assertEquals(emptyList(), w.parts)
            assertEquals(0, dir.listFiles()!!.size)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun writesAfterCloseAreIgnored() {
        val dir = createTempDirectory("parts").toFile()
        try {
            val w = PartWriter(dir, framesPerPart = 1_000)
            w.write(ShortArray(300) { it.toShort() })
            w.close()
            val before = w.parts
            w.write(ShortArray(300) { it.toShort() })
            assertEquals(before, w.parts)
            assertEquals(300L, w.totalFrames)
            assertEquals(listOf("part-000.wav"), dir.list()!!.sorted())
        } finally {
            dir.deleteRecursively()
        }
    }
}
