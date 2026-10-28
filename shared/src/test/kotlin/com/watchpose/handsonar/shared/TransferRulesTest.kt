package com.watchpose.handsonar.shared

import java.io.File
import java.io.RandomAccessFile
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals

class TransferRulesTest {
    @Test
    fun sendsPartsInNumberOrderThenRecJsonAndSkipsLeftovers() {
        val files = listOf(
            RecFile("rec.json", 900),
            RecFile("part-010.wav", 44),
            RecFile("part-002.wav", 44),
            RecFile("rec.json.tmp", 900),
            RecFile("part-001.wav.part", 10),
            RecFile("part-1000.wav", 44),
            RecFile("notes.txt", 3),
        )
        assertEquals(listOf("part-002.wav", "part-010.wav", "part-1000.wav", "rec.json"), TransferRules.transferable(files).map { it.name })
    }

    @Test
    fun pendingIsOldestFirstWithoutSentOrLiveRecordings() {
        val folders = listOf("20261028-101500-2", "20261028-120000", "20261027-235407", "20261028-090000", "tmp", "20261028")
        assertEquals(
            listOf("20261027-235407", "20261028-101500-2"),
            TransferRules.pending(folders, sent = setOf("20261028-090000"), live = "20261028-120000"),
        )
        assertEquals(4, TransferRules.pending(folders, sent = emptySet(), live = null).size)
    }

    @Test
    fun missingListsAbsentOrDifferentSizesInOfferOrder() {
        val offer = listOf(RecFile("part-000.wav", 100), RecFile("part-001.wav", 50), RecFile("rec.json", 9))
        assertEquals(listOf("part-001.wav", "rec.json"), TransferRules.missing(offer, mapOf("part-000.wav" to 100L, "part-001.wav" to 49L, "extra.wav" to 1L)))
        assertEquals(emptyList(), TransferRules.missing(offer, mapOf("part-000.wav" to 100L, "part-001.wav" to 50L, "rec.json" to 9L)))
    }

    @Test
    fun verifyChecksSizeAndTheWavHeaderShape() {
        val dir = createTempDirectory("transfer").toFile()
        try {
            val closed = File(dir, "part-000.wav")
            WavWriter(closed, SonarSpec.SAMPLE_RATE).use { it.write(ShortArray(480) { i -> i.toShort() }) }
            val size = closed.length()
            assertEquals(44L + 960, size)
            assertEquals(TransferProtocol.CODE_OK, TransferRules.verify("part-000.wav", size, size, TransferRules.readHeader(closed)))
            assertEquals(TransferProtocol.CODE_BAD_SIZE, TransferRules.verify("part-000.wav", size - 2, size, TransferRules.readHeader(closed)))
            // 앱이 죽어 닫히지 않은 조각: 길이 칸이 RIFF 36·data 0이어도 받는다
            val open = File(dir, "part-001.wav").apply { writeBytes(closed.readBytes()) }
            RandomAccessFile(open, "rw").use { f ->
                f.seek(4)
                f.write(le32(36))
                f.seek(40)
                f.write(le32(0))
            }
            assertEquals(TransferProtocol.CODE_OK, TransferRules.verify("part-001.wav", size, size, TransferRules.readHeader(open)))
            // 닫혔다고 적혔는데 길이가 파일과 다르다
            assertEquals(TransferProtocol.CODE_BAD_WAV, TransferRules.verify("part-000.wav", size + 2, size + 2, TransferRules.readHeader(closed)))
            // 다른 표본율, 망가진 앞머리, 머리말보다 짧은 파일
            val other = File(dir, "part-002.wav")
            WavWriter(other, 44_100).use { it.write(ShortArray(480)) }
            assertEquals(TransferProtocol.CODE_BAD_WAV, TransferRules.verify("part-002.wav", other.length(), other.length(), TransferRules.readHeader(other)))
            assertEquals(TransferProtocol.CODE_BAD_WAV, TransferRules.verify("part-003.wav", size, size, ByteArray(44) { 7 }))
            val short = File(dir, "part-004.wav").apply { writeBytes(ByteArray(10)) }
            assertEquals(10, TransferRules.readHeader(short).size)
            assertEquals(TransferProtocol.CODE_BAD_WAV, TransferRules.verify("part-004.wav", 10, 10, TransferRules.readHeader(short)))
            assertEquals(0, TransferRules.readHeader(File(dir, "none.wav")).size)
            // rec.json은 크기만 본다
            assertEquals(TransferProtocol.CODE_OK, TransferRules.verify("rec.json", 5, 5, "{...}".toByteArray()))
            assertEquals(TransferProtocol.CODE_BAD_SIZE, TransferRules.verify("rec.json", 4, 5, "{..}".toByteArray()))
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun le32(v: Int) = byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte())
}
