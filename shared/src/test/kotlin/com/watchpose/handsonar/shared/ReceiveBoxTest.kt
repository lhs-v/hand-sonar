package com.watchpose.handsonar.shared

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ReceiveBoxTest {
    private val rec = "20261028-101500"

    /** 받는 폴더 하나와, 앱 프로세스가 다시 뜬 것처럼 같은 폴더·기록으로 새 상자를 만드는 함수. */
    private class Setup(val dir: File) {
        val root = File(dir, "watch-recordings")
        val log = File(dir, "watch-received.txt")
        fun box() = ReceiveBox(root, NameLog(log), settleMs = 0)
    }

    private fun withBox(body: (Setup) -> Unit) {
        val dir = createTempDirectory("inbox").toFile()
        try {
            body(Setup(dir))
        } finally {
            dir.deleteRecursively()
        }
    }

    /** 워치 WavWriter가 쓰는 것과 같은 닫힌 조각의 바이트. */
    private fun wavBytes(dir: File, frames: Int): ByteArray {
        val f = File(dir, "source-$frames.wav")
        WavWriter(f, SonarSpec.SAMPLE_RATE).use { it.write(ShortArray(frames) { i -> i.toShort() }) }
        return f.readBytes()
    }

    @Test
    fun needListsMissingFilesAndMarksTheRecordingDoneOnce() = withBox { s ->
        val box = s.box()
        val offer = Offer(rec, listOf(RecFile("part-000.wav", 100), RecFile("rec.json", 9)))
        assertEquals(ReceiveBox.Need(listOf("part-000.wav", "rec.json"), newlyDone = false), box.need(offer))
        File(s.root, rec).mkdirs()
        File(s.root, "$rec/part-000.wav").writeBytes(ByteArray(100))
        File(s.root, "$rec/rec.json").writeBytes(ByteArray(8)) // 크기가 다르면 아직 없는 것이다
        assertEquals(listOf("rec.json"), box.need(offer).missing)
        assertEquals(1, box.filesLeft(rec))
        File(s.root, "$rec/rec.json").writeBytes(ByteArray(9))
        assertEquals(ReceiveBox.Need(emptyList(), newlyDone = true), box.need(offer))
        assertEquals(ReceiveBox.Need(emptyList(), newlyDone = false), box.need(offer)) // 같은 요청을 다시 받아도 한 번만 센다
        assertEquals(1, s.box().doneCount)
    }

    @Test
    fun beginSkipsASameSizeFileAndClearsAnOldPart() = withBox { s ->
        val box = s.box()
        File(s.root, rec).mkdirs()
        File(s.root, "$rec/rec.json").writeBytes(ByteArray(9))
        assertNull(box.begin(FileRef(rec, "rec.json", 9), "A")) // 같은 크기: 받지 않는다(HAVE)
        File(s.root, "$rec/part-000.wav.part").writeBytes(ByteArray(500) { 1 }) // 지난번에 끊긴 조각
        val part = box.begin(FileRef(rec, "part-000.wav", 1_004), "B")
        assertNotNull(part)
        assertEquals(File(s.root, "$rec/part-000.wav.part"), part)
        assertFalse(part.exists()) // receiveFile은 자르지 않고 쓰므로 먼저 지운다
        assertEquals(ReceiveBox.Active(FileRef(rec, "part-000.wav", 1_004), part), box.active())
    }

    @Test
    fun finishChecksTheSizeAndHeaderThenRenames() = withBox { s ->
        val box = s.box()
        val bytes = wavBytes(s.dir, 480)
        val ref = FileRef(rec, "part-000.wav", bytes.size.toLong())
        box.begin(ref, "A")!!.writeBytes(bytes)
        assertEquals(TransferProtocol.CODE_OK, box.finish(ref, "A"))
        assertContentEquals(bytes, File(s.root, "$rec/part-000.wav").readBytes())
        assertFalse(File(s.root, "$rec/part-000.wav.part").exists())
        assertNull(box.active())
        // 잘린 조각: 지우고 BAD_SIZE, 최종 이름은 그대로
        val ref2 = FileRef(rec, "part-001.wav", bytes.size.toLong())
        box.begin(ref2, "B")!!.writeBytes(bytes.copyOf(bytes.size - 2))
        assertEquals(TransferProtocol.CODE_BAD_SIZE, box.finish(ref2, "B"))
        assertFalse(File(s.root, "$rec/part-001.wav").exists())
        assertFalse(File(s.root, "$rec/part-001.wav.part").exists())
        // 크기는 맞는데 WAV 머리말이 아님: BAD_WAV
        val ref3 = FileRef(rec, "part-002.wav", bytes.size.toLong())
        box.begin(ref3, "C")!!.writeBytes(ByteArray(bytes.size) { 7 })
        assertEquals(TransferProtocol.CODE_BAD_WAV, box.finish(ref3, "C"))
        // WAV가 아닌 파일(센서 기록 등)은 크기만 본다
        val ref4 = FileRef(rec, "imu-000.csv", 5)
        box.begin(ref4, "D")!!.writeText("1,2,3")
        assertEquals(TransferProtocol.CODE_OK, box.finish(ref4, "D"))
    }

    @Test
    fun lateEventsOfAnOlderChannelForTheSamePathAreIgnored() = withBox { s ->
        val box = s.box()
        val bytes = wavBytes(s.dir, 100)
        val ref = FileRef(rec, "part-000.wav", bytes.size.toLong())
        box.begin(ref, "old")
        val part = box.begin(ref, "new")!! // 워치가 같은 파일을 새 채널로 다시 보냄
        part.writeBytes(bytes)
        assertNull(box.finish(ref, "old")) // 앞 채널의 늦은 입력 닫힘
        assertFalse(box.abandon("old")) // 앞 채널의 늦은 채널 닫힘
        assertTrue(part.exists())
        assertEquals(TransferProtocol.CODE_OK, box.finish(ref, "new"))
        assertTrue(File(s.root, "$rec/part-000.wav").isFile)
    }

    @Test
    fun afterTheAppProcessRestartsFinishFindsThePartByItsPath() = withBox { s ->
        val bytes = wavBytes(s.dir, 100)
        val ref = FileRef(rec, "part-000.wav", bytes.size.toLong())
        s.box().begin(ref, "A")!!.writeBytes(bytes) // Play 서비스는 프로세스가 죽어도 계속 쓴다
        val fresh = s.box() // 다시 뜬 프로세스: 받는 중인 채널을 모른다
        assertNull(fresh.active())
        assertEquals(TransferProtocol.CODE_OK, fresh.finish(ref, "A-again"))
        assertTrue(File(s.root, "$rec/part-000.wav").isFile)
        // 끊긴 채 다시 뜬 경우는 지우고 BAD_SIZE
        val ref2 = FileRef(rec, "part-001.wav", bytes.size.toLong())
        s.box().begin(ref2, "B")!!.writeBytes(bytes.copyOf(10))
        assertEquals(TransferProtocol.CODE_BAD_SIZE, s.box().finish(ref2, "B"))
        assertFalse(File(s.root, "$rec/part-001.wav.part").exists())
    }

    @Test
    fun abandonDeletesOnlyThePartOfTheActiveChannel() = withBox { s ->
        val box = s.box()
        val part = box.begin(FileRef(rec, "part-000.wav", 1_000), "A")!!
        part.writeBytes(ByteArray(300))
        assertTrue(box.abandon("A"))
        assertFalse(part.exists())
        assertNull(box.active())
        assertFalse(box.abandon("A"))
    }
}
