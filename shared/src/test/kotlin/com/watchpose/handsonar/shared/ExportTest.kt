package com.watchpose.handsonar.shared

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.util.zip.ZipInputStream
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class ExportTest {
    @Test
    fun zipKeepsThePcLayoutAndLeavesOutPartialFiles() {
        val dir = createTempDirectory("export").toFile()
        try {
            val session = File(dir, "sessions/20261027-235412").apply { mkdirs() }
            File(session, "meta.json").writeText("{}")
            File(session, "landmarks.csv").writeText("frameNs\n1\n")
            File(session, "meta.json.tmp").writeText("x")
            val rec = File(dir, "watch-recordings/20261027-235407").apply { mkdirs() }
            File(rec, "part-000.wav").writeBytes(ByteArray(100_000) { (it % 251).toByte() })
            File(rec, "part-001.wav.part").writeBytes(ByteArray(7))
            File(rec, "rec.json").writeText("{\"format\":1}")
            val folders = listOf(ExportRules.folder(ExportRules.SESSIONS, session), ExportRules.folder(ExportRules.RECORDINGS, rec))
            assertEquals(listOf("phone/sessions/20261027-235412", "watch/recordings/20261027-235407"), folders.map { it.prefix })
            assertEquals(listOf("landmarks.csv", "meta.json"), folders[0].files.map { it.name })
            assertEquals(listOf("part-000.wav", "rec.json"), folders[1].files.map { it.name })

            val out = ByteArrayOutputStream()
            val total = folders.sumOf { it.bytes }
            var last = 0L
            ExportZip.write(out, folders) { done, all ->
                assertEquals(total, all)
                last = done
            }
            assertEquals(total, last)
            val entries = linkedMapOf<String, ByteArray>()
            ZipInputStream(ByteArrayInputStream(out.toByteArray())).use { zip ->
                generateSequence { zip.nextEntry }.forEach { entries[it.name] = zip.readBytes() }
            }
            assertEquals(
                listOf(
                    "phone/sessions/20261027-235412/landmarks.csv",
                    "phone/sessions/20261027-235412/meta.json",
                    "watch/recordings/20261027-235407/part-000.wav",
                    "watch/recordings/20261027-235407/rec.json",
                ),
                entries.keys.toList(),
            )
            assertContentEquals(File(rec, "part-000.wav").readBytes(), entries.getValue("watch/recordings/20261027-235407/part-000.wav"))
            assertEquals("{}", String(entries.getValue("phone/sessions/20261027-235412/meta.json"), Charsets.UTF_8))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun onlyNewSkipsFoldersWhoseCountSizeAndTimeAreUnchanged() {
        val a = ExportFolder("phone/sessions/a", File("a"), listOf(ExportFile("meta.json", 10, 1_000), ExportFile("video.mp4", 500, 2_000)))
        val b = ExportFolder("watch/recordings/b", File("b"), listOf(ExportFile("rec.json", 9, 3_000)))
        val empty = ExportFolder("watch/recordings/c", File("c"), emptyList())
        assertEquals("2:510:2000", a.signature)
        val before = mapOf(a.prefix to a.signature, b.prefix to "1:8:3000")
        assertEquals(listOf(b), ExportRules.select(listOf(a, b, empty), before, onlyNew = true))
        assertEquals(listOf(a, b), ExportRules.select(listOf(a, b, empty), before, onlyNew = false))
        val touched = a.copy(files = listOf(ExportFile("meta.json", 10, 4_000), ExportFile("video.mp4", 500, 2_000)))
        assertEquals(listOf(touched), ExportRules.select(listOf(touched), before, onlyNew = true))
    }

    @Test
    fun zipNameUsesLocalTime() {
        val wallMs = Instant.parse("2026-10-28T05:30:05Z").toEpochMilli()
        assertEquals("hand-sonar-20261028-143005.zip", ExportRules.zipName(wallMs, ZoneId.of("Asia/Seoul")))
    }
}
