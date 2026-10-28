package com.watchpose.handsonar.shared

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NameLogTest {
    @Test
    fun keepsNamesAcrossInstancesAndIgnoresRepeats() {
        val dir = createTempDirectory("names").toFile()
        try {
            val file = File(dir, "sent-recordings.txt")
            val log = NameLog(file)
            assertEquals(emptySet(), log.names())
            assertTrue(log.add("20261028-101500"))
            assertFalse(log.add("20261028-101500"))
            assertTrue(log.add("20261028-120000"))
            assertEquals(setOf("20261028-101500", "20261028-120000"), NameLog(file).names())
            assertEquals("20261028-101500\n20261028-120000\n", file.readText())
            assertFalse(File(dir, "sent-recordings.txt.tmp").exists())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun readsBlankLinesAsNothingAndRejectsLineBreaksInNames() {
        val dir = createTempDirectory("names").toFile()
        try {
            val file = File(dir, "watch-received.txt").apply { writeText("\n20261028-101500\n\n") }
            assertEquals(setOf("20261028-101500"), NameLog(file).names())
            assertFailsWith<IllegalArgumentException> { NameLog(file).add("a\nb") }
            assertFailsWith<IllegalArgumentException> { NameLog(file).add(" ") }
        } finally {
            dir.deleteRecursively()
        }
    }
}
