package com.watchpose.handsonar.shared

import java.io.File
import java.io.IOException
import java.io.Writer
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SensorLogTest {
    private fun inTemp(block: (File) -> Unit) {
        val dir = createTempDirectory("sensors").toFile()
        try {
            block(dir)
        } finally {
            dir.deleteRecursively()
        }
    }

    private val xyz = floatArrayOf(1f, 2f, 3f)

    /** [failAt]번째 쓰기에서 [error]를 던지는 Writer. 머리줄은 쓰기 두 번(머리줄, 줄바꿈), 줄은 한 번씩이다. */
    private class FailingWriter(private val failAt: Int, private val error: Exception) : Writer() {
        private var writes = 0

        override fun write(cbuf: CharArray, off: Int, len: Int) {
            if (++writes == failAt) throw error
        }

        override fun flush() = Unit

        override fun close() = Unit
    }

    @Test
    fun plansListWatchAndPhoneSensors() {
        assertEquals(listOf("acc", "gyro", "mag", "grav"), SensorPlan.WATCH.map { it.kind.id })
        assertTrue(SensorPlan.WATCH.all { it.periodUs == 10_000 })
        assertEquals(listOf("acc" to 10_000, "gyro" to 10_000, "light" to 200_000), SensorPlan.PHONE.map { it.kind.id to it.periodUs })
        assertEquals(listOf(1, 4, 2, 9, 5), SensorKind.entries.map { it.androidType }) // android.hardware.Sensor.TYPE_* 값
        assertEquals("sensors" to "phone_sensors", SensorPlan.WATCH_BASE to SensorPlan.PHONE_BASE)
    }

    @Test
    fun rowHasBothTimesKindValuesAndAccuracy() {
        val sb = StringBuilder()
        SensorCsv.appendRow(sb, 1_000L, 1_500L, SensorKind.ACCEL, floatArrayOf(0.5f, -9.75f, 1.0E-4f), 3)
        SensorCsv.appendRow(sb, 2_000L, 2_100L, SensorKind.LIGHT, floatArrayOf(120f, 7f, 8f), -1) // 밝기는 x만
        SensorCsv.appendRow(sb, 3_000L, 3_100L, SensorKind.GYRO, floatArrayOf(1f), 0) // 기기가 값을 덜 주면 그 칸은 빈칸
        assertEquals("1000,1500,acc,0.5,-9.75,1.0E-4,3\n2000,2100,light,120.0,,,-1\n3000,3100,gyro,1.0,,,0\n", sb.toString())
    }

    @Test
    fun rowsSplitIntoPartsByReceiveTimeAndEachPartHasAHeader() = inTemp { dir ->
        val w = SensorCsvWriter(dir, "sensors", partNs = 1_000)
        w.add(SensorKind.ACCEL, 10, 100, xyz, 3) // 첫 조각을 받은 시각 100에 연다
        w.add(SensorKind.GYRO, 20, 600, floatArrayOf(4f, 5f, 6f), 3)
        w.add(SensorKind.ACCEL, 30, 1_100, floatArrayOf(7f, 8f, 9f), 2) // 연 지 1,000 ns: 새 조각
        val expected = listOf(SensorPart("sensors-000.csv", 2), SensorPart("sensors-001.csv", 1))
        assertEquals(expected, w.parts) // 쓰는 중인 조각도 지금까지의 줄 수로 나온다
        w.close()
        assertEquals(expected, w.parts)
        assertEquals(mapOf(SensorKind.ACCEL to 2L, SensorKind.GYRO to 1L), w.counts)
        assertNull(w.error)
        assertEquals(listOf(SensorCsv.HEADER, "10,100,acc,1.0,2.0,3.0,3", "20,600,gyro,4.0,5.0,6.0,3"), File(dir, "sensors-000.csv").readLines())
        assertEquals(listOf(SensorCsv.HEADER, "30,1100,acc,7.0,8.0,9.0,2"), File(dir, "sensors-001.csv").readLines())
    }

    @Test
    fun bufferGoesToTheFileOncePerFlushPeriodWhileRecording() = inTemp { dir ->
        val w = SensorCsvWriter(dir, "s", flushNs = 1_000)
        val file = File(dir, "s-000.csv")
        w.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        w.add(SensorKind.ACCEL, 1, 999, xyz, 3)
        assertEquals(0L, file.length()) // 아직 버퍼에만 있다
        w.add(SensorKind.ACCEL, 2, 1_000, xyz, 3)
        assertEquals(4, file.readLines().size) // 머리줄 + 3줄: 닫기 전에도 파일에 있다(앱이 죽어도 남는다)
        w.close()
    }

    @Test
    fun noRowsMeansNoFileAndRowsAfterCloseAreDropped() = inTemp { dir ->
        val w = SensorCsvWriter(dir, "s")
        w.close()
        w.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        assertEquals(emptyList(), w.parts)
        assertEquals(emptyMap(), w.counts)
        assertEquals(0, dir.listFiles()!!.size)
    }

    @Test
    fun openFailureIsRecordedAndLaterRowsAreDroppedWithoutThrowing() = inTemp { dir ->
        val w = SensorCsvWriter(File(dir, "missing/dir"), "s")
        w.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        w.add(SensorKind.ACCEL, 1, 1, xyz, 3)
        val error = w.error
        assertNotNull(error)
        assertTrue(error.startsWith("센서 기록 쓰기 실패"), error)
        assertEquals(emptyList(), w.parts)
        assertEquals(emptyMap(), w.counts)
        w.close()
    }

    @Test
    fun writeFailureMidPartKeepsThePartListedAndDropsLaterRows() = inTemp { dir ->
        // 머리줄 둘과 첫 줄은 되고 둘째 줄에서 디스크 오류: 그 조각은 쓴 줄 수로 남고, 뒤 줄은 버리며, 예외는 밖으로 나가지 않는다
        val w = SensorCsvWriter(dir, "s", openWriter = { _, _ -> FailingWriter(failAt = 4, error = IOException("디스크가 가득 참")) })
        w.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        w.add(SensorKind.GYRO, 1, 1, xyz, 3)
        w.add(SensorKind.ACCEL, 2, 2, xyz, 3)
        assertEquals("센서 기록 쓰기 실패: 디스크가 가득 참", w.error)
        assertEquals(listOf(SensorPart("s-000.csv", 1)), w.parts)
        assertEquals(mapOf(SensorKind.ACCEL to 1L), w.counts)
        w.close()
        assertEquals(listOf(SensorPart("s-000.csv", 1)), w.parts)
    }

    @Test
    fun anyExceptionWhileWritingIsRecordedNotThrown() = inTemp { dir ->
        // IOException 밖의 예외도 센서 스레드를 죽이지 않는다(앱과 함께 소리 녹음이 멈추지 않게)
        val w = SensorCsvWriter(dir, "s", openWriter = { _, _ -> FailingWriter(failAt = 3, error = IllegalStateException("뜻밖의 오류")) })
        w.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        assertEquals("센서 기록 쓰기 실패: 뜻밖의 오류", w.error)
        assertEquals(listOf(SensorPart("s-000.csv", 0)), w.parts)
        w.close()
        val broken = SensorCsvWriter(dir, "t", openWriter = { _, _ -> throw SecurityException("열 수 없음") })
        broken.add(SensorKind.ACCEL, 0, 0, xyz, 3)
        assertEquals("센서 기록 쓰기 실패: 열 수 없음", broken.error)
        assertEquals(emptyList(), broken.parts)
    }

    @Test
    fun reportJsonListsSensorsMissingKindsOtherSensorsPartsCountsAndError() {
        val acc = SensorInfo(SensorKind.ACCEL, 1, "LSM6DSV Accelerometer", "STM", 1, 10_000, 10_000, 160_000, 0.25f, 78.5f, 300, 0, false)
        val light = SensorInfo(SensorKind.LIGHT, 5, "Light", "x", 2, 200_000, 0, 0, Float.NaN, 60_000f, 0, 1, false)
        val otherGyro = SensorInfo(SensorKind.GYRO, 4, "Corrected Gyroscope Sensor", "AOSP", 1, 0, 0, 0, 0.5f, 34.5f, 0, 0, false)
        val report = SensorReport(
            sensors = listOf(acc, light),
            missing = listOf(SensorKind.MAG),
            parts = listOf(SensorPart("sensors-000.csv", 5)),
            counts = mapOf(SensorKind.ACCEL to 5L),
            error = "센서 등록 실패: gyro",
            others = listOf(otherGyro),
        )
        assertEquals(
            """{"columns":"t_ns,recv_ns,sensor,x,y,z,accuracy","list":[""" +
                """{"kind":"acc","type":1,"name":"LSM6DSV Accelerometer","vendor":"STM","version":1,"periodUs":10000,"minDelayUs":10000,""" +
                """"maxDelayUs":160000,"maxRateHz":100.0,"resolution":0.25,"maxRange":78.5,"fifoMax":300,"reportingMode":0,"wakeUp":false},""" +
                """{"kind":"light","type":5,"name":"Light","vendor":"x","version":2,"periodUs":200000,"minDelayUs":0,""" +
                """"maxDelayUs":0,"maxRateHz":null,"resolution":null,"maxRange":60000.0,"fifoMax":0,"reportingMode":1,"wakeUp":false}],""" +
                """"missing":["mag"],""" +
                """"others":[{"kind":"gyro","type":4,"name":"Corrected Gyroscope Sensor","vendor":"AOSP","version":1,"periodUs":0,"minDelayUs":0,""" +
                """"maxDelayUs":0,"maxRateHz":null,"resolution":0.5,"maxRange":34.5,"fifoMax":0,"reportingMode":0,"wakeUp":false}],""" +
                """"parts":[{"file":"sensors-000.csv","rows":5}],"counts":{"acc":5},"error":"센서 등록 실패: gyro"}""",
            JsonOut.write(report.toMap()),
        )
    }
}
