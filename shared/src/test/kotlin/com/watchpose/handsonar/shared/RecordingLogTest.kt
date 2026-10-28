package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertTrue

class RecordingLogTest {
    private fun log() = RecordingLog(
        app = "0.1.0",
        device = "samsung SM-L320",
        android = "16 (API 36)",
        start = ClockPoint(wallMs = 1_000L, elapsedNs = 5_000L),
        source = "VOICE_RECOGNITION",
        window = SweepWindow.NONE,
        amplitude = 0.5f,
        volumeStream = "music",
        volumeIndex = 10,
        volumeMax = 15,
    )

    @Test
    fun runningLogHasNullStopAndCurrentParts() {
        val l = log()
        l.addRecordClock(48_000, 1_005_000_000, 47_040)
        l.setParts(listOf(PartInfo("part-000.wav", 0, 48_000)))
        val json = l.toJson()
        assertTrue(
            json.startsWith(
                """{"format":1,"app":"0.1.0","device":"samsung SM-L320","android":"16 (API 36)","start":{"wallMs":1000,"elapsedNs":5000},"stop":null,""",
            ),
            json,
        )
        assertTrue(""""recordClockSource":"timestamp","recordClock":[[48000,1005000000,47040]],""" in json, json)
        // 센서 기록을 붙이지 않은 녹음(원격 시작 sensors=false)은 sensors가 null이다
        assertTrue(json.endsWith(""""parts":[{"file":"part-000.wav","firstFrame":0,"frames":48000}],"sensors":null}"""), json)
    }

    @Test
    fun finishedLogHasStopReasonAndAllLists() {
        val l = log()
        l.addPlayClock(4_800, 6_000)
        l.addUnderrun(7_000, 2)
        l.useReadTimeClock()
        l.finish(ClockPoint(2_000L, 9_000L), StopReason.STORAGE, emptyList())
        val json = l.toJson()
        assertTrue(""""stop":{"wallMs":2000,"elapsedNs":9000,"reason":"storage"},""" in json, json)
        assertTrue(""""audio":{"rate":48000,"bits":16,"channels":1,"source":"VOICE_RECOGNITION"},""" in json, json)
        assertTrue(""""tx":{"f0Hz":18000.0,"bandwidthHz":3000.0,"samples":600,"window":"none","amplitude":0.5},""" in json, json)
        // 녹음 볼륨 "그대로": 고른 값·되돌린 값이 없고 원래 볼륨은 녹음 동안 볼륨과 같다
        assertTrue(""""volume":{"stream":"music","index":10,"max":15,"choice":null,"original":10,"restored":null,"changes":[]},""" in json, json)
        assertTrue(""""recordClockSource":"read","recordClock":[],""" in json, json)
        assertTrue(""""playClock":[[4800,6000]],"underruns":[[7000,2]],"parts":[],"sensors":null}""" in json, json)
    }

    @Test
    fun timerStopIsWrittenAsTimer() {
        val l = log()
        l.finish(ClockPoint(3_000L, 8_000L), StopReason.TIMER, emptyList())
        assertTrue(""""stop":{"wallMs":3000,"elapsedNs":8000,"reason":"timer"},""" in l.toJson(), l.toJson())
    }

    @Test
    fun chosenRecordVolumeHasOriginalRestoredAndChanges() {
        val l = RecordingLog(
            app = "0.1.0",
            device = "samsung SM-L320",
            android = "16 (API 36)",
            start = ClockPoint(wallMs = 1_000L, elapsedNs = 5_000L),
            source = "VOICE_RECOGNITION",
            window = SweepWindow.HANN,
            amplitude = 0.5f,
            volumeStream = "music",
            volumeIndex = 13,
            volumeMax = 15,
            volumeChoice = 13,
            volumeOriginal = 7,
        )
        l.addVolumeChange(6_000L, 12) // 녹음 중 사람이 볼륨을 바꿈
        assertTrue(""""volume":{"stream":"music","index":13,"max":15,"choice":13,"original":7,"restored":null,"changes":[[6000,12]]},""" in l.toJson(), l.toJson())
        l.setVolumeRestored(7)
        assertTrue(""""choice":13,"original":7,"restored":7,"changes":[[6000,12]]},""" in l.toJson(), l.toJson())
    }

    @Test
    fun sensorReportIsWrittenLast() {
        val l = log()
        l.setSensors(SensorReport(emptyList(), listOf(SensorKind.MAG), listOf(SensorPart("sensors-000.csv", 4)), mapOf(SensorKind.ACCEL to 4L), null))
        val json = l.toJson()
        assertTrue(
            json.endsWith(
                ""","parts":[],"sensors":{"columns":"t_ns,recv_ns,sensor,x,y,z,accuracy","list":[],"missing":["mag"],""" +
                    """"parts":[{"file":"sensors-000.csv","rows":4}],"counts":{"acc":4},"error":null}}""",
            ),
            json,
        )
    }
}
