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
        assertTrue(""""parts":[{"file":"part-000.wav","firstFrame":0,"frames":48000}]}""" in json, json)
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
        assertTrue(""""volume":{"stream":"music","index":10,"max":15},""" in json, json)
        assertTrue(""""recordClockSource":"read","recordClock":[],""" in json, json)
        assertTrue(""""playClock":[[4800,6000]],"underruns":[[7000,2]],"parts":[]}""" in json, json)
    }

    @Test
    fun timerStopIsWrittenAsTimer() {
        val l = log()
        l.finish(ClockPoint(3_000L, 8_000L), StopReason.TIMER, emptyList())
        assertTrue(""""stop":{"wallMs":3000,"elapsedNs":8000,"reason":"timer"},""" in l.toJson(), l.toJson())
    }
}
