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
        // 음원을 고르지 않은(옛 방식) 기록은 sourceRequested·sourceFallback이 null이다
        assertTrue(
            """"audio":{"rate":48000,"bits":16,"channels":1,"source":"VOICE_RECOGNITION","sourceRequested":null,"sourceFallback":null},""" in json,
            json,
        )
        assertTrue(""""tx":{"f0Hz":18000.0,"bandwidthHz":3000.0,"samples":600,"window":"none","amplitude":0.5},""" in json, json)
        // 녹음 볼륨 "그대로": 고른 값·되돌린 값이 없고 원래 볼륨은 녹음 동안 볼륨과 같다. 외부 출력·바뀜·건너뜀·남은 기록도 없다
        assertTrue(
            """"volume":{"stream":"music","index":10,"max":15,"choice":null,"original":10,"restored":null,"changes":[],""" +
                """"target":null,"outputs":[],"outputChanges":[],"skipped":null,"restoreSkipped":null,"leftover":null},""" in json,
            json,
        )
        assertTrue(""""recordClockSource":"read","recordClock":[],""" in json, json)
        assertTrue(""""playClock":[[4800,6000]],"underruns":[[7000,2]],"parts":[],"sensors":null}""" in json, json)
    }

    @Test
    fun chosenSourceAndFallbackAreRecorded() {
        // UNPROCESSED를 골랐지만 열리지 않아 VOICE_RECOGNITION으로 녹음함: 기록은 녹음을 연 뒤 실제 음원으로 고친다
        val l = RecordingLog(
            app = "0.1.0",
            device = "samsung SM-L320",
            android = "16 (API 36)",
            start = ClockPoint(wallMs = 1_000L, elapsedNs = 5_000L),
            source = "UNPROCESSED",
            window = SweepWindow.TUKEY,
            amplitude = 0.5f,
            volumeStream = "music",
            volumeIndex = 13,
            volumeMax = 15,
            sourceRequested = RecordSource.UNPROCESSED,
        )
        assertTrue(""""source":"UNPROCESSED","sourceRequested":"unprocessed","sourceFallback":null},""" in l.toJson(), l.toJson())
        l.setSource("VOICE_RECOGNITION", "UNPROCESSED 녹음 초기화 실패")
        assertTrue(
            """"source":"VOICE_RECOGNITION","sourceRequested":"unprocessed","sourceFallback":"UNPROCESSED 녹음 초기화 실패"},""" in l.toJson(),
            l.toJson(),
        )
        assertTrue(""""window":"tukey",""" in l.toJson(), l.toJson())
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
            volumeTarget = 13,
        )
        l.addVolumeChange(6_000L, 12) // 녹음 중 사람이 볼륨을 바꿈
        assertTrue(""""volume":{"stream":"music","index":13,"max":15,"choice":13,"original":7,"restored":null,"changes":[[6000,12]],"target":13,""" in l.toJson(), l.toJson())
        l.setVolumeRestore(7, null)
        assertTrue(""""choice":13,"original":7,"restored":7,"changes":[[6000,12]],"target":13,""" in l.toJson(), l.toJson())
    }

    @Test
    fun externalOutputsSkippedRestoreAndLeftoverAreRecorded() {
        // 블루투스 이어폰이 이어진 채 시작(녹음 볼륨을 걸지 않음), 녹음 중 끊겼다 이어짐, 정지 때 이어져 있어 되돌리기를 미룸,
        // 그 전에 앞 녹음이 남긴 기록은 그사이 바뀌어 지웠음
        val l = RecordingLog(
            app = "0.1.0",
            device = "samsung SM-L320",
            android = "16 (API 36)",
            start = ClockPoint(wallMs = 1_000L, elapsedNs = 5_000L),
            source = "VOICE_RECOGNITION",
            window = SweepWindow.HANN,
            amplitude = 0.5f,
            volumeStream = "music",
            volumeIndex = 9,
            volumeMax = 15,
            volumeChoice = 13,
            volumeOriginal = 9,
            volumeOutputs = listOf("bluetooth_a2dp"),
            volumeSkipped = RecordVolumeRules.SKIP_EXTERNAL,
            volumeLeftover = VolumeRestore(null, RecordVolumeRules.SKIP_CHANGED, VolumePending(7, 13)),
        )
        l.addOutputChange(6_000L, emptyList())
        l.addOutputChange(7_000L, listOf("bluetooth_a2dp"))
        l.setVolumeRestore(null, RecordVolumeRules.SKIP_EXTERNAL)
        assertTrue(
            """"choice":13,"original":9,"restored":null,"changes":[],"target":null,"outputs":["bluetooth_a2dp"],""" +
                """"outputChanges":[[6000,[]],[7000,["bluetooth_a2dp"]]],"skipped":"external_output","restoreSkipped":"external_output",""" +
                """"leftover":{"original":7,"applied":13,"restored":null,"skipped":"changed"}},""" in l.toJson(),
            l.toJson(),
        )
    }

    @Test
    fun sensorReportIsWrittenLast() {
        val l = log()
        l.setSensors(SensorReport(emptyList(), listOf(SensorKind.MAG), listOf(SensorPart("sensors-000.csv", 4)), mapOf(SensorKind.ACCEL to 4L), null))
        val json = l.toJson()
        assertTrue(
            json.endsWith(
                ""","parts":[],"sensors":{"columns":"t_ns,recv_ns,sensor,x,y,z,accuracy","list":[],"missing":["mag"],"others":[],""" +
                    """"parts":[{"file":"sensors-000.csv","rows":4}],"counts":{"acc":4},"error":null}}""",
            ),
            json,
        )
    }
}
