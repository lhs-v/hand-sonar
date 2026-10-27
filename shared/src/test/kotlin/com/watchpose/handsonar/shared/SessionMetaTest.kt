package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SessionMetaTest {
    private val plan = PromptPlanner.plan(seed = 9L, freeMotionSec = 0)

    private fun meta(stats: SessionStats?, endWallMs: Long?, status: String = SessionMeta.STATUS_KEPT) = SessionMeta(
        app = "0.1.0",
        device = "samsung SM-F971N",
        android = "17 (API 37)",
        camera = CameraMeta("3", "inner", 640, 480, 30, "realtime", mirrored = true, rotation = 270),
        hand = "left",
        startWallMs = 1_000L,
        startNs = 2_000L,
        endWallMs = endWallMs,
        endNs = endWallMs?.let { 3_000L },
        plan = plan,
        status = status,
        flags = listOf("suspect-hand"),
        memo = "",
        stats = stats,
        rewear = false,
        video = VideoMeta("video.mp4", 480, 360, 100, ok = true),
        audioFile = "phone_audio.wav",
        audioSource = "UNPROCESSED",
        cancelled = false,
    )

    @Test
    fun startOfSessionHasNullEndAndStats() {
        val json = meta(stats = null, endWallMs = null).toJson()
        assertTrue(json.startsWith("{\"format\":1,\"app\":\"0.1.0\",\"device\":\"samsung SM-F971N\",\"android\":\"17 (API 37)\",\"camera\":{\"id\":\"3\",\"display\":\"inner\""), json)
        assertTrue(json.contains("\"rotation\":270},\"hand\":\"left\",\"start\""), json)
        assertTrue(json.contains("\"start\":{\"wallMs\":1000,\"elapsedNs\":2000},\"end\":null"), json)
        assertTrue(json.contains("\"stats\":null"), json)
        assertTrue(json.contains("\"status\":\"kept\",\"flags\":[\"suspect-hand\"],\"memo\":\"\""), json)
        assertTrue(json.contains("\"version\":\"watchhand-1\",\"seed\":9,\"freeMotionSec\":0,\"promptMs\":2000"), json)
        assertTrue(json.contains("\"segments\":[[" + plan.segments[0].joinToString(",") { it.classId.toString() } + "]"), json)
        assertTrue(json.endsWith("\"audio\":{\"file\":\"phone_audio.wav\",\"source\":\"UNPROCESSED\"},\"cancelled\":false}"), json)
    }

    @Test
    fun finishedSessionHasStatsAndMarkers() {
        val stats = SessionStats(
            frames = 6_000,
            handRatio = 0.95,
            watchHeardRatio = 1.0,
            watchGaps = listOf(longArrayOf(1, 2)),
            startMarker = MarkerMeta(found = true, playNs = 5, micFrame = 10.5, elapsedNs = 7, ratio = 20.0),
            endMarker = null,
            durationSec = 218.5,
        )
        val json = meta(stats, endWallMs = 9_000L, status = SessionMeta.STATUS_TRASH).toJson()
        assertTrue(json.contains("\"end\":{\"wallMs\":9000,\"elapsedNs\":3000}"), json)
        assertTrue(json.contains("\"status\":\"trash\""), json)
        assertTrue(
            json.contains(
                "\"stats\":{\"frames\":6000,\"handRatio\":0.95,\"watchHeardRatio\":1.0,\"watchGaps\":[[1,2]]," +
                    "\"markers\":{\"start\":{\"found\":true,\"playNs\":5,\"micFrame\":10.5,\"elapsedNs\":7,\"ratio\":20.0},\"end\":null}," +
                    "\"durationSec\":218.5}",
            ),
            json,
        )
        assertTrue(json.contains("\"video\":{\"file\":\"video.mp4\",\"width\":480,\"height\":360,\"frames\":100,\"ok\":true}"), json)
    }

    @Test
    fun phoneAudioJson() {
        val clock = listOf(ClockLine(0, 10, 960), ClockLine(48_000, 1_000_000_010, 48_960))
        assertEquals(
            "{\"format\":1,\"file\":\"phone_audio.wav\",\"rate\":48000,\"bits\":16,\"channels\":1,\"source\":\"UNPROCESSED\"," +
                "\"frames\":96000,\"recordClockSource\":\"timestamp\",\"recordClock\":[[0,10,960],[48000,1000000010,48960]]}",
            PhoneAudioJson.write("UNPROCESSED", "timestamp", clock, 96_000),
        )
    }
}
