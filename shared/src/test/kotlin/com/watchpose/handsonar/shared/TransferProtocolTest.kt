package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class TransferProtocolTest {
    @Test
    fun channelPathCarriesRecordingFileAndSize() {
        val path = TransferProtocol.channelPath("20261028-101500", "part-003.wav", 5_760_044)
        assertEquals("/hand-sonar/rec/20261028-101500/part-003.wav/5760044", path)
        assertEquals(FileRef("20261028-101500", "part-003.wav", 5_760_044), TransferProtocol.parseChannelPath(path))
        assertEquals(FileRef("20261028-101500-2", "rec.json", 0), TransferProtocol.parseChannelPath("/hand-sonar/rec/20261028-101500-2/rec.json/0"))
    }

    @Test
    fun channelPathRejectsOtherPathsEscapesAndOddSizes() {
        val bad = listOf(
            TransferProtocol.OFFER_PATH,
            "/hand-sonar/rec/20261028-101500/part-003.wav", // 크기 없음
            "/hand-sonar/rec/20261028-101500/part-003.wav/12/x",
            "/hand-sonar/rec/../part-003.wav/12",
            "/hand-sonar/rec/20261028-101500/../rec.json/12",
            "/hand-sonar/rec/20261028-101500/notes.txt/12",
            "/hand-sonar/rec/20261028-101500/rec.json.tmp/12",
            "/hand-sonar/rec/20261028-101500/part-003.wav.part/12",
            "/hand-sonar/rec/20261028-101500/part-003.wav/-1",
            "/hand-sonar/rec/20261028-101500/part-003.wav/+12",
            "/hand-sonar/rec/20261028-101500/part-003.wav/012",
            "/hand-sonar/rec/20261028-101500/part-003.wav/${TransferProtocol.MAX_FILE_BYTES + 1}",
            "/hand-sonar/rec/sessions/part-003.wav/12",
            "/other/rec/20261028-101500/part-003.wav/12",
        )
        for (path in bad) assertNull(TransferProtocol.parseChannelPath(path), path)
        assertFailsWith<IllegalArgumentException> { TransferProtocol.channelPath("../x", "rec.json", 1) }
        assertFailsWith<IllegalArgumentException> { TransferProtocol.channelPath("20261028-101500", "../rec.json", 1) }
        assertFailsWith<IllegalArgumentException> { TransferProtocol.channelPath("20261028-101500", "rec.json", -1) }
    }

    @Test
    fun offerAndNeedRoundTrip() {
        val offer = Offer("20261028-101500", listOf(RecFile("part-000.wav", 5_760_044), RecFile("part-001.wav", 1_000_044), RecFile("rec.json", 2_345)))
        val body = TransferProtocol.encodeOffer(offer)
        assertEquals(
            "hand-sonar-offer 1\n20261028-101500\npart-000.wav 5760044\npart-001.wav 1000044\nrec.json 2345\n",
            String(body, Charsets.UTF_8),
        )
        assertEquals(offer, TransferProtocol.decodeOffer(body))
        assertEquals(listOf("part-001.wav", "rec.json"), TransferProtocol.decodeNeed(TransferProtocol.encodeNeed(listOf("part-001.wav", "rec.json"))))
        assertEquals(emptyList(), TransferProtocol.decodeNeed(TransferProtocol.encodeNeed(emptyList())))
    }

    @Test
    fun brokenBodiesAreRejected() {
        fun offer(text: String) = TransferProtocol.decodeOffer(text.toByteArray(Charsets.UTF_8))
        assertNull(TransferProtocol.decodeOffer(null))
        assertNull(offer("hand-sonar-offer 2\n20261028-101500\nrec.json 1\n")) // 모르는 판
        assertNull(offer("hand-sonar-offer 1\n20261028-101500\n")) // 파일 없음
        assertNull(offer("hand-sonar-offer 1\n../x\nrec.json 1\n"))
        assertNull(offer("hand-sonar-offer 1\n20261028-101500\nrec.json\n"))
        assertNull(offer("hand-sonar-offer 1\n20261028-101500\nrec.json -5\n"))
        assertNull(offer("hand-sonar-offer 1\n20261028-101500\nrec.json 1\nrec.json 1\n")) // 같은 이름 둘
        assertNull(offer("hand-sonar-offer 1\n20261028-101500\nrec.json 1")) // 끝 줄바꿈 없음(잘림)
        assertNull(offer("hand-sonar-offer 1\r\n20261028-101500\r\nrec.json 1\r\n"))
        fun need(text: String) = TransferProtocol.decodeNeed(text.toByteArray(Charsets.UTF_8))
        assertNull(TransferProtocol.decodeNeed(null))
        assertNull(need("hand-sonar-offer 1\n"))
        assertNull(need("hand-sonar-need 1\n../rec.json\n"))
        assertNull(need("hand-sonar-need 1\nrec.json\nrec.json\n"))
    }
}
