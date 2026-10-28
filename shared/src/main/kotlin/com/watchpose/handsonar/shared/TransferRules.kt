package com.watchpose.handsonar.shared

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 무엇을 어떤 차례로 보내고, 받은 것이 맞는지(전송 설계 §2). 워치(보낼 녹음·파일)와 폰(없는 파일·받은 파일 확인)이 같이 쓴다.
 */
object TransferRules {
    const val REC_JSON = "rec.json"
    private val PART = Regex("""part-(\d{3,})\.wav""")

    /** 폰으로 보낼 파일 이름인가: `rec.json`과 `part-NNN.wav`. 쓰다 남은 `rec.json.tmp`, 받는 중인 `.part`는 아니다. */
    fun isTransferable(name: String): Boolean = name == REC_JSON || PART.matches(name)

    /**
     * 녹음 폴더의 파일 중 보낼 것과 그 차례: 조각을 번호 순으로 먼저, `rec.json`을 마지막에.
     * PC 도구는 `rec.json`이 있는 폴더만 녹음으로 보므로(`hs_io.recording_folders`) 도중에 끊긴 사본이 녹음으로 잡히지 않는다.
     */
    fun transferable(files: List<RecFile>): List<RecFile> =
        files.filter { isTransferable(it.name) }.sortedWith(compareBy<RecFile>({ it.name == REC_JSON }, { partNumber(it.name) }))

    /** 보낼 녹음(오래된 것부터): 녹음 이름 규칙에 맞고, 아직 "보냄"이 아니고([sent]), 지금 녹음 중인 폴더([live])가 아닌 것. */
    fun pending(folders: Collection<String>, sent: Set<String>, live: String?): List<String> =
        folders.filter { TransferProtocol.isRecordingName(it) && it !in sent && it != live }.sorted()

    /** [offer]의 파일 중 [local](폰의 이름 → 크기)에 없거나 크기가 다른 것의 이름(목록 차례 그대로). 비면 그 녹음을 다 받은 것이다. */
    fun missing(offer: List<RecFile>, local: Map<String, Long>): List<String> =
        offer.filter { local[it.name] != it.size }.map { it.name }

    /** 받은 파일 확인: 크기가 [expected]와 같고 WAV면 머리말([header], 앞 44바이트)이 [WavShape]에 맞을 때 CODE_OK, 아니면 그 까닭의 코드. */
    fun verify(name: String, size: Long, expected: Long, header: ByteArray): Int = when {
        size != expected -> TransferProtocol.CODE_BAD_SIZE
        name.endsWith(".wav") && !WavShape.ok(header, size) -> TransferProtocol.CODE_BAD_WAV
        else -> TransferProtocol.CODE_OK
    }

    /** 파일의 앞 44바이트(더 짧으면 있는 만큼, 없으면 빈 배열). */
    fun readHeader(file: File): ByteArray {
        if (!file.isFile) return ByteArray(0)
        RandomAccessFile(file, "r").use { f ->
            val bytes = ByteArray(minOf(WavWriter.HEADER_BYTES.toLong(), f.length()).toInt())
            f.readFully(bytes)
            return bytes
        }
    }

    private fun partNumber(name: String): Long = PART.matchEntire(name)?.groupValues?.get(1)?.toLongOrNull() ?: -1L
}

/**
 * 워치 [WavWriter]가 쓰는 44바이트 머리말 모양(전송 설계 §2의 "WAV 머리말 형식"): PCM 16비트 1채널 [SonarSpec.SAMPLE_RATE] Hz.
 * 길이 칸은 닫힌 조각이면 data = 크기 − 44, RIFF = 크기 − 8이고, 앱이 죽어 닫히지 않은 조각이면 data 0, RIFF 36이다. 둘 다 받는다.
 */
object WavShape {
    fun ok(header: ByteArray, fileSize: Long, rate: Int = SonarSpec.SAMPLE_RATE): Boolean {
        if (header.size < WavWriter.HEADER_BYTES || fileSize < WavWriter.HEADER_BYTES) return false
        val b = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        fun tag(at: Int, text: String) = String(header, at, 4, Charsets.US_ASCII) == text
        if (!tag(0, "RIFF") || !tag(8, "WAVE") || !tag(12, "fmt ") || !tag(36, "data")) return false
        val pcm16Mono = b.getInt(16) == 16 && b.getShort(20).toInt() == 1 && b.getShort(22).toInt() == 1 &&
            b.getInt(24) == rate && b.getInt(28) == rate * 2 && b.getShort(32).toInt() == 2 && b.getShort(34).toInt() == 16
        if (!pcm16Mono) return false
        val riff = b.getInt(4).toLong() and 0xFFFF_FFFFL
        val data = b.getInt(40).toLong() and 0xFFFF_FFFFL
        val closed = data == fileSize - WavWriter.HEADER_BYTES && riff == fileSize - 8
        val unclosed = data == 0L && riff == 36L
        return closed || unclosed
    }
}
