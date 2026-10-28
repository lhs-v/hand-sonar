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

    /** 기기 사이를 오가도 되는 파일 이름: 영문·숫자로 시작하고 영문·숫자·`.`·`_`·`-`만, 100자까지. 경로 구분자·`..`·숨김 파일이 될 수 없다. */
    private val SAFE_NAME = Regex("""[A-Za-z0-9][A-Za-z0-9._-]{0,99}""")

    /** 쓰다 남은(`.tmp`, 예: `rec.json.tmp`)·받는 중인(`.part`) 파일. 보내지도, 내보내지도, PC로 받지도 않는다. */
    fun isLeftover(name: String): Boolean = name.endsWith(".tmp") || name.endsWith(".part")

    /**
     * 폰으로 보낼 수 있는 파일 이름인가: 남은 파일이 아니고 이름이 안전하다. 이름 목록을 정해 두지 않는다 — 녹음 폴더의 일반 파일은
     * (`part-NNN.wav`, `rec.json`, 앞으로 더해질 센서 기록 등) 모두 보낸다.
     */
    fun isSendable(name: String): Boolean = SAFE_NAME.matches(name) && !isLeftover(name)

    /**
     * 녹음 폴더의 일반 파일 중 보낼 것과 그 차례: 조각(번호 순) → 다른 파일(이름 순) → `rec.json`. 남은 파일과 이름이 안전하지 않은 파일은 뺀다
     * (안전하지 않은 이름은 [unsendable]로 따로 보고 그 녹음을 막는다). PC 도구는 `rec.json`이 있는 폴더만 녹음으로 보므로
     * (`hs_io.recording_folders`) 도중에 끊긴 사본이 녹음으로 잡히지 않는다.
     */
    fun transferable(files: List<RecFile>): List<RecFile> =
        files.filter { isSendable(it.name) }.sortedWith(compareBy<RecFile>({ it.name == REC_JSON }, { partNumber(it.name) }, { it.name }))

    /** 남은 파일이 아닌데 이름이 안전하지 않아 보낼 수 없는 파일. 말없이 빼면 폰 사본이 모자라므로 보내는 쪽이 그 녹음을 막고 알린다. */
    fun unsendable(files: List<RecFile>): List<String> = files.map { it.name }.filter { !isLeftover(it) && !SAFE_NAME.matches(it) }

    /**
     * 보낼 녹음(새것부터): 녹음 이름 규칙에 맞고, 아직 "보냄"이 아니고([sent]), 지금 녹음 중인 폴더([live])가 아닌 것.
     * 새것부터 보내는 까닭: 앱을 처음 깔면 워치에 쌓인 녹음이 모두 목록에 드는데, 방금 녹음한 세션이 그 뒤에 서면 몇 시간 늦는다.
     * 밀린 녹음은 녹음과 녹음 사이에 마저 간다.
     */
    fun pending(folders: Collection<String>, sent: Set<String>, live: String?): List<String> =
        folders.filter { TransferProtocol.isRecordingName(it) && it !in sent && it != live }.sortedDescending()

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

    /** 조각의 번호. 조각이 아니면 가장 큰 값이라 조각들 뒤에 선다. */
    private fun partNumber(name: String): Long = PART.matchEntire(name)?.groupValues?.get(1)?.toLongOrNull() ?: Long.MAX_VALUE
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
