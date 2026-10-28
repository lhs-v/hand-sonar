package com.watchpose.handsonar.shared

/** 녹음 폴더 안 파일 하나와 그 크기(바이트). */
data class RecFile(val name: String, val size: Long)

/** 녹음 하나의 목록(워치 → 폰 목록 요청의 몸통): 녹음 이름과 보낼 파일들. */
data class Offer(val recording: String, val files: List<RecFile>)

/** 채널 경로 하나가 가리키는 파일: 녹음 이름, 파일 이름, 워치에서 잰 크기. */
data class FileRef(val recording: String, val file: String, val size: Long)

/**
 * 워치 → 폰 녹음 전송 규약(전송 설계 §2). 두 앱이 같은 값을 쓰도록 capability·경로·결과 코드·몸통 형식을 여기에만 둔다.
 *
 * 1. 워치는 폰 앱이 선언한 [CAPABILITY]로 폰을 찾는다(CapabilityClient, 닿는 노드만).
 * 2. 녹음마다 [OFFER_PATH]로 목록 요청을 보낸다(MessageClient.sendRequest, 몸통 [encodeOffer]).
 *    폰은 아직 없거나 크기가 다른 파일 이름을 답한다([encodeNeed]).
 * 3. 워치는 그 파일마다 채널 하나를 열어([channelPath]) 처음부터 끝까지 흘려 보낸다. 폰은 `.part`로 받아 크기와 WAV 머리말을 확인한 뒤
 *    원래 이름으로 바꾸고, 결과 코드(CODE_…)로 채널을 닫는다. 이어 받기는 없다: 끊기면 그 파일을 처음부터 다시 보낸다.
 * 4. 다 보낸 뒤 같은 목록 요청을 한 번 더 보낸다. 빈 답이 그 녹음을 다 받았다는 폰의 확인이고, 워치는 그때 그 녹음을 "보냄"으로 적는다.
 *
 * 채널 경로 끝에 크기를 붙이는 까닭: 폰은 채널 하나만 보고도(목록 요청 뒤에 앱이 다시 떴어도) 받은 크기를 견줄 수 있다.
 * 이름은 규칙으로 걸러 받는 쪽 폴더 밖으로 나가지 못하게 한다(`..`, `/`).
 */
object TransferProtocol {
    const val CAPABILITY = "hand_sonar_phone"
    const val OFFER_PATH = "/hand-sonar/offer"
    const val CHANNEL_PREFIX = "/hand-sonar/rec/"

    /** 폰이 확인하고 원래 이름으로 바꿨다. 채널을 닫을 때의 코드는 0이 "코드 없음"이라 1부터 쓴다. */
    const val CODE_OK = 1

    /** 같은 크기의 파일이 이미 있어 받지 않았다. */
    const val CODE_HAVE = 2

    /** 받은 크기가 경로의 크기와 다르다(끊김). */
    const val CODE_BAD_SIZE = 3

    /** 크기는 맞는데 WAV 머리말 모양이 아니다. */
    const val CODE_BAD_WAV = 4

    /** 폰에서 파일을 쓰거나 이름을 바꾸지 못했다. */
    const val CODE_IO = 5

    /** 경로를 읽지 못했다. */
    const val CODE_BAD_PATH = 6

    /** 워치가 녹음을 시작해 보내던 파일을 끊었다. */
    const val CODE_CANCELLED = 7

    /** 워치가 오래 못 보내 끊었다. */
    const val CODE_STALLED = 8

    /** 파일 하나의 크기 상한(1 GiB). 1분 조각은 5,760,044바이트다. 경로·목록의 엉뚱한 값을 거른다. */
    const val MAX_FILE_BYTES = 1L shl 30

    /** 목록·답 몸통의 크기 상한. 10분 녹음의 목록은 300바이트쯤이다. */
    private const val MAX_BODY_BYTES = 64 * 1024
    private const val OFFER_HEAD = "hand-sonar-offer 1"
    private const val NEED_HEAD = "hand-sonar-need 1"
    private val RECORDING = Regex("""\d{8}-\d{6}(-\d{1,3})?""")

    /** 워치 녹음 폴더 이름인가(RecorderService의 yyyyMMdd-HHmmss, 같은 초에 겹칠 때를 위해 -숫자도 받는다). */
    fun isRecordingName(name: String): Boolean = RECORDING.matches(name)

    /** `/hand-sonar/rec/<녹음>/<파일>/<바이트 수>`. 이름이 규칙에 맞지 않거나 크기가 범위 밖이면 IllegalArgumentException. */
    fun channelPath(recording: String, file: String, size: Long): String {
        require(isRecordingName(recording)) { "녹음 이름이 아님: $recording" }
        require(TransferRules.isTransferable(file)) { "보낼 파일 이름이 아님: $file" }
        require(size in 0..MAX_FILE_BYTES) { "크기가 범위 밖: $size" }
        return "$CHANNEL_PREFIX$recording/$file/$size"
    }

    /** [channelPath]를 거꾸로 읽는다. 규칙에 맞지 않으면(다른 경로, `..`, 칸 수, 이상한 크기) null. */
    fun parseChannelPath(path: String): FileRef? {
        if (!path.startsWith(CHANNEL_PREFIX)) return null
        val parts = path.substring(CHANNEL_PREFIX.length).split('/')
        if (parts.size != 3) return null
        val size = sizeOf(parts[2]) ?: return null
        if (!isRecordingName(parts[0]) || !TransferRules.isTransferable(parts[1])) return null
        return FileRef(parts[0], parts[1], size)
    }

    /** 목록 요청 몸통: `hand-sonar-offer 1` / 녹음 이름 / `파일 크기` 줄들. 줄마다 `\n`으로 끝난다(UTF-8). */
    fun encodeOffer(offer: Offer): ByteArray = buildString {
        append(OFFER_HEAD).append('\n')
        append(offer.recording).append('\n')
        for (f in offer.files) append(f.name).append(' ').append(f.size).append('\n')
    }.toByteArray(Charsets.UTF_8)

    /** [encodeOffer]를 거꾸로 읽는다. 머리 줄·이름·크기가 틀리거나, 파일이 없거나, 같은 이름이 두 번이면 null. */
    fun decodeOffer(bytes: ByteArray?): Offer? {
        val lines = lines(bytes) ?: return null
        if (lines.size < 3 || lines[0] != OFFER_HEAD || !isRecordingName(lines[1])) return null
        val files = lines.drop(2).map { line ->
            val cut = line.lastIndexOf(' ')
            if (cut <= 0) return null
            val name = line.substring(0, cut)
            val size = sizeOf(line.substring(cut + 1)) ?: return null
            if (!TransferRules.isTransferable(name)) return null
            RecFile(name, size)
        }
        if (files.distinctBy { it.name }.size != files.size) return null
        return Offer(lines[1], files)
    }

    /** 목록 요청의 답 몸통: `hand-sonar-need 1` / 폰에 아직 없는 파일 이름 줄들. 이름이 없으면 그 녹음을 다 받았다는 확인이다. */
    fun encodeNeed(names: List<String>): ByteArray = buildString {
        append(NEED_HEAD).append('\n')
        for (n in names) append(n).append('\n')
    }.toByteArray(Charsets.UTF_8)

    /** [encodeNeed]를 거꾸로 읽는다. 머리 줄이 틀리거나 보낼 파일 이름이 아닌 줄, 같은 이름이 있으면 null. */
    fun decodeNeed(bytes: ByteArray?): List<String>? {
        val lines = lines(bytes) ?: return null
        if (lines.isEmpty() || lines[0] != NEED_HEAD) return null
        val names = lines.drop(1)
        if (names.any { !TransferRules.isTransferable(it) } || names.toSet().size != names.size) return null
        return names
    }

    /** 결과 코드를 화면용 짧은 글로. */
    fun describe(code: Int): String = when (code) {
        CODE_OK -> "확인됨"
        CODE_HAVE -> "이미 있음"
        CODE_BAD_SIZE -> "받은 크기가 다름"
        CODE_BAD_WAV -> "WAV 머리말이 이상함"
        CODE_IO -> "폰에서 쓰지 못함"
        CODE_BAD_PATH -> "경로를 못 읽음"
        CODE_CANCELLED -> "녹음을 시작해 끊음"
        CODE_STALLED -> "오래 멈춰 끊음"
        else -> "알 수 없는 코드 $code"
    }

    /** 0 이상 [MAX_FILE_BYTES] 이하의 십진수(부호·앞 0 없이)만 받는다. */
    private fun sizeOf(text: String): Long? = text.toLongOrNull()?.takeIf { it in 0..MAX_FILE_BYTES && it.toString() == text }

    /** 몸통을 줄로 나눈다. 끝이 `\n`이 아니거나(잘림) `\r`이 있거나 너무 크면 null. */
    private fun lines(bytes: ByteArray?): List<String>? {
        if (bytes == null || bytes.size > MAX_BODY_BYTES) return null
        val text = String(bytes, Charsets.UTF_8)
        if (!text.endsWith('\n') || '\r' in text) return null
        return text.dropLast(1).split('\n')
    }
}
