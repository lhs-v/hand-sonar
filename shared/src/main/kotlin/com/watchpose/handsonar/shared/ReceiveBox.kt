package com.watchpose.handsonar.shared

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * 폰이 워치 녹음을 받는 폴더의 상태 기계(전송 설계 §2·§3). `[root]/<녹음>/<파일>`에 워치와 같은 파일을 두고, 받는 동안만
 * `<파일>.part`를 쓴다. 크기(와 WAV 머리말)를 확인한 뒤에만 원래 이름으로 바꾼다(있던 것은 덮는다). 다 받은 녹음 이름은 [done]에 적는다
 * (녹음 폴더에 표시 파일을 두지 않아 PC로 받는 모양이 워치와 같다). 안드로이드·Play 서비스를 모르므로 JVM에서 시험한다.
 *
 * 한 번에 채널 하나만 받는다(워치가 하나씩 보낸다). 채널은 [begin]의 `token`(Play 서비스의 채널 객체 — 토큰·노드·경로로 견준다)으로 가린다:
 * - 새 채널이 begin하면 앞 채널의 늦은 사건([finish]·[abandon])은 무시한다(같은 경로로 다시 연 채널도 토큰이 다르다).
 * - 받는 중인 채널이 없으면(받는 동안 앱 프로세스가 다시 떠 이 객체가 새로 만들어졌다 — receiveFile은 Play 서비스가 쓰므로 그사이에도
 *   이어진다) [finish]는 경로로 `.part`를 찾아 확인한다.
 * 리스너 서비스의 작업 스레드와 화면의 진행률 읽기가 함께 쓰므로 모든 공개 멤버를 잠근다. [finish]의 기다림은 잠그지 않고 한다.
 */
class ReceiveBox(val root: File, private val done: NameLog, private val settleMs: Long = SETTLE_MS) {
    /** 받는 중인 채널의 파일과 그 `.part`. */
    data class Active(val ref: FileRef, val part: File)

    /** 목록 요청의 답: 없거나 크기가 다른 파일 이름(목록 차례), 이번 요청으로 그 녹음을 처음 다 받았는지. */
    data class Need(val missing: List<String>, val newlyDone: Boolean)

    private val lock = Any()
    private var token: Any? = null
    private var current: Active? = null

    /** 녹음마다 마지막 목록 요청에서 아직 없던 파일(화면의 남은 파일 수). */
    private val left = HashMap<String, MutableSet<String>>()

    /** 다 받은 녹음 수. */
    val doneCount: Int get() = done.names().size

    fun active(): Active? = synchronized(lock) { current }

    /** 그 녹음에서 아직 받을 파일 수(마지막 목록 요청 기준, 확인할 때마다 준다). */
    fun filesLeft(recording: String): Int = synchronized(lock) { left[recording]?.size ?: 0 }

    /** 목록 요청의 답. 비면 그 녹음을 다 받은 것으로 적는다. */
    fun need(offer: Offer): Need = synchronized(lock) {
        val dir = File(root, offer.recording)
        val local = HashMap<String, Long>()
        for (f in offer.files) {
            val file = File(dir, f.name)
            if (file.isFile) local[f.name] = file.length()
        }
        val missing = TransferRules.missing(offer.files, local)
        if (missing.isEmpty()) {
            left.remove(offer.recording)
            Need(missing, newlyDone = done.add(offer.recording))
        } else {
            left[offer.recording] = missing.toMutableSet()
            Need(missing, newlyDone = false)
        }
    }

    /**
     * 채널 하나를 받기 시작한다. 같은 크기의 파일이 이미 있으면 null(받지 않는다 — HAVE). 아니면 받을 `.part`를 돌려준다.
     * Play 서비스의 receiveFile은 열 때 자르지 않으므로(WRITE_ONLY|CREATE) 지난 시도의 바이트가 남지 않게 먼저 지운다.
     */
    fun begin(ref: FileRef, token: Any): File? {
        synchronized(lock) {
            val dir = File(root, ref.recording)
            val target = File(dir, ref.file)
            if (target.isFile && target.length() == ref.size) {
                left[ref.recording]?.remove(ref.file)
                return null
            }
            dir.mkdirs()
            val part = File(dir, ref.file + PART)
            part.delete()
            this.token = token
            current = Active(ref, part)
            return part
        }
    }

    /**
     * 입력이 닫혔다: `.part`를 확인해 맞으면 원래 이름으로 바꾸고 CODE_OK, 아니면 `.part`를 지우고 그 까닭의 코드.
     * 다른 채널이 받는 중이면 null(앞 채널의 늦은 사건). 받는 중인 채널이 없으면 [ref]의 경로로 `.part`를 찾는다(앱 프로세스가 다시 뜸).
     * 마지막 쓰기가 늦게 닿을 수 있어 크기가 모자라면 [settleMs]까지 다시 잰다.
     */
    fun finish(ref: FileRef, token: Any): Int? {
        val part = synchronized(lock) {
            val a = current
            when {
                a == null -> File(File(root, ref.recording), ref.file + PART)
                this.token == token -> a.part
                else -> return null
            }
        }
        var waited = 0L
        while (part.length() < ref.size && waited < settleMs) {
            Thread.sleep(SETTLE_STEP_MS)
            waited += SETTLE_STEP_MS
        }
        synchronized(lock) {
            val mine = current != null
            if (mine && this.token != token) return null
            val size = if (part.isFile) part.length() else -1L
            var code = TransferRules.verify(ref.file, size, ref.size, TransferRules.readHeader(part))
            if (code == TransferProtocol.CODE_OK) {
                code = try {
                    Files.move(part.toPath(), File(part.parentFile, ref.file).toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
                    left[ref.recording]?.remove(ref.file)
                    TransferProtocol.CODE_OK
                } catch (e: IOException) {
                    lastError = e
                    TransferProtocol.CODE_IO
                }
            }
            if (code != TransferProtocol.CODE_OK) part.delete()
            if (mine) clear()
            return code
        }
    }

    /** 채널이 끝났다(끊김·워치가 끊음·우리가 닫음). 받는 중이던 채널이면 `.part`를 지우고 true(이어 받기가 없으므로). */
    fun abandon(token: Any): Boolean {
        synchronized(lock) {
            val a = current ?: return false
            if (this.token != token) return false
            a.part.delete()
            clear()
            return true
        }
    }

    /** 마지막으로 이름을 못 바꾼 까닭(기록용, 없으면 null). */
    @Volatile
    var lastError: IOException? = null
        private set

    private fun clear() {
        token = null
        current = null
    }

    companion object {
        const val PART = ".part"
        const val SETTLE_MS = 2_000L
        private const val SETTLE_STEP_MS = 100L
    }
}
