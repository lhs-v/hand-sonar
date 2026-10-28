package com.watchpose.handsonar.mobile.transfer

import android.content.Context
import android.util.Log
import com.watchpose.handsonar.shared.FileRef
import com.watchpose.handsonar.shared.NameLog
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.TransferProtocol
import com.watchpose.handsonar.shared.TransferRules
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** 받는 중인 파일(화면의 진행률). [filesLeft]는 그 녹음에서 아직 받을 파일 수(지금 것 포함, 마지막 목록 요청 기준). */
data class Receiving(val recording: String, val file: String, val bytes: Long, val total: Long, val filesLeft: Int)

/** 화면이 보는 받기 상태(전송 설계 §3): 다 받은 녹음 수, 마지막으로 다 받은 때(wallMs, 없으면 0), 받는 중인 파일. */
data class InboxStatus(val received: Int = 0, val lastMs: Long = 0L, val receiving: Receiving? = null)

/**
 * 폰의 워치 녹음 폴더(전송 설계 §3): 앱 외부 저장소 `files/watch-recordings/<녹음>/`, 워치 `recordings/<녹음>/`과 같은 파일.
 * 받는 동안만 `<파일>.part`가 있고, 크기(와 WAV 머리말)를 확인한 뒤 원래 이름으로 바꾼다(있던 것은 덮는다).
 * 다 받은 녹음 이름은 앱 내부 저장소의 [NameLog]에 적는다(녹음 폴더에 표시 파일을 두지 않아 PC로 받는 모양이 워치와 같다).
 * [WatchReceiverService]의 콜백이 그 서비스의 작업 스레드에서 부르고, 화면은 [status]만 본다.
 * 한 번에 파일 하나만 받는다(워치가 하나씩 보낸다): 새 채널이 [begin]하면 앞 채널의 늦은 사건([finish]·[abandon])은
 * 채널 객체(`token`, 토큰·노드·경로로 견준다)가 달라 무시한다.
 */
class WatchInbox(context: Context) {
    val root: File = File(context.getExternalFilesDir(null) ?: context.filesDir, ROOT).apply { mkdirs() }
    private val done = NameLog(File(context.filesDir, DONE_FILE))
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val lock = Any()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mutable = MutableStateFlow(InboxStatus(received = done.names().size, lastMs = prefs.getLong(KEY_LAST_MS, 0L)))
    val status: StateFlow<InboxStatus> = mutable.asStateFlow()

    /** 받는 중인 채널 하나. */
    private class Active(val token: Any, val ref: FileRef, val part: File)

    private var active: Active? = null
    private var ticker: Job? = null

    /** 녹음마다 마지막 목록 요청에서 아직 없던 파일(화면의 남은 파일 수). */
    private val left = HashMap<String, MutableSet<String>>()

    /** 목록 요청의 답: [offer]의 파일 중 없거나 크기가 다른 이름. 비면 그 녹음을 다 받은 것으로 적는다(처음이면 수와 마지막 시각을 고친다). */
    fun need(offer: Offer): List<String> = synchronized(lock) {
        val dir = File(root, offer.recording)
        val local = HashMap<String, Long>()
        for (f in offer.files) {
            val file = File(dir, f.name)
            if (file.isFile) local[f.name] = file.length()
        }
        val missing = TransferRules.missing(offer.files, local)
        if (missing.isEmpty()) {
            left.remove(offer.recording)
            if (done.add(offer.recording)) {
                val now = System.currentTimeMillis()
                prefs.edit().putLong(KEY_LAST_MS, now).apply()
                mutable.update { it.copy(received = done.names().size, lastMs = now) }
            }
        } else {
            left[offer.recording] = missing.toMutableSet()
        }
        missing
    }

    /**
     * 채널 하나를 받기 시작한다. 같은 크기의 파일이 이미 있으면 null(받지 않는다). 아니면 받을 `.part`를 돌려준다.
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
            active = Active(token, ref, part)
            val filesLeft = left[ref.recording]?.size?.takeIf { it > 0 } ?: 1
            mutable.update { it.copy(receiving = Receiving(ref.recording, ref.file, 0L, ref.size, filesLeft)) }
            ticker?.cancel()
            ticker = scope.launch {
                while (isActive) {
                    delay(TICK_MS)
                    val bytes = part.length()
                    mutable.update { s ->
                        val r = s.receiving
                        if (r != null && r.recording == ref.recording && r.file == ref.file) s.copy(receiving = r.copy(bytes = bytes)) else s
                    }
                }
            }
            return part
        }
    }

    /**
     * 입력이 닫혔다: 받은 `.part`를 확인해 맞으면 원래 이름으로 바꾸고 CODE_OK, 아니면 `.part`를 지우고 그 까닭의 코드.
     * [token]이 받는 중인 채널이 아니면 null(앞 채널의 늦은 사건). 마지막 쓰기가 늦게 닿을 수 있어 크기가 모자라면 2 s까지 다시 잰다.
     */
    fun finish(token: Any): Int? {
        val a = synchronized(lock) { active?.takeIf { it.token == token } } ?: return null
        var waited = 0L
        while (a.part.length() < a.ref.size && waited < SETTLE_MS) {
            Thread.sleep(SETTLE_STEP_MS)
            waited += SETTLE_STEP_MS
        }
        synchronized(lock) {
            if (active?.token != token) return null
            val size = if (a.part.isFile) a.part.length() else -1L
            var code = TransferRules.verify(a.ref.file, size, a.ref.size, TransferRules.readHeader(a.part))
            if (code == TransferProtocol.CODE_OK) {
                code = try {
                    Files.move(
                        a.part.toPath(),
                        File(a.part.parentFile, a.ref.file).toPath(),
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE,
                    )
                    left[a.ref.recording]?.remove(a.ref.file)
                    TransferProtocol.CODE_OK
                } catch (e: IOException) {
                    Log.w(TAG, "이름을 못 바꿈: ${a.part}", e)
                    TransferProtocol.CODE_IO
                }
            }
            if (code != TransferProtocol.CODE_OK) a.part.delete()
            clearActive()
            return code
        }
    }

    /** 채널이 끝났다(끊김·워치가 끊음·우리가 닫음). 아직 받는 중이던 채널이면 `.part`를 지운다(이어 받기가 없으므로). */
    fun abandon(token: Any) {
        synchronized(lock) {
            val a = active?.takeIf { it.token == token } ?: return
            a.part.delete()
            clearActive()
        }
    }

    private fun clearActive() {
        active = null
        ticker?.cancel()
        ticker = null
        mutable.update { it.copy(receiving = null) }
    }

    companion object {
        /** 앱 외부 저장소 안 폴더 이름. PC `pull.py`가 같은 이름을 쓴다. */
        const val ROOT = "watch-recordings"
        const val PART = ".part"
        private const val TAG = "HandSonarInbox"
        private const val DONE_FILE = "watch-received.txt"
        private const val PREFS = "watch-inbox"
        private const val KEY_LAST_MS = "lastMs"
        private const val TICK_MS = 500L
        private const val SETTLE_MS = 2_000L
        private const val SETTLE_STEP_MS = 100L
    }
}
