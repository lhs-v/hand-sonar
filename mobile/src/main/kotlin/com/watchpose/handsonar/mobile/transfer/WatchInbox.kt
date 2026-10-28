package com.watchpose.handsonar.mobile.transfer

import android.content.Context
import android.util.Log
import com.watchpose.handsonar.shared.FileRef
import com.watchpose.handsonar.shared.NameLog
import com.watchpose.handsonar.shared.Offer
import com.watchpose.handsonar.shared.ReceiveBox
import com.watchpose.handsonar.shared.TransferProtocol
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

/** 받는 중인 파일(화면의 진행률). [filesLeft]는 그 녹음에서 아직 받을 파일 수(지금 것 포함, 마지막 목록 요청 기준). */
data class Receiving(val recording: String, val file: String, val bytes: Long, val total: Long, val filesLeft: Int)

/** 화면이 보는 받기 상태(전송 설계 §3): 다 받은 녹음 수, 마지막으로 다 받은 때(wallMs, 없으면 0), 받는 중인 파일. */
data class InboxStatus(val received: Int = 0, val lastMs: Long = 0L, val receiving: Receiving? = null)

/**
 * 폰의 워치 녹음 폴더(전송 설계 §3): 앱 외부 저장소 `files/watch-recordings/<녹음>/`, 워치 `recordings/<녹음>/`과 같은 파일.
 * 받는 규칙(`.part`로 받고 확인한 뒤 이름 바꾸기, 늦은 사건 가리기, 프로세스가 다시 떴을 때 경로로 확인)은 shared [ReceiveBox]에 있고
 * (JVM 시험), 이 클래스는 그 둘레의 안드로이드 쪽만 맡는다: 화면 상태([status]), 받는 중 진행률(`.part` 크기를 0.5 s마다),
 * 마지막으로 다 받은 시각(앱 설정 `watch-inbox`), 기록. [WatchReceiverService]의 콜백이 그 서비스의 작업 스레드에서 부른다.
 */
class WatchInbox(context: Context) {
    private val box = ReceiveBox(
        File(context.getExternalFilesDir(null) ?: context.filesDir, ROOT).apply { mkdirs() },
        NameLog(File(context.filesDir, DONE_FILE)),
    )
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mutable = MutableStateFlow(InboxStatus(received = box.doneCount, lastMs = prefs.getLong(KEY_LAST_MS, 0L)))
    val status: StateFlow<InboxStatus> = mutable.asStateFlow()
    private var ticker: Job? = null

    /** 받은 녹음 폴더들의 부모(내보내기가 읽는다). */
    val root: File get() = box.root

    /** 목록 요청의 답: 없거나 크기가 다른 파일 이름. 비면 그 녹음을 다 받은 것이고, 처음이면 수와 마지막 시각을 고친다. */
    fun need(offer: Offer): List<String> {
        val result = box.need(offer)
        if (result.newlyDone) {
            val now = System.currentTimeMillis()
            prefs.edit().putLong(KEY_LAST_MS, now).apply()
            mutable.update { it.copy(received = box.doneCount, lastMs = now) }
        }
        return result.missing
    }

    /** 채널 하나를 받기 시작한다([ReceiveBox.begin]). 받을 `.part`를 돌려주고 진행률을 보이기 시작한다. 같은 크기가 있으면 null. */
    fun begin(ref: FileRef, token: Any): File? {
        val part = box.begin(ref, token) ?: return null
        showReceiving(ref, part)
        return part
    }

    /** 입력이 닫혔다([ReceiveBox.finish]): 결과 코드, 앞 채널의 늦은 사건이면 null. */
    fun finish(ref: FileRef, token: Any): Int? {
        val code = box.finish(ref, token) ?: return null
        if (code == TransferProtocol.CODE_IO) Log.w(TAG, "이름을 못 바꿈: ${ref.recording}/${ref.file}", box.lastError)
        if (box.active() == null) hideReceiving()
        return code
    }

    /** 채널이 끝났다([ReceiveBox.abandon]). */
    fun abandon(token: Any) {
        if (box.abandon(token)) hideReceiving()
    }

    @Synchronized
    private fun showReceiving(ref: FileRef, part: File) {
        val filesLeft = box.filesLeft(ref.recording).coerceAtLeast(1)
        mutable.update { it.copy(receiving = Receiving(ref.recording, ref.file, 0L, ref.size, filesLeft)) }
        ticker?.cancel()
        ticker = scope.launch {
            while (isActive) {
                delay(TICK_MS)
                if (box.active()?.part != part) break
                val bytes = part.length()
                mutable.update { s ->
                    val r = s.receiving
                    if (r != null && r.recording == ref.recording && r.file == ref.file) s.copy(receiving = r.copy(bytes = bytes)) else s
                }
            }
        }
    }

    @Synchronized
    private fun hideReceiving() {
        ticker?.cancel()
        ticker = null
        mutable.update { it.copy(receiving = null) }
    }

    companion object {
        /** 앱 외부 저장소 안 폴더 이름. PC `pull.py`가 같은 이름을 쓴다. */
        const val ROOT = "watch-recordings"
        private const val TAG = "HandSonarInbox"
        private const val DONE_FILE = "watch-received.txt"
        private const val PREFS = "watch-inbox"
        private const val KEY_LAST_MS = "lastMs"
        private const val TICK_MS = 500L
    }
}
