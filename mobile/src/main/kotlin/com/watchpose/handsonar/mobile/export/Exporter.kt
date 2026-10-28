package com.watchpose.handsonar.mobile.export

import android.content.ContentResolver
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.os.Bundle
import android.os.Environment
import android.provider.BaseColumns
import android.provider.MediaStore
import com.watchpose.handsonar.mobile.session.SessionStore
import com.watchpose.handsonar.mobile.transfer.WatchInbox
import com.watchpose.handsonar.shared.ExportRules
import com.watchpose.handsonar.shared.ExportZip
import com.watchpose.handsonar.shared.SessionMeta
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * 기록 화면의 내보내기 상태. [progress]는 0..1(바이트 기준), [message]는 끝난 뒤의 결과나 오류([error]).
 * [lastZip]·[lastMs]는 지난번에 다 쓴 zip 이름과 그때(wallMs, 없으면 null·0) — "새 것만"의 기준이 그 zip이라 확인 창이 보인다.
 */
data class ExportStatus(
    val busy: Boolean = false,
    val progress: Float = 0f,
    val message: String? = null,
    val error: Boolean = false,
    val lastZip: String? = null,
    val lastMs: Long = 0L,
)

/**
 * 보관한 세션과 받은 워치 녹음을 zip 하나로 공용 `Download/HandSonar/`에 저장한다(전송 설계 §5). MediaStore로 쓰므로 저장소 권한이 필요 없다.
 * USB 파일 전송만 되는 PC에서 그 zip을 복사해 `hand-sonar-data`에 풀면 `pull.py`로 받은 것과 같다(zip 안 `phone/sessions/…`, `watch/recordings/…`).
 * "새 것만"이면 지난번 내보낸 폴더 중 파일 수·크기·가장 늦은 수정 시각이 그대로인 것을 뺀다(앱 설정 `export`에 폴더마다 남긴다).
 * 세션은 휴지통이 아닌 것(끝나지 않은 세션 포함 — pull.py도 받는다), 워치 녹음은 받은 폴더 전부다. 받는 중인 `.part`·쓰다 남은 `.tmp`는 넣지 않는다.
 * 새로 넣을 것이 없으면 zip을 만들지 않는다. 쓰다 실패하면 만들던 zip을 지운다.
 * 앱 프로세스 안의 일이라 쓰는 동안 앱을 닫으면 끊길 수 있다(화면은 액티비티가 켜 둔다 — 기록 화면이 "앱을 닫지 마세요"를 보인다).
 * 끊겨 남은 숨은 `.pending-…` zip은 다음 내보내기를 시작할 때 지운다(이 앱이 만든 것만 보인다).
 * "새 것만"의 기준은 zip을 다 쓴 때 남긴다: 그 zip을 PC로 옮기지 않았으면 다음 "새 것만"에서 빠지므로, 확인 창에 지난번 zip을 보이고
 * 옮긴 뒤에만 "새 것만"을 쓰라고 적는다.
 */
class Exporter(private val context: Context, private val sessions: SessionStore, private val inbox: WatchInbox) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val last = context.getSharedPreferences(LAST_PREFS, Context.MODE_PRIVATE)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mutable = MutableStateFlow(ExportStatus(lastZip = last.getString(KEY_ZIP, null), lastMs = last.getLong(KEY_MS, 0L)))
    val status: StateFlow<ExportStatus> = mutable.asStateFlow()

    /** 내보내기를 시작한다(이미 하는 중이면 아무것도 하지 않는다). 주 스레드에서 부른다. */
    fun start(onlyNew: Boolean) {
        if (mutable.value.busy) return
        mutable.update { it.copy(busy = true, progress = 0f, message = null, error = false) }
        scope.launch {
            val result = try {
                export(onlyNew)
            } catch (e: Exception) {
                Result("내보내기 실패: ${e.message ?: e.javaClass.simpleName}", error = true)
            }
            mutable.update {
                it.copy(busy = false, message = result.message, error = result.error, lastZip = result.zip ?: it.lastZip, lastMs = if (result.zip != null) result.atMs else it.lastMs)
            }
        }
    }

    /** 한 번 내보낸 결과: 글, 오류인지, 새로 쓴 zip 이름과 그때(없으면 null). */
    private class Result(val message: String, val error: Boolean = false, val zip: String? = null, val atMs: Long = 0L)

    private fun export(onlyNew: Boolean): Result {
        val resolver = context.contentResolver
        runCatching { removeUnfinished(resolver) } // 앞서 끊긴 내보내기가 남긴 숨은 zip(가장 하기 쉬운 일만, 실패는 넘긴다)
        val sessionFolders = sessions.list()
            .filter { it.status != SessionMeta.STATUS_TRASH }
            .sortedBy { it.name }
            .map { ExportRules.folder(ExportRules.SESSIONS, it.dir) }
        val recordingFolders = (inbox.root.listFiles() ?: emptyArray())
            .filter { it.isDirectory }
            .sortedBy { it.name }
            .map { ExportRules.folder(ExportRules.RECORDINGS, it) }
        val all = sessionFolders + recordingFolders
        val previous = prefs.all.mapNotNull { (key, value) -> (value as? String)?.let { key to it } }.toMap()
        val chosen = ExportRules.select(all, previous, onlyNew)
        if (chosen.isEmpty()) {
            val nothing = ExportRules.select(all, emptyMap(), onlyNew = false).isEmpty()
            return Result(if (nothing) "내보낼 것이 없습니다" else "새로 내보낼 것이 없습니다(지난번과 같음)")
        }
        val atMs = System.currentTimeMillis()
        val name = ExportRules.zipName(atMs)
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/zip")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/$FOLDER")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: error("내려받기 폴더에 파일을 못 만듦")
        try {
            val out = resolver.openOutputStream(uri) ?: error("zip 파일을 못 엶")
            var shown = -1
            ExportZip.write(out, chosen) { done, total ->
                val percent = if (total > 0) (done * 100 / total).toInt() else 100
                if (percent != shown) {
                    shown = percent
                    mutable.update { it.copy(progress = percent / 100f) }
                }
            }
            resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
        prefs.edit().apply { for (f in chosen) putString(f.prefix, f.signature) }.apply()
        last.edit().putString(KEY_ZIP, name).putLong(KEY_MS, atMs).apply()
        val sessionCount = chosen.count { it.prefix.startsWith(ExportRules.SESSIONS + "/") }
        val message = String.format(
            Locale.ROOT,
            "내보냄: Download/%s/%s · 세션 %d개 · 워치 녹음 %d개 · %.0f MB",
            FOLDER, name, sessionCount, chosen.size - sessionCount, chosen.sumOf { it.bytes } / 1e6,
        )
        return Result(message, zip = name, atMs = atMs)
    }

    /** `Download/HandSonar/`에 남은, 이 앱이 쓰다 만(IS_PENDING) zip을 지운다. 다른 앱의 것은 보이지 않는다. */
    private fun removeUnfinished(resolver: ContentResolver) {
        val args = Bundle().apply {
            putInt(MediaStore.QUERY_ARG_MATCH_PENDING, MediaStore.MATCH_ONLY)
            putString(ContentResolver.QUERY_ARG_SQL_SELECTION, "${MediaStore.MediaColumns.RELATIVE_PATH} = ?")
            putStringArray(ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf("${Environment.DIRECTORY_DOWNLOADS}/$FOLDER/"))
        }
        resolver.query(MediaStore.Downloads.EXTERNAL_CONTENT_URI, arrayOf(BaseColumns._ID), args, null)?.use { c ->
            while (c.moveToNext()) {
                resolver.delete(ContentUris.withAppendedId(MediaStore.Downloads.EXTERNAL_CONTENT_URI, c.getLong(0)), null, null)
            }
        }
    }

    private companion object {
        const val PREFS = "export"
        const val LAST_PREFS = "export-last"
        const val KEY_ZIP = "zip"
        const val KEY_MS = "ms"
        const val FOLDER = "HandSonar"
    }
}
