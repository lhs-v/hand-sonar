package com.watchpose.handsonar.mobile.session

import android.content.Context
import com.watchpose.handsonar.shared.PromptRecord
import com.watchpose.handsonar.shared.PromptsCsv
import com.watchpose.handsonar.shared.SessionMeta
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** 기록 화면 한 줄(meta.json에서 읽음). 끝나지 않은 세션(앱이 도중에 죽음)은 finished=false, stats 없음. */
data class SessionSummary(
    val dir: File,
    val name: String,
    val startWallMs: Long,
    val durationSec: Double,
    val status: String,
    val flags: List<String>,
    val handRatio: Double?,
    val watchRatio: Double?,
    /** 워치 기울기를 받고 있던 시간 비율(meta.json `stats.watchLinkRatio`). 이 칸 전 세션은 null. */
    val watchLinkRatio: Double? = null,
    val startMarker: Boolean?,
    val endMarker: Boolean?,
    val cancelled: Boolean,
    val finished: Boolean,
) {
    val suspect: Boolean get() = flags.isNotEmpty() || !finished
}

/**
 * 세션 폴더(설계 §7.2): 앱 외부 저장소 Android/data/com.watchpose.handsonar/files/sessions/<YYYYMMDD-HHMMSS>/.
 * 휴지통은 meta.json의 status("kept" | "trash")로만 나타내고 폴더는 그대로 둔다.
 * 완전 삭제는 휴지통 세션만, 사용자가 기록 화면에서 직접 할 때만 한다.
 */
class SessionStore(context: Context) {
    val root: File = File(context.getExternalFilesDir(null), "sessions").apply { mkdirs() }

    /**
     * 파일 쓰기와 meta.json 고쳐 쓰기를 한 번에 하나만 하게 한다. 세션을 닫는 쓰기와 "되돌리기"·휴지통의 setStatus가
     * 같은 `meta.json.tmp`를 쓰거나, 낡은 meta.json을 읽어 새 것을 덮어쓰는 일을 막는다.
     */
    private val writeLock = Any()

    fun newFolder(wallMs: Long): File {
        val base = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(Date(wallMs))
        var dir = File(root, base)
        var k = 2
        while (dir.exists()) {
            dir = File(root, "$base-$k")
            k++
        }
        check(dir.mkdirs()) { "세션 폴더를 못 만듦: $dir" }
        return dir
    }

    fun writeMeta(dir: File, json: String) = writeAtomically(File(dir, META), json)

    fun writePrompts(dir: File, records: List<PromptRecord>) = writeAtomically(
        File(dir, PROMPTS),
        buildString {
            append(PromptsCsv.HEADER).append('\n')
            for (r in records) append(PromptsCsv.row(r)).append('\n')
        },
    )

    fun writeText(dir: File, name: String, text: String) = writeAtomically(File(dir, name), text)

    /** meta.json의 status만 바꾼다. 읽고 쓰는 동안 다른 쓰기를 막는다. */
    fun setStatus(dir: File, status: String) {
        synchronized(writeLock) {
            val file = File(dir, META)
            val o = JSONObject(file.readText())
            o.put("status", status)
            writeAtomically(file, o.toString())
        }
    }

    /** 시작 시각 내림차순. 읽지 못하는 폴더는 건너뛴다. */
    fun list(): List<SessionSummary> = (root.listFiles() ?: emptyArray())
        .filter { it.isDirectory && File(it, META).isFile }
        .mapNotNull { runCatching { read(it) }.getOrNull() }
        .sortedByDescending { it.startWallMs }

    /** 휴지통 세션 폴더를 지운다(되돌릴 수 없음). 휴지통이 아니면 지우지 않고 false. */
    fun deleteForever(dir: File): Boolean {
        val summary = runCatching { read(dir) }.getOrNull() ?: return false
        if (summary.status != SessionMeta.STATUS_TRASH) return false
        return dir.deleteRecursively()
    }

    fun freeBytes(): Long = root.usableSpace

    private fun read(dir: File): SessionSummary {
        val o = JSONObject(File(dir, META).readText())
        val stats = o.optJSONObject("stats")
        val markers = stats?.optJSONObject("markers")
        val flags = o.optJSONArray("flags")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList()
        return SessionSummary(
            dir = dir,
            name = dir.name,
            startWallMs = o.getJSONObject("start").getLong("wallMs"),
            durationSec = stats?.optDouble("durationSec", 0.0) ?: 0.0,
            status = o.optString("status", SessionMeta.STATUS_KEPT),
            flags = flags,
            handRatio = stats?.optDouble("handRatio"),
            watchRatio = stats?.optDouble("watchHeardRatio"),
            watchLinkRatio = stats?.optDouble("watchLinkRatio")?.takeUnless { it.isNaN() },
            startMarker = markers?.optJSONObject("start")?.optBoolean("found"),
            endMarker = markers?.optJSONObject("end")?.optBoolean("found"),
            cancelled = o.optBoolean("cancelled", false),
            finished = !o.isNull("end"),
        )
    }

    /**
     * 임시 파일에 쓰고 이름을 바꿔, 반쯤 쓴 파일이 남지 않게 한다. 안드로이드의 renameTo는 있는 대상을 덮어쓰므로 먼저 지우지 않는다
     * (이름 바꾸기가 실패했을 때 정상 meta.json까지 지워질 수 있었다).
     */
    private fun writeAtomically(target: File, text: String) {
        synchronized(writeLock) {
            val tmp = File(target.parentFile, target.name + ".tmp")
            tmp.writeText(text)
            check(tmp.renameTo(target)) { "${target.name}을(를) 못 씀" }
        }
    }

    companion object {
        const val META = "meta.json"
        const val PROMPTS = "prompts.csv"
        const val AUDIO = "phone_audio.wav"
        const val AUDIO_JSON = "phone_audio.json"
    }
}
