package com.watchpose.handsonar.shared

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** 내보낼 파일 하나(전송 설계 §5): 이름, 크기, 마지막 수정 시각(wallMs). */
data class ExportFile(val name: String, val bytes: Long, val modifiedMs: Long)

/** 내보낼 폴더 하나: zip 안 앞머리([prefix], 예 `phone/sessions/20261027-235412`), 기기의 폴더, 그 안 파일들(이름 순). */
data class ExportFolder(val prefix: String, val dir: File, val files: List<ExportFile>) {
    val bytes: Long get() = files.sumOf { it.bytes }

    /** `파일 수:크기 합:가장 늦은 수정 시각`. 지난번 내보낸 값과 같으면 "새 것만"에서 뺀다(설계 §5의 크기·시각 비교). */
    val signature: String get() = "${files.size}:$bytes:${files.maxOfOrNull { it.modifiedMs } ?: 0}"
}

/**
 * 폰 "내보내기"의 규칙(전송 설계 §5). zip 안 구조는 PC의 `hand-sonar-data`와 같아(`phone/sessions/<세션>/…`,
 * `watch/recordings/<녹음>/…`) 그대로 풀면 `pull.py`로 받은 것과 같다.
 */
object ExportRules {
    const val SESSIONS = "phone/sessions"
    const val RECORDINGS = "watch/recordings"
    private val STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")

    /** zip에 넣을 파일인가: 받는 중인 `.part`, 쓰다 남은 `.tmp`는 넣지 않는다([TransferRules.isLeftover]). */
    fun includes(name: String): Boolean = !TransferRules.isLeftover(name)

    /** [dir]의 파일(하위 폴더는 없다)을 이름 순으로 읽어 `[parent]/<폴더 이름>` 앞머리의 폴더로. */
    fun folder(parent: String, dir: File): ExportFolder = ExportFolder(
        "$parent/${dir.name}",
        dir,
        (dir.listFiles() ?: emptyArray())
            .filter { it.isFile && includes(it.name) }
            .sortedBy { it.name }
            .map { ExportFile(it.name, it.length(), it.lastModified()) },
    )

    /** 넣을 폴더: 빈 폴더는 빼고, [onlyNew]면 [previous](앞머리 → 지난번 [ExportFolder.signature])와 같은 것도 뺀다. */
    fun select(folders: List<ExportFolder>, previous: Map<String, String>, onlyNew: Boolean): List<ExportFolder> =
        folders.filter { it.files.isNotEmpty() && !(onlyNew && previous[it.prefix] == it.signature) }

    /** `hand-sonar-yyyyMMdd-HHmmss.zip`([zone]의 시각). */
    fun zipName(wallMs: Long, zone: ZoneId = ZoneId.systemDefault()): String =
        "hand-sonar-" + STAMP.format(Instant.ofEpochMilli(wallMs).atZone(zone)) + ".zip"

    fun entryName(folder: ExportFolder, file: ExportFile): String = "${folder.prefix}/${file.name}"
}

/** 폴더들을 zip 하나로 쓴다. 녹음·영상은 거의 줄지 않아 Deflate를 가장 빠른 단계로 둔다. */
object ExportZip {
    private const val BUFFER = 64 * 1024

    /**
     * [folders]의 파일을 [ExportRules.entryName] 차례로 [out]에 쓰고 [out]을 닫는다.
     * [onProgress]는 덩어리(64 KB)마다 (지금까지 쓴 바이트, 모두 바이트)로 불린다.
     */
    fun write(out: OutputStream, folders: List<ExportFolder>, onProgress: (done: Long, total: Long) -> Unit = { _, _ -> }) {
        val total = folders.sumOf { it.bytes }
        var done = 0L
        val buffer = ByteArray(BUFFER)
        ZipOutputStream(BufferedOutputStream(out, BUFFER)).use { zip ->
            zip.setLevel(Deflater.BEST_SPEED)
            for (folder in folders) {
                for (file in folder.files) {
                    zip.putNextEntry(ZipEntry(ExportRules.entryName(folder, file)).apply { time = file.modifiedMs })
                    FileInputStream(File(folder.dir, file.name)).use { input ->
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            zip.write(buffer, 0, n)
                            done += n
                            onProgress(done, total)
                        }
                    }
                    zip.closeEntry()
                }
            }
        }
    }
}
