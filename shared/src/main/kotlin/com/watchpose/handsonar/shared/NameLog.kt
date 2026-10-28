package com.watchpose.handsonar.shared

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * 이름 한 줄씩의 작은 기록 파일(전송 설계 §2): 워치는 폰이 다 받았다고 답한 녹음을, 폰은 다 받은 녹음을 적는다.
 * 녹음 폴더 밖(앱 내부 저장소)에 두어 PC로 받는 폴더 모양(rec.json, part-NNN.wav)을 바꾸지 않는다.
 * 임시 파일에 다 쓴 뒤 이름을 바꿔 덮으므로 쓰다가 죽어도 앞 판이 남는다. 한 파일에는 객체 하나만 쓴다(앱 객체가 들고 있다).
 */
class NameLog(private val file: File) {
    private val lock = Any()
    private var cache: Set<String>? = null

    fun names(): Set<String> = synchronized(lock) { loaded() }

    /** 새로 넣었으면 true, 이미 있었으면 false. 빈 이름과 줄바꿈이 든 이름은 받지 않는다(IllegalArgumentException). */
    fun add(name: String): Boolean {
        require(name.isNotBlank() && name.none { it == '\n' || it == '\r' }) { "기록할 수 없는 이름: $name" }
        synchronized(lock) {
            val now = loaded()
            if (name in now) return false
            val next = now + name
            write(next)
            cache = next
        }
        return true
    }

    private fun loaded(): Set<String> = cache ?: read().also { cache = it }

    private fun read(): Set<String> {
        if (!file.isFile) return emptySet()
        return file.readLines(Charsets.UTF_8).map { it.trim() }.filter { it.isNotEmpty() }.toCollection(LinkedHashSet())
    }

    private fun write(names: Set<String>) {
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(names.joinToString("") { "$it\n" }, Charsets.UTF_8)
        Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    }
}
