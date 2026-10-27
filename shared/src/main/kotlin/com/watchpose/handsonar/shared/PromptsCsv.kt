package com.watchpose.handsonar.shared

/** prompts.csv(설계 §7.2): 실제로 보인 동작과 일시정지 구간, 한 줄에 하나. */
object PromptsCsv {
    const val HEADER = "index,segment,classId,name,startNs,endNs,flag"

    fun row(r: PromptRecord): String = listOf(r.index, r.segment, r.classId, quote(r.name), r.startNs, r.endNs, r.flag).joinToString(",")

    private fun quote(s: String): String =
        if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
}
