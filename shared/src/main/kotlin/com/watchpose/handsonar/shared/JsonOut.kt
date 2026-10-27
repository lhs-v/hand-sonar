package com.watchpose.handsonar.shared

import java.util.Locale

/**
 * 기록 파일(rec.json)용 작은 JSON 쓰개. 순수 Kotlin 모듈이라 org.json을 쓸 수 없어 직접 쓴다.
 * 값은 null, 문자열, 참거짓, 정수, 실수(유한한 값만), Map(키는 문자열로), Iterable, LongArray, IntArray만 받는다.
 * Map은 넣은 순서대로 쓰므로 linkedMapOf를 넘긴다. Float는 사람이 읽기 좋게 Float 표기("0.6")를 그대로 쓴다.
 */
object JsonOut {
    fun write(value: Any?): String = StringBuilder().also { append(it, value) }.toString()

    private fun append(sb: StringBuilder, v: Any?) {
        when (v) {
            null -> sb.append("null")
            is String -> string(sb, v)
            is Boolean -> sb.append(v)
            is Int, is Long, is Short, is Byte -> sb.append(v)
            is Float -> number(sb, v.toString().toDouble())
            is Double -> number(sb, v)
            is Map<*, *> -> {
                sb.append('{')
                var first = true
                for ((k, value) in v) {
                    if (!first) sb.append(',')
                    first = false
                    string(sb, k.toString())
                    sb.append(':')
                    append(sb, value)
                }
                sb.append('}')
            }
            is Iterable<*> -> {
                sb.append('[')
                var first = true
                for (e in v) {
                    if (!first) sb.append(',')
                    first = false
                    append(sb, e)
                }
                sb.append(']')
            }
            is LongArray -> append(sb, v.asList())
            is IntArray -> append(sb, v.asList())
            else -> throw IllegalArgumentException("JSON으로 쓸 수 없는 값: ${v::class.simpleName}")
        }
    }

    private fun number(sb: StringBuilder, d: Double) {
        require(d.isFinite()) { "JSON에 유한하지 않은 수: $d" }
        sb.append(d)
    }

    private fun string(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when {
                c == '"' -> sb.append("\\\"")
                c == '\\' -> sb.append("\\\\")
                c == '\n' -> sb.append("\\n")
                c == '\r' -> sb.append("\\r")
                c == '\t' -> sb.append("\\t")
                c < ' ' -> sb.append(String.format(Locale.ROOT, "\\u%04x", c.code))
                else -> sb.append(c)
            }
        }
        sb.append('"')
    }
}
