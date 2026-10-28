package com.watchpose.handsonar.shared

/**
 * 보내기 실패를 녹음마다 센다(전송 설계 §2). 실패한 녹음은 쉬는 동안([backoffMs]) 건너뛰고 다른 녹음은 그대로 보낸다 —
 * 한 녹음이 계속 실패해도 그 뒤 녹음이 막히지 않는다. [maxTries]번 실패한 녹음은 이 장부가 사는 동안(보내기 한 번 실행) 뺀다.
 * 시각은 부팅 후 ms처럼 줄지 않는 값을 넘긴다. 한 스레드(보내기 작업 스레드)만 쓴다.
 */
class RetryBook(
    private val maxTries: Int = 5,
    private val firstBackoffMs: Long = 60_000L,
    private val maxBackoffMs: Long = 8 * 60_000L,
) {
    private class Entry(var tries: Int = 0, var notBeforeMs: Long = 0L, var reason: String = "")

    private val entries = LinkedHashMap<String, Entry>()

    /** [name]이 [nowMs]에 [reason]으로 실패했다: 한 번 더 세고 쉬는 시간을 늘린다. */
    fun failed(name: String, reason: String, nowMs: Long) {
        val e = entries.getOrPut(name) { Entry() }
        e.tries++
        e.notBeforeMs = nowMs + backoffMs(e.tries)
        e.reason = reason
    }

    fun succeeded(name: String) {
        entries.remove(name)
    }

    /** [pending] 중 시도가 남은 녹음(차례 그대로). */
    fun tryable(pending: List<String>): List<String> = pending.filter { (entries[it]?.tries ?: 0) < maxTries }

    /** [pending] 중 지금 해 볼 녹음(차례 그대로): 시도가 남았고 쉬는 시간이 지났다. */
    fun due(pending: List<String>, nowMs: Long): List<String> = tryable(pending).filter { (entries[it]?.notBeforeMs ?: 0L) <= nowMs }

    /** [pending] 중 쉬고 있는(시도가 남은) 녹음의 가장 이른 다시 할 때. 없으면 null. */
    fun nextAt(pending: List<String>): Long? = tryable(pending).mapNotNull { entries[it]?.notBeforeMs }.minOrNull()

    /** 마지막 실패 까닭(없으면 null). */
    fun reasonOf(name: String): String? = entries[name]?.reason

    /** 시도를 다 쓴 녹음과 마지막 까닭(실패한 차례). */
    fun exhausted(): List<Pair<String, String>> = entries.filterValues { it.tries >= maxTries }.map { (name, e) -> name to e.reason }

    /** n번째 실패 뒤 쉬는 시간: [firstBackoffMs]에서 두 배씩, [maxBackoffMs]까지. */
    fun backoffMs(tries: Int): Long = minOf(firstBackoffMs shl (tries - 1).coerceIn(0, 20), maxBackoffMs)
}
