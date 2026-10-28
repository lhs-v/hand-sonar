package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class RetryBookTest {
    private val a = "20261028-120000"
    private val b = "20261028-101500"
    private val c = "20261027-235407"

    @Test
    fun aFailingRecordingWaitsWhileTheOthersGoOn() {
        val book = RetryBook(maxTries = 3, firstBackoffMs = 60_000, maxBackoffMs = 480_000)
        val pending = listOf(a, b, c)
        assertEquals(pending, book.due(pending, nowMs = 0))
        book.failed(b, "part-003.wav: 연결이 끊김", nowMs = 1_000)
        assertEquals(listOf(a, c), book.due(pending, nowMs = 1_000)) // 차례를 막지 않는다
        assertEquals(61_000L, book.nextAt(pending))
        assertEquals(listOf(a, b, c), book.due(pending, nowMs = 61_000))
        book.succeeded(b)
        assertNull(book.nextAt(pending))
    }

    @Test
    fun backoffDoublesUpToTheCapAndTriesRunOut() {
        val book = RetryBook(maxTries = 5, firstBackoffMs = 60_000, maxBackoffMs = 480_000)
        assertEquals(listOf(60_000L, 120_000L, 240_000L, 480_000L, 480_000L), (1..5).map { book.backoffMs(it) })
        var now = 0L
        repeat(4) {
            book.failed(a, "폰이 답하지 않음", now)
            now = book.nextAt(listOf(a))!!
            assertEquals(listOf(a), book.tryable(listOf(a, b)).filter { it == a })
        }
        book.failed(a, "폰이 답하지 않음", now)
        assertEquals(listOf(b), book.tryable(listOf(a, b))) // 다섯 번째 실패: 이번 실행에서 뺀다
        assertEquals(listOf(b), book.due(listOf(a, b), nowMs = Long.MAX_VALUE))
        assertEquals(listOf(a to "폰이 답하지 않음"), book.exhausted())
        assertNull(book.nextAt(listOf(a)))
    }
}
