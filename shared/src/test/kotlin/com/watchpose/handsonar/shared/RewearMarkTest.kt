package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RewearMarkTest {
    @Test
    fun theMarkGoesToTheFirstKeptSessionAndThenTurnsOff() {
        val mark = RewearMark<String>()
        mark.toggle()
        assertTrue(mark.started()) // 이 세션 meta.json rewear = 참
        mark.closed("s1", marked = true, kept = false) // 취소(휴지통): 다시 찬 뒤 첫 보관 세션은 아직이다
        assertTrue(mark.on)
        assertTrue(mark.started())
        mark.closed("s2", marked = true, kept = true)
        assertFalse(mark.on)
        assertFalse(mark.started())
        mark.closed("s3", marked = false, kept = true)
        assertFalse(mark.on)
    }

    @Test
    fun trashingTheJustSavedSessionTurnsItBackOnAndRestoringItTurnsItOff() {
        val mark = RewearMark<String>(on = true)
        mark.started()
        mark.closed("s1", marked = true, kept = true)
        mark.trashed("other") // 다른 세션을 버려도 그대로
        assertFalse(mark.on)
        mark.trashed("s1") // 되돌리기·기록 화면 휴지통: 첫 보관 세션은 다음 세션이 된다
        assertTrue(mark.on)
        mark.restored("s1") // 다시 살리면 그 세션이 다시 첫 보관 세션: 두 세션에 표시가 남지 않게 끈다
        assertFalse(mark.on)
        mark.trashed("s1") // 또 버리면 또 켠다
        assertTrue(mark.on)
    }

    @Test
    fun aNewSessionOrAPersonsToggleEndsTheJustSavedSessionRule() {
        val mark = RewearMark<String>(on = true)
        mark.started()
        mark.closed("s1", marked = true, kept = true)
        mark.started() // 다음 세션이 같은 착용으로 이어졌다: 이제 s1을 버려도 다시 켜면 착용 경계가 둘이 된다
        mark.closed("s2", marked = false, kept = true)
        mark.trashed("s1")
        assertFalse(mark.on)
        val other = RewearMark<String>(on = true)
        other.started()
        other.closed("s1", marked = true, kept = true)
        other.toggle() // 사람이 또 다시 찼다고 켬: 그 표시는 사람이 정한 것이라
        other.trashed("s1")
        other.restored("s1") // 되살려도 끄지 않는다
        assertTrue(other.on)
    }

    @Test
    fun anUnmarkedOrCancelledSessionIsNotTracked() {
        val mark = RewearMark<String>()
        mark.started()
        mark.closed("s1", marked = false, kept = true)
        mark.trashed("s1")
        assertFalse(mark.on)
        assertEquals(false, mark.started())
    }
}
