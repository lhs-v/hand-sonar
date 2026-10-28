package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SessionContextTest {
    @Test
    fun idsAreMetaValuesAndLabelsAreScreenText() {
        assertEquals(listOf("sitting" to "앉음", "standing" to "섬"), Posture.entries.map { it.id to it.label })
        assertEquals(listOf("desk" to "책상", "air" to "공중"), ArmSupport.entries.map { it.id to it.label })
        assertEquals(listOf("loose" to "헐렁", "normal" to "보통", "tight" to "꽉"), Strap.entries.map { it.id to it.label })
    }

    @Test
    fun unsetFieldsAreNullInJson() {
        assertEquals("""{"place":null,"posture":null,"armSupport":null,"strap":null}""", JsonOut.write(SessionContext().toMap()))
        assertEquals("""{"place":null,"posture":null,"armSupport":null,"strap":null}""", JsonOut.write(SessionContext(place = "  ").toMap()))
        assertEquals(
            """{"place":"집 책상","posture":"standing","armSupport":"air","strap":"tight"}""",
            JsonOut.write(SessionContext("집 책상", Posture.STANDING, ArmSupport.AIR, Strap.TIGHT).toMap()),
        )
    }

    @Test
    fun summaryLineSkipsUnsetFieldsAndShowsRewear() {
        assertEquals("세션 정보 없음", SessionContext().line(rewear = false))
        assertEquals("워치 다시 참", SessionContext(place = " ").line(rewear = true))
        assertEquals("앉음 · 팔 책상 · 줄 보통 · 집", SessionContext("집", Posture.SITTING, ArmSupport.DESK, Strap.NORMAL).line(rewear = false))
        assertEquals("섬 · 줄 헐렁 · 워치 다시 참", SessionContext(posture = Posture.STANDING, strap = Strap.LOOSE).line(rewear = true))
    }

    @Test
    fun placeIsOneTrimmedLineOfAtMostFortyChars() {
        assertEquals("집 책상", SessionContext.cleanPlace("  집\n  책상 "))
        assertEquals("집 책상", SessionContext.cleanPlace("집　　책상　")) // 전각 공백도 공백이다
        assertEquals(SessionContext.MAX_PLACE, SessionContext.cleanPlace("가".repeat(100)).length)
        assertEquals("", SessionContext.cleanPlace(" \t "))
    }

    @Test
    fun cuttingNeverSplitsASurrogatePair() {
        // 이모지는 UTF-16 두 칸이다: 40자 자르기가 반쪽을 남기면 meta.json에 깨진 글자가 된다
        val cut = SessionContext.limit("가" + "😀".repeat(50))
        assertEquals(SessionContext.MAX_PLACE, cut.codePointCount(0, cut.length))
        assertTrue(Character.isSurrogatePair(cut[cut.length - 2], cut[cut.length - 1]))
        assertEquals(cut, SessionContext.cleanPlace("가" + "😀".repeat(50)))
        assertEquals("짧음", SessionContext.limit("짧음"))
    }
}
