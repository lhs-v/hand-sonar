package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

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
        assertEquals(SessionContext.MAX_PLACE, SessionContext.cleanPlace("가".repeat(100)).length)
        assertEquals("", SessionContext.cleanPlace(" \t "))
    }
}
