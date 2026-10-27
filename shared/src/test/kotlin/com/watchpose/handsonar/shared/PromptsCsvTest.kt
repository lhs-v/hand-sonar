package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class PromptsCsvTest {
    @Test
    fun headerAndRow() {
        assertEquals("index,segment,classId,name,startNs,endNs,flag", PromptsCsv.HEADER)
        assertEquals("3,1,2,중지 굽히기,10,20,wrong", PromptsCsv.row(PromptRecord(3, 1, 2, "중지 굽히기", 10, 20, "wrong")))
        assertEquals("0,0,-2,일시정지,1,2,pause", PromptsCsv.row(PromptRecord(0, 0, -2, "일시정지", 1, 2, "pause")))
    }

    @Test
    fun nameWithCommaIsQuoted() {
        assertEquals("1,7,-1,\"a,\"\"b\"\"\",0,1,", PromptsCsv.row(PromptRecord(1, 7, -1, "a,\"b\"", 0, 1, "")))
    }
}
