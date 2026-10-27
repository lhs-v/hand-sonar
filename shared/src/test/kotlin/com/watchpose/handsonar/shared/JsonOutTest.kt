package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class JsonOutTest {
    @Test
    fun writesNestedMapsAndListsInOrder() {
        val json = JsonOut.write(
            linkedMapOf<String, Any?>("a" to 1, "b" to listOf(1L, 2.5, "x"), "c" to null, "d" to true, "e" to linkedMapOf("f" to 0.6f)),
        )
        assertEquals("""{"a":1,"b":[1,2.5,"x"],"c":null,"d":true,"e":{"f":0.6}}""", json)
    }

    @Test
    fun writesLongArraysAsLists() {
        assertEquals("[[48000,5,47040]]", JsonOut.write(listOf(longArrayOf(48_000, 5, 47_040))))
    }

    @Test
    fun escapesStrings() {
        assertEquals("\"q\\\"\\\\\\n\\u0001한\"", JsonOut.write("q\"\\\n\u0001한"))
    }

    @Test
    fun rejectsNonFiniteNumbers() {
        assertFailsWith<IllegalArgumentException> { JsonOut.write(Double.NaN) }
    }
}
