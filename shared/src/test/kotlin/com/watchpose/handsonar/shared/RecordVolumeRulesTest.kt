package com.watchpose.handsonar.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RecordVolumeRulesTest {
    /** 앱 설정 대신. */
    private class Store(var value: VolumePending? = null) : VolumeStore {
        override fun load() = value

        override fun save(pending: VolumePending?) {
            value = pending
        }
    }

    /** AudioManager 대신: 지금 출력 기기의 볼륨 하나와 외부 출력 목록. [accept]가 바꾼 값을 고친다(무시·잘림 흉내). */
    private class Control(var current: Int, private val store: Store) : VolumeControl {
        var outputs: List<String> = emptyList()
        var accept: (Int) -> Int = { it }
        val sets = mutableListOf<Int>()
        var pendingWhenSet: VolumePending? = null

        override fun volume() = current

        override fun setVolume(index: Int) {
            sets += index
            pendingWhenSet = store.value // 바꾸기 전에 원래 값이 적혀 있었는가
            current = accept(index)
        }

        override fun externalOutputs() = outputs
    }

    private val store = Store()
    private val control = Control(7, store)
    private val rules = RecordVolumeRules(control, store)

    @Test
    fun keepLeavesTheVolumeAndRecordsTheOutputs() {
        control.outputs = listOf("bluetooth_a2dp")
        val start = rules.start(choice = 0, max = 15)
        assertEquals(VolumeStart(original = 7, volume = 7, target = null, outputs = listOf("bluetooth_a2dp"), skipped = null), start)
        assertEquals(emptyList(), control.sets)
        assertNull(store.value)
        assertEquals(VolumeRestore(null, null, null), rules.stop())
    }

    @Test
    fun externalOutputAtStartSkipsTheChange() {
        control.outputs = listOf("ble_headset")
        val start = rules.start(choice = 13, max = 15)
        assertEquals(RecordVolumeRules.SKIP_EXTERNAL, start.skipped) // 바꾸면 스피커가 아니라 이어폰 볼륨이 바뀐다
        assertNull(start.target)
        assertFalse(start.mismatch)
        assertEquals(emptyList(), control.sets)
        assertNull(store.value)
    }

    @Test
    fun raiseWritesTheOriginalFirstAndStopRestoresEvenAfterAPersonChangedIt() {
        val start = rules.start(choice = 13, max = 15)
        assertEquals(VolumeStart(7, 13, 13, emptyList(), null), start)
        assertEquals(VolumePending(7, 13), control.pendingWhenSet) // 앱이 죽어도 되돌릴 수 있게 바꾸기 전에 적었다
        control.current = 10 // 녹음 중 사람이 바꿈
        assertEquals(VolumeRestore(restored = 7, skipped = null, pending = VolumePending(7, 13)), rules.stop())
        assertEquals(7, control.current)
        assertNull(store.value)
    }

    @Test
    fun targetTheSameAsNowChangesNothingAndLowerChoicesLower() {
        control.current = 13
        assertEquals(VolumeStart(13, 13, 13, emptyList(), null), rules.start(choice = 13, max = 15))
        assertEquals(emptyList(), control.sets)
        control.current = 15
        assertEquals(VolumeStart(15, 11, 11, emptyList(), null), rules.start(choice = 11, max = 15))
        assertEquals(VolumeRestore(15, null, VolumePending(15, 11)), rules.stop())
    }

    @Test
    fun aChangeTheDeviceIgnoresOrClampsIsAMismatch() {
        control.accept = { 7 } // 방해 금지 등으로 무시
        val ignored = rules.start(choice = 13, max = 15)
        assertTrue(ignored.mismatch)
        assertEquals(7, ignored.volume)
        assertNull(store.value) // 바뀌지 않았으니 되돌릴 것이 없다
        control.accept = { minOf(it, 12) } // 잘림
        val clamped = rules.start(choice = 13, max = 15)
        assertTrue(clamped.mismatch)
        assertEquals(VolumePending(7, 12), store.value) // 실제로 바뀐 값을 적는다
        assertEquals(7, rules.stop().restored)
    }

    @Test
    fun externalOutputAtStopKeepsTheRecordUntilTheSpeakerIsBack() {
        rules.start(choice = 13, max = 15)
        control.outputs = listOf("bluetooth_a2dp") // 녹음 중 블루투스가 이어짐: 되돌리면 이어폰 볼륨이 바뀐다
        assertEquals(VolumeRestore(null, RecordVolumeRules.SKIP_EXTERNAL, VolumePending(7, 13)), rules.stop())
        assertEquals(listOf(13), control.sets)
        assertEquals(VolumePending(7, 13), store.value)
        assertEquals(VolumeRestore(null, RecordVolumeRules.SKIP_EXTERNAL, VolumePending(7, 13)), rules.restoreLeftover())
        control.outputs = emptyList()
        control.current = 13 // 스피커로 돌아옴, 그대로 13
        assertEquals(VolumeRestore(7, null, VolumePending(7, 13)), rules.restoreLeftover())
        assertNull(store.value)
    }

    @Test
    fun leftoverAfterACrashIsRestoredOnlyWhenNobodyChangedIt() {
        store.value = VolumePending(7, 13) // 녹음 중 앱이 죽어 남은 기록(새 프로세스)
        control.current = 13
        assertEquals(VolumeRestore(7, null, VolumePending(7, 13)), RecordVolumeRules(control, store).restoreLeftover())
        store.value = VolumePending(7, 13)
        control.current = 5 // 그사이 사람이 바꿈: 덮지 않고 기록만 지운다
        assertEquals(VolumeRestore(null, RecordVolumeRules.SKIP_CHANGED, VolumePending(7, 13)), RecordVolumeRules(control, store).restoreLeftover())
        assertEquals(5, control.current)
        assertNull(store.value)
    }

    @Test
    fun stopOnlyTouchesWhatThisRecordingChanged() {
        store.value = VolumePending(4, 15) // 앞 녹음이 남긴 기록(블루투스로 미룸)
        control.outputs = listOf("bluetooth_a2dp")
        rules.start(choice = 13, max = 15) // 외부 출력이라 걸지 않음
        control.outputs = emptyList()
        assertEquals(VolumeRestore(null, null, null), rules.stop()) // 남은 기록은 미뤄진 되돌리기가 맡는다
        assertEquals(VolumePending(4, 15), store.value)
        assertEquals(emptyList(), control.sets)
    }
}
