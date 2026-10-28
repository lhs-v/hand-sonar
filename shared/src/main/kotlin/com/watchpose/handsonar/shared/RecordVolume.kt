package com.watchpose.handsonar.shared

/** 녹음 볼륨으로 바꾼 동안의 기록(계획 6): 원래 볼륨과 맞춘 볼륨. 워치 앱 설정에 바로(commit) 남겨, 앱이 죽어도 다음에 되돌린다. */
data class VolumePending(val original: Int, val applied: Int)

/** 미디어 볼륨을 읽고 바꾸는 쪽(워치는 AudioManager, 시험은 가짜). 볼륨은 지금 미디어가 나가는 출력 기기의 값이다. */
interface VolumeControl {
    fun volume(): Int

    fun setVolume(index: Int)

    /** 지금 이어진 외부 출력(블루투스 오디오·BLE·보청기·유선·USB)의 이름. 비면 미디어가 내장 스피커로 나간다. */
    fun externalOutputs(): List<String>
}

/** [VolumePending]을 두는 곳(워치는 앱 설정, 시험은 메모리). [save]는 돌아오기 전에 디스크에 닿아야 한다. null이면 지운다. */
interface VolumeStore {
    fun load(): VolumePending?

    fun save(pending: VolumePending?)
}

/**
 * 녹음 시작 때 볼륨을 다룬 결과(rec.json `volume`). [original]은 시작 전 볼륨, [volume]은 맞춘 뒤 다시 읽은 볼륨(녹음 동안 볼륨),
 * [target]은 맞추려던 값(그대로이거나 걸지 않았으면 null), [outputs]는 그때 이어진 외부 출력, [skipped]는 녹음 볼륨을 걸지 않은 까닭
 * ([RecordVolumeRules.SKIP_EXTERNAL]) 또는 null이다.
 */
data class VolumeStart(val original: Int, val volume: Int, val target: Int?, val outputs: List<String>, val skipped: String?) {
    /** 맞추려던 값에 정확히 닿지 못했다(기기가 무시하거나 잘랐다). */
    val mismatch: Boolean get() = target != null && volume != target
}

/**
 * 되돌리기 결과: [restored]는 되돌린 뒤 다시 읽은 볼륨(되돌리지 않았으면 null), [skipped]는 되돌리지 않은 까닭
 * ([RecordVolumeRules.SKIP_EXTERNAL] — 기록은 남겨 미룸, [RecordVolumeRules.SKIP_CHANGED] — 그사이 바뀌어 기록만 지움), [pending]은 다룬 기록.
 * 셋 다 null이면 다룰 것이 없었다.
 */
data class VolumeRestore(val restored: Int?, val skipped: String?, val pending: VolumePending?)

/**
 * 워치 녹음 볼륨의 규칙(전송 설계 §8, 계획 6 고침 I1·M2·M3). 안드로이드는 미디어 볼륨을 출력 기기마다 따로 두고, 처프는 내장 스피커로 고정해
 * 낸다(SonarPlayer). 그래서 미디어가 외부 출력(블루투스 이어폰 등)으로 갈 때 볼륨을 바꾸거나 되돌리면 스피커가 아니라 그 기기의 볼륨이 바뀐다.
 * - [start]: 외부 출력이 이어져 있으면 바꾸지 않고 까닭을 돌려준다. 바꿀 때는 원래 값과 맞출 값을 먼저 [VolumeStore]에 적고 바꾼 뒤 다시
 *   읽는다(실제로 바뀐 값을 다시 적는다). 맞출 값에 닿지 못하면 [VolumeStart.mismatch].
 * - [stop]: 이 녹음이 바꾼 볼륨만 다룬다. 외부 출력이 없으면(스피커로 나갈 때) 원래 값으로 되돌린다 — 녹음 중 사람이 바꿨어도(결정 11).
 *   외부 출력이 이어져 있으면 되돌리지 않고 기록을 남겨 미룬다.
 * - [restoreLeftover](앱이 죽었거나 정지 때 미룸): 외부 출력이 없고 지금 볼륨이 맞춘 값 그대로일 때만 되돌린다. 그사이 사람이 바꿨으면 그 값을
 *   두고 기록만 지운다(덮지 않는다). 외부 출력이 있으면 기록을 남겨 또 미룬다.
 * 이 녹음이 바꿨는지는 프로세스 안에서만 안다([start]가 적고 [stop]이 지운다): 앱이 죽어 다시 뜨면 남은 기록은 미뤄진 되돌리기로 다룬다.
 * 한 스레드(워치는 주 스레드)에서만 부른다.
 */
class RecordVolumeRules(private val control: VolumeControl, private val store: VolumeStore) {
    private var ownPending = false

    /** 되돌릴 기록(없으면 null). */
    val pending: VolumePending? get() = store.load()

    /** 녹음 시작: [choice]는 고른 단계(0 = 그대로), [max]는 최대 단계. */
    fun start(choice: Int, max: Int): VolumeStart {
        val current = control.volume()
        val outputs = control.externalOutputs()
        if (choice <= 0) return VolumeStart(current, current, null, outputs, null)
        if (outputs.isNotEmpty()) return VolumeStart(current, current, null, outputs, SKIP_EXTERNAL)
        val target = SonarSpec.recordVolumeTarget(choice, current, max)
        if (target == current) return VolumeStart(current, current, target, outputs, null)
        store.save(VolumePending(current, target)) // 바꾸기 전에 적는다: 바꾼 직후 앱이 죽어도 되돌릴 수 있게
        ownPending = true
        control.setVolume(target)
        val volume = control.volume()
        when (volume) {
            target -> Unit
            current -> { // 바뀌지 않았다: 되돌릴 것이 없다
                store.save(null)
                ownPending = false
            }
            else -> store.save(VolumePending(current, volume))
        }
        return VolumeStart(current, volume, target, outputs, null)
    }

    /** 녹음 정지: 이 녹음이 바꾼 볼륨을 되돌린다(외부 출력이 있으면 미룸). 이 녹음이 바꾸지 않았으면 아무것도 하지 않는다. */
    fun stop(): VolumeRestore {
        if (!ownPending) return VolumeRestore(null, null, null)
        ownPending = false
        val p = store.load() ?: return VolumeRestore(null, null, null)
        if (control.externalOutputs().isNotEmpty()) return VolumeRestore(null, SKIP_EXTERNAL, p)
        control.setVolume(p.original)
        store.save(null)
        return VolumeRestore(control.volume(), null, p)
    }

    /** 미뤄진 되돌리기: 녹음하지 않을 때(화면이 보일 때, 다음 녹음을 시작하기 전) 부른다. */
    fun restoreLeftover(): VolumeRestore {
        if (ownPending) return VolumeRestore(null, null, null) // 녹음 중: 정지가 맡는다
        val p = store.load() ?: return VolumeRestore(null, null, null)
        if (control.externalOutputs().isNotEmpty()) return VolumeRestore(null, SKIP_EXTERNAL, p)
        if (control.volume() != p.applied) {
            store.save(null)
            return VolumeRestore(null, SKIP_CHANGED, p)
        }
        control.setVolume(p.original)
        store.save(null)
        return VolumeRestore(control.volume(), null, p)
    }

    companion object {
        /** 외부 출력이 이어져 있어 바꾸거나 되돌리지 않았다(rec.json 값). */
        const val SKIP_EXTERNAL = "external_output"

        /** 그사이 볼륨이 바뀌어(사람이 바꿈) 되돌리지 않고 기록만 지웠다. */
        const val SKIP_CHANGED = "changed"
    }
}
