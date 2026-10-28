package com.watchpose.handsonar.wear.record

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import com.watchpose.handsonar.shared.SensorCsvWriter
import com.watchpose.handsonar.shared.SensorInfo
import com.watchpose.handsonar.shared.SensorKind
import com.watchpose.handsonar.shared.SensorReport
import com.watchpose.handsonar.shared.SensorSpec
import java.io.File

/**
 * 녹음하는 동안 워치 센서를 녹음 폴더의 `[base]-NNN.csv`에 적는다(전송 설계 §8, 계획 6). 워치는 재고 적기만 하고 값을 처리하지 않는다.
 * [specs]의 종류마다 기본 센서(getDefaultSensor)를 찾아 센서 스레드 하나(HandlerThread, 보통 우선순위 — 소리 스레드는 URGENT_AUDIO)에서
 * 배치 없이(최대 지연 0) 듣고, 이벤트마다 이벤트 시각(SensorEvent.timestamp)과 받은 시각(SystemClock.elapsedRealtimeNanos)을
 * [SensorCsvWriter]에 넘긴다. 줄 쓰기·1분 조각·1초마다 내보내기·정해진 버퍼는 그쪽이 맡는다.
 * 이벤트 시각의 기준: SDK 문서와 CDD 7.3은 elapsedRealtimeNanos와 같은 시계를 요구하지만 맞춤 오차는 100 ms 아래를 권할 뿐이고, 이 기기에서
 * 아직 재지 않았다. 그래서 받은 시각을 줄마다 함께 적는다(PC `hs_io.SensorTrack.event_clock`이 견주고, 어긋나면 받은 시각을 쓴다).
 * 소리 녹음을 방해하지 않게: 시작·정지는 예외를 내지 않고 문제를 [report]·[note]에 남기며, 기기에 없는 센서는 건너뛴다.
 * [start]·[stop]은 주 스레드에서 한 번씩 부른다. 폰 앱 `mobile.session.SensorRecorder`와 같은 틀이다(두 앱이 함께 쓰는 안드로이드 모듈은 없다).
 */
class SensorRecorder(context: Context, dir: File, base: String, private val specs: List<SensorSpec>) {
    private val manager: SensorManager? = context.getSystemService(SensorManager::class.java)
    private val writer = SensorCsvWriter(dir, base)
    private var thread: HandlerThread? = null

    // 아래 넷은 start가 센서 스레드를 띄우기 전에 한 번 정하고, 센서 스레드와 서비스 틱이 읽는다
    @Volatile private var kinds: Map<Int, SensorKind> = emptyMap()
    @Volatile private var found: List<SensorInfo> = emptyList()
    @Volatile private var missing: List<SensorKind> = emptyList()
    @Volatile private var problem: String? = null

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val received = SystemClock.elapsedRealtimeNanos()
            val type = event.sensor?.type ?: return
            val kind = kinds[type] ?: return
            writer.add(kind, event.timestamp, received, event.values, event.accuracy)
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit // 정확도는 줄마다 accuracy 칸에 있다
    }

    /** 센서를 찾아 센서 스레드에서 듣기 시작한다. 문제가 있어도 예외를 내지 않는다. */
    fun start() {
        val m = manager ?: run {
            problem = "센서 관리자가 없음"
            return
        }
        try {
            val chosen = specs.mapNotNull { spec -> m.getDefaultSensor(spec.kind.androidType)?.let { spec to it } }
            missing = specs.map { it.kind } - chosen.map { it.first.kind }.toSet()
            found = chosen.map { (spec, sensor) -> info(spec, sensor) }
            kinds = chosen.associate { (spec, _) -> spec.kind.androidType to spec.kind }
            if (chosen.isEmpty()) return
            val t = HandlerThread(THREAD_NAME).also { it.start() }
            thread = t
            val handler = Handler(t.looper)
            val failed = chosen.filterNot { (spec, sensor) -> m.registerListener(listener, sensor, spec.periodUs, handler) }
            if (failed.isNotEmpty()) problem = "센서 등록 실패: ${failed.joinToString(", ") { it.first.kind.id }}"
        } catch (e: Exception) {
            problem = "센서 기록 시작 실패: ${e.message ?: e.javaClass.simpleName}"
        }
    }

    /** 듣기를 멈추고 센서 스레드를 끝낸 뒤 CSV를 닫는다(보내기 전에 파일이 다 닫혀 있게). 두 번 불러도 된다. */
    fun stop() {
        runCatching { manager?.unregisterListener(listener) }
        thread?.let {
            it.quitSafely()
            runCatching { it.join(JOIN_MS) }
        }
        thread = null
        writer.close()
    }

    /** rec.json `sensors`. */
    fun report(): SensorReport = SensorReport(found, missing, writer.parts, writer.counts, problems().joinToString(" · ").ifEmpty { null })

    /** 화면 한 줄: 문제가 있으면 그것, 기기에 없는 센서가 있으면 그 이름, 아니면 null. */
    fun note(): String? = problems().firstOrNull() ?: missing.takeIf { it.isNotEmpty() }?.let { m -> "센서 없음: ${m.joinToString(", ") { it.id }}" }

    private fun problems(): List<String> = listOfNotNull(problem, writer.error)

    private fun info(spec: SensorSpec, s: Sensor) = SensorInfo(
        kind = spec.kind,
        type = s.type,
        name = s.name ?: "",
        vendor = s.vendor ?: "",
        version = s.version,
        periodUs = spec.periodUs,
        minDelayUs = s.minDelay,
        maxDelayUs = s.maxDelay,
        resolution = s.resolution,
        maxRange = s.maximumRange,
        fifoMax = s.fifoMaxEventCount,
        reportingMode = s.reportingMode,
        wakeUp = s.isWakeUpSensor,
    )

    private companion object {
        const val THREAD_NAME = "handsonar-sensors"
        const val JOIN_MS = 500L
    }
}
