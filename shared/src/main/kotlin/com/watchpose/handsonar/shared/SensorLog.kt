package com.watchpose.handsonar.shared

import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.io.Writer
import java.util.Locale

/**
 * 기록하는 센서 종류(전송 설계 §8, 계획 6). [id]는 CSV `sensor` 칸과 json `kind`의 이름, [axes]는 적는 값 수(1이면 y·z 칸을 비운다),
 * [androidType]은 android.hardware.Sensor.TYPE_* 값이다(shared는 안드로이드를 모르므로 값을 적어 둔다).
 * 단위는 안드로이드 그대로다: 가속도·중력 m/s², 자이로 rad/s, 지자기 µT, 밝기 lux.
 */
enum class SensorKind(val id: String, val axes: Int, val androidType: Int) {
    ACCEL("acc", 3, 1),
    GYRO("gyro", 3, 4),
    MAG("mag", 3, 2),
    GRAVITY("grav", 3, 9),
    LIGHT("light", 1, 5),
}

/** 기기에서 듣는 센서 하나: 종류와 건 표본 주기(µs, registerListener의 samplingPeriodUs). */
data class SensorSpec(val kind: SensorKind, val periodUs: Int)

/** 기기마다 기록하는 센서와 파일 이름(계획 6). 처리에는 쓰지 않고, 나중에 환경 변수를 가려 보려고 남긴다. */
object SensorPlan {
    /**
     * 약 100 Hz. 워치8의 가속도·자이로·지자기·중력은 모두 최대 100 Hz다(2026-10-21 dumpsys). 200 Hz를 넘기면 권한이 필요하다(안드로이드 12+).
     * 배치(최대 지연)는 쓰지 않는다: 받은 시각이 이벤트 시각에 가깝게.
     */
    const val IMU_PERIOD_US = 10_000

    /** 폰 주변 밝기는 값이 바뀔 때만 내는 센서라 이 값은 가장 짧은 간격이다(5 Hz). 카메라 조건만 본다. */
    const val LIGHT_PERIOD_US = 200_000

    /** 워치 녹음 폴더의 `sensors-NNN.csv`. */
    const val WATCH_BASE = "sensors"

    /** 폰 세션 폴더의 `phone_sensors-NNN.csv`(`phone_audio.*`처럼 폰에서 잰 것임을 이름에 둔다). */
    const val PHONE_BASE = "phone_sensors"

    /** 워치 센서 기록의 대략 바이트/초: 4종 × 100 Hz × 한 줄 약 80바이트. 남은 녹음 시간 계산([StorageBudget])에 쓴다. */
    const val WATCH_BYTES_PER_SECOND = 32_000

    /** 워치: 팔뚝·손목 방향, 워치가 움직이거나 눌린 순간(설계 §8). */
    val WATCH: List<SensorSpec> = listOf(
        SensorSpec(SensorKind.ACCEL, IMU_PERIOD_US),
        SensorSpec(SensorKind.GYRO, IMU_PERIOD_US),
        SensorSpec(SensorKind.MAG, IMU_PERIOD_US),
        SensorSpec(SensorKind.GRAVITY, IMU_PERIOD_US),
    )

    /**
     * 폰: 가속도·자이로(폰이 그대로였는지), 주변 밝기(카메라 조건). 화면이 둘인 접는 폰(폴드8)은 밝기 센서도 둘일 수 있고, 기본 밝기 센서
     * (getDefaultSensor)가 덮개 화면 쪽이면 펼쳐 세운 폰에서는 사용자 반대쪽 밝기를 잰다(계획 6 검토 M7). 쓴 센서는 json `sensors.list`,
     * 같은 종류의 다른 센서는 `sensors.others`에 이름으로 남겨 기기 확인(작업 9)에서 어느 쪽인지 가린다. 이름만으로 안쪽 화면 것을 가를 수
     * 있다고 확인되면 그때 그것을 고른다(지금은 기기 정보가 없어 기본 센서를 쓴다).
     */
    val PHONE: List<SensorSpec> = listOf(
        SensorSpec(SensorKind.ACCEL, IMU_PERIOD_US),
        SensorSpec(SensorKind.GYRO, IMU_PERIOD_US),
        SensorSpec(SensorKind.LIGHT, LIGHT_PERIOD_US),
    )
}

/**
 * 센서 CSV 한 줄(계획 6): `t_ns,recv_ns,sensor,x,y,z,accuracy`. t_ns는 SensorEvent.timestamp, recv_ns는 앱이 그 이벤트를 받은 때의
 * SystemClock.elapsedRealtimeNanos다. 이벤트 시각이 elapsedNs 기준이라는 것은 SDK 문서·CDD의 요구일 뿐 이 기기에서 아직 재지 않아,
 * PC가 견줄 수 있게 둘 다 적는다. 값은 Float 표기 그대로(`9.80665`, 작으면 `1.0E-4`)이고, 종류의 축 수를 넘는 칸과 기기가 주지 않은 칸은
 * 비운다. accuracy는 SensorEvent.accuracy(−1 접촉 없음, 0 믿을 수 없음, 1–3 낮음–높음).
 */
object SensorCsv {
    const val HEADER = "t_ns,recv_ns,sensor,x,y,z,accuracy"

    /** 한 줄(끝에 `\n`)을 [sb]에 덧붙인다. */
    fun appendRow(sb: StringBuilder, tNs: Long, recvNs: Long, kind: SensorKind, values: FloatArray, accuracy: Int) {
        sb.append(tNs).append(',').append(recvNs).append(',').append(kind.id)
        for (i in 0 until 3) {
            sb.append(',')
            if (i < kind.axes && i < values.size) sb.append(values[i])
        }
        sb.append(',').append(accuracy).append('\n')
    }
}

/** 센서 CSV 조각 하나: 파일 이름과 머리줄을 뺀 줄 수(json `sensors.parts`). */
data class SensorPart(val file: String, val rows: Long)

/**
 * 기기가 준 센서 하나(json `sensors.list`·`sensors.others`, 계획 6). android.hardware.Sensor의 값을 그대로 옮기고, [periodUs]는 우리가 건 표본
 * 주기다(`others`의 듣지 않은 센서는 0).
 * [maxRateHz]는 1e6 / [minDelayUs]이고, minDelay가 0 이하(값이 바뀔 때만 내는 센서 등)면 null이다.
 * [reportingMode]는 Sensor.REPORTING_MODE_*(0 연속, 1 바뀔 때, 2 한 번, 3 특수)다.
 */
data class SensorInfo(
    val kind: SensorKind,
    val type: Int,
    val name: String,
    val vendor: String,
    val version: Int,
    val periodUs: Int,
    val minDelayUs: Int,
    val maxDelayUs: Int,
    val resolution: Float,
    val maxRange: Float,
    val fifoMax: Int,
    val reportingMode: Int,
    val wakeUp: Boolean,
) {
    val maxRateHz: Double? get() = if (minDelayUs > 0) 1e6 / minDelayUs else null
}

/**
 * rec.json·meta.json의 `sensors`(계획 6): 기기가 준 센서들, 기기에 없던 종류, 같은 종류의 듣지 않은 다른 센서([others] — 폴드8 밝기 센서 같은
 * 것을 나중에 가리려고, 검토 M7), CSV 조각과 줄 수, 종류별 줄 수, 기록 문제(없으면 null).
 * 녹음 중에도 rec.json이 10초마다 다시 써지므로 그때의 줄 수는 그때까지다(끝난 녹음은 다 닫은 뒤의 값).
 */
data class SensorReport(
    val sensors: List<SensorInfo>,
    val missing: List<SensorKind>,
    val parts: List<SensorPart>,
    val counts: Map<SensorKind, Long>,
    val error: String?,
    val others: List<SensorInfo> = emptyList(),
) {
    /** [JsonOut]으로 쓸 모양. 유한하지 않은 수는 null로 적는다(JsonOut이 거절해 rec.json 쓰기가 멈추지 않게). */
    fun toMap(): Map<String, Any?> = linkedMapOf<String, Any?>(
        "columns" to SensorCsv.HEADER,
        "list" to sensors.map(::infoMap),
        "missing" to missing.map { it.id },
        "others" to others.map(::infoMap),
        "parts" to parts.map { linkedMapOf<String, Any?>("file" to it.file, "rows" to it.rows) },
        "counts" to linkedMapOf<String, Any?>().apply { for ((kind, n) in counts) put(kind.id, n) },
        "error" to error,
    )

    private fun infoMap(s: SensorInfo) = linkedMapOf<String, Any?>(
        "kind" to s.kind.id,
        "type" to s.type,
        "name" to s.name,
        "vendor" to s.vendor,
        "version" to s.version,
        "periodUs" to s.periodUs,
        "minDelayUs" to s.minDelayUs,
        "maxDelayUs" to s.maxDelayUs,
        "maxRateHz" to s.maxRateHz,
        "resolution" to finite(s.resolution),
        "maxRange" to finite(s.maxRange),
        "fifoMax" to s.fifoMax,
        "reportingMode" to s.reportingMode,
        "wakeUp" to s.wakeUp,
    )

    private fun finite(x: Float): Float? = x.takeIf { it.isFinite() }
}

/**
 * 센서 줄을 `[base]-NNN.csv` 조각에 쓴다(계획 6). 조각마다 첫 줄은 [SensorCsv.HEADER]이고, 첫 줄을 받을 때 그 조각을 연다(줄이 없으면 파일도 없다).
 * 때는 줄의 받은 시각(그 기기의 elapsedNs)으로 잰다: 조각을 연 지 [partNs]가 지난 줄은 새 조각에 쓰고, 마지막으로 내보낸 지 [flushNs]가 지난
 * 줄 뒤에 버퍼를 파일로 내보낸다(따로 타이머를 두지 않는다). 그래서 파일이 녹음 중에도 1초쯤마다 자라고, 앱이 죽어도 그만큼 남는다.
 * 조각은 처음부터 제 이름으로 쓴다(WAV 조각과 같다): 녹음 서비스가 정지 때 보내기 전에 [close]하므로 보내기는 다 닫힌 파일만 옮긴다.
 * 버퍼는 [bufferChars] 크기로 정해져 있어 메모리가 늘지 않는다. 파일을 열거나 쓰지 못하면(어떤 예외든 — 검토 M6) [error]를 남기고 그 뒤 줄은
 * 버린다: 센서 스레드에서 예외가 새면 앱과 함께 소리 녹음이 멈추므로 예외를 내지 않는다. [close] 뒤의 줄도 버린다. 줄마다 할당하지 않는다
 * (줄 글·문자 배열은 다시 쓰고, 종류별 셈은 [LongArray] — 검토 M9). 센서 스레드가 쓰고 다른 스레드가 [parts]·[counts]·[error]를 읽으므로
 * 모두 잠근다. [openWriter]는 시험이 쓰다 실패하는 Writer를 넣는 자리다(파일, 버퍼 문자 수).
 */
class SensorCsvWriter(
    private val dir: File,
    private val base: String,
    private val partNs: Long = PART_NS,
    private val flushNs: Long = FLUSH_NS,
    private val bufferChars: Int = BUFFER_CHARS,
    private val openWriter: (File, Int) -> Writer = { file, chars -> BufferedWriter(OutputStreamWriter(FileOutputStream(file), Charsets.UTF_8), chars) },
) : AutoCloseable {
    private val lock = Any()
    private val row = StringBuilder(ROW_CHARS)
    private var chars = CharArray(ROW_CHARS)
    private var out: Writer? = null
    private var currentName = ""
    private var currentRows = 0L
    private var openedNs = 0L
    private var flushedNs = 0L
    private val done = mutableListOf<SensorPart>()
    private val perKind = LongArray(SensorKind.entries.size)
    private var closed = false
    private var failure: String? = null

    /** 닫힌 조각과, 쓰는 중인 조각(지금까지 쓴 줄 수). */
    val parts: List<SensorPart> get() = synchronized(lock) { done + listOfNotNull(out?.let { SensorPart(currentName, currentRows) }) }

    /** 종류마다 쓴 줄 수(종류 차례, 줄이 없는 종류는 뺀다). */
    val counts: Map<SensorKind, Long>
        get() = synchronized(lock) { SensorKind.entries.filter { perKind[it.ordinal] > 0 }.associateWith { perKind[it.ordinal] } }

    /** 파일을 열거나 쓰지 못한 까닭(없으면 null). 그 뒤 줄은 버렸다. */
    val error: String? get() = synchronized(lock) { failure }

    /** 줄 하나. [recvNs]는 받은 시각(elapsedNs)이고 조각 나누기·내보내기의 때가 된다. */
    fun add(kind: SensorKind, tNs: Long, recvNs: Long, values: FloatArray, accuracy: Int) {
        synchronized(lock) {
            if (closed || failure != null) return
            try {
                if (out != null && recvNs - openedNs >= partNs) closeCurrent()
                val w = out ?: open(recvNs)
                row.setLength(0)
                SensorCsv.appendRow(row, tNs, recvNs, kind, values, accuracy)
                val n = row.length
                if (chars.size < n) chars = CharArray(n)
                row.getChars(0, n, chars, 0)
                w.write(chars, 0, n)
                currentRows++
                perKind[kind.ordinal]++
                if (recvNs - flushedNs >= flushNs) {
                    w.flush()
                    flushedNs = recvNs
                }
            } catch (e: Exception) {
                fail(e)
            }
        }
    }

    /** 쓰는 중인 조각을 닫는다. 두 번 불러도 된다. */
    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            try {
                closeCurrent()
            } catch (e: Exception) {
                fail(e)
            }
        }
    }

    private fun open(nowNs: Long): Writer {
        val file = File(dir, String.format(Locale.ROOT, "%s-%03d.csv", base, done.size))
        val w = openWriter(file, bufferChars)
        currentName = file.name
        currentRows = 0
        openedNs = nowNs
        flushedNs = nowNs
        out = w
        w.write(SensorCsv.HEADER)
        w.write("\n")
        return w
    }

    /** 쓰는 중인 조각을 목록에 넣고 닫는다(닫기가 실패해도 목록에는 남는다 — 파일은 있다). */
    private fun closeCurrent() {
        val w = out ?: return
        out = null
        done += SensorPart(currentName, currentRows)
        w.close()
    }

    private fun fail(e: Exception) {
        if (failure == null) failure = "센서 기록 쓰기 실패: ${e.message ?: e.javaClass.simpleName}"
        val w = out ?: return
        out = null
        done += SensorPart(currentName, currentRows)
        runCatching { w.close() }
    }

    companion object {
        /** 조각 길이: 1분(워치 WAV 조각과 같다). 끊기면 파일 하나를 처음부터 다시 보내므로(전송 설계 §2) 작게(약 1.8 MB) 나눈다. */
        const val PART_NS = 60_000_000_000L

        /** 이만큼(받은 시각)마다 버퍼를 파일로 내보낸다. */
        const val FLUSH_NS = 1_000_000_000L

        /** 버퍼 크기(문자). 워치 4종 × 100 Hz × 약 75자 = 초당 약 30,000자라 2초 치다. */
        const val BUFFER_CHARS = 64 * 1024

        private const val ROW_CHARS = 128
    }
}
