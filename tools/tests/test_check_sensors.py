import json
import os
import tempfile
import unittest

import check_sensors
import hs_io

T0 = 1_000_000_000_000_000
STEP = 10_000_000  # 100 Hz


def info(kind, sensor_type, name, mode=0, max_rate=100.0):
    return {"kind": kind, "type": sensor_type, "name": name, "vendor": "STM", "version": 1, "periodUs": 10000, "minDelayUs": 10000,
            "maxDelayUs": 160000, "maxRateHz": max_rate, "resolution": 0.0024, "maxRange": 78.4, "fifoMax": 300, "reportingMode": mode, "wakeUp": False}


def report(listed, parts, missing=(), error=None):
    return {"columns": ",".join(hs_io.SENSOR_COLUMNS), "list": listed, "missing": list(missing),
            "parts": [{"file": f, "rows": 0} for f in parts], "counts": {}, "error": error}


def rows(kind, n, lag_ns=2_000_000, offset_ns=0, skip=()):
    """100 Hz 줄 n개. skip에 든 번호는 빠진다(빈틈). offset_ns는 이벤트 시각만 옮긴다(다른 시계)."""
    return [(T0 + k * STEP + offset_ns, T0 + k * STEP + lag_ns, kind, 0.1, 0.2, 9.8, 3) for k in range(n) if k not in skip]


def write(folder, name, value):
    with open(os.path.join(folder, name), "w", encoding="utf-8", newline="") as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False)


def csv_text(lines):
    return ",".join(hs_io.SENSOR_COLUMNS) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in lines)


class CheckSensorsTest(unittest.TestCase):
    def test_watch_recording_reports_rates_lag_gaps_and_volume(self):
        with tempfile.TemporaryDirectory() as d:
            volume = {"stream": "music", "index": 13, "max": 15, "choice": 13, "original": 7, "restored": 7, "changes": []}
            write(d, "rec.json", {"volume": volume, "sensors": report([info("acc", 1, "LSM6DSV Accelerometer"), info("gyro", 4, "LSM6DSO Gyroscope")],
                                                                    ["sensors-000.csv"], missing=["mag"])})
            write(d, "sensors-000.csv", csv_text(rows("acc", 201) + rows("gyro", 201, skip=range(100, 110))))
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            text = "\n".join(lines)
            self.assertIn("녹음 볼륨: 고른 값 13, 녹음 동안 13/15, 원래 7, 되돌린 뒤 7", text)
            self.assertIn("센서 acc: LSM6DSV Accelerometer (STM), 최대 100 Hz, 분해능 0.0024, 건 주기 10000 µs", text)
            self.assertIn("기기에 없는 종류: mag", text)
            self.assertIn("acc: 201줄, 100.0 Hz, 받은 시각 − 이벤트 시각 중앙 2.0 ms · 99% 2.0 ms · 최대 2.0 ms, 빈틈 0개", text)
            self.assertIn("gyro: 191줄,", text)
            self.assertIn("빈틈 1개", text)  # 자이로 10줄이 빠진 곳

    def test_other_clock_missing_part_silent_sensor_and_volume_failures_are_problems(self):
        with tempfile.TemporaryDirectory() as d:
            volume = {"stream": "music", "index": 7, "max": 15, "choice": 13, "original": 7, "restored": 9, "changes": [[T0, 8]]}
            listed = [info("acc", 1, "Acc"), info("mag", 2, "Mag")]
            write(d, "rec.json", {"volume": volume, "sensors": report(listed, ["sensors-000.csv", "sensors-001.csv"], error="센서 등록 실패: gyro")})
            write(d, "sensors-000.csv", csv_text(rows("acc", 50, offset_ns=-3_600_000_000_000)))  # 이벤트 시각이 한 시간 다른 시계
            lines, problems = check_sensors.check(d)
            text = "\n".join(problems)
            self.assertIn("센서 기록 오류: 센서 등록 실패: gyro", text)
            self.assertIn("기록에 있는데 없는 조각: sensors-001.csv", text)
            self.assertIn("mag: 기기가 준 센서인데 줄이 없음", text)
            self.assertIn("acc: 이벤트 시각이 elapsedNs 기준이 아님", text)
            self.assertIn("녹음 볼륨을 고른 값(13)에 맞추지 못함(녹음 동안 7)", text)
            self.assertIn("원래 볼륨(7)으로 되돌리지 못함(되돌린 뒤 9)", text)
            self.assertIn("녹음 중 바뀜 1번", "\n".join(lines))

    def test_phone_session_reports_context_and_does_not_count_light_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"rewear": True, "context": {"place": "집 책상", "posture": "sitting", "armSupport": "desk", "strap": None},
                    "sensors": report([info("acc", 1, "Acc"), info("light", 5, "Light", mode=1, max_rate=None)], ["phone_sensors-000.csv"])}
            write(d, "meta.json", meta)
            light = [(T0, T0 + 1_000_000, "light", 120.0, "", "", 3), (T0 + 2_000_000_000, T0 + 2_001_000_000, "light", 80.0, "", "", 3),
                     (T0 + 2_100_000_000, T0 + 2_101_000_000, "light", 81.0, "", "", 3)]
            write(d, "phone_sensors-000.csv", csv_text(rows("acc", 300) + light))
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            text = "\n".join(lines)
            self.assertIn("세션 정보: 장소 집 책상 · 자세 앉음 · 팔 책상 · 줄 — · 워치 다시 참", text)
            self.assertIn("센서 light: Light (STM), 최대 — Hz", text)
            light_line = next(line for line in lines if line.startswith("light:"))
            self.assertNotIn("빈틈", light_line)  # 바뀔 때만 내는 센서는 빈틈을 세지 않는다

    def test_folder_from_before_sensors_is_not_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "rec.json", {"volume": {"stream": "music", "index": 11, "max": 15}})
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems)
            self.assertIn("녹음 볼륨: 그대로(미디어 볼륨 11/15)", lines)
            self.assertIn("센서 기록 없음(이 기능 전 녹음·세션이거나 센서 기록을 끈 녹음)", lines)

    def test_folder_without_rec_or_meta_is_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            _, problems = check_sensors.check(d)
            self.assertEqual(1, len(problems))


if __name__ == "__main__":
    unittest.main()
