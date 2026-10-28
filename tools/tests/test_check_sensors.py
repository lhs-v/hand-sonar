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


def report(listed, parts, missing=(), error=None, others=()):
    return {"columns": ",".join(hs_io.SENSOR_COLUMNS), "list": listed, "missing": list(missing), "others": list(others),
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


def csv_text(lines, header=hs_io.SENSOR_COLUMNS):
    return ",".join(header) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in lines)


def volume(**changes):
    v = {"stream": "music", "index": 13, "max": 15, "choice": 13, "original": 7, "restored": 7, "changes": [], "target": 13, "outputs": [],
         "outputChanges": [], "skipped": None, "restoreSkipped": None, "leftover": None}
    v.update(changes)
    return v


class CheckSensorsTest(unittest.TestCase):
    def test_watch_recording_reports_rates_lags_clock_gaps_and_volume(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": volume(),
                                  "sensors": report([info("acc", 1, "LSM6DSV Accelerometer"), info("gyro", 4, "LSM6DSO Gyroscope")],
                                                    ["sensors-000.csv"], missing=["mag"], others=[info("gyro", 4, "Corrected Gyroscope Sensor")])})
            write(d, "sensors-000.csv", csv_text(rows("acc", 201) + rows("gyro", 201, skip=range(100, 110))))
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            text = "\n".join(lines)
            self.assertIn("녹음 볼륨: 고른 값 13, 녹음 동안 13/15, 원래 7, 되돌린 뒤 7", text)
            self.assertIn("센서 acc: LSM6DSV Accelerometer (STM), 최대 100 Hz, 분해능 0.0024, 건 주기 10000 µs", text)
            self.assertIn("기기에 없는 종류: mag", text)
            self.assertIn("같은 종류의 다른 센서: gyro Corrected Gyroscope Sensor (STM)", text)
            self.assertIn("시계: 연속 센서 받은 시각 − 이벤트 시각 1% 값 2.0 ms → 이벤트 시각을 그대로 씀", text)
            self.assertIn("acc: 201줄, 100.0 Hz, 받은 시각 − 이벤트 시각 1% 2.0 · 중앙 2.0 · 99% 2.0 · 최대 2.0 ms, 빈틈 0개", text)
            self.assertIn("gyro: 191줄,", text)
            self.assertIn("빈틈 1개", text)  # 자이로 10줄이 빠진 곳

    def test_other_clock_missing_part_silent_sensor_and_volume_failures_are_problems(self):
        with tempfile.TemporaryDirectory() as d:
            v = volume(index=7, restored=9, changes=[[T0, 8]])
            listed = [info("acc", 1, "Acc"), info("mag", 2, "Mag")]
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": v,
                                  "sensors": report(listed, ["sensors-000.csv", "sensors-001.csv"], error="센서 등록 실패: gyro")})
            write(d, "sensors-000.csv", csv_text(rows("acc", 50, offset_ns=-3_600_000_000_000)))  # 이벤트 시각이 한 시간 다른 시계
            lines, problems = check_sensors.check(d)
            text = "\n".join(problems)
            self.assertIn("센서 기록 오류: 센서 등록 실패: gyro", text)
            self.assertIn("기록에 있는데 없는 조각: sensors-001.csv", text)
            self.assertIn("mag: 기기가 준 센서인데 줄이 없음", text)
            self.assertIn("이벤트 시각이 elapsedNs 기준이 아님", text)
            self.assertIn("녹음 볼륨을 고른 값(13)에 맞추지 못함(녹음 동안 7)", text)
            self.assertIn("원래 볼륨(7)으로 되돌리지 못함(되돌린 뒤 9)", text)
            self.assertIn("녹음 중 볼륨이 바뀜 1번", "\n".join(lines))

    def test_external_output_skips_and_leftover_are_reported_not_counted_as_failures(self):
        with tempfile.TemporaryDirectory() as d:
            v = volume(index=9, original=9, restored=None, target=None, skipped="external_output", outputs=["bluetooth_a2dp"],
                       outputChanges=[[T0, []], [T0 + 1, ["bluetooth_a2dp"]]], restoreSkipped=None,
                       leftover={"original": 4, "applied": 15, "restored": None, "skipped": "changed"})
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": v})
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)  # 걸지 않은 것은 기록대로 한 것이다(못 맞춤이 아니다)
            text = "\n".join(lines)
            self.assertIn("녹음 볼륨을 걸지 않음: 외부 출력이 이어져 있어(그대로 녹음)", text)
            self.assertIn("시작 때 외부 출력: bluetooth_a2dp — 볼륨 값은 그 기기 것", text)
            self.assertIn("녹음 중 외부 출력이 바뀜 2번(끝: bluetooth_a2dp)", text)
            self.assertIn("앞 녹음이 남긴 볼륨(원래 4, 맞춘 15): 그사이 볼륨이 바뀌어 그대로 둠", text)
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": volume(restored=None, restoreSkipped="external_output")})
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            self.assertIn("정지 때 되돌리지 않음: 외부 출력이 이어져 있어 — 스피커로 돌아온 뒤 워치 앱을 열 때 되돌림", "\n".join(lines))

    def test_keep_shows_changes_and_an_unfinished_recording_says_restore_is_pending(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": {"stream": "music", "index": 11, "max": 15, "changes": [[T0, 9], [T0 + 1, 11]]}})
            lines, _ = check_sensors.check(d)
            self.assertIn("녹음 볼륨: 그대로(미디어 볼륨 11/15)", lines)
            self.assertIn("녹음 중 볼륨이 바뀜 2번", lines)  # 그대로여도 사람이 바꾼 것은 보인다(볼륨별로 견줄 때 가려야 한다)
            write(d, "rec.json", {"stop": None, "volume": volume(restored=None)})  # 앱이 죽어 stop이 없음
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            self.assertIn("되돌린 기록 없음(녹음이 끝나지 않음 — 다음에 워치 앱을 열 때 되돌림)", lines)

    def test_phone_session_reports_context_and_light_never_trips_the_clock_check(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"rewear": True, "context": {"place": "집 책상", "posture": "sitting", "armSupport": "desk", "strap": None},
                    "sensors": report([info("acc", 1, "Acc"), info("light", 5, "Light", mode=1, max_rate=None)], ["phone_sensors-000.csv"])}
            write(d, "meta.json", meta)
            # 밝기 첫 줄은 들을 때 받은 오래된 값이라 이벤트 시각이 40 s 이르다: 바뀔 때만 내는 센서는 시계 판정·빈틈에서 뺀다
            light = [(T0 - 40_000_000_000, T0 + 1_000_000, "light", 120.0, "", "", 3), (T0 + 2_000_000_000, T0 + 2_001_000_000, "light", 80.0, "", "", 3)]
            write(d, "phone_sensors-000.csv", csv_text(rows("acc", 300) + light))
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems, lines)
            text = "\n".join(lines)
            self.assertIn("세션 정보: 장소 집 책상 · 자세 앉음 · 팔 책상 · 줄 — · 워치 다시 참", text)
            self.assertIn("센서 light: Light (STM), 최대 — Hz", text)
            light_line = next(line for line in lines if line.startswith("light:"))
            self.assertNotIn("빈틈", light_line)

    def test_folder_from_before_sensors_is_not_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": {"stream": "music", "index": 11, "max": 15}})
            lines, problems = check_sensors.check(d)
            self.assertEqual([], problems)
            self.assertIn("녹음 볼륨: 그대로(미디어 볼륨 11/15)", lines)
            self.assertIn("센서 기록 없음(이 기능 전 녹음·세션이거나 센서 기록을 끈 녹음)", lines)

    def test_unreadable_folder_is_one_problem_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            _, problems = check_sensors.check(d)
            self.assertEqual(1, len(problems))  # rec.json도 meta.json도 없음
            write(d, "rec.json", {"stop": {"reason": "user"}, "volume": volume(), "sensors": report([], ["sensors-000.csv"])})
            write(d, "sensors-000.csv", csv_text(rows("acc", 3), header=["\0" * 4]))  # 전원이 꺼져 0으로 찬 머리줄
            _, problems = check_sensors.check(d)
            self.assertEqual(1, len(problems), problems)
            self.assertIn("점검하지 못함", problems[0])


if __name__ == "__main__":
    unittest.main()
