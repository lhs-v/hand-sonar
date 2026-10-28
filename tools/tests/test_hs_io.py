import os
import tempfile
import unittest

import hs_io  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402

RATE = hs_io.RATE
T0 = 1_000_000_000_000_000


def write_csv(path, rows, header=hs_io.SENSOR_COLUMNS, tail=""):
    """센서 CSV(계획 6): 머리줄, 줄들, 그리고 줄바꿈 없는 꼬리(앱이 죽어 잘린 마지막 줄)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
        f.write(tail)


def clock_rows(n, late_ms=None, step=None, seed=1):
    """1 s마다 [위치, elapsedNs, 읽은 수](흔들림 ±0.3 ms). late_ms = {줄: 늦게 찍힌 ms}, step = (줄, ms): 그 줄 앞에서 샘플이 빠져
    그 줄부터 시각이 ms만큼 앞선다."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        t = T0 + k * 1e9 + rng.uniform(-0.3e6, 0.3e6) + (late_ms or {}).get(k, 0.0) * 1e6
        if step and k >= step[0]:
            t += step[1] * 1e6
        rows.append([k * RATE, int(t), k * RATE])
    return rows


class ClockTest(unittest.TestCase):
    def test_one_row_spikes_are_dropped_not_steps(self):
        # 실데이터 폰 recordClock: 한 줄이 +10.8 ms 늦고 다음 줄이 +2.75 ms까지 되돌아옴. 25 ms 튐 하나는 문턱(20 ms)을 넘어도 계단이 아니다.
        clock = hs_io.Clock(clock_rows(220, late_ms={59: 10.8, 60: 2.75, 120: 25.0}))
        self.assertEqual([], clock.steps)
        self.assertEqual([59 * RATE, 120 * RATE], [p for p, _ in clock.spikes])
        frames = np.linspace(0, 219 * RATE, 500)
        error_ms = np.abs(clock.ns(frames) - (T0 + frames / RATE * 1e9)) / 1e6
        self.assertLess(float(error_ms.max()), 0.05)

    def test_step_is_a_span_between_clock_rows(self):
        # 99번 줄과 100번 줄 사이 어딘가에서 샘플이 빠지면(시각 +25 ms) 계단 구간은 (99 s, 100 s]이고, 그 구간과 겹치는 범위는 모두 계단을 본다.
        clock = hs_io.Clock(clock_rows(220, step=(100, 25.0)))
        self.assertEqual(1, len(clock.steps))
        a, b, ns = clock.steps[0]
        self.assertEqual((99 * RATE, 100 * RATE), (a, b))
        self.assertAlmostEqual(25e6, ns, delta=1e6)
        self.assertTrue(clock.steps_between(10 * RATE, 99.5 * RATE))  # 끝이 계단 구간 안
        self.assertTrue(clock.steps_between(99.5 * RATE, 150 * RATE))  # 시작이 계단 구간 안
        self.assertFalse(clock.steps_between(100 * RATE, 150 * RATE))
        self.assertFalse(clock.steps_between(10 * RATE, 99 * RATE))
        after = np.linspace(100 * RATE, 219 * RATE, 100)
        error_ms = np.abs(clock.ns(after) - (T0 + after / RATE * 1e9 + 25e6)) / 1e6
        self.assertLess(float(error_ms.max()), 0.05)


class SensorReadTest(unittest.TestCase):
    def test_parts_join_in_number_order_and_cut_rows_are_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            # 조각은 번호 순으로 잇는다(1000번은 1번 뒤). 1번 조각은 마지막 줄이 잘렸고(앱이 죽음) 0번 조각에는 열이 모자란 줄이 있다.
            write_csv(os.path.join(d, "sensors-001.csv"), [(T0 + 20, T0 + 25, "acc", 0.5, -9.75, "1.0E-4", 3)], tail="1000000000000030,10")
            write_csv(os.path.join(d, "sensors-000.csv"), [(T0, T0 + 5, "acc", 1.0, 2.0, 3.0, 3), (T0 + 10, T0 + 12, "gyro", 0.1, 0.2, 0.3, 2),
                                                           (T0 + 11, T0 + 13, "acc", 1.0)])
            write_csv(os.path.join(d, "sensors-1000.csv"), [(T0 + 30, T0 + 31, "light", 120.0, "", "", 0)])
            open(os.path.join(d, "sensors-002.csv"), "w").close()  # 머리줄도 내보내기 전에 앱이 죽은 조각
            write_csv(os.path.join(d, "phone_sensors-000.csv"), [(1, 2, "acc", 0, 0, 0, 3)])  # 다른 기기의 이름은 섞지 않는다
            write_csv(os.path.join(d, "sensors-003.csv.tmp"), [(1, 2, "acc", 0, 0, 0, 3)])  # 쓰다 남은 파일도
            log = hs_io.read_sensors(d)
            self.assertEqual(["sensors-000.csv", "sensors-001.csv", "sensors-002.csv", "sensors-1000.csv"], [os.path.basename(p) for p in log.files])
            self.assertEqual(2, log.dropped)
            self.assertEqual(["acc", "gyro", "light"], list(log.tracks))  # 처음 나온 차례
            acc = log.tracks["acc"]
            self.assertEqual([T0, T0 + 20], acc.t_ns.tolist())
            self.assertEqual([5, 5], acc.lag_ns.tolist())
            np.testing.assert_allclose(acc.values, [[1.0, 2.0, 3.0], [0.5, -9.75, 1e-4]])
            self.assertEqual([3, 3], acc.accuracy.tolist())
            light = log.tracks["light"]
            self.assertEqual(120.0, light.values[0, 0])
            self.assertTrue(np.isnan(light.values[0, 1:]).all())  # 밝기는 x만
            phone = hs_io.read_sensors(d, hs_io.PHONE_SENSORS)
            self.assertEqual(["phone_sensors-000.csv"], [os.path.basename(p) for p in phone.files])

    def test_no_sensor_files_means_an_older_recording(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(hs_io.read_sensors(d))
            self.assertIsNone(hs_io.read_sensors(d, hs_io.PHONE_SENSORS))
            self.assertIsNone(hs_io.read_sensors(os.path.join(d, "없음")))

    def test_another_header_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            write_csv(os.path.join(d, "sensors-000.csv"), [(1, 2, "acc", 0, 0, 0)], header=["t_ns", "sensor", "x", "y", "z", "accuracy"])
            with self.assertRaises(ValueError):
                hs_io.read_sensors(d)

    def test_event_clock_is_judged_per_device_from_the_floor_of_continuous_sensor_lags(self):
        t = T0 + np.arange(200, dtype=np.int64) * 10_000_000
        z, a = np.zeros((200, 3)), np.full(200, 3)
        lag = np.full(200, 3_000_000, dtype=np.int64)
        lag[::10] = 300_000_000  # 가끔 300 ms 늦게 받음(배치·스레드 지연): 중앙값이 아니라 바닥(1%)을 보므로 그대로 믿는다
        stale = np.array([40_000_000_000, 1_000_000, 1_000_000])  # 밝기 첫 줄은 들을 때 받은 오래된 값(40 s 전 시각)
        light = hs_io.SensorTrack("light", t[:3], t[:3] + stale, z[:3], a[:3])
        log = hs_io.SensorLog([], {"acc": hs_io.SensorTrack("acc", t, t + lag, z, a), "light": light}, 0)
        self.assertAlmostEqual(3e6, log.lag_floor_ns)
        self.assertTrue(log.event_clock)  # 밝기(바뀔 때만)는 판정에서 뺀다
        self.assertIs(log.tracks["acc"].t_ns, log.ns("acc"))
        self.assertAlmostEqual(100.0, log.tracks["acc"].rate_hz(), places=9)
        # 다른 시계: 이벤트 시각이 늘 300 ms 앞섬(바닥 301 ms) → 이벤트 시각 + 바닥(이벤트 사이 간격은 그대로)
        skewed = hs_io.SensorLog([], {"gyro": hs_io.SensorTrack("gyro", t - 300_000_000, t + 1_000_000, z, a)}, 0)
        self.assertFalse(skewed.event_clock)
        np.testing.assert_array_equal(t + 1_000_000, skewed.ns("gyro"))
        ahead = hs_io.SensorLog([], {"acc": hs_io.SensorTrack("acc", t + 20_000_000, t, z, a)}, 0)  # 이벤트 시각이 받은 시각보다 20 ms 뒤(미래)
        self.assertFalse(ahead.event_clock)
        np.testing.assert_array_equal(t, ahead.ns("acc"))
        only_light = hs_io.SensorLog([], {"light": light}, 0)
        self.assertIsNone(only_light.lag_floor_ns)
        self.assertTrue(only_light.event_clock)  # 연속 센서가 없으면 판정하지 않고 이벤트 시각 그대로
        empty = hs_io.SensorTrack("acc", t[:0], t[:0], z[:0], a[:0])
        self.assertTrue(np.isnan(empty.rate_hz()))


class MetaFieldsTest(unittest.TestCase):
    def test_session_context_and_rewear_with_defaults_for_older_sessions(self):
        self.assertEqual({"place": None, "posture": None, "armSupport": None, "strap": None, "rewear": False}, hs_io.session_context({"rewear": False}))
        meta = {"rewear": True, "context": {"place": "집 책상", "posture": "sitting", "armSupport": "air", "strap": "tight"}}
        self.assertEqual({"place": "집 책상", "posture": "sitting", "armSupport": "air", "strap": "tight", "rewear": True}, hs_io.session_context(meta))

    def test_recording_volume_with_defaults_for_older_recordings(self):
        empty = {"target": None, "restoreSkipped": None, "skipped": None, "outputs": [], "outputChanges": [], "leftover": None}
        old = hs_io.recording_volume({"volume": {"stream": "music", "index": 11, "max": 15}})
        self.assertEqual({"index": 11, "max": 15, "choice": None, "original": 11, "restored": None, "changes": [], **empty}, old)
        rec = {"volume": {"stream": "music", "index": 13, "max": 15, "choice": 13, "original": 7, "restored": 7, "changes": [[5, 12]],
                          "target": 13, "outputs": [], "outputChanges": [[6, ["bluetooth_a2dp"]]], "skipped": None, "restoreSkipped": "external_output",
                          "leftover": {"original": 4, "applied": 15, "restored": None, "skipped": "changed"}}}
        self.assertEqual({"index": 13, "max": 15, "choice": 13, "original": 7, "restored": 7, "changes": [[5, 12]], "target": 13,
                          "restoreSkipped": "external_output", "skipped": None, "outputs": [], "outputChanges": [[6, ["bluetooth_a2dp"]]],
                          "leftover": {"original": 4, "applied": 15, "restored": None, "skipped": "changed"}}, hs_io.recording_volume(rec))
        self.assertEqual({"index": None, "max": None, "choice": None, "original": None, "restored": None, "changes": [], **empty},
                         hs_io.recording_volume({}))


if __name__ == "__main__":
    unittest.main()
