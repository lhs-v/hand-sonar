import unittest

import hs_io  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402

RATE = hs_io.RATE
T0 = 1_000_000_000_000_000


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


if __name__ == "__main__":
    unittest.main()
