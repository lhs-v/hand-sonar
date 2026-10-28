import unittest

import markers  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402

RATE = markers.RATE


def marker_at(n, start, amplitude):
    """길이 n 신호에 표지음을 소수 샘플 start에서 시작하게 식으로 바로 그린다(샘플 사이 지연도 정확하다)."""
    u = np.arange(n) - start
    inside = (u >= 0) & (u <= markers.SAMPLES - 1)
    t = u / RATE
    k = (markers.END_HZ - markers.START_HZ) / (markers.SAMPLES / RATE)
    wave = np.sin(2 * np.pi * (markers.START_HZ * t + 0.5 * k * t * t)) * (0.5 - 0.5 * np.cos(2 * np.pi * u / (markers.SAMPLES - 1)))
    return np.where(inside, wave, 0.0) * amplitude


def watch_chirp(n, amplitude):
    """워치 처프(18→21 kHz, 600샘플, 한 창)를 쉬지 않고 반복한 신호. 표지음 대역 바로 아래의 센 소리다."""
    i = np.arange(600)
    t = i / RATE
    one = np.sin(2 * np.pi * (18_000 * t + 0.5 * (3_000 / (600 / RATE)) * t * t)) * (0.5 - 0.5 * np.cos(2 * np.pi * i / 599))
    return np.tile(one, n // 600 + 1)[:n] * amplitude


class MarkerTimingTest(unittest.TestCase):
    def test_sub_sample_start_is_within_a_tenth_of_a_millisecond(self):
        rng = np.random.default_rng(7)
        n = RATE  # 1 s
        for start in (20_000.0, 20_000.25, 20_000.5, 20_000.73, 31_234.99):
            x = watch_chirp(n, 3_000.0) + marker_at(n, start, 300.0) + rng.normal(0, 30.0, n)
            hit = markers.find(x)
            self.assertIsNotNone(hit, start)
            self.assertGreater(hit.ratio, markers.MIN_RATIO)
            error_ms = abs(hit.sample - start) / RATE * 1e3
            self.assertLess(error_ms, 0.1, f"시작 {start}: 찾음 {hit.sample:.3f}, 오차 {error_ms:.4f} ms")


if __name__ == "__main__":
    unittest.main()
