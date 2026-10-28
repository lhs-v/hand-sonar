import unittest

import verify_sync  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402


def motion(t_ns, bursts):
    """손 움직임 세기: 시각 t_ns(ns)에서 가우스 덩어리들의 합. bursts = [(가운데 s, 폭 s, 높이)]."""
    t = np.asarray(t_ns, dtype=np.float64) / 1e9
    return sum(h * np.exp(-0.5 * ((t - c) / w) ** 2) for c, w, h in bursts)


class VerifySyncTest(unittest.TestCase):
    def test_finds_the_injected_delay(self):
        rng = np.random.default_rng(5)
        bursts = [(c, rng.uniform(0.1, 0.4), rng.uniform(0.5, 2.0)) for c in np.sort(rng.uniform(1, 59, 40))]
        sound_ns = np.arange(0, 60e9, 12.5e6)  # 프로파일 줄(80 Hz)
        energy = 0.1 + motion(sound_ns, bursts) + rng.uniform(0, 0.05, len(sound_ns))  # 소리 세기는 |차분| 합이라 늘 양수
        for delay_ms in (47.0, -60.0):
            cam_ns = np.sort(np.arange(0, 60e9, 1e9 / 30) + rng.uniform(-2e6, 2e6, 1800))  # 30 fps, 시각 ±2 ms 흔들림
            speed = 0.1 + motion(cam_ns - delay_ms * 1e6, bursts) + rng.uniform(0, 0.05, len(cam_ns))  # 카메라 시각이 delay만큼 늦게 찍힘
            lag = verify_sync.estimate_lag(sound_ns, energy, cam_ns, speed)
            self.assertAlmostEqual(delay_ms, lag.lag_ns / 1e6, delta=5.0)  # 줄 간격 12.5 ms에 포물선 보간이라 잡음 없이도 1 ms 안팎 치우친다(설계 기준은 33 ms)
            self.assertGreater(lag.peak, 0.8)


if __name__ == "__main__":
    unittest.main()
