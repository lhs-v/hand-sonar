import unittest

import profiles  # 사용자 설치 numpy 경로도 여기서 붙는다
import signal_check

import numpy as np  # noqa: E402


class SignalCheckTest(unittest.TestCase):
    def test_speed_energy_correlation_peaks_at_the_paired_row(self):
        # 프레임마다 짝 줄이 4줄씩 늘고, 그 줄로 들어오는 차분 세기(열 r − 1)가 관절 속도를 따라간다
        rng = np.random.default_rng(0)
        m = 400
        y = np.cumsum(rng.normal(0, 1, (m, 60)) * rng.uniform(0.1, 2.0, (m, 1)), axis=0)
        speed = np.linalg.norm(np.diff(y, axis=0), axis=1)
        rows = 10 + 4 * np.arange(m)
        energy = rng.uniform(0.5, 1.5, rows[-1] + 20)
        energy[rows[1:] - 1] = speed
        t = np.arange(m) * 50_000_000
        corr = signal_check.speed_energy_corr(y, t, rows, energy)
        self.assertGreater(corr[8], 0.99)  # 지연 0(LAGS의 가운데)
        self.assertLess(max(corr[:8] + corr[9:]), 0.5)

    def test_frames_after_a_gap_are_not_used(self):
        y = np.zeros((3, 60))
        y[2] = 1.0  # 큰 움직임이지만 앞 프레임과 0.2 s 떨어져 있어 쓰지 않는다
        t = np.array([0, 50_000_000, 250_000_000])
        energy = np.ones(50)
        with np.errstate(invalid="ignore"), self.assertWarns(RuntimeWarning):
            corr = signal_check.speed_energy_corr(y, t, np.array([10, 14, 30]), energy, lags=[0])
        self.assertTrue(np.isnan(corr[0]))  # 한 쌍뿐이라 상관을 정할 수 없다

    def test_motion_contrast_is_p95_over_median(self):
        e = np.ones(100)
        e[-10:] = 6.0
        self.assertAlmostEqual(6.0, signal_check.motion_contrast(e))

    def test_peak_width_separates_hann_from_tukey(self):
        def track(window):
            tx = profiles.tx_template(window, 0.5)
            p = profiles.compute(np.tile(tx, 40), tx)
            return signal_check.peak_track(p.raw[signal_check.PEAK_BINS, 5:35])
        pos_t, width_t = track("tukey")
        _, width_h = track("hann")
        self.assertLessEqual(np.median(width_t), 26)
        self.assertGreater(np.median(width_h), 1.5 * np.median(width_t))
        self.assertLess(np.percentile(np.abs(np.diff(pos_t)), 90), 0.01)  # 같은 처프가 되풀이되면 봉우리가 흔들리지 않는다


if __name__ == "__main__":
    unittest.main()
