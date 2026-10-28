import unittest

import profiles  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

N = profiles.N


def received(delays, echo_gain=0.2, echo_extra=40, moving=False, seed=0):
    """줄마다 직접 경로가 delays[r]샘플 늦게 들어온 녹음. 처프가 쉬지 않고 반복되므로 줄 하나는 송신 처프를 원형으로 민 것이다.
    직접 경로 echo_extra칸 뒤에 메아리 하나(moving이면 10줄마다 1칸씩 멀어진다)와 잡음을 더한다."""
    tx = profiles.tx_template("hann", 0.5)
    rng = np.random.default_rng(seed)
    rows = []
    for r, d in enumerate(delays):
        e = d + echo_extra + (r // 10 if moving else 0)
        rows.append(np.roll(tx, d) + echo_gain * np.roll(tx, e) + rng.normal(0, 30.0, N))
    return np.concatenate(rows), tx


def reference(x, tx, shift):
    """따로 쓴 기준 구현: 대역 필터를 직접 설계하고, 원형 상관은 순환 행렬 곱, 칸 옮기기는 줄마다 np.roll, 3줄 중앙값은 줄마다 np.median."""
    y = sosfiltfilt(butter(5, [18_000, 21_000], btype="bandpass", fs=48_000, output="sos"), x)
    rows = y[: len(y) // N * N].reshape(-1, N)
    circulant = np.stack([np.roll(tx, k) for k in range(N)])  # [k, m] = tx[(m − k) mod N]
    corr = rows @ circulant.T  # [r, k] = Σ_j 줄[(j + k) mod N]·tx[j]
    rolled = np.stack([np.roll(c, 300 - shift) for c in corr])
    last = len(rolled) - 1
    med = np.stack([np.median(rolled[[max(r - 1, 0), r, min(r + 1, last)]], axis=0) for r in range(len(rolled))])
    return med.T, (np.abs(med[1:]) - np.abs(med[:-1])).T


class ProfilesTest(unittest.TestCase):
    def test_direct_path_survives_90_sample_jumps(self):
        base = 123
        delays = np.full(250, base)
        delays[50] = base + 90  # 한 줄 튐: 무시
        delays[100:102] = base + 90  # 두 줄 튐: 무시
        delays[150:] = base + 90  # 여기서부터 옮겨 감: 세 줄째에 따라가고 그 세 줄부터 새 칸
        x, tx = received(delays)
        p = profiles.compute(x, tx)
        expected = np.where(np.arange(250) >= 150, base + 90, base)
        np.testing.assert_array_equal(expected, p.direct)
        np.testing.assert_array_equal(np.flatnonzero(~p.ok), [50, 100, 101])  # 무시한 튐 줄은 쓰지 않게 표시
        peak = np.abs(p.raw).argmax(axis=0)
        steady = np.ones(250, dtype=bool)
        steady[[100, 101]] = False  # 두 줄 튐은 3줄 중앙값으로도 남는다(한 줄 튐은 중앙값이 지운다)
        self.assertTrue(np.all(peak[steady] == 300), np.flatnonzero(peak != 300))

    def test_profiles_match_reference(self):
        x, tx = received(np.full(40, 123), moving=True)
        p = profiles.compute(x, tx)
        raw, diff = reference(x, tx, 123)
        self.assertEqual((600, 40), p.raw.shape)
        self.assertEqual((600, 39), p.diff.shape)
        scale = np.abs(raw).max()
        np.testing.assert_allclose(p.raw, raw, rtol=0, atol=1e-9 * scale)
        np.testing.assert_allclose(p.diff, diff, rtol=0, atol=1e-9 * scale)
        self.assertGreater(np.abs(diff).max(), 1e-3 * scale)  # 메아리가 움직여 차분이 0만은 아니다


if __name__ == "__main__":
    unittest.main()
