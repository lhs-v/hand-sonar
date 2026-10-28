import unittest

import dataset  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402


class WindowTest(unittest.TestCase):
    def test_window_ends_at_the_paired_row_and_diff_columns_are_one_behind(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(600, 400)) * 1e9
        diff = np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])  # profiles.py와 같은 정의: 차분 열 j = |원본 j+1| − |원본 j|
        rows = np.array([96, 250, 399])
        raw_cols, diff_cols = dataset.window_columns(rows)
        np.testing.assert_array_equal(raw_cols[:, [0, -1]], [[1, 96], [155, 250], [304, 399]])  # 원본 열 r−95 … r
        np.testing.assert_array_equal(diff_cols[:, [0, -1]], [[0, 95], [154, 249], [303, 398]])  # 차분 열 r−96 … r−1
        x = dataset.windows(raw, diff, rows)
        self.assertEqual((3, 2, 96, 60), x.shape)
        for i, r in enumerate(rows):
            block = raw[300:360, r - 95: r + 1]
            change = np.abs(block) - np.abs(raw[300:360, r - 96: r])  # 창 줄마다 그 원본 줄로 들어온 변화
            np.testing.assert_allclose(x[i, 0], dataset.standardize(block.T), atol=1e-5)
            np.testing.assert_allclose(x[i, 1], dataset.standardize(change.T), atol=1e-5)
            self.assertAlmostEqual(0.0, float(x[i, 1].mean()), places=5)
            self.assertAlmostEqual(1.0, float(x[i, 1].std()), places=5)

    def test_window_needs_every_row_the_diff_channel_touches(self):
        ok = np.ones(400, dtype=bool)
        ok[54] = False
        # r = 150은 원본 열 55…150만 쓰지만 첫 차분 열 54가 원본 줄 54를 쓴다
        np.testing.assert_array_equal(dataset.window_ok(ok, [95, 149, 150, 151, 399, 400]), [False, False, False, True, True, False])


class TimeSplitTest(unittest.TestCase):
    def test_parts_do_not_overlap_and_keep_two_second_gaps(self):
        rng = np.random.default_rng(8)
        t, session = [], []
        for s, n in enumerate((900, 600)):
            times = np.cumsum(rng.uniform(0.03, 0.07, n)) * 1e9 + s * 1e12
            times[400:] += 5e9  # 카메라가 5 s 끊긴 자리
            t.append(times.astype(np.int64))
            session.append(np.full(n, s))
        t, session = np.concatenate(t), np.concatenate(session)
        order = rng.permutation(len(t))  # 시각 순서가 아니어도 된다
        t, session = t[order], session[order]
        train, val, test = dataset.time_split(t, session)
        everything = np.concatenate([train, val, test])
        self.assertEqual(len(everything), len(np.unique(everything)))  # 겹침 없음
        for s, n in enumerate((900, 600)):
            a, b, c = (t[part[session[part] == s]] for part in (train, val, test))
            self.assertEqual(round(0.7 * n), len(a))
            self.assertTrue(len(b) > 0 and len(c) > 0)
            self.assertGreaterEqual(b.min() - a.max(), 2e9)
            self.assertGreaterEqual(c.min() - b.max(), 2e9)


if __name__ == "__main__":
    unittest.main()
