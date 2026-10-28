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


class BlockSplitTest(unittest.TestCase):
    @staticmethod
    def synthetic():
        """세션 둘(약 225 s · 150 s, 프레임 간격 0.03–0.07 s), 시각 순서를 섞은 (t_ns, session)."""
        rng = np.random.default_rng(9)
        t, session = [], []
        for s, n in enumerate((4500, 3000)):
            t.append((np.cumsum(rng.uniform(0.03, 0.07, n)) * 1e9 + s * 1e12).astype(np.int64))
            session.append(np.full(n, s))
        t, session = np.concatenate(t), np.concatenate(session)
        order = rng.permutation(len(t))
        return t[order], session[order]

    def test_pattern_is_14_3_3_spread_out_and_offset_rotates_it(self):
        roles = dataset.block_roles(40)
        self.assertEqual([14, 3, 3], [int((roles[:20] == r).sum()) for r in range(3)])
        np.testing.assert_array_equal(roles[:20], roles[20:])  # mod 20으로 되풀이
        held = roles != 0
        self.assertFalse((held[1:] & held[:-1]).any())  # 검증·시험 블록이 연달아 붙지 않는다
        np.testing.assert_array_equal(dataset.block_roles(20, 3), np.roll(dataset.block_roles(20), -3))  # offset은 무늬를 블록 단위로 돌린다

    def test_parts_do_not_overlap_and_other_sets_stay_two_seconds_apart(self):
        t, session = self.synthetic()
        parts = dataset.block_split(t, session)
        everything = np.concatenate(parts)
        self.assertEqual(len(everything), len(np.unique(everything)))  # 겹침 없음
        tag = np.full(len(t), -1)
        for r, part in enumerate(parts):
            tag[part] = r
        for s in (0, 1):
            idx = np.flatnonzero((session == s) & (tag >= 0))
            idx = idx[np.argsort(t[idx])]
            near = np.diff(t[idx]) < 2e9
            self.assertFalse((near & (np.diff(tag[idx]) != 0)).any())  # 이웃한 두 프레임이 2 s 안이면 같은 자료다
            half = (t[idx].min() + t[idx].max()) / 2
            for r in range(3):  # 세 자료가 모두 세션 앞쪽 반과 뒤쪽 반에 걸친다
                self.assertTrue(((tag[idx] == r) & (t[idx] < half)).any() and ((tag[idx] == r) & (t[idx] >= half)).any())

    def test_only_frames_two_seconds_after_a_role_change_are_dropped(self):
        t, session = self.synthetic()
        block, role, keep = dataset.block_assign(t, session)
        for s in (0, 1):
            idx = np.flatnonzero(session == s)
            idx = idx[np.argsort(t[idx])]
            ts, rs, bs = t[idx], role[idx].astype(int), block[idx]
            self.assertEqual(0, bs[0])  # 블록은 그 세션의 첫 프레임부터 센다
            into = (ts - ts[0]) / 1e9 - bs * dataset.BLOCK_SEC  # 블록 시작부터의 s
            roles = dataset.block_roles(int(bs.max()) + 1)
            changed = (bs > 0) & (roles[bs] != roles[np.maximum(bs - 1, 0)])  # 앞 블록과 역할이 다른 블록
            np.testing.assert_array_equal(keep[idx], ~(changed & (into < dataset.GAP_SEC)))  # 정확히 그 블록의 앞 2 s만 비운다
            lo = np.searchsorted(ts, ts - 2_000_000_000, side="right")  # 각 프레임의 (t − 2 s, t) 안 앞 프레임들
            other = np.zeros(len(idx), dtype=bool)  # 그 안에 다른 역할의 프레임이 있는가
            for r in range(3):
                prefix = np.concatenate([[0], np.cumsum(rs == r)])
                other |= (rs == r) & ((np.arange(len(idx)) - lo) != prefix[np.arange(len(idx))] - prefix[lo])
            self.assertFalse((keep[idx] & other).any())  # 남긴 프레임 앞 2 s 안에는 다른 역할이 없다

    def test_shares_are_about_70_15_15_on_a_whole_period(self):
        t = np.arange(4000) * 50_000_000  # 200 s = 블록 20개, 프레임 간격 0.05 s
        train, val, test = dataset.block_split(t, np.zeros(len(t), dtype=int))
        total = len(train) + len(val) + len(test)
        self.assertTrue(0.62 <= len(train) / total <= 0.78 and 0.08 <= len(val) / total <= 0.2 and 0.08 <= len(test) / total <= 0.2)


class SessionSplitTest(unittest.TestCase):
    @staticmethod
    def frames(count, rng):
        """세션 count개(세션마다 30 s, 0.03–0.07 s 간격), 순서를 섞은 (t_ns, session)."""
        t, session = [], []
        for s in range(count):
            n = int(rng.integers(500, 700))
            t.append((np.cumsum(rng.uniform(0.03, 0.07, n)) * 1e9 + s * 1e12).astype(np.int64))
            session.append(np.full(n, s))
        t, session = np.concatenate(t), np.concatenate(session)
        order = rng.permutation(len(t))
        return t[order], session[order]

    def assertWhole(self, parts, session, sets):
        """나눈 프레임이 겹치지 않고, 세션마다 통째로 한 자료에 들어가는가."""
        everything = np.concatenate(parts)
        self.assertEqual(len(everything), len(np.unique(everything)))
        for part, want in zip(parts, sets):
            self.assertEqual(sorted(want), sorted(set(session[part].tolist())))
            for s in want:
                self.assertEqual(int((session == s).sum()), int((session[part] == s).sum()))

    def test_rewear_sessions_are_tested_and_the_latest_other_session_validates(self):
        t, session = self.frames(5, np.random.default_rng(11))
        wall = np.array([30.0, 10.0, 50.0, 20.0, 40.0])  # 시각 순서 1 · 3 · 0 · 4 · 2(세션 번호 순서가 아님)
        parts, _, sets = dataset.session_split(t, session, wall, np.array([False, True, False, False, False]))
        self.assertEqual(([3, 0, 4], [2], [1]), sets)  # 다시 찬 세션 1이 시험, 시험이 아닌 가장 늦은 세션 2가 검증
        self.assertWhole(parts, session, sets)
        parts, notes, sets = dataset.session_split(t, session, wall, np.zeros(5, dtype=bool))
        self.assertEqual(([1, 3, 0], [4], [2]), sets)  # 표시가 없으면 가장 늦은 세션이 시험, 그 앞이 검증
        self.assertTrue(any("rewear" in line for line in notes))  # 규칙을 알린다
        self.assertWhole(parts, session, sets)
        t10, s10 = self.frames(10, np.random.default_rng(12))
        _, _, sets = dataset.session_split(t10, s10, np.arange(10.0), np.zeros(10, dtype=bool))
        self.assertEqual(([0, 1, 2, 3, 4, 5, 6], [7], [8, 9]), sets)  # 세션 10개: 마지막 15%(2세션)가 시험
        _, _, sets = dataset.session_split(t, session, wall, np.zeros(5, dtype=bool), test=[0], val=[3])
        self.assertEqual(([1, 4, 2], [3], [0]), sets)  # 고른 세션을 그대로 쓴다

    def test_two_sessions_validate_on_the_training_tail_and_one_session_stops(self):
        t, session = self.frames(2, np.random.default_rng(13))
        (train, val, test), notes, sets = dataset.session_split(t, session, np.array([2.0, 1.0]), np.zeros(2, dtype=bool))
        self.assertEqual(([1], [], [0]), sets)  # 검증 세션을 따로 둘 수 없다
        self.assertTrue(set(session[test].tolist()) == {0} and set(session[np.r_[train, val]].tolist()) == {1})
        self.assertGreaterEqual(t[val].min() - t[train].max(), 2e9)  # 검증은 학습 세션의 끝, 앞 2 s 비움
        idx = np.flatnonzero(session == 1)
        idx = idx[np.argsort(t[idx])]
        k = round(0.85 * len(idx))
        np.testing.assert_array_equal(idx[:k], train)  # 학습 = 앞 85%
        np.testing.assert_array_equal(idx[t[idx] >= t[idx[k]] + 2e9], val)  # 검증 = 나머지 15% 중 경계 2 s 뒤
        self.assertEqual(int((session == 0).sum()), len(test))
        with self.assertRaises(ValueError):  # 세션 하나는 세션 단위로 나눌 수 없다
            dataset.session_split(t[session == 0], session[session == 0], np.array([2.0]), np.zeros(1, dtype=bool))


if __name__ == "__main__":
    unittest.main()
