import csv
import json
import os
import tempfile
import unittest

import dataset  # 사용자 설치 numpy 경로도 여기서 붙는다

import hs_io  # noqa: E402
import joints  # noqa: E402
import numpy as np  # noqa: E402
import pairs  # noqa: E402
import verify_sync  # noqa: E402

NAME = "20990101-000005"
T0 = 1_600_000_000_000_000  # 첫 카메라 프레임(폰 elapsedNs)
SEC = 1_000_000_000
FRAMES = 240  # 30 fps 8 s
ROWS = 880  # 프로파일 줄(12.5 ms): 첫 프레임 1 s 앞부터 11 s
ROW0 = T0 - SEC  # 줄 0 가운데 시각
ROW_NS = 12_500_000
LAG_MS = 7.5
NO_HAND = (20, 21, 150)  # 손이 안 보인 프레임
BIG = (40, 170)  # MediaPipe 크기가 1.3배로 튄 프레임(길이 거르기로 빠진다)
BAD_ROW = 300  # ok가 아닌 프로파일 줄(이 줄을 창에 품는 프레임이 빠진다)
CLASSES = ((0, "엄지 굽히기"), (10, "숫자 5"), (15, "손목 젖히기"), (-1, "자유 동작"))  # 2 s씩


def real_hand(n):
    """프레임 n의 실제 왼손 21점(m, 손바닥이 아래, 엄지 +x, 손가락 +y, 손등 +z). 검지가 프레임마다 다르게 굽어 프레임 번호가 손 모양에 새겨진다."""
    p = np.zeros((21, 3))
    p[1:5] = ((0.025, 0.025, -0.012), (0.045, 0.045, -0.022), (0.058, 0.065, -0.030), (0.066, 0.085, -0.035))  # 엄지는 손바닥 쪽(−z)
    for base, mcp in zip((5, 9, 13, 17), ((0.020, 0.090, 0.0), (0.0, 0.095, 0.002), (-0.017, 0.088, 0.001), (-0.033, 0.078, -0.002))):
        p[base] = mcp
        bend = 0.004 * n if base == 5 else 0.2
        for s, length in enumerate((0.040, 0.025, 0.022)):
            angle = bend * (s + 1)
            p[base + s + 1] = p[base + s] + length * np.array([0.0, np.cos(angle), -np.sin(angle)])
    return p


def file_world(n):
    """landmarks.csv에 적히는 world(거울 그림 기준, wx → −wx): 크기는 프레임마다 ±1% 흔들리고 BIG 프레임만 1.3배."""
    scale = 1.3 if n in BIG else 1.0 + 0.01 * np.sin(n)
    return real_hand(n) * scale * joints.MIRROR


def raw_profiles():
    """원본 프로파일 (600, ROWS): 열 j는 씨앗값 j의 무늬라 줄 번호가 값에 새겨진다(창이 한 줄만 어긋나도 다르다)."""
    return np.stack([np.random.default_rng(j).normal(size=600) for j in range(ROWS)], axis=1)


def write_session(data, lag_ms=LAG_MS):
    """합성 세션 하나(meta·landmarks·prompts·align·rows·profiles·sync)를 쓰고 pairs.py로 pairs.csv를 만든다. 영상은 쓸 수 없는 세션이다."""
    folder = os.path.join(data, "phone", "sessions", NAME)
    out = os.path.join(data, "processed", NAME)
    os.makedirs(folder)
    os.makedirs(out)
    meta = {"format": 1, "status": "kept", "cancelled": False, "hand": "left", "rewear": True, "camera": {"timeBase": "realtime"},
            "start": {"wallMs": 1_000, "elapsedNs": T0 - SEC}, "end": {"wallMs": 10_000, "elapsedNs": T0 + 9 * SEC},
            "stats": {"frames": FRAMES}, "video": {"ok": False, "frames": 0}}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(folder, "landmarks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frameNs", "hands", "handedness", "score"] + [f"v{i}" for i in range(126)])
        for n in range(FRAMES):
            t = T0 + n * SEC // 30
            if n in NO_HAND:
                w.writerow([t, 0, "", ""] + [""] * 126)
            else:
                w.writerow([t, 1, "Left", 0.95] + [0.5] * 63 + file_world(n).ravel().tolist())
    with open(os.path.join(folder, "prompts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["index", "segment", "classId", "name", "startNs", "endNs", "flag"])
        for i, (cls, label) in enumerate(CLASSES):
            w.writerow([i, i + 1, cls, label, T0 + 2 * i * SEC, T0 + 2 * (i + 1) * SEC, ""])
    hs_io.write_json(os.path.join(out, "align.json"), {"format": 1, "session": NAME, "method": "two"})
    ok = np.ones(ROWS, dtype=bool)
    ok[BAD_ROW] = False
    np.savez(os.path.join(out, "rows.npz"), ns=ROW0 + np.arange(ROWS, dtype=np.float64) * ROW_NS, ok=ok,
             alignSha=np.array(hs_io.sha256(os.path.join(out, "align.json"))))
    raw = raw_profiles()
    np.save(os.path.join(out, "profiles.npy"), raw)
    np.save(os.path.join(out, "diff_profiles.npy"), np.abs(raw[:, 1:]) - np.abs(raw[:, :-1]))
    write_sync(data, lag_ms)
    _, done = pairs.run(data, NAME)
    assert done
    return raw


def write_sync(data, lag_ms):
    sync = {"format": 2, "session": NAME, "lagMs": lag_ms, "peak": 0.2, "withinFrame": True, "inputs": verify_sync.inputs(data, NAME)}
    sync["usable"] = verify_sync.usable(sync)[0]
    hs_io.write_json(os.path.join(data, "processed", NAME, "sync.json"), sync)


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


class KindTest(unittest.TestCase):
    def test_calibration_hold_and_pauses_are_not_gesture_kinds(self):
        kinds = dataset.kind_of(np.array([0, 7, 16, -1, -3, -2, 3]), np.array([True, True, True, True, True, True, False]))
        np.testing.assert_array_equal([0, 1, 2, 3, 4, 4, 4], kinds)


class SessionFolderTest(unittest.TestCase):
    def test_windows_targets_and_labels_stay_paired_through_the_filters(self):
        with tempfile.TemporaryDirectory() as data:
            raw = write_session(data)
            diff = np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])
            ds, lines = dataset.build(data)
            self.assertIsNotNone(ds, "\n".join(lines))
            ns = T0 + np.arange(FRAMES) * SEC // 30
            rows = np.rint((ns - LAG_MS * 1e6 - ROW0) / ROW_NS).astype(int)  # 프레임마다 소리 시각에 가장 가까운 줄(지연 뺌)
            expect = [n for n in range(FRAMES) if n not in NO_HAND and n not in BIG and rows[n] >= 96 and not rows[n] - 96 <= BAD_ROW <= rows[n]]
            frame = np.rint((ds["frame_ns"] - T0) * 30 / SEC).astype(int)  # 시각에서 되짚은 프레임 번호
            np.testing.assert_array_equal(expect, frame)  # 손 없음·크기 튐·앞 줄 모자람·ok 아닌 줄을 품은 창만 빠진다(영상 칸이 비어도 쓴다)
            self.assertTrue(any("영상" in line for line in lines))  # 영상 조건은 거르지 않고 알리기만 한다
            np.testing.assert_array_equal(rows[frame], ds["row"])
            targets = np.stack([joints.targets(file_world(n)[None], "left")[0][0] for n in frame])
            np.testing.assert_allclose(ds["y"].reshape(-1, 20, 3), targets, atol=1e-6)  # 목표는 그 프레임의 손(검지 굽힘에 새긴 번호)
            for i, r in enumerate(rows[frame]):  # 창은 그 프레임의 짝 줄에서 끝난다(열 무늬에 새긴 줄 번호)
                np.testing.assert_allclose(ds["x"][i, 0], dataset.standardize(raw[300:360, r - 95: r + 1].T), atol=1e-5)
                np.testing.assert_allclose(ds["x"][i, 1], dataset.standardize(diff[300:360, r - 96: r].T), atol=1e-5)
            r = rows[frame[0]]
            self.assertGreater(float(np.abs(ds["x"][0, 0] - dataset.standardize(raw[300:360, r - 94: r + 2].T)).max()), 0.5)  # 한 줄 어긋나면 다르다
            np.testing.assert_array_equal([CLASSES[min(n // 60, 3)][0] for n in frame], ds["class_id"])
            np.testing.assert_array_equal(frame % 60 >= 15, ds["labelled"])  # 동작 시작 0.5 s(15프레임)는 경계라 라벨이 아니다
            self.assertFalse(ds["wrist_ok"].any())  # 워치·폰 센서 기록이 없는 세션: 손목 정답 없음
            self.assertEqual((len(frame), 3), ds["wrist"].shape)
            self.assertEqual([""], ds["wrist_method"].tolist())
            self.assertEqual([True], ds["rewear"].tolist())
            self.assertEqual([1], ds["wear"].tolist())  # 다시 찬 세션이라 새 착용(1)이다
            path = os.path.join(data, "dataset.npz")
            dataset.save(path, ds)
            saved = dataset.load(path)
            self.assertNotIn("x", saved)  # 기본은 창 없이 쓴다
            self.assertEqual(list(verify_sync.inputs(data, NAME).values()), saved["inputs"][0].tolist())
            np.testing.assert_array_equal(ds["x"], dataset.windows(raw, diff, saved["row"]))  # 창은 profiles에서 row로 다시 만든다
            dataset.save(path, ds, with_x=True)
            self.assertIn("x", dataset.load(path))

    def test_rewear_marks_come_from_every_session_folder(self):
        with tempfile.TemporaryDirectory() as data:
            for name, meta in (("A", {"status": "trash", "rewear": True, "start": {"wallMs": 500}}),
                               ("B", {"status": "kept", "rewear": False, "start": {"wallMs": 900}, "end": {"wallMs": 990}}),
                               ("C", {"status": "kept", "rewear": True, "start": {"wallMs": 1_200}}),  # 끝나지 않은 세션
                               ("D", {"status": "kept", "rewear": True})):  # 시각이 없으면 경계로 쓸 수 없다
                folder = os.path.join(data, "phone", "sessions", name)
                os.makedirs(folder)
                with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
            os.makedirs(os.path.join(data, "phone", "sessions", "E"))  # meta.json 없음
            self.assertEqual([500.0, 1_200.0], sorted(dataset.rewear_marks(data)))

    def test_hamer_ground_truth_replaces_only_the_finger_targets(self):
        drop = 200  # MediaPipe 거르기는 지나지만 HaMeR 관절이 없는 프레임
        with tempfile.TemporaryDirectory() as data:
            write_session(data)
            path = os.path.join(data, "processed", NAME, "hamer.npz")
            ns = T0 + np.arange(FRAMES) * SEC // 30
            kp = np.stack([real_hand(FRAMES - 1 - n) * joints.MIRROR for n in range(FRAMES)])  # 검지 굽힘이 거꾸로 — MediaPipe와 다른 손
            kp[list(NO_HAND) + [drop]] = np.nan
            np.savez(path, frame_ns=ns, keypoints=kp.astype(np.float32))
            mp, _ = dataset.build(data)
            hm, lines = dataset.build(data, gt="hamer")
            self.assertIsNotNone(hm, "\n".join(lines))
            frame = np.rint((mp["frame_ns"] - T0) * 30 / SEC).astype(int)
            self.assertIn(drop, frame)
            np.testing.assert_array_equal(mp["frame_ns"][frame != drop], hm["frame_ns"])  # 거르기는 MediaPipe 그대로, HaMeR 관절 없는 프레임만 더 빠진다
            kept = frame[frame != drop]
            targets = np.stack([joints.targets(kp[n][None].astype(np.float64), "left")[0][0] for n in kept])
            np.testing.assert_allclose(hm["y"].reshape(-1, 20, 3), targets, atol=1e-5)
            self.assertGreater(float(np.abs(hm["y"] - mp["y"][frame != drop]).max()), 0.1)  # MediaPipe 목표와는 다르다
            np.testing.assert_array_equal(mp["length_mm"], hm["length_mm"])  # L(mm)은 둘 다 MediaPipe
            self.assertEqual(("mediapipe", "hamer"), (str(mp["gt"]), str(hm["gt"])))
            self.assertTrue(any("정답 HaMeR" in line for line in lines) and any("HaMeR 관절 없음" in line for line in lines), "\n".join(lines))
            dataset.save(os.path.join(data, "dataset.npz"), hm)
            self.assertEqual("hamer", str(dataset.load(os.path.join(data, "dataset.npz"))["gt"]))
            np.savez(path, frame_ns=ns + 1, keypoints=kp.astype(np.float32))  # 프레임이 landmarks.csv와 다르면 섞지 않고 건너뛴다
            ds, lines = dataset.build(data, gt="hamer")
            self.assertIsNone(ds)
            self.assertTrue(any("landmarks.csv와 다름" in line for line in lines), "\n".join(lines))
            os.remove(path)
            ds, lines = dataset.build(data, gt="hamer")
            self.assertIsNone(ds)
            self.assertTrue(any("hamer.npz" in line and "hamer_keypoints.py" in line for line in lines), "\n".join(lines))
            with self.assertRaises(ValueError):
                dataset.session_data(data, NAME, gt="openpose")

    def test_pairs_made_with_an_old_lag_are_skipped(self):
        with tempfile.TemporaryDirectory() as data:
            write_session(data)
            write_sync(data, LAG_MS + 1.0)  # verify_sync.py만 다시 돌아 지연이 바뀌고 pairs.py는 다시 돌지 않은 경우(입력 지문은 그대로)
            ds, lines = dataset.build(data)
            self.assertIsNone(ds)
            self.assertTrue(any("soundNs" in line and "pairs.py" in line for line in lines), "\n".join(lines))


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

    def test_the_last_wear_is_held_out_whole_and_the_latest_other_session_validates(self):
        t, session = self.frames(5, np.random.default_rng(11))
        wall = np.array([30.0, 10.0, 50.0, 20.0, 40.0])  # 시각 순서 1 · 3 · 0 · 4 · 2(세션 번호 순서가 아님)
        # 세션 4(시각 순 넷째)에서 워치를 다시 찼다: 4와 그 뒤 2가 한 착용이고 통째로 시험이다(2만 학습에 들면 같은 착용 누수)
        parts, notes, sets = dataset.session_split(t, session, wall, np.array([False, False, False, False, True]))
        self.assertEqual(([1, 3], [0], [4, 2]), sets)  # 시험이 아닌 가장 늦은 세션 0이 검증
        self.assertTrue(any("마지막 착용" in line for line in notes), notes)
        self.assertWhole(parts, session, sets)
        # 다시 찬 첫 세션을 휴지통으로 보냈어도(자료에 없음) 그 시각(35)부터 새 착용이다
        _, _, sets = dataset.session_split(t, session, wall, np.zeros(5, dtype=bool), wear=dataset.wear_episodes(wall, np.zeros(5, dtype=bool), [35.0]))
        self.assertEqual(([1, 3], [0], [4, 2]), sets)
        # 표시가 맨 앞 세션에만 있으면 착용이 하나뿐이다: 시각 순 마지막 세션이 시험(같은 착용 안 — 알림)
        parts, notes, sets = dataset.session_split(t, session, wall, np.array([False, True, False, False, False]))
        self.assertEqual(([1, 3, 0], [4], [2]), sets)
        self.assertTrue(any("착용이 하나뿐" in line for line in notes), notes)
        parts, notes, sets = dataset.session_split(t, session, wall, np.zeros(5, dtype=bool))
        self.assertEqual(([1, 3, 0], [4], [2]), sets)  # 표시가 없으면 가장 늦은 세션이 시험, 그 앞이 검증
        self.assertTrue(any("rewear" in line for line in notes))  # 규칙을 알린다
        self.assertWhole(parts, session, sets)
        t10, s10 = self.frames(10, np.random.default_rng(12))
        _, _, sets = dataset.session_split(t10, s10, np.arange(10.0), np.zeros(10, dtype=bool))
        self.assertEqual(([0, 1, 2, 3, 4, 5, 6], [7], [8, 9]), sets)  # 세션 10개: 마지막 15%(2세션)가 시험
        _, notes, sets = dataset.session_split(t, session, wall, np.zeros(5, dtype=bool), test=[0], val=[3])
        self.assertEqual(([1, 4, 2], [3], [0]), sets)  # 고른 세션을 그대로 쓴다
        self.assertTrue(any("같은 착용의 세션 4개" in line for line in notes), notes)  # 고른 시험과 같은 착용이 학습·검증에 있음을 알린다

    def test_wear_episodes_start_at_each_rewear_mark_and_hold_out_only_the_last_wear(self):
        wall = np.arange(10.0)
        marked = np.zeros(10, dtype=bool)
        marked[[3, 7]] = True
        np.testing.assert_array_equal([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dataset.wear_episodes(wall, marked))
        np.testing.assert_array_equal([0, 0, 1, 2, 2, 2, 2, 3, 3, 3], dataset.wear_episodes(wall, marked, marks_ms=[1.5]))  # 휴지통의 표시도 경계
        t, session = self.frames(10, np.random.default_rng(14))
        parts, _, sets = dataset.session_split(t, session, wall, marked)
        self.assertEqual(([0, 1, 2, 3, 4, 5], [6], [7, 8, 9]), sets)  # 착용 셋: 마지막 착용(7–9)만 통째로 시험, 앞 착용들은 학습·검증
        self.assertWhole(parts, session, sets)

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
