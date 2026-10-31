import math
import unittest

import wrist  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402


def rot(axis, angle):
    return wrist.rot_axis(np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis), angle)


def angle_between(a, b):
    """두 회전 사이 각(도)."""
    c = (np.trace(a.T @ b) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class MahonyTest(unittest.TestCase):
    def test_tracks_turns_and_keeps_the_tilt_from_gravity(self):
        dt, n = 0.01, 800
        t = (np.arange(n) * dt * 1e9).astype(np.int64)
        true = np.zeros((n, 3, 3))
        gyro = np.zeros((n, 3))
        r = rot([1, 0, 0], 0.6) @ rot([0, 1, 0], -0.3)  # 첫 자세: 기울어 있고 방위도 임의
        for k in range(n):
            if k:
                gyro[k] = (0.0, 0.0, 0.5) if k < 400 else (0.4, -0.2, 0.0)  # 워치 축 회전 속도(rad/s): 앞은 화면 둘레, 뒤는 기울이기
                r = r @ rot(gyro[k], np.linalg.norm(gyro[k]) * dt)
            true[k] = r
        up = np.einsum("kba,b->ka", true, [0.0, 0.0, 1.0])  # 중력 센서 = 워치 축으로 본 위쪽
        est, bias = wrist.mahony(t, gyro, up)
        np.testing.assert_allclose(np.einsum("kba,b->ka", est, [0.0, 0.0, 1.0]), up, atol=2e-3)  # 기울기는 중력으로 맞는다
        # 방위는 처음 자세 기준으로만 뜻이 있다: 처음부터의 상대 회전이 같다
        self.assertLess(angle_between(est[0].T @ est[-1], true[0].T @ true[-1]), 1.0)
        self.assertLess(float(np.abs(bias).max()), 0.01)  # 치우침 없는 자이로라 추정도 0 근처


class YawTest(unittest.TestCase):
    def test_fit_yaw_recovers_the_turn_about_vertical(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(50, 3))
        b = a @ rot([0, 0, 1], 2.4).T + 0.01 * rng.normal(size=(50, 3))
        self.assertAlmostEqual(2.4, wrist.fit_yaw(a, b), places=2)
        self.assertTrue(math.isnan(wrist.fit_yaw(np.array([[0, 0, 1.0]]), np.array([[0, 0, 1.0]]))))  # 수직 축만으로는 방위를 모른다

    def test_up_camera_follows_display_rotation(self):
        np.testing.assert_allclose(wrist.up_camera([0.0, 9.5, 2.6]), np.array([0.0, -9.5, 2.6]) / math.hypot(9.5, 2.6), atol=1e-9)
        np.testing.assert_allclose(wrist.up_camera([9.8, 0.0, 0.0], 1), [0.0, -1.0, 0.0], atol=1e-9)  # 기기를 시계 반대로 90°: 기기 +x가 위
        v = np.array([0.3, -0.4, 0.866])
        np.testing.assert_allclose(wrist.rotation_to_z(v) @ (v / np.linalg.norm(v)), [0, 0, 1.0], atol=1e-9)

    def test_hand_axes_are_a_proper_frame_along_the_middle_finger(self):
        rng = np.random.default_rng(2)
        p = rng.normal(size=(5, 21, 3))
        axes = wrist.hand_axes(p)
        for m in range(5):
            np.testing.assert_allclose(axes[m] @ axes[m].T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(1.0, float(np.linalg.det(axes[m])), places=9)
            v = p[m, 9] - p[m, 0]
            np.testing.assert_allclose(axes[m, 0], v / np.linalg.norm(v), atol=1e-12)

    def test_zero_rotation_sends_the_neutral_direction_to_straight(self):
        mean = np.array([math.cos(0.3), 0.05, math.sin(0.3)])  # 정면 기준 자세에서 읽힌 젖힘 약 17°
        q = wrist.zero_rotation(mean)
        np.testing.assert_allclose(q @ (mean / np.linalg.norm(mean)), [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(q @ q.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(1.0, float(np.linalg.det(q)), places=12)
        e, _ = wrist.extension_deviation((q @ np.array([math.cos(0.5), 0.0, math.sin(0.5)]))[None])
        self.assertAlmostEqual(math.degrees(0.5 - 0.3), float(e[0]), delta=0.5)  # 같은 축으로 더 젖힌 것은 차이만 남는다
        np.testing.assert_allclose(wrist.zero_rotation([2.0, 0.0, 0.0]), np.eye(3), atol=1e-12)

    def test_extension_and_deviation_angles(self):
        e, d = wrist.extension_deviation(np.array([[math.cos(0.5), 0.0, math.sin(0.5)], [math.cos(0.3), math.sin(0.3), 0.0]]))
        np.testing.assert_allclose(e, [math.degrees(0.5), 0.0], atol=1e-9)
        np.testing.assert_allclose(d, [0.0, math.degrees(0.3)], atol=1e-9)


class HamerPointsTest(unittest.TestCase):
    def test_hamer_file_is_used_only_when_frames_match(self):
        import os
        import tempfile
        frames = np.array([10, 20, 30], dtype=np.int64)
        kp = np.random.default_rng(5).normal(size=(3, 21, 3)).astype(np.float32)
        kp[1] = np.nan  # HaMeR가 손을 못 본 프레임
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hamer.npz")
            self.assertIsNone(wrist.hamer_points(path, frames))  # 파일 없음 → MediaPipe
            np.savez(path, frame_ns=frames, keypoints=kp, global_orient=np.zeros((3, 3, 3), np.float32))
            points, has = wrist.hamer_points(path, frames)
            np.testing.assert_array_equal([True, False, True], has)
            np.testing.assert_allclose(points[0], kp[0], atol=1e-6)
            with self.assertRaises(ValueError):  # landmarks.csv가 바뀌었으면 쓰지 않는다
                wrist.hamer_points(path, frames + 1)


class CompositionTest(unittest.TestCase):
    """카메라 기울기(폰 가속도) + 방위 맞춤 + 워치 자세로 손 방향을 워치 축에 되돌린다(session_wrist의 셈을 파일 없이)."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.accel = np.array([-0.2, 9.5, 2.6])  # 폰: 세워 두고 뒤로 약 15° 기욺
        self.c0 = wrist.rotation_to_z(wrist.up_camera(self.accel))
        self.cam = rot([0, 0, 1], -1.1) @ self.c0  # 참 카메라 → 세계(방위 −1.1 rad)
        n = 300
        self.watch = np.stack([rot([0, 1, 0], -0.8 + 0.2 * math.sin(k / 30)) @ rot([1, 0, 0], 0.3 * math.sin(k / 17)) @ rot([0, 0, 1], 0.4 * math.cos(k / 23))
                               for k in range(n)])  # 워치 → 세계: 팔뚝이 기울고 비틀리며 움직임
        self.ext = 0.4 + 0.3 * np.sin(np.arange(n) / 11)  # 손목 젖힘(워치 y축 둘레, rad) — 새끼 쪽 축은 돌지 않는다
        self.hand_w = np.stack([rot([0, 1, 0], e) for e in self.ext])  # 손 축(행)을 워치 축으로: Ry(e)의 행 0 = (cos e, 0, sin e) = 손등 쪽 젖힘
        hand_world = np.einsum("mab,mkb->mka", self.watch, self.hand_w)
        self.hand_k = np.einsum("ba,mkb->mka", self.cam, hand_world)  # 카메라가 본 손 축(행) = C^T R q
        self.rng = rng

    def test_ulnar_fit_recovers_heading_and_extension(self):
        psi = wrist.fit_yaw(*wrist.yaw_pairs(self.c0, self.hand_k, self.watch, "ulnar"))
        cam = wrist.rot_axis([0, 0, 1.0], psi) @ self.c0
        np.testing.assert_allclose(cam, self.cam, atol=1e-6)
        u = np.einsum("mba,bc,mc->ma", self.watch, cam, self.hand_k[:, 0])
        ext, dev = wrist.extension_deviation(u)
        np.testing.assert_allclose(ext, np.degrees(self.ext), atol=1e-4)
        np.testing.assert_allclose(dev, 0.0, atol=1e-4)

    def test_calibration_fit_uses_all_three_axes(self):
        straight = np.einsum("ba,mbk->mka", self.cam, self.watch)  # 평행 자세: 손 축 k = 워치 축 k(R의 열)를 카메라 축으로
        psi = wrist.fit_yaw(*wrist.yaw_pairs(self.c0, straight[:50], self.watch[:50], "calibration"))
        np.testing.assert_allclose(wrist.rot_axis([0, 0, 1.0], psi) @ self.c0, self.cam, atol=1e-6)

    def test_classes_of_frames_mark_the_settled_part_of_each_prompt(self):
        sec = 1_000_000_000
        prompts = [{"startNs": 0, "endNs": 2 * sec, "classId": -3, "flag": ""}, {"startNs": 2 * sec, "endNs": 4 * sec, "classId": 5, "flag": "wrong"}]
        cls, flag, settled = wrist.class_of_frames(prompts, np.array([0, sec // 4, sec, 3 * sec, 5 * sec]))
        np.testing.assert_array_equal([-3, -3, -3, 5, -99], cls)
        self.assertEqual(["", "", "", "wrong", ""], list(flag))
        np.testing.assert_array_equal([False, False, True, True, False], settled)


if __name__ == "__main__":
    unittest.main()
