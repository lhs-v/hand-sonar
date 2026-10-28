import unittest

import joints  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402

# 합성 왼손(오른손 좌표계, m): X = 엄지 쪽, Y = 손가락 쪽, Z = 손등 쪽. 손바닥을 아래로 둔 왼손은 엄지가 +X에 온다.
MCP = {5: (0.020, 0.090, 0.0), 9: (0.0, 0.095, 0.002), 13: (-0.017, 0.088, 0.001), 17: (-0.033, 0.078, -0.002)}
THUMB = ((0.025, 0.025, -0.012), (0.045, 0.045, -0.022), (0.058, 0.065, -0.030), (0.066, 0.085, -0.035))  # 점 1–4, 손바닥 쪽(−Z)
SEGMENT_M = (0.040, 0.025, 0.022)


def left_hand(bend):
    """굽힘 bend (4,)(손가락마다 마디 하나당 rad, 손바닥 쪽으로)인 왼손 21점 (21, 3)."""
    p = np.zeros((21, 3))
    p[1:5] = THUMB
    for k, base in enumerate((5, 9, 13, 17)):
        p[base] = MCP[base]
        angle = 0.0
        for s in range(3):
            angle += bend[k]
            p[base + s + 1] = p[base + s] + SEGMENT_M[s] * np.array([0.0, np.cos(angle), -np.sin(angle)])
    return p


def rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def hands(n, rng):
    """(카메라 world의 실제 왼손 (n, 21, 3), 회전·이동 전 왼손 (n, 21, 3)). 프레임마다 굽힘·회전·이동·크기가 다르다."""
    canon = np.stack([left_hand(rng.uniform(0.0, 0.6, 4)) for _ in range(n)])
    world = np.stack([rng.uniform(0.8, 1.2) * c @ rotation(rng).T + rng.normal(0, 0.05, 3) for c in canon])
    return world, canon


class PalmFrameTest(unittest.TestCase):
    def test_frame_is_orthonormal_right_handed_and_puts_index_on_x(self):
        world, _ = hands(50, np.random.default_rng(1))
        coords, rot = joints.palm_frame(world)
        np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rot.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
        np.testing.assert_allclose(coords[:, 0], 0.0, atol=1e-15)  # 손목 원점
        np.testing.assert_allclose(coords[:, 5, 1:], 0.0, atol=1e-12)  # 검지 MCP는 +x 축 위
        self.assertTrue((coords[:, 5, 0] > 0).all())
        np.testing.assert_allclose(coords[:, 17, 2], 0.0, atol=1e-12)  # 새끼 MCP는 xy 평면, +y 쪽
        self.assertTrue((coords[:, 17, 1] > 0).all())

    def test_unmirror_gives_the_real_left_hand_and_folds_right_hands(self):
        rng = np.random.default_rng(2)
        world, canon = hands(40, rng)
        expected, _ = joints.palm_frame(canon)
        file_left = world * joints.MIRROR  # 폰 앱은 거울 그림의 world를 적는다
        got, _ = joints.palm_frame(joints.to_left(file_left, "left"))
        np.testing.assert_allclose(got / np.linalg.norm(got[:, 17], axis=1)[:, None, None],
                                   expected / np.linalg.norm(expected[:, 17], axis=1)[:, None, None], atol=1e-9)
        self.assertEqual(1.0, joints.thumb_palmar(got))  # 실제 왼손: 엄지가 손바닥 쪽
        mirrored, _ = joints.palm_frame(file_left)
        self.assertEqual(0.0, joints.thumb_palmar(mirrored))  # 되돌리지 않으면 법선 쪽(z만 뒤집힘)
        np.testing.assert_allclose(mirrored[..., 2], -joints.palm_frame(world)[0][..., 2], atol=1e-9)
        right = canon * joints.MIRROR  # 실제 오른손(왼손의 거울상)
        file_right = np.stack([r @ rotation(rng).T for r in right]) * joints.MIRROR
        folded, _ = joints.palm_frame(joints.to_left(file_right, "right"))
        np.testing.assert_allclose(folded / np.linalg.norm(folded[:, 17], axis=1)[:, None, None],
                                   expected / np.linalg.norm(expected[:, 17], axis=1)[:, None, None], atol=1e-9)

    def test_unit_hand_sets_wrist_to_pinky_mcp_to_one_and_ignores_size(self):
        world, _ = hands(30, np.random.default_rng(3))
        coords, _ = joints.palm_frame(world)
        unit, length = joints.unit_hand(coords)
        np.testing.assert_allclose(np.linalg.norm(unit[:, 17], axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(length, np.linalg.norm(world[:, 17] - world[:, 0], axis=1), atol=1e-12)
        bigger, _ = joints.unit_hand(joints.palm_frame(world * 1.7)[0])
        np.testing.assert_allclose(bigger, unit, atol=1e-12)


class InstrumentTest(unittest.TestCase):
    def setUp(self):
        _, canon = hands(200, np.random.default_rng(4))
        self.true = joints.unit_hand(joints.palm_frame(canon)[0])[0][:, 1:]  # (200, 20, 3) 단위 손 목표

    def test_mpjpe_and_moving_points(self):
        shift = np.zeros((20, 3))
        shift[:, 0] = 0.1  # 모든 점을 x로 0.1
        self.assertTrue(np.allclose(joints.frame_errors(self.true + shift, self.true), 0.1))
        palm_only = np.zeros((20, 3))
        palm_only[[p - 1 for p in joints.PALM], 2] = 0.3  # 손바닥 점만 0.3
        self.assertTrue(np.allclose(joints.frame_errors(self.true + palm_only, self.true), 0.3 * 4 / 20))
        self.assertTrue(np.allclose(joints.frame_errors(self.true + palm_only, self.true, joints.MOVING), 0.0))

    def test_collapse_ratio_and_bone_shortfall(self):
        true = np.random.default_rng(5).normal(size=(200, 20, 3))  # 모든 점·축이 흔들리는 자료(합성 손의 엄지는 고정이라 분모가 0)
        mean = true.mean(axis=0)
        half = mean + 0.5 * (true - mean)  # 평균 쪽으로 반만 움직이는 예측
        np.testing.assert_allclose(joints.collapse(half, true), 0.5, atol=1e-12)
        self.assertEqual((16, 3), joints.collapse(half, true).shape)
        short = self.true.copy()
        for tip in joints.TIPS:  # 끝마디만 20% 짧게
            short[:, tip - 1] = self.true[:, tip - 2] + 0.8 * (self.true[:, tip - 1] - self.true[:, tip - 2])
        np.testing.assert_allclose(joints.by_segment(joints.bone_shortfall(short, self.true)), [0, 0, 0, 20], atol=1e-9)

    def test_mean_pose_and_medoid_baselines(self):
        base = self.true[0]
        step = self.true[1] - self.true[0]
        train = np.stack([base + c * step for c in (-2.0, -1.0, 0.1, 1.0, 2.0)])
        np.testing.assert_allclose(joints.mean_pose(train), base + 0.02 * step, atol=1e-12)
        np.testing.assert_allclose(joints.medoid(train), train[2], atol=0)  # 평균(c = 0.02)에 가장 가까운 실제 프레임
        test = (base + step)[None]
        self.assertAlmostEqual(0.98 * float(np.linalg.norm(step, axis=1).mean()),
                               float(joints.frame_errors(joints.mean_pose(train)[None], test)[0]), places=12)


if __name__ == "__main__":
    unittest.main()
