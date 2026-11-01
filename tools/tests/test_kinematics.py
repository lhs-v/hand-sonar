import unittest

import kinematics  # 사용자 설치 numpy·torch 경로도 여기서 붙는다

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 단위 손 손바닥 틀의 손바닥 점 다섯(엄지 CMC, 검지·중지·약지·소지 MCP): 검지 MCP는 x축 위, 소지 MCP는 손목에서 거리 1
PALM = np.array([[0.25, -0.30, -0.10], [1.10, 0.0, 0.0], [1.00, 0.28, 0.02], [0.88, 0.52, 0.01], [0.70, 0.71, 0.0]])
LENGTHS = np.array([0.45, 0.35, 0.30] + [0.45, 0.28, 0.24] + [0.50, 0.32, 0.25] + [0.46, 0.30, 0.24] + [0.36, 0.22, 0.21])


def random_angles(rng, n):
    return rng.uniform(-1.0, 1.0, size=(n, 15, 2))  # 라디안, ±57°


class ForwardTest(unittest.TestCase):
    def test_straight_fingers_extend_along_their_metacarpals(self):
        pos = kinematics.fk(np.zeros((1, 15, 2)), PALM, LENGTHS)[0]
        np.testing.assert_allclose(pos[list(kinematics.PALM)], PALM)  # 손바닥 점은 그대로
        for f, root in enumerate(kinematics.PALM):
            along = PALM[f] / np.linalg.norm(PALM[f])
            for k in range(3):
                np.testing.assert_allclose(pos[root + 1 + k], PALM[f] + along * LENGTHS[3 * f: 3 * f + k + 1].sum(), atol=1e-12)

    def test_flexion_bends_toward_the_palm_side_and_abduction_toward_the_little_finger(self):
        ang = np.zeros((2, 15, 2))
        ang[0, 3, 0] = np.pi / 2  # 검지 첫 뼈 90° 굽힘
        ang[1, 3, 1] = np.pi / 6  # 검지 첫 뼈 30° 벌림
        pos = kinematics.fk(ang, PALM, LENGTHS)
        np.testing.assert_allclose(pos[0, 5] - PALM[1], [0.0, 0.0, -LENGTHS[3]], atol=1e-12)  # 손바닥 쪽 = −z
        step = pos[1, 5] - PALM[1]
        np.testing.assert_allclose(step, LENGTHS[3] * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0]), atol=1e-12)  # 소지 쪽 = +y

    def test_torch_and_numpy_agree(self):
        ang = random_angles(np.random.default_rng(1), 4)
        got = kinematics.fk_t(torch.as_tensor(ang), torch.as_tensor(PALM), torch.as_tensor(LENGTHS)).numpy()
        np.testing.assert_allclose(got, kinematics.fk(ang, PALM, LENGTHS), atol=1e-12)


class InverseTest(unittest.TestCase):
    def test_ik_recovers_the_angles_fk_used(self):
        ang = random_angles(np.random.default_rng(2), 50)
        np.testing.assert_allclose(kinematics.ik(kinematics.fk(ang, PALM, LENGTHS)), ang, atol=1e-9)

    def test_a_finger_bending_in_one_plane_has_no_distal_abduction(self):
        y = kinematics.fk(np.zeros((1, 15, 2)), PALM, LENGTHS)
        root = PALM[1]  # 검지 MCP (1.1, 0, 0): x–z 평면 안에서 30° · 60° · 80° 누적으로 굽힌다(기구학과 따로 만든 점)
        p, total = root.copy(), 0.0
        for k, bend in enumerate((30.0, 30.0, 20.0)):
            total += np.radians(bend)
            p = p + LENGTHS[3 + k] * np.array([np.cos(total), 0.0, -np.sin(total)])
            y[0, 5 + k] = p
        ang = np.degrees(kinematics.ik(y)[0])
        np.testing.assert_allclose(ang[3:6, 0], [30.0, 30.0, 20.0], atol=1e-9)  # 굽힘은 부모 뼈에 대한 각
        np.testing.assert_allclose(ang[3:6, 1], 0.0, atol=1e-9)

    def test_fk_of_ik_keeps_points_when_bones_have_template_lengths(self):
        y = kinematics.fk(random_angles(np.random.default_rng(3), 20), PALM, LENGTHS)
        np.testing.assert_allclose(kinematics.fk(kinematics.ik(y), PALM, LENGTHS), y, atol=1e-9)


class TemplateTest(unittest.TestCase):
    def test_palm_is_the_training_mean_and_bones_the_training_median(self):
        rng = np.random.default_rng(4)
        y = kinematics.fk(random_angles(rng, 200) * 0.5, PALM, LENGTHS)
        y[:, list(kinematics.PALM)] += rng.normal(scale=0.01, size=(200, 5, 3))
        t = kinematics.fit_template(y, bins=0)
        np.testing.assert_allclose(t["palm"], y[:, list(kinematics.PALM)].mean(axis=0))
        bone = np.linalg.norm(y[:, 5] - y[:, 4], axis=1)
        self.assertAlmostEqual(float(np.median(bone)), float(t["lengths"][3]))
        self.assertNotIn("centers", t)

    def test_bins_cover_the_training_angles(self):
        ang = random_angles(np.random.default_rng(5), 300) * 0.6
        t = kinematics.fit_template(kinematics.fk(ang, PALM, LENGTHS), bins=16)
        self.assertEqual((30, 16), t["centers"].shape)
        flat = ang.reshape(-1, 30)
        self.assertTrue(np.all(t["centers"][:, 0] < np.percentile(flat, 1, axis=0)))
        self.assertTrue(np.all(t["centers"][:, -1] > np.percentile(flat, 99, axis=0)))


class BinTest(unittest.TestCase):
    def test_soft_argmax_of_a_peaked_logit_is_that_bin_center(self):
        centers = torch.linspace(-1.0, 1.0, 9).repeat(30, 1)
        logits = torch.full((1, 30, 9), -1e4)
        logits[..., 6] = 0.0
        np.testing.assert_allclose(kinematics.soft_argmax(logits, centers).numpy(), 0.5, atol=1e-6)

    def test_soft_labels_sum_to_one_and_peak_at_the_true_bin(self):
        centers = torch.linspace(-1.0, 1.0, 9).repeat(30, 1)
        q = kinematics.soft_labels(torch.full((2, 30), 0.25), centers)
        np.testing.assert_allclose(q.sum(-1).numpy(), 1.0, atol=1e-6)
        self.assertTrue(torch.all(q.argmax(-1) == 5))  # 0.25에 가장 가까운 칸 중심

    def test_bin_loss_is_smallest_when_logits_match_the_labels(self):
        centers = torch.linspace(-1.0, 1.0, 9).repeat(30, 1)
        truth = torch.full((1, 30), -0.5)
        good = torch.log(kinematics.soft_labels(truth, centers) + 1e-9)
        bad = torch.log(kinematics.soft_labels(-truth, centers) + 1e-9)
        self.assertLess(float(kinematics.bin_loss(good, truth, centers)), float(kinematics.bin_loss(bad, truth, centers)))


class HeadTest(unittest.TestCase):
    def backbone(self, outputs):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6, outputs))

    def test_regression_head_turns_30_angles_into_60_positions_and_passes_extras(self):
        t = kinematics.fit_template(kinematics.fk(random_angles(np.random.default_rng(6), 100) * 0.5, PALM, LENGTHS), bins=0)
        head = kinematics.Head(self.backbone(30 + 3), t, extra=3).double()
        out = head(torch.ones(4, 6, dtype=torch.float64))
        self.assertEqual((4, 63), tuple(out.shape))
        self.assertEqual(63, head.num_classes)
        self.assertIsNone(head.logits)
        bones = torch.linalg.norm(out[:, 15:18] - out[:, 12:15], dim=1)  # 검지 MCP(자리 4) → PIP(자리 5)
        np.testing.assert_allclose(bones.detach().numpy(), t["lengths"][3], rtol=1e-6)  # 템플릿은 float32 버퍼

    def test_bins_head_keeps_logits_and_reloads_from_a_state_dict(self):
        t = kinematics.fit_template(kinematics.fk(random_angles(np.random.default_rng(7), 100) * 0.5, PALM, LENGTHS), bins=8)
        head = kinematics.Head(self.backbone(30 * 8), t).double()
        x = torch.randn(3, 6, dtype=torch.float64)
        out = head(x)
        self.assertEqual((3, 30, 8), tuple(head.logits.shape))
        fresh = kinematics.Head(self.backbone(30 * 8), kinematics.empty_template(bins=8)).double()
        fresh.load_state_dict(head.state_dict())
        np.testing.assert_allclose(fresh(x).detach().numpy(), out.detach().numpy())


class AngleErrorTest(unittest.TestCase):
    def test_angle_errors_are_zero_for_the_truth_and_measure_flexion_in_degrees(self):
        ang = random_angles(np.random.default_rng(8), 10) * 0.5
        y = kinematics.fk(ang, PALM, LENGTHS)
        ang2 = ang.copy()
        ang2[:, :, 0] += np.radians(10.0)
        e = kinematics.angle_errors(kinematics.fk(ang2, PALM, LENGTHS), y)
        self.assertAlmostEqual(0.0, kinematics.angle_errors(y, y)["flexionDeg"], places=6)
        self.assertAlmostEqual(10.0, e["flexionDeg"], places=6)
        self.assertAlmostEqual(0.0, e["abductionDeg"], places=6)


if __name__ == "__main__":
    unittest.main()
