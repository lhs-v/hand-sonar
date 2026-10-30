import os
import pickle
import sys
import tempfile
import types
import unittest

import mano  # 사용자 설치 numpy·scipy·torch 경로도 여기서 붙는다

import numpy as np  # noqa: E402
import scipy.sparse  # noqa: E402
import torch  # noqa: E402

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]  # MANO 순서: 검지 1–3 · 중지 4–6 · 새끼 7–9 · 약지 10–12 · 엄지 13–15


def fake_model(v=800):
    """MANO와 같은 배열 모양의 작은 손 모형(m): 관절 16개는 손바닥에서 위(+y)로 뻗은 다섯 사슬, 정점 0–15는 관절 자리, 손끝 정점(mano.TIPS)은
    끝마디 너머, 나머지는 가장 가까운 관절에 붙는다. β0 = 1이면 10% 커지고, 자세 보정은 없다."""
    rng = np.random.default_rng(0)
    J = np.zeros((16, 3))
    for first, (x, y) in {1: (0.025, 0.09), 4: (0.0, 0.095), 7: (-0.04, 0.08), 10: (-0.02, 0.09), 13: (0.035, 0.03)}.items():
        for s in range(3):
            J[first + s] = (x * (1 + 0.2 * s), y + 0.025 * s, 0.003 * s)
    verts = J[rng.integers(0, 16, v)] + rng.normal(0, 0.005, (v, 3))
    verts[:16] = J
    for tip, joint in zip(mano.TIPS, mano.TIP_JOINTS):
        verts[tip] = J[joint] + 0.8 * (J[joint] - J[PARENTS[joint]])
    weights = np.zeros((v, 16))
    weights[np.arange(v), np.linalg.norm(verts[:, None] - J[None], axis=2).argmin(1)] = 1.0
    for tip, joint in zip(mano.TIPS, mano.TIP_JOINTS):
        weights[tip] = np.eye(16)[joint]
    shapedirs = np.zeros((v, 3, 10))
    shapedirs[:, :, 0] = 0.1 * verts
    return {"v_template": verts, "shapedirs": shapedirs, "posedirs": np.zeros((v, 3, 135)), "J_regressor": np.eye(16, v), "weights": weights,
            "hands_mean": np.zeros(45), "faces": np.array([[0, 1, 2], [0, 4, 5]]), "parents": np.array(PARENTS)}


def raw_pickle(m):
    """m을 MANO pkl처럼 바이트로: shapedirs는 실제 파일처럼 chumpy 고르기(Select — (V, 3, 20) 배열에서 앞 10성분), J_regressor는 희소 행렬,
    kintree_table 첫 칸은 4294967295."""
    mod, sel_mod, top = types.ModuleType("chumpy.ch"), types.ModuleType("chumpy.reordering"), types.ModuleType("chumpy")
    ch = type("Ch", (), {"__module__": "chumpy.ch"})
    select = type("Select", (), {"__module__": "chumpy.reordering"})
    mod.Ch, sel_mod.Select = ch, select
    sys.modules["chumpy"], sys.modules["chumpy.ch"], sys.modules["chumpy.reordering"] = top, mod, sel_mod
    try:
        v = len(m["v_template"])
        a = ch()
        a.x = np.concatenate([m["shapedirs"], np.full((v, 3, 10), 7.0)], axis=2)
        a.extra = {"chumpy가 담는 다른 값"}
        shapedirs = select()
        shapedirs.a, shapedirs.idxs, shapedirs.preferred_shape = a, np.arange(v * 3 * 20).reshape(v, 3, 20)[..., :10].ravel(), (v, 3, 10)
        kintree = np.stack([np.r_[4294967295, m["parents"][1:]], np.arange(16)])
        raw = {"v_template": m["v_template"], "shapedirs": shapedirs, "posedirs": m["posedirs"], "J_regressor": scipy.sparse.csc_matrix(m["J_regressor"]),
               "weights": m["weights"], "hands_mean": m["hands_mean"], "hands_components": np.eye(45), "f": m["faces"].astype(np.uint32),
               "kintree_table": kintree}
        return pickle.dumps(raw, protocol=2)
    finally:
        del sys.modules["chumpy"], sys.modules["chumpy.ch"], sys.modules["chumpy.reordering"]


class ManoTest(unittest.TestCase):
    def setUp(self):
        self.m = fake_model()
        self.layer = mano.Layer(self.m)
        self.zero = torch.zeros(10)

    def joints(self, orient, pose, beta=None):
        with torch.no_grad():
            return self.layer.joints21(self.zero if beta is None else beta, torch.as_tensor(orient, dtype=torch.float32)[None],
                                       torch.as_tensor(pose, dtype=torch.float32)[None])[0].numpy()

    def test_rest_pose_gives_the_template_in_mediapipe_order(self):
        want = np.concatenate([self.m["v_template"][:16], self.m["v_template"][list(mano.TIPS)]])[list(mano.TO_MEDIAPIPE)]
        np.testing.assert_allclose(self.joints(np.zeros(3), np.zeros(45)), want, atol=1e-6)
        np.testing.assert_allclose(self.joints(np.zeros(3), np.zeros(45), torch.tensor([1.0] + [0.0] * 9)), 1.1 * want, atol=1e-6)

    def test_bending_the_index_knuckle_turns_only_the_index_finger(self):
        rest = self.joints(np.zeros(3), np.zeros(45))
        pose = np.zeros(45)
        pose[2] = np.pi / 2  # MANO 관절 1(검지 MCP)을 z축으로 90°
        bent = self.joints(np.zeros(3), pose)
        rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(bent[6:9] - bent[5], (rest[6:9] - rest[5]) @ rz.T, atol=1e-6)  # MediaPipe 6–8 = 검지 PIP·DIP·끝
        others = [k for k in range(21) if k not in (6, 7, 8)]
        np.testing.assert_allclose(bent[others], rest[others], atol=1e-6)

    def test_pose_fit_recovers_the_joints(self):
        rng = np.random.default_rng(1)
        orient = torch.as_tensor(rng.normal(0, 0.6, (8, 3)), dtype=torch.float32)
        pose = torch.as_tensor(rng.normal(0, 0.3, (8, 45)), dtype=torch.float32)
        with torch.no_grad():
            j = self.layer.joints21(self.zero, orient, pose)
            target = ((j - j[:, :1]) * 1000.0).numpy()
        _, _, res = mano.fit_pose(self.layer, target, self.zero, torch.tensor(0.0))
        self.assertLess(res.max(), 1.0)

    def test_shape_fit_absorbs_a_bigger_hand(self):
        rng = np.random.default_rng(2)
        with torch.no_grad():
            j = self.layer.joints21(self.zero, torch.as_tensor(rng.normal(0, 0.6, (40, 3)), dtype=torch.float32),
                                    torch.as_tensor(rng.normal(0, 0.2, (40, 45)), dtype=torch.float32))
            target = ((j - j[:, :1]) * 1150.0).numpy()  # 15% 큰 손
        beta, log_s, res = mano.fit_shape(self.layer, target, frames=40)
        self.assertLess(res.mean(), 1.0)
        self.assertAlmostEqual(float(torch.exp(log_s)) * (1 + 0.1 * float(beta[0])), 1.15, delta=0.02)
        _, _, res = mano.fit_pose(self.layer, target, beta, log_s)
        verts = mano.meshes(self.layer, beta, log_s, *mano.fit_pose(self.layer, target[:3], beta, log_s)[:2])
        self.assertEqual(verts.shape, (3, 800, 3))
        self.assertLess(res.mean(), 1.0)

    def test_the_pickle_is_read_without_chumpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "mano_v1_2", "models")
            os.makedirs(folder)
            with open(os.path.join(folder, "MANO_LEFT.pkl"), "wb") as f:
                f.write(raw_pickle(self.m))
            m = mano.load(tmp)  # 푼 폴더를 주면 그 아래에서 찾는다
        for k in ("v_template", "shapedirs", "J_regressor", "weights", "parents", "faces"):
            np.testing.assert_allclose(m[k], self.m[k])

    def test_a_pickle_that_calls_a_function_is_refused(self):
        class Evil:
            def __reduce__(self):
                return (os.getcwd, ())

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "MANO_LEFT.pkl")
            with open(path, "wb") as f:
                pickle.dump({"v_template": Evil()}, f, protocol=2)
            with self.assertRaises(pickle.UnpicklingError):
                mano.load(path)

    def test_tips_on_the_wrong_joint_are_refused(self):
        bad = dict(self.m, weights=self.m["weights"].copy())
        bad["weights"][mano.TIPS[0]] = np.eye(16)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "MANO_LEFT.pkl")
            with open(path, "wb") as f:
                f.write(raw_pickle(bad))
            with self.assertRaises(ValueError):
                mano.load(path)


if __name__ == "__main__":
    unittest.main()
