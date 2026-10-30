import os
import unittest

import render_video  # 사용자 설치 numpy 경로도 여기서 붙는다

import dataset  # noqa: E402
import hand_seg  # noqa: E402
import hs_io  # noqa: E402
import joints  # noqa: E402
import numpy as np  # noqa: E402
import render_draw as D  # noqa: E402


def hand_world():
    """파일 좌표(m)의 손 하나 (21, 3): 손가락 다섯이 손바닥에서 위(+y)로 뻗고 끝으로 갈수록 조금 앞(+z)으로 굽는다."""
    pts = np.zeros((21, 3))
    for start, (bx, by) in zip((1, 5, 9, 13, 17), ((0.03, 0.02), (0.025, 0.08), (0.0, 0.085), (-0.02, 0.08), (-0.04, 0.07))):
        for s in range(4):
            pts[start + s] = (bx * (1 + 0.3 * s), by + 0.025 * s, 0.004 * s * s)
    return pts


def state(world, hand="left", size=(360, 480)):
    """render_draw가 쓰는 칸만 채운 S(프레임 len(world)개). 화면 좌표는 world xy를 그대로 늘린 것."""
    real = joints.to_left(world, hand)
    coords, rot = joints.palm_frame(real)
    unit, Lm = joints.unit_hand(coords)
    img_px = world[:, :, :2] * np.array([1500.0, -1500.0]) + np.array([180.0, 400.0])
    c, side = D.crop_params(img_px, size)
    return {"hand": hand, "n": len(world), "world": world, "real": real, "rot": rot, "unit": unit, "Lm": Lm, "L_mm": 87.0, "img_px": img_px,
            "crop_c": c, "crop_side": side, "ymax": 40.0}


class GeometryTest(unittest.TestCase):
    def test_similarity_fit_recovers_rotation_scale_and_shift(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(21, 2))
        A = 2.5 * np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
        dst = src @ A.T + np.array([3.0, -1.0])
        got, sm, dm = D.fit_similarity(src, dst)
        np.testing.assert_allclose(got, A, atol=1e-12)
        np.testing.assert_allclose((src - sm) @ got.T + dm, dst, atol=1e-12)

    def test_prediction_is_undone_back_to_the_file_world_for_both_hands(self):
        world = hand_world()[None]
        for hand in ("left", "right"):
            S = state(world, hand)
            y = joints.targets(world, hand)[0]
            np.testing.assert_allclose(D.pred_world(S, 0, y[0]), world[0, 1:], atol=1e-12)

    def test_image_overlay_moves_only_the_joint_the_prediction_moves(self):
        world = hand_world()[None]
        S = state(world)
        y = joints.targets(world, "left")[0][0]
        np.testing.assert_allclose(D.pred_to_image(S, 0, y), S["img_px"][0], atol=1e-9)  # 예측 = 정답이면 정답 영상점 그대로
        bent = y.copy()
        bent[7] += (0.0, 0.3, 0.0)  # 검지 끝(점 8)만 새끼 쪽으로(손바닥 면 안 — 이 손은 면이 화면과 나란해 깊이 방향은 영상에 안 보인다)
        moved = np.linalg.norm(D.pred_to_image(S, 0, bent) - S["img_px"][0], axis=1)
        self.assertGreater(moved[8], 1.0)
        self.assertLess(np.delete(moved, 8).max(), 1e-9)

    def test_crop_fills_frames_without_a_hand(self):
        px = np.full((5, 21, 2), np.nan)
        px[1] = np.random.default_rng(1).uniform(100, 200, (21, 2))
        px[3] = px[1] + 40.0
        c, side = D.crop_params(px, (360, 480), half=0)
        self.assertTrue(np.isfinite(c).all() and np.isfinite(side).all())
        np.testing.assert_allclose(c[2], (c[1] + c[3]) / 2)
        self.assertTrue((side >= 110.0).all())
        c, side = D.crop_params(np.full((2, 21, 2), np.nan), (360, 480))
        np.testing.assert_allclose(c, [[180.0, 240.0]] * 2)  # 손이 하나도 없으면 영상 가운데


@unittest.skipUnless(D.find_fonts(), "한글 글꼴이 없음(render_draw.FONTS)")
class RenderTest(unittest.TestCase):
    def setUp(self):
        world = np.repeat(hand_world()[None], 3, axis=0)
        self.S = state(world)
        self.y = joints.targets(world, "left")[0]
        self.bg = D.static_background("제목", "부제", ("설명 하나", "설명 둘"), 40.0, mean_mm=np.vstack([np.zeros((1, 3)), self.y[0] * 87.0]))
        self.info = {"pred": self.y[0] + 0.02, "err": 12.3, "prompt": "숫자 2", "kind": "ASL 숫자", "edge": True, "time": 1.0, "tag": "시험 프레임",
                     "tag_color": D.PURPLE, "tag_note": "", "seg_text": "시험 블록 0", "meanpose": 14.0}
        self.tl = {"t": np.array([0.0, 0.1, 0.2]), "err": np.array([10.0, np.nan, 12.0]), "mean": np.array([14.0, 15.0, 13.0]),
                   "role": np.array([2, 2, -1], np.int8), "a": 0.0, "b": 0.2, "now": 0.1}
        self.frame = np.zeros((480, 360, 3), np.uint8)
        self.frame[:] = (0, 255, 0)  # 초록 영상: 손 패널에 원본이 보이는지 가린다

    def panel(self, img):
        return img[D.PA[1] + 2: D.PA[1] + D.PA[3] - 2, D.PA[0] + 2: D.PA[0] + D.PA[2] - 2]

    def test_every_kind_of_frame_renders_to_the_canvas_size(self):
        for frame, seg in ((self.frame, None), (None, None), (self.frame, {"alpha": np.full((320, 320), 255, np.uint8)})):
            img = D.render_frame(self.S, self.bg, 0, frame, self.info, seg, self.tl)
            self.assertEqual((img.shape, img.dtype), ((D.CH, D.CW, 3), np.uint8))
        card = D.card_image("제목", "부제", "시험 블록 0", "세션 0–10 s", ["줄 하나", "12.3 mm", "참고"], "1 / 4")
        self.assertEqual(card.shape, (D.CH, D.CW, 3))

    def test_hand_only_view_never_shows_the_frame_outside_the_alpha(self):
        def green(img):  # 영상의 초록(RGB)이 남은 픽셀 수 — 흰 정답 선·파란 예측은 세지 않는다
            p = self.panel(img).astype(int)
            return int(((p[..., 1] > 180) & (p[..., 0] < 80) & (p[..., 2] < 80)).sum())

        self.assertGreater(green(D.render_frame(self.S, self.bg, 0, self.frame, self.info, None, self.tl)), 1000)
        self.assertEqual(green(D.render_frame(self.S, self.bg, 0, self.frame, self.info, {"alpha": np.zeros((320, 320), np.uint8)}, self.tl)), 0)
        self.assertEqual(green(D.render_frame(self.S, self.bg, 0, self.frame, self.info, {"alpha": None}, self.tl)), 0)


class HandSegTest(unittest.TestCase):
    def test_no_landmarks_means_no_hand(self):
        alpha, status, _ = hand_seg.segment(np.zeros((320, 320, 3), np.uint8), np.full((21, 2), np.nan), 100.0)
        self.assertEqual((status, int(alpha.max())), (2, 0))

    def test_a_skin_hand_on_a_blue_background_is_kept_and_the_background_dropped(self):
        world = hand_world()
        pts = world[:, :2] * np.array([1300.0, -1300.0]) + np.array([170.0, 270.0])
        size = hand_seg.hand_size(pts)
        img = np.zeros((320, 320, 3), np.uint8)
        img[:] = (190, 110, 40)  # 파란 바탕(BGR)
        prior = hand_seg.build_masks(pts, size)[0]
        img[prior] = (120, 160, 225)  # 피부색 손
        img = np.clip(img + np.random.default_rng(2).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        alpha, status, _ = hand_seg.segment(img, pts, size)
        self.assertEqual(status, 0)
        self.assertGreater(alpha[prior].mean(), 0.8 * 255)
        self.assertEqual(int(alpha[:20, :20].max()), 0)


class PlanTest(unittest.TestCase):
    def fake(self):
        t = np.arange(60) * 0.5 - 1.0  # −1 … 28.5 s
        role = np.full(60, -2, np.int8)
        role[2:] = 0
        role[(t >= 10) & (t < 20)] = 2
        role[(t >= 10) & (t < 12)] = -1  # 블록 앞 2 s 비움
        return {"name": "s", "n": 60, "t": t, "role": role, "block": np.where(t >= 0, np.floor(t / 10), -1).astype(int), "test_blocks": [1],
                "err": np.where(role == 2, 10.0, np.nan), "mean_err": np.where(role == 2, 12.0, np.nan)}

    def test_blocks_render_whole_test_blocks_and_all_renders_the_session(self):
        S = self.fake()
        segs = render_video.plan(S, "blocks", "test")
        self.assertEqual([s["title"] for s in segs], ["시험 블록 1"])
        np.testing.assert_array_equal(segs[0]["frames"], np.flatnonzero((S["t"] >= 10) & (S["t"] < 20)))  # 비운 2 s도 블록에 든다(정답만)
        self.assertEqual(segs[0]["range"], (10.0, 20.0))
        self.assertEqual(len(render_video.plan(S, "blocks", "all")[0]["frames"]), 60)
        session = render_video.plan(S, "session", "test")[0]["frames"]
        self.assertEqual((session[0], session[-1]), (26, 41))  # 첫 시험 프레임(12 s)부터 마지막(19.5 s)까지
        S["role"][S["role"] == 2] = 0
        with self.assertRaises(ValueError):
            render_video.plan(S, "blocks", "test")

    def test_time_strip_follows_the_frame_in_long_segments(self):
        S = self.fake()
        seg = {"frames": np.arange(60)}
        old, render_video.TL_WINDOW_SEC = render_video.TL_WINDOW_SEC, 10.0
        try:
            tl = render_video.timeline(S, seg, 30)  # 14 s
            self.assertEqual((tl["a"], tl["b"]), (9.0, 19.0))
            tl = render_video.timeline(S, seg, 1)  # 앞 끝에 붙는다
            self.assertEqual((tl["a"], tl["b"]), (-1.0, 9.0))
        finally:
            render_video.TL_WINDOW_SEC = old

    def test_output_inside_the_repository_is_refused(self):
        self.assertTrue(render_video.inside(os.path.join(hs_io.REPO, "tools", "out"), hs_io.REPO))
        self.assertTrue(render_video.inside(hs_io.REPO, hs_io.REPO))
        self.assertFalse(render_video.inside(os.path.join(os.path.dirname(hs_io.REPO), "hand-sonar-data"), hs_io.REPO))


class SplitTest(unittest.TestCase):
    def saved(self):
        n = 400
        session = np.repeat(np.arange(2), n // 2).astype(np.int16)
        frame_ns = 10 ** 12 + np.tile(np.arange(n // 2, dtype=np.int64) * 200_000_000, 2)  # 세션마다 40 s
        return {"y": np.zeros((n, 60), np.float32), "frame_ns": frame_ns, "session": session, "sessions": np.array(["a", "b"]),
                "class_id": np.full(n, -99, np.int16), "labelled": np.zeros(n, bool), "wall_ms": np.array([0.0, 1e6]), "rewear": np.array([False, True]),
                "wear": np.array([0, 1])}

    def test_the_run_split_is_rebuilt_and_checked_against_metrics(self):
        saved = self.saved()
        tr, va, te = dataset.block_split(saved["frame_ns"], saved["session"])
        m = {"mode": "blocks", "split": {"offset": 0}, "frames": {"train": len(tr), "val": len(va), "test": len(te)}}
        parts = render_video.run_split(saved, m)
        np.testing.assert_array_equal(parts[2], te)
        m = {"mode": "session", "split": {"train": ["a"], "val": [], "test": ["b"]}, "frames": {"train": 170, "val": 20, "test": 200}}
        np.testing.assert_array_equal(render_video.run_split(saved, m)[2], np.arange(200, 400))
        m["frames"]["test"] = 199
        with self.assertRaises(ValueError):
            render_video.run_split(saved, m)


if __name__ == "__main__":
    unittest.main()
