import unittest

import train  # 사용자 설치 numpy·torch·timm 경로도 여기서 붙는다

import joints  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CLASSES = (0, 10, 1, 10, 2, -1, 15, 10, 3, -1)  # 2 s 동작 줄 10개의 classId(숫자 5 = 10이 셋, 자유 동작 둘)


def synthetic(rng, frames=600):
    """세션 하나(30 fps, 20 s)의 학습 자료 dict와 (학습, 검증, 시험) 프레임 번호. 동작 줄마다 다른 자세 + 작은 흔들림, 줄 앞 0.5 s는 경계.
    학습 0–399(줄 0–6) · 검증 400–499 · 시험 500–599(약지 굽히기 유지 구간과 자유 동작)."""
    prompt = np.arange(frames) // 60
    poses = rng.normal(size=(len(CLASSES), 20, 3)) * 0.2
    y = poses[prompt] + 0.01 * rng.normal(size=(frames, 20, 3))
    ds = {"y": y.reshape(frames, 60).astype(np.float32), "frame_ns": 10 ** 12 + np.arange(frames, dtype=np.int64) * 33_333_333,
          "session": np.zeros(frames, dtype=np.int16), "prompt": prompt.astype(np.int32), "class_id": np.array(CLASSES)[prompt].astype(np.int16),
          "labelled": np.arange(frames) % 60 >= 15, "length_mm": np.array([87.0]), "sessions": np.array(["20990101-000000"]),
          "label_ids": np.array([-1, 0, 1, 2, 3, 10, 15], dtype=np.int16),
          "label_names": np.array(["자유 동작", "엄지 굽히기", "검지 굽히기", "중지 굽히기", "약지 굽히기", "숫자 5", "손목 젖히기"])}
    return ds, np.arange(400), np.arange(400, 500), np.arange(500, frames)


class VelocityLossTest(unittest.TestCase):
    def test_velocity_loss_compares_motion_not_position_and_skips_chunk_edges(self):
        true = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [10.0, 10.0]])
        pair = torch.tensor([True, True, False])  # 셋째 쌍(2→3)은 묶음 경계
        self.assertAlmostEqual(0.0, float(train.velocity_loss(true + 5.0, true, pair)))  # 위치가 틀려도 움직임이 같으면 0
        # 움직이지 않는 예측: 정답 차분 (1, 2)·(2, 0)의 제곱 평균 (2.5 + 2.0) / 2, 경계 쌍 (7, 8)은 넣지 않음
        self.assertAlmostEqual(2.25, float(train.velocity_loss(torch.zeros_like(true), true, pair)))

    def test_batch_pairs_never_cross_chunks_or_breaks(self):
        t = np.r_[np.arange(100), np.arange(150, 220)] * 50_000_000  # 50 ms 간격, 100번 뒤 2.5 s 끊김
        session = np.zeros(len(t), dtype=np.int16)
        segs = train.segments(t, session, np.random.default_rng(9).permutation(len(t)))
        self.assertEqual([100, 70], [len(s) for s in segs])
        epoch = train.batches(segs, np.random.default_rng(10), chunk=32, batch=64)
        np.testing.assert_array_equal(np.sort(np.concatenate([idx for idx, _ in epoch])), np.arange(len(t)))  # epoch마다 모든 프레임 한 번
        for idx, pair in epoch:
            self.assertEqual(len(idx) - 1, len(pair))
            np.testing.assert_array_equal(np.diff(t[idx])[pair], 50_000_000)  # 이웃 쌍은 늘 바로 다음 프레임


class WristTest(unittest.TestCase):
    def test_depth_error_counts_a_quarter_and_frames_without_truth_are_ignored(self):
        u = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        pred = torch.tensor([[1.0, 0.0, 0.2], [1.0, 0.2, 0.0]])  # 첫째는 깊이 쪽 0.2, 둘째는 그림 평면 쪽 0.2 틀림
        self.assertAlmostEqual((train.DEPTH_WEIGHT * 0.04 + 0.04) / 2, float(train.wrist_loss(pred, u, d, torch.tensor([1.0, 1.0]))), places=6)
        self.assertAlmostEqual(train.DEPTH_WEIGHT * 0.04, float(train.wrist_loss(pred, u, d, torch.tensor([1.0, 0.0]))), places=6)
        self.assertEqual(0.0, float(train.wrist_loss(pred, u, d, torch.tensor([0.0, 0.0]))))  # 정답이 없으면 0

    def test_loss_fn_adds_the_wrist_term_only_when_weighted(self):
        pred = torch.zeros(4, 63)
        true = torch.zeros(4, 67)
        true[:, 60] = 1.0  # u = (1, 0, 0)
        true[:, 65] = 1.0  # 깊이 축 = (0, 0, 1)
        true[:, 66] = 1.0  # 정답 있음
        pair = torch.tensor([True, True, True])
        self.assertEqual(0.0, float(train.loss_fn(pred, true, pair)))  # 손가락 60값만 본다
        self.assertAlmostEqual(0.5, float(train.loss_fn(pred, true, pair, wrist_weight=0.5)), places=6)  # 그림 평면 오차 1 × 무게 0.5

    def test_model_takes_more_input_channels(self):
        model = train.make_model(train.FINGER_OUT, in_chans=4).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 4, 96, 60))
        self.assertEqual((2, train.FINGER_OUT), tuple(out.shape))

    def test_model_outputs_follow_the_head_size(self):
        model = train.make_model(train.FINGER_OUT + train.WRIST_OUT)
        x = torch.zeros(2, 2, 96, 60)
        self.assertEqual((2, 63), tuple(train.predict(model, x, np.arange(2)).shape))
        self.assertEqual((0, 63), train.predict(model, x, np.arange(0)).shape)

    def test_wrist_metrics_compare_against_the_training_mean_direction(self):
        rng = np.random.default_rng(4)
        n = 200
        ang = rng.uniform(-0.6, 0.6, size=n)
        u = np.stack([np.cos(ang), np.sin(ang) * 0.5, np.sin(ang)], axis=1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        depth = np.tile([0.0, 0.0, 1.0], (n, 1))
        ok = np.ones(n, dtype=bool)
        ok[150:160] = False
        ds = {"wrist": u.astype(np.float32), "wrist_depth": depth.astype(np.float32), "wrist_ok": ok}
        train_idx, test_idx = np.arange(100), np.arange(100, n)
        m = train.wrist_metrics(ds, u * 3.0, train_idx, test_idx)  # 길이가 달라도 방향만 본다
        self.assertEqual({"train": 100, "test": 90}, m["frames"])
        self.assertLess(m["angleDeg"]["model"]["median"], 1e-3)
        self.assertGreater(m["angleDeg"]["meanDirection"]["median"], 5.0)
        self.assertAlmostEqual(1.0, m["slope"]["extension"], places=6)
        empty = train.wrist_metrics(dict(ds, wrist_ok=np.zeros(n, dtype=bool)), u, train_idx, test_idx)
        self.assertEqual({"frames": {"train": 0, "test": 0}}, empty)


class AugmentTest(unittest.TestCase):
    def setUp(self):
        self.segs = [np.arange(0, 150), np.arange(150, 260)]
        self.rng_x = np.random.default_rng(3)

    def offsets_of(self, aug, x_wide, i, out):
        """증강 결과의 각 샘플이 넓은 창의 몇 번째 칸부터 자른 것인지(0 … 2·shift)."""
        found = []
        for k, frame in enumerate(i.tolist()):
            hits = [o for o in range(2 * aug.shift + 1)
                    if torch.allclose(out[k], train.standardize_t(x_wide[frame: frame + 1, :, :, o: o + 60])[0], atol=1e-5)]
            self.assertEqual(1, len(hits))
            found.append(hits[0])
        return np.array(found)

    def test_batch_pieces_draw_the_same_minibatches_as_batches(self):
        a = train.batches(self.segs, np.random.default_rng(7))
        b = train.batch_pieces(self.segs, np.random.default_rng(7))
        self.assertEqual(len(a), len(b))
        for (ia, pa), (ib, pb, lengths) in zip(a, b):
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_array_equal(pa, pb)
            self.assertEqual(len(ib), sum(lengths))

    def test_center_is_the_plain_window(self):
        x_wide = torch.as_tensor(self.rng_x.normal(3.0, 2.0, (6, 2, 96, 70)), dtype=torch.float32)
        center = train.Augment(x_wide, 5, False).center()
        want = train.dataset.standardize(x_wide.numpy()[:, :, :, 5:65].astype(np.float64))
        np.testing.assert_allclose(center.numpy(), want, atol=1e-5)

    def test_shift_moves_whole_chunks_and_untouched_chunks_stay_centered(self):
        x_wide = torch.as_tensor(self.rng_x.normal(0.0, 1.0, (260, 2, 96, 70)), dtype=torch.float32)
        aug = train.Augment(x_wide, 5, False)
        rng = np.random.default_rng(11)
        seen = []
        for idx, _, lengths in train.batch_pieces(self.segs, rng):
            i = torch.as_tensor(idx)
            offs = self.offsets_of(aug, x_wide, i, aug.batch(i, lengths, rng))
            for chunk in np.split(offs, np.cumsum(lengths)[:-1]):
                self.assertEqual(1, len(set(chunk.tolist())))  # 한 묶음은 같은 이동
                seen.append(int(chunk[0]))
        self.assertIn(5, seen)  # 증강하지 않은 묶음은 가운데
        self.assertTrue(any(o != 5 for o in seen))
        self.assertTrue(all(0 <= o <= 10 for o in seen))

    def test_mask_hides_paper_sized_bands(self):
        x = torch.as_tensor(self.rng_x.normal(0.0, 1.0, (260, 2, 96, 60)), dtype=torch.float32)
        aug = train.Augment(x, 0, True)
        rng = np.random.default_rng(5)
        hidden_any = False
        for idx, _, lengths in train.batch_pieces(self.segs, rng):
            out = aug.batch(torch.as_tensor(idx), lengths, rng)
            zero = (out == 0).all(dim=1)  # (B, 96, 60): 두 채널 모두 0
            rows = zero.all(dim=2).sum(dim=1).numpy()  # 시간 띠(모든 칸이 0인 줄)
            cols = zero.all(dim=1).sum(dim=1).numpy()  # 거리 띠
            for r, c in zip(rows, cols):
                self.assertTrue(r == 0 or 5 <= r <= 15, r)
                self.assertTrue(c == 0 or 5 <= c <= 10, c)
                self.assertEqual(r == 0, c == 0)  # 시간·거리 띠를 함께 가린다
                hidden_any |= bool(r)
            untouched = rows == 0
            np.testing.assert_array_equal(out[torch.as_tensor(untouched)].numpy(), x[torch.as_tensor(idx[untouched])].numpy())
        self.assertTrue(hidden_any)


class FitTest(unittest.TestCase):
    def test_non_finite_loss_stops_with_a_reason(self):
        x = torch.full((8, 2, 96, 60), float("nan"))  # 첫 epoch부터 손실이 nan
        segs = [np.arange(8)]
        with self.assertRaises(FloatingPointError):
            train.fit(x, torch.zeros((8, 60)), segs, segs, epochs=2, seed=0, log=lambda line: None)

    def test_wall_clock_limit_stops_after_the_epoch_and_keeps_the_best_weights(self):
        x = torch.as_tensor(np.random.default_rng(23).normal(size=(8, 2, 96, 60)).astype(np.float32))
        segs = [np.arange(8)]
        model, history, best, stopped = train.fit(x, torch.zeros((8, 60)), segs, segs, epochs=5, seed=0, log=lambda line: None, max_minutes=0.0)
        self.assertEqual(("time", 1, 1), (stopped, len(history), best))  # 상한 0분: 첫 epoch 뒤 멈춤


class InstrumentsTest(unittest.TestCase):
    def test_baselines_and_templates_come_from_training_frames_only(self):
        rng = np.random.default_rng(21)
        ds, tr, va, te = synthetic(rng)
        pred = ds["y"] + 0.05 * rng.normal(size=ds["y"].shape).astype(np.float32)
        before = train.instruments(ds, pred, tr, te)
        changed = dict(ds, y=ds["y"].copy())
        changed["y"][va] += 0.3  # 검증 프레임만 바꾼다
        self.assertEqual(before, train.instruments(changed, pred, tr, te))  # 검증 프레임은 어떤 계기에도 쓰이지 않는다
        changed["y"][te] *= 1.5  # 시험 정답만 바꾼다: 기준선 자세는 그대로여야 한다
        after = train.instruments(changed, pred, tr, te)
        y = ds["y"].reshape(-1, 20, 3).astype(np.float64)
        test_true = changed["y"].reshape(-1, 20, 3)[te].astype(np.float64)
        self.assertAlmostEqual(float(np.mean(joints.frame_errors(y[tr].mean(axis=0)[None], test_true)) * 87.0), after["baselines"]["meanPoseMm"], places=9)
        self.assertAlmostEqual(float(np.mean(joints.frame_errors(joints.medoid(y[tr])[None], test_true)) * 87.0), after["baselines"]["medoidMm"], places=9)

    def test_jitter_does_not_read_as_tracking_and_bias_is_measured_toward_the_open_hand(self):
        rng = np.random.default_rng(22)
        ds, tr, _, te = synthetic(rng)
        y = ds["y"].reshape(-1, 20, 3).astype(np.float64)
        mean = y[tr].mean(axis=0)
        stuck = y.copy()  # 시험 프레임에서 평균 자세에 붙어 프레임마다 정답 σ만큼 번갈아 떨리는 예측
        stuck[te] = mean + (-1.0) ** np.arange(len(te))[:, None, None] * y[te].std(axis=0)
        c = train.instruments(ds, stuck.reshape(-1, 60), tr, te)["collapse"]
        self.assertGreater(c["sigmaRatioRaw"]["mean"], 0.9)  # 평활 전 σ비는 '쏠리지 않음'으로 읽힌다
        self.assertLess(c["sigmaRatioSmoothed"]["mean"], 0.2)  # 9프레임 평활 뒤에는 평균에 붙은 것이 드러난다
        self.assertLess(abs(c["slope"]["mean"]), 0.1)  # 기울기 ≈ 0: 정답을 따라가지 않는다
        five = y[tr][ds["labelled"][tr] & (ds["class_id"][tr] == 10)].mean(axis=0)
        u = (five - mean) / np.linalg.norm(five - mean)
        shifted = y + 0.1 * u  # 모든 프레임이 편 손 쪽으로 0.1(단위 손)
        m = train.instruments(ds, shifted.reshape(-1, 60), tr, te)
        other = te[ds["labelled"][te] & (ds["class_id"][te] != 10)]
        self.assertAlmostEqual(0.1 * 87.0, m["asl5Bias"]["modelMm"], places=6)
        self.assertAlmostEqual(float(np.mean(((mean - y[other]).reshape(len(other), -1) @ u.ravel()))) * 87.0, m["asl5Bias"]["meanPoseMm"], places=6)
        self.assertEqual(len(other), m["asl5Bias"]["frames"])
        self.assertEqual("test", m["noiseFloorSource"])  # 시험에 2 s 동작 유지 구간이 있으면 시험 정답에서 잰다
        free = np.arange(555, 600)  # 자유 동작뿐인 시험
        self.assertEqual("all", train.instruments(ds, shifted.reshape(-1, 60), tr, free)["noiseFloorSource"])  # 없으면 전체 정답(참고)


if __name__ == "__main__":
    unittest.main()
