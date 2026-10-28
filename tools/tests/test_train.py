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
