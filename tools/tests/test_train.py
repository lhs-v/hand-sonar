import unittest

import train  # 사용자 설치 numpy·torch·timm 경로도 여기서 붙는다

import numpy as np  # noqa: E402
import torch  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
