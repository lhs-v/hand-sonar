import unittest

import train_seq  # 사용자 설치 numpy·torch·timm 경로도 여기서 붙는다

import kinematics  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tests.test_train import kinematic_targets  # noqa: E402

DIM = 16


def tiny_encoder():
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten(), torch.nn.Linear(2 * 16, DIM))


def model(design, seed=0):
    torch.manual_seed(seed)
    t = kinematics.fit_template(kinematic_targets(40), bins=8)
    return train_seq.SeqModel(design, t, dim=DIM, deltas=(1, 2), encoder=tiny_encoder()).eval()


def windows(n, seed=1):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=(n, 2, 96, 60)).astype(np.float32))


class PaddingTest(unittest.TestCase):
    def test_padding_round_trips_and_marks_real_steps(self):
        flat = torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3)
        padded, mask = train_seq.to_padded(flat, [3, 5, 2])
        self.assertEqual((3, 5, 3), tuple(padded.shape))
        np.testing.assert_array_equal([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], mask.int().numpy())
        np.testing.assert_array_equal(flat.numpy(), train_seq.from_padded(padded, [3, 5, 2]).numpy())

    def test_delta_pairs_stay_inside_each_clip(self):
        src, dst, which = train_seq.delta_pairs([3, 5], (1, 2))
        got = sorted(zip(src.tolist(), dst.tolist(), which.tolist()))
        want = sorted([(0, 1, 0), (1, 2, 0), (0, 2, 1),
                       (3, 4, 0), (4, 5, 0), (5, 6, 0), (6, 7, 0), (3, 5, 1), (4, 6, 1), (5, 7, 1)])
        self.assertEqual(want, got)


class HopTest(unittest.TestCase):
    def test_hop_steps_are_spaced_and_phases_cover_the_frames(self):
        t = np.r_[np.zeros(100), np.cumsum(np.tile([0.0333, 0.0667], 60))]  # 프레임 100–219: 카메라 간격 33/67 ms 반반, 6 s
        seg = np.arange(100, 220)
        seqs = train_seq.hop_sequences(seg, t, 0.6)
        self.assertEqual(12, len(seqs))  # 위상 수 = hop ÷ 프레임 간격 평균(50 ms)
        for s in seqs:
            gaps = np.diff(t[s])
            self.assertTrue(np.all(np.abs(gaps - 0.6) < 0.07), gaps)
        self.assertGreater(len(np.unique(np.concatenate(seqs))), 0.9 * len(seg))  # 위상을 모두 쓰면 거의 모든 프레임이 학습에 든다
        first = train_seq.hop_sequences(seg, t, 0.6, all_phases=False)
        self.assertEqual(1, len(first))
        self.assertEqual(100, int(first[0][0]))
        same = train_seq.hop_sequences(seg, t, 0.0)
        self.assertEqual(1, len(same))
        np.testing.assert_array_equal(seg, same[0])

    def test_crop_keeps_the_latest_rows_and_restandardizes(self):
        x = windows(3, seed=7) * 3.0 + 1.0
        c = train_seq.crop(x, 24)
        self.assertEqual((3, 2, 24, 60), tuple(c.shape))
        np.testing.assert_allclose(c.mean(dim=(-2, -1)).numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(c.std(dim=(-2, -1), correction=0).numpy(), 1.0, atol=1e-4)
        self.assertIs(x, train_seq.crop(x, 96))


class SeqModelTest(unittest.TestCase):
    def test_none_design_reads_each_window_alone(self):
        m = model("none")
        x = windows(5, seed=8)
        with torch.no_grad():
            base, aux = m(x, [5])
            y = x.clone()
            y[1] += 1.0
            moved, _ = m(y, [5])
        np.testing.assert_allclose(np.delete(moved.numpy(), 1, axis=0), np.delete(base.numpy(), 1, axis=0), atol=1e-6)
        self.assertIsNone(aux["zhat"])

    def test_both_designs_look_only_at_the_past_and_carry_it_forward(self):
        x = windows(6)
        for design in ("state", "world"):
            m = model(design)
            with torch.no_grad():
                base, _ = m(x, [6])
                y = x.clone()
                y[3] += 1.0
                moved, _ = m(y, [6])
            np.testing.assert_allclose(moved[:3].numpy(), base[:3].numpy(), atol=1e-6, err_msg=design)  # 앞 프레임은 그대로
            self.assertGreater(float((moved[5] - base[5]).abs().max()), 1e-6, design)  # 바뀐 창 두 걸음 뒤에도 남는다(상태로 넘어감)

    def test_world_design_feeds_the_predicted_latent_as_the_next_hidden_state(self):
        m = model("world")
        x = windows(4, seed=2)
        with torch.no_grad():
            _, aux = m(x, [4])
            z = m.encoder(x)
            h = m.h0[None]
            for t in range(4):
                prior = m.prior(h)
                np.testing.assert_allclose(aux["zhat"][t].numpy(), prior[0].numpy(), atol=1e-5)
                h = m.cell(z[t: t + 1], prior)
                np.testing.assert_allclose(aux["h"][t].numpy(), h[0].numpy(), atol=1e-5)

    def test_world_prior_is_layer_normalized_and_the_state_stays_bounded(self):
        m = model("world")
        with torch.no_grad():
            for p in m.predictor.parameters():
                p.mul_(10.0)  # 크기를 키우는 예측기(발산했던 씨앗값 1·2처럼)
            prior = m.prior(torch.randn(5, DIM) * 100.0)
            np.testing.assert_allclose(prior.mean(dim=1).numpy(), 0.0, atol=1e-4)  # 목표 z̄와 같은 공간(층 정규화)
            np.testing.assert_allclose(prior.std(dim=1, correction=0).numpy(), 1.0, atol=1e-3)
            _, aux = m(windows(300, seed=9), [300])
        self.assertTrue(bool(torch.isfinite(aux["h"]).all()))
        self.assertLess(float(aux["h"].abs().max()), DIM ** 0.5 + 1.0)

    def test_state_design_has_no_predictor_in_the_loop(self):
        m = model("state")
        x = windows(5, seed=3)
        with torch.no_grad():
            base, _ = m(x, [5])
            for p in m.predictor.parameters():
                p.add_(1.0)
            again, _ = m(x, [5])
        np.testing.assert_allclose(again.numpy(), base.numpy(), atol=1e-6)

    def test_jepa_targets_come_from_the_ema_copy_without_gradient(self):
        for design in ("state", "world"):
            m = model(design).train()
            x = windows(6, seed=4)
            _, aux = m(x, [4, 2])
            loss = m.jepa_loss(aux, m.targets(x), [4, 2])
            self.assertGreater(float(loss), 0.0, design)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in m.target.parameters()), design)
            self.assertIsNotNone(next(m.encoder.parameters()).grad, design)

    def test_ema_moves_the_target_a_little_toward_the_online_encoder(self):
        m = model("state")
        with torch.no_grad():
            for p in m.encoder.parameters():
                p.add_(1.0)
        before = [p.clone() for p in m.target.parameters()]
        m.ema(0.9)
        for b, t, o in zip(before, m.target.parameters(), m.encoder.parameters()):
            np.testing.assert_allclose(t.detach().numpy(), (0.9 * b + 0.1 * o.detach()).numpy(), atol=1e-6)


class PredictTest(unittest.TestCase):
    def test_state_starts_fresh_at_every_segment(self):
        x = windows(9, seed=5)
        for design in train_seq.DESIGNS:
            m = model(design)
            both = train_seq.predict_seq(m, x, [np.arange(0, 4), np.arange(4, 9)])
            alone = train_seq.predict_seq(m, x, [np.arange(4, 9)])
            np.testing.assert_allclose(both[4:9], alone[4:9], atol=1e-5, err_msg=design)
            self.assertEqual((9, 60), both.shape)


class FitTest(unittest.TestCase):
    def test_short_fit_records_pose_and_jepa_losses_and_skips_pose_on_unlabelled_clips(self):
        n = 48
        y_np = kinematic_targets(41, n=n).reshape(n, 60).astype(np.float32)
        x = windows(n, seed=6)
        y = torch.as_tensor(y_np)
        angles = torch.as_tensor(kinematics.ik(y_np.reshape(n, 20, 3)).reshape(n, 30), dtype=torch.float32)
        t = kinematics.fit_template(y_np[:32].reshape(-1, 20, 3), bins=8)
        segs = [np.arange(0, 32)]
        extra = [np.arange(40, 48)]  # 라벨 없이 JEPA에만 쓰는 덩어리
        for design in train_seq.DESIGNS:
            for rows in (96, 24):
                m, history, best_epoch, stopped = train_seq.fit(x, y, angles, segs, [np.arange(32, 40)], 2, 0, design, t, jepa=0.1,
                                                                extra_segs=extra, log=lambda *_: None, encoder=tiny_encoder(), dim=DIM,
                                                                clip=8, batch=16, rows=rows)
                self.assertEqual(2, len(history))
                self.assertTrue(all(np.isfinite([h["train"], h["val"]]).all() for h in history), design)
                if design == "none":
                    self.assertIsNone(history[0]["jepa"])  # 상태 없는 판은 JEPA를 쓰지 않는다
                else:
                    self.assertGreater(history[0]["jepa"], 0.0)
                self.assertIn(best_epoch, (1, 2))


if __name__ == "__main__":
    unittest.main()
