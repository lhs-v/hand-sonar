"""마이크가 둘이면 얼마나 더 얻나(2026-11-02): hand_echo_sim(실측에 맞춘 떨림 0.1 mm, 움직이는 몫 0.125)에 두 번째 마이크를 두고
톱니·Tukey(60) 송신으로 되읽기(ridge·kNN, 평균 자세 대비 비)를 견준다. 결과와 해석은 docs/echo-profile.md §4.

    python -I tools/experiments/two_mics.py [--data data] [--session 20261028-215926]

- 스피커는 하나(손 쪽 끝, hand_echo_sim along 배치). 마이크 A는 지금 모형의 자리(팔꿈치 쪽 끝).
- 두 번째 마이크 후보: B1 = A 옆 1.2 cm(가까운 쌍), B2 = 손목 옆면(12시 쪽), B3 = 손 쪽 끝(스피커 옆 1.2 cm).
- 대조 A′: A와 같은 자리에 마이크를 하나 더 둔다(같은 메아리, 덧붙는 잡음만 독립) — 잡음 평균 효과만 있는 경우.
- 채널마다 (|원본|, 차분) 창 특징을 이어 붙여 hand_echo_sim.Sim.decode와 같은 ridge·kNN으로 되읽는다.
CPU 4코어에서 약 4분 걸린다(전달 함수 열 개).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hand_echo_sim as H  # noqa: E402
import sonar_signals as SG  # noqa: E402

K, SIGMA = 0.125, 0.1  # hand_echo_sim 맞추기 값(docs/sonar-signal.md §8)


def decode(sim, chans):
    """채널 여럿의 (|원본|, 차분) 창 특징을 이어 붙여 Sim.decode와 같은 ridge·kNN. (ridge 비, kNN 비, 평균 자세 MPJPE mm)."""
    rows = sim.prow[sim.use]
    mats = [(np.abs(med[:, H.HZ]), np.vstack([np.zeros((1, 60)), diff[:, H.HZ]])) for med, diff in chans]

    def feats(idx):
        r = rows[idx]
        return np.concatenate([np.concatenate([a_[np.maximum(r - lag, 0)], d_[np.maximum(r - lag, 0)]], 1)
                               for a_, d_ in mats for lag in H.LAGS], 1)
    Xtr, Xva, Xte = feats(sim.tr_i), feats(sim.va_i), feats(sim.te_i)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ytr, yva, yte = sim.y[sim.tr_i], sim.y[sim.va_i], sim.y[sim.te_i]
    mean = ytr.mean(0)

    def mp(pred, y):
        return float(np.linalg.norm((pred - y).reshape(-1, 20, 3), axis=2).mean())
    base = mp(np.broadcast_to(mean, yte.shape), yte)
    G = Xtr.T @ Xtr
    best = None
    for lam in (1e2, 1e3, 1e4, 1e5):
        Wt = np.linalg.solve(G + lam * np.eye(len(G)), Xtr.T @ (ytr - mean))
        e = mp(Xva @ Wt + mean, yva)
        if best is None or e < best[0]:
            best = (e, Wt)
    ridge = mp(Xte @ best[1] + mean, yte) / base
    _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
    P = Vt[:64].T
    Ztr, Zte = Xtr @ P, Xte @ P
    d2 = (Zte ** 2).sum(1)[:, None] - 2 * Zte @ Ztr.T + (Ztr ** 2).sum(1)[None]
    nn = np.argsort(d2, axis=1)[:, :10]
    w = 1 / (np.sqrt(np.maximum(np.take_along_axis(d2, nn, 1), 0)) + 1e-6)
    knn = mp((ytr[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True), yte) / base
    return ridge, knn, base * sim.L_mm


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="두 번째 마이크의 이득(손 메아리 시뮬레이션)")
    parser.add_argument("--data", default=H.DEFAULT_DATA, help="데이터 묶음 폴더(기본: 저장소 data/)")
    parser.add_argument("--session", default="20261028-215926")
    args = parser.parse_args()
    sim = H.Sim(args.data, args.session)
    a, y_a, Wc = sim.a, sim.y_a, sim.Wc
    spk = (Wc + 0.021 * a, a)
    mics = {
        "A 팔꿈치 쪽 끝(지금 모형)": (Wc - 0.021 * a, -a),
        "B1 A 옆 1.2 cm(가까운 쌍)": (Wc - 0.021 * a + 0.012 * y_a, -a),
        "B2 손목 옆면(12시 쪽)": (Wc + 0.021 * y_a, y_a),
        "B3 손 쪽 끝(스피커 옆 1.2 cm)": (Wc + 0.021 * a + 0.012 * y_a, a),
    }
    x = SG.repo("tukey", "saw")
    rng = np.random.default_rng(99)
    white0 = sim.white.copy()

    def fresh_noise():
        return (rng.normal(size=white0.shape) + 1j * rng.normal(size=white0.shape)) / np.sqrt(2)

    def run(name):
        Hh0, Hs = sim.channel(name, 0.0)
        Hh1 = sim.channel(name, SIGMA)[0]
        return sim.process(x, x, SG.BAND, 1 + H.SH * (K * Hh1 + (1 - K) * Hh0.mean(0)) + H.SC * Hs)

    out = {}
    for i, (name, mic) in enumerate(mics.items()):
        sim.configs[name] = (spk, mic)
        sim.white = white0 if i == 0 else fresh_noise()
        out[name] = run(name)
        print(f"채널 {name} 완료", flush=True)
    name_a = list(mics)[0]
    sim.white = fresh_noise()
    out["A′ A와 같은 자리(잡음만 독립)"] = run(name_a)
    names = list(out)
    print("\n되읽기 비(평균 자세 대비, 작을수록 좋음, 괄호 mm) — 송신 톱니·Tukey(60), 스피커 하나")
    combos = ([names[0]], [names[1]], [names[2]], [names[3]], [names[0], names[4]], [names[0], names[1]], [names[0], names[2]],
              [names[0], names[3]], [names[0], names[2], names[3]])
    for combo in combos:
        ridge, knn, base = decode(sim, [out[n] for n in combo])
        print(f"  {' + '.join(n.split()[0] for n in combo):12s} ridge {ridge:.3f} ({ridge * base:4.1f} mm) · kNN {knn:.3f} ({knn * base:4.1f} mm)"
              f"   [{' + '.join(combo)}]", flush=True)


if __name__ == "__main__":
    main()
