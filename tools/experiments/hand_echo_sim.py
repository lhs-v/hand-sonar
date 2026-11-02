"""손 메아리 시뮬레이션: 실제 세션의 손 관절로 메아리를 합성해, 송신 신호 후보를 저장소와 같은 처리·잣대로 견준다(2026-11-02).
결과와 해석은 docs/sonar-signal.md.

    python -I tools/experiments/hand_echo_sim.py --out <저장소 밖 폴더> [--data data] [--session 20261028-215926]
    python -I tools/experiments/hand_echo_sim.py --out <폴더> --sigma 0.1 --k 0.125   # 맞추기를 건너뛰고 값을 준다

데이터 묶음(data/, export_data.py)만으로 돈다: 세션의 echo.npz(칸 250–479·줄 시각)·landmarks.csv.gz·pairs.csv.gz·sync.json, 시험 녹음 echo.npz.
CPU 4코어에서 약 15분 걸린다(전달 함수 하나에 약 40초).

1. 실측 잡음(시험 녹음·세션): 직접 경로 복소 값의 줄 사이 세기·위상 흔들림, 손 자리 포락선 ÷ 직접 경로, 차분 바닥.
   위상으로 잰 시간 흔들림은 0.0004–0.0009칸이라(봉우리 위치로 재면 추정 잡음 탓에 0.04칸) 시계는 안정적이고, 세기 흔들림(줄 사이 0.12–0.27%)이 남는다.
2. 모형(가정):
   - 관절: MediaPipe 세계 좌표를 거울 되돌림, 3프레임 평균, 줄 가운데 시각(+ sync.json 지연)으로 보간한 뒤 손바닥 틀로 옮긴다. 워치를 손바닥 틀에
     붙인 셈이라 손가락 굽힘만 메아리를 바꾼다. 팔뚝을 카메라 축에 고정하면(손 방향 변화를 모두 손목 꺾임으로 봄) 손 자리 차분이 실측보다
     15–23 dB 커서 쓰지 않았다.
   - 워치: 손목에서 팔뚝 쪽 3.5 cm, 손등 쪽 2 cm. 팔뚝 축 = 세션 중앙 손 방향을 손바닥 쪽으로 25°(평소 젖힘, 연구 현황 5.8).
   - 소리 구멍 배치 셋: along = 몸체 양 끝(손 쪽·팔꿈치 쪽, ±2.1 cm), hand = 둘 다 손 쪽 끝(옆으로 ±1.2 cm), across = 손목 양옆.
     상반성 때문에 스피커와 마이크를 맞바꿔도 메아리 경로는 같다. 구멍 방향성 0.2 + 0.8·(1 + cos θ)/2(뒤쪽 −14 dB, 몸체 그늘 근사).
   - 반사점: 손가락 마디(첫 마디 9점, 나머지 6점, 폭 1.6 cm), 손바닥 부채꼴 30점, 손목·팔뚝 피부 27점(고정). 피부는 공기와 임피던스 차가 커서
     거의 다 반사하므로 크기 = 면적 × 방향성 ÷ (스피커 거리 × 마이크 거리). 가려짐·다중 반사·한 줄 안 움직임(도플러)은 넣지 않는다.
   - 메아리 비례 상수(손 SH 3.16, 고정 SC 0.1)와 덧붙는 잡음(SIG 0.0867)은 첫 판(팔뚝 고정 모형)에서 창 없음 시험 녹음 20261027-203239의
     손 자리 포락선(칸 303–360 일곱 점, RMS 3.6 dB)과 먼 칸 차분 바닥(−70.8 dB)에 맞춘 값을 그대로 쓴다.
   - 잡음: 직접 경로 세기 줄 사이 σ 0.2%, 시간 0.0005칸, 덧붙는 백색 잡음, 손 떨림(4–12 Hz, 손바닥 + 손가락 다섯, 3축). 모든 신호에 같은 표본.
3. 맞추기: 손 메아리 = k·움직이는 손 + (1 − k)·세션 평균(움직이지 않는 몫, 손 자리 평균 세기는 그대로). 떨림 σ × k 격자에서 hann 시뮬레이션이
   같은 세션 실제 hann 녹음의 손 자리 |차분| 평균(직접 경로 대비 중앙·95%), signal_check 상관·대비에 가장 가까운 값을 고른다
   (오차 = (dB 차 ÷ 2)² 둘 + (상관 차 ÷ 0.05)² + (ln 대비 비 ÷ 0.15)²). 넷을 함께 맞추는 값은 없고 대비가 실측보다 높게 남는다.
4. 견주기: 후보마다 같은 잡음으로 처리해 signal_check 잣대(상관·대비), 손 자리 차분·원본 SNR(포즈 몫 ÷ 떨림·흔들림·잡음),
   되읽기(MediaPipe 정답, 10 s 블록 5개 중 3 학습·1 검증·1 시험, 창 1.2 s의 8줄 × 손 자리 60칸 × 원본·차분 → ridge·kNN, 평균 자세 대비 비).
   떨림을 두 배로 한 대안, 소리 구멍 배치 셋도 본다.
5. 대역: 3·4.4·6 kHz 톱니(창 없음·Tukey)의 꼬리를 손가락·손바닥 메아리가 몰리는 거리(세기 가중)로 평균한다.
그림(--out): sonar-handsim.png, sonar-null-vs-hand.png. 숫자: hand_echo_sim.json.
"""
import argparse
import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools
sys.path.insert(0, HERE)  # experiments(sonar_signals)
import hs_io  # noqa: E402  사용자 설치 numpy·scipy 경로도 붙는다
import joints  # noqa: E402
import signal_check  # noqa: E402
import sonar_signals as SG  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

C, RATE, N = SG.C, SG.RATE, SG.N
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
HZ = slice(301, 361)  # 손 자리 칸
SH, SC, SIG = 3.16, 0.1, 0.0867
GAIN_STEP, DELAY_STEP = 0.002, 0.0005  # 직접 경로 줄 사이 세기(비율)·시간(칸) 흔들림
GRID_SIGMA = (0.0, 0.05, 0.1, 0.2, 0.4)  # 떨림 mm
GRID_K = (1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015)
MSEL = np.flatnonzero((SG.F >= 15_500) & (SG.F <= 23_000))
FM = SG.F[MSEL]
CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
LAGS = (0, 4, 8, 16, 32, 48, 64, 95)  # 되읽기 창(1.2 s) 안에서 쓰는 줄
TEST_RECORDINGS = (("20261027-203239", "창 없음·볼륨 11"), ("20261027-202740", "창 없음·볼륨 15"), ("20261027-203600", "hann·볼륨 11"))


def rstd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def measure_noise(path):
    """시험 녹음·세션 echo.npz에서 직접 경로 흔들림과 손 자리 세기."""
    z = np.load(path)
    raw = z["raw"].astype(np.float64)
    b0 = int(z["bin0"])
    ok = z["rows_ok"].astype(bool) & (np.abs(raw).sum(axis=0) > 0)
    an = hilbert(raw, axis=0)
    env = np.abs(an)
    pair = ok[1:] & ok[:-1]
    A = an[300 - b0]
    dlog = np.diff(np.log(np.abs(A)))[pair]
    dph = np.angle(A[1:] * np.conj(A[:-1]))[pair]
    pk = env[290 - b0: 311 - b0].max(axis=0)
    rel = np.median(env[301 - b0: 361 - b0][:, ok] / pk[ok], axis=1)
    diff = np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])
    near = np.median(np.abs(diff[301 - b0: 361 - b0][:, pair]) / pk[1:][pair])
    far = np.median(np.abs(diff[440 - b0: 480 - b0][:, pair]) / pk[1:][pair])
    return dict(gain=rstd(dlog), phase_rad=rstd(dph), timing_bins=rstd(dph) / (2 * np.pi * 19_500) * RATE,
                rel={b: float(rel[b - 301]) for b in (303, 310, 320, 330, 340, 350, 360)}, near=float(near), far=float(far))


class Sim:
    def __init__(self, data, name):
        folder = os.path.join(data, "sessions", name)
        E = np.load(os.path.join(folder, "echo.npz"))
        self.rows_ns = E["rows_ns"].astype(np.float64)
        self.R = len(self.rows_ns)
        self.raw = E["raw"].astype(np.float64)
        self.b0 = int(E["bin0"])
        with gzip.open(os.path.join(folder, "landmarks.csv.gz"), "rt", encoding="utf-8") as f:
            lrows = list(csv.reader(f))[1:]
        self.frame_ns = np.array([int(r[0]) for r in lrows], dtype=np.int64)
        world = np.array([[float(v) for v in r[67:130]] if r[2] else [np.nan] * 63 for r in lrows]).reshape(-1, 21, 3)
        score = np.array([float(r[3]) if r[2] else np.nan for r in lrows])
        with gzip.open(os.path.join(folder, "pairs.csv.gz"), "rt", encoding="utf-8") as f:
            prs = list(csv.DictReader(f))
        self.prow = np.array([int(r["row"]) if r["row"] else -1 for r in prs])
        direct = np.array([r["direct"] == "1" for r in prs])
        valid = np.isfinite(world).all(axis=(1, 2))
        good, _ = joints.keep_frames(world, score, valid)
        self.use = np.flatnonzero(valid & direct & good & (self.prow >= 1))
        y, _, _ = joints.targets(world[self.use], "left")
        self.y = y.reshape(len(self.use), -1)
        real = joints.to_left(world[self.use], "left")
        self.L_mm = float(np.median(np.linalg.norm(real[:, 17] - real[:, 0], axis=1)) * 1000)
        lag_ms = hs_io.num(hs_io.read_json(os.path.join(folder, "sync.json")), "lagMs") or 0.0
        p = joints.to_left(world[valid], "left")
        q = p - p[:, :1]
        k3 = np.ones(3) / 3
        q = np.stack([np.stack([np.convolve(q[:, j, c], k3, mode="same") for c in range(3)], -1) for j in range(21)], 1)
        q[0], q[-1] = q[1], q[-2]
        tq = self.frame_ns[valid].astype(np.float64)
        tr = self.rows_ns + lag_ms * 1e6
        Q = np.stack([np.stack([np.interp(tr, tq, q[:, j, c]) for c in range(3)], -1) for j in range(21)], 1)
        self.Q, _ = joints.palm_frame(Q + 1e-12)  # 워치를 손바닥 틀에 붙임
        h = self.Q[:, 9] / np.linalg.norm(self.Q[:, 9], axis=1, keepdims=True)
        _, rot = joints.palm_frame(self.Q + 1e-12)
        h_med = np.median(h, axis=0)
        h_med /= np.linalg.norm(h_med)
        n_med = np.median(rot[:, 2], axis=0)
        n_med -= h_med * (n_med @ h_med)
        n_med /= np.linalg.norm(n_med)
        ext = np.deg2rad(25)
        self.a = np.cos(ext) * h_med - np.sin(ext) * n_med
        self.n_a = np.sin(ext) * h_med + np.cos(ext) * n_med
        self.y_a = np.cross(self.n_a, self.a)
        self.Wc = -0.035 * self.a + 0.02 * self.n_a
        p_dist, p_prox = self.Wc + 0.021 * self.a, self.Wc - 0.021 * self.a
        self.configs = {"along": ((p_dist, self.a), (p_prox, -self.a)),
                        "hand": ((p_dist + 0.012 * self.y_a, self.a), (p_dist - 0.012 * self.y_a, self.a)),
                        "across": ((self.Wc + 0.021 * self.y_a, self.y_a), (self.Wc - 0.021 * self.y_a, -self.y_a))}
        rng = np.random.default_rng(7)
        self.gain = 1 + rng.normal(0, GAIN_STEP / np.sqrt(2), self.R)
        self.delay = rng.normal(0, DELAY_STEP / np.sqrt(2), self.R)
        self.white = (rng.normal(size=(self.R, len(MSEL))) + 1j * rng.normal(size=(self.R, len(MSEL)))) / np.sqrt(2)
        fr_t = self.frame_ns[self.use].astype(np.float64)
        blk = ((fr_t - fr_t[0]) / 10e9).astype(int)
        keep = (fr_t - fr_t[0]) / 1e9 - blk * 10 >= 1.3  # 창이 블록 경계를 넘지 않게
        role = blk % 5
        self.tr_i = np.flatnonzero(keep & (role <= 2))
        self.va_i = np.flatnonzero(keep & (role == 3))
        self.te_i = np.flatnonzero(keep & (role == 4))

    # ----- 반사점과 전달 함수 -----
    @staticmethod
    def points(Qr):
        """(점 (K, 3), 면적 무게 (K,), 무리 번호 (K,): 0–4 손가락, 5 손바닥)."""
        pts, wts, grp = [], [], []
        for g, ch in enumerate(CHAINS):
            for s in range(3):
                i, j = ch[s], ch[s + 1]
                n = 9 if s == 0 else 6
                u = (np.arange(n) + 0.5) / n
                pts.append(Qr[i] + u[:, None] * (Qr[j] - Qr[i]))
                wts.append(np.full(n, np.linalg.norm(Qr[j] - Qr[i]) / n * 0.016))
                grp.append(np.full(n, g))
        for s in np.linspace(0.25, 1.0, 6):
            for u in np.linspace(0, 1, 5):
                pts.append((s * ((1 - u) * Qr[5] + u * Qr[17]))[None])
                wts.append(np.array([0.0045 * s]))
                grp.append(np.array([5]))
        return np.concatenate(pts), np.concatenate(wts), np.concatenate(grp)

    def static_points(self):
        pts = [-s * self.a + 0.010 * self.n_a + b * self.y_a for s in (0.004, 0.010, 0.016) for b in (-0.015, 0.0, 0.015)]
        pts += [-s * self.a + 0.012 * self.n_a + b * self.y_a for s in np.arange(0.065, 0.17, 0.02) for b in (-0.02, 0.0, 0.02)]
        return np.array(pts), np.full(len(pts), 0.00006)

    @staticmethod
    def port_gain(x, port, facing):
        v = x - port
        r = np.linalg.norm(v, axis=-1)
        return r, 0.2 + 0.8 * 0.5 * (1 + (v @ facing) / r)

    def tremor(self, sigma_mm, seed=11):
        if sigma_mm <= 0:
            return np.zeros((self.R, 6, 3))
        sos = butter(2, [4, 12], btype="bandpass", fs=RATE / N, output="sos")
        z = sosfiltfilt(sos, np.random.default_rng(seed).normal(size=(self.R, 6, 3)), axis=0)
        z /= z.std(axis=0, keepdims=True)
        z[:, :5] = (z[:, :5] + z[:, 5:6]) / np.sqrt(2)
        return z * sigma_mm * 1e-3

    def channel(self, cfg, sigma_mm=0.0, chunk=256):
        """(R, M) 손 메아리와 (M,) 고정 메아리 전달 함수(직접 경로 = 1, 지연 0)."""
        (spk, sf), (mic, mf) = self.configs[cfg]
        d0 = np.linalg.norm(spk - mic)
        tz = self.tremor(sigma_mm)
        Hh = np.zeros((self.R, len(MSEL)), complex)
        _, _, grp = self.points(self.Q[0])
        for s in range(0, self.R, chunk):
            e = min(self.R, s + chunk)
            pw = [self.points(self.Q[r]) for r in range(s, e)]
            P = np.stack([v[0] for v in pw]) + tz[s:e][:, grp]
            wt = np.stack([v[1] for v in pw])
            rs, ds = self.port_gain(P, spk, sf)
            rm, dm = self.port_gain(P, mic, mf)
            Hh[s:e] = np.einsum("rk,rkm->rm", wt * ds * dm / (rs * rm), np.exp(-2j * np.pi * ((rs + rm - d0) / C)[..., None] * FM))
        Ps, ws = self.static_points()
        rs, ds = self.port_gain(Ps, spk, sf)
        rm, dm = self.port_gain(Ps, mic, mf)
        Hs = (ws * ds * dm / (rs * rm)) @ np.exp(-2j * np.pi * ((rs + rm - d0) / C)[:, None] * FM)
        return Hh, Hs

    # ----- 처리와 잣대 -----
    def process(self, x, template, band, H, noisy=True, peak=0.5):
        """profiles.py와 같은 처리: 대역 통과, 원형 상관, 칸 300 = 직접 경로, 3줄 중앙값, 차분 |p[t+1]| − |p[t]|."""
        X = np.fft.rfft(x * peak)[MSEL]
        Tm = np.fft.rfft(template * peak)[MSEL]
        Hb = SG.bandpass_mag2(*band)[MSEL]
        Y = X * H
        if noisy:
            Y = Y * (self.gain[:, None] * np.exp(-2j * np.pi * FM * self.delay[:, None] / RATE)) + SIG * self.white
        full = np.zeros((self.R, len(SG.F)), complex)
        full[:, MSEL] = Y * np.conj(Tm) * Hb
        prof = np.roll(np.fft.irfft(full, n=N, axis=1), N // 2, axis=1)
        a = np.vstack([prof[:1], prof[:-1]])
        c = np.vstack([prof[1:], prof[-1:]])
        med = np.maximum(np.minimum(a, prof), np.minimum(np.maximum(a, prof), c))
        return med, np.abs(med[1:]) - np.abs(med[:-1])

    def sc_metrics(self, diff_hand):
        """signal_check와 같은 계산(손 자리 |차분| 평균 → 속도–세기 로그 상관 지연 0, 움직임 대비)."""
        energy = np.abs(diff_hand).mean(axis=1)
        corr = signal_check.speed_energy_corr(self.y, self.frame_ns[self.use], self.prow[self.use], energy, lags=[0])[0]
        lo, hi = int(self.prow[self.use].min()), int(self.prow[self.use].max())
        return float(corr), float(signal_check.motion_contrast(energy[lo:hi]))

    def levels(self, med, diff_hand):
        """손 자리 |차분| 평균의 중앙·95%(직접 경로 대비 dB)."""
        lo, hi = int(self.prow[self.use].min()), int(self.prow[self.use].max())
        es = np.abs(diff_hand).mean(axis=1)[lo:hi]
        pk = np.median(np.abs(med[:, 295:306]).max(axis=1))
        return float(20 * np.log10(np.median(es) / pk)), float(20 * np.log10(np.percentile(es, 95) / pk))

    def real_targets(self):
        b0 = self.b0
        diff = (np.abs(self.raw[:, 1:]) - np.abs(self.raw[:, :-1]))[301 - b0: 361 - b0].T
        corr, con = self.sc_metrics(diff)
        lo, hi = int(self.prow[self.use].min()), int(self.prow[self.use].max())
        es = np.abs(diff).mean(axis=1)[lo:hi]
        pk = np.median(np.abs(self.raw[295 - b0: 306 - b0]).max(axis=0))
        return dict(m50=float(20 * np.log10(np.median(es) / pk)), m95=float(20 * np.log10(np.percentile(es, 95) / pk)), corr=corr, contrast=con)

    def decode(self, med, diff):
        """창 1.2 s 안 8줄 × 손 자리 60칸 × (|원본|, 차분) → ridge(검증으로 λ)·kNN(PCA 64, k 10) 평균 자세 대비 비."""
        a = np.abs(med[:, HZ])
        d = np.vstack([np.zeros((1, 60)), diff[:, HZ]])
        rows = self.prow[self.use]

        def feats(idx):
            r = rows[idx]
            return np.concatenate([np.concatenate([a[np.maximum(r - lag, 0)], d[np.maximum(r - lag, 0)]], 1) for lag in LAGS], 1)
        Xtr, Xva, Xte = feats(self.tr_i), feats(self.va_i), feats(self.te_i)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        ytr, yva, yte = self.y[self.tr_i], self.y[self.va_i], self.y[self.te_i]
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
        return ridge, knn, base * self.L_mm

    def echo_distances(self, cfg="along", step=50):
        """손가락·손바닥 메아리 거리(cm, 칸과 같은 뜻 = 경로 차 ÷ 2)와 세기, 고정 반사 거리."""
        (spk, sf), (mic, mf) = self.configs[cfg]
        d0 = np.linalg.norm(spk - mic)
        out = {"finger": ([], []), "palm": ([], [])}
        for r in range(0, self.R, step):
            P, wt, grp = self.points(self.Q[r])
            rs, ds = self.port_gain(P, spk, sf)
            rm, dm = self.port_gain(P, mic, mf)
            d = (rs + rm - d0) / 2 * 100
            amp = wt * ds * dm / (rs * rm)
            for key, m in (("finger", grp < 5), ("palm", grp == 5)):
                out[key][0].append(d[m])
                out[key][1].append(amp[m])
        Ps, _ = self.static_points()
        rs, _ = self.port_gain(Ps, spk, sf)
        rm, _ = self.port_gain(Ps, mic, mf)
        res = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in out.items()}
        res["static"] = (rs + rm - d0) / 2 * 100
        return res


def runs():
    cand = SG.candidates()
    saw, han, tri = cand["톱니·창 없음"][0], cand["톱니·hann"][0], cand["삼각·창 없음"][0]
    out = {k: (x, x, b) for k, (x, b) in cand.items()}
    out["hann + 평탄화 수신"] = (han, SG.eq_template(han, saw, lam=0.02), SG.BAND)
    out["삼각 + 평탄화 수신"] = (tri, SG.eq_template(tri, saw, lam=0.05), SG.BAND)
    out["창 없음 + 수신 Tukey"] = (saw, SG.spectral_template(saw, "tukey"), SG.BAND)
    return out


SUBSET = ("톱니·창 없음", "톱니·Tukey(60)", "톱니·hann", "삼각·창 없음", "넓은 대역 17–21.4 kHz")


def compare(sim, cfg, Hh0, Hh1, Hs, k, names, examples=None):
    mean0 = Hh0.mean(axis=0)
    H0 = 1 + SH * (k * Hh0 + (1 - k) * mean0) + SC * Hs  # 포즈 몫만
    H1 = 1 + SH * (k * Hh1 + (1 - k) * mean0) + SC * Hs
    R = runs()
    out = {}
    for name in names:
        x, tm, band = R[name]
        med_c, diff_c = sim.process(x, tm, band, H0, noisy=False)
        med, diff = sim.process(x, tm, band, H1)
        raw_c = np.abs(med_c[:, HZ])
        snr_d = 10 * np.log10(diff_c[:, HZ].var() / (diff[:, HZ] - diff_c[:, HZ]).var())
        snr_r = 10 * np.log10((raw_c - raw_c.mean(0)).var() / (np.abs(med[:, HZ]) - raw_c).var())
        corr, con = sim.sc_metrics(diff[:, HZ])
        ridge, knn, base_mm = sim.decode(med, diff)
        out[name] = dict(corr=corr, contrast=con, snr_d=float(snr_d), snr_r=float(snr_r), ridge=ridge, knn=knn, base_mm=base_mm)
        print(f"  [{cfg:6s}] {name:22s} 상관 {corr:5.3f} · 대비 {con:4.2f} · 차분 SNR {snr_d:5.1f} dB · 원본 SNR {snr_r:5.1f} dB · "
              f"되읽기 비 ridge {ridge:5.3f} ({ridge * base_mm:4.1f} mm) kNN {knn:5.3f} ({knn * base_mm:4.1f} mm)", flush=True)
        if examples is not None and name in SUBSET[:4]:
            examples[name] = diff[6000:6640, 300:361].astype(np.float32)
    return out


def bandwidth_table(sim):
    dist = sim.echo_distances()
    (df, af), (dp, ap) = dist["finger"], dist["palm"]

    def wq(d, w, q):
        o = np.argsort(d)
        return float(d[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), q)])
    print(f"\n메아리가 몰리는 거리(세기 가중 10·50·90%): 손가락 {wq(df, af, .1):.1f}·{wq(df, af, .5):.1f}·{wq(df, af, .9):.1f} cm, "
          f"손바닥 {wq(dp, ap, .1):.1f}·{wq(dp, ap, .5):.1f}·{wq(dp, ap, .9):.1f} cm, 손목·팔뚝 피부 {dist['static'].min():.1f}–{dist['static'].max():.1f} cm")
    table = {}
    print("대역별 직접 경로 꼬리(직접 경로 = 100): 0이 되는 거리 · 3.6/7/10/15 cm · 손가락 자리 평균 · 손바닥 자리 평균")
    for lab, f0, f1, tk in (("3 kHz 18–21 창 없음", 18e3, 21e3, False), ("3 kHz 18–21 Tukey", 18e3, 21e3, True),
                            ("4.4 kHz 17–21.4 창 없음", 17e3, 21.4e3, False), ("4.4 kHz 17–21.4 Tukey", 17e3, 21.4e3, True),
                            ("6 kHz 16–22 창 없음", 16e3, 22e3, False), ("6 kHz 16–22 Tukey", 16e3, 22e3, True)):
        x = SG.repo("tukey" if tk else "none", "saw") if f0 == 18e3 else SG.wide_saw(f0, f1, tk)
        env = SG.psf(x, band=(f0, f1))
        row = dict(null_cm=SG.first_null(env) * SG.CM, tails=[float(v) for v in SG.tail(env, SG.DIST)],
                   finger=float(np.sum(af * SG.tail(env, df)) / af.sum()), palm=float(np.sum(ap * SG.tail(env, dp)) / ap.sum()))
        table[lab] = row
        print(f"  {lab:22s} {row['null_cm']:4.1f} cm · " + "/".join(f"{v:.0f}" for v in row["tails"]) + f" · {row['finger']:5.1f} · {row['palm']:5.1f}")
    return table, dict(finger=(wq(df, af, .1), wq(df, af, .5), wq(df, af, .9)), palm=(wq(dp, ap, .1), wq(dp, ap, .5), wq(dp, ap, .9)),
                       static=(float(dist["static"].min()), float(dist["static"].max()))), dist


def figures(out_dir, sim, res, alt, examples, sigma, k, dist):
    plt = SG.korean_pyplot()
    from matplotlib.patches import Rectangle
    keys = list(runs())
    fig = plt.figure(figsize=(18.5, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
    fig.suptitle(f"실제 손 움직임으로 메아리를 합성해 송신 신호를 견줌 — 잡음은 같은 세션 hann 실측에 맞춤(떨림 {sigma:.2f} mm, 움직이는 몫 {k:.3f}; "
                 f"점 = 떨림 두 배)", fontsize=11.5)
    a = fig.add_subplot(gs[0, 0:2])
    qa, qn = sim.Q @ sim.a, sim.Q @ sim.n_a
    mid = sim.R // 2
    for ch in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17)):
        a.plot(qa[mid, list(ch)] * 100, qn[mid, list(ch)] * 100, "-o", color="#555555", ms=2.5, lw=1.2)
    wa, wn = sim.Wc @ sim.a * 100, sim.Wc @ sim.n_a * 100
    a.add_patch(Rectangle((wa - 2.1, wn - 0.6), 4.2, 1.2, color="#bbbbbb"))
    a.plot([-12, 0], [0.9, 0.9], color="#c9a27e", lw=6, alpha=0.6, solid_capstyle="round")
    a.plot(wa + 2.1, wn, "o", color="#1f6fb4", ms=7, label="along: 몸체 양 끝(손 쪽·팔꿈치 쪽)")
    a.plot(wa - 2.1, wn, "s", color="#1f6fb4", ms=7, mfc="none", mew=1.8)
    a.plot(wa + 2.1, wn + 0.35, "o", color="#2ca25f", ms=6, label="hand: 둘 다 손 쪽")
    a.plot(wa, wn, "^", color="#d95f02", ms=7, label="across: 손목 양옆(그림 평면 밖)")
    for r_cm in (5, 10, 15, 20):
        th = np.linspace(-1.2, 1.2, 100)
        a.plot(wa + 2.1 + r_cm * np.cos(th), wn + r_cm * np.sin(th), color="#fdae6b", lw=0.6, alpha=0.6)
    a.set(xlim=(-13, 22), ylim=(-9, 9), aspect="equal", xlabel="팔뚝 축 (cm, 손 쪽 +)", ylabel="손등 쪽 (cm)", title="(a) 모형(한 프레임, 동심원 5 cm 간격)")
    a.legend(loc="lower left", fontsize=8)
    for i, name in enumerate(SUBSET[:4]):
        axi = fig.add_subplot(gs[0, 2 + i])
        d = examples[name]
        v = np.percentile(np.abs(d), 99)
        axi.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=(300, 360, 8, 0))
        axi.set_title(f"(b) {name}", fontsize=9.5)
        axi.set_xlabel("거리 칸")
        if i == 0:
            axi.set_ylabel("시간 (s) · 손 자리 차분(그림마다 색 범위 99%)")
    y = np.arange(len(keys))
    c = fig.add_subplot(gs[1, 0:2])
    c.barh(y - 0.2, [res["along"][kk]["corr"] for kk in keys], 0.4, color="#1f6fb4", label="속도–세기 상관")
    c2 = c.twiny()
    con = [res["along"][kk]["contrast"] for kk in keys]
    c2.barh(y + 0.2, con, 0.4, color="#fdae6b", label="움직임 대비(위 축)")
    for kk, (lo, hi), lab in (("톱니·hann", (0.12, 0.36), "10-28 hann 실측"), ("톱니·창 없음", (0.30, 0.40), "10-31 톱니 실측"),
                              ("삼각·창 없음", (0.21, 0.37), "11-01 삼각 실측")):
        i = keys.index(kk)
        c.plot([lo, hi], [i - 0.2, i - 0.2], color="k", lw=3, alpha=0.6)
        c.text(hi + 0.01, i - 0.35, lab, fontsize=7.5)
    c.set_yticks(y, keys)
    c.invert_yaxis()
    c.set(xlabel="상관(검은 막대 = 실측 범위, 연구 현황 5.5·5.11)", xlim=(0, 0.75), title="(c) signal_check 잣대: 시뮬레이션 대 실측")
    c2.set(xlim=(0, max(con) * 1.6))
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    d = fig.add_subplot(gs[1, 2:4])
    kn = [res["along"][kk]["knn"] for kk in keys]
    rg = [res["along"][kk]["ridge"] for kk in keys]
    d.barh(y - 0.2, kn, 0.4, color="#2ca25f", label="kNN 되읽기 비(작을수록 좋음)")
    d.barh(y + 0.2, rg, 0.4, color="#a6d96a", label="ridge 되읽기 비")
    d.plot([alt[kk]["knn"] for kk in keys], y - 0.2, "k.", ms=6, label="떨림 두 배")
    d.plot([alt[kk]["ridge"] for kk in keys], y + 0.2, "k.", ms=6)
    d.axvline(1, color="k", lw=0.8)
    d.set_yticks(y, keys)
    d.invert_yaxis()
    d.set(xlim=(min(min(kn), min(rg)) - 0.03, 1.02), xlabel="시험 MPJPE ÷ 평균 자세 MPJPE(MediaPipe 정답, 10 s 블록)", title="(d) 관절 되읽기")
    d3 = d.twiny()
    d3.plot([res["along"][kk]["snr_d"] for kk in keys], y, "D", color="#7b3294", ms=5, label="손 자리 차분 SNR(위 축, dB)")
    h1, l1 = d.get_legend_handles_labels()
    h2, l2 = d3.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=2, fontsize=8)
    e = fig.add_subplot(gs[1, 4:6])
    for j, (cf, col, lab) in enumerate((("along", "#1f6fb4", "양 끝(손·팔꿈치)"), ("hand", "#2ca25f", "둘 다 손 쪽"), ("across", "#d95f02", "손목 양옆"))):
        e.bar(np.arange(len(SUBSET)) + (j - 1) * 0.27, [res[cf][kk]["snr_d"] for kk in SUBSET], 0.27, color=col, label=lab)
    e.set_xticks(np.arange(len(SUBSET)), SUBSET, rotation=15, fontsize=8.5)
    e.set(ylabel="손 자리 차분 SNR (dB)", title="(e) 소리 구멍 배치(스피커·마이크 맞바꿈은 같음)")
    e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    for axx in (c, d, e):
        axx.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "sonar-handsim.png"), dpi=105)
    plt.close(fig)

    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8.6), height_ratios=[2.2, 1], sharex=True)
    for lab, f0, f1, col in (("3 kHz 18–21 (지금, 창 없음)", 18e3, 21e3, "#1f6fb4"), ("3 kHz 18–21 Tukey", None, None, "#2ca25f"),
                             ("4.4 kHz 17–21.4 (가정)", 17e3, 21.4e3, "#e6ab02"), ("6 kHz 16–22 (가정, 들릴 위험 큼)", 16e3, 22e3, "#d95f02")):
        if f0 is None:
            env = SG.psf(SG.repo("tukey", "saw"))
        elif f0 == 18e3:
            env = SG.psf(SG.repo("none", "saw"))
        else:
            env = SG.psf(SG.wide_saw(f0, f1, False), band=(f0, f1))
        z = SG.first_null(env)
        a.plot(np.arange(70) * SG.CM, env[N // 2: N // 2 + 70], color=col, lw=2.2, label=f"{lab}: {z * SG.CM:.1f} cm에서 0")
        a.plot([z * SG.CM], [env[N // 2 + z]], "o", color=col, ms=7)
    a.set(ylim=(0, 102), ylabel="직접 경로 봉우리 = 100", title="대역폭이 넓을수록 직접 경로가 빨리 꺼진다(c ÷ 2B)")
    a.legend(loc="upper right", fontsize=9.5)
    a.grid(alpha=0.25)
    bins = np.arange(0, 25, 0.5)
    b.hist(dist["static"], bins=bins, weights=np.ones_like(dist["static"]) / len(dist["static"]), color="#bbbbbb", label="손목·팔뚝 피부(고정)")
    (dp, ap), (df, af) = dist["palm"], dist["finger"]
    b.hist(dp, bins=bins, weights=ap / ap.sum(), color="#fdae6b", alpha=0.8, label="손바닥")
    b.hist(df, bins=bins, weights=af / af.sum(), color="#7b3294", alpha=0.6, label="손가락")
    b.set(xlim=(0, 25), xlabel="워치에서 반사체까지 거리 (cm, 메아리 칸과 같은 뜻)", ylabel="메아리 세기 비율", title="모형에서 메아리가 몰리는 거리(세기 가중)")
    b.legend(loc="upper right", fontsize=9.5)
    b.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sonar-null-vs-hand.png"), dpi=110)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="손 메아리 시뮬레이션으로 송신 신호 후보 견주기")
    parser.add_argument("--out", required=True, help="그림·숫자를 쓸 폴더(저장소 밖)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="데이터 묶음 폴더(기본: 저장소 data/)")
    parser.add_argument("--session", default="20261028-215926", help="손 움직임과 맞추기에 쓸 hann 세션")
    parser.add_argument("--sigma", type=float, help="떨림 σ(mm). --k와 함께 주면 맞추기를 건너뛴다")
    parser.add_argument("--k", type=float, help="손 메아리 중 움직이는 몫")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result = {"noise": {}}
    print("1. 실측 잡음(직접 경로 = 1): 줄 사이 세기 σ · 위상 σ → 시간 흔들림 · 손 자리 포락선(칸 303/310/320/330/340/350/360) · 차분 중앙(손 자리/먼 칸)")
    paths = [(os.path.join(args.data, "recordings", r, "echo.npz"), f"{r} {lab}") for r, lab in TEST_RECORDINGS]
    paths.append((os.path.join(args.data, "sessions", args.session, "echo.npz"), f"세션 {args.session}"))
    for path, lab in paths:
        if not os.path.exists(path):
            print(f"  {lab}: 없음")
            continue
        m = measure_noise(path)
        result["noise"][lab] = m
        print(f"  {lab}: 세기 {100 * m['gain']:.2f}% · 위상 {m['phase_rad']:.4f} rad → {m['timing_bins']:.4f}칸 · 포락선 "
              + "/".join(f"{100 * v:.0f}" for v in m["rel"].values()) + f" · 차분 {100 * m['near']:.3f}/{100 * m['far']:.3f}")
    sim = Sim(args.data, args.session)
    han = SG.repo("hann", "saw")
    target = sim.real_targets()
    result["target"] = target
    print(f"\n2. 세션 {args.session}: 줄 {sim.R} · 쓴 프레임 {len(sim.use)} · L {sim.L_mm:.0f} mm · 실측 손 자리 |차분| 중앙 {target['m50']:.1f} dB · "
          f"95% {target['m95']:.1f} dB · 상관 {target['corr']:.3f} · 대비 {target['contrast']:.2f}")
    Hh0, Hs = sim.channel("along", 0.0)
    chans = {0.0: Hh0}
    if args.sigma is not None and args.k is not None:
        sigma, k = args.sigma, args.k
    else:
        print("\n3. 맞추기(hann 시뮬레이션, 떨림 σ × 움직이는 몫 k)")
        mean0 = Hh0.mean(axis=0)
        best = None
        for s in GRID_SIGMA:
            if s not in chans:
                chans[s] = sim.channel("along", s)[0]
            for kk in GRID_K:
                med, diff = sim.process(han, han, SG.BAND, 1 + SH * (kk * chans[s] + (1 - kk) * mean0) + SC * Hs)
                m50, m95 = sim.levels(med, diff[:, HZ])
                corr, con = sim.sc_metrics(diff[:, HZ])
                err = ((m50 - target["m50"]) / 2) ** 2 + ((m95 - target["m95"]) / 2) ** 2 + ((corr - target["corr"]) / 0.05) ** 2 \
                    + (np.log(con / target["contrast"]) / 0.15) ** 2
                print(f"  σ {s:4.2f} mm · k {kk:5.3f}: 중앙 {m50:6.1f} · 95% {m95:6.1f} dB · 상관 {corr:.3f} · 대비 {con:.2f} · 오차 {err:6.2f}", flush=True)
                if best is None or err < best[0]:
                    best = (err, s, kk)
        _, sigma, k = best
        print(f"  고른 값: 떨림 {sigma} mm · 움직이는 몫 {k}")
    result["sigma"], result["k"] = sigma, k
    for s in (sigma, 2 * sigma):
        if s not in chans:
            chans[s] = sim.channel("along", s)[0]
    print(f"\n4. 견주기(떨림 {sigma} mm, k {k})")
    examples = {}
    res = {"along": compare(sim, "along", Hh0, chans[sigma], Hs, k, list(runs()), examples)}
    print(f"  떨림 두 배({2 * sigma} mm)")
    alt = compare(sim, "along", Hh0, chans[2 * sigma], Hs, k, list(runs()))
    for cfg in ("hand", "across"):
        h0, hs = sim.channel(cfg, 0.0)
        h1 = sim.channel(cfg, sigma)[0] if sigma > 0 else h0
        res[cfg] = compare(sim, cfg, h0, h1, hs, k, list(SUBSET))
    result["compare"], result["compare_double_tremor"] = res, alt
    print("\n5. 대역")
    table, spots, dist = bandwidth_table(sim)
    result["bandwidth"], result["echo_spots_cm"] = table, spots
    with open(os.path.join(args.out, "hand_echo_sim.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    figures(args.out, sim, res, alt, examples, sigma, k, dist)
    print(f"\n그림·숫자: {args.out}/sonar-handsim.png, sonar-null-vs-hand.png, hand_echo_sim.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
