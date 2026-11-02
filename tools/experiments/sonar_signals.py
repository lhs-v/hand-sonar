"""소나 송신 신호 후보를 저장소와 같은 처리(18–21 kHz 5차 버터워스 앞뒤 대역 통과 → 600표본 원형 상관)로 견준다(2026-11-02).
결과와 해석은 docs/sonar-signal.md.

    python -I tools/experiments/sonar_signals.py --out <저장소 밖 폴더> [--no-figures]

후보는 모두 600표본(12.5 ms, 48 kHz) 한 주기를 쉬지 않고 되풀이하고, 최대 |x| = 1로 맞춘다(스피커·DAC가 최대 진폭으로 묶인다는 가정).
- 지금 쓰는 넷: 톱니(창 없음·Tukey 60표본·한 창)와 삼각(창 없음). profiles.tx_phase와 같은 식이다.
- 이음새 손질: 위상 이은 톱니(18 → 21.04 kHz, 한 주기 244바퀴), 매끄러운 되돌림 톱니(뒤 48표본 동안 21 → 18 kHz를 반 코사인으로).
- 다른 분야에서 온 꼴: NLFM(스펙트럼 양 끝을 진폭 대신 머무는 시간으로 낮춤), HFM(1/f가 시간에 곧게, 도플러 불변),
  2차 위상 다중 사인(이산 처프·Zadoff–Chu와 같은 꼴, 마루율을 줄임), 무작위 위상 다중 사인(PN·부호 신호와 같은 스펙트럼, 같은 마루율 꼴).
- 넓은 대역(가정): 17–21.4 kHz 되돌림 톱니. 워치 스피커가 이 대역을 내는지는 재지 않았다.

잰 것(모두 직접 경로 봉우리 = 100인 진폭 비율, 또는 dB):
- 같은 최대 진폭에서의 에너지(일정 진폭 사인 = 100%), 들리는 대역(0.1–15 kHz) 누설(딸깍, 16비트 반올림 출력), 15–17.5 kHz 누설.
- 직접 경로 봉우리(주기 상관 = 파워 스펙트럼의 역DFT): 반높이 폭, 0이 되는 거리(첫 골), 3.6·7·10·15 cm의 꼬리, 한 주기 바닥(봉우리에서
  100–290칸 떨어진 곳의 중앙값). 칸 하나 = 3.57 mm(반사체까지 거리).
- 도플러: 10 cm 점 반사체가 0.1–0.5 m/s로 다가올 때 줄마다 포락선 봉우리의 거리 치우침과 크기(정지 대비).
- 수신 쪽 가중(송신은 그대로, 상관 템플릿만 바꿈): 수신 Tukey·수신 한 창, 한 창·삼각으로 보낸 녹음의 평탄화 수신(역필터, λ로 조절).
  손실은 같은 백색 잡음에서 정합 필터 대비 신호 대 잡음.
- Tukey 경사 길이(한쪽 0–120표본)별 딸깍·바닥·폭·에너지.
그림(--out): sonar-waveforms.png(후보 비교 여섯 칸), sonar-floor-tail.png(바닥·꼬리를 직접 경로 = 100으로 둔 선형 그림).
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools
import hs_io  # noqa: E402  사용자 설치 numpy·scipy 경로도 붙는다
import profiles  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, sosfreqz  # noqa: E402

RATE, N = profiles.RATE, profiles.N
T = N / RATE
C = 343.0
CM = C / RATE / 2 * 100  # 칸 하나 = 0.357 cm
F = np.fft.rfftfreq(N, 1 / RATE)  # 한 주기 배음(80 Hz 간격)
FINE = 64  # 위상 적분 세분
BAND = (18_000.0, 21_000.0)


# ---------- 신호 ----------
def tukey_win(m=60):
    i = np.arange(N)
    j = np.minimum(i, N - 1 - i)
    return np.where(j >= m, 1.0, 0.5 - 0.5 * np.cos(np.pi * j / max(m, 1)))


def hann_win():
    i = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))


def unit(x):
    return x / np.max(np.abs(x))


def repo(window, shape):
    """저장소 송신 처프(profiles.tx_phase, 반올림 없는 실수, 최대 1)."""
    x = np.sin(profiles.tx_phase(shape, np.arange(N)))
    if window == "tukey":
        x = x * tukey_win()
    elif window == "hann":
        x = x * hann_win()
    return unit(x)


def _phase(fn):
    tf = (np.arange(N * FINE) + 0.5) / (RATE * FINE)
    ph = np.concatenate([[0.0], np.cumsum(fn(tf)) / (RATE * FINE)])  # 바퀴
    return ph[::FINE][:N], ph[-1]


def fm(fn, integer_cycles=True):
    """순간 주파수 fn(t)의 일정 진폭 한 주기. integer_cycles면 전체에 작은 δ(±40 Hz 안)를 더해 한 주기 바퀴 수를 정수로(이음새 위상 연속)."""
    _, cyc = _phase(fn)
    delta = (np.round(cyc) - cyc) / T if integer_cycles else 0.0
    ph, _ = _phase(lambda t: fn(t) + delta)
    return unit(np.sin(2 * np.pi * ph))


def flyback(f0=18_000.0, f1=21_000.0, nfb=48, sweep=None):
    """앞 N−nfb 표본은 f0 → f1(sweep(u), 기본 직선), 뒤 nfb 표본은 f1 → f0을 반 코사인으로 되돌리는 순간 주파수."""
    tu = (N - nfb) / RATE
    sweep = sweep or (lambda u: u)

    def fn(t):
        u = np.clip(t / tu, 0, 1)
        v = np.clip((t - tu) / (nfb / RATE), 0, 1)
        return np.where(t < tu, f0 + (f1 - f0) * sweep(u), f1 - (f1 - f0) * (1 - np.cos(np.pi * v)) / 2)
    return fn


def nlfm_sweep(taper=0.3, floor=0.25):
    """주파수마다 머무는 시간 ∝ 원하는 파워 스펙트럼(양 끝 taper 비율만 반 코사인으로 floor까지)."""
    g = np.linspace(0, 1, 4001)
    w = np.ones_like(g)
    lo, hi = g < taper, g > 1 - taper
    w[lo] = floor + (1 - floor) * 0.5 * (1 - np.cos(np.pi * g[lo] / taper))
    w[hi] = floor + (1 - floor) * 0.5 * (1 - np.cos(np.pi * (1 - g[hi]) / taper))
    cdf = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2)])
    return lambda u: np.interp(u, cdf / cdf[-1], g)


def hfm_sweep(f0=18_000.0, f1=21_000.0):
    return lambda u: (1 / (1 / f0 - u * (1 / f0 - 1 / f1)) - f0) / (f1 - f0)


def multisine(lo=18_000.0, hi=21_000.0, phases="newman", iters=300, seed=0):
    """lo–hi 안 80 Hz 배음만 같은 크기로 둔 주기 신호(대역 밖 에너지 0). newman = 2차 위상 뒤 마루율 줄이기(RMS의 1.3배에서 자르고
    대역 안 배음 크기를 1로 되돌리기를 되풀이), random = 무작위 위상."""
    m = np.flatnonzero((F >= lo) & (F <= hi))
    K = len(m)
    ph = np.pi * np.arange(K) ** 2 / K if phases == "newman" else np.random.default_rng(seed).uniform(0, 2 * np.pi, K)
    X = np.zeros(len(F), complex)
    X[m] = np.exp(1j * ph)
    x = np.fft.irfft(X, n=N)
    if phases == "newman":
        for _ in range(iters):
            c = 1.3 * np.sqrt(np.mean(x ** 2))
            Y = np.fft.rfft(np.clip(x, -c, c))
            X = np.zeros(len(F), complex)
            X[m] = np.exp(1j * np.angle(Y[m]))
            x = np.fft.irfft(X, n=N)
    return unit(x)


def candidates():
    """{이름: (한 주기, 대역 통과 (lo, hi))}"""
    return {
        "톱니·창 없음": (repo("none", "saw"), BAND),
        "톱니·Tukey(60)": (repo("tukey", "saw"), BAND),
        "톱니·hann": (repo("hann", "saw"), BAND),
        "삼각·창 없음": (repo("none", "triangle"), BAND),
        "위상 이은 톱니": (fm(lambda t: 18_000 + 3_040 * t / T, integer_cycles=False), BAND),
        "매끄러운 되돌림 톱니": (fm(flyback()), BAND),
        "NLFM+되돌림": (fm(flyback(sweep=nlfm_sweep())), BAND),
        "HFM+되돌림": (fm(flyback(sweep=hfm_sweep())), BAND),
        "다중 사인(ZC 꼴)": (multisine(phases="newman"), BAND),
        "무작위 위상(PN 꼴)": (multisine(phases="random"), BAND),
        "넓은 대역 17–21.4 kHz": (fm(flyback(17_000, 21_400)), (17_000.0, 21_400.0)),
    }


def wide_saw(f0, f1, tukey):
    """대역을 바꾼 톱니(위상 연속), tukey면 Tukey(60)."""
    x = fm(lambda t: f0 + (f1 - f0) * t / T)
    return unit(x * tukey_win()) if tukey else x


# ---------- 잣대 ----------
def bandpass_mag2(lo, hi):
    """profiles.py와 같은 5차 버터워스를 앞뒤로 건 크기 |H|²(배음에서)."""
    sos = butter(5, [lo, hi], btype="bandpass", fs=RATE, output="sos")
    return np.abs(sosfreqz(sos, worN=F, fs=RATE)[1]) ** 2


def psf(x, template=None, band=BAND):
    """주기 상관(대역 통과 뒤)의 포락선, 직접 경로 = 100, 0 지연이 가운데(칸 300)."""
    p = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(x if template is None else template)) * bandpass_mag2(*band), n=N)
    env = np.abs(hilbert(np.roll(p, N // 2)))
    return 100 * env / env.max()


def half_width(env):
    k = int(np.argmax(env))
    h = env[k] / 2
    lo, hi = k, k
    while lo > 0 and env[lo] > h:
        lo -= 1
    while hi < len(env) - 1 and env[hi] > h:
        hi += 1
    return (hi - 1 + (env[hi - 1] - h) / (env[hi - 1] - env[hi])) - (lo + (h - env[lo]) / (env[lo + 1] - env[lo]))


def first_null(env):
    """봉우리에서 첫 골까지 칸 수."""
    k = int(np.argmax(env))
    j = k + 1
    while j < len(env) - 1 and not (env[j] <= env[j - 1] and env[j] <= env[j + 1]):
        j += 1
    return j - k


def tail(env, dist_cm):
    """직접 경로에서 dist_cm(반사체까지 거리) 떨어진 곳의 꼬리(직접 경로 = 100)."""
    return np.interp(np.asarray(dist_cm) / CM, np.arange(N // 2), env[N // 2:])


def floor(env):
    """한 주기 바닥: 봉우리에서 100–290칸 떨어진 곳(앞뒤)의 중앙값."""
    k = N // 2
    return float(np.median(np.r_[env[k + 100: k + 290], env[k - 290: k - 100]]))


def leakage(x, lo=100.0, hi=15_000.0, bits=16, peak=0.5):
    """한 주기 출력(최대 진폭 peak, bits 비트 반올림)에서 [lo, hi] 에너지 ÷ 전체(dB)."""
    y = x * peak
    if bits:
        q = 2 ** (bits - 1) - 1
        y = np.round(y * q) / q
    s = np.abs(np.fft.rfft(y)) ** 2
    return 10 * np.log10(s[(F >= lo) & (F <= hi)].sum() / s.sum())


def energy(x):
    """같은 최대 진폭에서 한 표본 평균 에너지(일정 진폭 사인 = 1)."""
    return float(np.mean(x ** 2) / 0.5)


def doppler(x, band, v, d0=0.10, rows=24, template=None):
    """10 cm 점 반사체가 v m/s로 다가올 때 (거리 치우침 mm, 봉우리 크기 정지 대비 비율). 송신은 한 주기를 대역 제한 보간한 연속 신호."""
    X = np.fft.rfft(x)
    sel = F >= 12_000
    t = np.arange(rows * N) / RATE
    Tm = np.fft.rfft(x if template is None else template)
    H = bandpass_mag2(*band)

    def peaks(vv):
        tau = 2 * (d0 - vv * t) / C
        sig = (2 / N) * np.real(np.exp(2j * np.pi * np.outer(t - tau, F[sel])) @ X[sel])
        p = np.fft.irfft(np.fft.rfft(sig.reshape(rows, N), axis=1) * np.conj(Tm) * H, n=N, axis=1)
        env = np.abs(hilbert(p, axis=1))
        kk = env.argmax(axis=1)
        r = np.arange(rows)
        a, b, c = env[r, kk - 1], env[r, kk], env[r, (kk + 1) % N]
        return kk + 0.5 * (a - c) / (a - 2 * b + c), b
    p0, m0 = peaks(0.0)
    pv, mv = peaks(v)
    true_shift = -2 * v * (np.arange(rows) + 0.5) * T / C * RATE
    bias = ((pv - p0[0]) - true_shift + N / 2) % N - N / 2
    return float(np.median(bias[2:]) * CM * 10), float(np.median(mv[2:]) / np.median(m0[2:]))


def mismatch(x, template, band=BAND):
    """같은 백색 잡음에서 정합 필터 대비 신호 대 잡음 비율(1 = 손실 없음)."""
    X = np.fft.rfft(x)
    H = bandpass_mag2(*band)
    R = np.conj(np.fft.rfft(template)) * H
    Ro = np.conj(X) * H
    return float((np.abs(np.sum(X * R)) ** 2 / np.sum(np.abs(R) ** 2)) / (np.abs(np.sum(X * Ro)) ** 2 / np.sum(np.abs(Ro) ** 2)))


def spectral_template(x, shape, band=BAND, edge=0.25):
    """수신 템플릿: 보낸 신호 스펙트럼에 대역 안 가중을 곱한다. tukey = 양 끝 edge 비율만 반 코사인, hann = 대역 전체 한 창."""
    u = np.clip((F - band[0]) / (band[1] - band[0]), 0, 1)
    if shape == "tukey":
        w = np.ones_like(u)
        a, b = u < edge, u > 1 - edge
        w[a] = 0.5 - 0.5 * np.cos(np.pi * u[a] / edge)
        w[b] = 0.5 - 0.5 * np.cos(np.pi * (1 - u[b]) / edge)
    else:
        w = 0.5 - 0.5 * np.cos(2 * np.pi * u)
    return np.fft.irfft(np.fft.rfft(x) * np.where((F >= band[0]) & (F <= band[1]), w, 0.0), n=N)


def eq_template(x, flat_ref, band=BAND, lam=0.02, edge=0.25):
    """평탄화 수신: x로 보낸 녹음의 상관이 flat_ref(창 없는 톱니) 스펙트럼에 Tukey 가중을 건 모양이 되게 |X|²로 나누는 템플릿(λ·최대로 조절)."""
    X = np.fft.rfft(x)
    want = np.abs(np.fft.rfft(spectral_template(flat_ref, "tukey", band, edge))) * np.abs(np.fft.rfft(flat_ref))
    P = np.abs(X) ** 2
    return np.fft.irfft(X * want / (P + lam * P.max()), n=N)


def rx_options():
    """{이름: (송신, 수신 템플릿)} 송신은 그대로 두고 PC에서 템플릿만 바꾸는 판."""
    saw, tuk, han, tri = repo("none", "saw"), repo("tukey", "saw"), repo("hann", "saw"), repo("none", "triangle")
    return {
        "창 없음 송신 · 정합": (saw, saw),
        "창 없음 송신 · 수신 Tukey 25%": (saw, spectral_template(saw, "tukey")),
        "창 없음 송신 · 수신 한 창": (saw, spectral_template(saw, "hann")),
        "Tukey 송신 · 정합": (tuk, tuk),
        "hann 송신 · 정합(옛 처리)": (han, han),
        "hann 송신 · 평탄화 수신 λ=0.02": (han, eq_template(han, saw, lam=0.02)),
        "삼각 송신 · 정합": (tri, tri),
        "삼각 송신 · 평탄화 수신 λ=0.05": (tri, eq_template(tri, saw, lam=0.05)),
    }


DIST = (3.6, 7.0, 10.0, 15.0)


def report():
    print("직접 경로 봉우리 = 100. 꼬리는 그 거리 칸에 남는 직접 경로, 바닥은 먼 칸에 고르게 깔리는 몫. 칸 하나 = 3.57 mm.")
    print(f"{'신호':22s} {'에너지':>5s} {'딸깍':>7s} {'15–17.5':>7s} {'폭':>5s} {'0이 되는':>7s} " + " ".join(f"{d:>4.1f}cm" for d in DIST)
          + f" {'바닥':>6s} | 0.3 m/s 봉우리·치우침")
    out = {}
    for name, (x, band) in candidates().items():
        env = psf(x, band=band)
        bias, size = doppler(x, band, 0.3)
        row = dict(energy=energy(x), leak=leakage(x), near=leakage(x, 15_000, 17_500), width=half_width(env), null_cm=first_null(env) * CM,
                   tails=[float(v) for v in tail(env, DIST)], floor=floor(env), dop_size=size, dop_mm=bias)
        out[name] = row
        print(f"{name:22s} {100 * row['energy']:4.0f}% {row['leak']:5.0f}dB {row['near']:5.0f}dB {row['width']:5.1f} {row['null_cm']:5.1f}cm "
              + " ".join(f"{v:6.1f}" for v in row["tails"]) + f" {row['floor']:6.2f} | {100 * size:3.0f}% · {bias:5.1f} mm")
    print("\n도플러(10 cm 반사체가 다가옴): 봉우리 크기(정지 대비)·거리 치우침")
    for name in ("톱니·창 없음", "삼각·창 없음", "HFM+되돌림", "무작위 위상(PN 꼴)", "넓은 대역 17–21.4 kHz"):
        x, band = candidates()[name]
        cells = []
        for v in (0.1, 0.3, 0.5):
            bias, size = doppler(x, band, v)
            cells.append(f"{v} m/s {100 * size:3.0f}% {bias:6.1f} mm")
        print(f"  {name:22s} " + " · ".join(cells))
    print("\n수신 쪽 가중(송신 그대로, 상관 템플릿만 바꿈). 신호 대 잡음은 같은 최대 진폭의 창 없음 톱니 정합 = 100%")
    rx = {}
    for name, (x, tm) in rx_options().items():
        env = psf(x, tm)
        snr = energy(x) * mismatch(x, tm)
        rx[name] = dict(width=half_width(env), null_cm=first_null(env) * CM, tails=[float(v) for v in tail(env, DIST)], floor=floor(env), snr=snr)
        print(f"  {name:28s} 폭 {rx[name]['width']:5.1f}칸 · 0이 되는 거리 {rx[name]['null_cm']:4.1f} cm · 7 cm {rx[name]['tails'][1]:5.1f} · "
              f"바닥 {rx[name]['floor']:5.2f} · 신호 대 잡음 {100 * snr:3.0f}%")
    print("\nTukey 경사 길이(한쪽 표본)")
    saw = repo("none", "saw")
    sweep = {}
    for m in (0, 15, 30, 60, 90, 120):
        x = unit(saw * tukey_win(m)) if m else saw
        env = psf(x)
        sweep[m] = dict(leak=leakage(x), floor=floor(env), width=half_width(env), energy=energy(x))
        print(f"  {m:3d}: 딸깍 {sweep[m]['leak']:6.1f} dB · 바닥 {sweep[m]['floor']:5.2f} · 폭 {sweep[m]['width']:4.1f}칸 · 에너지 {100 * sweep[m]['energy']:3.0f}%")
    return out, rx, sweep


# ---------- 그림 ----------
def korean_pyplot():
    """hs_io.pyplot(맑은 고딕)에 리눅스 한글 글꼴을 뒤에 더한다."""
    plt = hs_io.pyplot()
    plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["font.size"] = 9.5
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


COL = {"톱니·창 없음": "#1f6fb4", "톱니·Tukey(60)": "#2ca25f", "톱니·hann": "#d95f02", "삼각·창 없음": "#7b3294", "매끄러운 되돌림 톱니": "#e6ab02",
       "NLFM+되돌림": "#a6761d", "다중 사인(ZC 꼴)": "#666666", "무작위 위상(PN 꼴)": "#e7298a", "넓은 대역 17–21.4 kHz": "#1b9e77",
       "HFM+되돌림": "#66a61e"}


def figures(out_dir, res, rx, sweep):
    plt = korean_pyplot()
    cand = candidates()
    lag = np.arange(N) - N // 2
    fig, ax = plt.subplots(2, 3, figsize=(18.5, 10.4))
    fig.suptitle("소나 송신 신호 후보 — 저장소와 같은 대역 통과·원형 상관, 같은 최대 진폭(직접 경로 봉우리 = 100)", fontsize=13)
    a = ax[0, 0]
    t = np.arange(3 * N) / RATE * 1000
    for k in ("톱니·창 없음", "삼각·창 없음", "매끄러운 되돌림 톱니", "NLFM+되돌림", "넓은 대역 17–21.4 kHz"):
        f = np.gradient(np.unwrap(np.angle(hilbert(np.tile(cand[k][0], 3))))) * RATE / (2 * np.pi) / 1000
        m = (t >= 6.25) & (t <= 25)
        a.plot(t[m] - 6.25, np.convolve(f, np.ones(9) / 9, mode="same")[m], color=COL[k], lw=1.6, label=k)
    a.axhspan(21.5, 23, color="#c51b7d", alpha=0.12)
    a.text(0.3, 22.6, "폰 표지음 대역(겹치면 안 됨)", color="#c51b7d", fontsize=8.5)
    a.axhspan(15, 18, color="0.85", alpha=0.6)
    a.text(0.3, 15.25, "18 kHz 아래: 들릴 수 있는 쪽", fontsize=8.5, color="0.3")
    a.set(xlim=(0, 18.75), ylim=(15, 23.2), xlabel="시간 (ms, 1.5주기)", ylabel="순간 주파수 (kHz)", title="(a) 주파수 궤적(창은 진폭만 바꿈)")
    a.legend(loc="center right", fontsize=8)
    b = ax[0, 1]
    saw, han = repo("none", "saw"), repo("hann", "saw")
    for k in ("톱니·창 없음", "톱니·Tukey(60)", "톱니·hann", "삼각·창 없음", "넓은 대역 17–21.4 kHz"):
        env = psf(cand[k][0], band=cand[k][1])
        b.plot(lag * CM, env, color=COL[k], lw=1.6, label=f"{k}: {first_null(env) * CM:.1f} cm에서 0")
    env = psf(han, eq_template(han, saw))
    b.plot(lag * CM, env, "--", color=COL["톱니·hann"], lw=1.4, label=f"hann 송신+평탄화 수신: {first_null(env) * CM:.1f} cm에서 0")
    b.axvspan(0, 21, color="#fdd49e", alpha=0.35)
    b.set(xlim=(0, 25), ylim=(0, 102), xlabel="직접 경로에서 반사체까지 거리 (cm)", ylabel="직접 경로 = 100", title="(b) 꼬리: 직접 경로가 손 자리로 번지는 정도")
    b.legend(loc="upper right", fontsize=8)
    c = ax[0, 2]
    for k in ("삼각·창 없음", "매끄러운 되돌림 톱니", "톱니·창 없음", "다중 사인(ZC 꼴)", "톱니·Tukey(60)"):
        env = np.roll(psf(cand[k][0], band=cand[k][1]), -N // 2)
        c.plot(np.arange(N) * CM, env, color=COL[k], lw=1.1, label=f"{k}: 바닥 {res[k]['floor']:.2g}")
    c.axvspan(0, 21, color="#fdd49e", alpha=0.35)
    c.set(xlim=(0, 214), ylim=(0, 25), xlabel="직접 경로에서의 거리 (cm, 2.14 m 너머는 되감김)", ylabel="직접 경로 = 100(25에서 자름)",
          title="(c) 바닥: 모든 거리에 깔리는 뿌연 막")
    c.legend(loc="upper right", fontsize=8)
    d = ax[1, 0]
    ms = sorted(sweep)
    d.plot(ms, [sweep[m]["leak"] for m in ms], "o-", color="#d95f02", label="딸깍(들리는 대역 누설, dB)")
    d.axhline(res["삼각·창 없음"]["leak"], color="#7b3294", ls=":", lw=1)
    d.text(120, res["삼각·창 없음"]["leak"] + 1, "삼각의 딸깍", color="#7b3294", fontsize=8, ha="right")
    d.axvline(60, color="#2ca25f", lw=6, alpha=0.18)
    d.set(xlabel="Tukey 경사 길이(한쪽 표본, 60 = 앱 기본값) · 그 아래: 봉우리 폭 · 에너지", ylabel="딸깍 (dB)", ylim=(-95, -20),
          title="(d) Tukey 경사 길이: 딸깍·바닥 대 폭·에너지")
    d.set_xticks(ms, [f"{m}\n{sweep[m]['width']:.0f}칸\n{100 * sweep[m]['energy']:.0f}%" for m in ms], fontsize=8)
    d2 = d.twinx()
    d2.plot(ms, [sweep[m]["floor"] for m in ms], "s-", color="#7570b3", label="바닥(직접 경로 = 100, 오른쪽 축)")
    d2.set(ylim=(0, 2.5), ylabel="바닥(직접 경로 = 100)")
    h1, l1 = d.get_legend_handles_labels()
    h2, l2 = d2.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    e = ax[1, 1]
    vs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for k in ("톱니·창 없음", "삼각·창 없음", "무작위 위상(PN 꼴)", "넓은 대역 17–21.4 kHz", "HFM+되돌림"):
        x, band = cand[k]
        vals = [(0.0, 1.0) if v == 0 else doppler(x, band, v) for v in vs]
        e.plot(vs, [b_ for b_, _ in vals], "o-", color=COL[k], lw=1.6, label=k)
        e.plot(vs, [100 * (s - 1) for _, s in vals], "x--", color=COL[k], lw=1, alpha=0.8)
    e.axvspan(0, 0.1, color="0.9")
    e.set(xlabel="10 cm 반사체가 다가오는 속도 (m/s)", ylabel="실선: 거리 치우침 (mm) · 점선: 봉우리 크기 변화 (%)",
          title="(e) 도플러: 톱니는 치우치고, 삼각·PN 꼴은 봉우리가 줄어듦")
    e.legend(loc="lower left", fontsize=8)
    f_ = ax[1, 2]
    keys = list(res)
    cols = ["에너지(%)", "딸깍(dB)", "0이 되는\n거리(cm)", "7 cm\n꼬리", "10 cm\n꼬리", "바닥", "0.3 m/s\n봉우리(%)"]
    low_good = [False, True, True, True, True, True, False]
    V = np.array([[100 * res[k]["energy"], res[k]["leak"], res[k]["null_cm"], res[k]["tails"][1], res[k]["tails"][2], res[k]["floor"],
                   100 * res[k]["dop_size"]] for k in keys])
    Sx = np.zeros_like(V)
    for j in range(V.shape[1]):
        v = V[:, j] if low_good[j] else -V[:, j]
        Sx[:, j] = (v - v.min()) / (np.ptp(v) + 1e-9)
    f_.imshow(Sx, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            f_.text(j, i, f"{V[i, j]:.0f}" if abs(V[i, j]) >= 10 else f"{V[i, j]:.1f}", ha="center", va="center", fontsize=8)
    f_.set_xticks(range(len(cols)), cols, fontsize=8)
    f_.set_yticks(range(len(keys)), keys, fontsize=8.5)
    f_.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    f_.set_title("(f) 성적표(초록 = 좋음, 칸마다 상대 색)", fontsize=9.5)
    for axx in (a, b, c, d, e):
        axx.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "sonar-waveforms.png"), dpi=105)
    plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.2))
    for k in ("삼각·창 없음", "톱니·창 없음", "톱니·Tukey(60)"):
        env = np.roll(psf(cand[k][0]), -N // 2)
        a.plot(np.arange(N) * CM, env, color=COL[k], lw=1.3, label=k)
        a.axhline(res[k]["floor"], color=COL[k], ls="--", lw=1)
        a.text(212, res[k]["floor"] + 0.4, f"바닥 ≈ {res[k]['floor']:.2g}", color=COL[k], fontsize=10.5, ha="right", va="bottom")
    a.axvspan(0, 21, color="#fdd49e", alpha=0.5)
    for xx, lab in ((10.5, "손 자리"), (45, "몸·팔(30–60 cm)"), (150, "다른 사람·벽(1–2 m)")):
        a.text(xx, 28.3, lab, ha="center", fontsize=9.5, bbox=dict(fc="white", ec="none", alpha=0.8))
    a.set(xlim=(0, 214), ylim=(0, 30), xlabel="직접 경로에서의 거리 (cm)", ylabel="직접 경로 봉우리 = 100", title="(a) 바닥(위는 30에서 자름)")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=9.5)
    for k, dy in (("톱니·hann", 3), ("톱니·창 없음", 4), ("톱니·Tukey(60)", -6)):
        env = psf(cand[k][0])
        b.plot(lag[N // 2:] * CM, env[N // 2:], color=COL[k], lw=2, label=f"{k}: {first_null(env) * CM:.0f} cm에서 0")
        b.text(7.25, tail(env, 7.0) + dy, f"{tail(env, 7.0):.0f}", color=COL[k], fontsize=12, fontweight="bold")
    b.axvline(7.0, color="0.4", ls=":", lw=1)
    b.axvspan(0, 21, color="#fdd49e", alpha=0.35)
    b.set(xlim=(0, 22), ylim=(0, 102), xlabel="직접 경로에서 반사체까지 거리 (cm)", ylabel="직접 경로 봉우리 = 100", title="(b) 꼬리(점선 = 7 cm)")
    b.legend(loc="upper right", fontsize=9.5)
    for axx in (a, b):
        axx.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sonar-floor-tail.png"), dpi=110)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="소나 송신 신호 후보 비교(처리는 profiles.py와 같음)")
    parser.add_argument("--out", required=True, help="그림을 쓸 폴더(저장소 밖)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res, rx, sweep = report()
    if not args.no_figures:
        figures(args.out, res, rx, sweep)
        print(f"\n그림: {args.out}/sonar-waveforms.png, sonar-floor-tail.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
