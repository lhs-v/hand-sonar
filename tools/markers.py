"""폰 표지음(21.5→23 kHz, 40 ms, 한 창)을 녹음에서 찾는다(손 소나 계획 3, 설계 §6.5·§8·§10).

정합 필터를 해석 신호로 한다. 표지음 파형 t와 그 힐베르트 짝으로 만든 복소 형판 a = t + i·H{t}와의 상관
c[k] = Σ x[k+j]·conj(a[j])의 크기 |c|가 상관 포락선이다. 실수 상관은 22 kHz 반송파를 따라 봉우리가 몇 칸씩 건너뛰지만
(폰 MarkerDetector는 ±7칸) 포락선은 매끈해서, 가장 큰 칸과 이웃 두 칸에 포물선을 맞추면 샘플보다 잘게 잡힌다(설계 §10: 0.1 ms 안).
비(ratio) = 포락선 봉우리 / 실수 상관 Re c의 RMS(봉우리 ±GUARD칸, 곧 표지음 길이 안은 뺌). 폰 MarkerDetector는 ±64칸만 빼지만,
그러면 표지음 자기 옆봉우리가 분모에 들어가 짧은 창에서 비가 크게 낮아진다(실측 워치 끝 표지음 ±0.5 s 창: ±64칸 16, ±1,920칸 92).
잡음만 있으면 약 4–5다.
찾은 sample은 x 안에서 표지음이 시작하는 샘플(소수)이다(meta.json micFrame과 같은 뜻).
"""
import site
import sys
from dataclasses import dataclass

sys.path.append(site.getusersitepackages())  # python -I에서도 사용자 설치 numpy·scipy를 찾게
import numpy as np  # noqa: E402
from scipy.signal import fftconvolve, hilbert  # noqa: E402

RATE = 48_000
START_HZ = 21_500.0
END_HZ = 23_000.0
SAMPLES = 1_920  # 40 ms
MIN_RATIO = 8.0  # 폰 MarkerDetector.MIN_RATIO와 같다
GUARD = 1_920  # 비의 분모에서 뺄 봉우리 둘레(칸) = 표지음 길이(상관이 0이 아닌 범위)
SEPARATION = 4_800  # 후보끼리 0.1 s 넘게 떨어져야 다른 소리로 본다(포락선 옆봉우리는 ±1,920칸 안)


def template():
    """진폭 1의 표지음(shared MarkerSpec.wave()와 같은 식): 위상 2π(f0·t + ½·k·t²), k = (끝 − 시작)/길이, 대칭 한 창."""
    i = np.arange(SAMPLES)
    t = i / RATE
    k = (END_HZ - START_HZ) / (SAMPLES / RATE)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * i / (SAMPLES - 1))
    return np.sin(2 * np.pi * (START_HZ * t + 0.5 * k * t * t)) * window


_KERNEL = np.conj(hilbert(template()))[::-1]  # fftconvolve로 c[k] = Σ x[k+j]·conj(a[j])를 내는 뒤집은 켤레 형판


@dataclass
class Hit:
    sample: float  # x 안에서 표지음이 시작하는 샘플(소수)
    ratio: float


def _parabola(env, k):
    """봉우리 k와 이웃 두 칸에 맞춘 포물선 꼭짓점의 칸 오프셋(−0.5..0.5)."""
    if 0 < k < len(env) - 1:
        a, b, d = env[k - 1], env[k], env[k + 1]
        den = a - 2 * b + d
        if den < 0:
            return float(np.clip(0.5 * (a - d) / den, -0.5, 0.5))
    return 0.0


def candidates(x, max_count=5, min_ratio=MIN_RATIO):
    """x에서 표지음 후보를 센 것부터 최대 max_count개. 서로 SEPARATION 넘게 떨어진 봉우리만, 비가 min_ratio 이상인 것만."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < SAMPLES + 2 * GUARD + 2:
        return []
    c = fftconvolve(x, _KERNEL, mode="valid")
    env = np.abs(c)
    best = int(np.argmax(env))
    keep = np.ones(len(c), dtype=bool)
    keep[max(0, best - GUARD): best + GUARD + 1] = False
    rms = float(np.sqrt(np.mean(c.real[keep] ** 2)))
    if rms <= 0:
        return []
    hits, work = [], env.copy()
    for _ in range(max_count):
        k = int(np.argmax(work))
        if work[k] < min_ratio * rms:
            break
        hits.append(Hit(k + _parabola(env, k), float(env[k] / rms)))
        work[max(0, k - SEPARATION): k + SEPARATION + 1] = 0.0
    return hits


def find(x, min_ratio=MIN_RATIO):
    """가장 센 표지음 하나(없으면 None)."""
    hits = candidates(x, 1, min_ratio)
    return hits[0] if hits else None
