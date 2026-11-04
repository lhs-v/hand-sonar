"""이론·시뮬레이션: 처프 대역(가운데 주파수·대역폭)을 바꾸면 값이 어떻게 움직이나, 그리고 18–21 kHz를 셋으로 나눈 값이 실측 경향을 내나.

    python -I tools/experiments/knobs/sim_band.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

sim_floor.py와 같은 가상 녹음(sim.json에 맞춘 세기·시각 흔들림, 잡음, 정적 반사, 손가락 세기)을 대역만 바꿔 같은 처리에 넣는다.
profiles의 대역 필터(SOS)와 송신 템플릿을 그 대역으로 바꾼다(삼각 꼭대기는 한 주기가 정수 바퀴가 되게 — 18 kHz 시작이면 21.04 kHz로 기기와 같다).
모형에 없는 것: 스피커·마이크 주파수 응답, 주변 소음 스펙트럼, 주파수에 따른 손가락 반사·몸 흡수. 이 결과는 처리와 위상만의 대역 효과다.
결과는 sim_band.json.
"""
import json
import os
import sys
from contextlib import contextmanager

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import sim_floor as S  # noqa: E402
import band_check  # noqa: E402
import profiles as P  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter  # noqa: E402

N, FS = P.N, P.RATE
CENTER = [15_000.0, 16_500.0, 18_000.0, 19_500.0, 20_500.0]  # 대역폭 3 kHz의 시작 주파수
WIDTH = [1_500.0, 3_000.0, 4_500.0, 6_000.0]  # 가운데 19.5 kHz의 대역폭
CONFIGS = (("ours_tri", "triangle"), ("ours_saw", "saw"), ("public", "saw"))
SEEDS = 4


def chirp(f0, bw, shape, amplitude=0.5):
    """대역 f0 … f0 + bw의 처프 한 주기(profiles.tx_template와 같은 식, 창 없음, 16비트 반올림)."""
    t = np.arange(N) / FS
    T = N / FS
    if shape == "triangle":
        top = 2 * round((f0 + bw / 2) * T) / T - f0  # 한 주기 위상이 정수 바퀴(기기의 TRIANGLE_TOP_HZ와 같은 규칙)
        half, k = T / 2, (2 * round((f0 + bw / 2) * T) / T - 2 * f0) / (T / 2)
        u = t - half
        ph = 2 * np.pi * np.where(t <= half, f0 * t + 0.5 * k * t * t, f0 * half + 0.5 * k * half * half + top * u - 0.5 * k * u * u)
    else:
        ph = 2 * np.pi * (f0 * t + 0.5 * (bw / T) * t * t)
    return np.round(np.sin(ph) * amplitude * 32767)


@contextmanager
def windowed(name):
    """송신·템플릿을 잠시 그 창(hann·tukey·none)으로 — 18–21 kHz 그대로."""
    old = P.tx_template
    P.tx_template = lambda window, amplitude, shape="saw": old(name, amplitude, shape)
    try:
        yield
    finally:
        P.tx_template = old


@contextmanager
def band(f0, bw):
    """profiles의 대역 필터와 송신 템플릿을 잠시 그 대역으로."""
    old = P.SOS, P.tx_template
    P.SOS = butter(5, [f0, f0 + bw], btype="bandpass", fs=FS, output="sos")
    P.tx_template = lambda window, amplitude, shape="saw": chirp(f0, bw, shape, amplitude)
    try:
        yield
    finally:
        P.SOS, P.tx_template = old


def main():
    for shape in ("saw", "triangle"):
        assert np.array_equal(chirp(18_000.0, 3_000.0, shape), P.tx_template("none", 0.5, shape)), shape
    sim = json.load(open(common.out("sim.json"), encoding="utf-8"))
    fit, taps_db = sim["fit"], sim["static"]["tapsDb"]
    taps_db["ours_tri"] = taps_db["ours_saw"]  # 같은 기기·같은 사람: 정적 반사는 톱니 날 실측에 맞춘 것

    def raws(key, shape):
        """정적 반사 무늬·흔들림·잡음·손가락 자리(지연 24.4·29.4·34.4·39.4샘플 = 8.7–14.1 cm)를 SEEDS번 새로 뽑은 (원본, 움직임, 멈춤)들
        — 반사체 한 벌·손가락 한 자리의 우연한 간섭이 대역 효과로 보이지 않게. 반사체는 칸 사이(소수 지연, taps_frac)에 둔다: 정수 칸에만 두면
        반송파와 칸 격자가 맞물려 대역 자리에 따라 오르내림이 ±1 dB 엇갈렸다(18–21이 가장 낮고 16.5–19.5·19.5–22.5가 높게 — 소수 지연이면 사라짐)."""
        f = fit[key]
        for s in range(SEEDS):
            yield S.simulate_raw(shape, f["eps"], f["tau"], f["noiseDb"], fit["fingerDb"][key], S.taps_frac(*taps_db[key], seed=1 + s), seed=s,
                                 finger_delay=24.37 + 5.0 * s)

    def mean(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    fmt = lambda r: f"바닥 {r['floor']:.1f} 멈춤 {r['rest']:.1f} 움직임 {r['move']:.1f} 오르내림 {r['swing']:.1f}"  # noqa: E731
    path = common.out("sim_band.json")
    only_window = "--only-window" in sys.argv  # 이미 있는 sim_band.json에 창 비교만 더한다
    res = json.load(open(path, encoding="utf-8")) if only_window else {"seeds": SEEDS, "center": {}, "width": {}, "thirds": {}}
    for key, shape in ([] if only_window else CONFIGS):
        res["center"][key] = []
        for f0 in CENTER:
            with band(f0, 3_000.0):
                res["center"][key].append({"f0Hz": f0, "bwHz": 3_000.0, **mean([S.values(*r) for r in raws(key, shape)])})
        res["width"][key] = []
        for bw in WIDTH:
            with band(19_500.0 - bw / 2, bw):
                res["width"][key].append({"f0Hz": 19_500.0 - bw / 2, "bwHz": bw, **mean([S.values(*r) for r in raws(key, shape)])})
        per = {b: [] for b in band_check.BANDS}
        for raw, moving, rest in raws(key, shape):
            spec = np.fft.rfft(raw, axis=0)
            for b, (lo, hi) in band_check.BANDS.items():
                per[b].append(S.values(np.fft.irfft(spec * band_check.mask(lo, hi)[:, None], n=N, axis=0), moving, rest))
        res["thirds"][key] = {b: mean(v) for b, v in per.items()}
        print(f"[{key} {shape}] 가운데 옮기기(3 kHz 폭): " + " | ".join(f"{r['f0Hz'] / 1000:.1f}–{(r['f0Hz'] + 3000) / 1000:.1f} {fmt(r)}" for r in res["center"][key]))
        print(f"[{key} {shape}] 폭(가운데 19.5): " + " | ".join(f"{r['bwHz'] / 1000:.1f} kHz {fmt(r)}" for r in res["width"][key]))
        print(f"[{key} {shape}] 18–21 안 나누기: " + " | ".join(f"{b} {fmt(r)}" for b, r in res["thirds"][key].items()))
    # 창: 우리 톱니 모형 그대로 송신·템플릿의 창만 바꾼다(10-27·10-28 한 창 녹음과 10-31 창 없음 녹음의 차이를 창 하나로 얼마나 설명하나)
    res["window"] = {}
    for name in ("none", "tukey", "hann"):
        with windowed(name):
            res["window"][name] = mean([S.values(*r) for r in raws("ours_saw", "saw")])
    print("[ours_saw saw] 창만 바꾸기: " + " | ".join(f"{k} {fmt(r)}" for k, r in res["window"].items()))
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: sim_band.json")


if __name__ == "__main__":
    main()
