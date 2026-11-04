"""대역 확인 — 공개 데이터가 정말 18–21 kHz인가, 그리고 그 대역 안 어느 쪽이 손 신호를 더 싣나.

    python -I tools/experiments/knobs/band_check.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

1. 프로파일 한 줄은 받은 줄 y와 송신 템플릿 x의 부호 있는 원형 상관이라 칸 방향 600점 DFT가 곧 Y(f)·conj(X(f))(80 Hz 간격)다.
   받은 소리와 템플릿에 함께 있는 대역만 남으므로 이 스펙트럼이 실제로 쓰인 대역이다. 칸 300으로 옮기기는 원형 밀기라 크기를 바꾸지 않고,
   3줄 중앙값은 시간 방향이다. 직접 경로(0 dB)가 줄을 거의 정하므로 모양 = 송신 처프 × 스피커→마이크 응답 × 템플릿.
2. 같은 원본에서 칸 방향 스펙트럼의 한 조각(가장자리 0.2 kHz 코사인)만 남겨 knobs.py의 값을 다시 잰다. 그 조각 대역만 쓰는 처프로 상관한
   것과 같다(대역폭이 줄면 거리 분해능도 준다: 1.5 kHz면 11.4 cm, 1 kHz면 17 cm라 직접 경로가 손 자리에 더 번진다 — 조각끼리만 견준다).
결과는 band.json.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import knobs  # noqa: E402
import hs_io  # noqa: E402
import numpy as np  # noqa: E402

N, FS = 600, 48_000
FREQ = np.arange(N // 2 + 1) * FS / N  # 80 Hz 간격
BANDS = {"18–21": (18_000, 21_000), "18–19.5": (18_000, 19_500), "19.5–21": (19_500, 21_000),
         "18–19": (18_000, 19_000), "19–20": (19_000, 20_000), "20–21": (20_000, 21_000)}
RAMP = 200.0


def mask(lo, hi):
    """[lo, hi] Hz는 1, 가장자리 RAMP Hz에 걸쳐 코사인으로 0."""
    m = np.clip(np.minimum(FREQ - (lo - RAMP / 2), (hi + RAMP / 2) - FREQ) / RAMP, 0.0, 1.0)
    return np.sin(0.5 * np.pi * m) ** 2


def spectrum_stats(p):
    """p = 칸 방향 평균 세기 스펙트럼(301,). 무게 중심, 누적 5·95% 주파수, 17.9–21.1 kHz 안 몫."""
    c = np.cumsum(p) / p.sum()
    inband = (FREQ >= 17_900) & (FREQ <= 21_100)
    return {"centroidHz": float((FREQ * p).sum() / p.sum()), "f05Hz": float(FREQ[np.searchsorted(c, 0.05)]),
            "f95Hz": float(FREQ[np.searchsorted(c, 0.95)]), "inbandShare": float(p[inband].sum() / p.sum())}


def main():
    meta = {r["session"]: r for r in json.load(open(common.out("knobs.json"), encoding="utf-8"))}
    ours = sorted(n for n in os.listdir(os.path.join(common.ARGS.data, "processed")) if n[:4] == "2026")
    pubs = sorted(n for n in os.listdir(os.path.join(knobs.PUB, "processed")) if n[:4] == "2025")
    out = []
    for data, names, who in ((common.ARGS.data, ours, "ours"), (knobs.PUB, pubs, "public")):
        for name in names:
            if name not in meta:
                continue
            try:
                raw, idx, moving, rest = knobs.labels(data, name)
            except Exception as ex:  # noqa: BLE001
                print(f"{name}: 못 잼({ex})")
                continue
            S = np.fft.rfft(raw, axis=0)
            p = (np.abs(S) ** 2).mean(axis=1)
            row = {"who": who, "session": name, **{k: meta[name].get(k) for k in ("source", "window", "shape", "amplitude", "volume")},
                   "spectrumDb": (10 * np.log10(p / p.max() + 1e-30)).tolist(), **spectrum_stats(p), "bands": {}}
            for key, (lo, hi) in BANDS.items():
                r = np.fft.irfft(S * mask(lo, hi)[:, None], n=N, axis=0)
                v = knobs.values(r, idx, moving, rest)
                direct = np.abs(r[300]).mean()
                v["hand_raw"] = float(20 * np.log10(np.abs(r[knobs.HAND]).mean() / direct))  # 정적 반사 세기(평균 |원본|)
                v["far_raw"] = float(20 * np.log10(np.abs(r[knobs.FAR]).mean() / direct))
                row["bands"][key] = v
            out.append(row)
            b = row["bands"]
            print(f"{who} {name} {row['shape']} | 중심 {row['centroidHz'] / 1000:.2f} kHz, 90% {row['f05Hz'] / 1000:.2f}–{row['f95Hz'] / 1000:.2f} kHz,"
                  f" 18–21 안 {row['inbandShare'] * 100:.1f}% | 오르내림 전체 {meta[name]['swing']:.1f} · 18–21 {b['18–21']['swing']:.1f}"
                  f" · 아래 반 {b['18–19.5']['swing']:.1f} · 위 반 {b['19.5–21']['swing']:.1f}"
                  f" · 셋 {b['18–19']['swing']:.1f}/{b['19–20']['swing']:.1f}/{b['20–21']['swing']:.1f}")
    json.dump({"freqHz": FREQ.tolist(), "sessions": out}, open(common.out("band.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("저장: band.json")


if __name__ == "__main__":
    main()
