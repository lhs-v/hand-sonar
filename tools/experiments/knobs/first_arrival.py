"""공개 세션의 칸 300이 같은 도착인가 — 첫 도착에 맞춰 knobs.py의 값을 다시 잰다(거리 비교의 바로잡기, 2026-11-04).

    python -I tools/experiments/knobs/first_arrival.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

공개 10세션의 포락선(줄 중앙값)에는 약 23칸 간격의 봉우리가 있다. 처리가 가장 센 |원본|을 칸 300에 두므로, 둘째 봉우리가 첫째보다 센 세션은
칸 300이 둘째 도착이다(첫 도착이 칸 277 부근에 남는다). 그런 세션은 손 자리 칸 315–359가 첫 도착 기준으로 약 8 cm 더 멀다.
세션마다 첫 도착 칸 P를 찾아(칸 268–285의 봉우리가 −8 dB 넘으면 그 칸, 아니면 칸 300 둘레 최대) 다음을 첫 도착 기준으로 다시 잰다.
- 직접 경로 세기 = 칸 P의 평균 |원본|
- 손 자리 = P+15 … P+59(5–21 cm), 먼 칸 = P+150 … (54–107 cm, 줄 끝까지)
- 멈춤·움직임·오르내림·바닥, 평균 |원본|(손 자리), 움직일 때·멈췄을 때 거리별 변화 세기(첫 도착 기준 −10 … 54 cm)
결과는 first_arrival.json.
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
from scipy.signal import hilbert  # noqa: E402

EARLY = (268, 286)  # 칸 300이 둘째 도착이면 첫 도착이 있는 칸(칸 300 주봉의 어깨가 들지 않게 칸 285까지)
EARLY_DB = -8.0
SPAN = range(-28, 151)  # 첫 도착 기준 거리 칸(남길 거리 분포)


def first_arrival(raw):
    """(첫 도착 칸, 칸 EARLY 봉우리 세기 dB, 포락선 줄 중앙값 dB (600,)). 포락선은 줄 8개마다."""
    env = np.abs(hilbert(raw[:, ::8], axis=0))
    m = np.median(env, axis=1)
    level = 20 * np.log10(m / m[296:305].max())
    k = EARLY[0] + int(level[EARLY[0]:EARLY[1]].argmax())
    p = k if level[k] > EARLY_DB else 296 + int(level[296:305].argmax())
    return p, float(level[k]), level


def values(raw, p, idx, moving, rest):
    d = np.abs(np.abs(raw[:, 1:]) - np.abs(raw[:, :-1]))
    direct = np.abs(raw[p - 2: p + 3]).max(axis=0).mean()  # 첫 도착 봉우리 세기(반송파에 흔들리지 않게 ±2칸 최대)
    hand, far = slice(p + 15, p + 60), slice(p + 150, 600)
    e_hand = np.convolve(d[hand].mean(axis=0), np.ones(5) / 5, mode="same")
    e_far = d[far].mean(axis=0)
    db = lambda v: float(20 * np.log10(v / direct))  # noqa: E731
    r = {"floor": db(np.median(e_far)), "rest": db(np.median(e_hand[idx[rest]])), "move": db(np.median(e_hand[idx[moving]])),
         "handRaw": db(np.abs(raw[hand]).mean()), "farRaw": db(np.abs(raw[far]).mean())}
    r["swing"] = r["move"] - r["rest"]
    bins = [p + k for k in SPAN]
    r["distanceMove"] = (20 * np.log10(d[bins][:, idx[moving]].mean(axis=1) / direct)).tolist()
    r["distanceRest"] = (20 * np.log10(d[bins][:, idx[rest]].mean(axis=1) / direct)).tolist()
    r["static"] = (20 * np.log10(np.abs(hilbert(raw[:, ::8], axis=0)).mean(axis=1)[bins] / np.abs(hilbert(raw[:, ::8], axis=0)).mean(axis=1)[p - 2: p + 3].max())).tolist()
    return r


def main():
    meta = {r["session"]: r for r in json.load(open(common.out("knobs.json"), encoding="utf-8"))}
    ours = sorted(n for n in os.listdir(os.path.join(common.ARGS.data, "processed")) if n[:4] == "2026")
    pubs = sorted(n for n in os.listdir(os.path.join(knobs.PUB, "processed")) if n[:4] == "2025")
    res = {"spanBins": list(SPAN), "sessions": []}
    for data, names, who in ((knobs.PUB, pubs, "public"), (common.ARGS.data, ours, "ours")):
        for name in names:
            if name not in meta or (who == "ours" and meta[name].get("window") != "none"):
                continue
            raw, idx, moving, rest = knobs.labels(data, name)
            p, early, level = first_arrival(raw)
            r = {"who": who, "session": name, "shape": meta[name].get("shape"), "firstBin": p, "earlyDb": early, "level": level[250:400].tolist(), **values(raw, p, idx, moving, rest)}
            res["sessions"].append(r)
            m = meta[name]
            print(f"{who} {name} {str(r['shape']):8s} 첫 도착 칸 {p}(앞 봉우리 {early:.1f} dB) | 첫 도착 기준 멈춤 {r['rest']:.1f} 움직임 {r['move']:.1f} 오르내림 {r['swing']:.1f} 손 자리 |원본| {r['handRaw']:.1f}"
                  f" | 칸 300 기준 멈춤 {m['rest']:.1f} 움직임 {m['move']:.1f} 오르내림 {m['swing']:.1f}")
    json.dump(res, open(common.out("first_arrival.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("저장: first_arrival.json")


if __name__ == "__main__":
    main()
