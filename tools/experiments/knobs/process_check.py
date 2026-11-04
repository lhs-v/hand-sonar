"""신호 처리로 더 나아지나, 손 근처에 뭔가 있나 — 같은 원본에서 처리만 바꿔 멈춤·움직임·오르내림을 다시 재고(공개·우리 같게),
움직일 때·멈췄을 때 변화 세기의 거리 분포를 낸다. 같은 처리를 맞춘 모형(sim.json)에도 걸어 '흔들림이 줄 공통이면 얼마나 줄어야 하나'와 견준다.

    python -I tools/experiments/knobs/process_check.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

처리(모두 3줄 중앙값을 건 부호 있는 원본에서 시작, 칸 300 = 직접 경로):
- now: 지금 처리 ||원본_t| − |원본_t−1||
- env: 포락선(칸 방향 힐베르트 크기) 차분 — 반송파 위상에 둔감(줄 시각 흔들림의 위상 몫이 빠진다)
- env/direct: 포락선을 줄마다 직접 경로 포락선으로 나눈 뒤 차분 — 줄 공통 세기 흔들림까지 빠진다
- cplx/direct: 복소 상관을 줄마다 직접 경로 복소값으로 나눈 뒤 |A_t − A_t−1| — 줄 공통 세기·위상 흔들림이 빠지고, 손가락의 위상 변화는 남는다
- cplx/fit: |A_t − g_t·A_t−1|, g_t = 줄 전체 최소제곱 공통 복소 이득 — 직접 경로 한 칸이 아니라 줄 전체의 공통 변화를 뺀다
값은 knobs.py와 같은 정의(그 처리의 직접 경로 평균 세기 = 0 dB). 결과는 process.json.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import knobs  # noqa: E402
import sim_floor as S  # noqa: E402
import hs_io  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

PROCS = ("now", "env", "env/direct", "cplx/direct", "cplx/fit")
NEAR = slice(290, 451)  # 거리 분포를 남길 칸(직접 경로 앞 3.6 cm … 54 cm)


def changes(raw):
    """처리마다 (변화 세기 D (600, R−1) ≥ 0, 직접 경로 평균 세기)."""
    out = {"now": (np.abs(np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])), np.abs(raw[300]).mean())}
    A = hilbert(raw, axis=0)
    E = np.abs(A)
    out["env"] = (np.abs(E[:, 1:] - E[:, :-1]), E[300].mean())
    En = E / E[300][None, :]
    out["env/direct"] = (np.abs(En[:, 1:] - En[:, :-1]), 1.0)
    An = A / A[300][None, :]
    out["cplx/direct"] = (np.abs(An[:, 1:] - An[:, :-1]), 1.0)
    g = (A[:, 1:] * np.conj(A[:, :-1])).sum(axis=0) / (np.abs(A[:, :-1]) ** 2).sum(axis=0)
    out["cplx/fit"] = (np.abs(A[:, 1:] - g[None, :] * A[:, :-1]), E[300].mean())
    return out


def values(D, direct, idx, moving, rest):
    e_hand = np.convolve(D[knobs.HAND].mean(axis=0), np.ones(5) / 5, mode="same")
    e_far = D[knobs.FAR].mean(axis=0)
    db = lambda v: float(20 * np.log10(v / direct))  # noqa: E731
    r = {"floor": db(np.median(e_far)), "rest": db(np.median(e_hand[idx[rest]])), "move": db(np.median(e_hand[idx[moving]]))}
    r["swing"] = r["move"] - r["rest"]
    return r


def main():
    meta = {r["session"]: r for r in json.load(open(common.out("knobs.json"), encoding="utf-8"))}
    ours = sorted(n for n in os.listdir(os.path.join(common.ARGS.data, "processed")) if n[:4] == "2026")
    pubs = sorted(n for n in os.listdir(os.path.join(knobs.PUB, "processed")) if n[:4] == "2025")
    res = {"sessions": [], "sim": {}}
    for data, names, who in ((common.ARGS.data, ours, "ours"), (knobs.PUB, pubs, "public")):
        for name in names:
            if name not in meta:
                continue
            raw, idx, moving, rest = knobs.labels(data, name)
            row = {"who": who, "session": name, **{k: meta[name].get(k) for k in ("source", "window", "shape", "volume")}, "procs": {}, "distance": {}}
            for p, (D, direct) in changes(raw).items():
                row["procs"][p] = values(D, direct, idx, moving, rest)
                if p in ("now", "cplx/fit"):
                    db = lambda v: (20 * np.log10(v / direct)).tolist()  # noqa: E731
                    row["distance"][p] = {"move": db(D[NEAR][:, idx[moving]].mean(axis=1)), "rest": db(D[NEAR][:, idx[rest]].mean(axis=1))}
            res["sessions"].append(row)
            print(f"{who} {name} {row['shape']} {row['window']} | " + " · ".join(
                f"{p} 멈춤 {v['rest']:.1f} 오르내림 {v['swing']:.1f}" for p, v in row["procs"].items()))
    # 맞춘 모형(줄 공통 흔들림)에 같은 처리 — 공통이면 cplx 처리에서 멈춤이 크게 내려가야 한다
    sim = json.load(open(common.out("sim.json"), encoding="utf-8"))
    fit, taps_db = sim["fit"], sim["static"]["tapsDb"]
    for key, shape, tk in (("public", "saw", "public"), ("ours_saw", "saw", "ours_saw"), ("ours_tri", "triangle", "ours_saw")):
        f = fit[key]
        raw, moving, rest = S.simulate_raw(shape, f["eps"], f["tau"], f["noiseDb"], fit["fingerDb"][key], S.taps_frac(*taps_db[tk], seed=1), finger_delay=30.37)
        idx = np.arange(raw.shape[1] - 1)
        res["sim"][key] = {p: values(D, direct, idx, moving, rest) for p, (D, direct) in changes(raw).items()}
        print(f"모의 {key}: " + " · ".join(f"{p} 멈춤 {v['rest']:.1f} 오르내림 {v['swing']:.1f}" for p, v in res["sim"][key].items()))
    res["distanceBins"] = list(range(NEAR.start, NEAR.stop))
    json.dump(res, open(common.out("process.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("저장: process.json")


if __name__ == "__main__":
    main()
