"""잡음 바닥·손 신호 값들이 무엇에 따라 움직이나 — 우리 세션 전부(녹음 설정이 서로 다름)와 공개 10세션을 같은 방식으로 재서 설정별로 가른다.

    python -I tools/experiments/knobs/knobs.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

세션마다(직접 경로 = 0 dB, 짝지은 줄 범위):
- floor: 먼 칸(54–107 cm, 칸 450–599) |차분| 평균의 줄 중앙값
- move: 정답 손이 움직이는 프레임(움직이는 점 > 80 mm/s)의 손 자리(5–21 cm, 칸 315–359) |차분|(5줄 평활) 중앙값
- rest: 손이 멈춘 프레임(< 15 mm/s, 움직임에서 0.3 s 넘게 떨어짐)의 손 자리 |차분| 중앙값
- jitter: 직접 경로 세기(칸 300 |원본|)의 줄 사이 변화 ÷ 세기, 중앙값(%)
- gap: move − floor, swing: move − rest
줄 단위 정규화(dataset.row_normalized: 줄마다 600칸 |값| 합으로 나눔)를 켠 값(_rn)도 같은 세션에서 잰다. 결과는 knobs.json.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import dataset  # noqa: E402
import hs_io  # noqa: E402
import joints  # noqa: E402
import signal_check  # noqa: E402
import numpy as np  # noqa: E402

PUB = common.ARGS.pub
HAND, FAR = slice(315, 360), slice(450, 600)


def settings(data, name):
    """(녹음 설정 dict, 폰 context dict). 공개 데이터는 rec.json이 없어 비운다."""
    out = hs_io.processed_folder(data, name)
    rec = {}
    try:
        al = json.load(open(os.path.join(out, "align.json"), encoding="utf-8"))
        rj = json.load(open(os.path.join(data, "watch", "recordings", al["recording"], "rec.json"), encoding="utf-8"))
        rec = {"source": rj["audio"].get("source"), "window": rj["tx"].get("window"), "shape": rj["tx"].get("shape", "saw"),
               "amplitude": rj["tx"].get("amplitude"), "volume": (rj.get("volume") or {}).get("index")}
    except (OSError, KeyError, ValueError):
        pass
    try:
        meta = json.load(open(os.path.join(hs_io.session_folder(data, name), "meta.json"), encoding="utf-8"))
        ctx = meta.get("context") or {}
    except (OSError, ValueError):
        ctx = {}
    return rec, ctx


def labels(data, name):
    """(원본 (600, 줄) float64 — 짝지은 줄 범위, 프레임마다 차분 열 (F,), 움직이는 프레임 (F,), 멈춘 프레임 (F,))."""
    marks, row, direct = signal_check.mapping(data, name)
    good, _ = joints.keep_frames(marks.world, marks.score, marks.valid)
    use = np.flatnonzero(marks.valid & direct & good & (row >= 1))
    hand = hs_io.read_session(hs_io.session_folder(data, name)).hand
    y, _, _ = joints.targets(marks.world[use], hand)
    t = marks.frame_ns[use] / 1e9
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(y.reshape(len(use), 20, 3), axis=0), axis=2)[:, list(joints.MOVING)].mean(axis=1)
    speed = np.minimum(np.r_[0.0, np.where((dt > 0.02) & (dt < 0.1), step / np.maximum(dt, 1e-3), 0.0)] * 88.0, 1000.0)
    moving = speed > 80.0
    near = np.zeros(len(t), dtype=bool)
    for tm in t[moving]:
        near |= np.abs(t - tm) < 0.3
    rest = (speed < 15.0) & ~near
    out = hs_io.processed_folder(data, name)
    lo, hi = int(row[use].min()), int(row[use].max()) + 1
    raw = np.asarray(np.load(os.path.join(out, "profiles.npy"), mmap_mode="r")[:, lo:hi], dtype=np.float64)
    idx = np.clip(row[use] - lo - 1, 0, hi - lo - 2)  # 프레임의 줄 → 차분 열(줄 i−1 → i)
    return raw, idx, moving, rest


def values(r, idx, moving, rest):
    """원본 r (600, 줄)의 다섯 값(직접 경로 = 0 dB)과 간격·오르내림."""
    d = np.abs(r[:, 1:]) - np.abs(r[:, :-1])
    direct_mag = np.abs(r[300]).mean()
    kern = np.ones(5) / 5
    e_hand = np.convolve(np.abs(d[HAND]).mean(axis=0), kern, mode="same")
    e_far = np.abs(d[FAR]).mean(axis=0)
    db = lambda v: float(20 * np.log10(v / direct_mag))  # noqa: E731
    g = np.abs(r[300])
    res = {"floor": db(np.median(e_far)),
           "move": db(np.median(e_hand[idx[moving]])) if moving.any() else float("nan"),
           "rest": db(np.median(e_hand[idx[rest]])) if rest.any() else float("nan"),
           "jitter": float(np.median(np.abs(np.diff(g)) / g[1:]) * 100)}
    res["gap"] = res["move"] - res["floor"]
    res["swing"] = res["move"] - res["rest"]
    return res


def measure(data, name):
    raw, idx, moving, rest = labels(data, name)
    res = {}
    for tag, r in (("", raw), ("_rn", dataset.row_normalized(raw)[0])):
        res.update({k + tag: v for k, v in values(r, idx, moving, rest).items()})
    res.update({"movingShare": float(moving.mean()), "restShare": float(rest.mean()), "frames": int(len(idx))})
    return res


def main():
    rows = []
    ours = sorted(n for n in os.listdir(os.path.join(common.ARGS.data, "processed")) if n[:4] == "2026")
    pubs = sorted(n for n in os.listdir(os.path.join(PUB, "processed")) if n[:4] == "2025")
    for data, names, who in ((common.ARGS.data, ours, "ours"), (PUB, pubs, "public")):
        for name in names:
            try:
                m = measure(data, name)
            except Exception as ex:  # noqa: BLE001
                print(f"{name}: 못 잼({ex})")
                continue
            rec, ctx = settings(data, name)
            rows.append({"who": who, "session": name, **rec, **{f"ctx_{k}": v for k, v in ctx.items()}, **m})
            r = rows[-1]
            print(f"{who} {name} {rec.get('source', '')} {rec.get('window', '')} {rec.get('shape', '')} 세기 {rec.get('amplitude', '')} 볼륨 {rec.get('volume', '')}"
                  f" | 바닥 {r['floor']:.1f} 움직임 {r['move']:.1f} 멈춤 {r['rest']:.1f} 흔들림 {r['jitter']:.2f}% 간격 {r['gap']:.1f} 오르내림 {r['swing']:.1f}"
                  f" | 정규화 후 멈춤 {r['rest_rn']:.1f} 간격 {r['gap_rn']:.1f} 오르내림 {r['swing_rn']:.1f}")
    json.dump(rows, open(common.out("knobs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: knobs.json")


if __name__ == "__main__":
    main()
