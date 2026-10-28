"""세션의 신호 잣대(2026-10-28): 학습 없이, 녹음 신호가 손 움직임을 또렷이 담는지 거칠게 본다.

    python -I tools/signal_check.py [--data 데이터 폴더] [세션 …]

세션마다(processed/<세션>의 profiles·diff_profiles·rows·sync, 폰 landmarks.csv) 넷을 재어 화면에 내고 processed/<세션>/signal.json에 쓴다.
- 속도–세기 상관: 정답 관절(단위 손·손바닥 틀, dataset과 같은 거르기) 이웃 프레임 속도와 손 자리(칸 301–360) 차분 세기의 로그 상관.
  지연 0과 ±100 ms(±8줄) 안 최고를 낸다.
- 움직임 대비: 짝지은 줄 범위에서 손 자리 차분 세기의 95% 값 ÷ 중앙값. 손이 움직일 때 바닥 위로 얼마나 솟는가.
- 직접 경로 반높이 폭(칸): 원본 프로파일 칸 200–399 포락선 봉우리의 반높이 폭 중앙값. 처프 창 확인용이다(한 창 34–85, Tukey·창 없음 20–24).
- 직접 경로 흔들림(칸): 줄 사이 포락선 봉우리 위치(포물선 보간) 변화의 90% 값.
견줄 값(같은 계산): 공개 WatchHand 참가자 9 왼손 10세션은 상관 0.39–0.70, 대비 4.6–10.9, 폭 21칸, 흔들림 0.11–0.14칸이고 같은 처리로 학습하면
같은 세션 안에서 손 모양을 따라갔다. 우리 2026-10-28 세션(한 창, VOICE_RECOGNITION)은 상관 0.12–0.40, 대비 2.5–3.7이고 따라가지 못했다.
대응은 pairs.csv가 있으면 그 줄을, 없으면(맞춤 문턱에 걸린 세션) pairs.py와 같은 대응(rows.npz 줄 시각 + sync.json 지연)을 여기서만 만든다.
"""
import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·scipy 경로도 여기서 붙는다
import joints  # noqa: E402
import pairs  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

HAND = slice(301, 361)  # 손 자리 칸(손목에서 약 0–21 cm)
LAGS = range(-8, 9)  # 줄(12.5 ms) 단위 지연 ±100 ms
GAP_NS = 100_000_000  # 이웃 프레임 간격이 이보다 길면 속도를 재지 않는다
PEAK_BINS = slice(200, 400)  # 직접 경로(칸 300) 둘레
REFERENCE = "견줄 값: 공개 WatchHand(참가자 9 왼손) 상관 0.39–0.70 · 대비 4.6–10.9 · 폭 21칸 · 흔들림 0.11–0.14칸"


def speed_energy_corr(y, t_ns, rows, energy, lags=LAGS):
    """시각 순서 정답 y (M, 60)·시각 t_ns (M,)·짝 줄 rows (M,)와 줄마다 손 자리 차분 세기 energy (R−1,)에서 지연마다 로그 상관 [..]."""
    keep = np.diff(t_ns) < GAP_NS
    speed = np.linalg.norm(np.diff(y, axis=0), axis=1)[keep]
    r = np.asarray(rows)[1:][keep]
    out = []
    for lag in lags:
        e = energy[np.clip(r - 1 + lag, 0, len(energy) - 1)]  # 줄 r로 들어오는 변화는 차분 열 r − 1
        out.append(float(np.corrcoef(np.log(speed + 1e-6), np.log(e + 1e-9))[0, 1]))
    return out


def motion_contrast(energy):
    """손 자리 차분 세기의 95% 값 ÷ 중앙값."""
    return float(np.percentile(energy, 95) / np.median(energy))


def peak_track(raw):
    """원본 프로파일 블록 (칸, 줄)에서 줄마다 (포락선 봉우리 위치(포물선 보간, 칸), 반높이 폭(칸))."""
    env = np.abs(hilbert(np.asarray(raw, dtype=np.float64), axis=0))
    k = np.clip(env.argmax(axis=0), 1, env.shape[0] - 2)
    j = np.arange(env.shape[1])
    a, b, c = env[k - 1, j], env[k, j], env[k + 1, j]
    denom = a - 2 * b + c
    pos = k + 0.5 * (a - c) / np.where(denom != 0, denom, 1)
    widths = np.empty(env.shape[1])
    for col in j:
        e, p = env[:, col], k[col]
        half = e[p] / 2
        lo, hi = p, p
        while lo > 0 and e[lo] > half:
            lo -= 1
        while hi < len(e) - 1 and e[hi] > half:
            hi += 1
        widths[col] = hi - lo
    return pos, widths


def mapping(data, name):
    """(landmarks, 프레임마다 짝 줄 (M,) −1 = 없음, 직접 경로가 있는 줄인가 (M,))."""
    folder = hs_io.session_folder(data, name)
    out = hs_io.processed_folder(data, name)
    marks = hs_io.read_landmarks(folder)
    path = os.path.join(out, "pairs.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == len(marks.frame_ns):
            row = np.array([int(r["row"]) if r["row"] else -1 for r in rows])
            return marks, row, np.array([r["direct"] == "1" for r in rows])
    sync = hs_io.read_json(os.path.join(out, "sync.json"))
    rz = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)
    row, _ = pairs.nearest_rows(rz["ns"], marks.frame_ns.astype(np.float64) - hs_io.num(sync, "lagMs") * 1e6)
    ok = rz["ok"]
    return marks, row, (row >= 0) & ok[np.clip(row, 0, len(ok) - 1)]


def session_metrics(data, name, peak_rows=8000):
    marks, row, direct = mapping(data, name)
    hand = hs_io.read_session(hs_io.session_folder(data, name)).hand
    good, _ = joints.keep_frames(marks.world, marks.score, marks.valid)
    use = np.flatnonzero(marks.valid & direct & good & (row >= 1))
    if len(use) < 100:
        raise ValueError(f"쓸 프레임이 {len(use)}개뿐")
    y, _, _ = joints.targets(marks.world[use], hand)
    y = y.reshape(len(use), -1)
    out = hs_io.processed_folder(data, name)
    diff = np.load(os.path.join(out, "diff_profiles.npy"), mmap_mode="r", allow_pickle=False)
    raw = np.load(os.path.join(out, "profiles.npy"), mmap_mode="r", allow_pickle=False)
    energy = np.abs(np.asarray(diff[HAND], dtype=np.float64)).mean(axis=0)
    corr = speed_energy_corr(y, marks.frame_ns[use], row[use], energy)
    lo, hi = int(row[use].min()), int(row[use].max())
    pos, widths = peak_track(raw[PEAK_BINS, lo: min(hi, lo + peak_rows)])
    return {"session": name, "frames": int(len(use)), "corr0": corr[list(LAGS).index(0)], "corrMax": max(corr),
            "contrast": motion_contrast(energy[lo:hi]), "peakWidthBins": float(np.median(widths)),
            "peakJitter90Bins": float(np.percentile(np.abs(np.diff(pos)), 90))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="세션의 신호 잣대(학습 없이)")
    parser.add_argument("sessions", nargs="*", help="세션 이름(없으면 처리된 보관 세션 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    names = args.sessions or [n for n, why in hs_io.chosen(args.data) if not why
                              and os.path.exists(os.path.join(args.data, "processed", n, "sync.json"))]
    bad = 0
    for name in names:
        try:
            m = session_metrics(args.data, name)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 다른 세션은 잰다
            print(f"{name}: 재지 못함({type(e).__name__}: {e})")
            bad += 1
            continue
        hs_io.write_json(os.path.join(hs_io.processed_folder(args.data, name), "signal.json"), m)
        print(f"{name}: 속도–세기 상관 {m['corr0']:.2f}(±100 ms 최고 {m['corrMax']:.2f}) · 움직임 대비 {m['contrast']:.1f}"
              f" · 직접 경로 폭 {m['peakWidthBins']:.0f}칸 · 흔들림 {m['peakJitter90Bins']:.2f}칸 · 프레임 {m['frames']}")
    print(REFERENCE)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
