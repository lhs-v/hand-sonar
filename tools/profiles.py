"""세션 구간의 워치 녹음으로 에코 프로파일을 만든다(손 소나 계획 3, 설계 §3·§8).

    python -I tools/profiles.py [--data 데이터 폴더] [세션 이름 …]   (이름이 없으면 쓸 수 있는 세션 모두, align.json이 먼저 있어야 한다)

공개 데이터에서 확인한 처리(docs/2026-10-27-watchhand-dataset-check)를 따르고, 실험 코드 pose_probe.py에서 나왔다.
1. 세션 구간(meta 시작 − 1 s ~ 끝 + 1 s)을 align.json 맞춤으로 워치 프레임으로 바꾸고, 녹음 처음부터 센 600샘플 줄로 자른다.
   필터 가장자리를 버리려고 앞뒤 0.1 s를 더 읽는다.
2. 18–21 kHz 5차 버터워스 대역 필터(sosfiltfilt, 위상 0).
3. 줄마다 송신 처프(rec.json tx의 창·진폭, 16비트 반올림)와 부호 있는 원형 상관 600칸: c[k] = Σ 줄[(j+k) mod 600]·tx[j].
4. 직접 경로 따라가기(설계 §8): 줄마다 상관 포락선(칸 방향 힐베르트 크기)의 최대 칸을 본다. 지금 칸에서 3칸 넘게 벗어난 줄이
   한두 줄이면 무시하고, 서로 3칸 안인 줄이 3줄 연속 벗어나면 그 세 줄부터 새 칸으로 옮긴다(워치8 직접 경로는 약 90샘플씩 튄다).
   포락선 최대가 그 줄 칸 중앙값의 20배 아래인 줄은 직접 경로가 없는 줄(재생 끊김 등)로 보고 판단에 넣지 않는다. 그런 줄과,
   무시한 한두 줄 튐(직접 경로가 따라간 칸에서 3칸 넘게 벗어남)은 ok=False로 적어 뒤 도구가 쓰지 않게 한다.
   실측(워치8 녹음 2개): 포락선 최대 칸은 ±3칸 안에서만 흔들렸고 포락선 최대/중앙값은 중앙 194배였다. |원본| 최대 칸은 반송파 탓에
   4–7칸씩 건너뛰어 따라가기에 쓰지 않는다.
5. |원본| 최대가 칸 300에 오도록(공개 데이터는 98%가 그렇다) 포락선 칸에 둘의 차이(세션 전체 최빈값, ±3칸 안)를 더한 칸을 칸 300으로
   옮긴다.
6. 시간 방향 3줄 중앙값(끝 줄은 되풀이), 차분 |원본[t]| − |원본[t−1]|.
저장(processed/<세션>/): profiles.npy (600, R), diff_profiles.npy (600, R−1)는 공개 데이터와 같은 배치·자료형(float64)이고,
rows.npz에 줄마다 첫 워치 프레임(frame), 줄 가운데의 폰 elapsedNs(ns), 옮기기 전 직접 경로 칸(direct), 직접 경로가 제자리에 있음(ok)을
둔다.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·scipy 경로도 여기서 붙는다
import align  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

RATE = hs_io.RATE
N = 600  # 줄 하나 = 처프 하나(12.5 ms)
ROW0 = 300  # 직접 경로를 둘 칸(공개 데이터와 같음)
SOS = butter(5, [18_000, 21_000], btype="bandpass", fs=RATE, output="sos")
MARGIN_SEC = 1.0  # 세션 meta 시작·끝 앞뒤로 더 자르는 시간(첫 카메라 프레임이 시작 버튼보다 0.1 s 이르다)
PAD_SEC = 0.1  # 필터 가장자리용으로 더 읽는 시간
TOL = 3  # 이 칸 수 안이면 같은 직접 경로로 본다(실측 흔들림 ±3칸)
CONFIRM = 3  # 이만큼 줄이 연속으로 옮겨 가야 따라간다(설계 §8)
INIT_ROWS = 80  # 처음 칸은 직접 경로가 있는 첫 1 s 줄의 최빈값
DIRECT_MIN = 20.0  # 포락선 최대 / 칸 중앙값이 이보다 작으면 직접 경로가 없는 줄(실측 중앙 194)
NEAR = slice(ROW0 + 1, ROW0 + 61)  # 직접 경로 바로 뒤 60칸 = 0–21.4 cm(손 거리, 모델 입력 폭)
FAR = slice(ROW0 + 120, ROW0 + 180)  # 같은 폭의 손 밖(43–64 cm), 견줌용


def tx_template(window, amplitude):
    """워치가 낸 처프 한 주기(shared Sweep.pcm16과 같은 식: 18→21 kHz 600샘플, 한 창이면 대칭 한 창, 16비트 반올림)."""
    i = np.arange(N)
    t = i / RATE
    w = np.sin(2 * np.pi * (18_000 * t + 0.5 * (3_000 / (N / RATE)) * t * t))
    if window == "hann":
        w = w * (0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1)))
    return np.round(w * min(max(amplitude, 0.0), 1.0) * 32767)


def _dist(a, b):
    d = (a - b) % N
    return min(d, N - d)


def track(peaks, ok, tol=TOL, confirm=CONFIRM, init_rows=INIT_ROWS):
    """줄마다 직접 경로 칸. peaks는 줄마다 포락선 최대 칸, ok가 아닌 줄은 판단에 넣지 않고 지금 칸을 준다."""
    out = np.empty(len(peaks), dtype=np.int64)
    first = np.asarray(peaks)[np.asarray(ok)][:init_rows]
    cur = int(np.bincount(first, minlength=N).argmax()) if len(first) else 0
    pending = []  # 지금 칸에서 벗어나 서로 tol 안인 연속 줄
    for r, a in enumerate(peaks):
        out[r] = cur
        if not ok[r]:
            continue
        a = int(a)
        if _dist(a, cur) <= tol:
            pending = []  # 돌아왔으면 앞의 벗어난 줄은 튐이었다
            continue
        if pending and _dist(a, int(peaks[pending[-1]])) > tol:
            pending = []
        pending.append(r)
        if len(pending) >= confirm:
            cur = a
            out[pending] = cur
            pending = []
    return out


@dataclass
class Profiles:
    raw: np.ndarray  # (600, R) 칸 300 = 직접 경로, 3줄 중앙값
    diff: np.ndarray  # (600, R − 1) 열 i = |raw[:, i+1]| − |raw[:, i]|
    direct: np.ndarray  # (R,) 칸 300으로 옮긴 원래 칸
    ok: np.ndarray  # (R,) 직접 경로가 있고 따라간 칸에서 튀지 않은 줄
    offset: int  # 포락선 최대 칸 → |원본| 최대 칸


def compute(x, tx, front=0, back=0):
    """x = 앞 front샘플 + 600샘플 줄 R개 + 뒤 back샘플. 필터를 x 전체에 건 뒤 앞뒤를 버리고 줄로 나눈다."""
    y = sosfiltfilt(SOS, np.asarray(x, dtype=np.float64))
    y = y[front: len(y) - back]
    rows = y[: len(y) // N * N].reshape(-1, N)
    prof = np.fft.irfft(np.fft.rfft(rows, axis=1) * np.conj(np.fft.rfft(tx)), n=N, axis=1)
    env = np.abs(hilbert(prof, axis=1))
    peaks = env.argmax(axis=1)
    ok = env.max(axis=1) >= DIRECT_MIN * np.median(env, axis=1)
    tracked = track(peaks, ok)
    d = (np.abs(prof).argmax(axis=1) - tracked + N // 2) % N - N // 2
    near = ok & (np.abs(d) <= TOL)
    offset = int(np.bincount(d[near] + TOL, minlength=2 * TOL + 1).argmax()) - TOL if near.any() else 0
    direct = (tracked + offset) % N
    steady = np.abs((peaks - tracked + N // 2) % N - N // 2) <= TOL  # 무시한 한두 줄 튐은 제자리가 아니다
    idx = (np.arange(N)[None, :] + direct[:, None] - ROW0) % N  # aligned[r, 300] = prof[r, direct[r]]
    aligned = np.take_along_axis(prof, idx, axis=1)
    prev = np.vstack([aligned[:1], aligned[:-1]])
    nxt = np.vstack([aligned[1:], aligned[-1:]])
    med = np.maximum(np.minimum(prev, aligned), np.minimum(np.maximum(prev, aligned), nxt))  # 세 값의 중앙값
    diff = np.abs(med[1:]) - np.abs(med[:-1])
    return Profiles(np.ascontiguousarray(med.T), np.ascontiguousarray(diff.T), direct, ok & steady, offset)


def run(data, name):
    """세션 하나: (보고 줄, 만들었는가)."""
    info, a = align.load(data, name)
    if a is None:
        return [f"{name}: 건너뜀(맞춤 없음: {'; '.join(info['notes'])})"], False
    meta = hs_io.read_json(os.path.join(hs_io.session_folder(data, name), "meta.json"))
    rec = a.recording
    t0 = hs_io.nanos(meta["start"], "elapsedNs") - MARGIN_SEC * 1e9
    t1 = hs_io.nanos(meta["end"], "elapsedNs") + MARGIN_SEC * 1e9
    f0, f1 = (float(v) for v in a.frames([t0, t1]))
    r0, r1 = max(0, math.ceil(f0 / N)), min(rec.frames // N, math.floor(f1 / N))
    if r1 - r0 < 2:
        return [f"{name}: 건너뜀(세션 구간이 워치 녹음 {rec.name} 밖)"], False
    pad = int(PAD_SEC * RATE)
    a0, b0 = max(0, r0 * N - pad), min(rec.frames, r1 * N + pad)
    tx = rec.rec.get("tx") or {}
    p = compute(rec.read(a0, b0), tx_template(tx.get("window", "none"), hs_io.num(tx, "amplitude", 0.5)), r0 * N - a0, b0 - r1 * N)
    frames = (r0 + np.arange(p.raw.shape[1], dtype=np.int64)) * N
    out = hs_io.processed_folder(data, name)
    np.save(os.path.join(out, "profiles.npy"), p.raw)
    np.save(os.path.join(out, "diff_profiles.npy"), p.diff)
    np.savez(os.path.join(out, "rows.npz"), frame=frames, ns=a.phone_ns(frames + N / 2), direct=p.direct.astype(np.int16),
             ok=p.ok, recording=np.array(rec.name))
    return report(name, info, frames, p), True


def report(name, info, frames, p):
    rows = p.raw.shape[1]
    moves = np.flatnonzero(np.diff(p.direct) != 0)
    lines = [f"{name}: 줄 {rows}개({rows * N / RATE:.1f} s), 워치 {info['recording']} {frames[0] / RATE:.2f}–{(frames[-1] + N) / RATE:.2f} s,"
             f" 맞춤 {info['label']}"]
    shown = ", ".join(f"{frames[k + 1] / RATE:.2f} s {p.direct[k]}→{p.direct[k + 1]}" for k in moves[:5])
    lines.append(f"  직접 경로: 처음 칸 {p.direct[0]}(포락선 칸 {p.offset:+d}), 옮김 {len(moves)}번" + (f" ({shown}{' …' if len(moves) > 5 else ''})" if shown else ""))
    lines.append(f"  직접 경로가 없거나 튄 줄 {int((~p.ok).sum())}개, |원본| 최대가 칸 300인 줄 {np.mean(np.abs(p.raw).argmax(axis=0) == ROW0) * 100:.1f}%")
    near = np.abs(p.diff[NEAR]).sum(axis=0)
    far = np.abs(p.diff[FAR]).sum(axis=0)
    lines.append(f"  손 거리(칸 301–360) 차분 세기 / 손 밖(칸 420–479): 중앙 {np.median(near) / max(np.median(far), 1e-12):.1f}배")
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="세션마다 워치 에코 프로파일을 만든다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    missing = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why or not os.path.exists(os.path.join(args.data, "processed", name, "align.json")):
            print(f"{name}: 건너뜀({why or 'align.json 없음 — align.py를 먼저'})")
            continue
        lines, made = run(args.data, name)
        print("\n".join(lines))
        missing += not made
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
