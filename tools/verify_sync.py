"""카메라 관절 속도와 차분 프로파일 세기의 상호상관으로 카메라–소리 지연을 잰다(손 소나 계획 3, 설계 §8·§8.1·§9-3).

    python -I tools/verify_sync.py [--data 데이터 폴더] [세션 이름 …]   (profiles.py를 먼저 돌린다)

- 소리: 차분 프로파일의 가까운 60칸(칸 301–360, 직접 경로 바로 뒤 0–21.4 cm)의 |값| 합. 차분 열 i는 원본 줄 i와 i+1 사이의
  변화라 시각은 두 줄 가운데다.
- 카메라: landmarks.csv 세계 좌표(m) 21점이 이웃 프레임 사이에 움직인 거리의 평균 ÷ 시간(m/s). 두 프레임이 모두 valid이고
  0.1 s 안에 있을 때만 쓰고, 시각은 두 프레임 가운데다. 손 중심 기준 좌표라 손 전체가 옮겨 가는 것보다 손가락·손목 움직임을 본다.
- 카메라 값을 소리 줄 시각에 선형 보간으로 놓고(카메라 점 사이가 0.1 s 넘게 비면 뺌), 둘 다 로그를 취한 뒤 1 s 이동 평균을 빼고
  표준화해 ±1 s 안의 지연마다 상관을 구한다. 로그는 워치를 건드린 것 같은 큰 튐(실데이터 134 s에 표준점수 19)이 상관을 끌고 가지
  않게 하고, 이동 평균 빼기는 구간마다 다른 느린 세기를 지워 낱낱의 움직임 시각만 남긴다(실데이터 20261027-235412: 그대로 쓰면
  +704 ms에 거짓 봉우리 0.11, 이렇게 하면 +7.8 ms에 봉우리 0.18 하나). 봉우리와 이웃 두 점에 포물선을 맞춰 줄(12.5 ms)보다 잘게 잡는다.
- 지연이 +면 카메라 시각이 소리보다 늦게 찍힌 것이다. pairs.py는 frameNs에서 이 지연을 빼서 소리 시각에 놓는다(설계 §8.1).
- 설계 §9-3: 표지음 맞춤과 움직임 맞춤의 차이가 1프레임(33 ms) 안이어야 한다. 상관 봉우리 값은 소리가 손 움직임을 따라가는지의
  거친 표시다(포즈 신호 확인). 지연이 1프레임 밖이거나 봉우리가 MIN_PEAK 아래면 그 세션은 맞춤을 확인하지 못한 것이다:
  sync.json에 usable=false로 적고 끝 코드 1로 끝나며, pairs.py는 그 세션을 짝짓지 않는다(틀린 지연으로 짝짓느니 짝짓지 않는다).
- sync.json에는 쓴 입력(align.json·rows.npz·landmarks.csv)의 SHA-256을 적는다. pairs.py는 지금 파일과 같을 때만 쓴다. rows.npz가
  지금 align.json에서 나오지 않았으면(rows.npz alignSha가 다름 — align.py만 다시 돌림) 재지 않는다. 재지 못하고 끝나면(파일 없음,
  쌍 부족, 오류) 옛 sync.json을 지우지 않고 sync.stale.json으로 바꾼다.
결과: processed/<세션>/sync.json, sync.png(위: 첫 동작부터 20 s 동안 두 값을 겹친 시계열과 동작 이름, 아래: 지연별 상관).
그림에는 영상 프레임을 넣지 않는다.
"""
import argparse
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

NEAR = slice(301, 361)  # 직접 경로(칸 300) 바로 뒤 60칸 = 0–21.4 cm
MAX_LAG_SEC = 1.0
MAX_GAP_SEC = 0.1  # 카메라 점 사이가 이보다 비면 그 사이는 쓰지 않는다
FRAME_MS = 33.0  # 설계 §9-3의 1프레임
MIN_PEAK = 0.1  # 상관 봉우리 문턱. 실측 맞는 세션 235412 0.178, 3.66 s 틀린 맞춤(234828 되살림 흉내)의 거짓 봉우리 0.07(±100 ms 안 최대 0.024)
MIN_ROWS = 100  # 상관에 쓸 소리 줄(1.25 s)이 이보다 적으면 재지 않는다
DETREND_SEC = 1.0  # 로그 세기에서 빼는 이동 평균 폭(동작 하나 2 s보다 짧게)
SMOOTH_SEC = 0.1  # 그림의 시계열만 이만큼 이동 평균한다(상관에는 쓰지 않음)
EXCERPT_SEC = 20.0  # 그림 위 칸은 첫 동작 1 s 앞부터 이만큼만 그린다(세션 전체는 너무 빽빽하다)


def near_energy(diff):
    """차분 프로파일 (600, R−1)의 가까운 60칸 |값| 합 (R−1,)."""
    return np.abs(diff[NEAR]).sum(axis=0)


def joint_speed(frame_ns, world, valid):
    """(시각 ns, 속도 m/s): 이웃한 두 valid 프레임의 21점 이동 거리 평균 ÷ 시간. 시각은 두 프레임 가운데."""
    t = np.asarray(frame_ns, dtype=np.float64)
    dt = np.diff(t)
    use = valid[1:] & valid[:-1] & (dt > 0) & (dt <= MAX_GAP_SEC * 1e9)
    step = np.linalg.norm(world[1:] - world[:-1], axis=2).mean(axis=1)
    return (t[1:] + t[:-1])[use] / 2, step[use] / (dt[use] / 1e9)


@dataclass
class Lag:
    lag_ns: float  # +면 카메라가 늦다
    peak: float  # 상관 봉우리(표준화한 두 값의 평균 곱, −1..1)
    lags_ns: np.ndarray
    corr: np.ndarray
    used: int  # 상관에 쓴 소리 줄 수(지연 0일 때)
    sound: np.ndarray  # 소리 줄 시각의 소리 값(로그, 이동 평균 뺌, 표준화. 못 쓰는 곳은 nan)
    camera: np.ndarray  # 같은 시각에 놓은 카메라 값(같은 처리)


def _prepare(v, mask, width):
    """로그(mask 값 중앙값의 1%를 더해 0을 피함) → width줄 이동 평균(mask인 곳만) 빼기 → 표준화. mask 밖은 0."""
    v = np.log(v + 0.01 * np.median(v[mask]))
    count = uniform_filter1d(mask.astype(np.float64), width, mode="nearest")
    v = v - uniform_filter1d(np.where(mask, v, 0.0), width, mode="nearest") / np.maximum(count, 1e-9)
    out = np.zeros(len(v))
    s = float(np.std(v[mask]))
    out[mask] = (v[mask] - np.mean(v[mask])) / (s if s > 0 else 1.0)
    return out


def estimate_lag(sound_ns, energy, cam_ns, speed, max_lag_s=MAX_LAG_SEC):
    """소리 세기(고른 시각 sound_ns)와 카메라 속도(시각 cam_ns, 오름차순)의 지연. 상관(L) = 평균 z소리[t]·z카메라[t + L].
    쓸 줄이 MIN_ROWS보다 적거나 어느 지연에서도 상관을 못 재면 None."""
    sound_ns = np.asarray(sound_ns, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    cam_ns = np.asarray(cam_ns, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    if len(sound_ns) < 2 or len(cam_ns) < 2:
        return None
    step = float(np.median(np.diff(sound_ns)))
    j = np.searchsorted(cam_ns, sound_ns)
    jj = np.clip(j, 1, len(cam_ns) - 1)
    cam = np.interp(sound_ns, cam_ns, speed)
    valid = ((j > 0) & (j < len(cam_ns)) & (cam_ns[jj] - cam_ns[jj - 1] <= MAX_GAP_SEC * 1e9)
             & (np.nan_to_num(energy, nan=-1.0) >= 0) & (cam >= 0))
    if valid.sum() < MIN_ROWS:
        return None
    width = max(1, int(round(DETREND_SEC * 1e9 / step)))
    ze = _prepare(np.where(valid, energy, 1.0), valid, width)
    zc = _prepare(np.where(valid, cam, 1.0), valid, width)
    k = int(round(max_lag_s * 1e9 / step))
    lags = np.arange(-k, k + 1)
    corr = np.full(len(lags), np.nan)
    n = len(ze)
    for i, lag in enumerate(lags):
        a, b = (slice(0, n - lag), slice(lag, n)) if lag >= 0 else (slice(-lag, n), slice(0, n + lag))
        m = valid[a] & valid[b]
        if m.sum() > 10:
            corr[i] = float(np.mean(ze[a][m] * zc[b][m]))
    if not np.isfinite(corr).any():
        return None
    best = int(np.nanargmax(corr))
    frac = 0.0
    if 0 < best < len(corr) - 1 and np.isfinite(corr[best - 1]) and np.isfinite(corr[best + 1]):
        den = corr[best - 1] - 2 * corr[best] + corr[best + 1]
        if den < 0:
            frac = float(np.clip(0.5 * (corr[best - 1] - corr[best + 1]) / den, -0.5, 0.5))
    return Lag((lags[best] + frac) * step, float(corr[best]), lags * step, corr, int(valid.sum()),
               np.where(valid, ze, np.nan), np.where(valid, zc, np.nan))


def figure(path, sound_ns, lag, prompts, title):
    """그림 한 장: 위는 첫 동작 1 s 앞부터 20 s 동안 상관에 쓴 두 값(카메라는 지연을 뺀 자리)과 동작 이름, 아래는 지연별 상관."""
    plt = hs_io.pyplot()
    t0 = next((p["startNs"] for p in prompts if p["segment"] > 0), sound_ns[0] + 1e9) - 1e9
    width = max(1, int(round(SMOOTH_SEC * 1e9 / float(np.median(np.diff(sound_ns))))))

    def smooth(v):
        return uniform_filter1d(np.nan_to_num(v), width) / np.maximum(uniform_filter1d(np.isfinite(v) * 1.0, width), 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    ax = axes[0]
    for p in prompts:
        x = (p["startNs"] - t0) / 1e9
        if 0 <= x <= EXCERPT_SEC:
            ax.axvline(x, color="#bbbbbb", lw=0.6)
            ax.text(x + 0.05, 0.98, p["name"].split()[0], transform=ax.get_xaxis_transform(), va="top", fontsize=8)
    for times, values, color, label in (
            ((sound_ns - lag.lag_ns - t0) / 1e9, lag.camera, "#d9622b", f"카메라: 관절 속도(로그, 1 s 평균 뺌, 지연 {lag.lag_ns / 1e6:+.1f} ms 뺌)"),
            ((sound_ns - t0) / 1e9, lag.sound, "#1f6fb4", "소리: 차분 가까운 60칸 세기(같은 처리)")):
        shown = (times >= 0) & (times <= EXCERPT_SEC)
        ax.plot(times[shown], smooth(values)[shown], lw=1.0, color=color, label=label)
    ax.set_xlim(0, EXCERPT_SEC)
    ax.set_xlabel("첫 동작 1 s 앞부터의 시간(s)")
    ax.set_ylabel("표준화 세기")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11)
    ax = axes[1]
    ax.axvspan(-FRAME_MS, FRAME_MS, color="#e8f3e8", lw=0, label="±1프레임(33 ms)")
    ax.plot(lag.lags_ns / 1e6, lag.corr, color="#333333", lw=1)
    ax.axvline(lag.lag_ns / 1e6, color="#d9622b", lw=1, label=f"봉우리 {lag.lag_ns / 1e6:+.1f} ms, 상관 {lag.peak:.2f}")
    ax.set_xlabel("지연(ms, +면 카메라가 늦음) — 세션 전체")
    ax.set_ylabel("상관")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


INPUTS = ("align.json", "rows.npz", "landmarks.csv")  # sync.json이 지문을 적는 입력(landmarks.csv는 세션 폴더, 나머지는 processed)


def inputs(data, name):
    """입력 지문 {파일: SHA-256}. verify_sync가 sync.json에 적고 pairs.py가 지금 파일과 견준다."""
    out = os.path.join(data, "processed", name)
    folder = hs_io.session_folder(data, name)
    return {f: hs_io.sha256(os.path.join(folder if f == "landmarks.csv" else out, f)) for f in INPUTS}


def usable(sync):
    """sync.json 내용으로 짝지어도 되는가: (됨, 안 되는 이유)."""
    lag, peak = hs_io.num(sync, "lagMs"), hs_io.num(sync, "peak")
    if lag is None or peak is None:
        return False, "sync.json에 지연·봉우리가 없음"
    if not sync.get("withinFrame") or abs(lag) > FRAME_MS:
        return False, f"지연 {lag:+.1f} ms가 1프레임({FRAME_MS:.0f} ms) 밖"
    if peak < MIN_PEAK:
        return False, f"상관 봉우리 {peak:.3f}가 문턱 {MIN_PEAK} 아래"
    return True, ""


def _measure(data, name):
    """(보고 줄, 맞춤을 확인했는가, sync.json을 새로 썼는가)."""
    out = os.path.join(data, "processed", name)
    if not all(os.path.exists(os.path.join(out, f)) for f in ("align.json", "rows.npz", "diff_profiles.npy")):
        return [f"{name}: 건너뜀(align.json·rows.npz·diff_profiles.npy 중 없는 것이 있음 — align.py·profiles.py를 먼저)"], False, False
    fingerprints = inputs(data, name)
    rows = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)
    if "alignSha" not in rows.files or str(rows["alignSha"]) != fingerprints["align.json"]:
        return [f"{name}: 건너뜀(rows.npz가 지금 align.json에서 나오지 않음 — profiles.py를 다시)"], False, False
    diff = np.load(os.path.join(out, "diff_profiles.npy"), mmap_mode="r", allow_pickle=False)
    if diff.shape != (600, len(rows["ns"]) - 1):
        return [f"{name}: 건너뜀(diff_profiles.npy {diff.shape}와 rows.npz {len(rows['ns'])}줄이 맞지 않음 — profiles.py를 다시)"], False, False
    sound_ns = (rows["ns"][1:] + rows["ns"][:-1]) / 2
    energy = near_energy(diff)
    energy[~(rows["ok"][1:] & rows["ok"][:-1])] = np.nan  # 직접 경로가 없거나 옮김 둘레·계단 둘레인 줄은 쓰지 않는다
    folder = hs_io.session_folder(data, name)
    marks = hs_io.read_landmarks(folder)
    cam_ns, speed = joint_speed(marks.frame_ns, marks.world, marks.valid)
    if len(cam_ns) < 20:
        return [f"{name}: 건너뜀(쓸 카메라 프레임 쌍이 {len(cam_ns)}개뿐)"], False, False
    lag = estimate_lag(sound_ns, energy, cam_ns, speed)
    if lag is None:
        return [f"{name}: 건너뜀(상관에 쓸 소리 줄이 {MIN_ROWS}개 미만 — ok인 줄과 카메라 시간이 겹치지 않음)"], False, False
    meta = hs_io.read_json(os.path.join(folder, "meta.json"))
    time_base = (meta.get("camera") or {}).get("timeBase", "?")
    within = bool(abs(lag.lag_ns) / 1e6 <= FRAME_MS)
    sync = {"format": 2, "session": name, "lagMs": float(lag.lag_ns / 1e6), "peak": lag.peak, "withinFrame": within, "frameMs": FRAME_MS,
            "minPeak": MIN_PEAK, "soundRowsUsed": lag.used, "cameraPairs": int(len(cam_ns)), "timeBase": time_base, "inputs": fingerprints}
    sync["usable"], why = usable(sync)
    figure(os.path.join(out, "sync.png"), sound_ns, lag, hs_io.read_prompts(folder),
           f"{name}: 카메라 관절 속도와 손 거리 차분 세기(카메라 시각 {time_base})")
    hs_io.write_json(os.path.join(out, "sync.json"), sync)
    lines = [f"{name}: 카메라–소리 지연 {lag.lag_ns / 1e6:+.1f} ms(상관 봉우리 {lag.peak:.2f}, 문턱 {MIN_PEAK}), 1프레임(33 ms) {'안' if within else '밖'}",
             f"  카메라 프레임 쌍 {len(cam_ns)}개, 소리 줄 {lag.used}개 씀, 카메라 시각 {time_base}{'(근사)' if time_base == 'arrival' else ''}"]
    if not sync["usable"]:
        lines.append(f"  맞춤을 확인하지 못함({why}): pairs.py가 이 세션을 짝짓지 않는다")
    return lines, sync["usable"], True


def run(data, name):
    """세션 하나: (보고 줄, 맞춤을 확인했는가). sync.json을 새로 쓰지 못하고 끝나면 옛 sync.json을 지우지 않고 sync.stale.json으로 바꾼다."""
    try:
        lines, done, wrote = _measure(data, name)
    except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 뒤 세션을 한다
        lines, done, wrote = [f"{name}: 처리 중 오류로 실패({type(e).__name__}: {e})"], False, False
    if not wrote and hs_io.retire(os.path.join(data, "processed", name, "sync.json")):
        lines.append("  옛 sync.json을 sync.stale.json으로 바꿈(지금 입력에서 잰 것이 아님)")
    return lines, done


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="카메라 관절 속도와 소리 차분 세기로 카메라–소리 지연을 잰다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    bad = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        lines, done = run(args.data, name)
        print("\n".join(lines))
        bad += not done
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
