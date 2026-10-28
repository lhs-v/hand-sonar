"""손 소나 PC 도구가 함께 쓰는 읽기 모듈(계획 3). 기기에서 받은 파일은 이 모듈로만 읽는다.

- 워치 녹음 폴더(설계 §7.1): rec.json과 part-NNN.wav. 조각은 firstFrame으로 잇고, 닫히지 않았거나 잘린 조각은 파일 크기로 잰다
  (check_recording.wav_frames). 처음부터 빈틈 없이 이어지는 데까지만 읽는다.
- recordClock(설계 §7.1): [위치, elapsedNs, 읽은 수]. 위치나 시각이 앞으로 가지 않은 줄은 새 측정이 아니라 뺀다
  (check_recording.forward_rows). 한 줄만 3 ms 넘게 튀었다가 곧바로(반 넘게) 되돌아오는 줄은 늦게 찍힌 시각이라 뺀다(실데이터 폰
  +10.8·+7.9 ms. 빼지 않으면 25 ms 튐 하나가 계단 둘과 한 줄짜리 조각을 만들어 그 1 s를 통째로 틀리게 한다). 그다음 이웃 줄의
  |시각 간격 − 위치 간격/48 kHz|가 0.02 s를 넘으면 샘플이 빠진 계단이라 그 자리에서 직선을 나누고, 조각마다 최소제곱 직선으로
  위치 ↔ elapsedNs를 바꾼다(직선에서 1 ms 넘게 벗어난 줄은 빼고 한 번 다시 맞춘다). 줄마다 시각이 0.7 ms까지 흔들려 점을 그대로
  잇지 않는다. 계단은 두 시각 줄 사이 어딘가에서 났으므로 그 구간(앞 줄 위치, 뒤 줄 위치]으로 적는다.
- 폰 세션 폴더(설계 §7.2·§8.1): meta.json, phone_audio.wav·json, prompts.csv, landmarks.csv.
  휴지통·되살리기로 다시 쓴 meta.json은 org.json 모양이라 1.0이 1로 바뀌므로 숫자는 float()로 받고, 키가 없으면 기본값을 쓴다.
  쓸 세션은 status가 kept이고 end·stats가 있는 것뿐이다(chosen).
python -I에서도 사용자 설치 numpy를 찾게 경로를 붙인다(다른 도구는 numpy보다 먼저 이 모듈을 부른다).
"""
import csv
import json
import os
import site
import sys
from dataclasses import dataclass

sys.path.append(site.getusersitepackages())  # python -I에서도 사용자 설치 numpy·scipy를 찾게
import numpy as np  # noqa: E402

from check_recording import HEADER_BYTES, forward_rows, wav_frames  # noqa: E402

RATE = 48_000
STEP_SEC = 0.02  # check_recording.MAX_TIMESTAMP_STEP_SEC와 같다. 실기기 워치 녹음의 이웃 줄 어긋남은 최대 0.97 ms, 폰은 한 줄 튐 +10.8 ms
SPIKE_NS = 3e6  # 한 줄 튐: 앞 줄보다도 뒤 줄보다도 같은 쪽으로 3 ms 넘게 벗어나고 뒤 줄이 반 넘게 되돌아오면 그 줄만 뺀다(워치 흔들림 최대 0.97 ms의 3배)
OUTLIER_NS = 1e6  # 조각 직선에서 이만큼(흔들림이 크면 그 4σ) 넘게 벗어난 줄은 빼고 한 번 다시 맞춘다(실데이터 폰: 1–4 ms 늦은 줄 8개가 표지음 시각을 0.1 ms 치우침)
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
DEFAULT_DATA = os.path.join(os.path.dirname(REPO), "hand-sonar-data")  # 저장소 옆 데이터 폴더(설계 §7.3)
DEFAULT_PUBLIC = os.path.join(os.path.dirname(REPO), "watchhand-data", "Study1_sub9_left")  # 공개 데이터 왼손 표본


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num(d, key, default=None):
    """d[key]를 float로. 없거나 null이면 default."""
    v = d.get(key) if isinstance(d, dict) else None
    return default if v is None else float(v)


def nanos(d, key):
    """d[key]를 정수 나노초로. 없거나 null이면 None. (elapsedNs는 2^53보다 작아 float를 거쳐도 그대로다.)"""
    v = d.get(key) if isinstance(d, dict) else None
    return None if v is None else int(v)


class Clock:
    """녹음 위치(프레임) ↔ elapsedNs. 한 줄 튐을 뺀 뒤 계단마다 나눈 최소제곱 직선 조각들."""

    def __init__(self, rows, rate=RATE):
        kept, self.stale = forward_rows(rows)
        if len(kept) < 2:
            raise ValueError(f"앞으로 간 녹음 시각 줄이 {len(kept)}개뿐")
        pos = np.array([r[0] for r in kept], dtype=np.float64)
        t = np.array([r[1] for r in kept], dtype=np.float64)
        e = (t - t[0]) - (pos - pos[0]) / rate * 1e9  # 시각 − 위치 시간(ns): 고르면 거의 상수, 샘플이 빠지면 계단, 늦게 찍힌 줄은 튐
        u, v = e[1:-1] - e[:-2], e[2:] - e[1:-1]  # 줄 k의 앞 줄 대비·뒤 줄의 k 대비 어긋남(k = 1 … n−2)
        spike = np.zeros(len(pos), dtype=bool)
        spike[1:-1] = (np.abs(u) > SPIKE_NS) & (np.abs(v) > SPIKE_NS) & (u * v < 0) & (np.abs(u + v) <= 0.5 * np.abs(u))
        self.spikes = [(float(p), float(d)) for p, d in zip(pos[1:-1][spike[1:-1]], u[spike[1:-1]])]  # 뺀 줄 (위치, 튐 ns)
        pos, t, e = pos[~spike], t[~spike], e[~spike]
        gap = np.diff(e)  # 이웃 줄의 시각 간격 − 위치 간격(ns)
        cuts = [0] + [int(k) + 1 for k in np.flatnonzero(np.abs(gap) > STEP_SEC * 1e9)] + [len(pos)]
        self.steps = [(float(pos[k - 1]), float(pos[k]), float(gap[k - 1])) for k in cuts[1:-1]]  # (앞 줄 위치, 뒤 줄 위치, 계단 ns)
        starts, ns0, slope = [], [], []
        self.outliers = 0  # 조각 직선에서 벗어나 다시 맞출 때 뺀 줄 수
        for a, b in zip(cuts, cuts[1:]):
            p, q = pos[a:b] - pos[a], t[a:b] - t[a]
            k = np.polyfit(p, q, 1) if len(p) >= 2 else (1e9 / rate, 0.0)
            if len(p) >= 5:
                res = q - np.polyval(k, p)
                res -= np.median(res)
                keep = np.abs(res) <= max(OUTLIER_NS, 4 * 1.4826 * float(np.median(np.abs(res))))
                if 3 <= keep.sum() < len(p):
                    k = np.polyfit(p[keep], q[keep], 1)
                    self.outliers += int((~keep).sum())
            starts.append(pos[a])
            ns0.append(t[a] + k[1])
            slope.append(k[0])
        self.starts, self.ns0, self.ns_per_frame = np.array(starts), np.array(ns0), np.array(slope)

    def ns(self, frames):
        f = np.asarray(frames, dtype=np.float64)
        i = np.clip(np.searchsorted(self.starts, f, side="right") - 1, 0, len(self.starts) - 1)
        return self.ns0[i] + (f - self.starts[i]) * self.ns_per_frame[i]

    def frames(self, ns_values):
        t = np.asarray(ns_values, dtype=np.float64)
        i = np.clip(np.searchsorted(self.ns0, t, side="right") - 1, 0, len(self.ns0) - 1)
        return self.starts[i] + (t - self.ns0[i]) / self.ns_per_frame[i]

    def steps_between(self, f1, f2):
        """구간 (앞 줄 위치, 뒤 줄 위치]이 (f1, f2]와 겹치는 계단 [(앞 줄 위치, 뒤 줄 위치, ns)]. 계단은 그 구간 어딘가에서 났다."""
        return [s for s in self.steps if s[0] < f2 and s[1] > f1]


@dataclass
class WatchRecording:
    """워치 녹음 폴더 하나. frames는 처음부터 빈틈 없이 읽을 수 있는 프레임 수다."""
    folder: str
    rec: dict
    parts: list  # [(경로, 첫 프레임, 프레임 수)], 이어진 조각만
    frames: int
    clock: Clock
    problems: list

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.folder))

    def wall_ms_to_ns(self, wall_ms):
        """이 워치의 wallMs를 elapsedNs로(rec.json start의 한 쌍 기준)."""
        start = self.rec["start"]
        return nanos(start, "elapsedNs") + (wall_ms - num(start, "wallMs")) * 1e6

    def read(self, first, last):
        """프레임 [first, last)를 float64로. 0 ≤ first ≤ last ≤ frames여야 한다."""
        if not 0 <= first <= last <= self.frames:
            raise ValueError(f"{self.name}: 읽을 범위 {first}–{last}가 0–{self.frames} 밖")
        out = []
        for path, start, n in self.parts:
            a, b = max(first, start), min(last, start + n)
            if a < b:
                out.append(np.fromfile(path, dtype="<i2", count=b - a, offset=HEADER_BYTES + 2 * (a - start)))
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def read_recording(folder):
    """워치 녹음 폴더를 읽는다. 조각이 없거나 firstFrame이 어긋나면(빈틈) 거기서 멈추고 problems에 적는다."""
    rec = read_json(os.path.join(folder, "rec.json"))
    parts, problems, expected = [], [], 0
    for p in rec.get("parts") or []:
        path = os.path.join(folder, p["file"])
        if not os.path.exists(path):
            problems.append(f"{p['file']} 없음")
            break
        if int(p["firstFrame"]) != expected:
            problems.append(f"{p['file']}: firstFrame {p['firstFrame']} ≠ {expected}(빈틈)")
            break
        n = wav_frames(path)
        if n != int(p["frames"]):
            problems.append(f"{p['file']}: 파일 {n}프레임 ≠ 기록 {p['frames']}")
        parts.append((path, expected, n))
        expected += n
        if n < int(p["frames"]):
            break  # 잘린 조각 뒤는 이어지지 않는다
    return WatchRecording(folder, rec, parts, expected, Clock(rec.get("recordClock") or []), problems)


@dataclass
class PhoneAudio:
    """phone_audio.wav와 phone_audio.json. 위치는 WAV 첫 샘플을 0으로 센다(음수일 수 있음)."""
    path: str
    frames: int
    clock: Clock

    def read(self, first, last):
        first, last = max(0, first), min(self.frames, last)
        if first >= last:
            return np.zeros(0)
        return np.fromfile(self.path, dtype="<i2", count=last - first, offset=HEADER_BYTES + 2 * first).astype(np.float64)


@dataclass
class PhoneSession:
    folder: str
    meta: dict
    audio: PhoneAudio | None  # phone_audio.json이 없으면 None(그때는 폰 WAV를 쓰지 않는다, 설계 §8.1)

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.folder))

    @property
    def hand(self):
        return self.meta.get("hand") or "left"  # 2026-10-28 전 세션에는 키가 없고 왼손이다

    def wall_ms_to_ns(self, wall_ms):
        """폰 wallMs를 폰 elapsedNs로(meta start의 한 쌍 기준. start는 시작 버튼 시각이라 맞춤 점으로는 쓰지 않는다)."""
        start = self.meta["start"]
        return nanos(start, "elapsedNs") + (wall_ms - num(start, "wallMs")) * 1e6

    def ns_to_wall_ms(self, ns_value):
        start = self.meta["start"]
        return num(start, "wallMs") + (ns_value - nanos(start, "elapsedNs")) / 1e6

    def marker(self, key):
        """meta stats.markers.start·end(설계의 stats.*Marker). 없으면 빈 dict."""
        return ((self.meta.get("stats") or {}).get("markers") or {}).get(key) or {}


def usable(meta):
    """(쓸 수 있는가, 이유). 설계 §8.1: status가 kept이고 end·stats가 있는 세션만 쓴다."""
    if meta.get("status") != "kept":
        return False, f"상태 {meta.get('status')}"
    if not meta.get("end"):
        return False, "끝나지 않은 세션(end 없음)"
    if not meta.get("stats"):
        return False, "stats 없음"
    return True, ""


def read_session(folder):
    meta = read_json(os.path.join(folder, "meta.json"))
    audio = None
    info_path = os.path.join(folder, "phone_audio.json")
    wav_path = os.path.join(folder, "phone_audio.wav")
    if os.path.exists(info_path) and os.path.exists(wav_path):
        try:
            audio = PhoneAudio(wav_path, wav_frames(wav_path), Clock(read_json(info_path).get("recordClock") or []))
        except ValueError:
            audio = None  # 녹음 시각 줄이 2개 미만이거나 WAV 형식이 다르면 폰 소리를 쓰지 않는다
    return PhoneSession(folder, meta, audio)


def read_prompts(folder):
    """prompts.csv 줄 목록. index는 이 파일의 행 번호다(계획 순서 번호가 아님, 설계 §8.1). 파일이 없으면 []."""
    path = os.path.join(folder, "prompts.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            out.append({"index": int(r["index"]), "segment": int(r["segment"]), "classId": int(r["classId"]), "name": r["name"],
                        "startNs": int(r["startNs"]), "endNs": int(r["endNs"]), "flag": r["flag"] or ""})
        except (KeyError, TypeError, ValueError):
            continue  # 되살린 취소 세션은 마지막 줄이 잘렸을 수 있다
    return out


LANDMARK_COLUMNS = 130  # frameNs, hands, handedness, score + 21점 × (x, y, z) × (화면, 세계)
WRIST_JUMP = 0.1  # 손이 둘 보인 줄에서 손목(점 0)이 앞 줄보다 화면 폭의 10% 넘게 뛰면 다른 손을 고른 것으로 본다(30 fps에서 1 m/s ≈ 0.07)


@dataclass
class Landmarks:
    """landmarks.csv. 좌표는 파일 그대로(거울 그림 기준)다. 실제 손 모양은 x → 1 − x, wx → −wx(설계 §8.1)."""
    frame_ns: np.ndarray  # (M,) int64 카메라 시각(폰 elapsedNs). camera.timeBase가 arrival이면 근사
    hands: np.ndarray  # (M,) 찾은 손 개수
    tracked: np.ndarray  # (M,) 추적하는 손이 있음
    image: np.ndarray  # (M, 21, 3) x·y는 회전·거울 뒤 프레임 기준 0..1(조금 넘을 수 있음), z는 손목 기준 상대 깊이
    world: np.ndarray  # (M, 21, 3) 손 중심이 원점인 미터 좌표
    valid: np.ndarray  # (M,) 추적하는 손이 있고, 손이 둘 보인 줄이면 손목이 앞 줄과 이어짐
    dropped: int  # 열이 130개가 아니라(끝이 잘림) 버린 줄 수


def read_landmarks(folder):
    """landmarks.csv(설계 §7.2·§8.1). 열이 130개가 아닌 줄은 버린다. 데이터 줄 번호가 곧 영상 프레임 번호인지는 pairs.py가 meta로 따진다."""
    frame_ns, hands, tracked, values, dropped = [], [], [], [], 0
    with open(os.path.join(folder, "landmarks.csv"), encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                if len(row) != LANDMARK_COLUMNS:
                    raise ValueError
                has = row[2] != ""
                v = [float(s) for s in row[4:]] if has else [np.nan] * (LANDMARK_COLUMNS - 4)
                t, h = int(row[0]), int(row[1])
            except ValueError:
                dropped += 1
                continue
            frame_ns.append(t)
            hands.append(h)
            tracked.append(has)
            values.append(v)
    points = np.array(values, dtype=np.float64).reshape(-1, 2, 21, 3)
    tracked = np.array(tracked, dtype=bool)
    hands = np.array(hands, dtype=np.int64)
    valid = tracked.copy()
    last = None
    for i in np.flatnonzero(tracked):
        wrist = points[i, 0, 0, :2]
        if hands[i] >= 2 and last is not None and float(np.hypot(*(wrist - last))) > WRIST_JUMP:
            valid[i] = False
            continue
        last = wrist
    return Landmarks(np.array(frame_ns, dtype=np.int64), hands, tracked, points[:, 0], points[:, 1], valid, dropped)


def chosen(data, names=()):
    """[(세션 이름, 건너뛸 이유)]. 이유가 ""이면 쓴다. 이름이 없으면 phone/sessions 모두(설계 §8.1 고르기, 폰에서 지운 세션은 쓰지 않음)."""
    out = []
    for name in names or session_names(data):
        path = os.path.join(session_folder(data, name), "meta.json")
        try:
            ok, why = usable(read_json(path))
        except (OSError, ValueError) as e:
            ok, why = False, f"meta.json을 못 읽음({e})"
        out.append((name, "" if ok else why))
    return out


def session_names(data):
    root = os.path.join(data, "phone", "sessions")
    return sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, "meta.json"))) if os.path.isdir(root) else []


def session_folder(data, name):
    return os.path.join(data, "phone", "sessions", name)


def recording_folders(data):
    root = os.path.join(data, "watch", "recordings")
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n, "rec.json"))]


def processed_folder(data, name):
    path = os.path.join(data, "processed", name)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, value):
    """임시 파일에 다 쓴 뒤 이름을 바꾼다(중간에 멈춰도 반쯤 쓴 파일이 남지 않게)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def guarded(name, fn, *args):
    """fn(*args) → (보고 줄, 됐는가). 예외가 나면 그 세션만 실패로 적는다(한 세션이 넘어져도 뒤 세션을 처리한다)."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001  세션마다 따로 실패시키려고 모두 받는다
        return [f"{name}: 처리 중 오류로 실패({type(e).__name__}: {e})"], False


def pyplot():
    """그림용 matplotlib(창 없이, 맑은 고딕). 그림에는 영상 프레임이나 사람 모습을 넣지 않는다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt
