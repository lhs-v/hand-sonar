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
- 센서 기록(계획 6): 워치 녹음 폴더의 sensors-NNN.csv, 폰 세션 폴더의 phone_sensors-NNN.csv(read_sensors). 열은
  t_ns,recv_ns,sensor,x,y,z,accuracy이고 t_ns는 이벤트 시각, recv_ns는 앱이 받은 때의 elapsedNs다. 조각은 번호 순으로 잇고, 열이 7개가
  아니거나 숫자가 아닌 줄(앱이 죽어 잘린 마지막 줄)은 버린다. 이벤트 시각이 elapsedNs 기준인지는 받은 시각과 견줘 보고(SensorTrack.event_clock),
  아니면 받은 시각을 근사로 쓴다(SensorTrack.ns). 파일이 없으면 이 기능 전 녹음·세션이다(None). meta.json의 세션 정보·rewear는
  session_context, rec.json의 녹음 볼륨은 recording_volume으로 읽는다(없는 칸은 기본값).
python -I에서도 사용자 설치 numpy를 찾게 경로를 붙인다(다른 도구는 numpy보다 먼저 이 모듈을 부른다).
"""
import csv
import hashlib
import json
import os
import re
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
WRIST_MAX_AGE_SEC = 0.5  # 마지막으로 받아들인 손목이 이보다 오래면 견주지 않는다(손이 오래 안 보였다 돌아오면 자리가 달라도 정상이다)


@dataclass
class Landmarks:
    """landmarks.csv. 좌표는 파일 그대로(거울 그림 기준)다. 실제 손 모양은 x → 1 − x, wx → −wx(설계 §8.1)."""
    frame_ns: np.ndarray  # (M,) int64 카메라 시각(폰 elapsedNs). camera.timeBase가 arrival이면 근사
    hands: np.ndarray  # (M,) 찾은 손 개수
    tracked: np.ndarray  # (M,) 추적하는 손이 있음
    image: np.ndarray  # (M, 21, 3) x·y는 회전·거울 뒤 프레임 기준 0..1(조금 넘을 수 있음), z는 손목 기준 상대 깊이
    world: np.ndarray  # (M, 21, 3) 손 중심이 원점인 미터 좌표
    valid: np.ndarray  # (M,) 추적하는 손이 있고, 손이 둘 보인 줄이면 손목이 0.5 s 안의 앞 줄과 이어짐
    dropped: int  # 열이 130개가 아니라(끝이 잘림) 버린 줄 수
    score: np.ndarray  # (M,) 추적하는 손의 판정 점수(MediaPipe 손잡이 점수, 학습 설계 §3-1 하한 0.5). 손이 없으면 nan


def read_landmarks(folder):
    """landmarks.csv(설계 §7.2·§8.1). 열이 130개가 아닌 줄은 버린다. 데이터 줄 번호가 곧 영상 프레임 번호인지는 pairs.py가 meta로 따진다."""
    frame_ns, hands, tracked, values, scores, dropped = [], [], [], [], [], 0
    with open(os.path.join(folder, "landmarks.csv"), encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                if len(row) != LANDMARK_COLUMNS:
                    raise ValueError
                has = row[2] != ""
                v = [float(s) for s in row[4:]] if has else [np.nan] * (LANDMARK_COLUMNS - 4)
                score = float(row[3]) if has else np.nan
                t, h = int(row[0]), int(row[1])
            except ValueError:
                dropped += 1
                continue
            frame_ns.append(t)
            hands.append(h)
            tracked.append(has)
            values.append(v)
            scores.append(score)
    points = np.array(values, dtype=np.float64).reshape(-1, 2, 21, 3)
    tracked = np.array(tracked, dtype=bool)
    hands = np.array(hands, dtype=np.int64)
    valid = tracked.copy()
    last, last_ns = None, 0
    for i in np.flatnonzero(tracked):
        wrist = points[i, 0, 0, :2]
        recent = last is not None and frame_ns[i] - last_ns <= WRIST_MAX_AGE_SEC * 1e9
        if hands[i] >= 2 and recent and float(np.hypot(*(wrist - last))) > WRIST_JUMP:
            valid[i] = False
            continue
        last, last_ns = wrist, frame_ns[i]
    return Landmarks(np.array(frame_ns, dtype=np.int64), hands, tracked, points[:, 0], points[:, 1], valid, dropped,
                     np.array(scores, dtype=np.float64))


SENSOR_COLUMNS = ["t_ns", "recv_ns", "sensor", "x", "y", "z", "accuracy"]  # 센서 CSV 머리줄(계획 6, shared SensorCsv.HEADER)
WATCH_SENSORS = "sensors"  # 워치 녹음 폴더의 sensors-NNN.csv(acc·gyro·mag·grav, 약 100 Hz)
PHONE_SENSORS = "phone_sensors"  # 폰 세션 폴더의 phone_sensors-NNN.csv(acc·gyro 약 100 Hz, light는 바뀔 때만)
EVENT_LAG_MIN_NS = -0.1e9  # 받은 시각 − 이벤트 시각의 중앙값이 이 값 이상이고(CDD 7.3의 맞춤 오차 권장 100 ms)
EVENT_LAG_MAX_NS = 0.5e9  # 이 값 이하면 이벤트 시각을 elapsedNs로 믿는다. 배치 없이 받으므로 보통 수 ms다(넘으면 다른 시계로 본다)


@dataclass
class SensorTrack:
    """센서 한 종류(acc·gyro·mag·grav·light)의 줄들. 파일 차례 그대로다. 시각은 그 기기의 elapsedNs여야 한다(event_clock으로 본다)."""
    kind: str
    t_ns: np.ndarray  # (n,) int64 SensorEvent.timestamp
    recv_ns: np.ndarray  # (n,) int64 앱이 이벤트를 받은 때의 SystemClock.elapsedRealtimeNanos
    values: np.ndarray  # (n, 3) float64. light는 x(lux)만이고 y·z는 nan
    accuracy: np.ndarray  # (n,) int64 SensorEvent.accuracy(−1 접촉 없음, 0 믿을 수 없음, 1–3)

    @property
    def lag_ns(self):
        """받은 시각 − 이벤트 시각(ns)."""
        return self.recv_ns - self.t_ns

    @property
    def event_clock(self):
        """이벤트 시각이 elapsedNs 기준으로 보이는가: 받은 시각과의 차 중앙값이 EVENT_LAG_MIN_NS..EVENT_LAG_MAX_NS. 줄이 없으면 False."""
        if len(self.t_ns) == 0:
            return False
        lag = float(np.median(self.lag_ns))
        return EVENT_LAG_MIN_NS <= lag <= EVENT_LAG_MAX_NS

    @property
    def ns(self):
        """맞춤에 쓸 elapsedNs: event_clock이면 이벤트 시각, 아니면 받은 시각(근사 — 받을 때까지의 지연만큼 늦다)."""
        return self.t_ns if self.event_clock else self.recv_ns

    def rate_hz(self):
        """(줄 수 − 1) ÷ 첫·마지막 이벤트 시각 간격. 줄이 2개 미만이거나 간격이 0 이하면 nan."""
        if len(self.t_ns) < 2:
            return float("nan")
        span = float(self.t_ns[-1] - self.t_ns[0])
        return (len(self.t_ns) - 1) / span * 1e9 if span > 0 else float("nan")


@dataclass
class SensorLog:
    """한 폴더의 센서 CSV 조각들을 이은 것."""
    files: list  # 읽은 조각 경로(번호 순)
    tracks: dict  # {종류: SensorTrack}, 처음 나온 차례
    dropped: int  # 열이 7개가 아니거나 숫자가 아니라(끝이 잘림) 버린 줄 수


def sensor_files(folder, base=WATCH_SENSORS):
    """folder의 <base>-NNN.csv(NNN은 숫자 셋 이상)를 번호 순으로. 폴더가 없거나 파일이 없으면 []. 쓰다 남은 .tmp 등 다른 이름은 뺀다."""
    pattern = re.compile(re.escape(base) + r"-(\d{3,})\.csv")
    if not os.path.isdir(folder):
        return []
    found = [(int(m.group(1)), os.path.join(folder, name)) for name in os.listdir(folder) if (m := pattern.fullmatch(name))]
    return [path for _, path in sorted(found)]


def read_sensors(folder, base=WATCH_SENSORS):
    """센서 CSV(계획 6)를 읽어 SensorLog로. 파일이 없으면 None(이 기능 전의 녹음·세션이거나 센서 기록을 끈 녹음 — 문제가 아니다).
    머리줄이 다른 파일은 ValueError(형식이 바뀜). 빈 파일(머리줄도 내보내기 전에 앱이 죽음)은 건너뛴다."""
    files = sensor_files(folder, base)
    if not files:
        return None
    columns = {}
    dropped = 0
    for path in files:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            if header != SENSOR_COLUMNS:
                raise ValueError(f"{os.path.basename(path)}: 머리줄이 다름 {header}")
            for row in reader:
                try:
                    if len(row) != len(SENSOR_COLUMNS) or not row[2]:
                        raise ValueError
                    t, r = int(row[0]), int(row[1])
                    v = [float(s) if s else np.nan for s in row[3:6]]
                    a = int(row[6])
                except ValueError:
                    dropped += 1
                    continue
                c = columns.setdefault(row[2], ([], [], [], []))
                c[0].append(t)
                c[1].append(r)
                c[2].append(v)
                c[3].append(a)
    tracks = {kind: SensorTrack(kind, np.array(t, dtype=np.int64), np.array(r, dtype=np.int64),
                                np.array(v, dtype=np.float64).reshape(-1, 3), np.array(a, dtype=np.int64))
              for kind, (t, r, v, a) in columns.items()}
    return SensorLog(files, tracks, dropped)


def session_context(meta):
    """폰 세션 정보(계획 6): {"place", "posture", "armSupport", "strap", "rewear"}. 고르지 않은 칸과 이 기능 전 세션은 None, rewear는 False.
    posture는 sitting | standing, armSupport는 desk | air, strap은 loose | normal | tight다."""
    ctx = meta.get("context") or {}
    return {"place": ctx.get("place") or None, "posture": ctx.get("posture"), "armSupport": ctx.get("armSupport"),
            "strap": ctx.get("strap"), "rewear": bool(meta.get("rewear"))}


def recording_volume(rec):
    """워치 녹음 볼륨(계획 6): {"index": 녹음 동안 미디어 볼륨, "max", "choice": 고른 단계(None = 그대로), "original": 녹음 전 볼륨,
    "restored": 멈춘 뒤 되돌린 값(None = 바꾸지 않음), "changes": [[elapsedNs, 볼륨], …](녹음 중 바뀐 때)}.
    이 기능 전 녹음은 choice·restored가 None, original이 index, changes가 []다."""
    v = rec.get("volume") or {}

    def whole(key):
        return None if v.get(key) is None else int(v[key])

    index = whole("index")
    original = whole("original")
    return {"index": index, "max": whole("max"), "choice": whole("choice"), "original": index if original is None else original,
            "restored": whole("restored"), "changes": [[int(t), int(i)] for t, i in v.get("changes") or []]}


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


def sha256(path):
    """파일 내용의 SHA-256(16진). 처리 결과가 지금 입력에서 나왔는지 견주는 지문이다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retire(path):
    """옛 결과 path가 있으면 지우지 않고 '<이름>.stale<확장자>'로 바꾼다(학습이 옛 결과를 집지 않게). 바꿨으면 True."""
    if not os.path.exists(path):
        return False
    root, ext = os.path.splitext(path)
    os.replace(path, root + ".stale" + ext)
    return True


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
