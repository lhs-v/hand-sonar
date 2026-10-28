import json
import os
import tempfile
import unittest
import wave

import align  # 사용자 설치 numpy 경로도 여기서 붙는다
import hs_io

import numpy as np  # noqa: E402

from tests.test_markers import marker_at, watch_chirp

RATE = hs_io.RATE
PPM = 50e-6  # 워치 elapsed 시계가 폰보다 50 ppm 빠르다(설계 §10의 시계 빠르기 차)
P0 = 1_600_000_000_000_000  # 폰 elapsedNs 기준점
W0 = 500_000_000_000_000  # 같은 순간의 워치 elapsedNs
WATCH_START_NS = W0 + 1_000_000_000  # 워치 녹음 첫 샘플의 워치 elapsedNs
PHONE_START_NS = P0 + 3_000_000_000  # phone_audio.wav 첫 샘플의 폰 elapsedNs
P_START = P0 + 6_000_000_000 + 123_456  # 폰이 시작 표지음을 들은 폰 시각(ns)
P_END = P_START + 20_000_000_000  # 끝 표지음(20 s 뒤)
WALL_ERROR_MS = 1_200  # 워치 벽시계가 1.2 s 앞섬: 예상 위치가 실제보다 1.2 s 이르다


def watch_ns_of_phone(p):
    return W0 + (p - P0) * (1 + PPM)


def phone_ns_of_watch_frame(n):
    """정답: 워치 녹음 프레임 n(워치 ADC는 워치 elapsed 기준 정확히 48 kHz)의 폰 시각."""
    w = WATCH_START_NS + np.asarray(n, dtype=np.float64) / RATE * 1e9
    return P0 + (w - W0) / (1 + PPM)


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes())


def clock_rows(first_ns, n_frames, jitter_ns, rng, start_pos=0):
    """1 s마다 [위치, elapsedNs, 읽은 수]. 시각에 흔들림을 넣고, 11번 줄은 10번 줄을 되풀이한다(getTimestamp가 지난 값을 다시 줌)."""
    rows = []
    for pos in range(start_pos, n_frames, RATE):
        rows.append([pos, int(first_ns + pos / RATE * 1e9 + rng.uniform(-jitter_ns, jitter_ns)), max(pos, 0)])
    rows[11] = [rows[10][0], rows[10][1], rows[11][2]]
    return rows


def make_data(root, rng):
    n_watch = 30 * RATE
    s1 = (watch_ns_of_phone(P_START) - WATCH_START_NS) / 1e9 * RATE  # 시작 표지음이 워치 녹음에서 시작하는 소수 프레임
    s2 = (watch_ns_of_phone(P_END) - WATCH_START_NS) / 1e9 * RATE
    x = watch_chirp(n_watch, 3_000.0) + rng.normal(0, 30.0, n_watch)
    for start in (s1, s2, s1 - 2.0 * RATE):  # 시작·끝 표지음과, 2 s 앞 준비 화면의 "표지음 시험"
        x += marker_at(n_watch, start, 300.0)
    click = int(s1 - 0.3 * RATE)  # 시작 버튼 소리 같은 잡음(시작 표지음 0.3 s 앞)
    x[click] += 20_000.0
    folder = os.path.join(root, "watch", "recordings", "20990101-000000")
    os.makedirs(folder)
    half = n_watch // 2
    write_wav(os.path.join(folder, "part-000.wav"), x[:half])
    write_wav(os.path.join(folder, "part-001.wav"), x[half:])
    watch_wall0 = 1_000_000_000.0 + (WATCH_START_NS - W0) / (1 + PPM) / 1e6 + WALL_ERROR_MS  # 녹음 시작 순간의 워치 벽시계
    rec = {
        "format": 1, "app": "0.1.0", "device": "test", "android": "16",
        "start": {"wallMs": int(watch_wall0), "elapsedNs": WATCH_START_NS},
        "stop": {"wallMs": int(watch_wall0 + 30_000), "elapsedNs": WATCH_START_NS + 30_000_000_000, "reason": "user"},
        "audio": {"rate": RATE, "bits": 16, "channels": 1, "source": "VOICE_RECOGNITION"},
        "tx": {"f0Hz": 18000.0, "bandwidthHz": 3000.0, "samples": 600, "window": "hann", "amplitude": 0.5},
        "volume": {"stream": "music", "index": 11, "max": 15},
        "recordClockSource": "timestamp",
        "recordClock": clock_rows(WATCH_START_NS, n_watch, 300_000, rng),
        "playClock": [], "underruns": [],
        "parts": [{"file": "part-000.wav", "firstFrame": 0, "frames": half},
                  {"file": "part-001.wav", "firstFrame": half, "frames": n_watch - half}],
    }
    with open(os.path.join(folder, "rec.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)

    n_phone = 26 * RATE
    m1 = (P_START - PHONE_START_NS) / 1e9 * RATE
    m2 = (P_END - PHONE_START_NS) / 1e9 * RATE
    y = watch_chirp(n_phone, 100.0) + rng.normal(0, 30.0, n_phone) + marker_at(n_phone, m1, 3_000.0) + marker_at(n_phone, m2, 3_000.0)
    session = os.path.join(root, "phone", "sessions", "20990101-000005")
    os.makedirs(session)
    write_wav(os.path.join(session, "phone_audio.wav"), y)
    phone_clock = clock_rows(PHONE_START_NS, n_phone, 50_000, rng, start_pos=-480)  # 첫 줄 위치가 음수
    with open(os.path.join(session, "phone_audio.json"), "w", encoding="utf-8") as f:
        json.dump({"format": 1, "file": "phone_audio.wav", "rate": RATE, "bits": 16, "channels": 1, "source": "VOICE_RECOGNITION",
                   "frames": n_phone, "recordClockSource": "timestamp", "recordClock": phone_clock}, f)
    start_ns, end_ns = P_START - 400_000_000, P_END + 600_000_000
    marker = lambda p, m: {"found": True, "playNs": p - 112_000_000, "micFrame": m + 3.2, "elapsedNs": p, "ratio": 200.0}  # noqa: E731
    meta = {
        "format": 1, "app": "0.1.0", "device": "test", "android": "17",
        "camera": {"id": "3", "display": "inner", "width": 480, "height": 640, "fps": 30, "timeBase": "realtime", "mirrored": True, "rotation": 270},
        "start": {"wallMs": int(1_000_000_000 + (start_ns - P0) / 1e6), "elapsedNs": start_ns},
        "end": {"wallMs": int(1_000_000_000 + (end_ns - P0) / 1e6), "elapsedNs": end_ns},
        "status": "kept", "flags": [], "memo": "",
        "stats": {"frames": 0, "handRatio": 1.0, "watchHeardRatio": 0.05, "watchGaps": [],
                  "markers": {"start": marker(P_START, m1), "end": marker(P_END, m2)}, "durationSec": 21.0},
        "rewear": False, "video": None, "audio": {"file": "phone_audio.wav", "source": "VOICE_RECOGNITION"}, "cancelled": False,
    }
    with open(os.path.join(session, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return folder, session, s1, s2


class AlignTest(unittest.TestCase):
    def test_marker_line_is_within_a_tenth_of_a_millisecond(self):
        with tempfile.TemporaryDirectory() as root:
            folder, session_folder, s1, s2 = make_data(root, np.random.default_rng(3))
            recording = hs_io.read_recording(folder)
            info = align.align_session(hs_io.read_session(session_folder), [recording])
            self.assertEqual("two", info["method"], info["notes"])
            self.assertAlmostEqual(s1, info["watch"]["start"]["frame"], delta=0.1e-3 * RATE)  # 표지음 시험·잡음이 아닌 진짜 시작
            a = align.Alignment(info["method"], recording, **align._line_args(info["line"]))
            frames = np.linspace(s1 - RATE, s2 + RATE, 50)
            error_ms = np.abs(a.phone_ns(frames) - phone_ns_of_watch_frame(frames)) / 1e6
            self.assertLess(float(error_ms.max()), 0.1, f"최대 오차 {error_ms.max():.4f} ms")
            self.assertAlmostEqual(-WALL_ERROR_MS, info["wallDiffMs"]["start"], delta=5.0)  # 워치 벽시계가 앞서면 시스템 시각 맞춤이 그만큼 늦다


if __name__ == "__main__":
    unittest.main()
