import json
import os
import struct
import tempfile
import unittest
import wave

import check_recording as cr  # 사용자 설치 numpy 경로도 여기서 붙는다

import numpy as np  # noqa: E402

RATE = cr.RATE


def sweep_signal(n, amplitude=0.3, seed=0):
    """18→21 kHz 600샘플 처프를 이어 붙이고 작은 잡음을 더한 16비트 신호."""
    t = np.arange(600) / RATE
    one = np.sin(2 * np.pi * (18_000 * t + 0.5 * (3_000 / (600 / RATE)) * t ** 2))
    x = np.tile(one, n // 600 + 1)[:n] * amplitude
    x = x + np.random.default_rng(seed).normal(0, 0.001, n)
    return (x * 32767).astype("<i2")


def write_wav(path, samples, closed=True):
    if closed:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(samples.tobytes())
    else:
        # 워치 앱이 닫기 전에 죽은 조각: RIFF 길이 36, data 길이 0(WavWriter와 같음)
        header = (b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
                  + b"data" + struct.pack("<I", 0))
        with open(path, "wb") as f:
            f.write(header)
            f.write(samples.tobytes())


def make_recording(folder, frames=(cr.PART_FRAMES, RATE * 5), signal=True, last_closed=True, second_first_shift=0, silent_parts=()):
    parts, first = [], 0
    for i, n in enumerate(frames):
        name = f"part-{i:03d}.wav"
        samples = sweep_signal(n) if signal and i not in silent_parts else np.zeros(n, dtype="<i2")
        write_wav(os.path.join(folder, name), samples, closed=last_closed or i < len(frames) - 1)
        parts.append({"file": name, "firstFrame": first + (second_first_shift if i == 1 else 0), "frames": n})
        first += n
    total = sum(frames)
    t0 = 1_000_000_000
    seconds = total // RATE
    rec = {
        "format": 1, "app": "0.1.0", "device": "test", "android": "16",
        "start": {"wallMs": 0, "elapsedNs": t0},
        "stop": {"wallMs": 1, "elapsedNs": t0 + seconds * 1_000_000_000, "reason": "user"},
        "audio": {"rate": RATE, "bits": 16, "channels": 1, "source": "VOICE_RECOGNITION"},
        "tx": {"f0Hz": 18000.0, "bandwidthHz": 3000.0, "samples": 600, "window": "none", "amplitude": 0.5},
        "volume": {"stream": "music", "index": 10, "max": 15},
        "recordClockSource": "timestamp",
        "recordClock": [[k * RATE + 960, t0 + k * 1_000_000_000, k * RATE] for k in range(seconds)],
        "playClock": [[k * RATE, t0 + k * 1_000_000_000] for k in range(seconds)],
        "underruns": [],
        "parts": parts,
    }
    with open(os.path.join(folder, "rec.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)


def edit_rec(folder, edit):
    """rec.json을 읽어 edit(rec)로 고친 뒤 다시 쓴다(시험마다 기록의 한 곳만 바꿀 때 쓴다)."""
    path = os.path.join(folder, "rec.json")
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    edit(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)


class CheckRecordingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_recording_has_no_problems(self):
        make_recording(self.dir)
        lines, problems = cr.check(self.dir)
        self.assertEqual([], problems, lines)

    def test_chosen_source_and_fallback_are_reported_but_not_problems(self):
        make_recording(self.dir)
        edit_rec(self.dir, lambda rec: rec["audio"].update(sourceRequested="unprocessed", sourceFallback="UNPROCESSED 녹음 초기화 실패"))
        lines, problems = cr.check(self.dir)
        self.assertEqual([], problems, lines)
        self.assertTrue(any("unprocessed을 열지 못해 VOICE_RECOGNITION로 녹음함" in line for line in lines), lines)

    def test_unclosed_last_part_is_measured_by_file_size(self):
        make_recording(self.dir, last_closed=False)
        _, problems = cr.check(self.dir)
        self.assertEqual([], problems)

    def test_first_frame_gap_is_reported(self):
        make_recording(self.dir, second_first_shift=480)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("firstFrame" in p for p in problems), problems)

    def test_silence_is_reported(self):
        make_recording(self.dir, signal=False)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("약함" in p for p in problems), problems)

    def test_overflow_jump_is_reported(self):
        make_recording(self.dir)
        path = os.path.join(self.dir, "rec.json")
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        for p in rec["recordClock"][30:]:
            p[0] += 9_600  # 200 ms 넘침
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("넘침" in p for p in problems), problems)

    def test_error_stop_is_reported(self):
        make_recording(self.dir)
        edit_rec(self.dir, lambda rec: rec["stop"].update(reason="error"))
        _, problems = cr.check(self.dir)
        self.assertTrue(any("error" in p for p in problems), problems)

    def test_storage_stop_is_not_a_problem_but_is_labelled(self):
        make_recording(self.dir)
        edit_rec(self.dir, lambda rec: rec["stop"].update(reason="storage"))
        lines, problems = cr.check(self.dir)
        self.assertEqual([], problems, lines)
        self.assertTrue(any("저장 공간 부족으로 자동 정지" in line for line in lines), lines)

    def test_silent_later_part_is_reported_by_name(self):
        make_recording(self.dir, silent_parts=(1,))
        _, problems = cr.check(self.dir)
        self.assertTrue(any(p.startswith("part-001.wav") and "약함" in p for p in problems), problems)
        self.assertFalse(any("part-000.wav" in p for p in problems), problems)

    def test_playclock_stall_is_reported(self):
        make_recording(self.dir)

        def stall(rec):
            for p in rec["playClock"][30:]:
                p[0] = 30 * RATE  # 30초 뒤부터 재생 위치가 늘지 않음

        edit_rec(self.dir, stall)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("재생이 멈췄거나" in p for p in problems), problems)

    def test_empty_playclock_is_reported(self):
        make_recording(self.dir)
        edit_rec(self.dir, lambda rec: rec.update(playClock=[]))
        _, problems = cr.check(self.dir)
        self.assertTrue(any("재생 시각 기록이 없음" in p for p in problems), problems)

    def test_stop_length_mismatch_is_reported(self):
        make_recording(self.dir)
        edit_rec(self.dir, lambda rec: rec["stop"].update(elapsedNs=rec["stop"]["elapsedNs"] + 25 * 60 * 1_000_000_000))
        _, problems = cr.check(self.dir)
        self.assertTrue(any("오디오 길이" in p for p in problems), problems)

    def test_read_clock_stall_is_reported(self):
        make_recording(self.dir)

        def read_stall(rec):
            rec["recordClockSource"] = "read"
            rec["recordClock"] = [[p[2], p[1], p[2]] for p in rec["recordClock"]]  # read 줄: [읽은 수, 시각, 읽은 수]
            rec["recordClock"][30][1] += 200_000_000  # 한 줄의 시각만 0.2 s 늦음

        edit_rec(self.dir, read_stall)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("넘침" in p for p in problems), problems)

    def test_read_clock_without_stall_has_no_problems(self):
        make_recording(self.dir)

        def read_clean(rec):
            rec["recordClockSource"] = "read"
            rec["recordClock"] = [[p[2], p[1], p[2]] for p in rec["recordClock"]]

        edit_rec(self.dir, read_clean)
        lines, problems = cr.check(self.dir)
        self.assertEqual([], problems, lines)

    def test_stale_timestamp_row_is_not_overflow(self):
        make_recording(self.dir)

        def stale(rec):
            # getTimestamp가 지난 값을 다시 줌(실기기 205800의 32번 줄): 위치·시각은 31번 줄과 같고 읽은 수만 한 초 더 늘었다
            clock = rec["recordClock"]
            clock[32][0] = clock[31][0]
            clock[32][1] = clock[31][1]

        edit_rec(self.dir, stale)
        lines, problems = cr.check(self.dir)
        self.assertEqual([], problems, lines)
        self.assertTrue(any("1개" in line and "뺌" in line for line in lines), lines)

    def test_dropped_samples_show_as_time_step(self):
        make_recording(self.dir)

        def dropped(rec):
            # 앱이 늦게 읽어 서버가 50 ms 어치를 버림: 버린 샘플은 위치에도 읽은 수에도 안 들고 시각만 그만큼 흐른다
            for p in rec["recordClock"][30:]:
                p[1] += 50_000_000

        edit_rec(self.dir, dropped)
        _, problems = cr.check(self.dir)
        self.assertTrue(any("넘침" in p for p in problems), problems)

    def test_truncated_closed_part_is_reported(self):
        make_recording(self.dir)
        path = os.path.join(self.dir, "part-000.wav")
        with open(path, "r+b") as f:
            f.truncate(cr.HEADER_BYTES + 2 * RATE * 50)  # 닫힌 60 s 조각을 50 s 어치로 자름(무선 adb pull이 끊긴 경우)
        _, problems = cr.check(self.dir)
        self.assertTrue(any(p.startswith("part-000.wav") for p in problems), problems)

    def test_parts_shorter_than_one_second_cannot_be_measured(self):
        make_recording(self.dir, frames=(RATE // 2,))
        _, problems = cr.check(self.dir)
        self.assertTrue(any("1초보다 짧아 대역을 못 잼" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
