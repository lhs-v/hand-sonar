import csv
import json
import os
import tempfile
import unittest

import pairs  # 사용자 설치 numpy 경로도 여기서 붙는다
import hs_io
import verify_sync

import numpy as np  # noqa: E402

NAME = "20990101-000005"
T0 = 1_600_000_000_000_000  # 첫 카메라 프레임(폰 elapsedNs)
FRAMES = 120  # 30 fps 4 s
ROW_NS = 12.5e6 * (1 + 2e-6)  # 프로파일 줄 간격(시계 차 +2 ppm)
SEC = 1_000_000_000


def write_session(data):
    """합성 폰 세션(meta·landmarks·prompts)과 처리 결과(align.json·rows.npz). 동작 줄: 엄지 0.2–1.2 s, 약지 1.2–2.4 s(일시정지로 끊김),
    일시정지 2.4–2.9 s, 약지 2.9–4.9 s(틀림 표시)."""
    folder = os.path.join(data, "phone", "sessions", NAME)
    os.makedirs(folder)
    meta = {"format": 1, "status": "kept", "cancelled": False, "camera": {"timeBase": "realtime", "measuredFps": 30.0},
            "start": {"wallMs": 1, "elapsedNs": T0 - SEC}, "end": {"wallMs": 6_000, "elapsedNs": T0 + 5 * SEC},
            "stats": {"frames": FRAMES}, "video": {"ok": True, "frames": FRAMES}}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(folder, "landmarks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frameNs", "hands", "handedness", "score"] + [f"v{i}" for i in range(126)])
        for n in range(FRAMES):
            w.writerow([T0 + n * SEC // 30, 1, "Right", 0.9] + [0.5 + 0.001 * n] * 126)
    prompt_rows = [(1, 0, "엄지 굽히기", 0.2, 1.2, ""), (1, 3, "약지 굽히기", 1.2, 2.4, "pause"), (0, -2, "일시정지", 2.4, 2.9, "pause"),
                   (1, 3, "약지 굽히기", 2.9, 4.9, "wrong")]
    with open(os.path.join(folder, "prompts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["index", "segment", "classId", "name", "startNs", "endNs", "flag"])
        for i, (seg, cls, label, a, b, flag) in enumerate(prompt_rows):
            w.writerow([i, seg, cls, label, T0 + int(a * SEC), T0 + int(b * SEC), flag])
    out = os.path.join(data, "processed", NAME)
    os.makedirs(out)
    hs_io.write_json(os.path.join(out, "align.json"), {"format": 1, "session": NAME, "method": "two"})
    write_rows(data)
    return folder, out


def write_rows(data, ok_row=None):
    """rows.npz: 첫 프레임 1 s 앞부터 6 s. ok_row 줄은 ok가 아니다."""
    out = os.path.join(data, "processed", NAME)
    ns = T0 - SEC + np.arange(480) * ROW_NS
    ok = np.ones(480, dtype=bool)
    if ok_row is not None:
        ok[ok_row] = False
    np.savez(os.path.join(out, "rows.npz"), frame=np.arange(480, dtype=np.int64) * 600, ns=ns, direct=np.full(480, 300, dtype=np.int16),
             ok=ok, drift=np.zeros(480, dtype=np.int16), recording=np.array("20990101-000000"),
             alignSha=np.array(hs_io.sha256(os.path.join(out, "align.json"))))


def write_sync(data, lag_ms, peak):
    within = abs(lag_ms) <= verify_sync.FRAME_MS
    sync = {"format": 2, "session": NAME, "lagMs": lag_ms, "peak": peak, "withinFrame": within, "inputs": verify_sync.inputs(data, NAME)}
    sync["usable"] = verify_sync.usable(sync)[0]
    hs_io.write_json(os.path.join(data, "processed", NAME, "sync.json"), sync)


def read_pairs(out):
    with open(os.path.join(out, "pairs.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class PairsTest(unittest.TestCase):
    def test_pairs_only_from_a_confirmed_sync(self):
        with tempfile.TemporaryDirectory() as data:
            _, out = write_session(data)
            write_sync(data, 7.5, 0.18)
            lines, done = pairs.run(data, NAME)
            self.assertTrue(done, lines)
            got = read_pairs(out)
            self.assertEqual(FRAMES, len(got))
            frame_ns = np.array([int(r["frameNs"]) for r in got])
            np.testing.assert_array_equal(np.round(frame_ns - 7.5e6).astype(np.int64), [int(r["soundNs"]) for r in got])
            self.assertTrue(all(abs(float(r["dtMs"])) <= ROW_NS / 2e6 + 1e-6 for r in got))
            t = (frame_ns - T0) / SEC
            edge = np.array([r["edge"] == "1" for r in got])
            self.assertTrue(edge[(t >= 1.9) & (t < 2.4)].all())  # 일시정지로 끊긴 동작의 멈춘 때 앞 0.5 s
            self.assertFalse(edge[(t >= 1.7) & (t < 1.9)].any())
            self.assertEqual({"wrong"}, {r["flag"] for r, s in zip(got, t) if 2.9 <= s})
            self.assertEqual({""}, {r["classId"] for r, s in zip(got, t) if s < 0.2})  # 라벨 없는 프레임
            # 지연이 1프레임 밖, 상관이 문턱 아래, sync.json을 잰 뒤 rows.npz가 바뀜, sync.json 없음: 모두 짝짓지 않고 옛 pairs.csv는 stale로.
            for case in ("outside", "weak", "changed", "missing"):
                write_rows(data)
                write_sync(data, 7.5, 0.18)
                self.assertTrue(pairs.run(data, NAME)[1])
                if case == "outside":
                    write_sync(data, 50.0, 0.30)
                elif case == "weak":
                    write_sync(data, 7.5, 0.07)
                elif case == "changed":
                    write_rows(data, ok_row=100)
                else:
                    os.remove(os.path.join(out, "sync.json"))
                lines, done = pairs.run(data, NAME)
                self.assertFalse(done, (case, lines))
                self.assertFalse(os.path.exists(os.path.join(out, "pairs.csv")), case)
                self.assertTrue(os.path.exists(os.path.join(out, "pairs.stale.csv")), case)


if __name__ == "__main__":
    unittest.main()
