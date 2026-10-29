import csv
import gzip
import json
import os
import tempfile
import unittest

import export_data  # 사용자 설치 numpy 경로도 여기서 붙는다
import hs_io
import unpack_data
import verify_sync

import numpy as np  # noqa: E402

NAME = "20990101-000000"
REC = "20990101-000001"


def fake_data(root, rows=50):
    """처리까지 끝난 세션 하나와 그 워치 녹음. 영상·폰 소리·워치 소리·센서도 두어 묶음에 들지 않는지 본다."""
    sess = os.path.join(root, "phone", "sessions", NAME)
    proc = os.path.join(root, "processed", NAME)
    rec = os.path.join(root, "watch", "recordings", REC)
    for d in (sess, proc, rec):
        os.makedirs(d)
    meta = {"status": "kept", "end": {"wallMs": 2}, "stats": {"frames": 3, "durationSec": 1.0}, "hand": "left", "rewear": True,
            "start": {"wallMs": 1, "elapsedNs": 0}, "context": {"posture": "sitting", "armSupport": "desk", "strap": "normal"}}
    hs_io.write_json(os.path.join(sess, "meta.json"), meta)
    with open(os.path.join(sess, "prompts.csv"), "w", encoding="utf-8") as f:
        f.write("index,segment,classId,name,startNs,endNs,flag\n")
    with open(os.path.join(sess, "landmarks.csv"), "w", encoding="utf-8") as f:
        f.write("frameNs,hands,handedness,score\n1,1,Left,0.9\n")
    for f in ("video.mp4", "phone_audio.wav", "phone_sensors-000.csv"):
        with open(os.path.join(sess, f), "wb") as g:
            g.write(b"x")
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1e6, (600, rows))
    np.save(os.path.join(proc, "profiles.npy"), raw)
    np.save(os.path.join(proc, "diff_profiles.npy"), np.abs(raw[:, 1:]) - np.abs(raw[:, :-1]))
    np.savez(os.path.join(proc, "rows.npz"), frame=np.arange(rows) * 600, ns=np.arange(rows) * 12.5e6, direct=np.full(rows, 123, dtype=np.int16),
             ok=np.ones(rows, dtype=bool), drift=np.zeros(rows, dtype=np.int16), recording=np.array(REC), alignSha=np.array("old"))
    hs_io.write_json(os.path.join(proc, "align.json"), {"method": "two", "recording": REC})
    hs_io.write_json(os.path.join(proc, "sync.json"), {"lagMs": 10.0, "peak": 0.2, "withinFrame": True, "inputs": {"rows.npz": "old"}})
    with open(os.path.join(proc, "pairs.csv"), "w", encoding="utf-8") as f:
        f.write("frame,frameNs,soundNs,row,dtMs,hand,direct,prompt,classId,flag,edge,video\n0,1,1,5,0.1,1,1,,,,0,0\n")
    hs_io.write_json(os.path.join(rec, "rec.json"), {"volume": {"index": 13}, "tx": {"window": "hann"}, "audio": {"source": "VOICE_RECOGNITION"}})
    with open(os.path.join(rec, "part-000.wav"), "wb") as g:
        g.write(b"x")
    return raw


class ExportDataTest(unittest.TestCase):
    def test_export_leaves_out_video_audio_and_sensors_and_unpack_restores_training_input(self):
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as back:
            raw = fake_data(data)
            row = export_data.export_session(data, out, NAME)
            pack = os.path.join(out, "sessions", NAME)
            self.assertEqual({"meta.json", "prompts.csv", "landmarks.csv.gz", "align.json", "sync.json", "pairs.csv.gz", "rec.json", "echo.npz"},
                             set(os.listdir(pack)))
            self.assertEqual(13, row["volume"])
            self.assertEqual("hann", row["window"])
            self.assertTrue(row["usable"])
            with np.load(os.path.join(pack, "echo.npz")) as echo:
                self.assertEqual(np.float32, echo["raw"].dtype)
                np.testing.assert_array_equal(raw[250:480].astype(np.float32), echo["raw"])

            unpack_data.unpack_session(out, back, NAME)
            proc = os.path.join(back, "processed", NAME)
            prof = np.load(os.path.join(proc, "profiles.npy"))
            diff = np.load(os.path.join(proc, "diff_profiles.npy"))
            self.assertEqual((600, 50), prof.shape)
            np.testing.assert_allclose(raw[250:480], prof[250:480], rtol=1e-6)
            self.assertEqual(0.0, np.abs(prof[:250]).sum() + np.abs(prof[480:]).sum())
            np.testing.assert_allclose((np.abs(raw[:, 1:]) - np.abs(raw[:, :-1]))[300:360], diff[300:360], rtol=1e-5, atol=1.0)
            with open(os.path.join(data, "phone", "sessions", NAME, "landmarks.csv"), "rb") as a, \
                    open(os.path.join(back, "phone", "sessions", NAME, "landmarks.csv"), "rb") as b:
                self.assertEqual(a.read(), b.read())
            # 되돌린 rows.npz·align.json에 맞춘 지문이라 verify_sync·pairs·dataset이 낡은 입력으로 보지 않는다
            sync = hs_io.read_json(os.path.join(proc, "sync.json"))
            self.assertEqual(verify_sync.inputs(back, NAME), sync["inputs"])
            self.assertEqual(10.0, sync["lagMs"])
            with np.load(os.path.join(proc, "rows.npz")) as rows:
                self.assertEqual(hs_io.sha256(os.path.join(proc, "align.json")), str(rows["alignSha"]))
            self.assertTrue(os.path.exists(os.path.join(back, "watch", "recordings", REC, "rec.json")))
            self.assertFalse(os.path.exists(os.path.join(back, "watch", "recordings", REC, "part-000.wav")))

    def test_sessions_index_lists_each_exported_session(self):
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as out:
            fake_data(data)
            self.assertEqual([NAME], export_data.exportable(data))
            row = export_data.export_session(data, out, NAME)
            path = os.path.join(out, "sessions.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=export_data.COLUMNS, lineterminator="\n")
                w.writeheader()
                w.writerow(row)
            with open(path, encoding="utf-8") as f:
                got = list(csv.DictReader(f))
            self.assertEqual(NAME, got[0]["session"])
            self.assertEqual("desk", got[0]["armSupport"])
            with gzip.open(os.path.join(out, "sessions", NAME, "pairs.csv.gz"), "rt", encoding="utf-8") as f:
                self.assertEqual("frame", f.readline().split(",")[0])
            with open(os.path.join(out, "sessions", NAME, "align.json"), encoding="utf-8") as f:
                self.assertEqual(REC, json.load(f)["recording"])


if __name__ == "__main__":
    unittest.main()
