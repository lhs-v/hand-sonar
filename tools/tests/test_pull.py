import os
import tempfile
import unittest

import pull

DEVICES_OUT = """List of devices attached
adb-AAAA._adb-tls-connect._tcp device product:wear model:SM_L320 device:w transport_id:3
10.0.0.5:40001 device product:fold model:SM_F971N device:f transport_id:4
10.0.0.9:5555  offline
other-serial   device product:x model:Pixel_9 device:p transport_id:5
"""


class FakeAdb:
    """adb 대신 부르는 실행기. 명령을 적어 두고, stat에는 원격 파일 크기를, pull에는 그 크기대로 PC에 파일을 만들어 준다."""

    def __init__(self, remote):
        self.remote = remote  # {serial: {원격 폴더: {폴더: {파일: 크기}}}}
        self.pulled = []

    def __call__(self, args):
        if args[1:] == ["devices", "-l"]:
            return 0, DEVICES_OUT
        serial, command = args[2], args[3]
        if command == "shell":
            base = args[4].split()[-1][: -len("/*/*")]
            tree = self.remote[serial].get(base, {})
            if not tree:
                return 1, f"stat: '{base}/*/*': No such file or directory"
            return 0, "\n".join(f"{size} {base}/{folder}/{name}" for folder, files in tree.items() for name, size in files.items())
        if command == "pull":
            src, parent = args[4], args[5]
            base, folder = src.rsplit("/", 1)
            self.pulled.append((serial, src))
            os.makedirs(os.path.join(parent, folder), exist_ok=True)
            for name, size in self.remote[serial][base][folder].items():
                with open(os.path.join(parent, folder, name), "wb") as f:
                    f.write(b"\0" * size)
            return 0, "1 file pulled"
        raise AssertionError(f"모르는 명령 {args}")


def write(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


class PullTest(unittest.TestCase):
    def test_picks_devices_by_model_and_pulls_only_new_or_changed_folders(self):
        watch_dir = pull.APP + "/recordings"
        phone_dir = pull.APP + "/sessions"
        fake = FakeAdb({
            "adb-AAAA._adb-tls-connect._tcp": {watch_dir: {"20990101-000000": {"rec.json": 10, "part-000.wav": 300},
                                                          "20990101-001000": {"rec.json": 12, "part-000.wav": 400}}},
            "10.0.0.5:40001": {phone_dir: {"20990101-000005": {"meta.json": 20, "video.mp4": 500}}},
        })
        with tempfile.TemporaryDirectory() as data:
            write(os.path.join(data, "watch", "recordings", "20990101-000000", "rec.json"), 10)  # 이미 같은 크기로 받음
            write(os.path.join(data, "watch", "recordings", "20990101-000000", "part-000.wav"), 300)
            write(os.path.join(data, "phone", "sessions", "20990101-000005", "meta.json"), 20)
            write(os.path.join(data, "phone", "sessions", "20990101-000005", "video.mp4"), 123)  # 지난번에 끊겨 잘림
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            self.assertEqual([("adb-AAAA._adb-tls-connect._tcp", watch_dir + "/20990101-001000"),
                              ("10.0.0.5:40001", phone_dir + "/20990101-000005")], fake.pulled)
            self.assertEqual(500, os.path.getsize(os.path.join(data, "phone", "sessions", "20990101-000005", "video.mp4")))

    def test_recording_in_progress_is_not_a_failure_but_a_cut_pull_is(self):
        # 녹음 중인 폴더는 받는 동안 조각이 커져 받은 사본이 받은 뒤 원격 크기와 다르다: 실패가 아니라 "녹음 중"으로 알린다.
        # 원격 크기가 그대로인데 사본이 짧으면(무선 pull이 끊김) 받기 실패다.
        watch_dir = pull.APP + "/recordings"
        remote = {watch_dir: {"20990101-000000": {"rec.json": 10, "part-000.wav": 300}, "20990101-001000": {"rec.json": 12, "part-000.wav": 400}}}

        class Growing(FakeAdb):
            def __call__(self, args):
                folder = args[4].rsplit("/", 1)[1] if args[3:4] == ["pull"] else None
                if folder == "20990101-000000":
                    remote[watch_dir][folder]["part-000.wav"] += 4_800  # stat 뒤 pull 전에도 녹음이 이어진다
                code, out = super().__call__(args)
                if folder == "20990101-000000":
                    remote[watch_dir][folder]["part-000.wav"] += 4_800  # 받은 뒤에도 이어진다
                elif folder:
                    with open(os.path.join(args[5], folder, "part-000.wav"), "wb") as f:
                        f.write(b"\0" * 100)  # 끊겨 잘린 사본
                return code, out

        fake = Growing({"adb-AAAA._adb-tls-connect._tcp": remote})
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake, only="watch")
            self.assertEqual(1, len(problems), problems)
            self.assertIn("20990101-001000", problems[0])
            self.assertTrue(any("20990101-000000" in line and "녹음 중" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
