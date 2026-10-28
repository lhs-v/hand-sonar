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
PHONE_ONLY_OUT = """List of devices attached
PHONE0001 device product:fold model:SM_F971N device:f transport_id:4
"""
WATCH_ONLY_OUT = """List of devices attached
adb-AAAA._adb-tls-connect._tcp device product:wear model:SM_L320 device:w transport_id:3
"""
WATCH = "adb-AAAA._adb-tls-connect._tcp"
PHONE = "10.0.0.5:40001"
WATCH_DIR = pull.APP + "/recordings"
PHONE_DIR = pull.APP + "/sessions"
INBOX_DIR = pull.APP + "/watch-recordings"


class FakeAdb:
    """adb 대신 부르는 실행기. 명령을 적어 두고, stat에는 원격 파일 크기를, pull에는 그 크기대로 PC에 파일을 만들어 준다.
    pull은 폴더째(`pull <원격>/<폴더> <부모>`)와 파일 하나(`pull <원격>/<폴더>/<파일> <PC 파일>`)를 둘 다 흉내 낸다."""

    def __init__(self, remote, devices_out=DEVICES_OUT):
        self.remote = remote  # {serial: {원격 폴더: {폴더: {파일: 크기}}}}
        self.devices_out = devices_out
        self.pulled = []

    def __call__(self, args):
        if args[1:] == ["devices", "-l"]:
            return 0, self.devices_out
        serial, command = args[2], args[3]
        if command == "shell":
            base = args[4].split()[-1][: -len("/*/*")]
            tree = self.remote[serial].get(base, {})
            if not tree:
                return 1, f"stat: '{base}/*/*': No such file or directory"
            return 0, "\n".join(f"{size} {base}/{folder}/{name}" for folder, files in tree.items() for name, size in files.items())
        if command == "pull":
            src, dest = args[4], args[5]
            self.pulled.append((serial, src))
            head, last = src.rsplit("/", 1)
            if head in self.remote[serial]:  # 폴더째: dest는 부모 폴더
                os.makedirs(os.path.join(dest, last), exist_ok=True)
                for name, size in self.remote[serial][head][last].items():
                    write(os.path.join(dest, last, name), size)
            else:  # 파일 하나: dest는 PC 파일
                base, folder = head.rsplit("/", 1)
                write(dest, self.remote[serial][base][folder][last])
            return 0, "1 file pulled"
        raise AssertionError(f"모르는 명령 {args}")


def write(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def tree(root):
    """{폴더/파일: 크기}."""
    out = {}
    for folder in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for name in sorted(os.listdir(os.path.join(root, folder))):
            out[f"{folder}/{name}"] = os.path.getsize(os.path.join(root, folder, name))
    return out


class PullTest(unittest.TestCase):
    def test_picks_devices_by_model_and_pulls_only_new_or_changed_folders(self):
        fake = FakeAdb({
            WATCH: {WATCH_DIR: {"20990101-000000": {"rec.json": 10, "part-000.wav": 300},
                                "20990101-001000": {"rec.json": 12, "part-000.wav": 400}}},
            PHONE: {PHONE_DIR: {"20990101-000005": {"meta.json": 20, "video.mp4": 500}}},
        })
        with tempfile.TemporaryDirectory() as data:
            write(os.path.join(data, "watch", "recordings", "20990101-000000", "rec.json"), 10)  # 이미 같은 크기로 받음
            write(os.path.join(data, "watch", "recordings", "20990101-000000", "part-000.wav"), 300)
            write(os.path.join(data, "phone", "sessions", "20990101-000005", "meta.json"), 20)
            write(os.path.join(data, "phone", "sessions", "20990101-000005", "video.mp4"), 123)  # 지난번에 끊겨 잘림
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            # 폰을 먼저 받는다(계획 5). 폰에 워치 녹음 폴더가 없으면 그 줄은 "받을 폴더 없음"이다
            self.assertEqual([(PHONE, PHONE_DIR + "/20990101-000005"), (WATCH, WATCH_DIR + "/20990101-001000")], fake.pulled)
            self.assertEqual(500, os.path.getsize(os.path.join(data, "phone", "sessions", "20990101-000005", "video.mp4")))
            self.assertTrue(any("폰의 워치 녹음" in line and "받을 폴더 없음" in line for line in lines), lines)

    def test_recording_in_progress_is_not_a_failure_but_a_cut_pull_is(self):
        # 녹음 중인 폴더는 받는 동안 조각이 커져 받은 사본이 받은 뒤 원격 크기와 다르다: 실패가 아니라 "녹음 중"으로 알린다.
        # 원격 크기가 그대로인데 사본이 짧으면(무선 pull이 끊김) 받기 실패다.
        remote = {WATCH_DIR: {"20990101-000000": {"rec.json": 10, "part-000.wav": 300}, "20990101-001000": {"rec.json": 12, "part-000.wav": 400}}}

        class Growing(FakeAdb):
            def __call__(self, args):
                folder = args[4].rsplit("/", 1)[1] if args[3:4] == ["pull"] else None
                if folder == "20990101-000000":
                    remote[WATCH_DIR][folder]["part-000.wav"] += 4_800  # stat 뒤 pull 전에도 녹음이 이어진다
                code, out = super().__call__(args)
                if folder == "20990101-000000":
                    remote[WATCH_DIR][folder]["part-000.wav"] += 4_800  # 받은 뒤에도 이어진다
                elif folder:
                    with open(os.path.join(args[5], folder, "part-000.wav"), "wb") as f:
                        f.write(b"\0" * 100)  # 끊겨 잘린 사본
                return code, out

        fake = Growing({WATCH: remote})
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake, only="watch")
            self.assertEqual(1, len(problems), problems)
            self.assertIn("20990101-001000", problems[0])
            self.assertTrue(any("20990101-000000" in line and "녹음 중" in line for line in lines), lines)

    def test_phone_alone_brings_the_watch_recordings_in_the_watch_layout(self):
        # 워치를 adb로 붙일 수 없는 PC(전송 설계 §4): 폰의 watch-recordings가 watch/recordings로 같은 모양으로 온다.
        # 폰이 받는 중인 .part는 받지 않고, 워치가 없는 것은 문제가 아니다.
        fake = FakeAdb({"PHONE0001": {
            INBOX_DIR: {"20990101-000000": {"part-000.wav": 300, "rec.json": 10},
                        "20990101-001000": {"part-000.wav": 400, "part-001.wav.part": 50}},
            PHONE_DIR: {"20990101-000005": {"meta.json": 20}},
        }}, devices_out=PHONE_ONLY_OUT)
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            self.assertEqual(
                {"20990101-000000/part-000.wav": 300, "20990101-000000/rec.json": 10, "20990101-001000/part-000.wav": 400},
                tree(os.path.join(data, "watch", "recordings")),
            )
            self.assertEqual({"20990101-000005/meta.json": 20}, tree(os.path.join(data, "phone", "sessions")))
            self.assertFalse(any(src.endswith(".part") for _, src in fake.pulled), fake.pulled)
            self.assertIn(("PHONE0001", INBOX_DIR + "/20990101-000000/rec.json"), fake.pulled)  # 파일마다 받는다
            self.assertTrue(any("워치(SM-L320)가 adb에 없음" in line for line in lines), lines)
            # 다시 돌리면 받을 것이 없다
            fake.pulled.clear()
            self.assertEqual(([], []), (pull.pull_all(data, "adb", fake)[1], fake.pulled))

    def test_cut_file_from_the_phone_is_a_failure(self):
        # 파일마다 받을 때도 원격 크기가 그대로인데 사본이 짧으면 받기 실패다
        class Cut(FakeAdb):
            def __call__(self, args):
                code, out = super().__call__(args)
                if args[3:4] == ["pull"] and args[4].endswith("/part-000.wav"):
                    with open(args[5], "wb") as f:
                        f.write(b"\0" * 100)
                return code, out

        fake = Cut({"PHONE0001": {INBOX_DIR: {"20990101-000000": {"part-000.wav": 300, "rec.json": 10}}}}, devices_out=PHONE_ONLY_OUT)
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake, only="phone")
            self.assertEqual(1, len(problems), (lines, problems))
            self.assertIn("20990101-000000", problems[0])
            self.assertIn("part-000.wav", problems[0])

    def test_both_attached_pulls_each_recording_once_by_size(self):
        # 같은 녹음이 폰(블루투스로 받은 것)과 워치에 다 있으면 한 번만 받는다: 폰에서 먼저 받고, 워치에서는 크기가 다른 폴더만.
        same = {"part-000.wav": 300, "rec.json": 10}
        fake = FakeAdb({
            PHONE: {INBOX_DIR: {"20990101-000000": dict(same), "20990101-002000": {"part-000.wav": 300}}},
            WATCH: {WATCH_DIR: {"20990101-000000": dict(same),
                                "20990101-001000": {"part-000.wav": 500, "rec.json": 12},
                                "20990101-002000": {"part-000.wav": 300, "part-001.wav": 200, "rec.json": 11}}},
        })
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            self.assertEqual([
                (PHONE, INBOX_DIR + "/20990101-000000/part-000.wav"),
                (PHONE, INBOX_DIR + "/20990101-000000/rec.json"),
                (PHONE, INBOX_DIR + "/20990101-002000/part-000.wav"),
                (WATCH, WATCH_DIR + "/20990101-001000"),  # 폰에 없음
                (WATCH, WATCH_DIR + "/20990101-002000"),  # 폰에서는 일부만 받았음
            ], fake.pulled)
            self.assertEqual(
                {"20990101-000000/part-000.wav": 300, "20990101-000000/rec.json": 10,
                 "20990101-001000/part-000.wav": 500, "20990101-001000/rec.json": 12,
                 "20990101-002000/part-000.wav": 300, "20990101-002000/part-001.wav": 200, "20990101-002000/rec.json": 11},
                tree(os.path.join(data, "watch", "recordings")),
            )

    def test_missing_phone_is_a_problem_but_missing_watch_is_not(self):
        fake = FakeAdb({WATCH: {WATCH_DIR: {"20990101-000000": {"rec.json": 10}}}}, devices_out=WATCH_ONLY_OUT)
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual(1, len(problems), problems)
            self.assertIn("폰(SM-F971N)가 adb에 없음", problems[0])
            self.assertEqual([(WATCH, WATCH_DIR + "/20990101-000000")], fake.pulled)  # 워치는 그대로 받는다
            _, problems = pull.pull_all(data, "adb", FakeAdb({}, devices_out=PHONE_ONLY_OUT.replace("PHONE0001", "P1")), only="watch")
            self.assertEqual(1, len(problems), problems)  # --only로 고른 기기가 없으면 문제다
            self.assertIn("워치(SM-L320)가 adb에 없음", problems[0])


if __name__ == "__main__":
    unittest.main()
