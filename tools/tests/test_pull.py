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
    """adb 대신 부르는 실행기. 명령을 적어 두고, stat에는 원격 파일 크기를, 파일 하나 받기(`pull <원격>/<폴더>/<파일> <PC 파일>`)에는
    그 크기대로 PC에 파일을 만들어 준다. fail에 든 원격 경로는 끝 코드 1로 실패한다."""

    def __init__(self, remote, devices_out=DEVICES_OUT, fail=()):
        self.remote = remote  # {serial: {원격 폴더: {폴더: {파일: 크기}}}}
        self.devices_out = devices_out
        self.fail = set(fail)
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
            if src in self.fail:
                return 1, "adb: error: failed to copy"
            head, name = src.rsplit("/", 1)
            base, folder = head.rsplit("/", 1)
            write(dest, self.remote[serial][base][folder][name])
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


def short(pulled):
    """[(serial, 원격 경로)] → ["<폴더>/<파일>"]."""
    return ["/".join(src.rsplit("/", 2)[-2:]) for _, src in pulled]


class PullTest(unittest.TestCase):
    def test_picks_devices_by_model_and_pulls_only_new_or_changed_files(self):
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
            # 폰을 먼저, 크기가 다른 파일만 하나씩, rec.json(세션은 meta.json)은 그 폴더의 맨 끝에 받는다
            self.assertEqual([(PHONE, PHONE_DIR + "/20990101-000005/video.mp4"),
                              (WATCH, WATCH_DIR + "/20990101-001000/part-000.wav"),
                              (WATCH, WATCH_DIR + "/20990101-001000/rec.json")], fake.pulled)
            self.assertEqual(500, os.path.getsize(os.path.join(data, "phone", "sessions", "20990101-000005", "video.mp4")))
            self.assertTrue(any("폰의 워치 녹음" in line and "받을 폴더 없음" in line for line in lines), lines)

    def test_recording_in_progress_is_not_a_failure_but_a_cut_pull_is(self):
        # 녹음 중인 폴더는 받는 동안 조각이 커져 받은 사본이 받은 뒤 원격 크기와 다르다: 실패가 아니라 "녹음 중"으로 알린다.
        # 원격 크기가 그대로인데 사본이 짧으면(무선 pull이 끊김) 받기 실패다. 둘 다 그 폴더의 rec.json은 받지 않고 다음으로 미룬다.
        remote = {WATCH_DIR: {"20990101-000000": {"rec.json": 10, "part-000.wav": 300}, "20990101-001000": {"rec.json": 12, "part-000.wav": 400}}}

        class Growing(FakeAdb):
            def __call__(self, args):
                src = args[4] if args[3:4] == ["pull"] else ""
                live = src.endswith("20990101-000000/part-000.wav")
                if live:
                    remote[WATCH_DIR]["20990101-000000"]["part-000.wav"] += 4_800  # stat 뒤 pull 전에도 녹음이 이어진다
                code, out = super().__call__(args)
                if live:
                    remote[WATCH_DIR]["20990101-000000"]["part-000.wav"] += 4_800  # 받은 뒤에도 이어진다
                elif src.endswith("20990101-001000/part-000.wav"):
                    with open(args[5], "wb") as f:
                        f.write(b"\0" * 100)  # 끊겨 잘린 사본
                return code, out

        fake = Growing({WATCH: remote})
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake, only="watch")
            self.assertEqual(1, len(problems), problems)
            self.assertIn("20990101-001000", problems[0])
            self.assertTrue(any("20990101-000000" in line and "녹음 중" in line for line in lines), lines)
            self.assertEqual(["20990101-000000/part-000.wav", "20990101-001000/part-000.wav"], short(fake.pulled))  # rec.json은 둘 다 미룸

    def test_a_failed_part_pull_stops_the_folder_before_rec_json(self):
        # 조각 하나를 못 받으면 그 폴더는 거기서 멈춘다: rec.json이 있는(PC 도구가 녹음으로 보는) 빠진 사본을 만들지 않는다
        fake = FakeAdb({"PHONE0001": {INBOX_DIR: {"20990101-000000": {"part-000.wav": 300, "part-001.wav": 300, "rec.json": 10}}}},
                       devices_out=PHONE_ONLY_OUT, fail=[INBOX_DIR + "/20990101-000000/part-000.wav"])
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake, only="phone")
            self.assertEqual(1, len(problems), (lines, problems))
            self.assertIn("끝 코드 1", problems[0])
            self.assertEqual(["20990101-000000/part-000.wav"], short(fake.pulled))
            self.assertFalse(os.path.exists(os.path.join(data, "watch", "recordings", "20990101-000000", "rec.json")))

    def test_cut_file_from_the_phone_is_a_failure(self):
        # 파일마다 받을 때도 원격 크기가 그대로인데 사본이 짧으면 받기 실패다(그 폴더의 rec.json은 받지 않는다)
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
            self.assertEqual(["20990101-000000/part-000.wav"], short(fake.pulled))

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

    def test_both_attached_pulls_each_file_once_by_size(self):
        # 같은 녹음이 폰(블루투스로 받은 것)과 워치에 다 있으면 한 번만 받는다: 폰에서 먼저 받고, 워치에서는 크기가 다른 파일만.
        # 폰에 일부만 있는 녹음도 워치에서 빠진 파일만 받는다(폴더째 다시 받지 않음). 워치의 쓰다 남은 rec.json.tmp는 받지 않는다.
        same = {"part-000.wav": 300, "rec.json": 10}
        fake = FakeAdb({
            PHONE: {INBOX_DIR: {"20990101-000000": dict(same), "20990101-002000": {"part-000.wav": 300}}},
            WATCH: {WATCH_DIR: {"20990101-000000": dict(same),
                                "20990101-001000": {"part-000.wav": 500, "rec.json": 12, "rec.json.tmp": 5},
                                "20990101-002000": {"part-000.wav": 300, "part-001.wav": 200, "rec.json": 11}}},
        })
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            self.assertEqual([
                (PHONE, INBOX_DIR + "/20990101-000000/part-000.wav"),
                (PHONE, INBOX_DIR + "/20990101-000000/rec.json"),
                (PHONE, INBOX_DIR + "/20990101-002000/part-000.wav"),
                (WATCH, WATCH_DIR + "/20990101-001000/part-000.wav"),  # 폰에 없음
                (WATCH, WATCH_DIR + "/20990101-001000/rec.json"),
                (WATCH, WATCH_DIR + "/20990101-002000/part-001.wav"),  # 폰에서 받은 part-000은 다시 받지 않는다
                (WATCH, WATCH_DIR + "/20990101-002000/rec.json"),
            ], fake.pulled)
            self.assertEqual(
                {"20990101-000000/part-000.wav": 300, "20990101-000000/rec.json": 10,
                 "20990101-001000/part-000.wav": 500, "20990101-001000/rec.json": 12,
                 "20990101-002000/part-000.wav": 300, "20990101-002000/part-001.wav": 200, "20990101-002000/rec.json": 11},
                tree(os.path.join(data, "watch", "recordings")),
            )

    def test_sensor_csv_parts_come_along_before_rec_json_and_meta_json(self):
        # 계획 6의 센서 기록도 이름을 가리지 않고 받는다: 워치 sensors-NNN.csv, 폰 phone_sensors-NNN.csv. 맨 끝 파일(rec.json·meta.json)은
        # 그 뒤에 받고, 쓰다 남은 .tmp는 받지 않는다.
        fake = FakeAdb({
            PHONE: {INBOX_DIR: {"20990101-000000": {"part-000.wav": 300, "sensors-000.csv": 70, "sensors-001.csv": 30, "rec.json": 10}},
                    PHONE_DIR: {"20990101-000005": {"meta.json": 20, "phone_sensors-000.csv": 40, "landmarks.csv": 50}}},
            WATCH: {WATCH_DIR: {"20990101-001000": {"part-000.wav": 300, "sensors-000.csv": 70, "sensors-001.csv.tmp": 5, "rec.json": 10}}},
        })
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual([], problems, lines)
            self.assertEqual([
                "20990101-000000/part-000.wav", "20990101-000000/sensors-000.csv", "20990101-000000/sensors-001.csv", "20990101-000000/rec.json",
                "20990101-000005/landmarks.csv", "20990101-000005/phone_sensors-000.csv", "20990101-000005/meta.json",
                "20990101-001000/part-000.wav", "20990101-001000/sensors-000.csv", "20990101-001000/rec.json",
            ], short(fake.pulled))
            self.assertNotIn("20990101-001000/sensors-001.csv.tmp", tree(os.path.join(data, "watch", "recordings")))

    def test_missing_phone_is_a_problem_but_missing_watch_is_not(self):
        fake = FakeAdb({WATCH: {WATCH_DIR: {"20990101-000000": {"rec.json": 10}}}}, devices_out=WATCH_ONLY_OUT)
        with tempfile.TemporaryDirectory() as data:
            lines, problems = pull.pull_all(data, "adb", fake)
            self.assertEqual(1, len(problems), problems)
            self.assertIn("폰(SM-F971N)가 adb에 없음", problems[0])
            self.assertEqual([(WATCH, WATCH_DIR + "/20990101-000000/rec.json")], fake.pulled)  # 워치는 그대로 받는다
            _, problems = pull.pull_all(data, "adb", FakeAdb({}, devices_out=PHONE_ONLY_OUT.replace("PHONE0001", "P1")), only="watch")
            self.assertEqual(1, len(problems), problems)  # --only로 고른 기기가 없으면 문제다
            self.assertIn("워치(SM-L320)가 adb에 없음", problems[0])


if __name__ == "__main__":
    unittest.main()
