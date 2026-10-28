"""두 기기의 녹음·세션 폴더를 adb로 받는다(손 소나 계획 3, 설계 §7.3·§8).

    python -I tools/pull.py [--data 데이터 폴더] [--adb adb 경로] [--watch SERIAL] [--phone SERIAL] [--only watch|phone]

- 기기는 --watch·--phone으로 serial을 주거나, `adb devices -l`의 model로 고른다: SM-L320 = 워치, SM-F971N = 폰
  (adb는 model의 '-'를 '_'로 적는다). 두 기기 모두 무선 adb라 serial과 포트가 켤 때마다 바뀐다.
- 원격 폴더(워치 files/recordings, 폰 files/sessions) 아래 폴더마다 파일 크기를 `stat -c '%s %n'`으로 보고, PC 사본이 없거나
  크기가 다른 폴더만 `adb pull`로 같은 자리에 덮어 받는다(설계 §7.3). 받은 뒤 원격 크기를 다시 재서 PC 사본과 견준다.
  받는 동안 크기가 바뀐(또는 새로 생긴) 파일은 녹음 중인 폴더라 문제로 세지 않고 "녹음 중이라 다음에 다시 받음"으로 알린다.
  원격 크기가 그대로인데 PC 사본이 다르면 받기 실패다(무선 adb pull이 끊기면 조각이 잘린다).
- adb 실행은 run(args) -> (끝 코드, 출력)으로 주입한다. 시험은 가짜 실행기를 쓰고 실제 기기를 쓰지 않는다.
문제가 하나라도 있으면 끝 코드 1로 끝난다.
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402

APP = "/sdcard/Android/data/com.watchpose.handsonar/files"
DEVICES = {  # 종류: (모델, 원격 폴더, PC 폴더(데이터 폴더 기준), 이름)
    "watch": ("SM-L320", APP + "/recordings", ("watch", "recordings"), "워치"),
    "phone": ("SM-F971N", APP + "/sessions", ("phone", "sessions"), "폰"),
}


def default_adb():
    """PATH의 adb, 없으면 안드로이드 SDK 기본 자리(%LOCALAPPDATA%\\Android\\Sdk\\platform-tools\\adb.exe)."""
    return shutil.which("adb") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe")


def subprocess_run(args):
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def devices(run, adb):
    """{모델: serial}. `adb devices -l`에서 상태가 device인 줄만. 같은 모델이 두 번 보이면(mDNS 이름과 IP:포트) 앞의 것."""
    _, out = run([adb, "devices", "-l"])
    found = {}
    for line in out.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2 or fields[1] != "device":
            continue
        model = next((f.split(":", 1)[1].replace("_", "-") for f in fields[2:] if f.startswith("model:")), None)
        if model and model not in found:
            found[model] = fields[0]
    return found


def remote_sizes(run, adb, serial, remote):
    """{폴더 이름: {파일 이름: 크기}}. 원격 폴더가 없거나 비었으면 {}."""
    _, out = run([adb, "-s", serial, "shell", f"stat -c '%s %n' {remote}/*/*"])
    sizes = {}
    for line in out.splitlines():
        size, _, path = line.strip().partition(" ")
        if not size.isdigit() or not path.startswith(remote + "/"):
            continue  # "No such file or directory" 같은 줄
        folder, _, name = path[len(remote) + 1:].partition("/")
        if folder and name and "/" not in name:
            sizes.setdefault(folder, {})[name] = int(size)
    return sizes


def local_sizes(folder):
    if not os.path.isdir(folder):
        return {}
    return {n: os.path.getsize(os.path.join(folder, n)) for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))}


def differs(remote, local):
    """원격 파일 중 PC에 없거나 크기가 다른 이름들."""
    return sorted(n for n, s in remote.items() if local.get(n) != s)


def pull_device(run, adb, serial, kind, data):
    """기기 하나: (보고 줄, 문제)."""
    model, remote, local, label = DEVICES[kind]
    parent = os.path.join(data, *local)
    os.makedirs(parent, exist_ok=True)
    sizes = remote_sizes(run, adb, serial, remote)
    if not sizes:
        return [f"{label}({serial}): 받을 폴더 없음"], []
    wanted = [name for name in sorted(sizes) if differs(sizes[name], local_sizes(os.path.join(parent, name)))]
    lines = [f"{label}({serial}): 폴더 {len(sizes)}개, 새로 받을 것 {len(wanted)}개" + (f": {', '.join(wanted)}" if wanted else "")]
    problems = []
    results = {name: run([adb, "-s", serial, "pull", f"{remote}/{name}", parent]) for name in wanted}
    after = remote_sizes(run, adb, serial, remote) if wanted else {}  # 받은 뒤 다시 잰 원격 크기
    for name in wanted:
        code, out = results[name]
        now = after.get(name, sizes[name])
        bad = differs(now, local_sizes(os.path.join(parent, name)))
        growing = [n for n in bad if sizes[name].get(n) != now.get(n)]  # 받는 동안 크기가 바뀐 파일(녹음 중)
        broken = [n for n in bad if n not in growing]
        if code != 0 or broken:
            problems.append(f"{label} {name}: 받기 실패(끝 코드 {code}, 크기가 다른 파일 {', '.join(broken) or '없음'}) {out.strip()[-200:]}")
        elif growing:
            lines.append(f"{label} {name}: 녹음 중이라 다음에 다시 받음(받는 동안 크기가 바뀐 파일 {', '.join(growing)})")
    return lines, problems


def pull_all(data, adb, run, watch=None, phone=None, only=None):
    """(보고 줄, 문제). serial을 주지 않은 기기는 모델로 찾는다."""
    lines, problems = [], []
    found = devices(run, adb) if watch is None or phone is None else {}
    for kind, serial in (("watch", watch), ("phone", phone)):
        if only and kind != only:
            continue
        model, _, _, label = DEVICES[kind]
        serial = serial or found.get(model)
        if serial is None:
            problems.append(f"{label}({model})가 adb에 없음(무선 디버깅을 켰는지, `adb devices -l`을 본다)")
            continue
        more, bad = pull_device(run, adb, serial, kind, data)
        lines += more
        problems += bad
    return lines, problems


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="워치 녹음과 폰 세션을 adb로 받는다")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA, help="데이터 폴더(watch·phone 아래로 받는다)")
    parser.add_argument("--adb", default=default_adb())
    parser.add_argument("--watch", help="워치 serial(없으면 모델 SM-L320으로 찾음)")
    parser.add_argument("--phone", help="폰 serial(없으면 모델 SM-F971N으로 찾음)")
    parser.add_argument("--only", choices=["watch", "phone"])
    args = parser.parse_args()
    lines, problems = pull_all(args.data, args.adb, subprocess_run, args.watch, args.phone, args.only)
    print("\n".join(lines))
    if problems:
        print("문제:")
        for p in problems:
            print("  - " + p)
        return 1
    print("문제 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
