"""두 기기의 녹음·세션 폴더를 adb로 받는다(손 소나 계획 3·5, 수집 설계 §7.3·§8, 전송 설계 §4).

    python -I tools/pull.py [--data 데이터 폴더] [--adb adb 경로] [--watch SERIAL] [--phone SERIAL] [--only watch|phone]

- 기기는 --watch·--phone으로 serial을 주거나, `adb devices -l`의 model로 고른다: SM-L320 = 워치, SM-F971N = 폰
  (adb는 model의 '-'를 '_'로 적는다). 무선 adb면 serial과 포트가 켤 때마다 바뀐다.
- 받는 곳(SOURCES, 이 차례로):
  1. 폰 files/watch-recordings → watch/recordings: 워치가 블루투스로 폰에 넘긴 녹음(계획 5).
  2. 폰 files/sessions → phone/sessions
  3. 워치 files/recordings → watch/recordings: 1에서 같은 크기로 받은 파일은 다시 받지 않는다(같은 녹음은 한 번만, 크기 비교).
  폰을 먼저 받는 까닭: 폰은 USB라 빠르고, 워치에서는 그 뒤 없거나 크기가 다른 파일만 받으면 된다(그때는 워치 사본이 덮는다).
  워치가 adb에 없는 것은 문제가 아니다(폰만으로 두 기기의 데이터를 받는다). 폰이 없거나, --only로 고른 기기가 없으면 문제다.
- 원격 폴더 아래 폴더마다 파일 크기를 `stat -c '%s %n'`으로 보고, PC 사본이 없거나 크기가 다른 파일만 하나씩 같은 자리에 받는다
  (`adb pull <원격>/<폴더>/<파일> <PC 파일>`, 설계 §7.3). 받는 중인 `.part`(폰)·쓰다 남은 `.tmp`(rec.json.tmp 등)는 받지 않는다.
  폴더마다 맨 끝 파일(녹음 rec.json, 세션 meta.json — PC 도구는 그 파일이 있는 폴더만 녹음·세션으로 본다)은 다른 파일을 모두 같은
  크기로 받은 뒤에만 받고, 한 파일이 실패(끝 코드)하면 그 폴더는 거기서 멈춘다: 조각이 빠진 사본이 처리에 들어가지 않는다.
  받은 뒤 원격 크기를 다시 재서 PC 사본과 견준다. 받는 동안 크기가 바뀐(또는 새로 생긴) 파일은 녹음 중(폰은 받는 중)인 폴더라
  문제로 세지 않고 "다음에 다시 받음"으로 알린다(그 폴더의 맨 끝 파일도 다음으로 미룬다). 원격 크기가 그대로인데 PC 사본이 다르면
  받기 실패다(무선 adb pull이 끊기면 조각이 잘린다).
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
MODELS = {"watch": ("SM-L320", "워치"), "phone": ("SM-F971N", "폰")}  # 종류: (모델, 이름)
LEFTOVER = (".part", ".tmp")  # 폰이 받는 중인 파일, 쓰다 남은 임시 파일(rec.json.tmp·meta.json.tmp): 받지 않는다
SOURCES = (  # (기기, 원격 폴더, PC 폴더(데이터 폴더 기준), 이름, 받는 동안 크기가 바뀌는 까닭, 폴더의 맨 끝 파일)
    ("phone", APP + "/watch-recordings", ("watch", "recordings"), "폰의 워치 녹음", "폰이 받는 중", "rec.json"),
    ("phone", APP + "/sessions", ("phone", "sessions"), "폰 세션", "녹음 중", "meta.json"),
    ("watch", APP + "/recordings", ("watch", "recordings"), "워치 녹음", "녹음 중", "rec.json"),
)


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
    """{폴더 이름: {파일 이름: 크기}}. 원격 폴더가 없거나 비었으면 {}. 받는 중·쓰다 남은 파일(LEFTOVER)은 뺀다."""
    _, out = run([adb, "-s", serial, "shell", f"stat -c '%s %n' {remote}/*/*"])
    sizes = {}
    for line in out.splitlines():
        size, _, path = line.strip().partition(" ")
        if not size.isdigit() or not path.startswith(remote + "/"):
            continue  # "No such file or directory" 같은 줄
        folder, _, name = path[len(remote) + 1:].partition("/")
        if folder and name and "/" not in name and not name.endswith(LEFTOVER):
            sizes.setdefault(folder, {})[name] = int(size)
    return sizes


def local_sizes(folder):
    if not os.path.isdir(folder):
        return {}
    return {n: os.path.getsize(os.path.join(folder, n)) for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))}


def differs(remote, local):
    """원격 파일 중 PC에 없거나 크기가 다른 이름들."""
    return sorted(n for n, s in remote.items() if local.get(n) != s)


def pull_files(run, adb, serial, remote, parent, name, files, last):
    """PC에 없거나 크기가 다른 파일만 하나씩 받는다. 맨 끝 파일 last는 다른 파일이 모두 원격과 같은 크기로 있을 때만 받고,
    한 파일이 실패(끝 코드)하면 거기서 멈춘다. (끝 코드, 출력을 이은 것, last를 다음으로 미뤘는지)."""
    folder = os.path.join(parent, name)
    os.makedirs(folder, exist_ok=True)
    wanted = differs(files, local_sizes(folder))
    outs = []
    for f in (n for n in wanted if n != last):
        code, out = run([adb, "-s", serial, "pull", f"{remote}/{name}/{f}", os.path.join(folder, f)])
        outs.append(out)
        if code != 0:
            return code, "\n".join(outs), last in wanted
    if last not in wanted:
        return 0, "\n".join(outs), False
    if differs({n: s for n, s in files.items() if n != last}, local_sizes(folder)):
        return 0, "\n".join(outs), True  # 앞 파일이 아직 다르다(받는 동안 바뀜·잘림): 맨 끝 파일은 다음에
    code, out = run([adb, "-s", serial, "pull", f"{remote}/{name}/{last}", os.path.join(folder, last)])
    outs.append(out)
    return code, "\n".join(outs), False


def pull_source(run, adb, serial, source, data):
    """받는 곳 하나(SOURCES의 한 줄): (보고 줄, 문제)."""
    _, remote, local, label, busy, last = source
    parent = os.path.join(data, *local)
    os.makedirs(parent, exist_ok=True)
    sizes = remote_sizes(run, adb, serial, remote)
    if not sizes:
        return [f"{label}({serial}): 받을 폴더 없음"], []
    wanted = [name for name in sorted(sizes) if differs(sizes[name], local_sizes(os.path.join(parent, name)))]
    lines = [f"{label}({serial}): 폴더 {len(sizes)}개, 새로 받을 것 {len(wanted)}개" + (f": {', '.join(wanted)}" if wanted else "")]
    problems = []
    results = {name: pull_files(run, adb, serial, remote, parent, name, sizes[name], last) for name in wanted}
    after = remote_sizes(run, adb, serial, remote) if wanted else {}  # 받은 뒤 다시 잰 원격 크기
    for name in wanted:
        code, out, deferred = results[name]
        now = after.get(name, sizes[name])
        bad = differs(now, local_sizes(os.path.join(parent, name)))
        growing = [n for n in bad if sizes[name].get(n) != now.get(n)]  # 받는 동안 크기가 바뀐 파일(녹음 중·받는 중)
        broken = [n for n in bad if n not in growing]
        if deferred and growing and last in broken:
            broken.remove(last)  # 바뀌는 파일 때문에 미뤘다: 실패가 아니라 다음에 함께 받는다
        if code != 0 or broken:
            problems.append(f"{label} {name}: 받기 실패(끝 코드 {code}, 크기가 다른 파일 {', '.join(broken) or '없음'}) {out.strip()[-200:]}")
        elif growing:
            later = f", {last}은 다음에" if deferred else ""
            lines.append(f"{label} {name}: {busy}이라 다음에 다시 받음(받는 동안 크기가 바뀐 파일 {', '.join(growing)}{later})")
    return lines, problems


def pull_all(data, adb, run, watch=None, phone=None, only=None):
    """(보고 줄, 문제). serial을 주지 않은 기기는 모델로 찾는다. 폰부터 받는다(SOURCES 차례)."""
    lines, problems = [], []
    found = devices(run, adb) if watch is None or phone is None else {}
    serials = {"watch": watch or found.get(MODELS["watch"][0]), "phone": phone or found.get(MODELS["phone"][0])}
    for kind in ("phone", "watch"):
        if (only and kind != only) or serials[kind] is not None:
            continue
        model, name = MODELS[kind]
        message = f"{name}({model})가 adb에 없음(무선 디버깅을 켰는지, `adb devices -l`을 본다)"
        if kind == "watch" and not only:
            lines.append(message + " — 폰이 받은 워치 녹음만 받음")
        else:
            problems.append(message)
    for source in SOURCES:
        kind = source[0]
        if (only and kind != only) or serials[kind] is None:
            continue
        more, bad = pull_source(run, adb, serials[kind], source, data)
        lines += more
        problems += bad
    return lines, problems


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="폰의 세션·받은 워치 녹음과 워치 녹음을 adb로 받는다")
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
