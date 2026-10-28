"""센서 기록·녹음 볼륨·세션 정보를 점검한다(손 소나 계획 6, 기기 확인용).

    python -I tools/check_sensors.py <폴더> [<폴더> …]

- 폴더에 rec.json이 있으면 워치 녹음(sensors-NNN.csv, rec.json volume), meta.json이 있으면 폰 세션(phone_sensors-NNN.csv, meta.json
  context·rewear)으로 본다. 폴더 하나를 읽지 못해도 문제 한 줄로 적고 뒤 폴더를 점검한다.
- 보고: 기기가 준 센서(이름·제조사·최대 표본률·분해능·건 주기)와 기기에 없던 종류, 같은 종류의 다른 센서(듣지 않음), 시계 판정(연속 센서의
  받은 시각 − 이벤트 시각 1% 값, hs_io.SensorLog.event_clock), 종류마다 줄 수·실제 표본률(이벤트 시각)·받은 시각 − 이벤트 시각(1%·중앙·99%·최대
  ms)·빈틈 수(이웃 이벤트 간격이 중앙 간격의 GAP_FACTOR배를 넘는 곳, 바뀔 때만 내는 light는 세지 않음), 버린 줄 수, 조각 수와 크기.
  워치는 녹음 볼륨(고른 값·맞추려던 값·녹음 동안·원래·되돌린 뒤, 걸지 않은 까닭·외부 출력과 그 바뀜·되돌리기를 미룬 까닭·녹음 중 볼륨이 바뀐
  번수·앞 녹음이 남긴 기록, 녹음이 끝나지 않아 되돌린 기록이 없음), 폰은 세션 정보와 "워치 다시 참".
- 문제(끝 코드 1): 읽지 못함, 센서 기록 오류(json sensors.error), json이 적은 조각이 없음, 기기가 준 센서인데 줄이 없음, 이벤트 시각이 elapsedNs
  기준이 아님(그때 hs_io.SensorLog.ns는 이벤트 시각 + 지연 1% 값을 쓴다), 녹음 볼륨을 맞추려던 값에 맞추지 못함, 원래 볼륨으로 되돌리지 못함.
  외부 출력이라 걸지 않거나 되돌리기를 미룬 것은 기록대로 한 것이라 보고만 한다.
- 센서 CSV도 json의 sensors도 없으면 이 기능 전의 녹음·세션(또는 원격 시작 sensors=false 녹음)이라 "센서 기록 없음"만 적는다(문제 아님).
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import numpy as np  # noqa: E402

GAP_FACTOR = 5  # 이웃 이벤트 간격이 중앙 간격의 이 배를 넘으면 빈틈 하나로 센다(100 Hz면 50 ms 넘게 비었음)
LABELS = {"sitting": "앉음", "standing": "섬", "desk": "책상", "air": "공중", "loose": "헐렁", "normal": "보통", "tight": "꽉"}  # 폰 화면 글
SKIPS = {"external_output": "외부 출력이 이어져 있어", "changed": "그사이 볼륨이 바뀌어"}  # 볼륨을 걸지·되돌리지 않은 까닭(rec.json 값)


def fmt(value):
    """숫자를 짧게(없으면 —)."""
    return "—" if value is None else f"{float(value):g}"


def gaps(t_ns):
    d = np.diff(t_ns)
    if len(d) == 0:
        return 0
    median = float(np.median(d))
    return int((d > GAP_FACTOR * median).sum()) if median > 0 else 0


def volume_lines(rec):
    """(보고 줄, 문제) — 워치 rec.json volume(계획 6, 검토 고침 I1·M5)."""
    v = hs_io.recording_volume(rec)
    lines, problems = [], []
    if v["choice"] is None:
        lines.append(f"녹음 볼륨: 그대로(미디어 볼륨 {fmt(v['index'])}/{fmt(v['max'])})")
    else:
        aimed = f", 맞추려던 값 {v['target']}" if v["target"] is not None and v["target"] != v["choice"] else ""
        lines.append(f"녹음 볼륨: 고른 값 {v['choice']}{aimed}, 녹음 동안 {fmt(v['index'])}/{fmt(v['max'])}, 원래 {fmt(v['original'])}, "
                     f"되돌린 뒤 {fmt(v['restored'])}")
    if v["skipped"]:
        lines.append(f"녹음 볼륨을 걸지 않음: {SKIPS.get(v['skipped'], v['skipped'])}(그대로 녹음)")
    if v["outputs"]:
        lines.append(f"시작 때 외부 출력: {', '.join(v['outputs'])} — 볼륨 값은 그 기기 것")
    if v["outputChanges"]:
        last = ", ".join(v["outputChanges"][-1][1]) or "스피커"
        lines.append(f"녹음 중 외부 출력이 바뀜 {len(v['outputChanges'])}번(끝: {last})")
    if v["changes"]:
        lines.append(f"녹음 중 볼륨이 바뀜 {len(v['changes'])}번")
    if v["restoreSkipped"]:
        lines.append(f"정지 때 되돌리지 않음: {SKIPS.get(v['restoreSkipped'], v['restoreSkipped'])} — 스피커로 돌아온 뒤 워치 앱을 열 때 되돌림")
    changed = v["choice"] is not None and not v["skipped"] and v["index"] != v["original"]
    if changed and not rec.get("stop") and v["restored"] is None and not v["restoreSkipped"]:
        lines.append("되돌린 기록 없음(녹음이 끝나지 않음 — 다음에 워치 앱을 열 때 되돌림)")
    left = v["leftover"]
    if left:
        what = f"되돌림(지금 {fmt(left['restored'])})" if not left["skipped"] else (
            "그사이 볼륨이 바뀌어 그대로 둠" if left["skipped"] == "changed" else f"{SKIPS.get(left['skipped'], left['skipped'])} 또 미룸")
        lines.append(f"앞 녹음이 남긴 볼륨(원래 {fmt(left['original'])}, 맞춘 {fmt(left['applied'])}): {what}")
    if v["choice"] is not None and not v["skipped"]:
        want = v["target"] if v["target"] is not None else (min(v["choice"], v["max"]) if v["max"] is not None else v["choice"])
        if v["index"] != want:
            problems.append(f"녹음 볼륨을 고른 값({v['choice']})에 맞추지 못함(녹음 동안 {fmt(v['index'])})")
    if v["restored"] is not None and v["restored"] != v["original"]:
        problems.append(f"원래 볼륨({fmt(v['original'])})으로 되돌리지 못함(되돌린 뒤 {v['restored']})")
    return lines, problems


def context_line(c):
    parts = [f"장소 {c['place'] or '—'}", f"자세 {LABELS.get(c['posture'], '—')}", f"팔 {LABELS.get(c['armSupport'], '—')}",
             f"줄 {LABELS.get(c['strap'], '—')}", "워치 다시 참" if c["rewear"] else "다시 참 아님"]
    return "세션 정보: " + " · ".join(parts)


def track_line(track):
    lag_ms = track.lag_ns / 1e6
    line = (f"{track.kind}: {len(track.t_ns)}줄, {track.rate_hz():.1f} Hz, 받은 시각 − 이벤트 시각 1% {float(np.percentile(lag_ms, 1)):.1f}"
            f" · 중앙 {float(np.median(lag_ms)):.1f} · 99% {float(np.percentile(lag_ms, 99)):.1f} · 최대 {float(lag_ms.max()):.1f} ms")
    return line if track.kind not in hs_io.CONTINUOUS else line + f", 빈틈 {gaps(track.t_ns)}개"


def check(folder):
    """(보고 줄, 문제). 읽다가 넘어지면 그 폴더만 문제 한 줄로 낸다(검토 고침 M5)."""
    try:
        return check_folder(folder)
    except Exception as e:  # noqa: BLE001  폴더 하나가 넘어져도 뒤 폴더를 점검한다
        return [], [f"점검하지 못함({type(e).__name__}: {e})"]


def check_folder(folder):
    rec_path, meta_path = os.path.join(folder, "rec.json"), os.path.join(folder, "meta.json")
    lines, problems = [], []
    if os.path.exists(rec_path):
        doc, base = hs_io.read_json(rec_path), hs_io.WATCH_SENSORS
        more, bad = volume_lines(doc)
        lines += more
        problems += bad
    elif os.path.exists(meta_path):
        doc, base = hs_io.read_json(meta_path), hs_io.PHONE_SENSORS
        lines.append(context_line(hs_io.session_context(doc)))
    else:
        return [], [f"{folder}: rec.json도 meta.json도 없음"]
    report = doc.get("sensors")
    log = hs_io.read_sensors(folder, base)
    if report is None and log is None:
        lines.append("센서 기록 없음(이 기능 전 녹음·세션이거나 센서 기록을 끈 녹음)")
        return lines, problems
    report = report or {}
    listed = report.get("list") or []
    for s in listed:
        lines.append(f"센서 {s.get('kind')}: {s.get('name')} ({s.get('vendor')}), 최대 {fmt(s.get('maxRateHz'))} Hz, "
                     f"분해능 {fmt(s.get('resolution'))}, 건 주기 {s.get('periodUs')} µs")
    if report.get("missing"):
        lines.append("기기에 없는 종류: " + ", ".join(report["missing"]))
    if report.get("others"):
        lines.append("같은 종류의 다른 센서: " + ", ".join(f"{s.get('kind')} {s.get('name')} ({s.get('vendor')})" for s in report["others"]))
    if report.get("error"):
        problems.append(f"센서 기록 오류: {report['error']}")
    files = log.files if log else []
    have = {os.path.basename(p) for p in files}
    lost = [p.get("file") for p in report.get("parts") or [] if p.get("file") not in have]
    if lost:
        problems.append("기록에 있는데 없는 조각: " + ", ".join(lost))
    if files:
        lines.append(f"조각 {len(files)}개, {sum(os.path.getsize(p) for p in files) / 1e6:.2f} MB")
    tracks = log.tracks if log else {}
    if log and log.dropped:
        lines.append(f"버린 줄 {log.dropped}개(열 수가 다르거나 끝이 잘린 줄)")
    for s in listed:
        if s.get("kind") not in tracks:
            problems.append(f"{s.get('kind')}: 기기가 준 센서인데 줄이 없음")
    if log:
        floor = log.lag_floor_ns
        if floor is None:
            lines.append("시계: 연속 센서가 없어 판정하지 않음(이벤트 시각 그대로)")
        elif log.event_clock:
            lines.append(f"시계: 연속 센서 받은 시각 − 이벤트 시각 1% 값 {floor / 1e6:.1f} ms → 이벤트 시각을 그대로 씀")
        else:
            problems.append(f"이벤트 시각이 elapsedNs 기준이 아님(연속 센서 받은 시각 − 이벤트 시각 1% 값 {floor / 1e6:.0f} ms) — 이벤트 시각에 그만큼 더해 씀")
    for track in tracks.values():
        lines.append(track_line(track))
    return lines, problems


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="센서 기록·녹음 볼륨·세션 정보를 점검한다(계획 6)")
    parser.add_argument("folders", nargs="+", help="워치 녹음 폴더(rec.json) 또는 폰 세션 폴더(meta.json)")
    args = parser.parse_args()
    bad = False
    for folder in args.folders:
        lines, problems = check(folder)
        print(f"[{os.path.basename(os.path.normpath(folder))}]")
        for line in lines:
            print("  " + line)
        if problems:
            bad = True
            print("  문제:")
            for p in problems:
                print("    - " + p)
        else:
            print("  문제 없음")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
