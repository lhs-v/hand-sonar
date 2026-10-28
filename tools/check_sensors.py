"""센서 기록·녹음 볼륨·세션 정보를 점검한다(손 소나 계획 6, 기기 확인용).

    python -I tools/check_sensors.py <폴더> [<폴더> …]

- 폴더에 rec.json이 있으면 워치 녹음(sensors-NNN.csv, rec.json volume), meta.json이 있으면 폰 세션(phone_sensors-NNN.csv, meta.json
  context·rewear)으로 본다.
- 보고: 기기가 준 센서(이름·제조사·최대 표본률·분해능·건 주기)와 기기에 없던 종류, 종류마다 줄 수·실제 표본률(이벤트 시각)·받은 시각 − 이벤트
  시각(중앙·99%·최대 ms)·빈틈 수(이웃 이벤트 간격이 중앙 간격의 GAP_FACTOR배를 넘는 곳, 바뀔 때만 내는 light는 세지 않음), 버린 줄 수, 조각 수와 크기.
  워치는 녹음 볼륨(고른 값·녹음 동안·원래·되돌린 뒤·녹음 중 바뀐 번수), 폰은 세션 정보와 "워치 다시 참".
- 문제(끝 코드 1): 센서 기록 오류(json sensors.error), json이 적은 조각이 없음, 기기가 준 센서인데 줄이 없음, 이벤트 시각이 elapsedNs 기준이
  아님(그때 hs_io.SensorTrack.ns는 받은 시각을 쓴다), 녹음 볼륨을 고른 값에 맞추지 못함, 원래 볼륨으로 되돌리지 못함.
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


def fmt(value):
    """숫자를 짧게(없으면 —)."""
    return "—" if value is None else f"{float(value):g}"


def gaps(t_ns):
    d = np.diff(t_ns)
    if len(d) == 0:
        return 0
    median = float(np.median(d))
    return int((d > GAP_FACTOR * median).sum()) if median > 0 else 0


def volume_lines(v):
    """(보고 줄, 문제) — 워치 rec.json volume."""
    problems = []
    if v["choice"] is None:
        return [f"녹음 볼륨: 그대로(미디어 볼륨 {fmt(v['index'])}/{fmt(v['max'])})"], problems
    line = f"녹음 볼륨: 고른 값 {v['choice']}, 녹음 동안 {fmt(v['index'])}/{fmt(v['max'])}, 원래 {fmt(v['original'])}, 되돌린 뒤 {fmt(v['restored'])}"
    if v["changes"]:
        line += f", 녹음 중 바뀜 {len(v['changes'])}번"
    want = v["choice"] if v["max"] is None else min(v["choice"], v["max"])
    if v["index"] != want:
        problems.append(f"녹음 볼륨을 고른 값({v['choice']})에 맞추지 못함(녹음 동안 {fmt(v['index'])})")
    if v["restored"] is not None and v["restored"] != v["original"]:
        problems.append(f"원래 볼륨({fmt(v['original'])})으로 되돌리지 못함(되돌린 뒤 {v['restored']})")
    return [line], problems


def context_line(c):
    parts = [f"장소 {c['place'] or '—'}", f"자세 {LABELS.get(c['posture'], '—')}", f"팔 {LABELS.get(c['armSupport'], '—')}",
             f"줄 {LABELS.get(c['strap'], '—')}", "워치 다시 참" if c["rewear"] else "다시 참 아님"]
    return "세션 정보: " + " · ".join(parts)


def track_line(track):
    lag_ms = track.lag_ns / 1e6
    line = (f"{track.kind}: {len(track.t_ns)}줄, {track.rate_hz():.1f} Hz, 받은 시각 − 이벤트 시각 중앙 {float(np.median(lag_ms)):.1f} ms"
            f" · 99% {float(np.percentile(lag_ms, 99)):.1f} ms · 최대 {float(lag_ms.max()):.1f} ms")
    return line if track.kind == "light" else line + f", 빈틈 {gaps(track.t_ns)}개"


def check(folder):
    """(보고 줄, 문제)."""
    rec_path, meta_path = os.path.join(folder, "rec.json"), os.path.join(folder, "meta.json")
    lines, problems = [], []
    if os.path.exists(rec_path):
        doc, base = hs_io.read_json(rec_path), hs_io.WATCH_SENSORS
        more, bad = volume_lines(hs_io.recording_volume(doc))
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
    for track in tracks.values():
        lines.append(track_line(track))
        if not track.event_clock:
            lag = float(np.median(track.lag_ns)) / 1e6 if len(track.t_ns) else float("nan")
            problems.append(f"{track.kind}: 이벤트 시각이 elapsedNs 기준이 아님(받은 시각 − 이벤트 시각 중앙 {lag:.0f} ms) — 받은 시각을 씀")
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
