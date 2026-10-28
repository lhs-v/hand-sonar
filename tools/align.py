"""세션마다 폰 표지음으로 '워치 샘플 번호 → 폰 elapsedNs' 직선을 만든다(손 소나 계획 3, 설계 §4·§8·§8.1·§9-3).

    python -I tools/align.py [--data 데이터 폴더] [세션 이름 …]   (이름이 없으면 phone/sessions의 세션 모두)

1. 폰 표지음: meta.json stats.markers.start·end(설계의 stats.*Marker)의 micFrame 앞뒤 0.05 s를 phone_audio.wav에서 해석 신호
   포락선으로 다시 찾고(markers.py), phone_audio.json recordClock으로 폰 elapsedNs로 바꾼다. 낸 시각(playNs) 뒤 0–0.5 s
   (실측 111–113 ms) 밖이면 못 찾음이다. phone_audio.json이 없으면 폰 WAV를 쓰지 않는다.
2. 워치 녹음: 두 기기의 wallMs로 폰 표지음 시각을 워치 시각으로 옮겨, 그 시각을 담은 녹음을 고른다. 시스템 시각은 찾을 범위를
   좁히는 데만 쓴다(설계 §4).
3. 워치 시작 표지음: 시스템 시각으로 옮긴 예상 위치 ±2 s에서 후보를 찾는다. 같은 녹음에 준비 화면의 "표지음 시험"과 잡음 봉우리
   (비 20–36)가 있어서, 예상 위치에 가까운 후보부터 폰 표지음 간격만큼 뒤 ±0.5 s에서 끝 표지음을 찾고, 두 기기 간격이 0.02 s 안에서
   맞는 첫 후보를 쓴다(시작 버튼 소리 같은 잡음이 시작 표지음 0.3 s 앞에 있어도 간격이 0.3 s 어긋나 걸러진다).
4. 두 표지음이 한 녹음 안에 있고 그 사이에 녹음 계단(샘플 빠짐)이 없으면 두 점으로 직선을 만든다(method "two", "표지음 둘").
   조각 빈틈이 있으면 그 뒤는 읽지 않으므로 끝 표지음을 못 찾는다. 세션 도중 워치 녹음이 멈췄다 다시 켜졌으면 두 표지음이 다른 녹음에
   든다. 그런 때는 찾은 표지음 하나와 워치 녹음 시각(시스템 시각의 빠르기)으로 맞춘다(method "one", "표지음 하나"). 하나도 없으면
   method "none"과 이유를 적는다.
5. 시스템 시각 직선(워치 wallMs = 폰 wallMs로 봄)도 만들어, 표지음 직선과의 차이를 표지음 자리에서 ms로 적는다.
폰 스피커에서 폰 마이크와 워치 마이크까지의 소리 길이 차(50 cm면 약 1.5 ms)는 고정 지연이라 verify_sync.py가 재는 지연에 들어간다.
결과는 processed/<세션>/align.json. 모든 세션이 "표지음 둘"이 아니면 끝 코드 1로 끝난다.
"""
import argparse
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import markers  # noqa: E402
import numpy as np  # noqa: E402

RATE = hs_io.RATE
PHONE_SEARCH_SEC = 0.05  # 폰 WAV에서 micFrame 앞뒤로 다시 찾는 폭(폰 찾기의 흔들림 ±0.15 ms보다 넉넉히)
MAX_PLAY_DELAY_SEC = 0.5  # 낸 시각 뒤 이 안에 폰 마이크에 들어와야 같은 소리다(실측 111–113 ms, 설계 §8.1)
START_WINDOW_SEC = 2.0  # 워치 시작 표지음을 찾는 폭(예상 위치 ±). 실측 시스템 시각 어긋남 0.21 s
END_WINDOW_SEC = 0.5  # 워치 끝 표지음을 찾는 폭(시작 + 폰 간격 ±, 설계 §8.1)
PAIR_TOLERANCE_SEC = 0.02  # 두 기기 표지음 간격이 이만큼 안에서 같아야 짝이다(시계 차 90 ppm × 218 s도 덮음, 실측 2.7 ppm)
MAX_CANDIDATES = 5
STRONG_FRACTION = 0.5  # 짝이 없을 때: 가장 센 후보의 이 비율 이상인 것 중 예상 위치에 가장 가까운 후보(표지음 시험은 비슷하게 세고 잡음은 훨씬 약함)
LABELS = {"two": "표지음 둘", "one": "표지음 하나", "none": "표지음 없음"}


def phone_marker(session, key):
    """폰 표지음 하나: {"found", "playNs", 찾았으면 "frame"(phone_audio.wav 샘플), "ns"(폰 elapsedNs), "ratio", "delayMs", 못 찾았으면 "reason"}."""
    m = session.marker(key)
    out = {"found": False, "playNs": hs_io.nanos(m, "playNs")}
    if not m.get("found") or out["playNs"] is None:
        out["reason"] = "폰이 못 찾음(meta found=false)"
        return out
    if session.audio is None:
        out["reason"] = "phone_audio.json 없음"
        return out
    mic = hs_io.num(m, "micFrame")
    first = max(0, int(mic - PHONE_SEARCH_SEC * RATE))
    hit = markers.find(session.audio.read(first, int(mic + markers.SAMPLES + PHONE_SEARCH_SEC * RATE)))
    if hit is None:
        out["reason"] = "phone_audio.wav에서 다시 못 찾음"
        return out
    frame = first + hit.sample
    ns = float(session.audio.clock.ns(frame))
    delay = (ns - out["playNs"]) / 1e9
    if not 0 < delay < MAX_PLAY_DELAY_SEC:
        out["reason"] = f"낸 시각 뒤 {delay * 1e3:.1f} ms(0–500 ms 밖)"
        return out
    out.update(found=True, frame=float(frame), ns=ns, ratio=hit.ratio, delayMs=delay * 1e3)
    return out


def locate(session, recordings, phone_ns):
    """폰 시각을 시스템 시각으로 워치 녹음 위치로 옮긴다: (녹음, 예상 프레임) 또는 (None, None)."""
    wall = session.ns_to_wall_ms(phone_ns)
    margin = START_WINDOW_SEC * RATE
    for r in recordings:
        f = float(r.clock.frames(r.wall_ms_to_ns(wall)))
        if -margin <= f < r.frames + margin:
            return r, f
    return None, None


def search(recording, center, half_sec, max_count):
    """recording의 center ± half_sec에서 표지음 후보 [(녹음 프레임, 비)], 센 것부터."""
    a = max(0, int(center - half_sec * RATE))
    b = min(recording.frames, int(center + half_sec * RATE) + markers.SAMPLES)
    if b - a <= markers.SAMPLES:
        return []
    return [(a + h.sample, h.ratio) for h in markers.candidates(recording.read(a, b), max_count)]


def pick(cands, center):
    """짝 없이 하나를 고를 때: 가장 센 후보의 STRONG_FRACTION 이상인 것 중 예상 위치에 가장 가까운 것."""
    strong = max(r for _, r in cands) * STRONG_FRACTION
    return min((c for c in cands if c[1] >= strong), key=lambda c: abs(c[0] - center))


def align_session(session, recordings):
    """세션 하나의 align.json 내용(dict)."""
    out = {"format": 1, "session": session.name, "recording": None, "method": "none", "line": {}, "notes": []}
    notes = out["notes"]
    ps, pe = phone_marker(session, "start"), phone_marker(session, "end")
    out["phone"] = {"start": ps, "end": pe}
    if session.audio is not None:
        duration = hs_io.num(session.meta.get("stats"), "durationSec", 0.0)
        if session.audio.frames / RATE < duration - 2.0:
            notes.append(f"폰 마이크가 도중에 멈춤(녹음 {session.audio.frames / RATE:.1f} s < 세션 {duration:.1f} s)")
    rs, fs = locate(session, recordings, ps["ns"]) if ps["found"] else (None, None)
    re_, fe = locate(session, recordings, pe["ns"]) if pe["found"] else (None, None)
    starts = sorted(search(rs, fs, START_WINDOW_SEC, MAX_CANDIDATES), key=lambda c: abs(c[0] - fs)) if rs else []
    watch = {"start": None, "end": None, "startCandidates": len(starts)}
    out["watch"] = watch

    pair = None
    if rs is not None and rs is re_:
        gap = (pe["ns"] - ps["ns"]) * RATE / 1e9
        for s1, r1 in starts:
            ends = search(rs, s1 + gap, END_WINDOW_SEC, 1)
            if ends and abs(ends[0][0] - s1 - gap) <= PAIR_TOLERANCE_SEC * RATE:
                pair = ((s1, r1), ends[0])
                break
            if ends:
                notes.append(f"후보 {s1 / RATE:.3f} s: 끝 표지음 간격이 {(ends[0][0] - s1 - gap) / RATE * 1e3:+.0f} ms 어긋나 버림")
        if pair is None:
            notes.append("두 표지음의 짝을 못 찾음")
    elif ps["found"] and pe["found"]:
        notes.append("두 표지음이 다른 녹음에 있음(세션 도중 워치 녹음이 멈췄다 다시 켜졌을 수 있음)" if rs and re_ else
                     "표지음 시각을 담은 워치 녹음이 하나뿐이거나 없음")

    single = None  # (키, 녹음, 프레임, 비, 폰 표지음)
    if pair is not None:
        (s1, r1), (s2, r2) = pair
        watch["start"], watch["end"] = {"frame": s1, "ratio": r1}, {"frame": s2, "ratio": r2}
        steps = rs.clock.steps_between(s1, s2)
        if steps:
            notes.append("표지음 사이에 녹음 계단: " + ", ".join(f"{f / RATE:.1f} s {ns / 1e6:+.1f} ms" for f, ns in steps))
            single = ("start", rs, s1, r1, ps)
        else:
            slope = (pe["ns"] - ps["ns"]) / (s2 - s1)
            out.update(method="two", recording=rs.name, line={"frame0": s1, "ns0": ps["ns"], "nsPerFrame": slope})
            watch_ns = rs.clock.ns(np.array([s1, s2]))
            out["intervals"] = {"watchSec": (s2 - s1) / RATE, "phoneSec": (pe["ns"] - ps["ns"]) / 1e9,
                                "phoneWavSec": (pe["frame"] - ps["frame"]) / RATE}
            out["clockPpm"] = ((pe["ns"] - ps["ns"]) / float(watch_ns[1] - watch_ns[0]) - 1) * 1e6
    elif starts:
        s1, r1 = pick(starts, fs)
        watch["start"] = {"frame": s1, "ratio": r1}
        single = ("start", rs, s1, r1, ps)
    elif re_ is not None:
        ends = search(re_, fe, START_WINDOW_SEC, MAX_CANDIDATES)
        if ends:
            s2, r2 = pick(ends, fe)
            watch["end"] = {"frame": s2, "ratio": r2}
            single = ("end", re_, s2, r2, pe)
    if single is not None:
        key, rec, frame, _, pm = single
        out.update(method="one", recording=rec.name, line={"offsetNs": pm["ns"] - float(rec.clock.ns(frame))})
        notes.append(f"{'시작' if key == 'start' else '끝'} 표지음 하나와 워치 녹음 시각으로 맞춤")
    if out["method"] == "none":
        notes.append("워치 녹음에서 표지음을 못 찾음" if ps["found"] or pe["found"] else "폰 표지음이 둘 다 없음")
    out["label"] = LABELS[out["method"]]

    if out["recording"] is not None:
        rec = next(r for r in recordings if r.name == out["recording"])
        start = rec.rec["start"]
        out["wallOffsetNs"] = session.wall_ms_to_ns(hs_io.num(start, "wallMs")) - hs_io.nanos(start, "elapsedNs")
        a = Alignment(out["method"], rec, **_line_args(out["line"]))
        diffs = {}
        for key in ("start", "end"):
            if watch[key] is not None:
                f = watch[key]["frame"]
                diffs[key] = (float(a.phone_ns(f)) - (float(rec.clock.ns(f)) + out["wallOffsetNs"])) / 1e6
        out["wallDiffMs"] = diffs
    return out


@dataclass
class Alignment:
    """align.json의 맞춤: 워치 녹음 프레임 ↔ 폰 elapsedNs. "two"는 표지음 두 점의 직선, "one"은 워치 녹음 시각 + offsetNs."""
    method: str
    recording: object  # hs_io.WatchRecording
    frame0: float = 0.0
    ns0: float = 0.0
    ns_per_frame: float = 0.0
    offset_ns: float = 0.0

    def phone_ns(self, frames):
        f = np.asarray(frames, dtype=np.float64)
        if self.method == "two":
            return self.ns0 + (f - self.frame0) * self.ns_per_frame
        return self.recording.clock.ns(f) + self.offset_ns

    def frames(self, phone_ns):
        t = np.asarray(phone_ns, dtype=np.float64)
        if self.method == "two":
            return self.frame0 + (t - self.ns0) / self.ns_per_frame
        return self.recording.clock.frames(t - self.offset_ns)


def _line_args(line):
    return {"frame0": line.get("frame0", 0.0), "ns0": line.get("ns0", 0.0), "ns_per_frame": line.get("nsPerFrame", 0.0),
            "offset_ns": line.get("offsetNs", 0.0)}


def load(data, name):
    """processed/<name>/align.json → (내용, Alignment). 맞춤이 없으면 Alignment 자리는 None."""
    info = hs_io.read_json(os.path.join(data, "processed", name, "align.json"))
    if info["method"] == "none":
        return info, None
    rec = hs_io.read_recording(os.path.join(data, "watch", "recordings", info["recording"]))
    return info, Alignment(info["method"], rec, **_line_args(info["line"]))


def report(info):
    """사람이 읽는 몇 줄."""
    lines = [f"{info['session']} ↔ 워치 {info['recording'] or '-'}: {info['label']}"]
    parts = []
    for key, name in (("start", "시작"), ("end", "끝")):
        m = info["phone"][key]
        parts.append(f"{name} {m['frame']:.1f}(비 {m['ratio']:.0f}, 낸 뒤 {m['delayMs']:.1f} ms)" if m["found"] else f"{name} 없음({m['reason']})")
    lines.append("  폰 표지음: " + ", ".join(parts))
    parts = []
    for key, name in (("start", "시작"), ("end", "끝")):
        w = info["watch"][key]
        parts.append(f"{name} {w['frame'] / RATE:.3f} s(비 {w['ratio']:.0f})" if w else f"{name} 없음")
    lines.append(f"  워치 표지음: {', '.join(parts)}, 시작 후보 {info['watch']['startCandidates']}개")
    if "intervals" in info:
        iv = info["intervals"]
        lines.append(f"  간격: 워치 {iv['watchSec']:.4f} s(48 kHz로 셈), 폰 elapsedNs {iv['phoneSec']:.4f} s, 폰 WAV {iv['phoneWavSec']:.4f} s,"
                     f" 두 기기 시계 빠르기 차 {info['clockPpm']:+.1f} ppm")
    if info.get("wallDiffMs"):
        lines.append("  표지음 − 시스템 시각 맞춤: " + ", ".join(f"{'시작' if k == 'start' else '끝'} {v:+.1f} ms" for k, v in info["wallDiffMs"].items()))
    lines += [f"  메모: {n}" for n in info["notes"]]
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="폰 표지음으로 워치 녹음과 폰 세션의 시간을 맞춘다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA, help="데이터 폴더(watch·phone·processed가 있는 곳)")
    args = parser.parse_args()
    recordings = []
    for folder in hs_io.recording_folders(args.data):
        try:
            recordings.append(hs_io.read_recording(folder))
        except (OSError, KeyError, ValueError) as e:
            print(f"워치 {os.path.basename(folder)}: 못 읽음({e})")
    missing = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        info = align_session(hs_io.read_session(hs_io.session_folder(args.data, name)), recordings)
        hs_io.write_json(os.path.join(hs_io.processed_folder(args.data, name), "align.json"), info)
        print("\n".join(report(info)))
        missing += info["method"] != "two"
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
