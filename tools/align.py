"""세션마다 폰 표지음으로 '워치 샘플 번호 → 폰 elapsedNs' 직선을 만든다(손 소나 계획 3, 설계 §4·§8·§8.1·§9-3).

    python -I tools/align.py [--data 데이터 폴더] [세션 이름 …]   (이름이 없으면 phone/sessions의 세션 모두)

1. 폰 표지음: meta.json stats.markers.start·end(설계의 stats.*Marker)의 micFrame 앞뒤 0.05 s를 phone_audio.wav에서 해석 신호
   포락선으로 다시 찾고(markers.py), phone_audio.json recordClock으로 폰 elapsedNs로 바꾼다. 낸 시각(playNs) 뒤 0–0.5 s
   (실측 111–113 ms) 밖이면 못 찾음이다. phone_audio.json이 없으면 폰 WAV를 쓰지 않는다.
2. 워치 녹음: 두 기기의 wallMs로 폰 표지음 시각을 워치 시각으로 옮겨, 그 시각을 담은 녹음을 고른다. 시스템 시각은 찾을 범위를
   좁히는 데만 쓴다(설계 §4). 두 기기의 벽시계는 초 단위로 어긋날 수 있다(실측: 10-27 23:51 전 워치가 폰보다 약 4.07 s 앞섰고,
   녹음 235007 안에서 −3,850 ms 계단으로 고쳐졌다). 녹음마다 rec.json start·stop의 부팅 시각(wallMs − elapsedNs/1e6)을 견줘
   벽시계 계단을 알린다(평소 ±0.6 ms).
3. 워치 시작 표지음: 예상 위치 ±10 s에서 후보를 찾는다. 이 창에는 준비 화면의 "표지음 시험"(실측 두 세션 모두 시작 표지음 3.6 s 앞),
   딸깍 소리, 잡음 봉우리가 든다. 그래서 고르기는 짝 검사가 한다: 시작 후보마다 폰 표지음 간격만큼 뒤 ±0.5 s의 끝 후보 중 두 기기
   간격이 0.02 s 안에서 맞는 것을 찾고, 짝의 두 후보가 모두 처프일 수 있는 소리(아래 대역 상승 비 ≥ 5)여야 한다. 짝이 여럿이면
   간격이 가장 잘 맞는 것을 쓴다.
4. 두 표지음이 한 녹음 안에 있고 그 사이에 녹음 계단(샘플 빠짐)이 없으면 두 점으로 직선을 만든다(method "two", "표지음 둘").
   계단이 있으면 짝의 시작 표지음 하나로 맞춘다. 세 간격(워치 샘플, 폰 elapsedNs, 폰 WAV)이 2 ms 넘게 다르면 경고한다.
5. 짝이 없으면 표지음 하나로 맞추되(method "one", "표지음 하나": 찾은 표지음과 워치 녹음 시각의 빠르기), 아래를 모두 요구하고
   못 맞추면 method "none"과 이유를 적는다. 틀린 맞춤을 내보내느니 없음으로 둔다(학습 정답).
   - 처프 모양: 후보 40 ms의 21.5–23 kHz 세기가 둘레 ±1 s 40 ms 조각들의 중앙값보다 100배 넘게 솟고, 2–17 kHz는 10배 안이다
     (실측 표지음·표지음 시험 504–4,147배 / 0.7–3.6배, 딸깍은 2–17 kHz가 1,593–58,447배).
   - 창 안의 처프일 수 있는 소리(21.5–23 kHz 상승 ÷ 2–17 kHz 상승 ≥ 5) 중 하나뿐이거나, 시작 표지음이면 그중 마지막 것.
     표지음 시험은 늘 시작 표지음 앞에 오고 세션 중에는 표지음이 나지 않는다. 다만 세션이 창보다 짧아 다음 세션의 소리가 창에
     들 수 있으면(처음 후보 + 세션 길이 ≤ 창 끝) 고르지 않는다. 끝 표지음은 하나뿐일 때만 쓴다.
   - "표지음 하나"는 verify_sync.py의 지연 확인(1프레임 안, 상관 ≥ 문턱)을 지나야 pairs.py가 쓴다.
6. 시스템 시각 직선(워치 wallMs = 폰 wallMs로 봄)도 만들어, 표지음 직선과의 차이를 표지음 자리에서 ms로 적는다.
폰 스피커에서 폰 마이크와 워치 마이크까지의 소리 길이 차(50 cm면 약 1.5 ms)는 고정 지연이라 verify_sync.py가 재는 지연에 들어간다.
결과는 processed/<세션>/align.json. 모든 세션이 "표지음 둘"이 아니면(처리 중 오류 포함) 끝 코드 1로 끝난다.
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
START_WINDOW_SEC = 10.0  # 워치 시작 표지음을 찾는 폭(예상 위치 ±). 실측 벽시계 어긋남 4.07 s의 2배 넘게. 고르기는 짝 검사가 한다
END_WINDOW_SEC = 0.5  # 워치 끝 표지음을 찾는 폭(시작 + 폰 간격 ±, 설계 §8.1)
PAIR_TOLERANCE_SEC = 0.02  # 두 기기 표지음 간격이 이만큼 안에서 같아야 짝이다(시계 차 90 ppm × 218 s도 덮음, 실측 2.7 ppm)
MAX_CANDIDATES = 20  # 창 하나에서 볼 후보 수. 20 s 창에는 잡음 후보(비 8–27)가 여럿 든다(실측 4 s 창의 9–17%에 하나 이상)
INTERVAL_WARN_SEC = 0.002  # 세 간격이 이보다 다르면 경고(실측 0.2–0.7 ms. 폰 시각 튐이나 잘못 잡은 표지음)
WALL_STEP_NOTE_MS = 50.0  # 워치 녹음 start·stop 부팅 시각 차가 이보다 크면 녹음 중 벽시계가 바뀐 것으로 알린다(평소 ±0.6 ms)
SHAPE_REF_SEC = 1.0  # 처프 모양: 후보 둘레 이 폭의 40 ms 조각들 세기의 중앙값을 기준으로 삼는다(후보 ±0.1 s는 뺌)
MARKER_BAND = (21_500.0, 23_000.0)
OTHER_BAND = (2_000.0, 17_000.0)  # 워치 처프(18–21 kHz)를 뺀 아래 대역. 딸깍·말소리가 여기 든다
MIN_MARKER_RISE = 100.0  # 표지음 하나로 맞출 때 21.5–23 kHz 상승 하한(실측 처프 504–4,147배)
MAX_OTHER_RISE = 10.0  # 같은 때 2–17 kHz 상승 상한(실측 처프 0.7–3.6배, 처프가 아닌 후보 135개 중 위 하한과 함께 넘는 것 없음)
LIKE_RATIO = 5.0  # 처프일 수 있는 소리: 21.5–23 kHz 상승 ÷ 2–17 kHz 상승이 이 이상(실측 처프 139–6,334, 처프 아닌 후보 135개 중 6개, 최대 26)
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


def wall_step_ms(recording):
    """rec.json stop과 start의 부팅 시각(wallMs − elapsedNs/1e6) 차(ms). 녹음 중 벽시계가 바뀌면 0이 아니다. stop이 없으면 None."""
    start, stop = recording.rec.get("start"), recording.rec.get("stop")
    try:
        boot = [hs_io.num(p, "wallMs") - hs_io.nanos(p, "elapsedNs") / 1e6 for p in (start, stop)]
    except TypeError:  # 키가 없음(녹음 중이거나 앱이 죽음)
        return None
    return boot[1] - boot[0]


def locate(session, recordings, phone_ns):
    """폰 시각을 시스템 시각으로 워치 녹음 위치로 옮긴다: (녹음, 예상 프레임) 또는 (None, None).
    예상 위치를 담은 녹음을 먼저, 없으면 START_WINDOW_SEC 안에서 가장 가까운 녹음."""
    wall = session.ns_to_wall_ms(phone_ns)
    margin = START_WINDOW_SEC * RATE
    best = None
    for r in recordings:
        f = float(r.clock.frames(r.wall_ms_to_ns(wall)))
        outside = max(0.0, -f, f - r.frames)
        if outside <= margin and (best is None or outside < best[0]):
            best = (outside, r, f)
    return (best[1], best[2]) if best else (None, None)


def search(recording, center, half_sec, max_count):
    """recording의 center ± half_sec에서 표지음 후보 [(녹음 프레임, 비)], 센 것부터."""
    a = max(0, int(center - half_sec * RATE))
    b = min(recording.frames, int(center + half_sec * RATE) + markers.SAMPLES)
    if b - a <= markers.SAMPLES:
        return []
    return [(a + h.sample, h.ratio) for h in markers.candidates(recording.read(a, b), max_count)]


_FREQS = np.fft.rfftfreq(markers.SAMPLES, 1 / RATE)
_IN = (_FREQS >= MARKER_BAND[0]) & (_FREQS < MARKER_BAND[1])
_OUT = (_FREQS >= OTHER_BAND[0]) & (_FREQS < OTHER_BAND[1])


def band_rise(recording, frame):
    """후보(표지음 시작 프레임) 40 ms의 대역 세기 ÷ 둘레 ±1 s 40 ms 조각들의 중앙값: (21.5–23 kHz 상승, 2–17 kHz 상승).
    둘레가 모자라면(녹음 끝) (0, inf)."""
    n = markers.SAMPLES
    s = int(round(frame))
    a, b = max(0, s - int(SHAPE_REF_SEC * RATE)), min(recording.frames, s + int(SHAPE_REF_SEC * RATE) + n)
    if s < a or s + n > b:
        return 0.0, float("inf")
    x = recording.read(a, b)
    starts = [k for k in range(0, len(x) - n + 1, n) if abs(a + k - s) >= markers.SEPARATION]
    if len(starts) < 10:
        return 0.0, float("inf")
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(np.stack([x[k: k + n] for k in starts + [s - a]]) * window, axis=1)) ** 2
    ref, cand = spec[:-1], spec[-1]
    rise_in = cand[_IN].sum() / max(float(np.median(ref[:, _IN].sum(axis=1))), 1e-12)
    rise_out = cand[_OUT].sum() / max(float(np.median(ref[:, _OUT].sum(axis=1))), 1e-12)
    return float(rise_in), float(rise_out)


def maybe_chirp(rise):
    """처프일 수 있는 소리(짝 검사와 고르기의 셈에 넣음): 21.5–23 kHz만 2–17 kHz보다 LIKE_RATIO배 넘게 솟음."""
    return rise[0] >= LIKE_RATIO * rise[1]


def is_marker(rise):
    """표지음 하나로 맞출 때의 처프 모양: 21.5–23 kHz는 크게 솟고 2–17 kHz는 거의 그대로."""
    return rise[0] >= MIN_MARKER_RISE and rise[1] <= MAX_OTHER_RISE


def _hit(frame, ratio, rise):
    return {"frame": frame, "ratio": ratio, "markerRise": rise[0], "otherRise": rise[1]}


def single_marker(recording, center, key, span_frames, notes):
    """짝 없이 표지음 하나를 고른다(위 docstring 5). (프레임, 비, 상승) 또는 None(이유는 notes에)."""
    name = "시작" if key == "start" else "끝"
    cands = search(recording, center, START_WINDOW_SEC, MAX_CANDIDATES)
    rises = {f: band_rise(recording, f) for f, _ in cands}
    likes = sorted((f, r) for f, r in cands if maybe_chirp(rises[f]))
    if not likes:
        notes.append(f"{name} 표지음 하나: 창 안에 처프일 수 있는 소리가 없음(센 후보 {len(cands)}개)")
        return None
    window_end = center + START_WINDOW_SEC * RATE
    if len(likes) == 1:
        frame, ratio = likes[0]
    elif key == "start" and likes[0][0] + span_frames > window_end:
        frame, ratio = likes[-1]  # 표지음 시험은 시작 표지음 앞에 온다
        notes.append(f"시작 표지음 하나: 처프일 수 있는 소리 {len(likes)}개({', '.join(f'{f / RATE:.3f} s' for f, _ in likes)}) 중 마지막")
    else:
        why = "세션이 창보다 짧아 다음 세션 소리와 가를 수 없음" if key == "start" else "끝 표지음은 하나뿐일 때만 씀"
        notes.append(f"{name} 표지음 하나: 처프일 수 있는 소리 {len(likes)}개({', '.join(f'{f / RATE:.3f} s' for f, _ in likes)}) 중 고르지 않음({why})")
        return None
    rise = rises[frame]
    if not is_marker(rise):
        notes.append(f"{name} 표지음 하나: {frame / RATE:.3f} s가 처프 모양이 아님(21.5–23 kHz {rise[0]:.0f}배 < {MIN_MARKER_RISE:.0f}배"
                     f" 또는 2–17 kHz {rise[1]:.1f}배 > {MAX_OTHER_RISE:.0f}배)")
        return None
    return frame, ratio, rise


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
    out["wallStepMs"] = {}
    for r in {id(r): r for r in (rs, re_) if r is not None}.values():
        step = wall_step_ms(r)
        out["wallStepMs"][r.name] = step
        if step is not None and abs(step) > WALL_STEP_NOTE_MS:
            notes.append(f"워치 녹음 {r.name} 안에서 벽시계가 {step:+.1f} ms 바뀜(그만큼 시스템 시각 예상 위치가 틀릴 수 있음,"
                         f" 찾는 창 ±{START_WINDOW_SEC:.0f} s)")
    starts = search(rs, fs, START_WINDOW_SEC, MAX_CANDIDATES) if rs else []
    watch = {"start": None, "end": None, "startCandidates": len(starts)}
    out["watch"] = watch

    pair = None
    if rs is not None and rs is re_:
        gap = (pe["ns"] - ps["ns"]) * RATE / 1e9
        rises = {}

        def rise(f):
            if f not in rises:
                rises[f] = band_rise(rs, f)
            return rises[f]

        found, misses = [], []
        for s1, r1 in starts:
            ends = search(rs, s1 + gap, END_WINDOW_SEC, MAX_CANDIDATES)
            close = [(s2, r2) for s2, r2 in ends if abs(s2 - s1 - gap) <= PAIR_TOLERANCE_SEC * RATE]
            if not close:
                if ends:
                    miss = min((e[0] - s1 - gap for e in ends), key=abs)
                    misses.append(f"{s1 / RATE:.3f} s(간격 {miss / RATE * 1e3:+.0f} ms 어긋남)")
                continue
            s2, r2 = min(close, key=lambda e: abs(e[0] - s1 - gap))
            if maybe_chirp(rise(s1)) and maybe_chirp(rise(s2)):
                found.append((abs(s2 - s1 - gap), (s1, r1), (s2, r2)))
            else:
                misses.append(f"{s1 / RATE:.3f} s(간격은 맞으나 처프 모양이 아님)")
        if found:
            found.sort(key=lambda p: p[0])
            pair = found[0][1:]
            if len(found) > 1:
                notes.append(f"짝이 {len(found)}개라 간격이 가장 잘 맞는 것을 씀(어긋남 {found[0][0] / RATE * 1e3:.2f} ms, 다음 {found[1][0] / RATE * 1e3:.2f} ms)")
        else:
            notes.append("두 표지음의 짝을 못 찾음" + (f"(끝 후보는 있으나 간격이 안 맞는 시작 후보: {', '.join(misses[:5])})" if misses else ""))
    elif ps["found"] and pe["found"]:
        notes.append("두 표지음이 다른 녹음에 있음(세션 도중 워치 녹음이 멈췄다 다시 켜졌을 수 있음)" if rs and re_ else
                     "표지음 시각을 담은 워치 녹음이 하나뿐이거나 없음")

    single = None  # (키, 녹음, 프레임, 폰 표지음)
    if pair is not None:
        (s1, r1), (s2, r2) = pair
        watch["start"], watch["end"] = _hit(s1, r1, rises[s1]), _hit(s2, r2, rises[s2])
        steps = rs.clock.steps_between(s1, s2)
        if steps:
            notes.append("표지음 사이에 녹음 계단: " + ", ".join(f"{a / RATE:.1f}–{b / RATE:.1f} s {ns / 1e6:+.1f} ms" for a, b, ns in steps))
            single = ("start", rs, s1, ps)  # 짝으로 확인된 시작 표지음
        else:
            slope = (pe["ns"] - ps["ns"]) / (s2 - s1)
            out.update(method="two", recording=rs.name, line={"frame0": s1, "ns0": ps["ns"], "nsPerFrame": slope})
            watch_ns = rs.clock.ns(np.array([s1, s2]))
            iv = {"watchSec": (s2 - s1) / RATE, "phoneSec": (pe["ns"] - ps["ns"]) / 1e9, "phoneWavSec": (pe["frame"] - ps["frame"]) / RATE}
            out["intervals"] = iv
            out["intervalSpreadMs"] = (max(iv.values()) - min(iv.values())) * 1e3
            if out["intervalSpreadMs"] > INTERVAL_WARN_SEC * 1e3:
                notes.append(f"경고: 세 간격이 {out['intervalSpreadMs']:.2f} ms 다름({INTERVAL_WARN_SEC * 1e3:.0f} ms 넘음,"
                             " 폰 시각 튐이나 잘못 잡은 표지음일 수 있음)")
            out["clockPpm"] = ((pe["ns"] - ps["ns"]) / float(watch_ns[1] - watch_ns[0]) - 1) * 1e6
    else:
        end_ns = hs_io.nanos(session.meta.get("end"), "elapsedNs")
        if rs is not None:
            span = (end_ns - ps["ns"]) * RATE / 1e9 if end_ns is not None else 0.0  # 시작 표지음부터 세션 끝까지(워치 프레임)
            got = single_marker(rs, fs, "start", span, notes)
            if got:
                watch["start"] = _hit(*got)
                single = ("start", rs, got[0], ps)
        if single is None and re_ is not None:
            got = single_marker(re_, fe, "end", 0.0, notes)
            if got:
                watch["end"] = _hit(*got)
                single = ("end", re_, got[0], pe)
    if single is not None:
        key, rec, frame, pm = single
        out.update(method="one", recording=rec.name, line={"offsetNs": pm["ns"] - float(rec.clock.ns(frame))})
        notes.append(f"{'시작' if key == 'start' else '끝'} 표지음 하나와 워치 녹음 시각으로 맞춤(verify_sync 확인을 지나야 pairs가 씀)")
    if out["method"] == "none":
        notes.append("워치 녹음에서 쓸 표지음을 못 찾음" if ps["found"] or pe["found"] else "폰 표지음이 둘 다 없음")
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
        parts.append(f"{name} {w['frame'] / RATE:.3f} s(비 {w['ratio']:.0f}, 21.5–23 kHz {w['markerRise']:.0f}배, 2–17 kHz {w['otherRise']:.1f}배)"
                     if w else f"{name} 없음")
    lines.append(f"  워치 표지음: {', '.join(parts)}, 시작 창(±{START_WINDOW_SEC:.0f} s) 후보 {info['watch']['startCandidates']}개")
    if "intervals" in info:
        iv = info["intervals"]
        lines.append(f"  간격: 워치 {iv['watchSec']:.4f} s(48 kHz로 셈), 폰 elapsedNs {iv['phoneSec']:.4f} s, 폰 WAV {iv['phoneWavSec']:.4f} s"
                     f"(서로 {info['intervalSpreadMs']:.2f} ms 안), 두 기기 시계 빠르기 차 {info['clockPpm']:+.1f} ppm")
    if info.get("wallDiffMs"):
        lines.append("  표지음 − 시스템 시각 맞춤: " + ", ".join(f"{'시작' if k == 'start' else '끝'} {v:+.1f} ms" for k, v in info["wallDiffMs"].items()))
    steps = info.get("wallStepMs") or {}
    if steps:
        lines.append("  워치 녹음 벽시계 계단(stop − start 부팅 시각): " + ", ".join(
            f"{n} {'stop 없음' if v is None else f'{v:+.1f} ms'}" for n, v in steps.items()))
    lines += [f"  메모: {n}" for n in info["notes"]]
    return lines


def run(data, name, recordings):
    """세션 하나: (보고 줄, "표지음 둘"인가)."""
    info = align_session(hs_io.read_session(hs_io.session_folder(data, name)), recordings)
    hs_io.write_json(os.path.join(hs_io.processed_folder(data, name), "align.json"), info)
    return report(info), info["method"] == "two"


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
            continue
        step = wall_step_ms(recordings[-1])
        if step is not None and abs(step) > WALL_STEP_NOTE_MS:
            print(f"워치 {recordings[-1].name}: 녹음 중 벽시계 계단 {step:+.1f} ms(start·stop 부팅 시각 차)")
    missing = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        lines, two = hs_io.guarded(name, run, args.data, name, recordings)
        print("\n".join(lines))
        missing += not two
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
