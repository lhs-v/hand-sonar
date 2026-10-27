"""워치 녹음 폴더(rec.json + part-NNN.wav)를 점검한다(손 소나 계획 1, 작업 6).

    python -I tools/check_recording.py <녹음 폴더>

- 조각이 빠짐없이 이어지는지(firstFrame, 프레임 수, WAV 형식, 1분 조각 길이, 기록에 없는 조각).
  닫힌 조각이 머리말보다 짧게 잘렸으면(무선 adb pull이 끊김) 파일 크기로 재서 기록과 프레임 수가 달라 문제로 나온다
- 녹음 시각으로 잰 실제 샘플레이트와 넘침
  - 위치나 시각이 앞으로 가지 않은 줄(getTimestamp가 지난 값을 다시 준 것)은 새 측정이 아니라 빼고 센다
  - 계단: 이웃 줄의 시각 간격과 녹음 위치 간격이 어긋나는지. 앱이 늦게 읽어 서버가 버린 샘플은 위치에도 읽은 수에도 들지 않고 시각만 그만큼 흐른다
  - 시각 출처가 timestamp면 녹음 위치 − 읽은 수의 튐도 본다(서버 쪽 손실과 읽기 밀림)
- 재생 시각으로 잰 샘플레이트(멈춤·느림은 문제)와 재생 끊김 수(보고만)
- 18–21 kHz 소리가 녹음에 들어 있는지(조각마다 가운데 1초)
- 오디오 길이와 기록 시간(정지 − 시작)이 맞는지
- 정지 이유가 error면 문제(storage는 저장 공간 부족에 따른 자동 정지라 문제로 세지 않는다)
문제가 하나라도 있으면 끝 코드 1로 끝난다.
"""
import json
import os
import site
import sys
import wave

sys.path.append(site.getusersitepackages())  # python -I에서도 사용자 설치 numpy를 찾게
import numpy as np  # noqa: E402

RATE = 48_000
PART_FRAMES = RATE * 60
HEADER_BYTES = 44
MAX_RATE_ERROR = 500e-6  # 녹음 샘플레이트가 이보다 크게 어긋나면 시각 기록을 의심한다
MAX_LAG_JUMP = RATE // 10  # 녹음 위치 − 읽은 수가 100 ms 넘게 튀면 넘침으로 본다(시각 출처가 read면 이웃 줄의 시각 간격과 위치 간격의 차를 같은 100 ms로 견준다)
MAX_TIMESTAMP_STEP_SEC = 0.02  # 시각 출처가 timestamp일 때 이웃 줄의 |시각 간격 − 위치 간격|이 이보다 크면 샘플이 빠진 것으로 본다. 실기기 녹음 6개의 최대가 0.97 ms라 약 20배 여유다
MIN_BAND_CONTRAST_DB = 10.0  # 18–21 kHz가 10–17 kHz보다 이만큼은 세야 처프가 녹음에 들어간 것으로 본다
MAX_PLAY_RATE_ERROR = 0.01  # 재생 샘플레이트가 1% 넘게 어긋나면 재생이 멈췄거나 느린 것으로 본다(짧은 멈춤은 전체 기울기에 묻힌다)
MAX_LENGTH_GAP_SEC = 3.0  # 시작·정지 시각에는 준비와 정리 시간이 들어가 몇 초 차이는 정상이다. 3초 넘게 어긋나면 녹음이 오류 없이 중간에 멈춘 것으로 본다


def wav_frames(path):
    """WAV 프레임 수. 머리말의 data 길이를 믿되, 닫히지 않은 조각(data 길이 0)이거나 파일이 머리말이 말하는 것보다 짧으면(잘림)
    파일 크기에서 머리말 44바이트를 뺀 값으로 잰다. 그러면 무선 adb pull이 끊겨 잘린 조각은 rec.json의 프레임 수와 달라 문제로 나온다."""
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise ValueError(f"{os.path.basename(path)}: 형식이 다름 ({w.getnchannels()}채널, {w.getsampwidth() * 8}비트, {w.getframerate()} Hz)")
        n = w.getnframes()
    size_frames = (os.path.getsize(path) - HEADER_BYTES) // 2
    if n == 0 or size_frames < n:
        return size_frames
    return n


def forward_rows(clock):
    """(시각 기록에서 앞으로 간 줄만 남긴 목록, 뺀 줄 수). 마지막으로 남긴 줄보다 녹음 위치(p[0])나 시각(p[1])이 크지 않은 줄은 뺀다.
    getTimestamp가 지난 값을 다시 줄 수 있다(실기기 205800의 32번 줄은 31번 줄과 위치·시각이 같고 읽은 수만 한 초 늘었다).
    같거나 작은 줄은 새 측정이 아니라서, 남기면 거짓 넘침과 어긋난 샘플레이트가 나온다."""
    kept = []
    for p in clock:
        if kept and (p[0] <= kept[-1][0] or p[1] <= kept[-1][1]):
            continue
        kept.append(p)
    return kept, len(clock) - len(kept)


def fit_rate(pairs, frame_index):
    """[…, elapsedNs, …] 줄들에서 frame_index 칸의 프레임이 초당 얼마나 늘었는지(최소제곱 직선)."""
    if len(pairs) < 2:
        return float("nan")
    frames = np.array([p[frame_index] for p in pairs], dtype=np.float64)
    seconds = np.array([p[1] for p in pairs], dtype=np.float64) / 1e9
    return float(np.polyfit(seconds - seconds[0], frames - frames[0], 1)[0])


def band_contrast(samples):
    """(18–21 kHz 세기 dBFS, 18–21 kHz와 10–17 kHz의 세기 차 dB)."""
    x = samples.astype(np.float64) / 32768.0
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / RATE)
    sonar = power[(freqs >= 18_000) & (freqs <= 21_000)].sum()
    below = power[(freqs >= 10_000) & (freqs < 17_000)].sum()
    total = power.sum() + 1e-30
    level_db = 10 * np.log10(sonar / total * np.mean(x ** 2) + 1e-30)
    return float(level_db), float(10 * np.log10((sonar + 1e-30) / (below + 1e-30)))


def check(folder):
    """(보고 줄 목록, 문제 목록)."""
    lines, problems = [], []
    with open(os.path.join(folder, "rec.json"), encoding="utf-8") as f:
        rec = json.load(f)

    parts = rec.get("parts") or []
    if not parts:
        problems.append("rec.json에 조각 목록이 없음")
    expected_first, total = 0, 0
    existing = []  # (파일 이름, 경로, 프레임 수): 파일이 있는 조각
    for i, part in enumerate(parts):
        path = os.path.join(folder, part["file"])
        if not os.path.exists(path):
            problems.append(f"{part['file']} 없음")
            continue
        n = wav_frames(path)
        existing.append((part["file"], path, n))
        if part["firstFrame"] != expected_first:
            problems.append(f"{part['file']}: firstFrame {part['firstFrame']} ≠ 앞 조각까지의 합 {expected_first}")
        if n != part["frames"]:
            problems.append(f"{part['file']}: 파일 프레임 {n} ≠ 기록 {part['frames']}")
        if i < len(parts) - 1 and n != PART_FRAMES:
            problems.append(f"{part['file']}: 마지막 조각이 아닌데 {n}프레임(1분 = {PART_FRAMES})")
        expected_first += part["frames"]
        total += n
    listed = {p["file"] for p in parts}
    extra = sorted(name for name in os.listdir(folder) if name.startswith("part-") and name not in listed)
    if extra:
        problems.append("기록에 없는 조각: " + ", ".join(extra))
    lines.append(f"조각 {len(parts)}개, {total}프레임 = {total / RATE / 60:.2f}분")
    volume = rec.get("volume")
    if volume:  # 보고만 한다(문제로 세지 않음): 볼륨 0이면 처프가 안 나가 '약함'으로 걸리는데, 그 원인을 찾느라 시간이 들었다(작업 7)
        lines.append(f"미디어 볼륨 {volume['index']}/{volume['max']}")

    stop = rec.get("stop")
    if stop is None:
        problems.append("stop 없음(녹음 중이거나 앱이 죽음)")
    else:
        reason = stop["reason"]
        note = "(저장 공간 부족으로 자동 정지)" if reason == "storage" else ""
        lines.append(f"정지 이유 {reason}{note}, 녹음 소스 {rec['audio']['source']}, 창 {rec['tx']['window']}, 진폭 {rec['tx']['amplitude']}")
        if reason == "error":
            problems.append("정지 이유가 error(녹음이나 재생이 오류로 끝남)")
        audio_s = total / RATE
        clock_s = (stop["elapsedNs"] - rec["start"]["elapsedNs"]) / 1e9
        lines.append(f"오디오 길이 {audio_s:.2f} s, 기록 시간(정지 − 시작) {clock_s:.2f} s")
        if abs(audio_s - clock_s) > MAX_LENGTH_GAP_SEC:
            problems.append(f"오디오 길이 {audio_s:.2f} s ≠ 기록 시간 {clock_s:.2f} s")

    source = rec.get("recordClockSource")
    raw_clock = rec.get("recordClock") or []
    # getTimestamp가 지난 값을 다시 줄 수 있다(실기기 205800의 32번 줄). 위치나 시각이 앞으로 가지 않은 줄은 새 측정이 아니라 빼고, 이후 검사는 모두 남은 줄로 한다.
    clock, stale = forward_rows(raw_clock)
    rate = fit_rate(clock, 2)
    lines.append(f"녹음 샘플레이트(읽은 수 기준) {rate:.2f} Hz ({(rate / RATE - 1) * 1e6:+.0f} ppm), 시각 출처 {source}, 기록 {len(raw_clock)}줄")
    if stale:
        lines.append(f"녹음 시각이 앞으로 가지 않은 줄 {stale}개는 뺌")
    if not np.isfinite(rate) or abs(rate / RATE - 1) > MAX_RATE_ERROR:
        problems.append(f"녹음 샘플레이트가 이상함: {rate:.2f} Hz")
    if len(clock) > 1:
        if source == "read":
            # 읽은 직후 시각 줄은 [읽은 수, 시각, 읽은 수]라 위치 − 읽은 수가 늘 0이다. 그래서 넘침은 아래 시각 간격으로만 본다.
            lines.append("녹음 시각 출처가 read라 넘침은 시각 간격으로만 본다")
            max_step_sec = MAX_LAG_JUMP / RATE
        else:
            max_step_sec = MAX_TIMESTAMP_STEP_SEC
        # 앱이 늦게 읽어 서버가 이 녹음에 줄 샘플을 버리면, 버린 샘플은 녹음 위치에도 읽은 수에도 들지 않고 시각만 그만큼 흐른다.
        # 위치 − 읽은 수는 그대로라 아래 검사로는 안 보이므로, 이웃 줄의 시각 간격이 위치 간격보다 길어지는 계단으로 잡는다.
        # read 줄은 위치 = 읽은 수라 같은 검사다.
        for p0, p1 in zip(clock, clock[1:]):
            dt = (p1[1] - p0[1]) / 1e9
            dpos = (p1[0] - p0[0]) / RATE
            if abs(dt - dpos) > max_step_sec:
                problems.append(f"녹음 넘침 의심(시각 간격 {dt:.3f} s, 녹음 위치 간격 {dpos:.3f} s)")
        if source != "read":
            lag = np.array([p[0] - p[2] for p in clock], dtype=np.int64)
            jump = int(np.max(np.abs(np.diff(lag))))
            lines.append(f"녹음 위치 − 읽은 수: {int(lag.min())}…{int(lag.max())}프레임, 이웃 차 최대 {jump}")
            if jump > MAX_LAG_JUMP:
                problems.append(f"녹음 넘침 의심: 녹음 위치 − 읽은 수가 {jump}프레임 튐")

    play = rec.get("playClock") or []
    underruns = rec.get("underruns") or []
    play_rate = fit_rate(play, 0)
    lines.append(f"재생 샘플레이트 {play_rate:.2f} Hz, 재생 끊김 {underruns[-1][1] if underruns else 0}번")
    if len(play) < 2:
        problems.append(f"재생 시각 기록이 없음(playClock {len(play)}줄)")
    elif not np.isfinite(play_rate) or abs(play_rate / RATE - 1) > MAX_PLAY_RATE_ERROR:
        problems.append(f"재생이 멈췄거나 느림: {play_rate:.2f} Hz")

    contrasts = []  # (파일 이름, 18–21 kHz 대비 dB)
    for name, path, n in existing:
        if n < RATE:
            continue
        start = n // 2 - RATE // 2  # 조각 가운데 1초의 첫 프레임 번호
        x = np.fromfile(path, dtype="<i2", count=RATE, offset=HEADER_BYTES + 2 * start)
        _, contrast_db = band_contrast(x)
        contrasts.append((name, contrast_db))
        if contrast_db < MIN_BAND_CONTRAST_DB:
            problems.append(f"{name}: 18–21 kHz 소리가 약함({contrast_db:+.1f} dB)")
    if contrasts:
        low = min(contrasts, key=lambda c: c[1])
        high = max(contrasts, key=lambda c: c[1])
        lines.append(f"18–21 kHz 대비({len(contrasts)}개 조각): 최소 {low[1]:+.1f} dB({low[0]}) … 최대 {high[1]:+.1f} dB({high[0]})")
    elif parts:
        problems.append("조각이 1초보다 짧아 대역을 못 잼(1초 이상인 조각 없음)")
    return lines, problems


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    lines, problems = check(sys.argv[1])
    for line in lines:
        print(line)
    if problems:
        print("문제:")
        for p in problems:
            print("  - " + p)
        return 1
    print("문제 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
