"""모은 데이터를 저장소에 올릴 묶음(data/)으로 옮긴다(2026-10-29). 영상·이미지·원본 소리·센서 기록은 넣지 않는다.

    python -I tools/export_data.py [--data 데이터 폴더] [--out 묶음 폴더] [--recording 녹음 …] [세션 …]

원본 소리(워치 part-NNN.wav, 폰 phone_audio.wav)는 18–21 kHz만이 아니라 방 안의 들리는 소리(말소리 포함)를 담고 있어 넣지 않는다.
대신 18–21 kHz 대역 통과 뒤 송신 처프와 상관을 구한 에코 프로파일을 넣는다(공개 WatchHand 데이터도 프로파일만 공개했다).

세션마다 <묶음>/sessions/<세션>/:
- meta.json·prompts.csv: 폰 세션 그대로. landmarks.csv.gz: 관절(풀면 원본과 같은 바이트).
- rec.json: 짝지은 워치 녹음(align.json recording)의 기록. align.json·sync.json·pairs.csv.gz·signal.json: 처리 결과(있는 것만).
- echo.npz: raw (칸 BINS, 줄) float32 = processed profiles.npy의 그 칸(직접 경로를 칸 300에 맞추고 3줄 중앙값을 건 부호 있는 상관).
  차분은 |raw[:, t+1]| − |raw[:, t]|로 다시 만든다(profiles.py와 같은 식). bin0, rows_frame·rows_ns·rows_direct·rows_ok·rows_drift(rows.npz)를 함께 둔다.
--recording으로 고른 워치 녹음(세션 없는 시험 녹음)은 <묶음>/recordings/<녹음>/에 rec.json과 echo.npz(앞 RECORDING_SEC초를 profiles.compute로
처리한 칸 BINS, 정렬·3줄 중앙값까지 같음)를 둔다.
<묶음>/sessions.csv: 세션마다 한 줄(날짜, 워치 볼륨·창·음원, 프레임, 맞춤 지연·상관, 학습에 쓸 수 있는가, 신호 잣대).
"""
import argparse
import csv
import gzip
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import profiles  # noqa: E402
import verify_sync  # noqa: E402
import numpy as np  # noqa: E402

BINS = slice(250, 480)  # 직접 경로(칸 300) 둘레·손 자리(301–360)·손 밖(420–479). 학습 입력(300–359)과 signal_check가 쓰는 칸을 모두 담는다
RECORDING_SEC = 60
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = ["session", "date", "watchRecording", "volume", "window", "source", "hand", "rewear", "posture", "armSupport", "strap",
           "frames", "durationSec", "lagMs", "syncPeak", "usable", "pairs", "corr0", "contrast", "peakWidthBins", "peakJitter90Bins"]


def gzip_copy(src, dst):
    """gzip으로 복사한다. 머리말에 파일 이름·시각을 넣지 않는다(같은 입력이면 같은 바이트, 로컬 경로가 새지 않게)."""
    with open(src, "rb") as f, open(dst, "wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)


def echo_arrays(raw, rows=None):
    """echo.npz에 넣을 dict. raw (600, R)은 정렬 프로파일, rows는 rows.npz(없으면 줄 정보 없이)."""
    out = {"raw": np.asarray(raw[BINS], dtype=np.float32), "bin0": np.int16(BINS.start)}
    if rows is not None:
        for key in ("frame", "ns", "direct", "ok", "drift"):
            out["rows_" + key] = rows[key]
    return out


def export_session(data, out, name):
    """세션 하나를 묶음으로. sessions.csv 한 줄(dict)을 돌려준다."""
    src = hs_io.session_folder(data, name)
    proc = os.path.join(data, "processed", name)
    dst = os.path.join(out, "sessions", name)
    os.makedirs(dst, exist_ok=True)
    for f in ("meta.json", "prompts.csv"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    gzip_copy(os.path.join(src, "landmarks.csv"), os.path.join(dst, "landmarks.csv.gz"))
    for f in ("align.json", "sync.json", "signal.json"):
        if os.path.exists(os.path.join(proc, f)):
            shutil.copyfile(os.path.join(proc, f), os.path.join(dst, f))
    has_pairs = os.path.exists(os.path.join(proc, "pairs.csv"))
    if has_pairs:
        gzip_copy(os.path.join(proc, "pairs.csv"), os.path.join(dst, "pairs.csv.gz"))
    align = hs_io.read_json(os.path.join(proc, "align.json"))
    recording = align.get("recording")
    rec = {}
    rec_path = os.path.join(data, "watch", "recordings", str(recording), "rec.json")
    if recording and os.path.exists(rec_path):
        shutil.copyfile(rec_path, os.path.join(dst, "rec.json"))
        rec = hs_io.read_json(rec_path)
    raw = np.load(os.path.join(proc, "profiles.npy"), mmap_mode="r", allow_pickle=False)
    with np.load(os.path.join(proc, "rows.npz"), allow_pickle=False) as rows:
        arrays = echo_arrays(raw, rows)
    np.savez_compressed(os.path.join(dst, "echo.npz"), recording=np.str_(recording or ""), **arrays)

    meta = hs_io.read_json(os.path.join(src, "meta.json"))
    sync = hs_io.read_json(os.path.join(proc, "sync.json")) if os.path.exists(os.path.join(proc, "sync.json")) else {}
    signal = hs_io.read_json(os.path.join(proc, "signal.json")) if os.path.exists(os.path.join(proc, "signal.json")) else {}
    context, stats = meta.get("context") or {}, meta.get("stats") or {}
    usable, _ = verify_sync.usable(sync) if sync else (False, "")
    return {"session": name, "date": name[:8], "watchRecording": recording or "", "volume": (rec.get("volume") or {}).get("index", ""),
            "window": (rec.get("tx") or {}).get("window", ""), "source": (rec.get("audio") or {}).get("source", ""), "hand": meta.get("hand") or "left",
            "rewear": bool(meta.get("rewear")), "posture": context.get("posture") or "", "armSupport": context.get("armSupport") or "",
            "strap": context.get("strap") or "", "frames": stats.get("frames", ""), "durationSec": stats.get("durationSec", ""),
            "lagMs": round(sync["lagMs"], 1) if "lagMs" in sync else "", "syncPeak": round(sync["peak"], 3) if "peak" in sync else "",
            "usable": usable, "pairs": has_pairs, **{k: round(signal[k], 3) for k in ("corr0", "contrast", "peakWidthBins", "peakJitter90Bins") if k in signal}}


def export_recording(data, out, name, seconds=RECORDING_SEC):
    """세션 없는 워치 녹음 하나: rec.json과 앞 seconds초의 정렬 프로파일(칸 BINS)."""
    folder = os.path.join(data, "watch", "recordings", name)
    w = hs_io.read_recording(folder)
    window, amplitude, shape = profiles.tx_spec(w.rec)
    n = min(seconds * profiles.RATE, w.frames) // profiles.N * profiles.N
    p = profiles.compute(w.read(0, n), profiles.tx_template(window, amplitude, shape), direct_floor=profiles.direct_min(shape))
    dst = os.path.join(out, "recordings", name)
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(folder, "rec.json"), os.path.join(dst, "rec.json"))
    np.savez_compressed(os.path.join(dst, "echo.npz"), recording=np.str_(name), seconds=np.float32(n / profiles.RATE),
                        rows_direct=p.direct.astype(np.int16), rows_ok=p.ok, **echo_arrays(p.raw))
    return f"{name}: 창 {window}, 앞 {n / profiles.RATE:.0f} s"


def exportable(data):
    """보관된 세션 중 처리(정렬 프로파일·맞춤)까지 된 것."""
    return [n for n, why in hs_io.chosen(data) if not why
            and all(os.path.exists(os.path.join(data, "processed", n, f)) for f in ("profiles.npy", "rows.npz", "align.json"))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="모은 데이터를 저장소에 올릴 묶음으로(영상·원본 소리·센서 제외)")
    parser.add_argument("sessions", nargs="*", help="세션 이름(없으면 처리된 보관 세션 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--out", default=os.path.join(REPO, "data"))
    parser.add_argument("--recording", action="append", default=[], help="세션 없는 워치 녹음도 넣는다(여러 번)")
    args = parser.parse_args()
    rows = []
    for name in args.sessions or exportable(args.data):
        rows.append(export_session(args.data, args.out, name))
        print(f"{name}: 넣음(맞춤 상관 {rows[-1]['syncPeak']}, 학습에 쓸 수 있음 {rows[-1]['usable']})")
    with open(os.path.join(args.out, "sessions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for name in args.recording:
        print(export_recording(args.data, args.out, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
