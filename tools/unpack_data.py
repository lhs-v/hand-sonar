"""저장소의 데이터 묶음(data/, export_data.py)을 train.py·signal_check.py가 바로 읽는 데이터 폴더로 되돌린다(2026-10-29).

    python -I tools/unpack_data.py [--src 묶음 폴더] <데이터 폴더>

세션마다 쓰는 것:
- phone/sessions/<세션>/: meta.json, prompts.csv, landmarks.csv(gzip을 풀어 원본과 같은 바이트).
- processed/<세션>/: profiles.npy (600, R) float64 — 묶음에 있는 칸(bin0…)만 채우고 나머지는 0이다(학습 입력 300–359, signal_check가 쓰는
  칸은 모두 들어 있다), diff_profiles.npy = |raw[:, t+1]| − |raw[:, t]|, rows.npz(frame·ns·direct·ok·drift, alignSha = 그 align.json의
  SHA-256), pairs.csv(있으면), align.json, sync.json, signal.json(있으면). rows.npz를 새로 쓰므로 sync.json의 입력 지문(inputs)은 지금 파일로
  다시 잰다(지연·봉우리는 그대로).
- watch/recordings/<녹음>/rec.json(소리 조각은 없다. 점검·견줌용).
원본 소리가 없으므로 align.py·profiles.py는 다시 돌릴 수 없다. verify_sync.py·pairs.py·signal_check.py·train.py는 되돌린 폴더에서 돈다.
"""
import argparse
import gzip
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import verify_sync  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ("frame", "ns", "direct", "ok", "drift")


def gunzip(src, dst):
    with gzip.open(src, "rb") as g, open(dst, "wb") as f:
        shutil.copyfileobj(g, f)


def full_profiles(echo):
    """echo.npz의 칸 묶음 → (정렬 프로파일 (600, R) float64, 차분 (600, R − 1) float64)."""
    raw = np.zeros((600, echo["raw"].shape[1]), dtype=np.float64)
    b0 = int(echo["bin0"])
    raw[b0: b0 + echo["raw"].shape[0]] = echo["raw"]
    a = np.abs(raw)
    return raw, a[:, 1:] - a[:, :-1]


def unpack_session(src, data, name):
    pack = os.path.join(src, "sessions", name)
    sess = hs_io.session_folder(data, name)
    proc = os.path.join(data, "processed", name)
    os.makedirs(sess, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    for f in ("meta.json", "prompts.csv"):
        if os.path.exists(os.path.join(pack, f)):
            shutil.copyfile(os.path.join(pack, f), os.path.join(sess, f))
    gunzip(os.path.join(pack, "landmarks.csv.gz"), os.path.join(sess, "landmarks.csv"))
    for f in ("align.json", "signal.json"):
        if os.path.exists(os.path.join(pack, f)):
            shutil.copyfile(os.path.join(pack, f), os.path.join(proc, f))
    if os.path.exists(os.path.join(pack, "pairs.csv.gz")):
        gunzip(os.path.join(pack, "pairs.csv.gz"), os.path.join(proc, "pairs.csv"))
    with np.load(os.path.join(pack, "echo.npz"), allow_pickle=False) as echo:
        raw, diff = full_profiles(echo)
        np.save(os.path.join(proc, "profiles.npy"), raw)
        np.save(os.path.join(proc, "diff_profiles.npy"), diff)
        align_sha = hs_io.sha256(os.path.join(proc, "align.json"))  # verify_sync가 rows.npz의 바탕을 확인한다(같은 바이트라 원래 값과 같다)
        np.savez(os.path.join(proc, "rows.npz"), recording=echo["recording"], alignSha=np.array(align_sha), **{k: echo["rows_" + k] for k in ROWS})
        recording = str(echo["recording"])
    if recording and os.path.exists(os.path.join(pack, "rec.json")):
        rec_dir = os.path.join(data, "watch", "recordings", recording)
        os.makedirs(rec_dir, exist_ok=True)
        shutil.copyfile(os.path.join(pack, "rec.json"), os.path.join(rec_dir, "rec.json"))
    if os.path.exists(os.path.join(pack, "sync.json")):
        sync = hs_io.read_json(os.path.join(pack, "sync.json"))
        sync["inputs"] = verify_sync.inputs(data, name)  # rows.npz를 새로 썼다: 지문을 지금 파일로
        hs_io.write_json(os.path.join(proc, "sync.json"), sync)
    return f"{name}: 줄 {raw.shape[1]}개, 칸 {int(np.flatnonzero(np.abs(raw).sum(axis=1))[0])}–{int(np.flatnonzero(np.abs(raw).sum(axis=1))[-1])}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="데이터 묶음을 train.py가 읽는 데이터 폴더로 되돌린다")
    parser.add_argument("data", help="되돌릴 데이터 폴더(phone/sessions·processed·watch/recordings가 생긴다)")
    parser.add_argument("--src", default=os.path.join(REPO, "data"), help="묶음 폴더(export_data.py가 만든 data/)")
    args = parser.parse_args()
    names = sorted(n for n in os.listdir(os.path.join(args.src, "sessions")) if os.path.isfile(os.path.join(args.src, "sessions", n, "echo.npz")))
    for name in names:
        print(unpack_session(args.src, args.data, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
