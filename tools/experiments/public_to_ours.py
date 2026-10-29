"""공개 WatchHand 세션(사람×손 폴더 하나)을 우리 데이터 폴더 형식으로 옮긴다(우리 처리·학습 방식 검증용, 2026-10-28 실험).

    python -I tools/experiments/public_to_ours.py <받은 파일 폴더> <우리 형식 데이터 폴더> <hand_landmarker.task> [--prefix 접두] [세션 번호 …]

- 받은 파일은 zip 안 경로의 '/'를 '__'로 바꾼 이름이다(접두 기본 PREFIX = 참가자 9 왼손). __MACOSX 찌꺼기는 쓰지 않는다.
  docs/public-watchhand-data.md의 범위 요청(필요한 파일만 꺼내기) 방식으로 받는다. 영상 10개·프로파일 10개·시각표로 약 0.9 GB.
- 세션 i(0부터): 영상 videos[i], 프로파일 audio files[i], 동기 audio·ground_truth syncing_poses[i](hand-sonar/docs/public-watchhand-data.md §3·§5).
- 세션 폴더 phone/sessions/<날짜-시각>/: meta.json(보관·왼손·세션 사이 다시 참), landmarks.csv(130열, 우리 앱처럼 거울 그림 속 실제 왼손 =
  MediaPipe "Right" 중 점수가 가장 높은 손, handedness 칸에는 실제 손 이름 "Left").
- 처리 폴더 processed/<날짜-시각>/: profiles.npy(받은 파일의 하드 링크), diff_profiles.npy(|p[t+1]| − |p[t]|), rows.npz(ok = 칸 300 세기가
  세션 중앙값의 0.1배 이상), pairs.csv(프레임 → 줄: echo_row = audio_sync + round((frame_time[v] − frame_time[gt_sync]) × 80), 지연 0),
  align.json(method two — 표지음이 아니라 공개 동기 번호로 맞췄음을 적음), sync.json(지연 0, 봉우리 = 그 대응에서 잰 관절 속도–칸 301–360
  차분 세기 상관, ±1 s에서 가장 높은 지연도 적음, 입력 지문은 verify_sync.inputs와 같은 셈).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 붙는다
import verify_sync  # noqa: E402
import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

PREFIX = "Study-1_sub9,10,11,12__sub9_samsung_left_video__"
ROWS_PER_S = 80.0
HEADER = (["frameNs", "hands", "handedness", "score"] + [f"{a}{j}" for j in range(21) for a in "xyz"]
          + [f"w{a}{j}" for j in range(21) for a in "xyz"])


def find(src, name, prefix=PREFIX):
    path = os.path.join(src, prefix + name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def read_floats(path):
    with open(path, encoding="utf-8") as f:
        return np.array([float(x) for x in f if x.strip()])


def track(mp4, model, n_max):
    """[(손 개수, 점수 또는 None, 화면 21×3, 세계 21×3)] — 거울 그림 속 실제 왼손(MediaPipe "Right")만."""
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        while len(out) < n_max:
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, int(round(len(out) * 1000.0 / fps)))
            k = len(r.hand_world_landmarks)
            cand = [q for q in range(k) if r.handedness[q][0].category_name == "Right"]
            if not cand:
                out.append((k, None, None, None))
                continue
            b = max(cand, key=lambda q: r.handedness[q][0].score)
            out.append((k, float(r.handedness[b][0].score), [(p.x, p.y, p.z) for p in r.hand_landmarks[b]],
                        [(p.x, p.y, p.z) for p in r.hand_world_landmarks[b]]))
    cap.release()
    return out


def speed_energy(world, tracked, rows, diff, lags):
    """관절 속도(손목 기준 20점, 이웃 프레임)와 칸 301–360 |차분| 평균의 로그 상관을 줄 지연마다."""
    q = world - world[:, :1]
    sp = np.linalg.norm((q[1:] - q[:-1]).reshape(len(q) - 1, -1), axis=1)
    good = tracked[1:] & tracked[:-1] & (rows[1:] >= 1) & (rows[1:] < diff.shape[1])
    e = np.abs(np.asarray(diff[301:361], dtype=np.float64)).mean(axis=0)
    out = []
    for lag in lags:
        j = np.clip(rows[1:][good] - 1 + lag, 0, len(e) - 1)
        out.append(float(np.corrcoef(np.log(sp[good] + 1e-9), np.log(e[j] + 1e-9))[0, 1]))
    return np.array(out)


def convert(src, data, model, i, cfg, prefix=PREFIX):
    video = cfg["ground_truth"]["videos"][i]
    stem = video[: -len(".mp4")]
    name = stem[len("record_"): len("record_") + 15].replace("_", "-")  # 20250320-201714
    sess = os.path.join(data, "phone", "sessions", name)
    proc = os.path.join(data, "processed", name)
    os.makedirs(sess, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    ft = read_floats(find(src, stem + "_frame_time.txt", prefix))
    marks = track(find(src, video, prefix), model, len(ft))
    n = len(marks)
    frame_ns = np.round(ft[:n] * 1e9).astype(np.int64)
    tracked = np.array([m[1] is not None for m in marks])
    world = np.array([m[3] if m[1] is not None else [[np.nan] * 3] * 21 for m in marks], dtype=np.float64)
    with open(os.path.join(sess, "landmarks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for t, (k, score, image, wpts) in zip(frame_ns, marks):
            if score is None:
                w.writerow([int(t), k, "", ""] + [""] * 126)
            else:
                w.writerow([int(t), k, "Left", f"{score:.6f}"] + [f"{v:.6f}" for p in image for v in p] + [f"{v:.6f}" for p in wpts for v in p])

    prof_src = find(src, cfg["audio"]["files"][i], prefix)
    prof_dst = os.path.join(proc, "profiles.npy")
    if os.path.exists(prof_dst):
        os.remove(prof_dst)
    os.link(prof_src, prof_dst)  # 복사하지 않는다(디스크 여유가 적음)
    p = np.load(prof_dst, mmap_mode="r", allow_pickle=False)
    if p.ndim != 2 or p.shape[0] != 600:
        raise ValueError(f"{prof_src}: 모양 {p.shape}")
    a = np.abs(np.asarray(p, dtype=np.float64))
    diff = a[:, 1:] - a[:, :-1]
    np.save(os.path.join(proc, "diff_profiles.npy"), diff)
    a300 = a[300]
    ok = a300 >= 0.1 * np.median(a300[a300 > 0])
    np.savez(os.path.join(proc, "rows.npz"), ok=ok)
    R = p.shape[1]

    gt_sync, audio_sync = cfg["ground_truth"]["syncing_poses"][i], cfg["audio"]["syncing_poses"][i]
    real = audio_sync + (ft[:n] - ft[gt_sync]) * ROWS_PER_S
    rows = np.rint(real).astype(np.int64)
    inside = (rows >= 0) & (rows < R)
    with open(os.path.join(proc, "pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "frameNs", "soundNs", "row", "dtMs", "hand", "direct", "prompt", "classId", "flag", "edge", "video"])
        for v in range(n):
            direct = int(inside[v] and ok[rows[v]])
            w.writerow([v, int(frame_ns[v]), int(frame_ns[v]), int(rows[v]) if inside[v] else "", f"{(real[v] - rows[v]) * 12.5:.3f}",
                        int(tracked[v]), direct, "", "", "", 0, v])

    meta = {"format": 1, "app": "public-watchhand", "device": prefix.rstrip("_"), "hand": "left", "status": "kept",
            "rewear": i > 0, "start": {"wallMs": float(ft[0] * 1000.0), "elapsedNs": int(frame_ns[0])},
            "end": {"wallMs": float(ft[n - 1] * 1000.0), "elapsedNs": int(frame_ns[n - 1])},
            "stats": {"frames": n, "handRatio": float(tracked.mean())},
            "source": {"dataset": "doi:10.7298/qf1v-j805 (CC BY 4.0)", "session": i, "video": video, "profiles": cfg["audio"]["files"][i],
                       "gtSync": gt_sync, "audioSync": audio_sync, "anchorLength": cfg["ground_truth"].get("anchor_length")}}
    hs_io.write_json(os.path.join(sess, "meta.json"), meta)
    hs_io.write_json(os.path.join(proc, "align.json"), {"method": "two", "note": "공개 데이터: config syncing_poses로 맞춤(표지음 아님)"})

    lags = np.arange(-80, 81)
    cs = speed_energy(world, tracked, rows, diff, lags)
    peak0 = float(cs[80])
    best = int(lags[np.nanargmax(cs)])
    sync = {"format": 2, "session": name, "lagMs": 0.0, "peak": peak0, "withinFrame": True, "frameMs": 33.0, "minPeak": verify_sync.MIN_PEAK,
            "timeBase": "public frame_time", "usable": peak0 >= verify_sync.MIN_PEAK,
            "note": f"공개 동기 번호 대응 그대로(지연 0). ±1 s에서 가장 높은 상관은 지연 {best * 12.5:+.1f} ms({np.nanmax(cs):.3f})",
            "inputs": verify_sync.inputs(data, name)}
    hs_io.write_json(os.path.join(proc, "sync.json"), sync)
    return (f"{name}(세션 {i}): 프레임 {n}(시각표 {len(ft)}줄), 손 {tracked.mean():.1%}, 손 2개 {np.mean([m[0] >= 2 for m in marks]):.1%},"
            f" 줄 안 {inside.mean():.1%}, 직접 경로 ok {ok.mean():.1%}, 속도–차분 상관 지연 0 {peak0:.3f}, 최고 {np.nanmax(cs):.3f}(지연 {best * 12.5:+.1f} ms)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="공개 WatchHand 세션을 우리 데이터 폴더 형식으로")
    parser.add_argument("src", help="받은 파일 폴더(zip 안 경로의 '/'를 '__'로 바꾼 이름들)")
    parser.add_argument("data", help="만들 데이터 폴더(phone/sessions·processed)")
    parser.add_argument("model", help="hand_landmarker.task")
    parser.add_argument("sessions", nargs="*", type=int, help="세션 번호(0부터, 없으면 config의 모든 세션)")
    parser.add_argument("--prefix", default=PREFIX, help="받은 파일 이름 앞부분(사람×손 폴더)")
    args = parser.parse_args()
    with open(find(args.src, "config.json", args.prefix), encoding="utf-8") as f:
        cfg = json.load(f)
    for i in args.sessions or list(range(len(cfg["ground_truth"]["videos"]))):
        try:
            print(convert(args.src, args.data, args.model, i, cfg, args.prefix), flush=True)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 나머지는 옮긴다
            print(f"세션 {i}: 실패 {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
